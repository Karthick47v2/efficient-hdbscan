

import numpy as np
from collections import defaultdict
import heapq
import ctypes
import itertools
import math

np.random.seed(42)

random_generator = ctypes.CDLL('./random_generator.so')
random_generator.generate_random_numbers.argtypes = [
    ctypes.c_int, 
    ctypes.c_int,
    ctypes.c_int,
]
random_generator.generate_random_numbers.restype = ctypes.POINTER(ctypes.c_double)

nn = ctypes.CDLL('./nn.so')
nn.find_nearest_neighbour_dist.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]

nn_dist = ctypes.CDLL('./nn_dist.so')  # Replace with the actual library name

nn_dist.calculate_core_dist.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))),  # Input data
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # Number of elements in innermost lists
    ctypes.POINTER(ctypes.c_int),  #
    ctypes.c_int,  # Number of outer lists
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
nn_dist.calculate_core_dist.restype = ctypes.c_double


def generate_random_planes(num_hashtables, hash_size, input_dim):
    # Call the C function
    random_numbers_ptr = random_generator.generate_random_numbers(ctypes.c_int(num_hashtables), ctypes.c_int(hash_size), ctypes.c_int(input_dim))
    # Convert the C array to a NumPy array
    random_numbers = np.ctypeslib.as_array(random_numbers_ptr, shape=(num_hashtables, hash_size, input_dim))
    
    return random_numbers

class LSHash(object):
    def __init__(self, hash_size, input_dim, num_hashtables=1):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        # self.uniform_planes = np.random.randn(self.num_hashtables, self.hash_size, self.input_dim)
        self.uniform_planes = generate_random_planes(self.num_hashtables, self.hash_size, self.input_dim)
        self.hash_tables = [defaultdict(list) for _ in range(self.num_hashtables)]


    def add_points(self, input_points):
        self.input_points = input_points
        for i, table in enumerate(self.hash_tables):
            planes = self.uniform_planes[i]
            projections = np.dot(planes, input_points.T)
            hash_strings = (projections > 0).T

            for h, point in zip(hash_strings, input_points):
                table[h.tobytes()].append(point)

    def nn_dist(self, num_results=None):
        if num_results == 0:
            return

        dict_list = []
        num_elements_ = []
        num_bins = []

        for table in self.hash_tables:
            temp = list(table.values())
            num_elements_.append(list(map(len, temp)))
            num_bins.append(len(temp))
            dict_list.append(temp)

        n_list = len(dict_list)

        c_input_data = (ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))) * n_list)()
        num_inner_lists = (ctypes.c_int * n_list)()
        num_elements = (ctypes.POINTER(ctypes.c_int) * n_list)()
        
        for i, outer_list in enumerate(dict_list):
            num_inner_lists[i] = num_bins[i]
            num_elements[i] = (ctypes.c_int * num_bins[i])()
            c_input_data[i] = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * num_bins[i])()
            
            for j, inner_list in enumerate(outer_list):
                num_elements[i][j] = num_elements_[i][j]
                c_input_data[i][j] = (ctypes.POINTER(ctypes.c_double) * num_elements_[i][j])()
                
                for k, arr in enumerate(inner_list):
                    c_input_data[i][j][k] = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        c_array = (ctypes.POINTER(ctypes.c_double) * len(self.input_points))()
        for i, row in enumerate(self.input_points):
            c_array[i] = row.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

########## FIND BEST VALUES FOR MAX_NEIGHBOUR, HASH_TABLE, HYPERPLANE
# Speed depends on max neigh for C dym alloc
        result = nn_dist.calculate_core_dist(c_input_data, num_elements, num_inner_lists, ctypes.c_int(n_list), ctypes.c_int(self.input_dim), c_array, ctypes.c_int(len(self.input_points)), ctypes.c_int(2048), ctypes.c_int(num_results))

        print(result)



        output = {}
        
        # candidates = defaultdict(set)

        # for table in self.hash_tables:
        #     for values in table.values():
        #         temp = list(map(tuple, values))
        #         for v in values:
        #             candidates[tuple(v)].update(temp)

        # n_keys = len(candidates)
        # data_points = list(itertools.chain.from_iterable(candidates.keys()))

        # neighbours = []
        # n_neighbours = [None] * n_keys

        # for i, values in enumerate(candidates.values()):
        #     neighbours.extend(list(itertools.chain.from_iterable(values)))
        #     n_neighbours[i] = len(values)

        # data_points_c = (ctypes.c_double * (n_keys * self.input_dim))(*data_points)
        # neighbours_c = (ctypes.c_double * (len(neighbours)))(*neighbours)
        # n_neighbours_c = (ctypes.c_int * n_keys)(*n_neighbours)

        # nn.find_nearest_neighbour_dist(data_points_c, neighbours_c, n_neighbours_c, ctypes.c_int(n_keys), ctypes.c_int(self.input_dim))

        # for point, neighbours in candidates.items():
        #     p = np.array(point)
        #     candidates[point] = [(neighbour, LSHash.euclidean_dist_square(p, neighbour)) for neighbour in neighbours]

        #     heap = []

        #     for _, v in candidates[point]:
        #         if len(heap) < num_results:
        #             heapq.heappush(heap, -v)
        #         elif v < -heap[0]:
        #             heapq.heappushpop(heap, -v)

        #     output[point] = -heap[0]

        return output

    def euclidean_dist_square(x, y):
        diff = x - y
        return np.dot(diff, diff)
    