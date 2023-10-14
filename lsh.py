import numpy as np
from collections import defaultdict
import ctypes

np.random.seed(42)

random_generator = ctypes.CDLL('./rand_gen.so')
random_generator.generate_random_numbers.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
random_generator.generate_random_numbers.restype = ctypes.POINTER(
    ctypes.c_double)
random_generator.free_rand_ptr.argtypes = [ctypes.POINTER(ctypes.c_double)]

core_dist = ctypes.CDLL('./core_dist.so')
core_dist.calc_core_dist.argtypes = [
    ctypes.POINTER(ctypes.POINTER(
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int]

core_dist.calc_core_dist.restype = ctypes.POINTER(ctypes.c_double)


def generate_random_planes(num_hashtables, hash_size, input_dim):
    random_numbers_ptr = random_generator.generate_random_numbers(
        ctypes.c_int(num_hashtables), ctypes.c_int(hash_size), ctypes.c_int(input_dim))
    random_numbers = np.ctypeslib.as_array(
        random_numbers_ptr, shape=(num_hashtables, hash_size, input_dim))
    # random_generator.free_rand_ptr(random_numbers_ptr) ### FREEING RESULT NULL

    return random_numbers


class LSHash(object):
    def __init__(self, hash_size, input_dim, num_hashtables=1):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        # self.uniform_planes = np.random.randn(
        #     self.num_hashtables, self.hash_size, self.input_dim)
        # print(self.uniform_planes)
        # yep.start('random_check.prof')
        self.uniform_planes = generate_random_planes(
            self.num_hashtables, self.hash_size, self.input_dim)
        self.hash_tables = [defaultdict(list)
                            for _ in range(self.num_hashtables)]

    def add_points(self, input_points):
        self.input_points = input_points
        for i, table in enumerate(self.hash_tables):
            planes = self.uniform_planes[i]
            projections = np.dot(planes, input_points.T)
            hash_strings = [x.tobytes() for x in (projections > 0).T]

            for h, point in zip(hash_strings, input_points):
                table[h].append(point)
            # print(table.values())

            # print(len(table))
            # print(list(map(len, table.values())))

            # temp = []
            # keys = []
            # for k in table.keys():
            #     if len(table[k]) < 5:
            #         temp.extend(table[k])
            #         keys.append(k)

            # table[1] = temp

            # for k in keys:
            #     del table[k]

            # print(len(table))
            # print(list(map(len, table.values())))

    def nn_dist(self, num_results=0):
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


        max_arr = [v for n in num_elements_ for v in n]
        max_arr = int(np.percentile(max_arr, 80)) * self.num_hashtables

        # max_arr -= 1
        # for i in range(6):  # Perform a maximum of 6 iterations for a 32-bit integer
        #     max_arr |= max_arr >> (2 ** i)

        # max_arr += 1
        # # max_arr = 10

        print('dma size', max_arr)

        n_list = len(dict_list)


        c_input_data = (ctypes.POINTER(ctypes.POINTER(
            ctypes.POINTER(ctypes.c_double))) * n_list)()
        num_inner_lists = (ctypes.c_int * n_list)()
        num_elements = (ctypes.POINTER(ctypes.c_int) * n_list)()

        for i, outer_list in enumerate(dict_list):
            num_inner_lists[i] = num_bins[i]
            num_elements[i] = (ctypes.c_int * num_bins[i])()
            c_input_data[i] = (ctypes.POINTER(
                ctypes.POINTER(ctypes.c_double)) * num_bins[i])()

            for j, inner_list in enumerate(outer_list):
                num_elements[i][j] = num_elements_[i][j]
                c_input_data[i][j] = (ctypes.POINTER(
                    ctypes.c_double) * num_elements_[i][j])()

                for k, arr in enumerate(inner_list):
                    c_input_data[i][j][k] = arr.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double))

        c_array = (ctypes.POINTER(ctypes.c_double) * len(self.input_points))()
        for i, row in enumerate(self.input_points):
            c_array[i] = row.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# FIND BEST VALUES FOR MAX_NEIGHBOUR, HASH_TABLE, HYPERPLANE
# Speed depends on max neigh for C dym alloc
        result_ptr = core_dist.calc_core_dist(c_input_data, num_elements, num_inner_lists, ctypes.c_int(n_list), ctypes.c_int(
            self.input_dim), c_array, ctypes.c_int(len(self.input_points)), ctypes.c_int(max_arr), ctypes.c_int(num_results))

        results = np.ctypeslib.as_array(
            result_ptr, shape=(len(self.input_points),))

        return results
        # print(results)

        # output = {}
        # candidates = defaultdict(set)

        # for table in self.hash_tables:
        #     for values in table.values():
        #         temp = list(map(tuple, values))
        #         for v in values:
        #             candidates[tuple(v)].update(temp)

        # for point, neighbours in candidates.items():
        #     p = np.array(point)
        #     candidates[point] = [(neighbour, LSHash.euclidean_dist_square(
        #         p, neighbour)) for neighbour in neighbours]

        #     heap = []

        #     for _, v in candidates[point]:
        #         if len(heap) < num_results:
        #             heapq.heappush(heap, -v)
        #         elif v < -heap[0]:
        #             heapq.heappushpop(heap, -v)

        #     output[point] = np.sqrt(-heap[0])

        # print(list(output.keys()))
        # print(list(output.values()))
        # # return list(output.values())

    def euclidean_dist_square(x, y):
        diff = x - y
        return np.dot(diff, diff)
