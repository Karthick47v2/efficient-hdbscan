// # gcc-13 -shared -o nn_dist.so -fPIC nn_dist.c quickselect.c -lm -fopenmp
// # gcc -shared -o nn_dist.so -fPIC nn_dist.c euclidean_kernel.o -lcudart -L/usr/local/cuda/lib64

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

#include "quickselect.h"

struct Neighbours{
    int num_neighbours;
    int max_neighbours;
    int dims;
    double** data_points;
};

bool areArraysEqual(const double arr1[], const double arr2[], int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > 1e-9) {
            return false;
        }
    }
    return true;
}

void initNeighbors(struct Neighbours* neighbours, int max_neighbours, int dims) {
    neighbours->num_neighbours = 0;
    neighbours->max_neighbours = max_neighbours;
    neighbours->dims = dims;
    neighbours->data_points = malloc(max_neighbours * sizeof(double*));
    if (neighbours->data_points == NULL) {
        exit(1);
    }

    for (int i = 0; i < max_neighbours; i++) {
        neighbours->data_points[i] = malloc(dims * sizeof(double));
        if (neighbours->data_points[i] == NULL) {
            exit(1);
        }
    }
}

// Add a neighbor ID to the neighbors struct
void addNeighbor(struct Neighbours* neighbours, double data_point[]) {
    if (neighbours->num_neighbours < neighbours->max_neighbours) {
        for (int i = 0; i < neighbours->num_neighbours; i++) {
            if (areArraysEqual(neighbours->data_points[i], data_point, neighbours->dims)){
                return;
            }
        }
        memcpy(neighbours->data_points[neighbours->num_neighbours], data_point, neighbours->dims * sizeof(double));
        neighbours->num_neighbours++;
    }
}

// Calculate the Euclidean distance between two points (arrays)
double calculateEuclideanSquaredDistance(const double point1[], const double point2[], int dims) {
    double sum = 0.0;
    for (int i = 0; i < dims; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}


// Example function that performs some calculation on the input data
double* calculate_core_dist(double**** input_data, int** num_elements, int* num_lists, int num_table, int input_dim, double** train_data, int num_train_data, int max_neighbour, int k) {

    // printf("this is it %d", k)
    double* core_distances = (double*)malloc(num_train_data * sizeof(double));

    #pragma omp parallel for
        for (int i = 0; i < num_train_data; i++){
            struct Neighbours current_neighbours;
            initNeighbors(&current_neighbours, max_neighbour, input_dim);

            for (int j = 0; j < num_table; j++){
                bool is_found = false;

                for (int k = 0; k < num_lists[j] && !is_found; k++){

                    for (int l = 0; l < num_elements[j][k]; l++){
                        
                        if (areArraysEqual(train_data[i], input_data[j][k][l], input_dim)){
                            for (int m = 0; m < num_elements[j][k]; m++){
                                if (l != m){
                                    addNeighbor(&current_neighbours, input_data[j][k][m]);
                                }
                            }
                            is_found = true;
                            break;
                        }
                    }
                }
            }

            // printf("%d\n", current_neighbours.num_neighbours);

            double* distances = (double*)malloc(current_neighbours.num_neighbours * sizeof(double));

            // Calculate Euclidean distances with train_data[i]
            for (int n = 0; n < current_neighbours.num_neighbours; n++) {
                distances[n] = calculateEuclideanSquaredDistance(train_data[i], current_neighbours.data_points[n], input_dim);
            }

            if (k > current_neighbours.num_neighbours){
                k = current_neighbours.num_neighbours;
            }

            core_distances[i] = sqrt(quickSelect(distances, 0, current_neighbours.num_neighbours - 1, k-2));

            free(distances);
            

            for (int n = 0; n < current_neighbours.num_neighbours; n++) {
                free(current_neighbours.data_points[n]);
            }
            free(current_neighbours.data_points);
        }
    return core_distances;
}
