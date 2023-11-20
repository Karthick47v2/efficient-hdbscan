// # gcc-13 -shared -o nn_dist.so -fPIC nn_dist.c quickselect.c -lm -fopenmp

#include <omp.h>
#include <time.h>
// #include <cuda.h>

#include "neighbours.h"
#include "quickselect.h"

struct Edge
{
    double *src;   /**< Source of edge */
    double *dst;   /**< Destination of edge */
    double weight; /**< Weight of the edge */
};

// extern void kernel_wrapper(const double *train_data, const double *data_points, int input_dim);

// int compare(const void *a, const void *b)
// {
//     if (*(double *)a < *(double *)b)
//         return -1;
//     if (*(double *)a > *(double *)b)
//         return 1;
//     return 0;
// }

// void removeDuplicates(double arr[], int *size)
// {
//     if (*size <= 1)
//     {
//         return; // No duplicates to remove
//     }

//     // Sort the array in ascending order
//     qsort(arr, *size, sizeof(double), compare);

//     int newSize = 1;

//     for (int i = 1; i < *size; i++)
//     {
//         if (arr[i] != arr[i - 1])
//         {
//             arr[newSize] = arr[i];
//             newSize++;
//         }
//     }

//     *size = newSize; // Update the size parameter
// }

double *calc_core_dist(double ****data_points, int **num_data_points, int *num_lists, int num_hashtables, int input_dim, double **train_data, int num_train_data, int max_neighbour, int k)
{
    // double *core_distances = (double *)malloc(num_train_data * sizeof(double));
    double *edges = (Edge *)malloc(num_train_data * sizeof(Edge));

#pragma omp parallel for
    for (int i = 0; i < num_train_data; i++)
    {
        struct Neighbours current_neighbours;
        init_neighbors(&current_neighbours, max_neighbour, input_dim);
        add_neighbor(&current_neighbours, train_data[i]);

        for (int j = 0; j < num_hashtables; j++)
        {
            bool is_found = false;
            current_neighbours.search_stop_idx = current_neighbours.num_neighbours;

            for (int k = 0; k < num_lists[j] && !is_found; k++)
            {

                for (int l = 0; l < num_data_points[j][k]; l++)
                {

                    if (are_arrays_eq(train_data[i], data_points[j][k][l], input_dim))
                    {
                        for (int m = 0; m < num_data_points[j][k]; m++)
                        {
                            if (l != m)
                            {
                                add_neighbor(&current_neighbours, data_points[j][k][m]);
                            }
                        }
                        is_found = true;
                        break;
                    }
                }
            }
        }

        double *distances = (double *)malloc(current_neighbours.num_neighbours * sizeof(double));

        // kernel_wrapper(train_data[i], current_neighbours.data_points, input_dim, distances, current_neighbours.num_neighbours);

        for (int n = 0; n < current_neighbours.num_neighbours; n++)
        {
            distances[n] = calc_sq_dist(train_data[i], current_neighbours.data_points[n], input_dim); /////////////// USE CUDA HERE
        }

        // calc_sq_dist(train_data[i], )

        // removeDuplicates(distances, &current_neighbours.num_neighbours);

        if (k > current_neighbours.num_neighbours)
        {
            k = current_neighbours.num_neighbours;
        }

        int kth_point = quick_select(distances, 0, current_neighbours.num_neighbours - 1, k - 1);

        // core_distances[i] = sqrt(distances[kth_point]);
        // core_distances[i] = sqrt(distances[k - 1]);

        struct Edge edge;

        edge.dst = sqrt(distances[kth_point]);
        edge.src = train_data[i];
        edge.dst = current_neighbours.data_points[kth_point]; // need to assign all dims recursive;ly

        edges[i] = Edge()

            free(distances);

        for (int n = 0; n < current_neighbours.num_neighbours; n++)
        {
            free(current_neighbours.data_points[n]);
        }
        free(current_neighbours.data_points);
    }

    return core_distances;
}

// double *calc_mrd()

/**
 * @brief Free memory allocated for array of core distances.
 *
 * @param rand_ptr Pointer to array of core distances.
 */
void free_rand_ptr(double *rand_ptr)
{
    free(rand_ptr);
}

#define NUM_TRAIN_DATA 10000
#define NUM_LIST 10000
#define LEN_PER_LIST 1000
#define INPUT_DIM 10
#define MAX_NEIGH 1024

int main()
{
    int k = 2;
    double train_data[NUM_TRAIN_DATA][INPUT_DIM];
    double data_points[NUM_LIST][LEN_PER_LIST][INPUT_DIM];

    for (int i = 0; i < NUM_TRAIN_DATA; i++)
    {
        for (int j = 0; j < INPUT_DIM; j++)
        {
            // Replace these values with your actual data
            train_data[i][j] = i * INPUT_DIM + j + 1;
        }
    }

    // Adding data to the data_points array
    for (int i = 0; i < NUM_LIST; i++)
    {
        for (int j = 0; j < LEN_PER_LIST; j++)
        {
            int point = (int)(NUM_TRAIN_DATA * rand() / (double)RAND_MAX);
            for (int k = 0; k < INPUT_DIM; k++)
            {
                // Replace these values with your actual data
                data_points[i][j][k] = train_data[point][k];
            }
        }
        for (int k = 0; k < INPUT_DIM; k++)
        {
            data_points[i][i % LEN_PER_LIST][k] = train_data[i][k];
        }
    }

    double *core_distances = (double *)malloc(NUM_TRAIN_DATA * sizeof(double));

    // #pragma omp parallel for
    for (int i = 0; i < NUM_TRAIN_DATA; i++)
    {
        struct Neighbours current_neighbours;
        init_neighbors(&current_neighbours, MAX_NEIGH, INPUT_DIM);
        add_neighbor(&current_neighbours, train_data[i]);
        bool is_found = false;

        for (int j = 0; j < NUM_LIST && !is_found; j++)
        {
            for (int k = 0; k < LEN_PER_LIST; k++)
            {
                if (are_arrays_eq(train_data[i], data_points[j][k], INPUT_DIM))
                {
                    for (int m = 0; m < LEN_PER_LIST; m++)
                    {
                        if (m != k)
                        {
                            add_neighbor(&current_neighbours, data_points[j][m]);
                        }
                    }
                    is_found = true;
                    break;
                }
            }
        }

        double *distances = (double *)malloc(current_neighbours.num_neighbours * sizeof(double));

        for (int n = 0; n < current_neighbours.num_neighbours; n++)
        {
            distances[n] = calc_sq_dist(train_data[i], current_neighbours.data_points[n], INPUT_DIM); // ########## USE CUDA HERE
        }

        if (k > current_neighbours.num_neighbours)
        {
            k = current_neighbours.num_neighbours;
        }

        core_distances[i] = sqrt(quick_select(distances, 0, current_neighbours.num_neighbours - 1, k - 1));

        free(distances);

        for (int n = 0; n < current_neighbours.num_neighbours; n++)
        {
            free(current_neighbours.data_points[n]);
        }
        free(current_neighbours.data_points);
    }
}