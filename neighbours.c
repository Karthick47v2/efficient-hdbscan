/**
 * @file neighbours.c
 * @author Karthick T. Sharma
 * @brief Store nearest neighbours for data points.
 *
 * @date 2022-12-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "neighbours.h"

void init_neighbors(struct Neighbours *neighbours, int max_neighbours, int dims)
{
    neighbours->num_neighbours = 0;
    neighbours->max_neighbours = max_neighbours;
    neighbours->dims = dims;
    neighbours->search_stop_idx = 0;
    neighbours->data_points = malloc(max_neighbours * sizeof(double *));

    if (neighbours->data_points == NULL)
    {
        fprintf(stderr, "\nError: Out of Memory\n\n");
        exit(-1);
    }

    for (int i = 0; i < max_neighbours; i++)
    {
        neighbours->data_points[i] = malloc(dims * sizeof(double));
    }
    if (neighbours->data_points[max_neighbours - 1] == NULL)
    {
        fprintf(stderr, "\nError: Out of Memory\n\n");
        exit(-1);
    }
}

void add_neighbor(struct Neighbours *neighbours, double data_point[])
{
    if (neighbours->num_neighbours >= neighbours->max_neighbours)
    {
        int new_max_neighbours = neighbours->max_neighbours * 2;
        double **temp = realloc(neighbours->data_points, new_max_neighbours * sizeof(double *));

        if (temp == NULL)
        {
            fprintf(stderr, "\nError: Out of Memory\n\n");
            exit(-1);
        }

        for (int i = neighbours->max_neighbours; i < new_max_neighbours; i++)
        {
            temp[i] = malloc(neighbours->dims * sizeof(double));
        }

        if (temp[new_max_neighbours - 1] == NULL)
        {
            fprintf(stderr, "\nError: Out of Memory\n\n");
            exit(-1);
        }

        neighbours->max_neighbours = new_max_neighbours;
        neighbours->data_points = temp;
    }

    // for (int i = 0; i < neighbours->search_stop_idx; i++)
    // {
    //     if (are_arrays_eq(neighbours->data_points[i], data_point, neighbours->dims))
    //     {
    //         return;
    //     }
    // }
    memcpy(neighbours->data_points[neighbours->num_neighbours], data_point, neighbours->dims * sizeof(double));
    neighbours->num_neighbours++;
}