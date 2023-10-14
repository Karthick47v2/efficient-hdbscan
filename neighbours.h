/**
 * @file neighbours.h
 * @author Karthick T. Sharma
 * @brief Store nearest neighbours for data points.
 *
 * @date 2022-12-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "utils.h"

/**
 * @brief Structure to hold details about nearest neighbours.
 *
 */
struct Neighbours
{
    int num_neighbours;   /**< Current number of neighbours */
    int max_neighbours;   /**< Maximum number of neighbours */
    int dims;             /**< Number of features */
    int search_stop_idx;  /**< Stopping index for duplicate checking in each bucket */
    double **data_points; /**< Pointer to array of neighbours */
};

/**
 * @brief Initialize Neighbours struct with default values.
 *
 * Init struct and add default values for variables and create dynamic arrays.
 *
 * @param neighbours Pointer to Neighbours struct.
 * @param max_neighbours Maximum numbers for neighbours to current data point.
 * @param dims Number of features of data points.
 */
void init_neighbors(struct Neighbours *neighbours, int max_neighbours, int dims);

/**
 * @brief Add neighbours to Neighbours struct.
 *
 * Add neighbours to struct if and only if its not previously added.
 *
 * @param neighbours Pointer to Neighbours struct.
 * @param data_point Array of values of neighbour data point.
 */
void add_neighbor(struct Neighbours *neighbours, double data_point[]);

#endif