/**
 * @file utils.h
 * @author Karthick T. Sharma
 * @brief Helper functions to calculate core distances.
 *
 * @date 2022-12-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdbool.h>

#define EPSILON 1e-15 /**< Tolerance for double precision comparison */

/**
 * @brief Check if two arrays are equal.
 *
 * Iterate through each feature of data point and check if its same or not.
 *
 * @param arr_1 First array to be checked.
 * @param arr_2 Second array to be checked.
 * @param size Number of features in the array.
 * @return true If all features of both arrays are equal.
 * @return false If any single feature of both arrays are different.
 */
bool are_arrays_eq(const double arr_1[], const double arr_2[], int size);

/**
 * @brief Calculate Euclidean squared distance between two points in n-dims space.
 *
 * @param point_1 Point 1.
 * @param point_2 Point 2.
 * @param dims Number of features in points.
 * @return double Euclidean squared distance between points.
 */
double calc_sq_dist(const double point_1[], const double point_2[], int dims);

#endif