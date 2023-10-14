/**
 * @file utils.c
 * @author Karthick T. Sharma
 * @brief Helper functions to calculate core distances.
 *
 * @date 2022-12-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "utils.h"

bool are_arrays_eq(const double arr_1[], const double arr_2[], int size)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(arr_1[i] - arr_2[i]) > EPSILON)
        {
            return false;
        }
    }
    return true;
}

double calc_sq_dist(const double point_1[], const double point_2[], int dims)
{
    double sum = 0.0;
    for (int i = 0; i < dims; i++)
    {
        double diff = point_1[i] - point_2[i];
        sum += diff * diff;
    }
    return sum;
}