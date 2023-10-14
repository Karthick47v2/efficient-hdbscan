/**
 * @file quickselect.h
 * @author Karthick T. Sharma
 * @brief Select kth minimum value from array.
 *
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef QUICKSELECT_H
#define QUICKSELECT_H

/**
 * @brief Select kth minimum element from unsorted array.
 *
 * @param array Unsorted array.
 * @param low Left index range.
 * @param high Right index range.
 * @param k Rank of element being sought.
 * @return double
 */
double quick_select(double *array, int low, int high, int k);

#endif
