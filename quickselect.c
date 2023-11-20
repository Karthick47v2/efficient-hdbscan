/**
 * @file quickselect.c
 * @author Karthick T. Sharma
 * @brief Select kth minimum value from array.
 *
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "quickselect.h"
#include "stdio.h"

/**
 * @brief Swap variables.
 *
 * @param a Variable to be swapped.
 * @param b Variable to be swapped.
 */
void swap(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * @brief Partition the array by pivot.
 *
 * Find kth smallest element by only sorting partial array using partitioning until
 * kth elements comes as lower index.
 *
 * @param arr Array to be partitioned.
 * @param low lower index range.
 * @param high higer index range.
 * @return int lower index range after partitioned.
 */
int partition(double *arr, int low, int high)
{
    double pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    // Swap arr[i+1] and arr[high] (pivot)
    swap(&arr[i + 1], &arr[high]);

    return i + 1;
}

int quick_select(double *arr, int low, int high, int k)
{
    while (low <= high)
    {
        int pivot_idx = partition(arr, low, high);

        if (k == pivot_idx)
        {
            return pivot_idx;
        }
        else if (k < pivot_idx)
            high = pivot_idx - 1;
        else
            low = pivot_idx + 1;
    }

    return low;
}