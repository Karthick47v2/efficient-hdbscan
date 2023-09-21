#include "quickselect.h"

#include <stdio.h>

void swap(double *a, double *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(double* array, int low, int high) {
    double pivot = array[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (array[j] < pivot) { //////////////////////// TODO: CHECK IF ITS OK
            i++;
            swap(&array[i], &array[j]);
        }
    }

    // Swap array[i+1] and array[high] (pivot)
    swap(&array[i + 1], &array[high]);

    return i + 1;
}


double quickSelect(double* array, int low, int high, int k) {
    while (low <= high) {
        int pivotIndex = partition(array, low, high);

        if (k == pivotIndex)
            return array[pivotIndex];
        else if (k < pivotIndex)
            high = pivotIndex - 1;
        else
            low = pivotIndex + 1;
    }

    return array[low];
}