#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

// gcc -shared -o hdb.so hdb_c.c -O3 -march=native -lm -fopenmp -fPIC

int compare_doubles(const void *a, const void *b)
{
    return (*(double *)a > *(double *)b) - (*(double *)a < *(double *)b);
}

double euclidean_distance(double *x, double *y, int dim)
{
    double sum = 0.0;

    for (int i = 0; i < dim; i++)
    {
        double diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sqrt(sum);
    // return sum;
}

void calc_distance_matrix(double *data, unsigned long long n, int dim, double *distance_matrix)
{
    #pragma omp parallel for schedule(dynamic, 4)
    for (unsigned long long i = 0; i < n; i++)
    {
        for (unsigned long long j = 0; j < n; j++)
        {
            double distance = euclidean_distance(&data[i * dim], &data[j * dim], dim);
            distance_matrix[i * n + j] = distance;
        }
    }

    #pragma omp parallel for schedule(dynamic, 2)
    for (unsigned long long i = 0; i < n; i++)
    {
        for (unsigned long long j =  0; j < i; j++)
        {
            distance_matrix[i * n + j] = distance_matrix[j * n + i];
        }
    }
}

void swap(double *const a, double *const b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}

unsigned long long partition(double *const arr, const unsigned long long low, const unsigned long long high)
{
    const double pivot = arr[high];
    unsigned long long i = low - 1;

    for (unsigned long long j = low; j < high; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

double quickSelect(double *const arr, const unsigned long long low, const unsigned long long high, const int k)
{
    if (low <= high)
    {
        const int pivotIndex = partition(arr, low, high);

        if (pivotIndex == k)
        {
            return arr[pivotIndex];
        }
        else if (pivotIndex < k)
        {
            return quickSelect(arr, pivotIndex + 1, high, k);
        }
        else
        {
            return quickSelect(arr, low, pivotIndex - 1, k);
        }
    }

    return -1.0;
}

void calc_core_dist(unsigned long long n, int k, double *core_dist, double *distance_matrix)
{

#pragma omp parallel for
    for (unsigned long long i = 0; i < n; i++)
    {
        double *temp_array = (double *)malloc(n * sizeof(double));
        memcpy(temp_array, &distance_matrix[i * n], n * sizeof(double));
        core_dist[i] = quickSelect(temp_array, 0, n - 1, k - 1);
    }
}

double *calc_mutual_reachability_dist(double *data, unsigned long long n, int dim, int k)
{
    double *distance_matrix = (double *)malloc(n * n * sizeof(double));
    double *core_dist = (double *)malloc(n * sizeof(double));
    double *mutual_reach_dist = (double *)malloc(n * n * sizeof(double));

    calc_distance_matrix(data, n, dim, distance_matrix);
    calc_core_dist(n, k, core_dist, distance_matrix);

#pragma omp parallel for // schedule(static, 2)
    for (unsigned long long i = 0; i < n; i++)
    {
        for (unsigned long long j = i + 1; j < n; j++)
        {
            mutual_reach_dist[i * n + j] = fmax(fmax(core_dist[i], core_dist[j]), distance_matrix[i * n + j]);
            mutual_reach_dist[j * n + i] = mutual_reach_dist[i * n + j];
        }
    }
    return mutual_reach_dist;
}