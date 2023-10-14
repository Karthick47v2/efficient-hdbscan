// # gcc-13 -shared -o random_generator.so -fPIC random_generator.c -lm
/**
 * @file _random_generator.c
 * @author Karthick T. Sharma
 * @brief Generate random numbers in large-scale.
 * @date 2022-12-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Generate random numbers.
 *
 * @param num_hashtables Number of hashtables.
 * @param hash_size Number of bits to be used for hash function.
 * @param input_dim Number of features of input data.
 * @return double* Array of random numbers.
 */
double *generate_random_numbers(int num_hashtables, int hash_size, int input_dim)
{
    // Seed random number generator
    srand(0);

    // Total numbers to generate
    int total_elements = num_hashtables * hash_size * input_dim;
    double *random_numbers = (double *)malloc(total_elements * sizeof(double));

    if (random_numbers == NULL)
    {
        fprintf(stderr, "\nError: Out of Memory\n\n");
        exit(-1);
    }

    for (int i = 0; i < total_elements; i++)
    {
        random_numbers[i] = (rand() * 2.0 / (double)RAND_MAX) - 1.0;
    }

    return random_numbers;
}

/**
 * @brief Free memory allocated for array of random numbers.
 *
 * @param rand_ptr Pointer to array of random numbers.
 */
void free_rand_ptr(double *rand_ptr)
{
    free(rand_ptr);
}
