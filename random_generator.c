// # gcc-13 -shared -o random_generator.so -fPIC random_generator.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to generate random numbers
double* generate_random_numbers(int num_hashtables, int hash_size, int input_dim) {
    // Seed the random number generator
    srand(42);
    
    // Calculate the total number of elements
    int total_elements = num_hashtables * hash_size * input_dim;
    
    // Allocate memory for the array
    double* random_numbers = (double*)malloc(total_elements * sizeof(double));
    if (random_numbers == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }
    
    // Fill the array with random values between 0 and 1
    for (int i = 0; i < total_elements; i++) {
        random_numbers[i] = (rand() * 20.0 / (double) RAND_MAX) - 10.0;
    }
    
    return random_numbers;
}
