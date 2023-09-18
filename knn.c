#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    int k;
    double* X_train;
    int* y_train;
    int num_samples;
    int num_features;
} KNN;

KNN* knn_create(int k, double* X_train, int* y_train, int num_samples, int num_features) {
    KNN* knn = (KNN*)malloc(sizeof(KNN));
    knn->k = k;
    knn->X_train = X_train;
    knn->y_train = y_train;
    knn->num_samples = num_samples;
    knn->num_features = num_features;
    return knn;
}

void knn_destroy(KNN* knn) {
    free(knn);
}

double euclidean_distance(double* x1, double* x2, int num_features) {
    double distance = 0.0;
    for (int i = 0; i < num_features; i++) {
        double diff = x1[i] - x2[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

void knn_find_5th_nearest(KNN* knn, double* kth_distances) {
    #pragma omp parallel for    
    for (int i = 0; i < knn->num_samples; i++) {
        double* x = &knn->X_train[i * knn->num_features];
        double fifth_nearest = INFINITY;
        for (int j = 0; j < knn->num_samples; j++) {
            if (i != j) {
                double* neighbor = &knn->X_train[j * knn->num_features];
                double distance = euclidean_distance(x, neighbor, knn->num_features);
                if (distance < fifth_nearest) {
                    fifth_nearest = distance;
                }
            }
        }
        kth_distances[i] = sqrt(fifth_nearest); // Store the 5th nearest distance
    }
}



int main() {
    // Example usage
    int k = 5;
    int num_samples = 10000;
    int num_features = 100;
    
    double* X_train = (double*)malloc(sizeof(double) * num_samples * num_features);
    int* y_train = (int*)malloc(sizeof(int) * num_samples);
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            X_train[i * num_features + j] = (double)rand() / RAND_MAX; // Random values between 0 and 1
        }
        y_train[i] = rand() % 2; // Random binary label (0 or 1)
    }

    double start_time = omp_get_wtime();

    KNN* knn = knn_create(k, X_train, y_train, num_samples, num_features);
    
    double kth_distances[num_samples];
    knn_find_5th_nearest(knn, kth_distances);

    double end_time = omp_get_wtime(); // Record end time
    printf("Execution time: %f seconds\n", end_time - start_time);

    // // Print the 5th nearest distances for all training data
    // for (int i = 0; i < num_samples; i++) {
    //     printf("5th nearest distance for data point %d: %lf\n", i, kth_distances[i]);
    // }
    
    knn_destroy(knn);
    
    free(X_train);
    free(y_train);
    
    return 0;
}
