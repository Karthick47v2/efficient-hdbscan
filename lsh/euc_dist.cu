// # nvcc -o euclidean_kernel.o -c euclidean.cu

__global__ void calculateEuclideanDistances(double* train_point, double* neighbors, double* distances, int input_dim, int num_neighbors) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_neighbors) {
        double sum = 0.0;
        for (int i = 0; i < input_dim; i++) {
            double diff = train_point[i] - neighbors[tid * input_dim + i];
            sum += diff * diff;
        }
        distances[tid] = sum;
    }
}

// extern "C" void computeDistancesCuda(double* train_point, double* neighbor_points, double* distances, int input_dim, int num_neighbors) {
//     int numThreadsPerBlock = 256;
//     int numBlocks = (num_neighbors + numThreadsPerBlock - 1) / numThreadsPerBlock;
    
//     calculateEuclideanDistances<<<numBlocks, numThreadsPerBlock>>>(train_point, neighbor_points, distances, input_dim, num_neighbors);
//     cudaDeviceSynchronize();
// }