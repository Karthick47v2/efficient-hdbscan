__global__ void calc_sq_dist_kernel(const double *train_data, const double *data_points, double *distances, int input_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    for (int j = 0; j < input_dim; j++)
    {
        double diff = train_data[i * input_dim + j] - data_points[i * input_dim + j];
        sum += diff * diff;
    }
    distances[i] = sum;
}

void kernel_wrapper(const double *train_data, const double *data_points, int input_dim, double *distances, int num)
{
    double *d_distances; // Device array for distances
    cudaMalloc((void **)&d_distances, num * sizeof(double));

    // Calculate distances using the CUDA kernel
    calc_sq_dist_kernel<<<1, 512>>>(train_data, data_points, d_distances, input_dim);

    // Copy distances from the device to the host
    cudaMemcpy(distances, d_distances, num * sizeof(double), cudaMemcpyDeviceToHost);

    // cudaFree(d_distances);
}
