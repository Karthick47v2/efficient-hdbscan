#include <cuda_runtime.h>

extern "C" double callCudaCalcSqDist(const double *point1, const double *point2, int dims)
{
    double *dev_point1, *dev_point2, *dev_result;
    double result = 0.0;

    cudaMalloc((void **)&dev_point1, dims * sizeof(double));
    cudaMalloc((void **)&dev_point2, dims * sizeof(double));
    cudaMalloc((void **)&dev_result, sizeof(double));

    cudaMemcpy(dev_point1, point1, dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_point2, point2, dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dev_result, 0, sizeof(double));

    cudaCalcSqDist<<<1, dims>>>(dev_point1, dev_point2, dev_result, dims);

    cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_point1);
    cudaFree(dev_point2);
    cudaFree(dev_result);

    return result;
}