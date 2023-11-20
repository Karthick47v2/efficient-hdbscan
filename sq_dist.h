#ifndef __SQ_DIST__
#define __SQ_DIST__

#include "cuda.h"
#include "cuda_runtime.h"

extern "C" void kernel_wrapper(const double *train_data, const double *data_points, int input_dim, double *distances);
#endif