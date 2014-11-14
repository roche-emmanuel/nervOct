#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void CostFuncKernel(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* nn_params, double* X, double* yy, double lambda) {
  
}
