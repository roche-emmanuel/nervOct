#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void ComputeGradient(unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int grad_offset,
	unsigned int nrows, unsigned int ncols, unsigned int niter, double* nn_params, double* inputs, double* deltas, double* grads) 
{

}
