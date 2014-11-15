#ifndef NERV_KERNELS_H_
#define NERV_KERNELS_H_

#define BLOCK_SIZE 32

__global__ void MatMulKernel(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C);

__global__ void MatMulKernelTpB(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C);

__global__ void MatMulKernelTpA(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C);

__global__ void CostFuncKernel(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
		double* nn_params, double* X, double* yy, double lambda);

__global__ void ComputeActivation(unsigned int theta_offset, unsigned int input_offset, unsigned int next_input_offset,
	unsigned int nrows, unsigned int ncols, unsigned int ncolT, double* nn_params, double* inputs, double* X);

#endif
