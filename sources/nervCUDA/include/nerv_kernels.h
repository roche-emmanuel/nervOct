#ifndef NERV_KERNELS_H_
#define NERV_KERNELS_H_

#include <cuda_runtime.h>
#include <iostream>

#define logDEBUG(msg) std::cout << msg << std::endl;

#define BLOCK_SIZE 32

#define MAX_THREADS_PER_BLOCK 1024

#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

const char *_cudaGetErrorEnum(cudaError_t error);

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

extern "C" bool isPow2(unsigned int x);

unsigned int nextPow2(unsigned int x);

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

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

__global__ void InitLastDelta(unsigned int nrows, unsigned int ncols, double* deltas, double* hx, double* yy);

__global__ void ComputeDelta(unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int next_delta_offset,
    unsigned int nrows, unsigned int ncols, unsigned int niter, double* nn_params, double* inputs, double* deltas);

__global__ void ComputeGradient(unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int grad_offset,
    unsigned int nrows, unsigned int ncols, unsigned int niter, double* nn_params, double* inputs, double* deltas, double* grads); 

void reduce(int size, int threads, int blocks, int whichKernel, double *d_idata, double *d_odata);

#endif
