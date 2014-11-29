#ifndef NERV_KERNELS_H_
#define NERV_KERNELS_H_

#include <nervcuda.h>

#define logDEBUG(msg) std::cout << msg << std::endl;



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

#define CHECK_KERNEL() 

// checkCudaErrors( cudaPeekAtLastError() ); \
// checkCudaErrors( cudaDeviceSynchronize() );

extern "C" bool isPow2(unsigned int x);

unsigned int nextPow2(unsigned int x);

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatMult(unsigned int nrowA, unsigned int niter, unsigned int ncolB, const T* A, const T* B, T* C);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatMultTpB(unsigned int nrowA, unsigned int ncolA, const T* A,
    unsigned int nrowB, unsigned int ncolB, const T* B, T* C);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatMultTpA(unsigned int nrowA, unsigned int ncolA, const T* A,
    unsigned int nrowB, unsigned int ncolB, const T* B, T* C);

__global__ void CostFuncKernel(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
		double* nn_params, double* X, double* yy, double lambda);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void ComputeActivation(unsigned int theta_offset, unsigned int input_offset, unsigned int next_input_offset,
	unsigned int nrows, unsigned int ncols, unsigned int ncolT, T* nn_params, T* inputs, T* X);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void InitLastDelta(unsigned int nrows, unsigned int ncols, T* deltas, T* hx, T* yy);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void ComputeDelta(unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int next_delta_offset,
    unsigned int nrows, unsigned int ncols, unsigned int niter, T* nn_params, T* inputs, T* deltas);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void ComputeGradient(unsigned int theta_offset, int input_offset, unsigned int delta_offset, unsigned int grad_offset,
    unsigned int nrows, unsigned int ncols, unsigned int niter, T* X, T* nn_params, T* inputs, T* deltas, T* grads, T lambda); 

void reduce(int size, int threads, int blocks, int whichKernel, double *d_idata, double *d_odata);


#endif
