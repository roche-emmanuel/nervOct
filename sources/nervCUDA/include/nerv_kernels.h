#ifndef NERV_KERNELS_H_
#define NERV_KERNELS_H_

#include <nervcuda.h>


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

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatMult(unsigned int nrowC, unsigned int niter, unsigned int ncolC, const T *A, const T *B, T *C);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatMultTpB(unsigned int nrowC, unsigned int niter, unsigned int ncolC, const T *A, const T *B, T *C);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatMultTpA(unsigned int nrowC, unsigned int niter, unsigned int ncolC, const T *A, const T *B, T *C);

__global__ void CostFuncKernel(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
                               double *nn_params, double *X, double *yy, double lambda);

template <typename T, bool withDropout = false, bool debugMode = false, unsigned int blockSize = BLOCK_SIZE>
__global__ void ComputeActivation(BPComputeTraits<T> traits);

template <typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void ComputeActivationWithDropout(BPComputeTraits<T> traits);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void InitLastDelta(BPComputeTraits<T> traits);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void ComputeDelta(BPComputeTraits<T> traits);

template<typename T, bool withDropout = false, unsigned int blockSize = BLOCK_SIZE>
__global__ void ComputeGradient(BPComputeTraits<T> traits);

#endif
