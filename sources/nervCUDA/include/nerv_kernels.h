#ifndef NERV_KERNELS_H_
#define NERV_KERNELS_H_

#include <nervcuda.h>
#include <iostream>
#include <sstream>

#ifndef logDEBUG
#define logDEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl;
#endif

#ifndef logERROR
#define logERROR(msg) std::cout << "[ERROR] " << msg << std::endl;
#endif

#ifndef THROW
#define THROW(msg) { std::ostringstream os; os << msg; logERROR("Throwing exception: " << msg); throw std::runtime_error(os.str()); }
#endif

#ifndef THROW_IF
#define THROW_IF(cond,msg) if(cond) THROW(msg)
#endif


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
