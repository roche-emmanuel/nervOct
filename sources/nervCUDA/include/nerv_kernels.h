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

#define CHECK_KERNEL() checkCudaErrors( cudaPeekAtLastError() ); \
checkCudaErrors( cudaDeviceSynchronize() );

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}

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

__global__ void ComputeGradient(unsigned int theta_offset, int input_offset, unsigned int delta_offset, unsigned int grad_offset,
    unsigned int nrows, unsigned int ncols, unsigned int niter, double* X, double* nn_params, double* inputs, double* deltas, double* grads, double lambda); 

void reduce(int size, int threads, int blocks, int whichKernel, double *d_idata, double *d_odata);

void copy_vector_device(double* d_s, double* d_df1, unsigned int size, bool invert = false);
void mix_vectors_device(double* d_res, double* d_vec1, double* d_vec2, double w1, double w2, unsigned int size);

double compute_length2_device(double* d_vec, double* d_redtmp, unsigned int size);
double compute_dot_device(double* d_vec1, double* d_vec2, double* d_redtmp, unsigned int size);

#endif
