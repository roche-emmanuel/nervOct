#ifndef NERV_KERNELS_H_
#define NERV_KERNELS_H_

#include <nervcuda.h>

#define logDEBUG(msg) std::cout << msg << std::endl;

#define BLOCK_SIZE 32

#define MAX_THREADS_PER_BLOCK 1024


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
