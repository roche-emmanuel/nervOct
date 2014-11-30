
#ifndef NERV_CUDA_H_
#define NERV_CUDA_H_

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
    #  if defined( NERVCUDA_LIBRARY_STATIC )
    #    define NERVCUDA_EXPORT
    #  elif defined( NERVCUDA_LIBRARY )
    #    define NERVCUDA_EXPORT   __declspec(dllexport)
    #  else
    #    define NERVCUDA_EXPORT   __declspec(dllimport)
    #  endif
#else
    #  define NERVCUDA_EXPORT
#endif

#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

#define MAX_THREADS_PER_BLOCK 1024

#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

#define CHECK_KERNEL() 
// checkCudaErrors( cudaPeekAtLastError() ); \
// checkCudaErrors( cudaDeviceSynchronize() );

extern "C" bool isPow2(unsigned int x);

unsigned int nextPow2(unsigned int x);

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

const char *_cudaGetErrorEnum(cudaError_t error);

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


extern "C" {

void matmult(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

void matmult_f(unsigned int nrowA, unsigned int ncolA, const float* A,
    unsigned int nrowB, unsigned int ncolB, const float* B, float* C, bool tpA, bool tpB);

void costFunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
    double* nn_params, double* X, double* yy, double lambda, double& J, double* gradients, double* deltas, double* inputs);

void costFunc_device(unsigned int nl, unsigned int np, unsigned int* lsizes, unsigned int nsamples,
    double* d_params, double* d_X, double* d_yy, double lambda, double& J, double* d_grads, double* d_deltas, double* d_inputs, double* d_regw);

void costFuncCPU(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
    double* nn_params, double* X, double* yy, double lambda, double* activation, unsigned int ninputs, double* inputs, double& J, double* gradients, double* deltas);

void reductionCPU(double* inputs, unsigned int n, double& output);

void reduce_sum(double* inputs, unsigned int n, double& output);
void reduce_sum_f(float* inputs, unsigned int n, float& output);

void reduce_cost(double* hx, double* yy, unsigned int n, double& output);
void reduce_cost_f(float* hx, float* yy, unsigned int n, float& output);

void reduce_cost_reg(double* params, double* regweights, unsigned int n, double& output);
void reduce_cost_reg_f(float* params, float* regweights, unsigned int n, float& output);

void cgtrainCPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
    unsigned int* lsizes, double* X, double* yy, double* init_params, 
    double lambda, unsigned int maxiter, double* params);

void cgtrain(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
    unsigned int* lsizes, double* X, double* yy, double* init_params, 
    double lambda, unsigned int maxiter, double* params);

void copy_vector(double* dest, double* src, unsigned int size, bool invert = false);
void copy_vector_f(float* dest, float* src, unsigned int size, bool invert = false);

void mix_vectors(double* res, double* vec1, double* vec2, double w1, double w2, unsigned int size);
void mix_vectors_f(float* res, float* vec1, float* vec2, float w1, float w2, unsigned int size);

double compute_length2(double* vec, unsigned int size);
float compute_length2_f(float* vec, unsigned int size);

double compute_dot(double* vec1, double* vec2, unsigned int size);
float compute_dot_f(float* vec1, float* vec2, unsigned int size);

void gd_errfunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
    double* nn_params, double* X, double* yy, double lambda, double& J, double* gradients, double* deltas, double* inputs);
void gd_errfunc_f(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
    float* nn_params, float* X, float* yy, float lambda, float& J, float* gradients, float* deltas, float* inputs);

};

template<typename T>
void reduce_cost_device(T* d_hx, T* d_yy, unsigned int n, T& output, cudaStream_t stream = 0);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
void gd_errfunc_device(unsigned int nl, unsigned int np, unsigned int* lsizes, unsigned int nsamples,
    T* d_params, T* d_X, T* d_yy, T lambda, T& J, T* d_grads, T* d_deltas, T* d_inputs, T* d_regw, cudaStream_t stream = 0);

template<typename T>
T compute_dot_device(T* d_vec1, T* d_vec2, T* d_redtmp, unsigned int size);

template<typename T>
T compute_length2_device(T* d_vec, T* d_redtmp, unsigned int size);

template<typename T>
void copy_vector_device(T* d_s, T* d_df1, unsigned int size, bool invert = false);

template<typename T>
void mix_vectors_device(T* d_res, T* d_vec1, T* d_vec2, T w1, T w2, unsigned int size, cudaStream_t stream = 0);

template<typename T>
void matmult_device(unsigned int nrowA, unsigned int ncolA, unsigned int nrowB, unsigned int ncolB, 
    const T* d_A, const T* d_B, T* d_C, bool tpA, bool tpB);

template<typename T>
void reduce_sum_launcher(int size, int threads, int blocks, int whichKernel, T *d_idata, T *d_odata, cudaStream_t stream = 0);

template<typename T>
void reduce_cost_reg_device(T* d_params, T* d_regw, unsigned int n, T& output, cudaStream_t stream = 0);

#endif
