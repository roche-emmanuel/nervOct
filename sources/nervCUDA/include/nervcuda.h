
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

#include <nerv/BPTraits.h>
#include <nerv/BPDeviceTraits.h>
#include <nerv/RandDeviceTraits.h>

using namespace nerv;

namespace nerv
{

template<typename T>
T random_real(T mini, T maxi)
{
  return mini + (maxi - mini) * (T)rand() / (T)RAND_MAX;
}

inline unsigned int random_uint(unsigned int mini, unsigned int maxi)
{
  return mini + (unsigned int)floor(0.5 + (maxi - mini) * (double)rand() / (double)RAND_MAX);
}

}

extern "C" {

  void matmult(unsigned int nrowA, unsigned int ncolA, const double *A,
               unsigned int nrowB, unsigned int ncolB, const double *B, double *C, bool tpA, bool tpB);

  void matmult_f(unsigned int nrowA, unsigned int ncolA, const float *A,
                 unsigned int nrowB, unsigned int ncolB, const float *B, float *C, bool tpA, bool tpB);

  void costFunc(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
                double *nn_params, double *X, double *yy, double lambda, double &J, double *gradients, double *deltas, double *inputs);

  void costFunc_device(unsigned int nl, unsigned int np, unsigned int *lsizes, unsigned int nsamples,
                       double *d_params, double *d_X, double *d_yy, double lambda, double &J, double *d_grads, double *d_deltas, double *d_inputs, double *d_regw);

  void costFuncCPU(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
                   double *nn_params, double *X, double *yy, double lambda, double *activation, unsigned int ninputs, double *inputs, double &J, double *gradients, double *deltas);

  void reductionCPU(double *inputs, unsigned int n, double &output);

  void reduce_sum(double *inputs, unsigned int n, double &output);
  void reduce_sum_f(float *inputs, unsigned int n, float &output);

  void reduce_cost(double *hx, double *yy, unsigned int n, double &output);
  void reduce_cost_f(float *hx, float *yy, unsigned int n, float &output);

  void reduce_cost_reg(double *params, double *regweights, unsigned int n, double &output);
  void reduce_cost_reg_f(float *params, float *regweights, unsigned int n, float &output);

  void cgtrainCPU(unsigned int nl, unsigned int nsamples, unsigned int nparams,
                  unsigned int *lsizes, double *X, double *yy, double *init_params,
                  double lambda, unsigned int maxiter, double *params);

  void cgtrain(unsigned int nl, unsigned int nsamples, unsigned int nparams,
               unsigned int *lsizes, double *X, double *yy, double *init_params,
               double lambda, unsigned int maxiter, double *params);

  void copy_vector(double *dest, double *src, unsigned int size, bool invert = false);
  void copy_vector_f(float *dest, float *src, unsigned int size, bool invert = false);

  void mix_vectors(double *res, double *vec1, double *vec2, double w1, double w2, unsigned int size);
  void mix_vectors_f(float *res, float *vec1, float *vec2, float w1, float w2, unsigned int size);

  double compute_length2(double *vec, unsigned int size);
  float compute_length2_f(float *vec, unsigned int size);

  double compute_dot(double *vec1, double *vec2, unsigned int size);
  float compute_dot_f(float *vec1, float *vec2, unsigned int size);

  void gd_errfunc(BPTraits<double> &traits);
  void gd_errfunc_f(BPTraits<float> &traits);

  void gd_errfunc_cpu(BPTraits<double>& traits);

  void nn_predict(BPTraits<double> &traits);
  void nn_predict_f(BPTraits<float> &traits);

  void nn_predict_cpu(BPTraits<double> &traits);
  void nn_predict_cpu_f(BPTraits<float> &traits);

  void rand_weights(RandTraits<double>& traits);
  void rand_weights_f(RandTraits<float>& traits);
};

template<typename T>
void rand_weights_device(RandDeviceTraits<T>& traits);
// template<typename T>
// void rand_weights_device_debug(RandDeviceTraits<T>& traits);

template<typename T>
void reduce_cost_device(T *d_hx, T *d_yy, unsigned int n, T &output, cudaStream_t stream = 0);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
int nn_activation_device(BPDeviceTraits<T> &d_traits);

template<typename T, unsigned int blockSize = BLOCK_SIZE>
void gd_errfunc_device(BPDeviceTraits<T> &d_traits);

template<typename T>
T compute_dot_device(T *d_vec1, T *d_vec2, T *d_redtmp, unsigned int size);

template<typename T>
T compute_length2_device(T *d_vec, T *d_redtmp, unsigned int size);

template<typename T>
void copy_vector_device(T *d_s, T *d_df1, unsigned int size, bool invert = false);

template<typename T>
void mix_vectors_device(T *d_res, T *d_vec1, T *d_vec2, T w1, T w2, unsigned int size, cudaStream_t stream = 0);

template<typename T>
void matmult_device(unsigned int nrowA, unsigned int ncolA, unsigned int nrowB, unsigned int ncolB,
                    const T *d_A, const T *d_B, T *d_C, bool tpA, bool tpB);

template<typename T>
void reduce_sum_launcher(int size, int threads, int blocks, int whichKernel, T *d_idata, T *d_odata, cudaStream_t stream = 0);

template<typename T>
void reduce_cost_reg_device(T *d_params, T *d_regw, unsigned int n, T &output, cudaStream_t stream = 0);

#endif
