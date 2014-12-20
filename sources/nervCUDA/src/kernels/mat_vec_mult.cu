#include <nervCUDA.h>
#include <nerv_kernels.h>
#include "cublas_v2.h"

// Method used to compute op(A)*x using CUBLAS:
template<typename T>
void mat_vec_mult_device(cublasHandle_t handle, cublasOperation_t trans, unsigned int nrows, unsigned int ncols, T *A, T* x, T* y)
{
	T alpha = (T)1;
	cublasSgemv(handle, trans, nrows, ncols, &alpha, A, nrows, x, 1, NULL, y, 1);
}

template<>
void mat_vec_mult_device<double>(cublasHandle_t handle, cublasOperation_t trans, unsigned int nrows, unsigned int ncols, double *A, double* x, double* y)
{
	double alpha = (double)1;
	cublasDgemv(handle, trans, nrows, ncols, &alpha, A, nrows, x, 1, NULL, y, 1);
}

template <typename T>
void _mat_vec_mult(unsigned int nrows, unsigned int ncols, T *A, T* x, T* y, bool tpA)
{
  size_t size;

  size = nrows*ncols * sizeof(T);
  T *d_A = NULL;
  checkCudaErrors(cudaMalloc(&d_A, size));
  checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

  size = (tpA ? nrows : ncols) * sizeof(T);
  T *d_x = NULL;
  checkCudaErrors(cudaMalloc(&d_x, size));
  checkCudaErrors(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));

  size = (tpA ? ncols : nrows) * sizeof(T);
  T *d_y = NULL;
  checkCudaErrors(cudaMalloc(&d_y, size));
  // checkCudaErrors(cudaMemcpy(d_y, vec2, size, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
	checkCublasErrors(cublasCreate(&handle));

	// cudaStream_t stream;
	// checkCublasErrors(cublasSetStream(handle, stream));

  mat_vec_mult_device(handle, tpA ? CUBLAS_OP_T : CUBLAS_OP_N, nrows, ncols, d_A, d_x, d_y);

	checkCublasErrors(cublasDestroy(handle));

  copyFromDevice(y, d_y, (tpA ? ncols : nrows));

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
}

extern "C" {

  void mat_vec_mult(unsigned int nrows, unsigned int ncols, double *A, double* x, double* y, bool tpA)
  {
    _mat_vec_mult(nrows,ncols,A,x,y,tpA);
  }

  void mat_vec_mult_f(unsigned int nrows, unsigned int ncols, float *A, float* x, float* y, bool tpA)
  {
    _mat_vec_mult(nrows,ncols,A,x,y,tpA);
  }

}
