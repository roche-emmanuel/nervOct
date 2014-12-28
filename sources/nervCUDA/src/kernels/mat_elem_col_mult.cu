#include <nervCUDA.h>
#include <nerv_kernels.h>
#include "cublas_v2.h"

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatElemColMult(unsigned int nrows, unsigned int ncols, T *A, T *y)
{
  int c = blockIdx.x * blockSize + threadIdx.x;
  int r = blockIdx.y * blockSize + threadIdx.y;

  if (c < ncols && r < nrows)
  {
    A[nrows * c + r] *= y[c];
  }
}

template<typename T, unsigned int blockSize = BLOCK_SIZE>
__global__ void MatElemColDiv(unsigned int nrows, unsigned int ncols, T *A, T *y)
{
  int c = blockIdx.x * blockSize + threadIdx.x;
  int r = blockIdx.y * blockSize + threadIdx.y;

  if (c < ncols && r < nrows)
  {
    A[nrows * c + r] /= y[c];
  }
}

template<typename T, unsigned int blockSize>
void mat_elem_col_mult_device(unsigned int nrows, unsigned int ncols, T *A, T *y)
{
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((BLOCK_SIZE + ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + nrows-1)/BLOCK_SIZE);
  MatElemColMult<<<dimGrid, dimBlock>>>(nrows, ncols, A, y);
}

template<typename T, unsigned int blockSize>
void mat_elem_col_div_device(unsigned int nrows, unsigned int ncols, T *A, T *y)
{
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((BLOCK_SIZE + ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + nrows-1)/BLOCK_SIZE);
  MatElemColDiv<<<dimGrid, dimBlock>>>(nrows, ncols, A, y);
}

template <typename T>
void _mat_elem_col_mult(unsigned int nrows, unsigned int ncols, T *A, T *y, bool div)
{
  size_t size;

  size = nrows * ncols * sizeof(T);
  T *d_A = NULL;
  checkCudaErrors(cudaMalloc(&d_A, size));
  checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

  size = ncols * sizeof(T);
  T *d_y = NULL;
  checkCudaErrors(cudaMalloc(&d_y, size));
  checkCudaErrors(cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice));

  if (div)
  {
    mat_elem_col_div_device(nrows, ncols, d_A, d_y);
  }
  else
  {
    mat_elem_col_mult_device(nrows, ncols, d_A, d_y);
  }

  copyFromDevice(A, d_A, nrows * ncols);

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_y));
}

template <typename T>
void _mat_elem_col_mult_cpu(unsigned int nrows, unsigned int ncols, T *A, T *y, bool div)
{
  // perform the matrix multiplication:
  unsigned int nr = nrows;
  unsigned int nc = ncols;
  for (unsigned int r = 0; r < nr; ++r)
  {
    for (unsigned int c = 0; c < nc; ++c)
    {
      if (div)
      {
        A[nrows * c + r] /= y[c];
      }
      else
      {
        A[nrows * c + r] *= y[c];
      }
    }
  }
}

extern "C" {

  void mat_elem_col_mult(unsigned int nrows, unsigned int ncols, double *A, double *y, bool div)
  {
    _mat_elem_col_mult(nrows, ncols, A, y, div);
  }

  void mat_elem_col_mult_f(unsigned int nrows, unsigned int ncols, float *A, float *y, bool div)
  {
    _mat_elem_col_mult(nrows, ncols, A, y, div);
  }

  void mat_elem_col_mult_cpu(unsigned int nrows, unsigned int ncols, double *A, double *y, bool div)
  {
    _mat_elem_col_mult_cpu(nrows, ncols, A, y, div);
  }
}
