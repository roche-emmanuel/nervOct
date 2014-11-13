#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void MatMulKernel(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  double CValue = 0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int k = 0; k < (BLOCK_SIZE + ncolA - 1)/BLOCK_SIZE; k++) {

		if (k*BLOCK_SIZE + threadIdx.x < ncolA && row < nrowA) 
		 	As[threadIdx.y][threadIdx.x] = A[(k*BLOCK_SIZE + threadIdx.x)*nrowA + row];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		if (k*BLOCK_SIZE + threadIdx.y < nrowB && col < ncolB)
			Bs[threadIdx.y][threadIdx.x] = B[col*nrowB + k*BLOCK_SIZE + threadIdx.y];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrowA && col < ncolB)
  	C[ (blockIdx.x*BLOCK_SIZE + threadIdx.x)*nrowA + blockIdx.y*BLOCK_SIZE+threadIdx.y] = CValue;
}
