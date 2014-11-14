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

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

  int xx, yy;
  for (int k = 0; k < (BLOCK_SIZE + ncolA - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.x;
  	yy = row;
		if (xx < ncolA && yy < nrowA) 
		 	As[threadIdx.y][threadIdx.x] = A[xx*nrowA + yy];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		xx = col;
		yy = k*BLOCK_SIZE + threadIdx.y;

		if (yy < nrowB && xx < ncolB)
			Bs[threadIdx.y][threadIdx.x] = B[xx*nrowB + yy];
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

__global__ void MatMulKernelTpA(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  double CValue = 0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

  int xx, yy;
  for (int k = 0; k < (BLOCK_SIZE + nrowA - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.x;
  	yy = row;
		if (yy < ncolA && xx < nrowA) 
		 	As[threadIdx.y][threadIdx.x] = A[yy*nrowA + xx];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		xx = col;
		yy = k*BLOCK_SIZE + threadIdx.y;

		if (yy < nrowB && xx < ncolB)
			Bs[threadIdx.y][threadIdx.x] = B[xx*nrowB + yy];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < ncolA && col < ncolB)
  	C[ (blockIdx.x*BLOCK_SIZE + threadIdx.x)*ncolA + blockIdx.y*BLOCK_SIZE+threadIdx.y] = CValue;
}

__global__ void MatMulKernelTpB(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  double CValue = 0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

  int xx, yy;

  for (int k = 0; k < (BLOCK_SIZE + ncolA - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.x;
  	yy = row;

		if (xx < ncolA && yy < nrowA) 
		 	As[threadIdx.y][threadIdx.x] = A[xx*nrowA + yy];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		xx = col;
		yy = k*BLOCK_SIZE + threadIdx.y;

		if (xx < nrowB && yy < ncolB)
			Bs[threadIdx.y][threadIdx.x] = B[yy*nrowB + xx];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrowA && col < nrowB)
  	C[ (blockIdx.x*BLOCK_SIZE + threadIdx.x)*nrowA + blockIdx.y*BLOCK_SIZE+threadIdx.y] = CValue;
}
