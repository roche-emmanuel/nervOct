#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void MatMulKernel(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  double CValue = 0;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1]; // Adding +1 to avoid shared memory bank conflict
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

  int xx, yy;
  for (int k = 0; k < (BLOCK_SIZE + ncolA - 1)/BLOCK_SIZE; k++) {

  	// Here we try to access the A matrix data in a coaleased way:
  	// keeping in mind that A is row major. So we need to read A per column
  	// while the threads in the wrap are (probably) organized by row.
  	// So we invert the roles palyed by threadIdx.x and threadIdx.y.
  	xx = k*BLOCK_SIZE + threadIdx.y;
  	yy = blockIdx.y*BLOCK_SIZE + threadIdx.x;
		if (xx < ncolA && yy < nrowA) 
		 	As[threadIdx.x][threadIdx.y] = A[xx*nrowA + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;


		// Same for the B matrix, we need to invert the x and y coords:
		xx = blockIdx.x*BLOCK_SIZE + threadIdx.y;
		yy = k*BLOCK_SIZE + threadIdx.x;

		if (yy < nrowB && xx < ncolB)
			Bs[threadIdx.x][threadIdx.y] = B[xx*nrowB + yy];
		else
			Bs[threadIdx.x][threadIdx.y] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			CValue += As[threadIdx.x][n] * Bs[n][threadIdx.y];

		__syncthreads();
  }

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.x;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.y;

  if (row < nrowA && col < ncolB)
  	C[col*nrowA + row] = CValue;
}

__global__ void MatMulKernelTpA(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  double CValue = 0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

  int xx, yy;
  for (int k = 0; k < (BLOCK_SIZE + nrowA - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.x;
  	yy = row;
  	
		if (yy < ncolA && xx < nrowA) 
		 	As[threadIdx.y][threadIdx.x] = A[yy*nrowA + xx];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		xx = blockIdx.x*BLOCK_SIZE + threadIdx.y;
		yy = k*BLOCK_SIZE + threadIdx.x;

		if (yy < nrowB && xx < ncolB)
			Bs[threadIdx.x][threadIdx.y] = B[xx*nrowB + yy];
		else
			Bs[threadIdx.x][threadIdx.y] = 0.0;

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

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

  int xx, yy;

  for (int k = 0; k < (BLOCK_SIZE + ncolA - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.y;
  	yy = blockIdx.y*BLOCK_SIZE + threadIdx.x;

		if (xx < ncolA && yy < nrowA) 
		 	As[threadIdx.x][threadIdx.y] = A[xx*nrowA + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;

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
