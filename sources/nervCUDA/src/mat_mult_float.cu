#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>
#include <iostream>

__global__ void MatMulKernelf(unsigned int nrowA, unsigned int ncolA, const float* A,
    unsigned int nrowB, unsigned int ncolB, const float* B, float* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  float CValue = 0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+1]; // Adding +1 to avoid shared memory bank conflict
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+1];

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
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrowA && col < ncolB)
  	C[ (blockIdx.x*BLOCK_SIZE + threadIdx.x)*nrowA + blockIdx.y*BLOCK_SIZE+threadIdx.y] = CValue;
}

extern "C" {

void multiplyMatricesf(unsigned int nrowA, unsigned int ncolA, const float* A,
    unsigned int nrowB, unsigned int ncolB, const float* B, float* C, bool tpA, bool tpB)
{
	// Allocate the device memory:
	size_t size;

	size = nrowA * ncolA * sizeof(float);
	float* d_A = NULL;
	checkCudaErrors(cudaMalloc(&d_A, size));
	checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

	size = nrowB * ncolB * sizeof(float);
	float* d_B = NULL;
	checkCudaErrors(cudaMalloc(&d_B, size));
	checkCudaErrors(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

	size = (tpA ? ncolA : nrowA) * (tpB ? nrowB : ncolB) * sizeof(float);
	float* d_C = NULL;
	checkCudaErrors(cudaMalloc(&d_C, size));
	// cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); // no need to set this.

	// Call the kernel directly:
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((BLOCK_SIZE + (tpB ? nrowB : ncolB)-1)/BLOCK_SIZE, (BLOCK_SIZE + (tpA ? ncolA : nrowA)-1)/BLOCK_SIZE);
	// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");

	if(tpA) {
		logDEBUG("TpA not supported for mat_mult_float");
		// MatMulKernelTpA<<<dimGrid, dimBlock>>>(nrowA, ncolA, d_A, nrowB, ncolB, d_B, d_C);
	}
	else if(tpB) {
		logDEBUG("TpB not supported for mat_mult_float");
		// MatMulKernelTpB<<<dimGrid, dimBlock>>>(nrowA, ncolA, d_A, nrowB, ncolB, d_B, d_C);
	}
	else {
		MatMulKernelf<<<dimGrid, dimBlock>>>(nrowA, ncolA, d_A, nrowB, ncolB, d_B, d_C);
	}

	// Read C from device memory
	checkCudaErrors(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
	// logDEBUG("Copy C off of device: "<<cudaGetErrorString(err));

	// Free device memory
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
}

}
