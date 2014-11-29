#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
__global__ void MatMult(unsigned int nrowA, unsigned int niter, unsigned int ncolB, const T* A, const T* B, T* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  T CValue = 0;

  __shared__ T As[blockSize][blockSize+1]; // Adding +1 to avoid shared memory bank conflict
  __shared__ T Bs[blockSize][blockSize+1];

  int xx, yy;
  for (int k = 0; k < (blockSize + niter - 1)/blockSize; k++) {

  	// Here we try to access the A matrix data in a coaleased way:
  	// keeping in mind that A is row major. So we need to read A per column
  	// while the threads in the wrap are (probably) organized by row.
  	// So we invert the roles palyed by threadIdx.x and threadIdx.y.
  	xx = k*blockSize + threadIdx.y;
  	yy = blockIdx.y*blockSize + threadIdx.x;
		if (xx < niter && yy < nrowA) 
		 	As[threadIdx.x][threadIdx.y] = A[xx*nrowA + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;


		// Same for the B matrix, we need to invert the x and y coords:
		xx = blockIdx.x*blockSize + threadIdx.y;
		yy = k*blockSize + threadIdx.x;

		if (yy < niter && xx < ncolB)
			Bs[threadIdx.x][threadIdx.y] = B[xx*niter + yy];
		else
			Bs[threadIdx.x][threadIdx.y] = 0.0;

		__syncthreads();

		for (int n = 0; n < blockSize; ++n) 
			CValue += As[threadIdx.x][n] * Bs[n][threadIdx.y];

		__syncthreads();
  }

  int row = blockIdx.y*blockSize + threadIdx.x;
  int col = blockIdx.x*blockSize + threadIdx.y;

  if (row < nrowA && col < ncolB)
  	C[col*nrowA + row] = CValue;
}

template<typename T, unsigned int blockSize>
__global__ void MatMultTpA(unsigned int nrowA, unsigned int ncolA, const T* A,
    unsigned int nrowB, unsigned int ncolB, const T* B, T* C) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  T CValue = 0;

  int row = blockIdx.y*blockSize + threadIdx.y;
  int col = blockIdx.x*blockSize + threadIdx.x;

  __shared__ T As[blockSize][blockSize+1];
  __shared__ T Bs[blockSize][blockSize+1];

  int xx, yy;
  for (int k = 0; k < (blockSize + nrowA - 1)/blockSize; k++) {

  	xx = k*blockSize + threadIdx.x;
  	yy = row;
  	
		if (yy < ncolA && xx < nrowA) 
		 	As[threadIdx.y][threadIdx.x] = A[yy*nrowA + xx];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		xx = blockIdx.x*blockSize + threadIdx.y;
		yy = k*blockSize + threadIdx.x;

		if (yy < nrowB && xx < ncolB)
			Bs[threadIdx.x][threadIdx.y] = B[xx*nrowB + yy];
		else
			Bs[threadIdx.x][threadIdx.y] = 0.0;

		__syncthreads();

		for (int n = 0; n < blockSize; ++n) 
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < ncolA && col < ncolB)
  	C[ (blockIdx.x*blockSize + threadIdx.x)*ncolA + blockIdx.y*blockSize+threadIdx.y] = CValue;
}

template<typename T, unsigned int blockSize>
__global__ void MatMultTpB(unsigned int nrowC, unsigned int niter, unsigned int ncolC, const T* A, const T* B, T* C) 
{
	// unsigned int nrowA, unsigned int ncolA,
 //    unsigned int nrowB, unsigned int ncolB,) {

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  T CValue = 0;

  int row = blockIdx.y*blockSize + threadIdx.y;
  int col = blockIdx.x*blockSize + threadIdx.x;

  __shared__ T As[blockSize][blockSize+1];
  __shared__ T Bs[blockSize][blockSize+1];

  int xx, yy;

  for (int k = 0; k < (blockSize + niter - 1)/blockSize; k++) {

  	xx = k*blockSize + threadIdx.y;
  	yy = blockIdx.y*blockSize + threadIdx.x;

		if (xx < niter && yy < nrowC) 
		 	As[threadIdx.x][threadIdx.y] = A[xx*nrowC + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;

		xx = col;
		yy = k*blockSize + threadIdx.y;

		if (xx < ncolC && yy < niter)
			Bs[threadIdx.y][threadIdx.x] = B[yy*ncolC + xx];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < blockSize; ++n) 
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrowC && col < ncolC)
  	C[ (blockIdx.x*blockSize + threadIdx.x)*nrowC + blockIdx.y*blockSize+threadIdx.y] = CValue;
}

extern "C" {


void multiplyMatrices(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB)
{
	// Allocate the device memory:
	size_t size;

	size = nrowA * ncolA * sizeof(double);
	double* d_A = NULL;
	checkCudaErrors(cudaMalloc(&d_A, size));
	checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

	size = nrowB * ncolB * sizeof(double);
	double* d_B = NULL;
	checkCudaErrors(cudaMalloc(&d_B, size));
	checkCudaErrors(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

	size = (tpA ? ncolA : nrowA) * (tpB ? nrowB : ncolB) * sizeof(double);
	double* d_C = NULL;
	checkCudaErrors(cudaMalloc(&d_C, size));
	// cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); // no need to set this.

	// Call the kernel directly:
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((BLOCK_SIZE + (tpB ? nrowB : ncolB)-1)/BLOCK_SIZE, (BLOCK_SIZE + (tpA ? ncolA : nrowA)-1)/BLOCK_SIZE);
	// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");

	if(tpA) {
		MatMultTpA<<<dimGrid, dimBlock>>>(nrowA, ncolA, d_A, nrowB, ncolB, d_B, d_C);
	}
	else if(tpB) {
		MatMultTpB<<<dimGrid, dimBlock>>>(nrowA, ncolA, nrowB, d_A, d_B, d_C);
	}
	else {
		MatMult<<<dimGrid, dimBlock>>>(nrowA, ncolA, ncolB, d_A, d_B, d_C);
	}

	// Read C from device memory
	checkCudaErrors(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
	// logDEBUG("Copy C off of device: "<<cudaGetErrorString(err));

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

}
