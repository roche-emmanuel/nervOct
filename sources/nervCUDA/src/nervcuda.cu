#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>
#include <iostream>

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
		MatMulKernelTpA<<<dimGrid, dimBlock>>>(nrowA, ncolA, d_A, nrowB, ncolB, d_B, d_C);
	}
	else if(tpB) {
		MatMulKernelTpB<<<dimGrid, dimBlock>>>(nrowA, ncolA, d_A, nrowB, ncolB, d_B, d_C);
	}
	else {
		MatMulKernel<<<dimGrid, dimBlock>>>(nrowA, ncolA, d_A, nrowB, ncolB, d_B, d_C);
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
