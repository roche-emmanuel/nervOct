#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>
#include <iostream>

#define logDEBUG(msg) std::cout << msg << std::endl;

extern "C" {

void multiplyMatrices(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB)
{
	// Allocate the device memory:
	size_t size;
	cudaError_t err;

	size = nrowA * ncolA * sizeof(double);
	double* d_A = NULL;
	err = cudaMalloc(&d_A, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc A: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	size = nrowB * ncolB * sizeof(double);
	double* d_B = NULL;
	err = cudaMalloc(&d_B, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc B: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	size = (tpA ? ncolA : nrowA) * (tpB ? nrowB : ncolB) * sizeof(double);
	double* d_C = NULL;
	err = cudaMalloc(&d_C, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc C: "<<cudaGetErrorString(err));
	}
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
	err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	// logDEBUG("Copy C off of device: "<<cudaGetErrorString(err));

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void costFunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* nn_params, double* X, double* yy, double lambda, double* inputs)
{
	// Allocate the device memory:
	size_t size;
	cudaError_t err;

	size = nl * sizeof(unsigned int);
	double* d_lsizes = NULL;
	err = cudaMalloc(&d_lsizes, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc lsizes: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_lsizes, lsizes, size, cudaMemcpyHostToDevice);

	// Compute the total number of parameters in this network:
	unsigned int np = 0;
	unsigned int nt = nl-1; // number of matrices evolved.

	for(unsigned int i=0;i<nt;++i) {
		np += lsizes[i+1]*(lsizes[i]+1);
	}

	size = np * sizeof(double);
	double* d_params = NULL;
	err = cudaMalloc(&d_params, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc params: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_params, nn_params, size, cudaMemcpyHostToDevice);

	// Prepare the X matrix:
	size = sizeof(double) * nsamples * lsizes[0];
	double* d_X = NULL;
	err = cudaMalloc(&d_X, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc X: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);


	// Prepare the input data:
	// the size of each input matrix is lsize[i+1]*nsamples;
	// and we need input 0 to nt-1, inclusive.
	// So that's nl input matrices.
	unsigned int count = 0;
	for(unsigned int i=0;i<nt;++i) {
		count += lsizes[i+1];
	}

	size = nsamples * count * sizeof(double);
	size_t input_size = size;
	double* d_inputs = NULL;
	err = cudaMalloc(&d_inputs, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc inputs: "<<cudaGetErrorString(err));
	}
	cudaMemset(d_inputs,0,size); // This is needed for debugging only.

	// Copy the label matrix:	
	size = nsamples * lsizes[nt] * sizeof(double);
	double* d_yy = NULL;
	err = cudaMalloc(&d_yy, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc yy: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_yy, yy, size, cudaMemcpyHostToDevice);


	// offset used to locate the theta_i matrix in the d_params array.
	unsigned int theta_offset = 0;

	// Offset used for the z(i) matrix on iteration i
	unsigned int input_offset = 0;

	unsigned int next_input_offset = 0; //nsamples*lsizes[1];

  for(unsigned int i=0; i<nt;++i) {
  	// We compute the activation and input values for the given layer:

  	// The kernel compute the values of zi and a(i+1) 
  	// (note that the value or a(0) is already loaded in the Activation vector).
  	// even if we compute the a(i+1) matrix we actually discard completely the first column
  	// in this matrix (colu of intercept terms). As a result we just need to mapped the GPU grid to
  	// the dimension of of the sub z(i) matrix (which is transposed.)
  	// THe dimensions for z(i) are: lsize(i+1) * nsamples
  	// When this is transposed we get: nsamples * lsize(i+1);
		unsigned int nrows = lsizes[i+1];
		unsigned int ncolT = lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.
		unsigned int ncols = nsamples;

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((BLOCK_SIZE + ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + nrows-1)/BLOCK_SIZE);

		// Also we will need access to the theta_i matrix so we need to keep track of its global offset in the
		// network parameters array.
		// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");
		ComputeActivation<<<dimGrid, dimBlock>>>(theta_offset, input_offset, next_input_offset,
			nrows, ncols, ncolT, d_params, d_inputs, d_X);

		// update the offsets:
		theta_offset += lsizes[i+1]*(lsizes[i]+1);
		input_offset = next_input_offset;
		next_input_offset += nrows*ncols;
  }

	// Read inputs from device memory
	err = cudaMemcpy(inputs, d_inputs, input_size, cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA reading inputs: "<<cudaGetErrorString(err));
	}

	// Free device memory
	cudaFree(d_lsizes);
	cudaFree(d_params);
	cudaFree(d_inputs);	
	cudaFree(d_yy);	
	cudaFree(d_X);	
}

}