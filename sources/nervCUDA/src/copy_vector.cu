#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

__global__ void CopyVector(double* dest, double* src, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(i<n) {
		dest[i]	= src[i];
	}
}

__global__ void CopyVectorInv(double* dest, double* src, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(i<n) {
		dest[i]	= -src[i];
	}
}

void copy_vector_device(double* d_dest, double* d_src, unsigned int size, bool invert)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

  if(invert) {
		CopyVectorInv<<<dimGrid, dimBlock>>>(d_dest, d_src, size);
  }
  else {
		CopyVector<<<dimGrid, dimBlock>>>(d_dest, d_src, size);
  }
  CHECK_KERNEL()
}

extern "C" {

void copy_vector(double* dest, double* src, unsigned int n, bool invert)
{
	size_t size;

	size = n * sizeof(double);
	double* d_dest = NULL;
	checkCudaErrors(cudaMalloc(&d_dest, size));
	// checkCudaErrors(cudaMemcpy(d_dest, nn_params, size, cudaMemcpyHostToDevice));
	double* d_src = NULL;
	checkCudaErrors(cudaMalloc(&d_src, size));
	checkCudaErrors(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));

 	copy_vector_device(d_dest, d_src, n, invert);

	checkCudaErrors(cudaMemcpy(dest, d_dest, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_dest));
	checkCudaErrors(cudaFree(d_src));
}

}