#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

template<typename T, unsigned int blockSize>
__global__ void CopyVector(T* dest, T* src, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*blockSize + threadIdx.x;
	if(i<n) {
		dest[i]	= src[i];
	}
}

template<typename T, unsigned int blockSize>
__global__ void CopyVectorInv(T* dest, T* src, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*blockSize + threadIdx.x;
	if(i<n) {
		dest[i]	= -src[i];
	}
}

template<typename T>
void copy_vector_device(T* d_dest, T* d_src, unsigned int size, bool invert)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

  if(invert) {
		CopyVectorInv<T,BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_dest, d_src, size);
  }
  else {
		CopyVector<T,BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_dest, d_src, size);
  }
  //CHECK_KERNEL()
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