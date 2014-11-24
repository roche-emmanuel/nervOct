#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

__global__ void MixVectors(double* d_res, double* d_vec1, double* d_vec2, double w1, double w2, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(i<n) {
		d_res[i]= w1*d_vec1[i] + w2*d_vec2[i];
	}
}

void mix_vectors_device(double* d_res, double* d_vec1, double* d_vec2, double w1, double w2, unsigned int size)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

	MixVectors<<<dimGrid, dimBlock>>>(d_res, d_vec1, d_vec2, w1, w2, size);
  CHECK_KERNEL()
}
