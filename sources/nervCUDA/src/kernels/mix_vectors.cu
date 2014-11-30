#include <nervCUDA.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

template<typename T>
__global__ void MixVectors(T* d_res, T* d_vec1, T* d_vec2, T w1, T w2, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(i<n) {
		d_res[i]= w1*d_vec1[i] + w2*d_vec2[i];
	}
}

template<typename T>
void mix_vectors_device(T* d_res, T* d_vec1, T* d_vec2, T w1, T w2, unsigned int size, cudaStream_t stream)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

	MixVectors<<<dimGrid, dimBlock, 0, stream>>>(d_res, d_vec1, d_vec2, w1, w2, size);
  // CHECK_KERNEL()
}

template <typename T>
void _mix_vectors(T* res, T* vec1, T* vec2, T w1, T w2, unsigned int n)
{
	size_t size;

	size = n * sizeof(T);
	T* d_res = NULL;
	checkCudaErrors(cudaMalloc(&d_res, size));
	// checkCudaErrors(cudaMemcpy(d_res, nn_params, size, cudaMemcpyHostToDevice));
	T* d_vec1 = NULL;
	checkCudaErrors(cudaMalloc(&d_vec1, size));
	checkCudaErrors(cudaMemcpy(d_vec1, vec1, size, cudaMemcpyHostToDevice));

	T* d_vec2 = NULL;
	checkCudaErrors(cudaMalloc(&d_vec2, size));
	checkCudaErrors(cudaMemcpy(d_vec2, vec2, size, cudaMemcpyHostToDevice));

 	mix_vectors_device(d_res, d_vec1, d_vec2, w1, w2, n);

	checkCudaErrors(cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_vec1));
	checkCudaErrors(cudaFree(d_vec2));
}

extern "C" {

void mix_vectors(double* res, double* vec1, double* vec2, double w1, double w2, unsigned int n)
{
	_mix_vectors(res,vec1,vec2,w1,w2,n);
}

void mix_vectors_f(float* res, float* vec1, float* vec2, float w1, float w2, unsigned int n)
{
	_mix_vectors(res,vec1,vec2,w1,w2,n);
}

}
