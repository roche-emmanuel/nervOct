#include <nervCUDA.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

template<typename T>
__global__ void SpaeKLDivergence(T* d_kl, T* d_rho, T sp, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(i<n) {
		T r = d_rho[i];
		d_kl[i] = sp*log(sp/r) + (1.0 - sp)*log((1.0-sp)/(1.0-r));
	}
}

template<typename T>
void spae_kl_divergence_device(T* d_kl, T* d_rho, T sp, unsigned int size, cudaStream_t stream)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

	SpaeKLDivergence<<<dimGrid, dimBlock, 0, stream>>>(d_kl, d_rho, sp, size);
  // CHECK_KERNEL()
}

template <typename T>
void _spae_kl_divergence(T* kl, T* rho, T sp, unsigned int n)
{
	size_t size;

	size = n * sizeof(T);
	T* d_kl = NULL;
	checkCudaErrors(cudaMalloc(&d_kl, size));
	// checkCudaErrors(cudaMemcpy(d_kl, nn_params, size, cudaMemcpyHostToDevice));
	T* d_rho = NULL;
	checkCudaErrors(cudaMalloc(&d_rho, size));
	checkCudaErrors(cudaMemcpy(d_rho, rho, size, cudaMemcpyHostToDevice));

 	spae_kl_divergence_device(d_kl, d_rho, sp, n);

	checkCudaErrors(cudaMemcpy(kl, d_kl, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_kl));
	checkCudaErrors(cudaFree(d_rho));
}

extern "C" {

void spae_kl_divergence(double* kl, double* rho, double sp, unsigned int n)
{
	_spae_kl_divergence(kl,rho,sp,n);
}

void spae_kl_divergence_f(float* kl, float* rho, float sp, unsigned int n)
{
	_spae_kl_divergence(kl,rho,sp,n);
}

}
