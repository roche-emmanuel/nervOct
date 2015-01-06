#include <nervCUDA.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

template<typename T>
__global__ void SpaeSparseDelta(T* d_delta, T* d_rho, T beta, T sp, unsigned int n)
{
	// Retrieve the index for that thread:
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(i<n) {
		T r = d_rho[i];
		d_delta[i] = beta*( -sp/r + (1.0-sp)/(1.0-r));
	}
}

template<typename T>
void spae_sparse_delta_device(T* d_delta, T* d_rho, T beta, T sp, unsigned int size, cudaStream_t stream)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

	SpaeSparseDelta<<<dimGrid, dimBlock, 0, stream>>>(d_delta, d_rho, beta, sp, size);
  // CHECK_KERNEL()
}

template <typename T>
void _spae_sparse_delta(T* delta, T* rho, T beta, T sp, unsigned int n)
{
	size_t size;

	size = n * sizeof(T);
	T* d_delta = NULL;
	checkCudaErrors(cudaMalloc(&d_delta, size));
	// checkCudaErrors(cudaMemcpy(d_delta, nn_params, size, cudaMemcpyHostToDevice));
	
	T* d_rho = NULL;
	checkCudaErrors(cudaMalloc(&d_rho, size));
	checkCudaErrors(cudaMemcpy(d_rho, rho, size, cudaMemcpyHostToDevice));

 	spae_sparse_delta_device(d_delta, d_rho, beta, sp, n);

	checkCudaErrors(cudaMemcpy(delta, d_delta, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_delta));
	checkCudaErrors(cudaFree(d_rho));
}

template <typename T>
void _spae_sparse_delta_cpu(T* delta, T* rho, T beta, T sp, unsigned int n)
{
	T r;
	for(unsigned int i=0;i<n;++i) {
		r = rho[i];
		delta[i] = beta*( -sp/r + (1.0-sp)/(1.0-r));
	}
}

extern "C" {

void spae_sparse_delta(double* delta, double* rho, double beta, double sp, unsigned int n)
{
	_spae_sparse_delta(delta,rho,beta,sp,n);
}

void spae_sparse_delta_f(float* delta, float* rho, float beta, float sp, unsigned int n)
{
	_spae_sparse_delta(delta,rho,beta,sp,n);
}

void spae_sparse_delta_cpu(double* delta, double* rho, double beta, double sp, unsigned int n)
{
	_spae_sparse_delta_cpu(delta,rho,beta,sp,n);
}

}
