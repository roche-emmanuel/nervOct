#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

template<typename T>
__global__ void RandWeights( curandState *d_state, T* weights, T threshold, unsigned int n )
{
  unsigned int id = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if(id<n) {
  	curandState rState = d_state[id];
  	float val = curand_uniform(&rState);
  	d_state[id] = rState;
  	weights[id] = val<=threshold ? 1.0 : 0.0;
  }
}

template<typename T>
void rand_weights_device(curandState *d_state, T* weights, T threshold, unsigned int size)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

	RandWeights<<<dimGrid, dimBlock>>>(d_state, weights, threshold, size);		

}

// explicit instanciation:
template void rand_weights_device(curandState *d_state, double* weights, double threshold, unsigned int size);
template void rand_weights_device(curandState *d_state, float* weights, float threshold, unsigned int size);
