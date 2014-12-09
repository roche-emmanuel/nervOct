#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

__global__ void InitRandStates( curandState * d_state, unsigned int n, unsigned long seed )
{
  unsigned int id = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if(id<n) {
    curand_init ( seed, id, 0, &d_state[id] );
  }
} 

void init_rand_state_device(curandState* d_state, unsigned int size, unsigned long seed)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + size-1)/BLOCK_SIZE, 1, 1);	

	InitRandStates<<<dimGrid, dimBlock>>>(d_state, size, seed);		
}
