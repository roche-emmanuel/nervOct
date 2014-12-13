#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

template<typename T, bool debugMode, unsigned int blockSize = 32>
__global__ void RandWeights(RandDeviceTraits<T> traits)
{
  unsigned int id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (id < traits.size)
  {
    float val;
    if (debugMode)
    {
      val = (float)abs(sin((T)id));
    }
    else
    {
      // Compute the index to retrieve the rand state:
      int rid = blockSize * threadIdx.y + threadIdx.x;

      curandState rState = traits.randStates[rid];
      val = curand_uniform(&rState);
      traits.randStates[rid] = rState;
    }

    traits.target[id] = val <= traits.threshold ? traits.value : 0.0;
  }
}

template<typename T>
void rand_weights_device(RandDeviceTraits<T> &traits)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + traits.size - 1) / BLOCK_SIZE, 1, 1);

  if(traits.debug) {
    RandWeights<T,true><<< dimGrid, dimBlock>>>(traits);
  }
  else {
    RandWeights<T,false><<< dimGrid, dimBlock>>>(traits);
  }
  // CHECK_KERNEL();
}

template <typename T>
void _rand_weights(RandTraits<T> &traits)
{
  RandDeviceTraits<T> d_traits;
  d_traits = traits;

  // Now call the device kernel:
  rand_weights_device(d_traits);

  // copy the results back:
  copyFromDevice(traits.target, d_traits.target, traits.size);
}

extern "C" {

  void rand_weights(RandTraits<double> &traits)
  {
    _rand_weights(traits);
  }

  void rand_weights_f(RandTraits<float> &traits)
  {
    _rand_weights(traits);
  }
}
