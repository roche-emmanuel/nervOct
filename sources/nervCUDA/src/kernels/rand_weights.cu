#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#define BLOCK_SIZE 1024

template<typename T, bool debugMode, unsigned int blockSize = 32>
__global__ void RandWeights( curandState *d_states, T *weights, T threshold, unsigned int n, T value )
{
  unsigned int id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (id < n)
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

      curandState rState = d_states[rid];
      val = curand_uniform(&rState);
      d_states[rid] = rState;
    }

    weights[id] = val <= threshold ? value : 0.0;
  }
}

template<typename T>
void rand_weights_device(RandDeviceTraits<T> &traits) //curandState *d_state, T *weights, T threshold, unsigned int size, T value)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + traits.size - 1) / BLOCK_SIZE, 1, 1);

  if(traits.debug) {
    RandWeights<T,true><<< dimGrid, dimBlock>>>(traits.randStates, traits.target, traits.threshold, traits.size, traits.value);
  }
  else {
    RandWeights<T,false><<< dimGrid, dimBlock>>>(traits.randStates, traits.target, traits.threshold, traits.size, traits.value);
  }
  // CHECK_KERNEL();
}

template <typename T>
void _rand_weights(RandTraits<T> &traits) //T *weights, T threshold, unsigned int n, T value)
{
  RandDeviceTraits<T> d_traits;
  d_traits = traits;

  // Now call the device kernel:
  rand_weights_device(d_traits); //d_states, d_weights, threshold, n, value);

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
