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

#if 0
template<typename T, unsigned int blockSize = 32>
__global__ void RandWeightsDebug( curandState *d_states, T *weights, T threshold, unsigned int n, T value )
{
  unsigned int id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (id < n)
  {
    float val = (float)abs(sin((T)id));
    weights[id] = val <= threshold ? value : 0.0;
  }
}
#endif

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

#if 0
template<typename T>
void rand_weights_device_debug(RandDeviceTraits<T> &traits) //curandState *d_state, T *weights, T threshold, unsigned int size, T value)
{
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((BLOCK_SIZE + traits.size - 1) / BLOCK_SIZE, 1, 1);

  RandWeightsDebug <<< dimGrid, dimBlock>>>(traits.randStates, traits.target, traits.threshold, traits.size, traits.value);
  // CHECK_KERNEL();
}
#endif

template <typename T>
void _rand_weights(RandTraits<T> &traits) //T *weights, T threshold, unsigned int n, T value)
{
  T *weights = traits.target;
  T threshold = traits.threshold;
  unsigned int n = traits.size;
  T value = traits.value;

  unsigned int size = n * sizeof(T);
  T *d_weights = NULL;
  checkCudaErrors(cudaMalloc(&d_weights, size));
  checkCudaErrors(cudaMemcpy(d_weights, weights, size, cudaMemcpyHostToDevice));

  // We also need to prepare the curandState buffer:
  size = 1024;
  curandState *d_states = NULL;
  checkCudaErrors(cudaMalloc(&d_states, size * sizeof(curandState)));

  // Here we call the method to initialize the random states:
  init_rand_state_device(d_states, size, (unsigned long)time(NULL));

  RandDeviceTraits<T> d_traits;
  d_traits.randStates = d_states;
  d_traits.target = d_weights;
  d_traits.threshold = threshold;
  d_traits.size = n;
  d_traits.value = value;
  d_traits.debug = traits.debug;

  // Now call the device kernel:
  rand_weights_device(d_traits); //d_states, d_weights, threshold, n, value);

  // copy the results back:
  copyFromDevice(weights, d_weights, n);

  // Release the GPU buffers:
  checkCudaErrors(cudaFree(d_weights));
  checkCudaErrors(cudaFree(d_states));
}

#if 0
template <typename T>
void _rand_weights_debug(RandTraits<T> &traits) //T *weights, T threshold, unsigned int n, T value)
{
  T *weights = traits.target;
  T threshold = traits.threshold;
  unsigned int n = traits.size;
  T value = traits.value;

  unsigned int size = n * sizeof(T);
  T *d_weights = NULL;
  checkCudaErrors(cudaMalloc(&d_weights, size));
  checkCudaErrors(cudaMemcpy(d_weights, weights, size, cudaMemcpyHostToDevice));

  // We also need to prepare the curandState buffer:
  size = 1024;
  curandState *d_states = NULL;
  checkCudaErrors(cudaMalloc(&d_states, size * sizeof(curandState)));

  // Here we call the method to initialize the random states:
  init_rand_state_device(d_states, size, (unsigned long)time(NULL));

  // Now call the device kernel:
  rand_weights_device_debug(d_states, d_weights, threshold, n, value);

  // copy the results back:
  copyFromDevice(weights, d_weights, n);

  // Release the GPU buffers:
  checkCudaErrors(cudaFree(d_weights));
  checkCudaErrors(cudaFree(d_states));
}
#endif

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
