#include <nervCUDA.h>
#include <nerv_kernels.h>

/*
	Method used to evaluate the cost function when starting from the hx and yy matrices.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void ComputeLength2(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
    		T val = g_idata[i];
        mySum += val*val;

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
        	val = g_idata[i+blockSize];
        	mySum += val*val;
        }
         
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template<typename T>
void compute_length2(int size, int threads, int blocks,
       int whichKernel, T *d_idata, T *d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size))
  {
      switch (threads)
      {
          case 512:
              ComputeLength2<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 256:
              ComputeLength2<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 128:
              ComputeLength2<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 64:
              ComputeLength2<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 32:
              ComputeLength2<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 16:
              ComputeLength2<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  8:
              ComputeLength2<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  4:
              ComputeLength2<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  2:
              ComputeLength2<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  1:
              ComputeLength2<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;
      }
  }
  else
  {
      switch (threads)
      {
          case 512:
              ComputeLength2<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 256:
              ComputeLength2<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 128:
              ComputeLength2<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 64:
              ComputeLength2<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 32:
              ComputeLength2<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case 16:
              ComputeLength2<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  8:
              ComputeLength2<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  4:
              ComputeLength2<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  2:
              ComputeLength2<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;

          case  1:
              ComputeLength2<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
              break;
      }
  }
}

template<typename T>
T compute_length2_device(T* d_vec, T* d_odata, unsigned int n)
{
  int maxThreads = 256;
  int maxBlocks = 64;
  int whichKernel = 6;
  // bool cpuFinalReduction = false;
  int cpuFinalThreshold = 1;

  int numBlocks = 0;
  int numThreads = 0;
  getNumBlocksAndThreads(whichKernel, n, maxBlocks, maxThreads, numBlocks, numThreads);

  // Allocate output array:
  // size_t size = numBlocks*sizeof(T);
  // T* d_odata = NULL;
  // checkCudaErrors(cudaMalloc(&d_odata, size));

  // Allocate mem for the result on host side
  T *h_odata = (T *) malloc(numBlocks*sizeof(T));

  T gpu_result = 0.0;
  bool needReadBack = true;

  // execute the kernel
  compute_length2(n, numThreads, numBlocks, whichKernel, d_vec, d_odata);

  // sum partial block sums on GPU
  int s=numBlocks;
  int kernel = whichKernel;

  while (s > cpuFinalThreshold)
  {
      int threads = 0, blocks = 0;
      getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

      reduce_sum_launcher(s, threads, blocks, kernel, d_odata, d_odata);

      if (kernel < 3)
      {
          s = (s + threads - 1) / threads;
      }
      else
      {
          s = (s + (threads*2-1)) / (threads*2);
      }
  }

  if (s > 1)
  {
      // copy result from device to host
      checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));

      for (int i=0; i < s; i++)
      {
          gpu_result += h_odata[i];
      }

      needReadBack = false;
  }

  if (needReadBack)
  {
      // copy final sum from device to host
      checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
  }

  // Free host memory:
  free(h_odata);

  // Free device memory
  // checkCudaErrors(cudaFree(d_odata));
  return gpu_result;
}

template <typename T>
T _compute_length2(T* vec, unsigned int n)
{
  size_t size;

  size = n * sizeof(T);
  T* d_tmp = NULL;
  checkCudaErrors(cudaMalloc(&d_tmp, size));

  T* d_vec = NULL;
  checkCudaErrors(cudaMalloc(&d_vec, size));
  checkCudaErrors(cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice));

  T res = compute_length2_device(d_vec, d_tmp, n);

  checkCudaErrors(cudaFree(d_tmp));
  checkCudaErrors(cudaFree(d_vec));
  return res;
}

extern "C" {

double compute_length2(double* vec, unsigned int n)
{
  return _compute_length2(vec,n);
}

float compute_length2_f(float* vec, unsigned int n)
{
  return _compute_length2(vec,n);
}

}

