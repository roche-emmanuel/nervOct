#include <nervCUDA.h>
#include <nerv_kernels.h>

#include <curand_kernel.h>

__device__ float random_float(curandState *states, int rid)
{
  curandState rState = states[rid];
  float res = curand_uniform(&rState);
  states[rid] = rState;
  return res;
}

template <typename T, bool withDropout, bool debugMode, unsigned int blockSize>
__global__ void ComputeActivation(BPComputeTraits<T> traits)
{

  unsigned int nrows = traits.nrows;
  unsigned int ncols = traits.ncols;
  unsigned int ncolT = traits.niter;

  // Note that we assume here that the matrix coefficient are stored in row major order:
  // eg Aelem(i,jl) = A[j*nrowA+i]
  int row = blockIdx.y * blockSize + threadIdx.x;
  int col = blockIdx.x * blockSize + threadIdx.y;

  __shared__ T As[blockSize][blockSize + 1];
  __shared__ T Bs[blockSize][blockSize + 1];

  int xx, yy;

  // we can already add the element on the first row of theta_i to this element value:
  // but note that this element should be multiplied with the desired bias:
  T zval = traits.params[traits.theta_offset + row];

  if (withDropout)
  {
    // when dropout is enabled we use the weighted bias array:
    zval *= traits.wbias[traits.wbias_offset + col];
  }
  else
  {
    // Otherwise we use the fixed bias value:
    zval *= traits.bias;
  }

  T val;

  // Here we compute the product theta_i * a_i^T
  for (int k = 0; k < (blockSize + ncolT - 1) / blockSize; k++)
  {

    xx = k * blockSize + threadIdx.y;
    yy = blockIdx.y * blockSize + threadIdx.x;

    val = 0.0;

    if (xx < ncolT && yy < nrows)
    {
      // Note here that we should NOT use the first row of theta_i in those computation:
      // That row elemtn is already added to the zval value (matching the "virtual" 1 row
      // on top of the z_i matrix when used as activation.)
      val = traits.params[traits.theta_offset + (xx + 1) * nrows + yy];
    }

    As[threadIdx.x][threadIdx.y] = val;

    val = 0.0;

    xx = blockIdx.x * blockSize + threadIdx.y;
    yy = k * blockSize + threadIdx.x;

    if (yy < ncolT && xx < ncols)
    {
      val = traits.next_input_offset == 0 ? traits.wX[xx * ncolT + yy] : traits.inputs[traits.input_offset + xx * ncolT + yy];
    }

    Bs[threadIdx.x][threadIdx.y] = val;

    __syncthreads();

    for (int n = 0; n < blockSize; ++n)
    {
      zval += As[threadIdx.x][n] * Bs[n][threadIdx.y];
    }

    __syncthreads();
  }

  if (row < nrows && col < ncols)
  {
    // compute the sigmoid of the value:
    zval = 1.0 / (1.0 + exp(-zval * traits.wmult));

    if (withDropout)
    {
      // We might want to drop this unit completely.
      if (debugMode)
      {
        // Compute a fake random value:
        if(abs(sin((T)(nrows*col+row))) > traits.layer_dropout) {
          zval = 0.0;
        }
      }
      else
      {
        // Compute a real random value:
        // Compute the index to retrieve the rand state:
        int rid = blockSize * threadIdx.y + threadIdx.x;

        if (random_float(traits.randStates, rid) > traits.layer_dropout)
        {
          zval = 0.0;
        }
      }
    }

    // we just computed the value z_i(row,col), now we store it:
    traits.inputs[traits.next_input_offset + nrows * col + row] = zval;
  }

}

template __global__ void ComputeActivation<double>(BPComputeTraits<double> traits);
template __global__ void ComputeActivation<float>(BPComputeTraits<float> traits);
template __global__ void ComputeActivation<double, true>(BPComputeTraits<double> traits);
template __global__ void ComputeActivation<float, true>(BPComputeTraits<float> traits);
template __global__ void ComputeActivation<double, true, true>(BPComputeTraits<double> traits);
template __global__ void ComputeActivation<float, true, true>(BPComputeTraits<float> traits);
