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


template <typename T, unsigned int blockSize>
__global__ void ComputeActivationWithDropout
(BPComputeTraits<T> traits)
{
  unsigned int nrows = traits.nrows;
  unsigned int ncols = traits.ncols;
  unsigned int ncolT = traits.niter;
  curandState *states = traits.randStates;

  // Compute the index to retrieve the rand state:
  int rid = blockSize * threadIdx.y + threadIdx.x;

  // Retrieve the dropout threshold:
  T drop = traits.layer_dropout;

  // Note that we assume here that the matrix coefficient are stored in row major order:
  // eg Aelem(i,jl) = A[j*nrowA+i]
  int row = blockIdx.y * blockSize + threadIdx.x;
  int col = blockIdx.x * blockSize + threadIdx.y;

  __shared__ T As[blockSize][blockSize + 1];
  __shared__ T Bs[blockSize][blockSize + 1];

  int xx, yy;

  // we can already add the element on the first row of theta_i to this element value:
  // but note that this element should be multiplied with the desired bias:
  // Here we can use the wbias array, to decide if the bias unit is activated
  // for that sample or not:
  T zval = traits.params[traits.theta_offset + row] * traits.wbias[traits.wbias_offset + col];

  // Here we compute the product theta_i * a_i^T
  for (int k = 0; k < (blockSize + ncolT - 1) / blockSize; k++)
  {

    xx = k * blockSize + threadIdx.y;
    yy = blockIdx.y * blockSize + threadIdx.x;

    if (xx < ncolT && yy < nrows)
      // Note here that we should NOT use the first row of theta_i in those computation:
      // That row elemtn is already added to the zval value (matching the "virtual" 1 row
      // on top of the z_i matrix when used as activation.)
      As[threadIdx.x][threadIdx.y] = traits.params[traits.theta_offset + (xx + 1) * nrows + yy];
    else
      As[threadIdx.x][threadIdx.y] = 0.0;


    if (traits.next_input_offset == 0)
    {
      // In that case we need to retrieve the data from the X matrix.
      // actually we need the data from X^T.
      xx = blockIdx.x * blockSize + threadIdx.x;
      yy = k * blockSize + threadIdx.y;

      if (xx < ncols && yy < ncolT)
        Bs[threadIdx.y][threadIdx.x] = traits.X[yy * ncols + xx];
      else
        Bs[threadIdx.y][threadIdx.x] = 0.0;
    }
    else
    {
      xx = blockIdx.x * blockSize + threadIdx.y;
      yy = k * blockSize + threadIdx.x;

      if (yy < ncolT && xx < ncols)
        Bs[threadIdx.x][threadIdx.y] = traits.inputs[traits.input_offset + xx * ncolT + yy];
      else
        Bs[threadIdx.x][threadIdx.y] = 0.0;
    }

    __syncthreads();

    for (int n = 0; n < blockSize; ++n)
      zval += As[threadIdx.x][n] * Bs[n][threadIdx.y];

    __syncthreads();
  }

  if (row < nrows && col < ncols)
  {
    // check if we need to keep this unit activated or not
    // depending on the current dropout threshold:
    if (random_float(states, rid) <= drop)
    {
      // compute the sigmoid of the value:
      zval = 1.0 / (1.0 + exp(-zval * traits.wmult));
    }
    else
    {
      zval = 0.0; // desactivate this unit.
    }

    // we just computed the value z_i(row,col), now we store it:
    traits.inputs[traits.next_input_offset + nrows * col + row] = zval;
  }

}

template __global__ void ComputeActivationWithDropout<double>(BPComputeTraits<double> traits);
template __global__ void ComputeActivationWithDropout<float>(BPComputeTraits<float> traits);
