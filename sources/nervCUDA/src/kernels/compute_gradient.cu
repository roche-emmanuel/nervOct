#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

template<typename T, bool withDropout, unsigned int blockSize>
__global__ void ComputeGradient(BPComputeTraits<T> traits)
//unsigned int theta_offset, int input_offset,  unsigned int delta_offset, unsigned int grad_offset,
// unsigned int nrows, unsigned int ncols, unsigned int niter, T* X, T* nn_params, T* inputs, T* deltas, T* grads, T lambda, T bias)
{
  T CValue = 0;
  T val;

  unsigned int nrows = traits.nrows;
  unsigned int ncols = traits.ncols;
  unsigned int niter = traits.niter;

  int input_offset = traits.input_offset;

  __shared__ T As[blockSize][blockSize + 1]; // Adding +1 to avoid shared memory bank conflict
  __shared__ T Bs[blockSize][blockSize + 1];

  int xx, yy;
  for (int k = 0; k < (blockSize + niter - 1) / blockSize; k++)
  {

    // Here we try to access the A matrix data in a coaleased way:
    // keeping in mind that A is row major. So we need to read A per column
    // while the threads in the wrap are (probably) organized by row.
    // So we invert the roles palyed by threadIdx.x and threadIdx.y.
    xx = k * blockSize + threadIdx.y;
    yy = blockIdx.y * blockSize + threadIdx.x;

    val = 0.0;
    if (xx < niter && yy < nrows)
    {
      val = traits.deltas[traits.delta_offset + nrows * xx + yy];
    }

    As[threadIdx.x][threadIdx.y] = val;

    val = 0.0;

    // Same for the B matrix, we need to invert the x and y coords:
    xx = blockIdx.x * blockSize + threadIdx.x;
    yy = k * blockSize + threadIdx.y;

    if (yy < niter && xx < ncols)
    {
      // B(r,c)==1 if c==0 or B(r,c)=z_T(r,c-1)= z(c-1,r)
      if (xx == 0)
      {
        if (withDropout)
        {
          val = traits.wbias[traits.wbias_offset + yy];
        }
        else
        {
          val = traits.bias;
        }
      }
      else
      {
        val = input_offset < 0 ? traits.wX[(ncols - 1) * yy + xx - 1] : traits.inputs[input_offset + (ncols - 1) * yy + xx - 1];
      }
    }

    Bs[threadIdx.y][threadIdx.x] = val;

    __syncthreads();

    for (int n = 0; n < blockSize; ++n)
    {
      CValue += As[threadIdx.x][n] * Bs[n][threadIdx.y];
    }

    __syncthreads();
  }

  int row = blockIdx.y * blockSize + threadIdx.x;
  int col = blockIdx.x * blockSize + threadIdx.y;

  if (row < nrows && col < ncols)
  {
    int index = nrows * col + row;
    T reg = (col == 0 ? 0.0 : traits.params[traits.theta_offset + index]);
    CValue /= niter;
    CValue += traits.lambda * reg;

    traits.grads[traits.grad_offset + index] = CValue;
  }
}


// Explicit instanciation:
template __global__ void ComputeGradient(BPComputeTraits<double> traits);
template __global__ void ComputeGradient(BPComputeTraits<float> traits);
template __global__ void ComputeGradient<double, true>(BPComputeTraits<double> traits);
template __global__ void ComputeGradient<float, true>(BPComputeTraits<float> traits);
