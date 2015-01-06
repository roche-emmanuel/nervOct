#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, bool withSparsity, unsigned int blockSize>
__global__ void ComputeDelta(BPComputeTraits<T> traits)
// unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int next_delta_offset,
// unsigned int nrows, unsigned int ncols, unsigned int niter, T* nn_params, T* inputs, T* deltas)
{
  // This operation is basically a matrix multiplication with transposition on A:
  T dval = 0.0;
  T val;

  unsigned int nrows = traits.nrows;
  unsigned int ncols = traits.ncols;
  unsigned int niter = traits.niter;

  __shared__ T As[blockSize][blockSize + 1];
  __shared__ T Bs[blockSize][blockSize + 1];

  // So we want to compute the value d(row,col);
  // note that since we transpose A, the A number of cols is nrows and its number of row is ncols.
  int xx, yy;
  for (int k = 0; k < (blockSize + ncols - 1) / blockSize; k++)
  {

    xx = k * blockSize + threadIdx.x;
    yy = blockIdx.y * blockSize + threadIdx.y;

    val = 0.0;
    if (yy < nrows && xx < niter)
    {
      // We add 1 below because we do not want to use the first row of theta_T, so that's
      // actually the first col of theta.
      val = traits.params[traits.theta_offset + (yy + 1) * niter + xx];
    }

    As[threadIdx.y][threadIdx.x] = val;

    xx = blockIdx.x * blockSize + threadIdx.y;
    yy = k * blockSize + threadIdx.x;

    val = 0.0;
    if (yy < niter && xx < ncols)
    {
      val = traits.deltas[traits.delta_offset + xx * niter + yy];
    }

    Bs[threadIdx.x][threadIdx.y] = val;

    __syncthreads();

    for (int n = 0; n < blockSize; ++n)
    {
      dval += As[threadIdx.x][n] * Bs[n][threadIdx.y];
    }

    __syncthreads();
  }

  int row = blockIdx.y * blockSize + threadIdx.x;
  int col = blockIdx.x * blockSize + threadIdx.y;

  if (row < nrows && col < ncols)
  {
    // we have to multiply that value by the corresponding sigmoid gradient value from the input matrix at the same location.
    int index = nrows * col + row;
    T sig = traits.inputs[traits.input_offset + index];
    if(withSparsity) {
      dval += traits.spae_delta[row];
    }

    traits.deltas[traits.next_delta_offset + index] = dval * sig * (1.0 - sig);
  }
}

// Explicit instanciation:
template __global__ void ComputeDelta<double>(BPComputeTraits<double> traits);
template __global__ void ComputeDelta<float>(BPComputeTraits<float> traits);
template __global__ void ComputeDelta<double,true>(BPComputeTraits<double> traits);
template __global__ void ComputeDelta<float,true>(BPComputeTraits<float> traits);
