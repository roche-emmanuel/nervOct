#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
__global__ void ComputeDelta(BPComputeTraits<T> traits)
	// unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int next_delta_offset,
	// unsigned int nrows, unsigned int ncols, unsigned int niter, T* nn_params, T* inputs, T* deltas) 
{
	// This operation is basically a matrix multiplication with transposition on A:
  T dval = 0.0;

  unsigned int nrows = traits.nrows;
  unsigned int ncols = traits.ncols;
  unsigned int niter = traits.niter;

  __shared__ T As[blockSize][blockSize+1];
  __shared__ T Bs[blockSize][blockSize+1];

  // So we want to compute the value d(row,col);
  // note that since we transpose A, the A number of cols is nrows and its number of row is ncols.
  int xx, yy;
  for (int k = 0; k < (blockSize + ncols - 1)/blockSize; k++) {

  	xx = k*blockSize + threadIdx.x;
  	yy = blockIdx.y*blockSize + threadIdx.y;
  	
		if (yy < nrows && xx < niter) {
			// We add 1 below because we do not want to use the first row of theta_T, so that's
			// actually the first col of theta.
			As[threadIdx.y][threadIdx.x] = traits.params[traits.theta_offset + (yy+1)*niter + xx]; 
		}
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		xx = blockIdx.x*blockSize + threadIdx.y;
		yy = k*blockSize + threadIdx.x;

		if (yy < niter && xx < ncols)
			Bs[threadIdx.x][threadIdx.y] = traits.deltas[traits.delta_offset + xx*niter + yy];
		else
			Bs[threadIdx.x][threadIdx.y] = 0.0;

		__syncthreads();

		for (int n = 0; n < blockSize; ++n) 
			dval += As[threadIdx.x][n] * Bs[n][threadIdx.y];

		__syncthreads();
  }

  int row = blockIdx.y*blockSize + threadIdx.x;
  int col = blockIdx.x*blockSize + threadIdx.y;

  if (row < nrows && col < ncols) {
  	// we have to multiply that value by the corresponding sigmoid gradient value from the input matrix at the same location.
  	int index = nrows*col+row;
  	T sig = traits.inputs[traits.input_offset + index];
  	traits.deltas[traits.next_delta_offset + index] = dval *sig*(1.0 - sig);
  }
}

// Explicit instanciation:
template __global__ void ComputeDelta<double>(BPComputeTraits<double> traits);

template __global__ void ComputeDelta<float>(BPComputeTraits<float> traits);
