#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void ComputeDelta(unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int next_delta_offset,
	unsigned int nrows, unsigned int ncols, unsigned int niter, double* nn_params, double* inputs, double* deltas) 
{
	// This operation is basically a matrix multiplication with transposition on A:
  double dval = 0.0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

  // So we want to compute the value d(row,col);
  // note that since we transpose A, the A number of cols is nrows and its number of row is ncols.
  int xx, yy;
  for (int k = 0; k < (BLOCK_SIZE + ncols - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.x;
  	yy = row;
  	
		if (yy < nrows && xx < niter) {
			// We add 1 below because we do not want to use the first row of theta_T, so that's
			// actually the first col of theta.
			As[threadIdx.y][threadIdx.x] = nn_params[theta_offset + (yy+1)*niter + xx]; 
		}
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		xx = blockIdx.x*BLOCK_SIZE + threadIdx.y;
		yy = k*BLOCK_SIZE + threadIdx.x;

		if (yy < niter && xx < ncols)
			Bs[threadIdx.x][threadIdx.y] = deltas[delta_offset + xx*niter + yy];
		else
			Bs[threadIdx.x][threadIdx.y] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			dval += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrows && col < ncols) {
  	// we have to multiply that value by the corresponding sigmoid gradient value from the input matrix at the same location.
  	int index = nrows*col+row;
  	double sig = inputs[input_offset + index];
  	deltas[next_delta_offset + index] = dval *sig*(1.0 - sig);
  }
}
