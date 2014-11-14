#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void ComputeActivation(unsigned int theta_offset, unsigned int act_offset, unsigned int next_act_offset, unsigned int input_offset,
	unsigned int nrows, unsigned int ncols, unsigned int ncolT, double* nn_params, double* activation, double* inputs) 
{

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  double zval = 0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

  int xx, yy;

  // Here we compute the product theta_i * a_i^T
  for (int k = 0; k < (BLOCK_SIZE + ncolT - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.y;
  	yy = blockIdx.y*BLOCK_SIZE + threadIdx.x;

		if (xx < ncolT && yy < nrows) 
		 	As[threadIdx.x][threadIdx.y] = nn_params[theta_offset + xx*nrows + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;

		xx = col;
		yy = k*BLOCK_SIZE + threadIdx.y;

		if (xx < ncols && yy < ncolT)
			// In case we are trying to access the first row of the a_i matrix we just return 1.0
			// instead of reading the data (which is actually initialized to 0! and should not be read/written)
			Bs[threadIdx.y][threadIdx.x] = yy==0 ? 1.0 : activation[act_offset + yy*ncols + xx];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			zval += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrows && col < ncols) {
  	// compute the sigmoid of the value:
  	zval = 1.0 / (1.0 + exp(-zval));

  	// we just computed the value z_i(row,col), now we store it:
  	inputs[input_offset + nrows*col + row] = zval;

  	// we also store the value as activation (transposed)
  	activation[next_act_offset + (row+1)*ncols + col] = zval;
  }

}
