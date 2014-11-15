#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void ComputeActivation(unsigned int theta_offset, unsigned int input_offset, unsigned int next_input_offset,
	unsigned int nrows, unsigned int ncols, unsigned int ncolT, double* nn_params, double* inputs, double* X) 
{

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

  int xx, yy;

  // we can already add the element on the first row of theta_i to this element value:
  double zval = nn_params[theta_offset + row];

  // Here we compute the product theta_i * a_i^T
  for (int k = 0; k < (BLOCK_SIZE + ncolT - 1)/BLOCK_SIZE; k++) {

  	xx = k*BLOCK_SIZE + threadIdx.y;
  	yy = blockIdx.y*BLOCK_SIZE + threadIdx.x;

		if (xx < ncolT && yy < nrows) 
			// Note here that we should NOT use the first row of theta_i in those computation:
			// That row elemtn is already added to the zval value (matching the "virtual" 1 row
			// on top of the z_i matrix when used as activation.)
		 	As[threadIdx.x][threadIdx.y] = nn_params[theta_offset + (xx+1)*nrows + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;


		if(next_input_offset==0) {
			// In that case we need to retrieve the data from the X matrix.
			// actually we need the data from X^T.
			xx = col;
			yy = k*BLOCK_SIZE + threadIdx.y;

			if (xx < ncols && yy < ncolT)
				Bs[threadIdx.y][threadIdx.x] = X[yy*ncols + xx];
			else
				Bs[threadIdx.y][threadIdx.x] = 0.0;
		}
		else {
			xx = blockIdx.x*BLOCK_SIZE + threadIdx.y;
			yy = k*BLOCK_SIZE + threadIdx.x;

			if (yy < ncolT && xx < ncols)
				Bs[threadIdx.x][threadIdx.y] = inputs[input_offset + xx*ncolT + yy];
			else
				Bs[threadIdx.x][threadIdx.y] = 0.0;
		}


		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			zval += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrows && col < ncols) {
  	// compute the sigmoid of the value:
  	zval = 1.0 / (1.0 + exp(-zval));

  	// we just computed the value z_i(row,col), now we store it:
  	inputs[next_input_offset + nrows*col + row] = zval;

  	// we also store the value as activation (transposed)
  	// activation[next_act_offset + (row+1)*ncols + col] = zval;
  }

}
