#include <nervCUDA.h>
#include <nerv_kernels.h>

template <typename T, unsigned int blockSize>
__global__ void ComputeActivation(unsigned int theta_offset, unsigned int input_offset, unsigned int next_input_offset,
	unsigned int nrows, unsigned int ncols, unsigned int ncolT, T* nn_params, T* inputs, T* X, T bias, T wmult) 
{

	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  int row = blockIdx.y*blockSize + threadIdx.x;
  int col = blockIdx.x*blockSize + threadIdx.y;

  __shared__ T As[blockSize][blockSize+1];
  __shared__ T Bs[blockSize][blockSize+1];

  int xx, yy;

  // we can already add the element on the first row of theta_i to this element value:
  // but note that this element should be multiplied with the desired bias:
  T zval = nn_params[theta_offset + row]*bias;

  // Here we compute the product theta_i * a_i^T
  for (int k = 0; k < (blockSize + ncolT - 1)/blockSize; k++) {

  	xx = k*blockSize + threadIdx.y;
  	yy = blockIdx.y*blockSize + threadIdx.x;

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
			xx = blockIdx.x*blockSize + threadIdx.x;
			yy = k*blockSize + threadIdx.y;

			if (xx < ncols && yy < ncolT)
				Bs[threadIdx.y][threadIdx.x] = X[yy*ncols + xx];
			else
				Bs[threadIdx.y][threadIdx.x] = 0.0;
		}
		else {
			xx = blockIdx.x*blockSize + threadIdx.y;
			yy = k*blockSize + threadIdx.x;

			if (yy < ncolT && xx < ncols)
				Bs[threadIdx.x][threadIdx.y] = inputs[input_offset + xx*ncolT + yy];
			else
				Bs[threadIdx.x][threadIdx.y] = 0.0;
		}

		__syncthreads();

		for (int n = 0; n < blockSize; ++n) 
			zval += As[threadIdx.x][n] * Bs[n][threadIdx.y];

		__syncthreads();
  }

  if (row < nrows && col < ncols) {
  	// compute the sigmoid of the value:
  	zval = 1.0 / (1.0 + exp(-zval*wmult));

  	// we just computed the value z_i(row,col), now we store it:
  	inputs[next_input_offset + nrows*col + row] = zval;
  }

}

// Explicit specialization:
template __global__ void ComputeActivation<double>(unsigned int theta_offset, unsigned int input_offset, unsigned int next_input_offset,
	unsigned int nrows, unsigned int ncols, unsigned int ncolT, double* nn_params, double* inputs, double* X, double bias, double wmult);

template __global__ void ComputeActivation<float>(unsigned int theta_offset, unsigned int input_offset, unsigned int next_input_offset,
	unsigned int nrows, unsigned int ncols, unsigned int ncolT, float* nn_params, float* inputs, float* X, float bias, float wmult);
