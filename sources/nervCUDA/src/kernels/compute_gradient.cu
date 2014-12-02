#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
__global__ void ComputeGradient(unsigned int theta_offset, int input_offset,  unsigned int delta_offset, unsigned int grad_offset,
	unsigned int nrows, unsigned int ncols, unsigned int niter, T* X, T* nn_params, T* inputs, T* deltas, T* grads, T lambda, T bias) 
{
  T CValue = 0;

  __shared__ T As[blockSize][blockSize+1]; // Adding +1 to avoid shared memory bank conflict
  __shared__ T Bs[blockSize][blockSize+1];

  int xx, yy;
  for (int k = 0; k < (blockSize + niter - 1)/blockSize; k++) {

  	// Here we try to access the A matrix data in a coaleased way:
  	// keeping in mind that A is row major. So we need to read A per column
  	// while the threads in the wrap are (probably) organized by row.
  	// So we invert the roles palyed by threadIdx.x and threadIdx.y.
  	xx = k*blockSize + threadIdx.y;
  	yy = blockIdx.y*blockSize + threadIdx.x;

		if (xx < niter && yy < nrows) 
		 	As[threadIdx.x][threadIdx.y] = deltas[delta_offset +nrows*xx + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;


		if(input_offset<0) {
			xx = blockIdx.x*blockSize + threadIdx.y;
			yy = k*blockSize + threadIdx.x;

			if (yy < niter && xx < ncols) {
				// Here we use the matrix X instead of z_T:
				// B(r,c)= X(r,c-1) if c>0;
				Bs[threadIdx.x][threadIdx.y] = (xx==0 ? bias : X[niter*(xx-1) + yy]); //inputs[input_offset + (ncols-1)*yy + xx-1 ]; // memory access is coalesced, nothing to change.				
			}
			else
				Bs[threadIdx.x][threadIdx.y] = 0.0;
		}
		else {
			// Same for the B matrix, we need to invert the x and y coords:
			xx = blockIdx.x*blockSize + threadIdx.x;
			yy = k*blockSize + threadIdx.y;

			if (yy < niter && xx < ncols) {
					// B(r,c)==1 if c==0 or B(r,c)=z_T(r,c-1)= z(c-1,r)
					Bs[threadIdx.y][threadIdx.x] = (xx==0 ? bias : inputs[input_offset + (ncols-1)*yy + xx-1]); //inputs[input_offset + (ncols-1)*yy + xx-1 ]; // memory access is coalesced, nothing to change.				
			}
			else
				Bs[threadIdx.y][threadIdx.x] = 0.0;
		}

		__syncthreads();

		for (int n = 0; n < blockSize; ++n) 
			CValue += As[threadIdx.x][n] * Bs[n][threadIdx.y];

		__syncthreads();
  }

  int row = blockIdx.y*blockSize + threadIdx.x;
  int col = blockIdx.x*blockSize + threadIdx.y;

  if (row < nrows && col < ncols) {
  	int index = nrows*col+row;
    T reg = (col==0 ? 0.0 : nn_params[theta_offset + index]);
    CValue += lambda*reg;

  	grads[grad_offset + index] = CValue/niter;
  }
}


// Explicit instanciation:
template __global__ void ComputeGradient(unsigned int theta_offset, int input_offset,  unsigned int delta_offset, unsigned int grad_offset,
	unsigned int nrows, unsigned int ncols, unsigned int niter, double* X, double* nn_params, double* inputs, double* deltas, double* grads, double lambda, double bias);

template __global__ void ComputeGradient(unsigned int theta_offset, int input_offset,  unsigned int delta_offset, unsigned int grad_offset,
	unsigned int nrows, unsigned int ncols, unsigned int niter, float* X, float* nn_params, float* inputs, float* deltas, float* grads, float lambda, float bias);
