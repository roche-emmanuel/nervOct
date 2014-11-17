#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void ComputeGradient(unsigned int theta_offset, unsigned int input_offset,  unsigned int delta_offset, unsigned int grad_offset,
	unsigned int nrows, unsigned int ncols, unsigned int niter, double* nn_params, double* inputs, double* deltas, double* grads, double lambda) 
{
  double CValue = 0;

  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1]; // Adding +1 to avoid shared memory bank conflict
  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

  int xx, yy;
  for (int k = 0; k < (BLOCK_SIZE + niter - 1)/BLOCK_SIZE; k++) {

  	// Here we try to access the A matrix data in a coaleased way:
  	// keeping in mind that A is row major. So we need to read A per column
  	// while the threads in the wrap are (probably) organized by row.
  	// So we invert the roles palyed by threadIdx.x and threadIdx.y.
  	xx = k*BLOCK_SIZE + threadIdx.y;
  	yy = blockIdx.y*BLOCK_SIZE + threadIdx.x;
		if (xx < niter && yy < nrows) 
		 	As[threadIdx.x][threadIdx.y] = deltas[delta_offset + xx*nrows + yy];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;


		// Same for the B matrix, we need to invert the x and y coords:
		xx = blockIdx.x*BLOCK_SIZE + threadIdx.y;
		yy = k*BLOCK_SIZE + threadIdx.x;

		if (yy < niter && xx < ncols) {
			// B(r,c)==0 if c==0 or B(r,c)=z_T(r,c-1)= z(c-1,r)
			Bs[threadIdx.x][threadIdx.y] = (xx==0 ? 1.0 : inputs[input_offset + (ncols-1)*yy + xx-1]); //inputs[input_offset + (ncols-1)*yy + xx-1 ]; // memory access is coalesced, nothing to change.
		}
		else
			Bs[threadIdx.x][threadIdx.y] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) 
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
  }

  if (row < nrows && col < ncols) {
  	int index = nrows*col+row;
    double reg = (col==0 ? 0.0 : nn_params[theta_offset + index]);
    CValue += lambda*reg;

  	grads[grad_offset + index] = CValue/niter;
  }

	// // This operation is basically a matrix multiplication with transposition on A:
 //  double gval = 0.0;

 //  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
 //  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

 //  __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
 //  __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

 //  // So we want to compute the value d(row,col);
 //  int xx, yy;
 //  for (int k = 0; k < (BLOCK_SIZE + niter - 1)/BLOCK_SIZE; k++) {

 //  	// THe first matrix is delta[i] and it is not transposed:
 //  	xx = k*BLOCK_SIZE + threadIdx.x;
 //  	yy = blockIdx.y*BLOCK_SIZE + threadIdx.y;

 //  	// Need to access A(yy,xx)
	// 	if (xx < niter && yy < nrows) 
	// 	 	As[threadIdx.x][threadIdx.y] = 1.0; //deltas[delta_offset + xx*nrows + yy];
	// 	else
	// 		As[threadIdx.x][threadIdx.y] = 0.0;

	// 	// The second matrix contains a transposition:
	// 	xx = blockIdx.x*BLOCK_SIZE + threadIdx.x; // = col
	// 	yy = k*BLOCK_SIZE + threadIdx.y;

	// 	// We want to retrieve the value of B(yy,xx)
	// 	if (yy < niter && xx < ncols)
	// 		// B(r,c)==0 if c==0 or B(r,c)=z_T(r,c-1)= z(c-1,r)
	// 		Bs[threadIdx.x][threadIdx.y] = 1.0; //xx==0 ? 1.0 : inputs[input_offset + (ncols-1)*yy + xx-1 ]; // memory access is coalesced, nothing to change.
	// 	else
	// 		Bs[threadIdx.x][threadIdx.y] = 0.0;

	// 	__syncthreads();

	// 	for (int n = 0; n < BLOCK_SIZE; ++n) 
	// 		gval += As[threadIdx.y][n] * Bs[n][threadIdx.x];

	// 	__syncthreads();
 //  }

 //  if (row < nrows && col < ncols) {
 //  	// We should also compute the regularization term:
 //  	int index = nrows*col+row;
 //    double reg = (col==0 ? 0.0 : nn_params[theta_offset + index]);
 //    // gval += lambda*reg;

 //  	grads[grad_offset + index] = niter; //gval; //gval; //gval/(double)niter; //(niter==nsamples)
 // }
}
