#include <nervCUDA.h>

#include <cuda_runtime.h>
#include <nerv_kernels.h>

__global__ void InitLastDelta(unsigned int nrows, unsigned int ncols, double* deltas, double* hx, double* yy) 
{
	// Note that we assume here that the matrix coefficient are stored in row major order:
	// eg Aelem(i,jl) = A[j*nrowA+i]
  int row = blockIdx.y*BLOCK_SIZE + threadIdx.x; // we inverse x and y for coalesced global memory access
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.y;
  int index = nrows*col+row;

  if (row < nrows && col < ncols) {
  	deltas[index] = hx[index] - yy[index];
  }

}
