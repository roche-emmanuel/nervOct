#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
__global__ void InitLastDelta(unsigned int nrows, unsigned int ncols, T* deltas, T* hx, T* yy) 
{
  int row = blockIdx.y*blockSize + threadIdx.x; // we inverse x and y for coalesced global memory access
  int col = blockIdx.x*blockSize + threadIdx.y;

  if (row < nrows && col < ncols) {
	  int index = nrows*col+row;
   	deltas[index] = hx[index] - yy[index];
  }
}

// Explicit instanciation:
template __global__ void InitLastDelta(unsigned int nrows, unsigned int ncols, double* deltas, double* hx, double* yy);

template __global__ void InitLastDelta(unsigned int nrows, unsigned int ncols, float* deltas, float* hx, float* yy);
