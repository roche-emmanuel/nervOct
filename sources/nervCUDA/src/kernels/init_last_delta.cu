#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
__global__ void InitLastDelta(unsigned int input_offset, unsigned int nrows, unsigned int ncols, T* deltas, T* inputs, T* yy) 
{
  int row = blockIdx.y*blockSize + threadIdx.x; // we inverse x and y for coalesced global memory access
  int col = blockIdx.x*blockSize + threadIdx.y;

  if (row < nrows && col < ncols) {
	  int index = nrows*col+row;
   	deltas[index] = inputs[input_offset+index] - yy[index];
  }
}

// Explicit instanciation:
template __global__ void InitLastDelta(unsigned int input_offset,unsigned int nrows, unsigned int ncols, double* deltas, double* inputs, double* yy);

template __global__ void InitLastDelta(unsigned int input_offset,unsigned int nrows, unsigned int ncols, float* deltas, float* inputs, float* yy);
