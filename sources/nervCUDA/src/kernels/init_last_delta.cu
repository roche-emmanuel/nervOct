#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
__global__ void InitLastDelta(BPComputeTraits<T> traits)
// unsigned int input_offset, unsigned int nrows, unsigned int ncols, T* deltas, T* inputs, T* yy) 
{
  int row = blockIdx.y*blockSize + threadIdx.x; // we inverse x and y for coalesced global memory access
  int col = blockIdx.x*blockSize + threadIdx.y;

  unsigned int nrows = traits.nrows;
  unsigned int ncols = traits.ncols;

  if (row < nrows && col < ncols) {
	  int index = nrows*col+row;
   	traits.deltas[index] = traits.inputs[traits.input_offset+index] - traits.yy[index];
  }
}

// Explicit instanciation:
template __global__ void InitLastDelta(BPComputeTraits<double> traits);

template __global__ void InitLastDelta(BPComputeTraits<float> traits);
