#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
void gd_errfunc_device(unsigned int nl, unsigned int np, unsigned int* lsizes, unsigned int nsamples,
	T* d_params, T* d_X, T* d_yy, T lambda, T* J, T* d_grads, T* d_deltas, T* d_inputs, T* d_regw, T bias, cudaStream_t stream)
{
	unsigned int nt = nl-1; // number of matrices evolved.

	BPComputeTraits<T> traits;
	traits.params = d_params;
	traits.inputs = d_inputs;
	traits.deltas = d_deltas;
	traits.grads = d_grads;
	traits.yy = d_yy;
	traits.X = d_X;
	traits.bias = bias;
	traits.lambda = lambda;

	T* wmults = nullptr;

  traits.input_offset = nn_activation_device(nl, lsizes, nsamples, d_params, d_X, d_inputs, bias, wmults, stream);

  T* d_hx = d_inputs + traits.input_offset;

  // Here we can compute the cost now:
  // but only if requested:
  if(J) {
	  // The hx matrix is mapped to the last z matrix. eg at i=nt-1
	  // So its dimensions are lsizes[nt-1+1] * nsamples = lsizes[nl-1] * nsamples
	  // same dimensions for the yy matrix, and we want to perform reduction other those 2 matrices
		*J = 0.0;
		unsigned int count = nsamples*lsizes[nt];
		reduce_cost_device(d_hx, d_yy, count, *J, stream);
		// CHECK_KERNEL()

		*J /= (T)nsamples;

		T Jreg = 0.0;
		reduce_cost_reg_device(d_params, d_regw, np, Jreg, stream);
		// CHECK_KERNEL()

		*J += (T)((Jreg*lambda)/(2.0*nsamples));
  }

  if(!d_grads) {
  	// we don't need to compute the gradients.
  	return;
  }
  
	// Prepare the computation of the delta matrices in reverse order:

	// remove the last theta matrix size from the theta offset so that we can use
	// that offset to retrieve the proper theta matrix:
	// theta_offset -= lsizes[nt]*(lsizes[nt-1]+1);
	traits.theta_offset = np - lsizes[nt]*(lsizes[nt-1]+1);

	// initially the input_offset is pointing on the hx matrix which is z(nt-1) with our convention (eg. z(0) is not in the array.)
	// But the first one we will need is actually the one before that: z(nt-2)
	// So we need to update the offset, and remove the size of the matrix z(nt-2) ! (pointer is at the beginning of z(nt-1))
	// Note: This is now done inside the loop:
	// input_offset -= lsizes[nt-1]*nsamples;

	// Prepare the offset for the gradient array:
	// keep in mind we start with the latest theta matrix:
	traits.grad_offset = np - lsizes[nt]*(lsizes[nt-1]+1);

	for(unsigned int i=nt;i>0;--i) {
		traits.nrows = lsizes[i];
		traits.ncols = nsamples;
		traits.niter = lsizes[i+1];
		unsigned int count = traits.nrows*traits.ncols;

		dim3 dimBlock(blockSize, blockSize);
		dim3 dimGrid((blockSize + traits.ncols-1)/blockSize, (blockSize + traits.nrows-1)/blockSize);

		if(i==nt) {
			// we should just copy the difference of hx and yy into the z matrix.
			// CHECK_KERNEL()
			// InitLastDelta<<<dimGrid, dimBlock, 0, stream>>>(input_offset, nrows, ncols, d_deltas, d_inputs, d_yy);
			InitLastDelta<<<dimGrid, dimBlock, 0, stream>>>(traits);
			// CHECK_KERNEL()
		}
		else {
			// We compute the delta from the previous delta:
			// We start this computation for delta(nt-1). this matrix is build from theta(nt-1) and delta(nt).
			// also in the process we use the input matrix z(nt-2)
			// ComputeDelta<<<dimGrid, dimBlock, 0, stream>>>(theta_offset, input_offset, delta_offset, next_delta_offset, nrows, ncols, niter, d_params, d_inputs, d_deltas);
			ComputeDelta<<<dimGrid, dimBlock, 0, stream>>>(traits);
			// CHECK_KERNEL()

			// once the computation is done for that layer we move to the previous layer:
			traits.theta_offset -= lsizes[i]*(lsizes[i-1]+1);
		}

		traits.delta_offset = traits.next_delta_offset;
		traits.next_delta_offset += count;

		// At this point we have the previous theta matrix (eg. theta(i-1) pointed by theta_offset. (both when i=nt and i<nt).
		// and thats the matrix we need to compute the gradient values.
		// the gradient mat has the same size as the current theta matrix.
		// similarly, the input_offset is pointing on z(i-2) which is the one we need to perform the computation too.
		// and delta_offset points to the delta matrix we just wrote (eg. delta(i)).
		traits.nrows = lsizes[i];
		traits.ncols = lsizes[i-1]+1;
		traits.niter = nsamples;
		count = traits.nrows*traits.ncols;

		// Compute the gradient:
		dimBlock = dim3(blockSize, blockSize);
		dimGrid = dim3((blockSize + traits.ncols-1)/blockSize, (blockSize + traits.nrows-1)/blockSize);

    traits.input_offset -= lsizes[i-1]*nsamples; // we remove the size of the next delta matrix to be computed. which is also the size of the next z matrix we will use.
		// logDEBUG("GPU: Gradient at i="<<i<<" of size "<< nrows <<" x " << ncols<<", offset="<<grad_offset<<", input_offset="<<input_offset<<", nsamples="<<nsamples);

		// ComputeGradient<<<dimGrid, dimBlock, 0, stream>>>(theta_offset, input_offset, delta_offset, grad_offset, nrows, ncols, niter, d_X, d_params, d_inputs, d_deltas, d_grads, lambda, bias);
		ComputeGradient<<<dimGrid, dimBlock, 0, stream>>>(traits);
		// CHECK_KERNEL()

		// update the gradient offset by removing the size of the next gradient matrix to be computed:
		// except for the last iteration where the value is not available:
		if(i>1) {
			traits.grad_offset -= lsizes[i-1]*(lsizes[i-2]+1);
		}
	}
}

template <typename T>
void _gd_errfunc(BPTraits<T>& traits)
{
	// Compute the total number of parameters in this network:
	unsigned int np = traits.np();
	unsigned int nl = traits.nl;
	unsigned int* lsizes = traits.lsizes;
	unsigned int nsamples = traits.nsamples;

	// BPDeviceTraits<T> d_traits(traits);
	BPDeviceTraits<T> d_traits;
	d_traits = traits;

	// Compute the total number of delta coefficients:
	unsigned int nd = traits.nd();

	// Call the actual method to perform the computations:
	gd_errfunc_device<T>(nl, np, lsizes, nsamples, d_traits.params, d_traits.X, d_traits.yy, traits.lambda, &traits.cost, d_traits.grads, d_traits.deltas, 
		d_traits.inputs, d_traits.regw);

	// Here we should also read back the gradient values:
	copyFromDevice(traits.grads,d_traits.grads,np);
	
	// Read inputs from device memory
	if(traits.inputs) {
		copyFromDevice(traits.inputs, d_traits.inputs, nd);
	}

	if(traits.deltas) {
		copyFromDevice(traits.deltas, d_traits.deltas, nd); // only retrieve the deltas if requested.
	}
}

extern "C" {

void gd_errfunc(BPTraits<double>& traits)
{
		_gd_errfunc(traits);
}

void gd_errfunc_f(BPTraits<float>& traits)
{
		_gd_errfunc(traits);
}

}
