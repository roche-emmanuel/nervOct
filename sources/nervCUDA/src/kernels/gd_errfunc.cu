#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
void gd_errfunc_device(unsigned int nl, unsigned int np, unsigned int* lsizes, unsigned int nsamples,
	T* d_params, T* d_X, T* d_yy, T lambda, T& J, T* d_grads, T* d_deltas, T* d_inputs, T* d_regw, cudaStream_t stream)
{
	// getLastCudaError("Checkpoint1");

	unsigned int nt = nl-1; // number of matrices evolved.

	// offset used to locate the theta_i matrix in the d_params array.
	unsigned int theta_offset = 0;

	// Offset used for the z(i) matrix on iteration i
	int input_offset = 0;

	int next_input_offset = 0; //nsamples*lsizes[1];

  for(unsigned int i=0; i<nt;++i) {
  	// We compute the activation and input values for the given layer:

  	// The kernel compute the values of zi and a(i+1) 
  	// (note that the value or a(0) is already loaded in the Activation vector).
  	// even if we compute the a(i+1) matrix we actually discard completely the first column
  	// in this matrix (colu of intercept terms). As a result we just need to mapped the GPU grid to
  	// the dimension of of the sub z(i) matrix (which is transposed.)
  	// THe dimensions for z(i) are: lsize(i+1) * nsamples
  	// When this is transposed we get: nsamples * lsize(i+1);
		unsigned int nrows = lsizes[i+1];
		unsigned int ncolT = lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.
		unsigned int ncols = nsamples;

		dim3 dimBlock(blockSize, blockSize);
		dim3 dimGrid((blockSize + ncols-1)/blockSize, (blockSize + nrows-1)/blockSize);

		// Also we will need access to the theta_i matrix so we need to keep track of its global offset in the
		// network parameters array.
		// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");
		ComputeActivation<<<dimGrid, dimBlock, 0, stream>>>(theta_offset, input_offset, next_input_offset,
			nrows, ncols, ncolT, d_params, d_inputs, d_X);
		// CHECK_KERNEL();

		// update the offsets:
		theta_offset += lsizes[i+1]*(lsizes[i]+1);
		input_offset = next_input_offset;
		next_input_offset += nrows*ncols;
  }

  T* d_hx = d_inputs + input_offset;

  // Here we can compute the cost now:
  // The hx matrix is mapped to the last z matrix. eg at i=nt-1
  // So its dimensions are lsizes[nt-1+1] * nsamples = lsizes[nl-1] * nsamples
  // same dimensions for the yy matrix, and we want to perform reduction other those 2 matrices
	J = 0.0;
	unsigned int count = nsamples*lsizes[nt];
	reduce_cost_device(d_hx, d_yy, count, J, stream);
	// CHECK_KERNEL()

	J /= (T)nsamples;

	T Jreg = 0.0;
	reduce_cost_reg_device(d_params, d_regw, np, Jreg, stream);
	// CHECK_KERNEL()

	J += (T)((Jreg*lambda)/(2.0*nsamples));

	// Prepare the computation of the delta matrices in reverse order:

	// Offset to use when reading the delta matrix in the current iteration
	// except when next_delta_offset is 0, in that case we read the hx and yy matrices.
	unsigned int delta_offset = 0;

	// Offset to use when writing the delta matrix in the current iteration
	unsigned int next_delta_offset = 0;

	// remove the last theta matrix size from the theta offset so that we can use
	// that offset to retrieve the proper theta matrix:
	theta_offset -= lsizes[nt]*(lsizes[nt-1]+1);

	// initially the input_offset is pointing on the hx matrix which is z(nt-1) with our convention (eg. z(0) is not in the array.)
	// But the first one we will need is actually the one before that: z(nt-2)
	// So we need to update the offset, and remove the size of the matrix z(nt-2) ! (pointer is at the beginning of z(nt-1))
	// Note: This is now done inside the loop:
	// input_offset -= lsizes[nt-1]*nsamples;

	// Prepare the offset for the gradient array:
	// keep in mind we start with the latest theta matrix:
	unsigned int grad_offset = np - lsizes[nt]*(lsizes[nt-1]+1);

	for(unsigned int i=nt;i>0;--i) {
		unsigned int nrows = lsizes[i];
		unsigned int ncols = nsamples;
		unsigned int niter = lsizes[i+1];
		unsigned int count = nrows*ncols;

		dim3 dimBlock(blockSize, blockSize);
		dim3 dimGrid((blockSize + ncols-1)/blockSize, (blockSize + nrows-1)/blockSize);

		if(i==nt) {
			// we should just copy the difference of hx and yy into the z matrix.
			// CHECK_KERNEL()
			InitLastDelta<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, d_deltas, d_hx, d_yy);
			// CHECK_KERNEL()
		}
		else {
			// We compute the delta from the previous delta:
			// We start this computation for delta(nt-1). this matrix is build from theta(nt-1) and delta(nt).
			// also in the process we use the input matrix z(nt-2)
			ComputeDelta<<<dimGrid, dimBlock, 0, stream>>>(theta_offset, input_offset, delta_offset, next_delta_offset, nrows, ncols, niter, d_params, d_inputs, d_deltas);
			// CHECK_KERNEL()

			// once the computation is done for that layer we move to the previous layer:
			theta_offset -= lsizes[i]*(lsizes[i-1]+1);
		}

		delta_offset = next_delta_offset;
		next_delta_offset += count;

		// At this point we have the previous theta matrix (eg. theta(i-1) pointed by theta_offset. (both when i=nt and i<nt).
		// and thats the matrix we need to compute the gradient values.
		// the gradient mat has the same size as the current theta matrix.
		// similarly, the input_offset is pointing on z(i-2) which is the one we need to perform the computation too.
		// and delta_offset points to the delta matrix we just wrote (eg. delta(i)).
		nrows = lsizes[i];
		ncols = lsizes[i-1]+1;
		niter = nsamples;
		count = nrows*ncols;

		// Compute the gradient:
		dimBlock = dim3(blockSize, blockSize);
		dimGrid = dim3((blockSize + ncols-1)/blockSize, (blockSize + nrows-1)/blockSize);

    input_offset -= lsizes[i-1]*nsamples; // we remove the size of the next delta matrix to be computed. which is also the size of the next z matrix we will use.
		// logDEBUG("GPU: Gradient at i="<<i<<" of size "<< nrows <<" x " << ncols<<", offset="<<grad_offset<<", input_offset="<<input_offset<<", nsamples="<<nsamples);

		ComputeGradient<<<dimGrid, dimBlock, 0, stream>>>(theta_offset, input_offset, delta_offset, grad_offset, nrows, ncols, niter, d_X, d_params, d_inputs, d_deltas, d_grads, lambda);
		// CHECK_KERNEL()

		// update the gradient offset by removing the size of the next gradient matrix to be computed:
		// except for the last iteration where the value is not available:
		if(i>1) {
			grad_offset -= lsizes[i-1]*(lsizes[i-2]+1);
		}
	}
}

// Explicit specializations:
// Note that this is not needed since those templates are instantiated anyway below in the "C" functions.

// template void gd_errfunc_device<double, BLOCK_SIZE>(unsigned int nl, unsigned int np, unsigned int* lsizes, unsigned int nsamples,
// 	double* d_params, double* d_X, double* d_yy, double lambda, double& J, double* d_grads, double* d_deltas, double* d_inputs, double* d_regw);


template <typename T>
void _gd_errfunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	T* nn_params, T* X, T* yy, T lambda, T& J, T* gradients, T* deltas, T* inputs)
{
	// Allocate the device memory:
	size_t size;
	cudaError_t err;

	// cudaDeviceSynchronize();

	// size = nl * sizeof(unsigned int);
	// T* d_lsizes = NULL;
	// checkCudaErrors(cudaMalloc(&d_lsizes, size));
	// checkCudaErrors(cudaMemcpy(d_lsizes, lsizes, size, cudaMemcpyHostToDevice));

	// Compute the total number of parameters in this network:
	unsigned int np = 0;
	unsigned int nt = nl-1; // number of matrices evolved.

	for(unsigned int i=0;i<nt;++i) {
		np += lsizes[i+1]*(lsizes[i]+1);
	}

	size = np * sizeof(T);
	T* d_params = NULL;
	checkCudaErrors(cudaMalloc(&d_params, size));
	checkCudaErrors(cudaMemcpy(d_params, nn_params, size, cudaMemcpyHostToDevice));

	// prepare regularization weigths:
	T* h_regw = new T[size];
	memset(h_regw,0,size);

	// prepare the regularization correction:
	T* rptr = h_regw;

	for(unsigned int i=0; i<nt;++i) {
		unsigned int nrows = lsizes[i+1];
		unsigned int ncolT = lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.

		rptr += nrows;
		unsigned int count = nrows*ncolT;

		for(unsigned int j=0;j<count;++j) {
			(*rptr++) = 1.0;
		}
	}


	// Prepare the reg weights for this network:
	T* d_regw = NULL;
	checkCudaErrors(cudaMalloc(&d_regw, size));

	checkCudaErrors(cudaMemcpy(d_regw, h_regw, size, cudaMemcpyHostToDevice));

	// Also allocation the gradient array, with the same number of elements:
	T* d_grads = NULL;
	checkCudaErrors(cudaMalloc(&d_grads, size));
	checkCudaErrors(cudaMemset(d_grads,0,size));

	// Compute the total number of delta coefficients:
	unsigned int nd = 0;
	for(unsigned int i=1;i<nl;++i) {
		nd += lsizes[i]*nsamples;
	}

	size = nd*sizeof(T);
	T* d_deltas = NULL;
	checkCudaErrors(cudaMalloc(&d_deltas, size));
	checkCudaErrors(cudaMemset(d_deltas,0,size));

	// Prepare the X matrix:
	size = sizeof(T) * nsamples * lsizes[0];
	T* d_X = NULL;
	err = cudaMalloc(&d_X, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc X: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);

	// Prepare the input data:
	// the size of each input matrix is lsize[i+1]*nsamples;
	// and we need input 0 to nt-1, inclusive.
	// So that's nl input matrices.
	unsigned int count = 0;
	for(unsigned int i=0;i<nt;++i) {
		count += lsizes[i+1];
	}

	size = nsamples * count * sizeof(T);
	size_t input_size = size;
	T* d_inputs = NULL;
	err = cudaMalloc(&d_inputs, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc inputs: "<<cudaGetErrorString(err));
	}
	cudaMemset(d_inputs,0,size); // This is needed for debugging only.

	// Copy the label matrix:	
	size = nsamples * lsizes[nt] * sizeof(T);
	T* d_yy = NULL;
	err = cudaMalloc(&d_yy, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc yy: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_yy, yy, size, cudaMemcpyHostToDevice);

	// Call the actual method to perform the computations:
	// costFunc_device(nl, np, lsizes, nsamples, d_params, d_X, d_yy, lambda, J, d_grads, d_deltas, d_inputs, d_regw);
	gd_errfunc_device<T>(nl, np, lsizes, nsamples, d_params, d_X, d_yy, lambda, J, d_grads, d_deltas, d_inputs, d_regw);

	// Here we should also read back the gradient values:
	checkCudaErrors(cudaMemcpy(gradients, d_grads, sizeof(T)*np, cudaMemcpyDeviceToHost));
	// memset(gradients,0,sizeof(T)*np);
	
	// Read inputs from device memory
	if(inputs) {
		checkCudaErrors(cudaMemcpy(inputs, d_inputs, input_size, cudaMemcpyDeviceToHost));
	}

	if(deltas) {
		checkCudaErrors(cudaMemcpy(deltas, d_deltas, sizeof(T)*nd, cudaMemcpyDeviceToHost)); // only retrieve the deltas if requested.
	}

	// cudaDeviceSynchronize();

	// Free device memory
	// checkCudaErrors(cudaFree(d_lsizes));
	checkCudaErrors(cudaFree(d_params));
	checkCudaErrors(cudaFree(d_regw));
	checkCudaErrors(cudaFree(d_inputs));	
	checkCudaErrors(cudaFree(d_yy));	
	checkCudaErrors(cudaFree(d_X));	
	checkCudaErrors(cudaFree(d_deltas));	
	checkCudaErrors(cudaFree(d_grads));	
	delete [] h_regw;
}

// template <typename T>
// void gd_errfunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
// 	double* nn_params, double* X, double* yy, double lambda, double& J, double* gradients, double* deltas, double* inputs)


extern "C" {

void gd_errfunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* nn_params, double* X, double* yy, double lambda, double& J, double* gradients, double* deltas, double* inputs)
{
		_gd_errfunc(nl, lsizes, nsamples, nn_params, X, yy, lambda, J, gradients, deltas, inputs);
}

void gd_errfunc_f(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	float* nn_params, float* X, float* yy, float lambda, float& J, float* gradients, float* deltas, float* inputs)
{
		_gd_errfunc(nl, lsizes, nsamples, nn_params, X, yy, lambda, J, gradients, deltas, inputs);
}

}
