#include <nervCUDA.h>

#include <nerv_kernels.h>

extern "C" {

void costFunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* nn_params, double* X, double* yy, double lambda, double* inputs, double& J, double* gradients, double* deltas)
{
	// Allocate the device memory:
	size_t size;
	cudaError_t err;

	// cudaDeviceSynchronize();

	size = nl * sizeof(unsigned int);
	double* d_lsizes = NULL;
	checkCudaErrors(cudaMalloc(&d_lsizes, size));
	checkCudaErrors(cudaMemcpy(d_lsizes, lsizes, size, cudaMemcpyHostToDevice));

	// Compute the total number of parameters in this network:
	unsigned int np = 0;
	unsigned int nt = nl-1; // number of matrices evolved.

	for(unsigned int i=0;i<nt;++i) {
		np += lsizes[i+1]*(lsizes[i]+1);
	}

	size = np * sizeof(double);
	double* d_params = NULL;
	checkCudaErrors(cudaMalloc(&d_params, size));
	checkCudaErrors(cudaMemcpy(d_params, nn_params, size, cudaMemcpyHostToDevice));

#if 0
	// Also allocation the gradient array, with the same number of elements:
	double* d_grads = NULL;
	checkCudaErrors(cudaMalloc(&d_grads, size));
#endif

	// Compute the total number of delta coefficients:
	unsigned int nd = 0;
	for(unsigned int i=1;i<nl;++i) {
		nd += lsizes[i]*nsamples;
	}

	size = nd*sizeof(double);
	double* d_deltas = NULL;
	checkCudaErrors(cudaMalloc(&d_deltas, size));
	checkCudaErrors(cudaMemset(d_deltas,0,size));

	// Prepare the X matrix:
	size = sizeof(double) * nsamples * lsizes[0];
	double* d_X = NULL;
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

	size = nsamples * count * sizeof(double);
	size_t input_size = size;
	double* d_inputs = NULL;
	err = cudaMalloc(&d_inputs, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc inputs: "<<cudaGetErrorString(err));
	}
	cudaMemset(d_inputs,0,size); // This is needed for debugging only.

	// Copy the label matrix:	
	size = nsamples * lsizes[nt] * sizeof(double);
	double* d_yy = NULL;
	err = cudaMalloc(&d_yy, size);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA malloc yy: "<<cudaGetErrorString(err));
	}
	cudaMemcpy(d_yy, yy, size, cudaMemcpyHostToDevice);


	// offset used to locate the theta_i matrix in the d_params array.
	unsigned int theta_offset = 0;

	// Offset used for the z(i) matrix on iteration i
	unsigned int input_offset = 0;

	unsigned int next_input_offset = 0; //nsamples*lsizes[1];

	double reg_correction = 0.0;
	double* tptr = nn_params;
	double rval;

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

		for(unsigned int j=0;j<nrows;++j) {
			rval = (*tptr++);
			reg_correction += rval*rval;
		}
		tptr += nrows*ncolT;

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((BLOCK_SIZE + ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + nrows-1)/BLOCK_SIZE);

		// Also we will need access to the theta_i matrix so we need to keep track of its global offset in the
		// network parameters array.
		// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");
		ComputeActivation<<<dimGrid, dimBlock>>>(theta_offset, input_offset, next_input_offset,
			nrows, ncols, ncolT, d_params, d_inputs, d_X);

		// update the offsets:
		theta_offset += lsizes[i+1]*(lsizes[i]+1);
		input_offset = next_input_offset;
		next_input_offset += nrows*ncols;
  }

  double* d_hx = d_inputs + input_offset;

  // Here we can compute the cost now:
  // The hx matrix is mapped to the last z matrix. eg at i=nt-1
  // So its dimensions are lsizes[nt-1+1] * nsamples = lsizes[nl-1] * nsamples
  // same dimensions for the yy matrix, and we want to perform reduction other those 2 matrices
	J = 0.0;
	count = nsamples*lsizes[nt];
	reduction_cost_device(d_hx, d_yy, count, J);

	J /= (double)nsamples;

	double Jreg = 0.0;
	reduction_cost_reg_device(d_params, np, Jreg);
	Jreg -= reg_correction;

	J += (Jreg*lambda)/(2.0*nsamples);

	// Read inputs from device memory
	err = cudaMemcpy(inputs, d_inputs, input_size, cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess) {
		logDEBUG("CUDA reading inputs: "<<cudaGetErrorString(err));
	}

#if 0
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

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((BLOCK_SIZE + ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + nrows-1)/BLOCK_SIZE);

		if(i==nt) {
			// we should just copy the difference of hx and yy into the z matrix.
			InitLastDelta<<<dimGrid, dimBlock>>>(nrows, ncols, d_deltas, d_hx, d_yy);
		}
		else {
			// We compute the delta from the previous delta:
			// We start this computation for delta(nt-1). this matrix is build from theta(nt-1) and delta(nt).
			// also in the process we use the input matrix z(nt-2)
			ComputeDelta<<<dimGrid, dimBlock>>>(theta_offset, input_offset, delta_offset, next_delta_offset, nrows, ncols, niter, d_params, d_inputs, d_deltas);
	
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
		dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
		dimGrid = dim3((BLOCK_SIZE + ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + nrows-1)/BLOCK_SIZE);

		ComputeGradient<<<dimGrid, dimBlock>>>(theta_offset, input_offset, delta_offset, grad_offset, nrows, ncols, niter, d_params, d_inputs, d_deltas, d_grads, lambda);

    input_offset -= lsizes[i-1]*nsamples; // we remove the size of the next delta matrix to be computed. which is also the size of the next z matrix we will use.

		// update the gradient offset by removing the size of the next gradient matrix to be computed:
		// except for the last iteration where the value is not available:
		if(i>1) {
			grad_offset -= lsizes[i-1]*(lsizes[i-2]+1);
		}
	}

	// Here we should also read back the gradient values:
	// checkCudaErrors(cudaMemcpy(gradients, d_grads, sizeof(double)*np, cudaMemcpyDeviceToHost));
	// memset(gradients,0,sizeof(double)*np);

	if(deltas)
		checkCudaErrors(cudaMemcpy(deltas, d_deltas, sizeof(double)*nd, cudaMemcpyDeviceToHost)); // only retrieve the deltas if requested.
#endif

	// cudaDeviceSynchronize();

	// Free device memory
	cudaFree(d_lsizes);
	cudaFree(d_params);
	cudaFree(d_inputs);	
	cudaFree(d_yy);	
	cudaFree(d_X);	
	cudaFree(d_deltas);	
#if 0
	cudaFree(d_grads);	
#endif
}

}