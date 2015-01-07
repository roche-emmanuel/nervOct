#include <nervCUDA.h>

#include <nerv_kernels.h>

extern "C" {

void costFunc_device(unsigned int nl, unsigned int np, unsigned int* lsizes, unsigned int nsamples,
	double* d_params, double* d_X, double* d_yy, double lambda, double& J, double* d_grads, double* d_deltas, double* d_inputs, double* d_regw)
{
	// getLastCudaError("Checkpoint1");
	unsigned int nt = nl-1; // number of matrices evolved.

	BPComputeTraits<double> traits;
	traits.params = d_params;
	traits.inputs = d_inputs;
	traits.deltas = d_deltas;
	traits.grads = d_grads;
	traits.yy = d_yy;
	traits.wX = d_X;
	traits.lambda = lambda;

  for(unsigned int i=0; i<nt;++i) {
  	// We compute the activation and input values for the given layer:

  	// The kernel compute the values of zi and a(i+1) 
  	// (note that the value or a(0) is already loaded in the Activation vector).
  	// even if we compute the a(i+1) matrix we actually discard completely the first column
  	// in this matrix (colu of intercept terms). As a result we just need to mapped the GPU grid to
  	// the dimension of of the sub z(i) matrix (which is transposed.)
  	// THe dimensions for z(i) are: lsize(i+1) * nsamples
  	// When this is transposed we get: nsamples * lsize(i+1);
		traits.nrows = lsizes[i+1];
		traits.niter = lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.
		traits.ncols = nsamples;

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((BLOCK_SIZE + traits.ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + traits.nrows-1)/BLOCK_SIZE);

		// Also we will need access to the theta_i matrix so we need to keep track of its global offset in the
		// network parameters array.
		// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");
		// ComputeActivation<<<dimGrid, dimBlock>>>(theta_offset, input_offset, next_input_offset,
		// 	nrows, ncols, ncolT, d_params, d_inputs, d_X);
		ComputeActivation<<<dimGrid, dimBlock>>>(traits);

		// CHECK_KERNEL();

		// update the offsets:
		traits.theta_offset += lsizes[i+1]*(lsizes[i]+1);
		traits.input_offset = traits.next_input_offset;
		traits.next_input_offset += traits.nrows*traits.ncols;
  }

  double* d_hx = d_inputs + traits.input_offset;

  // Here we can compute the cost now:
  // The hx matrix is mapped to the last z matrix. eg at i=nt-1
  // So its dimensions are lsizes[nt-1+1] * nsamples = lsizes[nl-1] * nsamples
  // same dimensions for the yy matrix, and we want to perform reduction other those 2 matrices
	J = 0.0;
	unsigned int count = nsamples*lsizes[nt];
	reduce_cost_device(d_hx, d_yy, count, J);
	// CHECK_KERNEL()

	J /= (double)nsamples;

	double Jreg = 0.0;
	reduce_cost_reg_device(d_params, d_regw, np, Jreg);
	// CHECK_KERNEL()

	J += (Jreg*lambda)/(2.0);

	// Prepare the computation of the delta matrices in reverse order:

	// remove the last theta matrix size from the theta offset so that we can use
	// that offset to retrieve the proper theta matrix:
	traits.theta_offset -= lsizes[nt]*(lsizes[nt-1]+1);

	// Prepare the offset for the gradient array:
	// keep in mind we start with the latest theta matrix:
	traits.grad_offset = np - lsizes[nt]*(lsizes[nt-1]+1);

	for(unsigned int i=nt;i>0;--i) {
		traits.nrows = lsizes[i];
		traits.ncols = nsamples;
		traits.niter = lsizes[i+1];
		unsigned int count = traits.nrows * traits.ncols;

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((BLOCK_SIZE + traits.ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + traits.nrows-1)/BLOCK_SIZE);


		if(i==nt) {
			// we should just copy the difference of hx and yy into the z matrix.
			// CHECK_KERNEL()
			// InitLastDelta<<<dimGrid, dimBlock>>>(input_offset, nrows, ncols, d_deltas, d_inputs, d_yy);
			InitLastDelta<<<dimGrid, dimBlock>>>(traits);
			// CHECK_KERNEL()
		}
		else {
			// We compute the delta from the previous delta:
			// We start this computation for delta(nt-1). this matrix is build from theta(nt-1) and delta(nt).
			// also in the process we use the input matrix z(nt-2)
			// ComputeDelta<<<dimGrid, dimBlock>>>(theta_offset, input_offset, delta_offset, next_delta_offset, nrows, ncols, niter, d_params, d_inputs, d_deltas);
			ComputeDelta<<<dimGrid, dimBlock>>>(traits);
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
		dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
		dimGrid = dim3((BLOCK_SIZE + traits.ncols-1)/BLOCK_SIZE, (BLOCK_SIZE + traits.nrows-1)/BLOCK_SIZE);

    traits.input_offset -= lsizes[i-1]*nsamples; // we remove the size of the next delta matrix to be computed. which is also the size of the next z matrix we will use.
		// logDEBUG("GPU: Gradient at i="<<i<<" of size "<< nrows <<" x " << ncols<<", offset="<<grad_offset<<", input_offset="<<input_offset<<", nsamples="<<nsamples);

		// ComputeGradient<<<dimGrid, dimBlock>>>(theta_offset, input_offset, delta_offset, grad_offset, nrows, ncols, niter, d_X, d_params, d_inputs, d_deltas, d_grads, lambda);
		ComputeGradient<<<dimGrid, dimBlock>>>(traits);
		// CHECK_KERNEL()

		// update the gradient offset by removing the size of the next gradient matrix to be computed:
		// except for the last iteration where the value is not available:
		if(i>1) {
			traits.grad_offset -= lsizes[i-1]*(lsizes[i-2]+1);
		}
	}
}

void costFunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* nn_params, double* X, double* yy, double lambda, double& J, double* gradients, double* deltas, double* inputs)
{
	// Allocate the device memory:
	size_t size;
	cudaError_t err;

	// cudaDeviceSynchronize();

	// size = nl * sizeof(unsigned int);
	// double* d_lsizes = NULL;
	// checkCudaErrors(cudaMalloc(&d_lsizes, size));
	// checkCudaErrors(cudaMemcpy(d_lsizes, lsizes, size, cudaMemcpyHostToDevice));

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

	// prepare regularization weigths:
	double* h_regw = new double[size];
	memset(h_regw,0,size);

	// prepare the regularization correction:
	double* rptr = h_regw;

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
	double* d_regw = NULL;
	checkCudaErrors(cudaMalloc(&d_regw, size));

	checkCudaErrors(cudaMemcpy(d_regw, h_regw, size, cudaMemcpyHostToDevice));

	// Also allocation the gradient array, with the same number of elements:
	double* d_grads = NULL;
	checkCudaErrors(cudaMalloc(&d_grads, size));
	checkCudaErrors(cudaMemset(d_grads,0,size));

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

	// Call the actual method to perform the computations:
	costFunc_device(nl, np, lsizes, nsamples, d_params, d_X, d_yy, lambda, J, d_grads, d_deltas, d_inputs, d_regw);

	// Here we should also read back the gradient values:
	checkCudaErrors(cudaMemcpy(gradients, d_grads, sizeof(double)*np, cudaMemcpyDeviceToHost));
	// memset(gradients,0,sizeof(double)*np);
	
	// Read inputs from device memory
	if(inputs) {
		checkCudaErrors(cudaMemcpy(inputs, d_inputs, input_size, cudaMemcpyDeviceToHost));
	}

	if(deltas) {
		checkCudaErrors(cudaMemcpy(deltas, d_deltas, sizeof(double)*nd, cudaMemcpyDeviceToHost)); // only retrieve the deltas if requested.
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

}