#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
int nn_activation_device(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	T* d_params, T* d_X, T* d_inputs, T bias, T* wmults, cudaStream_t stream)
{
	unsigned int nt = nl-1; // number of matrices evolved.

	// offset used to locate the theta_i matrix in the d_params array.
	unsigned int theta_offset = 0;

	// Offset used for the z(i) matrix on iteration i
	int input_offset = 0;

	int next_input_offset = 0; //nsamples*lsizes[1];

	T wmult = 1.0;

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

		if(wmults)
			wmult = wmults[i];

		// Also we will need access to the theta_i matrix so we need to keep track of its global offset in the
		// network parameters array.
		// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");
		ComputeActivation<<<dimGrid, dimBlock, 0, stream>>>(theta_offset, input_offset, next_input_offset,
			nrows, ncols, ncolT, d_params, d_inputs, d_X, bias, wmult);
		// CHECK_KERNEL();

		// update the offsets:
		theta_offset += lsizes[i+1]*(lsizes[i]+1);
		input_offset = next_input_offset;
		next_input_offset += nrows*ncols;
  }

  return input_offset;
}


// Explicit instanciation:
// template int nn_activation_device(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
// 	double* d_params, double* d_X, double* d_inputs, double bias, cudaStream_t stream);

// template int nn_activation_device(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
// 	float* d_params, float* d_X, float* d_inputs, float bias, cudaStream_t stream);


template <typename T>
void _nn_predict(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	T* params, T* X, T* hx, T bias, T* wmults)
{
	size_t size;
	unsigned int nt = nl -1;

	// Compute the number of parameters:
	// and compute the number of activation (eg. inputs) values:
	unsigned int np = 0;
	unsigned int ni = 0;
	for(unsigned int i=0;i<nt;++i) {
		np += lsizes[i + 1] * (lsizes[i] + 1);
    ni += lsizes[i + 1] * nsamples;
	}

	size = np * sizeof(T);
	T* d_params = NULL;
	checkCudaErrors(cudaMalloc(&d_params, size));
	checkCudaErrors(cudaMemcpy(d_params, params, size, cudaMemcpyHostToDevice));
	
	size = lsizes[0]*nsamples * sizeof(T);
	T* d_X = NULL;
	checkCudaErrors(cudaMalloc(&d_X, size));
	checkCudaErrors(cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice));

	size = ni * sizeof(T);
	T* d_inputs = NULL;
	checkCudaErrors(cudaMalloc(&d_inputs, size));


 	int input_offset = nn_activation_device(nl,lsizes,nsamples,d_params,d_X,d_inputs,bias,wmults);

 	size = lsizes[nt]*nsamples * sizeof(T);
	checkCudaErrors(cudaMemcpy(hx, d_inputs+input_offset, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_params));
	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_inputs));
}

template <typename T>
void _nn_predict_cpu(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	T* params, T* X, T* hx, T bias, T* wmults)
{
	// method used to compute the activation on the CPU.
	unsigned int nt = nl -1;

	// Compute the number of parameters:
	// and compute the number of activation (eg. inputs) values:
	unsigned int np = 0;
	unsigned int ni = 0;
	for(unsigned int i=0;i<nt;++i) {
		np += lsizes[i + 1] * (lsizes[i] + 1);
    ni += lsizes[i + 1] * nsamples;
	}

	// Prepare the input array:
	T* inputs = new T[ni];

	// offset used to locate the theta_i matrix in the d_params array.
	unsigned int theta_offset = 0;

	// Offset used for the z(i) matrix on iteration i
	int input_offset = 0;

	int next_input_offset = 0; //nsamples*lsizes[1];
	T mult = 1.0; // default weight multiplier value.

  for(unsigned int i=0; i<nt;++i) {
  	// We compute the activation and input values for the given layer:

		unsigned int nrows = lsizes[i+1];
		unsigned int ncolT = lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.
		unsigned int ncols = nsamples;

		// Check if a weight multiplier is provided for this layer:
		if(wmults)
			mult = wmults[i];

		for(unsigned int r=0;r<nrows;++r) {
			for(unsigned int c=0;c<ncols;++c) {
				// compute the activation on unit r, for sample c;
				T val = bias*params[theta_offset + r];

				for(unsigned int j=0;j<ncolT;++j) {
					// Add the element theta(r,j+1)*a(j,c)
					// if we are on i==0, then we are using X as activation, and in that case
					// we need to transpose the value:
					if(i==0) {
						val += params[theta_offset + nrows*(j+1) + r] * X[nsamples*j + c];	
					}
					else {
						val += params[theta_offset + nrows*(j+1) + r] * inputs[input_offset + ncolT*c + j];	
					}
				}

				// Now assign the computed value to the input array:
				// The compute value is a(r,c)
				inputs[next_input_offset + nrows*c + r] = (T)(1.0/(1.0+exp(-val*mult)));
			}
		}

		// update the offsets:
		theta_offset += lsizes[i+1]*(lsizes[i]+1);
		input_offset = next_input_offset;
		next_input_offset += nrows*ncols;
  }

  // Now we need to copy the last input data in the hx matrix:
  memcpy(hx,inputs+input_offset,lsizes[nt]*nsamples*sizeof(T));

	// Delete 
	delete [] inputs;	
}

extern "C" {

void nn_predict(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, double* params, double* X, double* hx, double bias, double* wmults)
{
	_nn_predict(nl, lsizes, nsamples, params, X, hx, bias, wmults);
}

void nn_predict_f(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, float* params, float* X, float* hx, float bias, float* wmults)
{
	_nn_predict(nl, lsizes, nsamples, params, X, hx, bias, wmults);
}

void nn_predict_cpu(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, double* params, double* X, double* hx, double bias, double* wmults)
{
	_nn_predict_cpu(nl, lsizes, nsamples, params, X, hx, bias, wmults);
}

void nn_predict_cpu_f(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, float* params, float* X, float* hx, float bias, float* wmults)
{
	_nn_predict_cpu(nl, lsizes, nsamples, params, X, hx, bias, wmults);
}

}
