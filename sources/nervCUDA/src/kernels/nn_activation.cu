#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
int nn_activation_device(BPDeviceTraits<T>& d_traits)
{
	unsigned int nt = d_traits.nl-1; // number of matrices evolved.
	unsigned int* lsizes = d_traits.lsizes;
	unsigned int nsamples = d_traits.nsamples;
	T* wmults = d_traits.wmults;
	T* dropouts = d_traits.dropouts;

	cudaStream_t stream = d_traits.stream;

	BPComputeTraits<T> traits;
	traits = d_traits;
	
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
		// logDEBUG("Using grid size: ("<<dimGrid.x<<" x "<<dimGrid.y<<")");

		traits.nrows = nrows;
		traits.ncols = ncols;
		traits.niter = ncolT;

		if(wmults)
			traits.wmult = wmults[i];

		// Also we will need access to the theta_i matrix so we need to keep track of its global offset in the
		// network parameters array.
		if(dropouts) {
			// Update the bias weights to be used for this layer computation:
			rand_weights_device(traits.randStates, traits.wbias + traits.wbias_offset, dropouts[i], ncols, traits.bias);

			traits.layer_dropout = i==(nt-1) ? (T)1.0 : dropouts[i+1]; // we don't want to drop anything from the output layer.
			ComputeActivationWithDropout<<<dimGrid, dimBlock,0,stream>>>(traits);
		}
		else {
			ComputeActivation<<<dimGrid, dimBlock,0,stream>>>(traits);
		}

		// update the offsets:
		traits.wbias_offset += ncols;
		traits.theta_offset += lsizes[i+1]*(lsizes[i]+1);
		traits.input_offset = traits.next_input_offset;
		traits.next_input_offset += nrows*ncols;
  }

  return traits.input_offset;
}


template <typename T>
void _nn_predict(BPTraits<T>& traits)
{
	BPDeviceTraits<T> d_traits;
	d_traits = traits;

 	int input_offset = nn_activation_device(d_traits);

 	if(traits.hx) {
		copyFromDevice(traits.hx, d_traits.inputs+input_offset, traits.ny());
		// copyFromDevice(traits.hx, d_traits.predictions(), traits.ny()); 		
 	}
}

template <typename T>
void _nn_predict_cpu(BPTraits<T>& traits)
{
	unsigned int nl = traits.nl;
	unsigned int* lsizes = traits.lsizes;
	unsigned int nsamples = traits.nsamples_train;
	T* params = traits.params;
	T* X = traits.X;
	T* hx = traits.hx;
	T bias = traits.bias;
	T* wmults = traits.wmults;

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

void nn_predict(BPTraits<double>& traits)
{
	_nn_predict(traits);
}

void nn_predict_f(BPTraits<float>& traits)
{
	_nn_predict(traits);
}

void nn_predict_cpu(BPTraits<double>& traits)
{
	_nn_predict_cpu(traits); //nl, lsizes, nsamples, params, X, hx, bias, wmults);
}

void nn_predict_cpu_f(BPTraits<float>& traits)
{
	_nn_predict_cpu(traits); //nl, lsizes, nsamples, params, X, hx, bias, wmults);
}

}
