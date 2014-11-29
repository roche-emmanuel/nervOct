#include <nervCUDA.h>

#include <sgtcore.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

using namespace nerv;

GradientDescentClass::Traits::Traits() 
	: _nsamples(0), _maxiter(-1), _lambda(0.0),
	_lsizes(nullptr), _nl(0),
	_X_train(nullptr), _X_train_size(0),
	_y_train(nullptr), _y_train_size(0), 
	_params(nullptr), _nparams(0)
{
}

GradientDescentClass::Traits::~Traits()
{
}


unsigned int GradientDescentClass::Traits::nl() const
{
	return _nl;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::lsizes(unsigned int* layer_sizes, unsigned int nl)
{
	_nl = nl;
	_lsizes = layer_sizes;
	return *this;
}

unsigned int* GradientDescentClass::Traits::lsizes() const
{
	return _lsizes;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::nsamples(unsigned int num_samples)
{
	_nsamples = num_samples;
	return *this;
}

unsigned int GradientDescentClass::Traits::nsamples() const
{
	return _nsamples;
}

unsigned int GradientDescentClass::Traits::nparams() const
{
	return _nparams;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::X_train(GradientDescentClass::value_type* X, unsigned int size)
{
	_X_train_size = size;
	_X_train = X;
	return *this;
}

GradientDescentClass::value_type* GradientDescentClass::Traits::X_train() const
{
	return _X_train;
}

unsigned int GradientDescentClass::Traits::X_train_size() const
{
	return _X_train_size;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::y_train(GradientDescentClass::value_type* y, unsigned int size)
{
	_y_train_size = size;
	_y_train = y;
	return *this;
}

GradientDescentClass::value_type* GradientDescentClass::Traits::y_train() const
{
	return _y_train;
}

unsigned int GradientDescentClass::Traits::y_train_size() const
{
	return _y_train_size;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::params(GradientDescentClass::value_type* p, unsigned int nparams)
{
	_nparams = nparams;
	_params = p;
	return *this;
}

GradientDescentClass::value_type* GradientDescentClass::Traits::params() const
{
	return _params;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::maxiter(int num)
{
	_maxiter = num;
	return *this;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::lambda(value_type val)
{
	_lambda = val;
	return *this;
}

GradientDescentClass::value_type GradientDescentClass::Traits::lambda() const
{
	return _lambda;
}

int GradientDescentClass::Traits::maxiter() const
{
	return _maxiter;
}


GradientDescentClass::Traits::Traits(const GradientDescentClass::Traits& rhs)
{
	this->operator=(rhs);
}

GradientDescentClass::Traits& GradientDescentClass::Traits::operator=(const GradientDescentClass::Traits& rhs)
{
  _nl = rhs._nl;
  _nsamples = rhs._nsamples;
  _nparams = rhs._nparams;

  _lsizes = rhs._lsizes;
  _X_train = rhs._X_train;
  _y_train = rhs._y_train;
  _params = rhs._params;

	return *this;
}

GradientDescentClass::GradientDescentClass(const Traits& traits)
{
	// Assign the max number of iteration:
	_maxiter = traits.maxiter();

	// Assign regularization parameter:
	_lambda = traits.lambda();

	// ensure that the traits are usable:
	THROW_IF(traits.nl()<3,"Invalid nl value: "<<traits.nl())
	_nl = traits.nl();
	_nt = _nl-1;

	THROW_IF(!traits.lsizes(),"Invalid lsizes value.")
	_lsizes = traits.lsizes();

	THROW_IF(!traits.nsamples(),"Invalid nsamples value.")
	_nsamples = traits.nsamples();

	// Compute the number of parameters that are expected:
	unsigned int np = 0;
  for(unsigned int i=0;i<_nt;++i) {
    np += _lsizes[i+1]*(_lsizes[i]+1);
  }

	THROW_IF(traits.nparams()!=np,"Invalid nparams value: "<<traits.nparams()<<"!="<<np)
	THROW_IF(!traits.params(),"Invalid params value.")
	_np = np;

	// Compute the expected size for X:
	unsigned int nx = _nsamples*_lsizes[0];
	THROW_IF(traits.X_train_size()!=nx,"Invalid size for X: "<<traits.X_train_size()<<"!="<<nx)
	THROW_IF(!traits.X_train(),"Invalid X_train value.")

	// Compute the expected size for y:
	unsigned int ny = _nsamples*_lsizes[_nt];
	THROW_IF(traits.y_train_size()!=ny,"Invalid size for y: "<<traits.y_train_size()<<"!="<<ny)
	THROW_IF(!traits.y_train(),"Invalid y_train value.")
	
	// keep a copy of the traits:
	_traits = traits;

	// Now that we checked that the input data is valid, we should allocate the GPU resources:
	// First we allocate the stream that will be used for the main processing:
	checkCudaErrors(cudaStreamCreate(&_stream1));

	size_t size;

	// Load the X matrix on the GPU directly:
	size = sizeof(value_type) * nx;
	d_X_train = NULL;
	checkCudaErrors(cudaHostRegister(traits.X_train(), size,cudaHostRegisterDefault)); // register the memory as pinned memory.
	checkCudaErrors(cudaMalloc(&d_X_train, size));
	checkCudaErrors(cudaMemcpyAsync(d_X_train, traits.X_train(), size, cudaMemcpyHostToDevice,_stream1));

	// load the yy matrix on the GPU:
	size = sizeof(value_type) * ny;
	d_y_train = NULL;
	checkCudaErrors(cudaHostRegister(traits.y_train(), size,cudaHostRegisterDefault)); // register the memory as pinned memory.
	checkCudaErrors(cudaMalloc(&d_y_train, size));
	checkCudaErrors(cudaMemcpyAsync(d_y_train, traits.y_train(), size, cudaMemcpyHostToDevice,_stream1));

	// Load the parameters (weights) on the GPU:
	size = sizeof(value_type) * np;
	d_params = NULL;
	checkCudaErrors(cudaHostRegister(traits.params(), size,cudaHostRegisterDefault)); // register the memory as pinned memory.
	checkCudaErrors(cudaMalloc(&d_params, size));
	checkCudaErrors(cudaMemcpyAsync(d_params, traits.params(), size, cudaMemcpyHostToDevice,_stream1));

	// prepare regularization weigths:
	_regw = new value_type[size];
	memset(_regw,0,size);

	// prepare the regularization correction:
	value_type* rptr = _regw;

	for(unsigned int i=0; i<_nt;++i) {
		unsigned int nrows = _lsizes[i+1];
		unsigned int ncolT = _lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.

		rptr += nrows;
		unsigned int count = nrows*ncolT;

		for(unsigned int j=0;j<count;++j) {
			(*rptr++) = 1.0;
		}
	}

	// Prepare the reg weights for this network:
	d_regw = NULL;
	checkCudaErrors(cudaHostRegister(_regw, size,cudaHostRegisterDefault)); // register the memory as pinned memory.
	checkCudaErrors(cudaMalloc(&d_regw, size));
	checkCudaErrors(cudaMemcpyAsync(d_regw, _regw, size, cudaMemcpyHostToDevice,_stream1));
	

	// for the cost computation we will also need the grads and delta and input arrays:
	// Also allocation the gradient array, with the same number of elements:
	d_grads = NULL;
	checkCudaErrors(cudaMalloc(&d_grads, size));
	checkCudaErrors(cudaMemsetAsync(d_grads,0,size,_stream1));

	// Compute the total number of delta coefficients:
	unsigned int nd = 0;
	for(unsigned int i=1;i<_nl;++i) {
		nd += _lsizes[i]*_nsamples;
	}

	size = sizeof(value_type)*nd;
	d_deltas = NULL;
	checkCudaErrors(cudaMalloc(&d_deltas, size));
	checkCudaErrors(cudaMemsetAsync(d_deltas,0,size,_stream1));

	// finally we also need the inputs array:
	unsigned int ni = 0;
	for(unsigned int i=0;i<_nt;++i) {
		ni += _lsizes[i+1]*_nsamples;
	}

	size = sizeof(value_type) * ni;
	d_inputs = NULL;
	checkCudaErrors(cudaMalloc(&d_inputs, size));
	checkCudaErrors(cudaMemsetAsync(d_inputs,0,size,_stream1));
}

GradientDescentClass::~GradientDescentClass()
{
	// unregister the pinned memory:
	checkCudaErrors(cudaHostUnregister(_traits.X_train())); 
	checkCudaErrors(cudaHostUnregister(_traits.y_train())); 
	checkCudaErrors(cudaHostUnregister(_traits.params())); 
	checkCudaErrors(cudaHostUnregister(_regw)); 
	delete [] _regw;

	// free GPU buffers:
	checkCudaErrors(cudaFree(d_X_train));	
	checkCudaErrors(cudaFree(d_y_train));	
	checkCudaErrors(cudaFree(d_params));	
	checkCudaErrors(cudaFree(d_regw));	
	checkCudaErrors(cudaFree(d_grads));	
	checkCudaErrors(cudaFree(d_deltas));	
	checkCudaErrors(cudaFree(d_inputs));	

	// destroy the processing stream:
	checkCudaErrors(cudaStreamDestroy(_stream1));
}

void GradientDescentClass::run()
{
	int iter=0;
	value_type current_cost;

	// Run the iteration loop:
	while(_maxiter<0 || iter<_maxiter) {
		logDEBUG("Performing iteration "<<iter<<"...");

		// We start with the computation of the gradient from the current parameters.
		// costFunc_device(_nl, _np, _lsizes, _nsamples, d_params, 
		//   d_X_train, d_y_train, _lambda, current_cost, d_grads, d_deltas, d_inputs, d_regw);
		gd_errfunc_device(_nl, _np, _lsizes, _nsamples, d_params, 
			d_X_train, d_y_train, _lambda, current_cost, d_grads, d_deltas, d_inputs, d_regw);

		iter++;
	}

	// then compute the gradient.
	// update the weights with a learning rate.
	
}
