#include <nervCUDA.h>

#include <sgtcore.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

#define LOG2 0.6931471805599453

using namespace nerv;

GradientDescentClass::Traits::Traits() 
	: _nsamples(0), _maxiter(0), _lambda(0.0),
	_lsizes(nullptr), _nl(0),
	_X_train(nullptr), _X_train_size(0),
	_y_train(nullptr), _y_train_size(0), 
	_X_cv(nullptr), _X_cv_size(0),
	_y_cv(nullptr), _y_cv_size(0), 
	_params(nullptr), _nparams(0),
	_mu(0.0), _epsilon(0.0), _miniBatchSize(0),
	_validationWindowSize(0), _bias(1.0)
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

int GradientDescentClass::Traits::maxiter() const
{
	return _maxiter;
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

GradientDescentClass::Traits& GradientDescentClass::Traits::momentum(value_type mu)
{
	_mu = mu;
	return *this;
}

GradientDescentClass::value_type GradientDescentClass::Traits::momentum() const
{
	return _mu;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::learningRate(value_type lr)
{
	_epsilon = lr;
	return *this;
}

GradientDescentClass::value_type GradientDescentClass::Traits::learningRate() const
{
	return _epsilon;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::bias(value_type b)
{
	_bias = b;
	return *this;
}

GradientDescentClass::value_type GradientDescentClass::Traits::bias() const
{
	return _bias;
}


GradientDescentClass::Traits& GradientDescentClass::Traits::miniBatchSize(unsigned int size)
{
	_miniBatchSize = size;
	return *this;
}

unsigned int GradientDescentClass::Traits::miniBatchSize() const
{
	return _miniBatchSize;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::validationWindowSize(unsigned int size)
{
	_validationWindowSize = size;
	return *this;
}

unsigned int GradientDescentClass::Traits::validationWindowSize() const
{
	return _validationWindowSize;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::X_cv(GradientDescentClass::value_type* X, unsigned int size)
{
	_X_cv_size = size;
	_X_cv = X;
	return *this;
}

GradientDescentClass::value_type* GradientDescentClass::Traits::X_cv() const
{
	return _X_cv;
}

unsigned int GradientDescentClass::Traits::X_cv_size() const
{
	return _X_cv_size;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::y_cv(GradientDescentClass::value_type* y, unsigned int size)
{
	_y_cv_size = size;
	_y_cv = y;
	return *this;
}

GradientDescentClass::value_type* GradientDescentClass::Traits::y_cv() const
{
	return _y_cv;
}

unsigned int GradientDescentClass::Traits::y_cv_size() const
{
	return _y_cv_size;
}


GradientDescentClass::Traits::Traits(const GradientDescentClass::Traits& rhs)
{
	this->operator=(rhs);
}

GradientDescentClass::Traits& GradientDescentClass::Traits::operator=(const GradientDescentClass::Traits& rhs)
{
  _nl = rhs._nl;
  _nsamples = rhs._nsamples;

  _lsizes = rhs._lsizes;
  
  _X_train = rhs._X_train;
  _X_train_size = rhs._X_train_size;

  _y_train = rhs._y_train;
  _y_train_size = rhs._y_train_size;

  _params = rhs._params;
  _nparams = rhs._nparams;

  _maxiter = rhs._maxiter;
  _lambda = rhs._lambda;

  _mu = rhs._mu;
  _epsilon = rhs._epsilon;

  _miniBatchSize = rhs._miniBatchSize;

  _validationWindowSize = rhs._validationWindowSize;

  _X_cv = rhs._X_cv;
  _X_cv_size = rhs._X_cv_size;

  _y_cv = rhs._y_cv;
  _y_cv_size = rhs._y_cv_size;

  _bias = rhs._bias;

	return *this;
}

GradientDescentClass::Traits::Traits(const TrainingSet<value_type>& tr)
{
	_nl = tr.nl();
	_nsamples = tr.nsamples();

	_lsizes = tr.lsizes();
	
	_X_train = tr.X_train();
	_X_train_size = tr.X_train_size();

	_y_train = tr.y_train();
	_y_train_size = tr.y_train_size();

	_X_cv = tr.X_cv();
	_X_cv_size = tr.X_cv_size();

	_y_cv = tr.y_cv();
	_y_cv_size = tr.y_cv_size();

	_params = tr.params();
	_nparams = tr.np();

	_maxiter = tr.maxiter();
	_lambda = tr.lambda();

	_mu = 0.0;
	_epsilon = 0.0;
	_miniBatchSize = 0;
	_validationWindowSize = 0;
	_bias = 1.0;
}

GradientDescentClass::GradientDescentClass(const Traits& traits)
{
	_bestIter = 0;

	// Assign the max number of iteration:
	_maxiter = traits.maxiter();

	// Assign regularization parameter:
	_lambda = traits.lambda();

	_mumax = traits.momentum();
	_mu = 0.0; // will be initialized later.

	_epsilon = traits.learningRate();

	_miniBatchSize = traits.miniBatchSize();

	_validationWindowSize = traits.validationWindowSize();
	
	_minCvCostDec = 0.00001;

	// Retrieve the bias value:
	_bias = traits.bias();

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
	
	_nsamples_cv = traits.X_cv_size()/_lsizes[0];
	unsigned int ns_cv = traits.y_cv_size()/_lsizes[_nt];
	THROW_IF(_nsamples_cv!=ns_cv,"Mismatch in computation of _nsamples_cv"<<_nsamples_cv<<"!="<<ns_cv)

	THROW_IF(_miniBatchSize>_nsamples/2,"mini-batch size is too big: "<<_miniBatchSize<<">"<<(_nsamples/2));

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

	// Prepare the cv datasets if applicable:
	THROW_IF(traits.validationWindowSize()>0 && (!traits.X_cv() || !traits.y_cv()), "Invalid cv datasets.");
	d_X_cv = NULL;
	d_y_cv = NULL;
	
	if(traits.validationWindowSize()>0) {
		// Load the Xcv matrix on the GPU directly:
		size = sizeof(value_type) * traits.X_cv_size();
		checkCudaErrors(cudaHostRegister(traits.X_cv(), size,cudaHostRegisterDefault)); // register the memory as pinned memory.
		checkCudaErrors(cudaMalloc(&d_X_cv, size));
		checkCudaErrors(cudaMemcpyAsync(d_X_cv, traits.X_cv(), size, cudaMemcpyHostToDevice,_stream1));

		// load the ycv matrix on the GPU:
		size = sizeof(value_type) * traits.y_cv_size();
		checkCudaErrors(cudaHostRegister(traits.y_cv(), size,cudaHostRegisterDefault)); // register the memory as pinned memory.
		checkCudaErrors(cudaMalloc(&d_y_cv, size));
		checkCudaErrors(cudaMemcpyAsync(d_y_cv, traits.y_cv(), size, cudaMemcpyHostToDevice,_stream1));
	}

	// Load the parameters (weights) on the GPU:
	// params is the vector used for the evaluation of the cost function.
	size = sizeof(value_type) * np;
	d_params = NULL;
	checkCudaErrors(cudaMalloc(&d_params, size));
	checkCudaErrors(cudaMemsetAsync(d_params,0,size,_stream1));

	// velocity vector used to store the NAG velocity for each cycle:
	d_vel = NULL;
	checkCudaErrors(cudaMalloc(&d_vel, size));
	checkCudaErrors(cudaMemsetAsync(d_vel,0,size,_stream1));

	d_vel_bak = NULL;
	checkCudaErrors(cudaMalloc(&d_vel_bak, size));

	// Theta is the array containing the computed network weights at each cycle:
	d_theta = NULL;
	checkCudaErrors(cudaHostRegister(traits.params(), size,cudaHostRegisterDefault)); // register the memory as pinned memory.
	checkCudaErrors(cudaMalloc(&d_theta, size));
	checkCudaErrors(cudaMemcpyAsync(d_theta, traits.params(), size, cudaMemcpyHostToDevice,_stream1));

	d_theta_bak = NULL;
	checkCudaErrors(cudaMalloc(&d_theta_bak, size));


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

	if(d_X_cv) {
		checkCudaErrors(cudaHostUnregister(_traits.X_cv())); 
		checkCudaErrors(cudaFree(d_X_cv));	
	}

	if(d_y_cv) {
		checkCudaErrors(cudaHostUnregister(_traits.y_cv())); 
		checkCudaErrors(cudaFree(d_y_cv));
	}

	// free GPU buffers:
	checkCudaErrors(cudaFree(d_X_train));	
	checkCudaErrors(cudaFree(d_y_train));	
	checkCudaErrors(cudaFree(d_params));	
	checkCudaErrors(cudaFree(d_theta));	
	checkCudaErrors(cudaFree(d_vel));	
	checkCudaErrors(cudaFree(d_regw));	
	checkCudaErrors(cudaFree(d_grads));	
	checkCudaErrors(cudaFree(d_deltas));	
	checkCudaErrors(cudaFree(d_inputs));	

	// destroy the processing stream:
	checkCudaErrors(cudaStreamDestroy(_stream1));
}

template<typename T>
T WindowedMean<T>::push(T val) {
	_stack.push_back(val);
	_totalValue += val;

	if(_stack.size() > _maxSize) {
		// remove the initial value
		T pval = _stack.front();
		_stack.pop_front();
		_totalValue -= pval;
	}

	return _totalValue/_stack.size();
}


void GradientDescentClass::run()
{
	int iter=0;
	value_type* current_cost = NULL;

	value_type* X_train_ptr = d_X_train;
	value_type* y_train_ptr = d_y_train;

	unsigned int mbOffset = 0;

	// Check if we should use all samples or not:
	unsigned int ns = _nsamples;
	if(_miniBatchSize>0) {
		ns = _miniBatchSize;
		logDEBUG("Using mini batch size: "<<_miniBatchSize);
	}

	bool earlyStopping = false;
	if(_validationWindowSize>0) {
		logDEBUG("Using early stopping with window size: "<<_validationWindowSize)
		earlyStopping = true;
	}
	WindowedMean<value_type> mean(_validationWindowSize);
	value_type cur_mean = 0.0;
	value_type bestCvCost = std::numeric_limits<value_type>::max();


	// number of cycles after which to evaluation is performed on the cross validation set
	// when using early stopping:
	unsigned int evalFrequency = 128;
	
	// current number of invalid value for Jcv when using early stopping:
	unsigned int invalid_count = 0;
	unsigned int max_invalid_count = 5;


	// Run the iteration loop:
	while(_maxiter<0 || iter<_maxiter) {
		// logDEBUG("Performing iteration "<<iter<<"...");

		// Here we apply the Nesterov Accelerated gradient (NAG) method described in 
		// "On the importance of initialization and momentum in deep learning.pdf" [1]
		// so we compute the speed:
		// v(t+1) = mu * v(t) - epsilon * DeltaF(theta(t) + mu * v(t))
		// then theta(t+1) = theta(t)+v(t+1)
		// Where t is the current iteration number (or current optimization time).
		// fist we need to do the partial theta update:
		// d_params = d_theta + mu * v(t);

		// 1. We need to compute the desired value of mu for this cycle:
		// formula from [1]
		_mu = (value_type)std::min(1.0 - pow(2.0, -1.0 - log(floor((double)iter/250.0)+1)/LOG2), _mumax);

		// 2. prepare the parameter vector:
		mix_vectors_device(d_params, d_theta, d_vel, 1.0, _mu, _np, _stream1);

		// 3. Once we have the parameter vector, we compute the gradient at that location:
		gd_errfunc_device(_nl, _np, _lsizes, ns, d_params, 
			X_train_ptr, y_train_ptr, _lambda, current_cost, d_grads, d_deltas, d_inputs, d_regw, _bias, _stream1);

		// logDEBUG("Performing iteration "<<iter<<", Jtrain="<<current_cost);

		// 4. With the gradient we update the velocity vector:
		mix_vectors_device(d_vel, d_vel, d_grads, _mu, -_epsilon, _np, _stream1);

		// 5. Now that we have the new velocity, we can compute the new value for the theta vector:
		mix_vectors_device(d_theta, d_theta, d_vel, 1.0, 1.0, _np, _stream1);

		if(_miniBatchSize>0) {
			// we should move to the next mini batch lot.
			mbOffset += _miniBatchSize;
			// logDEBUG("Using mini-batch offset: "<<mbOffset);

			// check if we have enough samples left or if we should reset to the beginning:
			if((mbOffset+_miniBatchSize)>_nsamples)
				mbOffset = 0;

			// update the pointers:
			X_train_ptr = d_X_train + _lsizes[0]*mbOffset;
			y_train_ptr = d_y_train + _lsizes[_nt]*mbOffset;
		}

		if(earlyStopping && (iter%evalFrequency == 0)) {
			// perform evaluation of the current theta value:
			// logDEBUG("Performing cv evaluation at iteration "<<iter<<"...");
			value_type J = computeCvCost();
			if(J<bestCvCost) {
				logDEBUG("Updating best Cv cost to "<<J<<" at iteration "<<iter);
				// Here we need to save the best cv cost achieved so far and also
				// save all the relevant parameters in case we need to restart from this point:
				bestCvCost = J;
				saveState(iter,mean);
			}
			// else {
			// 	logDEBUG("Discarding worst cv cost: "<<J)
			// }

			// push the new cost on the window mean:
			value_type new_mean = mean.push(J);
			// logDEBUG("New mean value is: "<<new_mean);

			// if we have enough cycles already then check if something is going wrong
			// with the cv cost:
			if(mean.size()==_validationWindowSize) {
				value_type dec = (cur_mean-new_mean)/cur_mean;
				if(dec<_minCvCostDec) {
					logDEBUG("Invalid mean cv cost decrease ratio of: "<<dec); //new_mean<<">"<<cur_mean);
					// We count this as an error:
					invalid_count++;
					if(invalid_count>max_invalid_count) {
						iter = restoreState(mean);
						// logDEBUG("Max number of invalid Jcv count reached. Resetting the latest best state from iteration "<<iter);
						
						if(evalFrequency>1) {
							// we reduce the evaluation frequency (assuming it is a power of 2)
							evalFrequency /= 2;
							invalid_count = 0;
							logDEBUG("Resetting from iteration "<<iter<<" with eval frequency of "<<evalFrequency)
						}
						else {
							logDEBUG("Early stopping training with cv cost "<< bestCvCost << " from iteration "<<iter);
							break;
						}
					}
				}
				else {
					// logDEBUG("Current cv cost is: "<<J);

					// Reset the invalid count otherwise:
					// logDEBUG("Resetting invalid count.");
					invalid_count = 0;
				}
			}

			cur_mean = mean.getMean();
		}

		// Finally move to the next cycle:
		iter++;
	}

	downloadParameters();
}

void GradientDescentClass::downloadParameters()
{
	// Download the parameters from the theta buffer on the GPU:
	checkCudaErrors(cudaMemcpy(_traits.params(), d_theta, _np*sizeof(value_type), cudaMemcpyDeviceToHost));
}

GradientDescentClass::value_type GradientDescentClass::computeTrainCost()
{
	value_type J = 0.0;
	value_type* grads = nullptr;

	// compute the cost at d_theta location, on complete training set, and not accounting for regularization:
	gd_errfunc_device(_nl, _np, _lsizes, _nsamples, d_theta, d_X_train, d_y_train, 0.0, &J, grads, d_deltas, d_inputs, d_regw, _bias, _stream1);
	return J;
}

GradientDescentClass::value_type GradientDescentClass::computeCvCost()
{
	value_type J = 0.0;
	value_type* grads = nullptr;

	// compute the cost at d_theta location, on complete training set, and not accounting for regularization:
	gd_errfunc_device(_nl, _np, _lsizes, _nsamples_cv, d_theta, d_X_cv, d_y_cv, 0.0, &J, grads, d_deltas, d_inputs, d_regw, _bias, _stream1);
	return J;
}

void GradientDescentClass::saveState(unsigned int iter, const WindowedMean<value_type>& mean)
{
	_bestIter = iter;
	_bestMean = mean;
	// logDEBUG("Saved mean is: "<<_bestMean.getMean());
	copy_vector_device(d_theta_bak, d_theta, _np);
	copy_vector_device(d_vel_bak, d_vel, _np);
}

unsigned int GradientDescentClass::restoreState(WindowedMean<value_type>& mean)
{
	copy_vector_device(d_theta, d_theta_bak, _np);
	copy_vector_device(d_vel, d_vel_bak, _np);
	mean = _bestMean;
	// logDEBUG("Restored mean is: "<<mean.getMean());
	return _bestIter;
}

