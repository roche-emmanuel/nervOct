#include <nervCUDA.h>

#include <sgtcore.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

using namespace nerv;

GradientDescentClass::Traits::Traits() 
	: _nsamples(0),
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

	// unsigned int nl, unsigned int nsamples, unsigned int nparams, 
 //        unsigned int* lsizes, double lambda, unsigned int maxiter, double* params

	// _nl = nl;
	// _nsamples = nsamples;
	// _nparams = nparams;
	// _lsizes = lsizes;
	// _lambda = lambda;
	// _maxiter = maxiter;

	// _params = params;

	// Copy the init params in the current parameters:
	// memcpy(_params, init_params, sizeof(double)*nparams);
}

GradientDescentClass::~GradientDescentClass()
{

}

void GradientDescentClass::run()
{
	// Start from initial weights,
	// then compute the gradient.
	// update the weights with a learning rate.
	
}
