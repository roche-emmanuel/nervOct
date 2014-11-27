#include <nervCUDA.h>

#include <sgtcore.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

using namespace nerv;

GradientDescentClass::Traits::Traits() 
	: _nl(0), _nsamples(0), _nparams(0), 
	_lsizes(nullptr), _X_train(nullptr), _y_train(nullptr)
{
}

GradientDescentClass::Traits::~Traits()
{
}

GradientDescentClass::Traits& GradientDescentClass::Traits::nl(unsigned int num_layers)
{
	_nl = num_layers;
	return *this;
}

unsigned int GradientDescentClass::Traits::nl() const
{
	return _nl;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::lsizes(unsigned int* layer_sizes)
{
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

GradientDescentClass::Traits& GradientDescentClass::Traits::nparams(unsigned int num_params)
{
	_nparams = num_params;
	return *this;
}

unsigned int GradientDescentClass::Traits::nparams() const
{
	return _nparams;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::X_train(GradientDescentClass::value_type* X)
{
	_X_train = X;
	return *this;
}

GradientDescentClass::value_type* GradientDescentClass::Traits::X_train() const
{
	return _X_train;
}

GradientDescentClass::Traits& GradientDescentClass::Traits::y_train(GradientDescentClass::value_type* y)
{
	_y_train = y;
	return *this;
}

GradientDescentClass::value_type* GradientDescentClass::Traits::y_train() const
{
	return _y_train;
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

	return *this;
}

GradientDescentClass::GradientDescentClass(const Traits& traits)
{
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
