#include <nervCUDA.h>

#include "ConjugateGradientCPU.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

#define logDEBUG(msg) std::cout << std::setprecision(16) << msg << std::endl;

using namespace nerv;

ConjugateGradientCPU::ConjugateGradientCPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter, double* params) : ConjugateGradient(nl, nsamples, nparams, lsizes, lambda, maxiter, params)
{
	_X = X;
	_yy = yy;

	// prepare the variables we will need:
	_df0 = new double[nparams];
	_df1 = new double[nparams];
	_df2 = new double[nparams];
	_s = new double[nparams];

	_params0 = new double[nparams];

	// Copy the init params in the current parameters:
	memcpy(_params, init_params, sizeof(double)*nparams);
}

ConjugateGradientCPU::~ConjugateGradientCPU()
{
	delete [] _df0;
	delete [] _df1;
	delete [] _df2;
	delete [] _s;
	delete [] _params0;
}

void ConjugateGradientCPU::init()
{
	// evaluate the cost in _f1 and _df1 and assign value if _s.
	costFunc(_nl, _lsizes, _nsamples, _params, _X, _yy, _lambda, _f1, _df1, NULL, NULL);

	// copy the values from df1 to s:
	// memcpy(_s,_df1,sizeof(double)*_nparams);

	// compute d1 as the dot product of s by s:
	_d1 = 0.0;
	double* sptr = _s;
	double* fptr = _df1;
	double val;
	for(unsigned int i=0;i<_nparams;++i) {
	
		val = (*fptr++);
		(*sptr++) = -val; // s = -df1
		_d1 -= val*val;
	}
}

void ConjugateGradientCPU::evaluateCost(double zval)
{
	// move the current parameters to X = X + zval * s:
	for(unsigned int i=0;i<_nparams;++i) {
		_params[i] += zval * _s[i];
	}

	// Evaluate cost at that point and store in f2 and df2:
	costFunc(_nl, _lsizes, _nsamples, _params, _X, _yy, _lambda, _f2, _df2, NULL, NULL);

	// compute the value _d2:
	_d2 = 0.0;
	for(unsigned int i=0;i<_nparams;++i) {
		_d2 += _df2[i]*_s[i];
	}
}

void ConjugateGradientCPU::saveParameters()
{
	memcpy(_params0,_params,sizeof(double)*_nparams);
	memcpy(_df0,_df1,sizeof(double)*_nparams);
	_f0 = _f1;
}

void ConjugateGradientCPU::restoreParameters()
{
	memcpy(_params,_params0,sizeof(double)*_nparams);
	memcpy(_df1,_df0,sizeof(double)*_nparams);	
	_f1 = _f0;
}

void ConjugateGradientCPU::updateS()
{	
	// update the value in _s using _df1 and _df2:
	// Then we also swap the values for df1 and df2;
	double tmp;
	
	// Compute the coeff for the update of s:
	double df22 = 0.0;
	double df12 = 0.0;
	double df11 = 0.0;
	for(unsigned int i=0;i<_nparams; ++i) {
		df22 += _df2[i]*_df2[i];
		df12 += _df1[i]*_df2[i];
		df11 += _df1[i]*_df1[i];
	}

	double coeff = (df22 - df12)/df11;
	_d2 = 0.0;

	for(unsigned int i=0;i<_nparams; ++i) {
		_s[i] = coeff*_s[i] - _df2[i];
		tmp = _df1[i];
		_df1[i] = _df2[i];
		_df2[i] = tmp;
		_d2 += _df1[i]*_s[i];
	}
}

double ConjugateGradientCPU::resetS()
{	
	double d = 0.0;
	for(unsigned int i=0;i<_nparams; ++i) {
		_s[i] = -_df1[i];
		d -= _s[i]*_s[i];
	}
	return d;
}

void ConjugateGradientCPU::swapDfs()
{
	double* tmp = _df1;
	_df1 = _df2;
	_df2 = tmp;

	_d1 = resetS();
}
