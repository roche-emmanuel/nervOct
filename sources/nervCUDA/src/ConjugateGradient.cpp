#include <nervCUDA.h>

#include "ConjugateGradient.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

#define logDEBUG(msg) std::cout << std::setprecision(16) << msg << std::endl;

using namespace nerv;

ConjugateGradient::ConjugateGradient(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter, double* params)
{
	_nl = nl;
	_nsamples = nsamples;
	_nparams = nparams;
	_lsizes = lsizes;
	_X = X;
	_yy = yy;
	_lambda = lambda;
	_maxiter = maxiter;

	_params = params;

	// Copy the init params in the current parameters:
	memcpy(_params, init_params, sizeof(double)*nparams);

	// prepare the variables we will need:
	_f0 = 0.0;
	_f1 = 0.0;
	_f2 = 0.0;
	_df0 = new double[nparams];
	_df1 = new double[nparams];
	_df2 = new double[nparams];
	_s = new double[nparams];

	_params0 = new double[nparams];

	_realmin = std::numeric_limits<double>::min();
}

ConjugateGradient::~ConjugateGradient()
{

}

void ConjugateGradient::init()
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

void ConjugateGradient::evaluateCost(double zval)
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

void ConjugateGradient::saveParameters()
{
	memcpy(_params0,_params,sizeof(double)*_nparams);
	memcpy(_df0,_df1,sizeof(double)*_nparams);
	_f0 = _f1;
}

void ConjugateGradient::restoreParameters()
{
	memcpy(_params,_params0,sizeof(double)*_nparams);
	memcpy(_df1,_df0,sizeof(double)*_nparams);	
	_f1 = _f0;
}

void ConjugateGradient::updateS()
{	
	// update the value in _s using _df1 and _df2:
	// Then we also swap the values for df1 and df2;
	double tmp;
	double _d2 = 0.0;
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

	for(unsigned int i=0;i<_nparams; ++i) {
		_s[i] = coeff*_s[i] - _df2[i];
		tmp = _df1[i];
		_df1[i] = _df2[i];
		_df2[i] = tmp;
		_d2 += _df1[i]*_s[i];
	}
}

double ConjugateGradient::resetS()
{	
	double d = 0.0;
	for(unsigned int i=0;i<_nparams; ++i) {
		_s[i] = -_df1[i];
		d -= _s[i]*_s[i];
	}
	return d;
}

void ConjugateGradient::swapDfs()
{
	double* tmp = _df1;
	_df1 = _df2;
	_df2 = tmp;

	_d1 = resetS();
}

void ConjugateGradient::run()
{
	double RHO = 0.01;                           				// a bunch of constants for line searches
	double SIG = 0.5;  						// RHO and SIG are the constants in the Wolfe-Powell conditions
	double INT = 0.1;    			 // don't reevaluate within 0.1 of the limit of the current bracket
	double EXT = 3.0;                    			 // extrapolate maximum 3 times the current bracket
	int MAX = 20;                         			   // max 20 function evaluations per line search
	double RATIO = 100.0;                                     					 // maximum allowed slope ratio

	unsigned int i = 0;                                            // zero the run length counter
	bool ls_failed = false;                             // no previous line search has failed

	init();

	logDEBUG("cg: Value of f1: "<<_f1);

	double red = 1.0;
	double z1 = red/(1.0-_d1);

	double f3, d3, z2, z3, limit;
	int M;
	bool success;

	while(i<_maxiter) {
		i++;   // count iteration.

		saveParameters();
#if 1
		evaluateCost(z1);

		f3 = _f1;
		d3 = _d1;
		z3 = -z1;
		M = MAX;
		success = false;
		limit = -1;

		while(true) {
			while( ((_f2 > _f1+z1*RHO*_d1) || (_d2 > -SIG*_d1)) && M>0 ) {
				limit = z1;
				if(_f2 > _f1) {
					z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+_f2-f3);
				}
				else {
					double A = 6*(_f2-f3)/z3+3*(_d2+d3);
					double B = 3*(f3-_f2)-z3*(d3+2*_d2);
					z2 = (sqrt(B*B-A*_d2*z3*z3)-B)/A;
				}
				if(isnan(z2) || isinf(z2)) {
					z2 = z3/2.0; 					// if we had a numerical problem then bisect
				}
				z2 = std::max(std::min(z2, INT*z3),(1.0-INT)*z3);
				z1 = z1 + z2;
				evaluateCost(z2);
				M--;
				z3 = z3-z2;
			}

			if((_f2 > _f1+z1*RHO*_d1) || (_d2 > -SIG*_d1)) {
				break;
			}
			else if(_d2 > SIG*_d1) {
				success = true;
				break;
			}
			else if(M==0) {
				break;
			}

			double A = 6*(_f2-f3)/z3+3*(_d2+d3);
			double B = 3*(f3-_f2)-z3*(d3+2*_d2);
			z2 = -_d2*z3*z3/(B+sqrt(B*B-A*_d2*z3*z3));
			
			if(isnan(z2) || isinf(z2) || z2 < 0.0) {
				if(limit<-0.5) {
					z2 = z1 * (EXT-1);
				}
				else {
					z2 = (limit-z1)/2;
				}
			}
			else if((limit > -0.5) && ((z2+z1) > limit)) {
				z2 = (limit-z1)/2;
			}
			else if((limit < -0.5) && ((z2+z1) > z1*EXT)) {
				z2 = z1*(EXT-1.0);
			}
			else if(z2 < -z3*INT) {
				z2 = -z3*INT;
			}
			else if((limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))) {
				z2 = (limit-z1)*(1.0-INT);
			}

			f3 = _f2;
			d3 = _d2;
			z3 = -z2;
			z1 = z1 + z2;
			evaluateCost(z2);
			M--;			
		}

		if(success) {
			_f1 = _f2;
			logDEBUG("Iteration "<<i<<" | Cost: "<<_f1);
			updateS();
			if(_d2>0) {
				_d2 = resetS();
			}
			z1 = z1 * std::min(RATIO, _d1/(_d2-_realmin));
			_d1 = _d2;
			ls_failed = false;
		}
		else {
			restoreParameters();
			if(ls_failed || i > _maxiter) {
				break;
			}
			swapDfs();
			z1 = 1/(1-_d1);
			ls_failed = true;
		}
#endif
	}
}
