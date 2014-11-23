#include <nervCUDA.h>

#include "ConjugateGradient.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

#define logDEBUG(msg) std::cout << std::setprecision(16) << msg << std::endl;

using namespace nerv;

ConjugateGradient::ConjugateGradient(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double lambda, unsigned int maxiter, double* params)
{
	_nl = nl;
	_nsamples = nsamples;
	_nparams = nparams;
	_lsizes = lsizes;
	_lambda = lambda;
	_maxiter = maxiter;

	_params = params;

	// Copy the init params in the current parameters:
	// memcpy(_params, init_params, sizeof(double)*nparams);

	// prepare the variables we will need:
	_f0 = 0.0;
	_f1 = 0.0;
	_f2 = 0.0;
	_d1 = 0.0;
	_d2 = 0.0;
	_realmin = std::numeric_limits<double>::min();
}

ConjugateGradient::~ConjugateGradient()
{

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

	// logDEBUG("cg: Value of f1: "<<_f1);

	double red = 1.0;
	double z1 = red/(1.0-_d1);

	double f3, d3, z2, z3, limit;
	int M;
	bool success;

	while(i<_maxiter) {
		i++;   // count iteration.

		saveParameters();

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
	}

	retrieveParameters();
}
