#include <nervCUDA.h>
#include <nerv_kernels.h>

#include "ConjugateGradientGPU.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

#ifdef logDEBUG
#undef logDEBUG
#endif

#define logDEBUG(msg) std::cout << std::setprecision(16) << msg << std::endl;

using namespace nerv;

ConjugateGradientGPU::ConjugateGradientGPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter, double* params) : ConjugateGradient(nl, nsamples, nparams, lsizes, lambda, maxiter, params)
{
	size_t size;

	unsigned int nt = nl-1;

	cudaDeviceSynchronize();

	// Load the X matrix on the GPU directly:
	size = sizeof(double) * nsamples * lsizes[0];
	d_X = NULL;
	checkCudaErrors(cudaMalloc(&d_X, size));
	checkCudaErrors(cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice));

	// load the yy matrix on the GPU:
	size = nsamples * lsizes[nt] * sizeof(double);
	d_yy = NULL;
	checkCudaErrors(cudaMalloc(&d_yy, size));
	checkCudaErrors(cudaMemcpy(d_yy, yy, size, cudaMemcpyHostToDevice));

	// Load the parameters (weights) on the GPU:
	size = nparams * sizeof(double);
	d_params = NULL;
	checkCudaErrors(cudaMalloc(&d_params, size));
	checkCudaErrors(cudaMemcpy(d_params, init_params, size, cudaMemcpyHostToDevice));

	// Also prepare the backup array for the parameters:
	// d_params0 = NULL;
	// checkCudaErrors(cudaMalloc(&d_params0, size));
	// checkCudaErrors(cudaMemset(d_params0, 0, size));

	// for the cost computation we will also need the grads and delta arrays:
	// Also allocation the gradient array, with the same number of elements:
	d_grads = NULL;
	checkCudaErrors(cudaMalloc(&d_grads, size));
	checkCudaErrors(cudaMemset(d_grads,0,size));

	// we also need to prepare the _df0/_df1/df2 and _s arrays on GPU:
	// d_df0 = NULL;
	// checkCudaErrors(cudaMalloc(&d_df0, size));
	// checkCudaErrors(cudaMemset(d_df0,0,size));
	// d_df1 = NULL;
	// checkCudaErrors(cudaMalloc(&d_df1, size));
	// checkCudaErrors(cudaMemset(d_df1,0,size));
	// d_df2 = NULL;
	// checkCudaErrors(cudaMalloc(&d_df2, size));
	// checkCudaErrors(cudaMemset(d_df2,0,size));
	// d_s = NULL;
	// checkCudaErrors(cudaMalloc(&d_s, size));
	// checkCudaErrors(cudaMemset(d_s,0,size));


	// Compute the total number of delta coefficients:
	unsigned int nd = 0;
	for(unsigned int i=1;i<nl;++i) {
		nd += lsizes[i]*nsamples;
	}

	size = nd*sizeof(double);
	d_deltas = NULL;
	checkCudaErrors(cudaMalloc(&d_deltas, size));
	checkCudaErrors(cudaMemset(d_deltas,0,size));

	// finally we also need the inputs array:
	unsigned int count = 0;
	for(unsigned int i=0;i<nt;++i) {
		count += lsizes[i+1];
	}

	size = nsamples * count * sizeof(double);
	d_inputs = NULL;
	checkCudaErrors(cudaMalloc(&d_inputs, size));
	checkCudaErrors(cudaMemset(d_inputs,0,size));

	_df0 = new double[nparams];
	_df1 = new double[nparams];
	memset(_df1,0,sizeof(double)*nparams);
	_df2 = new double[nparams];
	_s = new double[nparams];
	_params0 = new double[nparams];

	// Copy the init params in the current parameters:
	memcpy(_params, init_params, sizeof(double)*nparams);
}

ConjugateGradientGPU::~ConjugateGradientGPU()
{
	checkCudaErrors(cudaFree(d_yy));	
	checkCudaErrors(cudaFree(d_X));	
	checkCudaErrors(cudaFree(d_params));	
	// checkCudaErrors(cudaFree(d_params0));	
	checkCudaErrors(cudaFree(d_grads));	
	checkCudaErrors(cudaFree(d_deltas));	
	checkCudaErrors(cudaFree(d_inputs));	
	// checkCudaErrors(cudaFree(d_df0));	
	// checkCudaErrors(cudaFree(d_df1));	
	// checkCudaErrors(cudaFree(d_df2));	
	// checkCudaErrors(cudaFree(d_s));	
	delete [] _df0;
	delete [] _df1;
	delete [] _df2;
	delete [] _s;
	delete [] _params0;
}

double ConjugateGradientGPU::computeRegCorrection()
{
	double reg_correction = 0.0;
	double* tptr = _params;
	double rval;
	unsigned int nt =_nl-1;

	for(unsigned int i=0; i<nt;++i) {
		unsigned int nrows = _lsizes[i+1];
		unsigned int ncolT = _lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.

		for(unsigned int j=0;j<nrows;++j) {
			rval = (*tptr++);
			reg_correction += rval*rval;
		}
		tptr += nrows*ncolT;
	}

 	return reg_correction;
}

void ConjugateGradientGPU::init()
{
	// evaluate the cost in _f1 and _df1 and assign value if _s.
	// here we need to compute the regularization from the current parameters:
	double reg_correction = computeRegCorrection();

	costFunc_device(_nl, _nparams, _lsizes, _nsamples, d_params, d_X, d_yy, _lambda, _f1, d_grads, d_deltas, d_inputs, reg_correction);

	// Here we should also read back the gradient values:
	checkCudaErrors(cudaMemcpy(_df1, d_grads, sizeof(double)*_nparams, cudaMemcpyDeviceToHost));

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

void ConjugateGradientGPU::evaluateCost(double zval)
{
	// move the current parameters to X = X + zval * s:
	for(unsigned int i=0;i<_nparams;++i) {
		_params[i] += zval * _s[i];
	}

	// update the params:
	checkCudaErrors(cudaMemcpy(d_params, _params, sizeof(double)*_nparams, cudaMemcpyHostToDevice));

	double reg_correction = computeRegCorrection();

	// Evaluate cost at that point and store in f2 and df2:
	costFunc_device(_nl, _nparams, _lsizes, _nsamples, d_params, d_X, d_yy, _lambda, _f2, d_grads, d_deltas, d_inputs, reg_correction);

	// Here we should also read back the gradient values:
	checkCudaErrors(cudaMemcpy(_df2, d_grads, sizeof(double)*_nparams, cudaMemcpyDeviceToHost));

	// compute the value _d2:
	_d2 = 0.0;
	for(unsigned int i=0;i<_nparams;++i) {
		_d2 += _df2[i]*_s[i];
	}
}

void ConjugateGradientGPU::saveParameters()
{
	memcpy(_params0,_params,sizeof(double)*_nparams);
	memcpy(_df0,_df1,sizeof(double)*_nparams);
	_f0 = _f1;
}

void ConjugateGradientGPU::restoreParameters()
{
	memcpy(_params,_params0,sizeof(double)*_nparams);
	checkCudaErrors(cudaMemcpy(d_params, _params, sizeof(double)*_nparams, cudaMemcpyHostToDevice));
	memcpy(_df1,_df0,sizeof(double)*_nparams);	
	_f1 = _f0;
}

void ConjugateGradientGPU::updateS()
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

double ConjugateGradientGPU::resetS()
{	
	double d = 0.0;
	for(unsigned int i=0;i<_nparams; ++i) {
		_s[i] = -_df1[i];
		d -= _s[i]*_s[i];
	}
	return d;
}

void ConjugateGradientGPU::swapDfs()
{
	double* tmp = _df1;
	_df1 = _df2;
	_df2 = tmp;

	_d1 = resetS();
}
