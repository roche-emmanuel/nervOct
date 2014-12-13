#include <nervCUDA.h>
#include <nerv_kernels.h>

#include "ConjugateGradientGPU.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace nerv;

ConjugateGradientGPU::ConjugateGradientGPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter, double* params) : ConjugateGradient(nl, nsamples, nparams, lsizes, lambda, maxiter, params)
{
	size_t size;

	unsigned int nt = nl-1;

	// checkCudaErrors(cudaDeviceSynchronize());

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

	// prepare regularization weigths:
	double* h_regw = new double[size];
	memset(h_regw,0,size);

	// prepare the regularization correction:
	double* rptr = h_regw;

	for(unsigned int i=0; i<nt;++i) {
		unsigned int nrows = lsizes[i+1];
		unsigned int ncolT = lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.

		rptr += nrows;
		unsigned int count = nrows*ncolT;

		for(unsigned int j=0;j<count;++j) {
			(*rptr++) = 1.0;
		}
	}

	// Prepare the reg weights for this network:
	d_regw = NULL;
	checkCudaErrors(cudaMalloc(&d_regw, size));
	checkCudaErrors(cudaMemcpy(d_regw, h_regw, size, cudaMemcpyHostToDevice));
	delete [] h_regw;

	// Also prepare the backup array for the parameters:
	d_params0 = NULL;
	checkCudaErrors(cudaMalloc(&d_params0, size));
	checkCudaErrors(cudaMemset(d_params0, 0, size));

	// for the cost computation we will also need the grads and delta arrays:
	// Also allocation the gradient array, with the same number of elements:
	d_grads = NULL;
	checkCudaErrors(cudaMalloc(&d_grads, size));
	checkCudaErrors(cudaMemset(d_grads,0,size));

	// we also need to prepare the _df0/_df1/df2 and _s arrays on GPU:
	d_df0 = NULL;
	checkCudaErrors(cudaMalloc(&d_df0, size));
	checkCudaErrors(cudaMemset(d_df0,0,size));
	d_df1 = NULL;
	checkCudaErrors(cudaMalloc(&d_df1, size));
	checkCudaErrors(cudaMemset(d_df1,0,size));
	d_df2 = NULL;
	checkCudaErrors(cudaMalloc(&d_df2, size));
	checkCudaErrors(cudaMemset(d_df2,0,size));
	d_s = NULL;
	checkCudaErrors(cudaMalloc(&d_s, size));
	checkCudaErrors(cudaMemset(d_s,0,size));
	d_redtmp = NULL;
	checkCudaErrors(cudaMalloc(&d_redtmp, size));
	checkCudaErrors(cudaMemset(d_redtmp,0,size));

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
}

ConjugateGradientGPU::~ConjugateGradientGPU()
{
	checkCudaErrors(cudaFree(d_yy));	
	checkCudaErrors(cudaFree(d_X));	
	checkCudaErrors(cudaFree(d_params));	
	checkCudaErrors(cudaFree(d_regw));	
	checkCudaErrors(cudaFree(d_params0));	
	checkCudaErrors(cudaFree(d_grads));	
	checkCudaErrors(cudaFree(d_deltas));	
	checkCudaErrors(cudaFree(d_inputs));	
	checkCudaErrors(cudaFree(d_df0));	
	checkCudaErrors(cudaFree(d_df1));	
	checkCudaErrors(cudaFree(d_df2));	
	checkCudaErrors(cudaFree(d_s));	
	checkCudaErrors(cudaFree(d_redtmp));	
}

void ConjugateGradientGPU::init()
{
	// evaluate the cost in _f1 and _df1 and assign value if _s.
	// here we need to compute the regularization from the current parameters:
	costFunc_device(_nl, _nparams, _lsizes, _nsamples, d_params, d_X, d_yy, _lambda, _f1, d_df1, d_deltas, d_inputs, d_regw);

	// compute d1 as the dot product of s by s after reseting s:
	_d1 = resetS();
}

void ConjugateGradientGPU::retrieveParameters()
{
	checkCudaErrors(cudaMemcpy(_params, d_params, sizeof(double)*_nparams, cudaMemcpyDeviceToHost));
}

void ConjugateGradientGPU::evaluateCost(double zval)
{
	// move the current parameters to X = X + zval * s:
	mix_vectors_device(d_params,d_params,d_s,1.0,zval,_nparams);

	// Evaluate cost at that point and store in f2 and df2:
	costFunc_device(_nl, _nparams, _lsizes, _nsamples, d_params, d_X, d_yy, _lambda, _f2, d_df2, d_deltas, d_inputs, d_regw);

	// compute the value _d2:
	_d2 = compute_dot_device(d_df2,d_s,d_redtmp,_nparams);
}

void ConjugateGradientGPU::saveParameters()
{
	copy_vector_device(d_params0, d_params, _nparams);
	copy_vector_device(d_df0, d_df1, _nparams);
	_f0 = _f1;
}

void ConjugateGradientGPU::restoreParameters()
{
	copy_vector_device(d_params, d_params0, _nparams);
	copy_vector_device(d_df1, d_df0, _nparams);
	_f1 = _f0;
}

void ConjugateGradientGPU::updateS()
{	
	// update the value in _s using _df1 and _df2:
	// Then we also swap the values for df1 and df2;
	
	double df22 = compute_length2_device(d_df2,d_redtmp,_nparams);
	double df11 = compute_length2_device(d_df1,d_redtmp,_nparams);
	double df12 = compute_dot_device(d_df1,d_df2,d_redtmp,_nparams);
	double coeff = (df22 - df12)/df11;

	mix_vectors_device(d_s,d_s,d_df2,coeff,-1.0,_nparams);
	swapDfs();
	_d2 = compute_dot_device(d_df1,d_s,d_redtmp,_nparams);
}

double ConjugateGradientGPU::resetS()
{
	copy_vector_device(d_s, d_df1, _nparams, true);
	return -compute_length2_device(d_s,d_redtmp,_nparams);
}

void ConjugateGradientGPU::swapDfs()
{
	double* tmp = d_df1;
	d_df1 = d_df2;
	d_df2 = tmp;
}
