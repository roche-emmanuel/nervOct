#include <nervCUDA.h>

#include "ConjugateGradient.h"
#include <iostream>

#define logDEBUG(msg) std::cout << msg << std::endl;

using namespace nerv;

ConjugateGradient::ConjugateGradient(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter)
{

}

ConjugateGradient::~ConjugateGradient()
{

}