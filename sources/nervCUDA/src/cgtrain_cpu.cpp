#include <nervCUDA.h>

#include <iostream>
#include "ConjugateGradient.h"

#define logDEBUG(msg) std::cout << msg << std::endl;

extern "C" {

void cgtrainCPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
    unsigned int* lsizes, double* X, double* yy, double* init_params, 
    double lambda, unsigned int maxiter, double* params)
{
  // we instanciate a need instance of the conjugate gradient class, and we run it:
  // logDEBUG("Creating conjugate gradient object...");
  nerv::ConjugateGradient cg(nl, nsamples, nparams, lsizes, X, yy, init_params, lambda, maxiter, params);

  // logDEBUG("Running conjugate gradient...");
  cg.run();
}

}
