
#ifndef NERV_CUDA_H_
#define NERV_CUDA_H_

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
    #  if defined( NERVCUDA_LIBRARY_STATIC )
    #    define NERVCUDA_EXPORT
    #  elif defined( NERVCUDA_LIBRARY )
    #    define NERVCUDA_EXPORT   __declspec(dllexport)
    #  else
    #    define NERVCUDA_EXPORT   __declspec(dllimport)
    #  endif
#else
    #  define NERVCUDA_EXPORT
#endif

extern "C" {

void multiplyMatrices(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

void costFunc(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
    double* nn_params, double* X, double* yy, double lambda, double* activation, double* inputs);

void costFuncCPU(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* nn_params, double* X, double* yy, double lambda, double* activation, double* inputs);

};

#endif
