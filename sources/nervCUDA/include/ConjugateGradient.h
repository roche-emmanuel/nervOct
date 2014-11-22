
#ifndef NERV_CONJUGATEGRADIENT_H_
#define NERV_CONJUGATEGRADIENT_H_

namespace nerv {

class ConjugateGradient {
public:
    // Constructor taking all the parameters needed for computation:
    ConjugateGradient(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter);

    ~ConjugateGradient();

protected:
    unsigned int _nl;
    unsigned int _nsamples;
    unsigned int _nparams;

    unsigned int* _lsizes;
    double* _X;
    double* _yy;
    double* _params;

    double _lambda;
    unsigned int _maxiter;
};

};

#endif
