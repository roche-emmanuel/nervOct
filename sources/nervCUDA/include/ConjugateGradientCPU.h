
#ifndef NERV_CONJUGATEGRADIENT_CPU_H_
#define NERV_CONJUGATEGRADIENT_CPU_H_

#include <ConjugateGradient.h>

namespace nerv {

class ConjugateGradientCPU : public ConjugateGradient {
public:
    // Constructor taking all the parameters needed for computation:
    ConjugateGradientCPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter, double* params);

    ~ConjugateGradientCPU();

    virtual void init();

    virtual void evaluateCost(double zval);

    virtual void saveParameters();
    virtual void restoreParameters();

    virtual void updateS();
    virtual double resetS();

    virtual void swapDfs();

protected:
    double* _X;
    double* _yy;


    double* _params0;
    double* _df0;
    double* _df1;
    double* _df2;
    double* _s;
};

};

#endif
