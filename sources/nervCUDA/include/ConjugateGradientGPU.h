
#ifndef NERV_CONJUGATEGRADIENT_GPU_H_
#define NERV_CONJUGATEGRADIENT_GPU_H_

#include <ConjugateGradient.h>

namespace nerv {

class ConjugateGradientGPU : public ConjugateGradient {
public:
    // Constructor taking all the parameters needed for computation:
    ConjugateGradientGPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter, double* params);

    ~ConjugateGradientGPU();

    virtual void init();

    virtual void evaluateCost(double zval);

    virtual void saveParameters();
    virtual void restoreParameters();

    virtual void updateS();
    virtual double resetS();

    virtual void swapDfs();

protected:
    double* d_X;
    double* d_yy;
    double* d_grads;
    double* d_deltas;
    double* d_inputs;
    double* d_regw;
    
    double* d_params;
    double* d_params0;

    double* d_df0;
    double* d_df1;
    double* d_df2;
    double* d_s;
    double* d_redtmp;

    double* _params0;
    double* _s;    
};

};

#endif
