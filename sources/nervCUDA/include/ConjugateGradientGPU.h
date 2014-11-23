
#ifndef NERV_CONJUGATEGRADIENT_GPU_H_
#define NERV_CONJUGATEGRADIENT_GPU_H_

namespace nerv {

class ConjugateGradientGPU {
public:
    // Constructor taking all the parameters needed for computation:
    ConjugateGradientGPU(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double* X, double* yy, double* init_params, 
        double lambda, unsigned int maxiter, double* params);

    ~ConjugateGradientGPU();

    void init();

    void evaluateCost(double zval);

    void run();

    void saveParameters();
    void restoreParameters();

    void updateS();
    double resetS();

    void swapDfs();

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

    double* _params0;
    double _f0;
    double* _df0;
    double _f1;
    double* _df1;
    double _f2;
    double* _df2;
    double* _s;

    double _d1;

    double _d2;
    double _realmin;
};

};

#endif
