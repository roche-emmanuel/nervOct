
#ifndef NERV_CONJUGATEGRADIENT_H_
#define NERV_CONJUGATEGRADIENT_H_

namespace nerv {

class ConjugateGradient {
public:
    // Constructor taking all the parameters needed for computation:
    ConjugateGradient(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
        unsigned int* lsizes, double lambda, unsigned int maxiter, double* params);

    ~ConjugateGradient();

    virtual void init() = 0;

    virtual void evaluateCost(double zval) = 0;

    void run();

    virtual void saveParameters() = 0;
    virtual void restoreParameters() = 0;

    virtual void updateS() = 0;
    virtual double resetS() = 0;

    virtual void swapDfs() = 0;

    // Method used to load the parameters in the params array if necessary.
    virtual void retrieveParameters() {}
    
protected:
    unsigned int _nl;
    unsigned int _nsamples;
    unsigned int _nparams;

    unsigned int* _lsizes;
    double _lambda;
    unsigned int _maxiter;

    double* _params;

    double _f0;
    double _f1;
    double _f2;

    double _d1;
    double _d2;
    double _realmin;
};

};

#endif
