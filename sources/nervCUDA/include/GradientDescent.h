
#ifndef NERV_GRADIENTDESCENT_H_
#define NERV_GRADIENTDESCENT_H_

#include <sgtcore.h>

namespace nerv {

// Basic implementation of gradient decsent on GPU.
class NERVCUDA_EXPORT GradientDescentClass {
public:
    typedef GradientDescentValueType value_type;

    class NERVCUDA_EXPORT Traits {
    public:
        Traits();
        Traits(const Traits& rhs);
        Traits& operator=(const Traits& rhs);

        virtual ~Traits();

        /** Specify the number of layers for this session. */
        Traits& nl(unsigned int num_layers);

        /** Retrieve the number of layers.*/
        unsigned int nl() const;

        /** Specify the layer sizes.*/
        Traits& lsizes(unsigned int* layer_sizes);

        /** Retrieve the layer sizes. */
        unsigned int* lsizes() const;

        /** Set the number of samples.*/
        Traits& nsamples(unsigned int num_samples);

        /** Retrieve the number of samples.*/
        unsigned int nsamples() const;

        /** Set the number of parameters (eg. weights).*/
        Traits& nparams(unsigned int num_params);

        /** Retrieve the number of parameters.*/
        unsigned int nparams() const;

        /** Set the training dataset.*/
        Traits& X_train(value_type* X);

        /** Retrieve the training dataset.*/
        value_type* X_train() const;

        /** Set the training labels.*/
        Traits& y_train(value_type* y);

        /** Retrieve the training labels.*/
        value_type* y_train() const;


    protected:
        unsigned int _nl;
        unsigned int _nsamples;
        unsigned int _nparams;

        unsigned int* _lsizes;
        value_type* _X_train;
        value_type* _y_train;
    };


public:
    // Constructor taking all the parameters needed for computation:
    GradientDescentClass(const Traits& traits);

// unsigned int nl, unsigned int nsamples, unsigned int nparams, 
//         unsigned int* lsizes, double* X, double* yy, double* init_params,
//          double lambda, unsigned int maxiter, double* params

    ~GradientDescentClass();

    void run();

protected:
    Traits _traits;

    // unsigned int _nl;
    // unsigned int _nsamples;
    // unsigned int _nparams;

    // unsigned int* _lsizes;
    // double _lambda;
    // unsigned int _maxiter;

    // double* _params;
};

};

#endif
