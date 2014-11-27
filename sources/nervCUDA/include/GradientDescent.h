
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

        /** Specify the layer sizes.*/
        Traits& lsizes(unsigned int* layer_sizes, unsigned int nl);

        /** Retrieve the layer sizes. */
        unsigned int* lsizes() const;

        /** Retrieve the number of layers.*/
        unsigned int nl() const;

        /** Set the number of samples.*/
        Traits& nsamples(unsigned int num_samples);

        /** Retrieve the number of samples.*/
        unsigned int nsamples() const;

        /** Set the training dataset.*/
        Traits& X_train(value_type* X, unsigned int size);

        /** Retrieve the training dataset.*/
        value_type* X_train() const;

        /** Retrieve the size of the training dataset.*/
        unsigned int X_train_size() const;

        /** Set the training labels.*/
        Traits& y_train(value_type* y, unsigned int size);

        /** Retrieve the training labels.*/
        value_type* y_train() const;

        /** Retrieve the size of the training labels.*/
        unsigned int y_train_size() const;

        /** Set the params array.*/
        Traits& params(value_type* p, unsigned int size);

        /** Retrieve the params array.*/
        value_type* params() const;

        /** Retrieve the number of parameters.*/
        unsigned int nparams() const;

    protected:
        unsigned int _nl;
        unsigned int _nsamples;

        unsigned int* _lsizes;

        value_type* _X_train;
        unsigned int _X_train_size;

        value_type* _y_train;
        unsigned int _y_train_size;

        value_type* _params;
        unsigned int _nparams;        
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

    unsigned int _nl; // number of layers
    unsigned int _nt; // number of theta matrices
    unsigned int _np; // number of parameters
    unsigned int _nsamples; // number of samples.
    unsigned int* _lsizes;

    // unsigned int _nsamples;
    // unsigned int _nparams;

    // double _lambda;
    // unsigned int _maxiter;

    // double* _params;
};

};

#endif
