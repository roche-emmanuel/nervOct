
#ifndef NERV_GRADIENTDESCENT_H_
#define NERV_GRADIENTDESCENT_H_

#include <nervcuda.h>
#include <sgtcore.h>

#include <nerv/TrainingSet.h>

namespace nerv {

// Basic implementation of gradient decsent on GPU.
class NERVCUDA_EXPORT GradientDescentClass {
public:
    typedef GradientDescentValueType value_type;

    class NERVCUDA_EXPORT Traits {
    public:
        Traits();
        Traits(const TrainingSet<value_type>& tr);
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

        /** Set the maximum number of iterations that can be performed.*/
        Traits& maxiter(int num);

        /** Retrieve the maximum number of iteration. */
        int maxiter() const;

        /** Set the regularizatino parameter.*/
        Traits& lambda(value_type val);

        /** Retrieve regularization parameter.*/
        value_type lambda() const;

        /** Set the maximum momentum value.*/
        Traits& momentum(value_type mu);

        /** Retrieve momentum value.*/
        value_type momentum() const;

    protected:
        unsigned int _nl;
        unsigned int _nsamples;
        unsigned int _maxiter;

        unsigned int* _lsizes;

        value_type* _X_train;
        unsigned int _X_train_size;

        value_type* _y_train;
        unsigned int _y_train_size;

        value_type* _params;
        unsigned int _nparams;

        value_type _lambda;
        value_type _mu;
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
    int _maxiter; // max number of iterations.

    value_type _mumax; // maximum value of the momentum.
    value_type _mu; // current value of the momentum.

    value_type _lambda; // regularization parameter.
    value_type* _regw; // host regularization buffer.

    // GPU buffers:
    value_type* d_X_train;
    value_type* d_y_train;
    value_type* d_params; // weights buffer.
    value_type* d_theta; // weights buffer.
    value_type* d_vel; // weights evolution velocity buffer.
    value_type* d_grads;
    value_type* d_deltas;
    value_type* d_inputs;
    
    // buffers for cost function evaluation:
    value_type* d_regw;

    cudaStream_t _stream1; // main processing stream. 
};

};

#endif
