
#ifndef NERV_GRADIENTDESCENT_H_
#define NERV_GRADIENTDESCENT_H_

#include <nervcuda.h>
#include <sgtcore.h>

#include <nerv/TrainingSet.h>
#include <nerv/WindowedMean.h>

namespace nerv {

// Basic implementation of gradient decsent on GPU.

template<typename T>
class GradientDescent {
public:
    typedef T value_type;

    class Traits {
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

        /** Set the initial learning rate value.*/
        Traits& learningRate(value_type lr);

        /** Retrieve learning rate value.*/
        value_type learningRate() const;

        /** Set the minibatch size, 0 to use full batch.*/
        Traits& miniBatchSize(unsigned int size);

        /** Retrieve the mini batch size.*/
        unsigned int miniBatchSize() const;

        /** Set validation window mean size when using early stopping.*/
        Traits& validationWindowSize(unsigned int size);

        /** Retrieve the mini batch size.*/
        unsigned int validationWindowSize() const;

        /** Set the cross validation dataset.*/
        Traits& X_cv(value_type* X, unsigned int size);

        /** Retrieve the cross validation dataset.*/
        value_type* X_cv() const;

        /** Retrieve the size of the cross validation dataset.*/
        unsigned int X_cv_size() const;

        /** Set the cross validation labels.*/
        Traits& y_cv(value_type* y, unsigned int size);

        /** Retrieve the cross validation labels.*/
        value_type* y_cv() const;

        /** Retrieve the size of the cross validation labels.*/
        unsigned int y_cv_size() const;

        /** Set the bias value used in the network.*/
        Traits& bias(value_type b);

        /** Retrieve the bias value.*/
        value_type bias() const;

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

        value_type _epsilon;
        unsigned int _miniBatchSize;

        unsigned int _validationWindowSize;

        value_type* _X_cv;
        unsigned int _X_cv_size;

        value_type* _y_cv;
        unsigned int _y_cv_size;

        value_type _bias;
    };


public:
    // Constructor taking all the parameters needed for computation:
    GradientDescent(const Traits& traits);

// unsigned int nl, unsigned int nsamples, unsigned int nparams, 
//         unsigned int* lsizes, double* X, double* yy, double* init_params,
//          double lambda, unsigned int maxiter, double* params

    ~GradientDescent();

    void run();

    value_type computeTrainCost();
    value_type computeCvCost();

    void downloadParameters();

    void saveState(unsigned int iter, const WindowedMean<value_type>& mean);
    unsigned int restoreState(WindowedMean<value_type>& mean);

protected:
    Traits _traits;

    unsigned int _nl; // number of layers
    unsigned int _nt; // number of theta matrices
    unsigned int _np; // number of parameters
    unsigned int _nsamples; // number of samples.
    unsigned int _nsamples_cv; // number of samples in cv datasets.
    unsigned int* _lsizes;
    int _maxiter; // max number of iterations.

    unsigned int _bestIter; // backup for the best iteration number so far when using early stopping.
    WindowedMean<value_type> _bestMean; // best windowed mean stack

    value_type _mumax; // maximum value of the momentum.
    value_type _mu; // current value of the momentum.
    value_type _epsilon; // Learning rate value.
    value_type _minCvCostDec; // minimal valid mean cv cost decrease
    value_type _bias; // Bias value used for the neurons in the network.

    value_type _lambda; // regularization parameter.
    value_type* _regw; // host regularization buffer.

    // GPU buffers:
    value_type* d_X_train;
    value_type* d_y_train;
    value_type* d_X_cv;
    value_type* d_y_cv;
    value_type* d_params; // weights buffer.
    value_type* d_theta; // weights buffer.
    value_type* d_theta_bak; // weights buffer.
    value_type* d_vel; // weights evolution velocity buffer.
    value_type* d_vel_bak; // weights evolution velocity buffer.
    value_type* d_grads;
    value_type* d_deltas;
    value_type* d_inputs;
    
    // buffers for cost function evaluation:
    value_type* d_regw;

    cudaStream_t _stream1; // main processing stream. 

    unsigned int _miniBatchSize; // size of the mini batch or 0 if full batch.

    unsigned int _validationWindowSize; // size of the windowed mean for the cross validation cost vector.
};

template NERVCUDA_EXPORT class GradientDescent<double>;
template NERVCUDA_EXPORT class GradientDescent<double>::Traits;
template NERVCUDA_EXPORT class GradientDescent<float>;
template NERVCUDA_EXPORT class GradientDescent<float>::Traits;

};

#endif
