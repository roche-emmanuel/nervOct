
#ifndef NERV_GRADIENTDESCENT_H_
#define NERV_GRADIENTDESCENT_H_

#include <nervcuda.h>
#include <sgtcore.h>

#include <nerv/TrainingSet.h>
#include <nerv/WindowedMean.h>

namespace nerv
{

// Basic implementation of gradient decsent on GPU.

template<typename T>
class GDTraits
{
public:
  typedef T value_type;

public:
  GDTraits();
  GDTraits(const TrainingSet<value_type> &tr);
  GDTraits(const GDTraits &rhs)
  {
    this->operator=(rhs);
  }

  GDTraits &operator=(const GDTraits &rhs);

  virtual ~GDTraits() {};

  /** Specify the layer sizes.*/
  GDTraits &lsizes(unsigned int *layer_sizes, unsigned int nl)
  {
    _nl = nl;
    _lsizes = layer_sizes;
    return *this;
  }

  /** Retrieve the layer sizes. */
  unsigned int *lsizes() const
  {
    return _lsizes;
  }

  /** Retrieve the number of layers.*/
  unsigned int nl() const
  {
    return _nl;
  }

  /** Set the number of samples.*/
  GDTraits &nsamples(unsigned int num_samples)
  {
    _nsamples = num_samples;
    return *this;
  }

  /** Retrieve the number of samples.*/
  unsigned int nsamples() const
  {
    return _nsamples;
  }

  /** Set the training dataset.*/
  GDTraits &X_train(value_type *X, unsigned int size)
  {
    _X_train_size = size;
    _X_train = X;
    return *this;
  }

  /** Retrieve the training dataset.*/
  value_type *X_train() const
  {
    return _X_train;
  }

  /** Retrieve the size of the training dataset.*/
  unsigned int X_train_size() const
  {
    return _X_train_size;
  }

  /** Set the training labels.*/
  GDTraits &y_train(value_type *y, unsigned int size)
  {
    _y_train_size = size;
    _y_train = y;
    return *this;
  }

  /** Retrieve the training labels.*/
  value_type *y_train() const
  {
    return _y_train;
  }

  /** Retrieve the size of the training labels.*/
  unsigned int y_train_size() const
  {
    return _y_train_size;
  }

  /** Set the params array.*/
  GDTraits &params(value_type *p, unsigned int size)
  {
    _nparams = size;
    _params = p;
    return *this;
  }

  /** Retrieve the params array.*/
  value_type *params() const
  {
    return _params;
  }

  /** Retrieve the number of parameters.*/
  unsigned int nparams() const
  {
    return _nparams;
  }

  /** Set the maximum number of iterations that can be performed.*/
  GDTraits &maxiter(int num)
  {
    _maxiter = num;
    return *this;
  }

  /** Retrieve the maximum number of iteration. */
  int maxiter() const
  {
    return _maxiter;
  }

  /** Set the regularizatino parameter.*/
  GDTraits &lambda(value_type val)
  {
    _lambda = val;
    return *this;
  }

  /** Retrieve regularization parameter.*/
  value_type lambda() const
  {
    return _lambda;
  }

  /** Set the maximum momentum value.*/
  GDTraits &momentum(value_type mu)
  {
    _mu = mu;
    return *this;
  }

  /** Retrieve momentum value.*/
  value_type momentum() const
  {
    return _mu;
  }

  /** Set the initial learning rate value.*/
  GDTraits &learningRate(value_type lr)
  {
    _epsilon = lr;
    return *this;
  }

  /** Retrieve learning rate value.*/
  value_type learningRate() const
  {
    return _epsilon;
  }

  /** Set the minibatch size, 0 to use full batch.*/
  GDTraits &miniBatchSize(unsigned int size)
  {
    _miniBatchSize = size;
    return *this;
  }

  /** Retrieve the mini batch size.*/
  unsigned int miniBatchSize() const
  {
    return _miniBatchSize;
  }

  /** Set validation window mean size when using early stopping.*/
  GDTraits &validationWindowSize(unsigned int size)
  {
    _validationWindowSize = size;
    return *this;
  }

  /** Retrieve the mini batch size.*/
  unsigned int validationWindowSize() const
  {
    return _validationWindowSize;
  }

  /** Set the cross validation dataset.*/
  GDTraits &X_cv(value_type *X, unsigned int size)
  {
    _X_cv_size = size;
    _X_cv = X;
    return *this;
  }

  /** Retrieve the cross validation dataset.*/
  value_type *X_cv() const
  {
    return _X_cv;
  }

  /** Retrieve the size of the cross validation dataset.*/
  unsigned int X_cv_size() const
  {
    return _X_cv_size;
  }

  /** Set the cross validation labels.*/
  GDTraits &y_cv(value_type *y, unsigned int size)
  {
    _y_cv_size = size;
    _y_cv = y;
    return *this;
  }

  /** Retrieve the cross validation labels.*/
  value_type *y_cv() const
  {
    return _y_cv;
  }

  /** Retrieve the size of the cross validation labels.*/
  unsigned int y_cv_size() const
  {
    return _y_cv_size;
  }

  /** Set the bias value used in the network.*/
  GDTraits &bias(value_type b)
  {
    _bias = b;
    return *this;
  }

  /** Retrieve the bias value.*/
  value_type bias() const
  {
    return _bias;
  }

protected:
  /** Init all the member values to default values. */
  void init();

  unsigned int _nl;
  unsigned int _nsamples;
  unsigned int _maxiter;

  unsigned int *_lsizes;

  value_type *_X_train;
  unsigned int _X_train_size;

  value_type *_y_train;
  unsigned int _y_train_size;

  value_type *_params;
  unsigned int _nparams;

  value_type _lambda;
  value_type _mu;

  value_type _epsilon;
  unsigned int _miniBatchSize;

  unsigned int _validationWindowSize;

  value_type *_X_cv;
  unsigned int _X_cv_size;

  value_type *_y_cv;
  unsigned int _y_cv_size;

  value_type _bias;
};

template<typename T>
class GradientDescent
{
public:
  typedef T value_type;
  typedef GDTraits<value_type> Traits;

  // Constructor taking all the parameters needed for computation:
  GradientDescent(const GDTraits<value_type> &traits);

  // unsigned int nl, unsigned int nsamples, unsigned int nparams,
  //         unsigned int* lsizes, double* X, double* yy, double* init_params,
  //          double lambda, unsigned int maxiter, double* params

  ~GradientDescent();

  void run();

  value_type computeTrainCost();
  value_type computeCvCost();

  void downloadParameters();

  void saveState(unsigned int iter, const WindowedMean<value_type> &mean);
  unsigned int restoreState(WindowedMean<value_type> &mean);

protected:
  GDTraits<value_type> _traits;

  unsigned int _nl; // number of layers
  unsigned int _nt; // number of theta matrices
  unsigned int _np; // number of parameters
  unsigned int _nsamples; // number of samples.
  unsigned int _nsamples_cv; // number of samples in cv datasets.
  unsigned int *_lsizes;
  int _maxiter; // max number of iterations.

  unsigned int _bestIter; // backup for the best iteration number so far when using early stopping.
  WindowedMean<value_type> _bestMean; // best windowed mean stack

  value_type _mumax; // maximum value of the momentum.
  value_type _mu; // current value of the momentum.
  value_type _epsilon; // Learning rate value.
  value_type _minCvCostDec; // minimal valid mean cv cost decrease
  value_type _bias; // Bias value used for the neurons in the network.

  value_type _lambda; // regularization parameter.
  value_type *_regw; // host regularization buffer.

  // GPU buffers:
  value_type *d_X_train;
  value_type *d_y_train;
  value_type *d_X_cv;
  value_type *d_y_cv;
  value_type *d_params; // weights buffer.
  value_type *d_theta; // weights buffer.
  value_type *d_theta_bak; // weights buffer.
  value_type *d_vel; // weights evolution velocity buffer.
  value_type *d_vel_bak; // weights evolution velocity buffer.
  value_type *d_grads;
  value_type *d_deltas;
  value_type *d_inputs;

  // buffers for cost function evaluation:
  value_type *d_regw;

  cudaStream_t _stream1; // main processing stream.

  unsigned int _miniBatchSize; // size of the mini batch or 0 if full batch.

  unsigned int _validationWindowSize; // size of the windowed mean for the cross validation cost vector.
};

template class NERVCUDA_EXPORT GradientDescent<double>;
template class NERVCUDA_EXPORT GDTraits<double>;
template class NERVCUDA_EXPORT GradientDescent<float>;
template class NERVCUDA_EXPORT GDTraits<float>;

typedef GradientDescent<double> GDd;
typedef GradientDescent<float> GDf;

};


#endif
