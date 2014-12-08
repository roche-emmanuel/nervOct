
#ifndef NERV_GRADIENTDESCENT_H_
#define NERV_GRADIENTDESCENT_H_

#include <nervcuda.h>
#include <sgtcore.h>

#include <nerv/TrainingSet.h>
#include <nerv/WindowedMean.h>
#include <nerv/BPTraits.h>

namespace nerv
{

// Basic implementation of gradient decsent on GPU.

template<typename T>
struct GDTraits : public BPTraits<T>
{
public:
  GDTraits();
  GDTraits(const TrainingSet<T> &tr);

  virtual ~GDTraits() {};

  unsigned int maxiter;
  unsigned int X_train_size;
  unsigned int y_train_size;
  unsigned int nparams;

  T momentum;

  T epsilon;
  unsigned int miniBatchSize;
  unsigned int validationWindowSize;

  unsigned int X_cv_size;
  unsigned int y_cv_size;

  void validate() const;

protected:
  void init();

};

template<typename T>
class GradientDescent
{
public:
  typedef T value_type;
  typedef GDTraits<value_type> Traits;

  // Constructor taking all the parameters needed for computation:
  GradientDescent(const GDTraits<value_type> &traits);

  void run();

  value_type computeTrainCost();
  value_type computeCvCost();

  void saveState(unsigned int iter, const WindowedMean<value_type> &mean);
  unsigned int restoreState(WindowedMean<value_type> &mean);

protected:
  GDTraits<value_type> _traits;
  BPDeviceTraits<value_type> _d_traits;

  unsigned int _nt; // number of theta matrices
  unsigned int _np; // number of parameters
  unsigned int *_lsizes;
  int _maxiter; // max number of iterations.

  unsigned int _bestIter; // backup for the best iteration number so far when using early stopping.
  WindowedMean<value_type> _bestMean; // best windowed mean stack

  value_type _mumax; // maximum value of the momentum.
  value_type _mu; // current value of the momentum.
  value_type _epsilon; // Learning rate value.
  value_type _minCvCostDec; // minimal valid mean cv cost decrease

  // GPU buffers:
  value_type *d_theta; // weights buffer.
  value_type *d_theta_bak; // weights buffer.
  
  value_type *d_vel; // weights evolution velocity buffer.
  value_type *d_vel_bak; // weights evolution velocity buffer.

  unsigned int _miniBatchSize; // size of the mini batch or 0 if full batch.

  unsigned int _validationWindowSize; // size of the windowed mean for the cross validation cost vector.
};

template class NERVCUDA_EXPORT GradientDescent<double>;
template struct NERVCUDA_EXPORT GDTraits<double>;
template class NERVCUDA_EXPORT GradientDescent<float>;
template struct NERVCUDA_EXPORT GDTraits<float>;

typedef GradientDescent<double> GDd;
typedef GradientDescent<float> GDf;

};


#endif
