#ifndef NERV_GDTRAITS_H_
#define NERV_GDTRAITS_H_

#include <nerv/BPTraits.h>
#include <vector>

namespace nerv
{

enum GradientDescentCode {
  GD_SUCCESS,
  GD_EXCEPTION_OCCURED
};

// forward declaration:
template<typename T>
class TrainingSet;

template<typename T>
struct GDTraits : public BPTraits<T>
{
public:
  typedef void (* CvCostFunc)(T cost, unsigned int iter, void* data);

  GDTraits()
  {
    init();
  }

  GDTraits(const TrainingSet<T> &tr);

  virtual ~GDTraits() {};

  unsigned int X_train_size;
  unsigned int y_train_size;
  unsigned int X_cv_size;
  unsigned int y_cv_size;

  unsigned int maxiter;
  unsigned int nparams;

  unsigned int evalFrequency;

  T momentum;
  T epsilon;
  T learningDecay;
  T minCostDecrease;

  bool verbose;

  unsigned int miniBatchSize;
  unsigned int validationWindowSize;

  CvCostFunc cvCostCB;
  void* userdata;

  void validate() const;

protected:

  void init()
  {
    X_train_size = 0;
    y_train_size = 0;
    X_cv_size = 0;
    y_cv_size = 0;

    maxiter = 0;
    nparams = 0;

    momentum = 0.0;
    epsilon = 0.0;
    learningDecay = 1.0;

    minCostDecrease = (T)0.00001;

    verbose = false;

    miniBatchSize = 0;
    validationWindowSize = 0;

    cvCostCB = nullptr;
    userdata = nullptr;

    evalFrequency = 128;
  }
};

};

#endif
