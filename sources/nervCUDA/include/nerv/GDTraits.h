#ifndef NERV_GDTRAITS_H_
#define NERV_GDTRAITS_H_

#include <nerv/BPTraits.h>
#include <vector>

namespace nerv
{

// forward declaration:
template<typename T>
class TrainingSet;

template<typename T>
struct GDTraits : public BPTraits<T>
{
public:
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

  T momentum;
  T epsilon;

  unsigned int miniBatchSize;
  unsigned int validationWindowSize;

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

    miniBatchSize = 0;
    validationWindowSize = 0;
  }
};

};

#endif
