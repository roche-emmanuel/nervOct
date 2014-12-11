#ifndef NERV_BPTRAITS_H_
#define NERV_BPTRAITS_H_

#include <vector>

namespace nerv
{

template <typename T>
struct BPTraitsBase
{
  BPTraitsBase()
    : bias(1.0), lambda(0.0),
      X(nullptr), yy(nullptr), params(nullptr),
      inputs(nullptr), deltas(nullptr), grads(nullptr) {}

  virtual ~BPTraitsBase() {}

  T bias;
  T lambda;

  T *X;
  T *yy;
  T *params;
  T *inputs;
  T *deltas;
  T *grads;
};

// template<typename T>
// T* newArray(const std::vector<T>& vec) {
//   size_t n = vec.size();
//   T* arr = new T[n];
//   memcpy(arr,&vec[0],n*sizeof(T));
//   return arr;
// }

// template<typename T>
// void deleteArray(T* arr) {
//   delete [] arr;
// }

template<typename T>
struct BPTraits : public BPTraitsBase<T>
{
  BPTraits()
    :  wmults(nullptr), cost(0.0), compute_cost(false), compute_grads(true),
       nsamples_train(0), nl(0), lsizes(nullptr), X_cv(nullptr), y_cv(nullptr),
       nsamples_cv(0), hx(nullptr), dropouts(nullptr), debug(false) {};

  virtual ~BPTraits() {}

  unsigned int nsamples_train;
  unsigned int nsamples_cv;

  unsigned int nl;
  unsigned int *lsizes;

  bool compute_cost;
  bool compute_grads;
    
  bool debug;

  T cost;
  T *wmults;
  T* dropouts;

  T *X_cv;
  T *y_cv;

  T* hx; // to store result of prediction

  // Compute the number of parameters:
  unsigned int np() const
  {
    unsigned int res = 0;
    unsigned int nt = nl - 1;
    for (unsigned int i = 0; i < nt; ++i)
    {
      res += lsizes[i + 1] * (lsizes[i] + 1);
    }
    return res;
  }

  // Compute the number of deltas (or inputs)
  unsigned int nd() const
  {
    unsigned int res = 0;
    for (unsigned int i = 1; i < nl; ++i)
    {
      res += lsizes[i];
    }
    return res * nsamples_train;
  }

  // Compute the number of elments in the X matrix:
  unsigned int nx() const
  {
    return nsamples_train * lsizes[0];
  }

  // Compute the number of elments in the yy matrix:
  unsigned int ny() const
  {
    return nsamples_train * lsizes[nl - 1];
  }

  // Compute the number of elments in the X matrix:
  unsigned int nx_cv() const
  {
    return nsamples_cv * lsizes[0];
  }

  // Compute the number of elments in the yy matrix:
  unsigned int ny_cv() const
  {
    return nsamples_cv * lsizes[nl - 1];
  }

};

};

#endif
