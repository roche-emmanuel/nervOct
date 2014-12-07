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

template<typename T>
struct BPTraits : public BPTraitsBase<T>
{
  typedef std::vector<T *> ArrayList;

  BPTraits()
    :  wmults(nullptr), cost(0.0),
       nsamples(0), nl(0), lsizes(nullptr) {};

  virtual ~BPTraits() {}

  unsigned int nsamples;
  unsigned int nl;
  unsigned int *lsizes;

  T cost;
  T *wmults;

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
    return res * nsamples;
  }

  // Compute the number of elments in the X matrix:
  unsigned int nx() const
  {
    return nsamples * lsizes[0];
  }

  // Compute the number of elments in the yy matrix:
  unsigned int ny() const
  {
    return nsamples * lsizes[nl - 1];
  }
};

};

#endif
