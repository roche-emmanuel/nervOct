#ifndef NERV_BPTRAITS_H_
#define NERV_BPTRAITS_H_

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
  BPTraits()
    :  wmults(nullptr), cost(0.0),
       nsamples(0), nl(0), lsizes(nullptr) {};

  unsigned int nsamples;
  unsigned int nl;
  unsigned int *lsizes;

  T cost;
  T *wmults;
};

template<typename T>
struct BPComputeTraits : public BPTraitsBase<T>
{
  BPComputeTraits()
    : theta_offset(0),
      input_offset(0), next_input_offset(0),
      delta_offset(0), next_delta_offset(0),
      grad_offset(0),
      nrows(0), ncols(0), niter(0),
      wmult(1.0) {};

  unsigned int theta_offset;

  int input_offset;
  unsigned int next_input_offset;

  unsigned int delta_offset;
  unsigned int next_delta_offset;

  unsigned int grad_offset;

  unsigned int nrows;
  unsigned int ncols;
  unsigned int niter;

  T wmult;
};

};

#endif
