#ifndef NERV_BPDEVICETRAITS_H_
#define NERV_BPDEVICETRAITS_H_

#include <nerv/BPTraits.h>
#include <nerv/Utils.h>

namespace nerv
{

template<typename T>
struct BPDeviceTraits : public BPTraits<T>
{
  typedef std::vector<T *> BufferList;

  BPDeviceTraits(cudaStream_t s = 0) : regw(nullptr), stream(s) {};

  BPDeviceTraits(const BPDeviceTraits &) = delete;
  BPDeviceTraits &operator=(const BPDeviceTraits &) = delete;

  BPDeviceTraits(const BPTraits<T> &rhs)
  {
    assign(rhs);
  }

  BPDeviceTraits &operator=(const BPTraits<T> &rhs)
  {
    release();

    assign(rhs);
    return *this;
  }

  ~BPDeviceTraits()
  {
    release();
  }

  T *regw; // array containing the L2 regularization weights.
  cudaStream_t stream;

  T *createDeviceBuffer(unsigned int n, T *data = NULL)
  {
    T *ptr = NULL;
    size_t size = n * sizeof(T);
    checkCudaErrors(cudaMalloc(&ptr, size));
    if (data)
    {
      checkCudaErrors(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
    }
    else
    {
      checkCudaErrors(cudaMemset(ptr, 0, size));
    }

    // If using pinned memory:
    // checkCudaErrors(cudaHostRegister(traits.X, size, cudaHostRegisterDefault)); // register the memory as pinned memory.
    // checkCudaErrors(cudaMalloc(&d_X_train, size));
    // checkCudaErrors(cudaMemcpyAsync(d_X_train, traits.X, size, cudaMemcpyHostToDevice, _stream1));

    // To unregister:
    // checkCudaErrors(cudaHostUnregister(_traits.X));

    _buffers.push_back(ptr);
    return ptr;
  }

protected:
  BufferList _buffers;

  void release()
  {
    size_t num = _buffers.size();
    for (size_t i = 0; i < num; ++i)
    {
      releaseDeviceBuffer(_buffers[i]);
    }

    _buffers.clear();

    params = regw = inputs = yy = X = deltas = grads = nullptr;
  }

  void assign(const BPTraits<T> &rhs)
  {
    bias = rhs.bias;
    lambda = rhs.lambda;
    nsamples = rhs.nsamples;
    nl = rhs.nl;
    lsizes = rhs.lsizes;
    cost = rhs.cost;
    wmults = rhs.wmults;

    X = createDeviceBuffer(nx(), rhs.X);
    yy = createDeviceBuffer(ny(), rhs.yy);

    params = createDeviceBuffer(np(), rhs.params);
    grads = createDeviceBuffer(np());

    inputs = createDeviceBuffer(nd());
    deltas = createDeviceBuffer(nd());

    buildL2RegWeights();
  }



  void buildL2RegWeights()
  {
    unsigned int n = np();
    unsigned int nt = nl - 1;

    T *h_regw = new T[n];
    memset(h_regw, 0, n * sizeof(T));

    // prepare the regularization correction:
    T *rptr = h_regw;

    for (unsigned int i = 0; i < nt; ++i)
    {
      unsigned int nrows = lsizes[i + 1];
      unsigned int ncolT = lsizes[i]; // we remove 1 here because we consider the intercept row as "virtual" in our calculation.

      rptr += nrows;
      unsigned int count = nrows * ncolT;

      for (unsigned int j = 0; j < count; ++j)
      {
        (*rptr++) = 1.0;
      }
    }

    regw = createDeviceBuffer(n, h_regw);
    delete [] h_regw;
  }

  void releaseDeviceBuffer(T *&ptr)
  {
    if (ptr)
    {
      checkCudaErrors(cudaFree(ptr));
      ptr = nullptr;
    }
  }
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
