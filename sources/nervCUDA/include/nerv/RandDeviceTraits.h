#ifndef NERV_RANDDEVICETRAITS_H_
#define NERV_RANDDEVICETRAITS_H_

#include <nervcuda.h>
#include <nerv/BPTraits.h>
#include <nerv/Utils.h>

#include <curand_kernel.h>

#include <ctime>

NERVCUDA_EXPORT void init_rand_state_device(curandState *d_state, unsigned int size, unsigned long seed);

namespace nerv
{

template<typename T>
struct RandDeviceTraits : public RandTraits<T>
{
  typedef std::vector<T *> BufferList;

  RandDeviceTraits(bool withStream = false)
    : randStates(nullptr), stream(nullptr), owned_stream(withStream), owned_states(false)
  {
    if (withStream)
    {
      checkCudaErrors(cudaStreamCreate(&stream));
    }
  };

  RandDeviceTraits(const RandDeviceTraits &rhs)
  {
    // use operator= from base class:
    RandTraits<T>::operator=(rhs);

    randStates = rhs.randStates;
    stream = rhs.stream;
    owned_stream = false;
    owned_states = false;
  };

  RandDeviceTraits &operator=(const RandDeviceTraits &) = delete;

  RandDeviceTraits(const RandTraits<T> &rhs)
  {
    assign(rhs);
  }

  RandDeviceTraits &operator=(const RandTraits<T> &rhs)
  {
    release();

    assign(rhs);
    return *this;
  }

  ~RandDeviceTraits()
  {
    release();
    if (owned_stream)
    {
      checkCudaErrors(cudaStreamDestroy(stream));
    }
  }

  T *createDeviceBuffer(unsigned int n, T *data = NULL)
  {
    T *ptr = NULL;
    if (n == 0)
    {
      throw std::runtime_error("ERROR: Invalid size if 0 when build device buffer.");
    }

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

    _buffers.push_back(ptr);
    return ptr;
  }

public:
  cudaStream_t stream;

  curandState *randStates;

protected:
  BufferList _buffers;
  bool owned_stream;
  bool owned_states;

  void release()
  {
    size_t num = _buffers.size();
    for (size_t i = 0; i < num; ++i)
    {
      releaseDeviceBuffer(_buffers[i]);
    }

    _buffers.clear();

    if (owned_states && randStates)
    {
      checkCudaErrors(cudaFree(randStates));
      randStates = nullptr;
    }
  }

  void assign(const RandTraits<T> &rhs)
  {
    // use operator= from base class:
    RandTraits<T>::operator=(rhs);

    if (rhs.target)
    {
      target = createDeviceBuffer(rhs.size, rhs.target);
    }

    if (rhs.values)
    {
      values = createDeviceBuffer(rhs.size, rhs.values);
    }

    initRandStates();
  }

  void initRandStates()
  {
    // we should allocate the curand states here:
    unsigned int size = BLOCK_SIZE * BLOCK_SIZE;
    checkCudaErrors(cudaMalloc(&randStates, size * sizeof(curandState)));

    // Here we call the method to initialize the random states:
    init_rand_state_device(randStates, size, (unsigned long)time(NULL));
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

};

#endif
