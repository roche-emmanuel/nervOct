#ifndef NERV_BPDEVICETRAITS_H_
#define NERV_BPDEVICETRAITS_H_

#include <nervcuda.h>
#include <nerv/BPTraits.h>
#include <nerv/Utils.h>

#include <curand_kernel.h>

#include <ctime>

NERVCUDA_EXPORT void init_rand_state_device(curandState *d_state, unsigned int size, unsigned long seed);

namespace nerv
{

template<typename T>
struct BPDeviceTraits : public BPTraits<T>
{
  typedef std::vector<T *> BufferList;

  BPDeviceTraits(bool withStream = false)
    : regw(nullptr), stream(nullptr), owned_stream(false), X_train(nullptr),
      y_train(nullptr), randStates(nullptr), wbias(nullptr), wX(nullptr), rX(nullptr),
      handle(nullptr), softmax_ones(nullptr), softmax_norms(nullptr),
      spae_ones(nullptr), spae_rho(nullptr), spae_kl(nullptr), spae_delta(nullptr), id(0)
  {
    if (withStream)
    {
      allocateStream();
    }
  };

  BPDeviceTraits(const BPDeviceTraits &) = delete;
  BPDeviceTraits &operator=(const BPDeviceTraits &rhs)
  {
    // copy the elements:
    BPTraits<T>::operator=(rhs);

    // Retrieve all the traits details from the dev traits:
    nsamples =rhs.nsamples;

    regw = rhs.regw;
    stream = rhs.stream;
    handle = rhs.handle;

    softmax_ones = rhs.softmax_ones;
    softmax_norms = rhs.softmax_norms;

    X_train = rhs.X_train;
    y_train = rhs.y_train;
    wbias = rhs.wbias;
    wX = rhs.wX;
    rX = rhs.rX;

    randStates = rhs.randStates;

    spae_ones = rhs.spae_ones;
    spae_rho = rhs.spae_rho;
    spae_kl = rhs.spae_kl;
    spae_delta = rhs.spae_delta;

    return *this;
  }

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
    if (owned_stream)
    {
      checkCudaErrors(cudaStreamDestroy(stream));
    }

    if (handle)
    {
      checkCublasErrors(cublasDestroy(handle));
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

    // If using pinned memory:
    // checkCudaErrors(cudaHostRegister(traits.X, size, cudaHostRegisterDefault)); // register the memory as pinned memory.
    // checkCudaErrors(cudaMalloc(&d_X_train, size));
    // checkCudaErrors(cudaMemcpyAsync(d_X_train, traits.X, size, cudaMemcpyHostToDevice, _stream1));

    // To unregister:
    // checkCudaErrors(cudaHostUnregister(_traits.X));

    _buffers.push_back(ptr);
    return ptr;
  }

  void allocateStream()
  {
    THROW_IF(stream, "Stream already allocated.")
    owned_stream = true;
    checkCudaErrors(cudaStreamCreate(&stream));
  }

  // T* predictions() {
  //   unsigned int offset = 0;
  //   unsigned int nt = nl - 1;

  //   for(unsigned int i=1;i<nt;++i) {
  //     offset += lsizes[i];
  //   }

  //   return inputs + offset * nsamples;
  // }

public:
  unsigned int nsamples;

  T *regw; // array containing the L2 regularization weights.
  cudaStream_t stream;
  cublasHandle_t handle;

  T *softmax_ones;
  T *softmax_norms;

  T *X_train;
  T *y_train;
  T *wbias;
  T *wX;
  T *rX; // Buffer used to hold the rand features computation when dropout is activated.

  curandState *randStates;

  // Sparse auto encoder variables:
  T *spae_ones;
  T *spae_rho;
  T *spae_kl;
  T *spae_delta;

  // Id used when using the BPTraitsManager:
  int id;

protected:
  BufferList _buffers;
  bool owned_stream;

  void release()
  {
    size_t num = _buffers.size();
    for (size_t i = 0; i < num; ++i)
    {
      releaseDeviceBuffer(_buffers[i]);
    }

    _buffers.clear();

    params = regw = inputs = yy = X = deltas = grads = nullptr;

    if (randStates)
    {
      checkCudaErrors(cudaFree(randStates));
      randStates = nullptr;
    }
  }

  void assign(const BPTraits<T> &rhs)
  {
    // use operator= from base class:
    BPTraits<T>::operator=(rhs);

    // Update the cost mode if needed:
    if (cost_mode == COST_DEFAULT)
    {
      if (use_softmax)
      {
        // logDEBUG("Updating cost mode to sotfmax cost");
        cost_mode = COST_SOFTMAX;
      }
      else if (spae_beta > 0.0)
      {
        // logDEBUG("Updating cost mode to SPAE cost");
        cost_mode = COST_RMS;
      }
      else
      {
        // logDEBUG("Updating cost mode to cross entropy cost");
        cost_mode = COST_CROSS_ENTROPY;
      }
    }

    nsamples = rhs.nsamples_train;

    X_train = createDeviceBuffer(nx(), rhs.X);
    y_train = createDeviceBuffer(ny(), rhs.yy);

    X = X_train;
    yy = y_train;

    if (rhs.X_cv)
    {
      X_cv = createDeviceBuffer(nx_cv(), rhs.X_cv);
    }

    if (rhs.y_cv)
    {
      y_cv = createDeviceBuffer(ny_cv(), rhs.y_cv);
    }

    params = createDeviceBuffer(np(), rhs.params);
    grads = createDeviceBuffer(np());

    inputs = createDeviceBuffer(nd());
    deltas = createDeviceBuffer(nd());

    buildL2RegWeights();

    if (rhs.dropouts)
    {
      initRandStates();

      // Prepare an array to hold the bias weights:
      wbias = createDeviceBuffer((nl - 1) * nsamples);
      rX = createDeviceBuffer(lsizes[0] * nsamples);
    }

    // check if softmax is enabled, and in that case create the handle
    if (use_softmax)
    {
      // allocate the CUBLAS handle:
      if (!handle)
      {
        checkCublasErrors(cublasCreate(&handle));
      }

      // Allocate the need device buffers:
      unsigned int nout = lsizes[nl - 1];
      T *ones = new T[nout];
      for (unsigned int i = 0; i < nout; ++i)
      {
        ones[i] = 1.0;
      }

      softmax_ones = createDeviceBuffer(nout, ones);
      softmax_norms = createDeviceBuffer(nsamples);

      delete [] ones;
    }

    if (spae_beta > 0.0)
    {
      // allocate the CUBLAS handle:
      if (!handle)
      {
        checkCublasErrors(cublasCreate(&handle));
      }

      // Allocate the need device buffers:
      T *ones = new T[nsamples];
      for (unsigned int i = 0; i < nsamples; ++i)
      {
        ones[i] = 1.0;
      }

      spae_ones = createDeviceBuffer(nsamples, ones);
      spae_rho = createDeviceBuffer(lsizes[1]);
      spae_kl = createDeviceBuffer(lsizes[1]);
      spae_delta = createDeviceBuffer(lsizes[1]);

      delete [] ones;
    }
  }

  void initRandStates()
  {
    // we should allocate the curand states here:
    unsigned int size = BLOCK_SIZE * BLOCK_SIZE;
    checkCudaErrors(cudaMalloc(&randStates, size * sizeof(curandState)));

    // Here we call the method to initialize the random states:
    init_rand_state_device(randStates, size, (unsigned long)time(NULL));
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
      wmult(1.0), layer_dropout(1.0),
      randStates(nullptr),
      wbias(nullptr), wbias_offset(0),
      wX(nullptr), with_softmax(false),
      spae_delta(nullptr) {};

  // BPComputeTraits(const BPComputeTraits &) = delete;
  BPComputeTraits &operator=(const BPComputeTraits &) = delete;

  BPComputeTraits &operator=(const BPDeviceTraits<T> &rhs)
  {
    params = rhs.params;
    inputs = rhs.inputs;
    deltas = rhs.deltas;
    grads = rhs.grads;
    spae_delta = rhs.spae_delta;
    yy = rhs.yy;
    X = rhs.X;
    bias = rhs.bias;
    lambda = rhs.lambda;
    randStates = rhs.randStates;
    wbias = rhs.wbias;
    wX = rhs.wX;
    THROW_IF(!wX, "Invalid wX buffer for BPComputeTraits.");

    return *this;
  }

  unsigned int theta_offset;

  int input_offset;
  unsigned int next_input_offset;

  unsigned int delta_offset;
  unsigned int next_delta_offset;

  unsigned int grad_offset;

  unsigned int wbias_offset;

  unsigned int nrows;
  unsigned int ncols;
  unsigned int niter;

  curandState *randStates;
  T *wbias;
  T *wX;

  T *spae_delta;

  T wmult;
  T layer_dropout;

  bool with_softmax;
};

};

#endif
