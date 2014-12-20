#ifndef NERV_UTILS_H_
#define NERV_UTILS_H_

#include <nervcuda.h>

#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

#define CHECK_KERNEL() checkCudaErrors( cudaPeekAtLastError() ); \
checkCudaErrors( cudaDeviceSynchronize() );

#include <iostream>
#include <sstream>

#ifndef logDEBUG
// #define logDEBUG(msg) std::cout << std::setprecision(16) << msg << std::endl;
#define logDEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl;
#endif

#ifndef logERROR
#define logERROR(msg) std::cout << "[ERROR] " << msg << std::endl;
#endif

#ifndef THROW
#define THROW(msg) { std::ostringstream os; os << msg; logERROR("Throwing exception: " << msg); throw std::runtime_error(os.str()); }
#endif

#ifndef THROW_IF
#define THROW_IF(cond,msg) if(cond) THROW(msg)
#endif

extern "C" bool isPow2(unsigned int x);

unsigned int nextPow2(unsigned int x);

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

NERVCUDA_EXPORT const char * _cudaGetErrorEnum(cudaError_t error);
NERVCUDA_EXPORT const char * _cublasGetErrorEnum(cublasStatus_t error);

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

template< typename T >
void checkBlas(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    fprintf(stderr, "CUBLAS error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), _cublasGetErrorEnum(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

#define checkCublasErrors(val) checkBlas ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, cudaGetErrorString(err));
    DEVICE_RESET
    exit(EXIT_FAILURE);
  }
}

namespace nerv
{

template<typename T>
T *createGPUBuffer(unsigned int n, bool init = true)
{
  T *ptr = NULL;
  size_t size = n * sizeof(T);
  checkCudaErrors(cudaMalloc(&ptr, size));
  if (init) {
    checkCudaErrors(cudaMemset(ptr, 0, size));
  }

  return ptr;
}

template<typename T>
T *createGPUBuffer(unsigned int n, T* data)
{
  T *ptr = NULL;
  size_t size = n * sizeof(T);
  checkCudaErrors(cudaMalloc(&ptr, size));
  if(data) {
    checkCudaErrors(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
  }

  return ptr;
}

template<typename T>
void destroyGPUBuffer(T *ptr)
{
  if (ptr)
  {
    checkCudaErrors(cudaFree(ptr));
    ptr = nullptr;
  }
}

template<typename T>
void copyFromDevice(T* dest, T* src, unsigned int n) {
  checkCudaErrors(cudaMemcpy(dest, src, sizeof(T)*n, cudaMemcpyDeviceToHost));
}

template<typename T>
void copyToDevice(T* dest, T* src, unsigned int n) {
  checkCudaErrors(cudaMemcpy(dest, src, sizeof(T)*n, cudaMemcpyHostToDevice));
}

};

#endif
