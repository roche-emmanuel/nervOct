#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <boost/chrono.hpp>

#define logDEBUG(msg) std::cout << msg << std::endl;

BOOST_AUTO_TEST_SUITE( gpu_perfs_suite )

BOOST_AUTO_TEST_CASE( test_mult_mat_perfs_gpu )
{
  cudaProfilerStart();

  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

  // We should be able to retrieve the train function:
  MultMatFunc mult_mat = (MultMatFunc) GetProcAddress(h, "matmult");
  BOOST_CHECK(mult_mat != nullptr);

  unsigned int nrowA = 2000;
  unsigned int ncolA = 2000;
  unsigned int nrowB = ncolA;
  unsigned int ncolB = 2000;

  // prepare the matrix data:
  unsigned int count = nrowA*ncolA;
  double* ptr;
  double* A = new double[count];
  ptr = A;
  for(unsigned int j=0;j<count;++j) {
    // (*ptr++) = random_double(-10.0,10.0);
    (*ptr++) = sin(j)*10.0;
  }

  count = nrowB*ncolB;
  double* B = new double[count];
  ptr = B;
  for(unsigned int j=0;j<count;++j) {
    // (*ptr++) = random_double(-10.0,10.0);
    (*ptr++) = cos(j)*10.0;
  }

  count = nrowA*ncolB;
  double* C = new double[count];
  memset((void*)C,0,sizeof(double)*count);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  // Compute the matrix onthe GPU:
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  
  for(unsigned int i = 0;i<num;++i) {
    mult_mat(nrowA, ncolA, A, nrowB, ncolB, B, C, false, false);
  }

  boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
  logDEBUG("GPU matrix mult taking " << (sec.count()/num) << " seconds.");

  delete [] A;
  delete [] B;
  delete [] C;

  BOOST_CHECK(FreeLibrary(h));
  cudaProfilerStop();
}

BOOST_AUTO_TEST_CASE( test_mult_mat_float_perfs_gpu )
{
  cudaProfilerStart();

  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const float* A,
    unsigned int nrowB, unsigned int ncolB, const float* B, float* C, bool tpA, bool tpB);

  // We should be able to retrieve the train function:
  MultMatFunc mult_mat = (MultMatFunc) GetProcAddress(h, "matmult_f");
  BOOST_CHECK(mult_mat != nullptr);

  unsigned int nrowA = 2000;
  unsigned int ncolA = 2000;
  unsigned int nrowB = ncolA;
  unsigned int ncolB = 2000;

  // prepare the matrix data:
  unsigned int count = nrowA*ncolA;
  float* ptr;
  float* A = new float[count];
  ptr = A;
  for(unsigned int j=0;j<count;++j) {
    // (*ptr++) = random_float(-10.0,10.0);
    (*ptr++) = (float)(sin(j)*10.0);
  }

  count = nrowB*ncolB;
  float* B = new float[count];
  ptr = B;
  for(unsigned int j=0;j<count;++j) {
    // (*ptr++) = random_float(-10.0,10.0);
    (*ptr++) = (float)(cos(j)*10.0);
  }

  count = nrowA*ncolB;
  float* C = new float[count];
  memset((void*)C,0,sizeof(float)*count);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  // Compute the matrix onthe GPU:
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  
  for(unsigned int i = 0;i<num;++i) {
    mult_mat(nrowA, ncolA, A, nrowB, ncolB, B, C, false, false);
  }

  boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
  logDEBUG("GPU matrix float mult taking " << (sec.count()/num) << " seconds.");

  delete [] A;
  delete [] B;
  delete [] C;

  BOOST_CHECK(FreeLibrary(h));
  cudaProfilerStop();
}

BOOST_AUTO_TEST_SUITE_END()
