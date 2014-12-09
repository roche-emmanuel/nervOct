#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <GradientDescent.h>

#include <boost/chrono.hpp>

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

BOOST_AUTO_TEST_CASE( test_gd_errfunc_performances )
{
  cudaProfilerStart();

  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  typedef void (*CostFunc)(BPTraits<double>& traits);

  CostFunc costfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(costfunc != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 1;
  for(unsigned int i=0;i<num;++i) {

    // prepare number of samples:
    unsigned int nsamples = 2000; //random_int(50,100);

    // Prepare the layer size vector:
    unsigned int nl = 3; //random_int(3,5);
    unsigned int nt = nl-1;

    // logDEBUG("Num samples: "<<nsamples<<", num layers: "<<nl);

    unsigned int* lsizes = new unsigned int[nl];
    lsizes[0] = 1441;
    lsizes[1] = 200;
    lsizes[2] = 3;

    // prepare the X matrix data:
    unsigned int count = nsamples*lsizes[0];
    double* ptr;
    double* X = new double[count];
    ptr = X;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = sin(j)*10.0;
      // (*ptr++) = random_double(-10.0,10.0);
    }

    // Prepare the yy matrix:
    count = nsamples*lsizes[nl-1];
    double* yy = new double[count];
    ptr = yy;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = abs(cos(j));
      // (*ptr++) = random_double(-10.0,10.0);
    }

    // Prepare the current weights matrices:
    count = 0;
    for(unsigned j=0;j<nt;++j) {
      count += lsizes[j+1]*(lsizes[j]+1);
    }
    unsigned int nparams = count;

    // logDEBUG("Allocating "<<count<<" bytes...")
    double* init_params = new double[count];
    ptr = init_params;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = sin(j+0.5);
    }    

    // prepare the lambda value:
    double lambda = 0.1; //random_double(0.0,1.0);

    // prepare the output gradient array:
    double* grads = new double[nparams];
    memset(grads,0,sizeof(double)*nparams);

    boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();


    BPTraits<double> traits;
    traits.nl = nl;
    traits.lsizes = lsizes;
    traits.nsamples_train = nsamples;
    traits.params = init_params;
    traits.X = X;
    traits.yy = yy;
    traits.lambda = lambda;
    traits.grads = grads;
    traits.compute_cost = true;
    traits.compute_grads = true;

    unsigned int niter=10;
    for(unsigned int j=0;j<niter;++j) {
      costfunc(traits);
    }

    boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
    logDEBUG("GPU gd_errfunc taking " << (sec.count()) << " seconds.");

    delete [] lsizes;
    delete [] X;
    delete [] yy;
    delete [] init_params;
    delete [] grads;
  }

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));

  cudaProfilerStop();
}

BOOST_AUTO_TEST_SUITE_END()
