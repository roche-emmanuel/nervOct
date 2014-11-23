#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <limits>

#include <boost/chrono.hpp>

#ifdef min
#undef min
#endif

#define logDEBUG(msg) std::cout << msg << std::endl;

int random_int(int mini, int maxi) {
  return mini + (int)floor(0.5 + (maxi-mini)*(double)rand()/(double)RAND_MAX);
}

double random_double(double mini, double maxi) {
  return mini + (maxi-mini)*(double)rand()/(double)RAND_MAX;
}

BOOST_AUTO_TEST_SUITE( train_backprop )

BOOST_AUTO_TEST_CASE( test_sanity_loading_module )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_retrieve_train_method_cpu )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // now load the train method:
  typedef void (* TrainFunc)(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
    unsigned int* lsizes, double* X, double* yy, double* init_params, 
    double lambda, unsigned int maxiter, double* params);

  TrainFunc cgtrain = (TrainFunc) GetProcAddress(h, "cgtrainCPU");
  BOOST_CHECK(cgtrain != nullptr);

  // logDEBUG("Double min value is: "<< std::numeric_limits<double>::min());

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_compare_cg_train_method )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // now load the train method:
  typedef void (* TrainFunc)(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
    unsigned int* lsizes, double* X, double* yy, double* init_params, 
    double lambda, unsigned int maxiter, double* params);

  TrainFunc cgtrain_cpu = (TrainFunc) GetProcAddress(h, "cgtrainCPU");
  BOOST_CHECK(cgtrain_cpu != nullptr);

  TrainFunc cgtrain = (TrainFunc) GetProcAddress(h, "cgtrain");
  BOOST_CHECK(cgtrain != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 10;
  for(unsigned int i=0;i<num;++i) {
    // prepare number of samples:
    unsigned int nsamples = random_int(50,100);

    // max number of iterations:
    unsigned int maxiter = 10;

    // Prepare the layer size vector:
    unsigned int nl = random_int(3,5);
    unsigned int nt = nl-1;

    // logDEBUG("Num samples: "<<nsamples<<", num layers: "<<nl);

    unsigned int* lsizes = new unsigned int[nl];

    for(unsigned int j = 0; j<nl; ++j) {
      lsizes[j] = random_int(3,6);
    }

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
    double lambda = random_double(0.0,1.0);

    // prepare the output gradient array:
    double* params = new double[nparams];
    memset(params,0,sizeof(double)*nparams);
    double* pred_params = new double[nparams];
    memset(pred_params,0,sizeof(double)*nparams);

    // Compute the predictions:
    cgtrain_cpu(nl, nsamples, nparams, lsizes, X, yy, init_params, lambda, maxiter, pred_params);
    cgtrain(nl, nsamples, nparams, lsizes, X, yy, init_params, lambda, maxiter, params);

    // Also compare the delta arrays:
    for(unsigned int j=0; j<nparams;++j) {
      double v1 = params[j];
      double v2 = pred_params[j];
      BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch on params element "<<j<<": "<<v1<<"!="<<v2);      
    }


    delete [] lsizes;
    delete [] X;
    delete [] yy;
    delete [] init_params;
    delete [] params;
    delete [] pred_params;
  }

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}


BOOST_AUTO_TEST_SUITE_END()
