#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <limits>

#include <boost/chrono.hpp>

#ifdef min
#undef min
#endif

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

BOOST_AUTO_TEST_CASE( test_copy_vector_method )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // now load the train method:
  typedef void (*Func)(double* dest, double* src, unsigned int size, bool invert);

  Func copy_vector = (Func) GetProcAddress(h, "copy_vector");
  BOOST_CHECK(copy_vector != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 10;
  for(unsigned int i=0;i<num;++i) {
    // prepare number of samples:
    unsigned int size = random_int(50,100);

    double* dest = new double[size];
    double* src = new double[size];

    for(unsigned int j=0; j<size; ++j) {
      src[j] = random_double(-10.0,10.0);
    }

    copy_vector(dest,src,size,false);

    for(unsigned int j=0;j<size;++j) {
      double v1 = src[j];
      double v2 = dest[j];
      BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch on copied element "<<j<<": "<<v1<<"!="<<v2); 
    }

    copy_vector(dest,src,size,true);

    for(unsigned int j=0;j<size;++j) {
      double v1 = src[j];
      double v2 = -dest[j];
      BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch on niverted copied element "<<j<<": "<<v1<<"!="<<v2); 
    }

    delete [] dest;
    delete [] src;
  }

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_mix_vectors_method )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // now load the train method:
  typedef void (* Func)(double* res, double* vec1, double* vec2, double w1, double w2, unsigned int size);

  Func mix_vectors = (Func) GetProcAddress(h, "mix_vectors");
  BOOST_CHECK(mix_vectors != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 10;
  for(unsigned int i=0;i<num;++i) {
    // prepare number of samples:
    unsigned int size = random_int(50,100);

    double* dest = new double[size];
    double* pred = new double[size];
    double* src1 = new double[size];
    double* src2 = new double[size];
    double w1 = random_double(-10.0,10.0);
    double w2 = random_double(-10.0,10.0);

    for(unsigned int j=0; j<size; ++j) {
      src1[j] = random_double(-10.0,10.0);
      src2[j] = random_double(-10.0,10.0);
      pred[j] = w1 * src1[j] + w2 * src2[j];
    }

    mix_vectors(dest,src1,src2,w1,w2,size);

    for(unsigned int j=0;j<size;++j) {
      double v1 = pred[j];
      double v2 = dest[j];
      BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch on copied element "<<j<<": "<<v1<<"!="<<v2); 
    }

    delete [] dest;
    delete [] pred;
    delete [] src1;
    delete [] src2;
  }

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_length2_method )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // now load the train method:
  typedef double (*Func)(double* src, unsigned int size);

  Func length2 = (Func) GetProcAddress(h, "compute_length2");
  BOOST_CHECK(length2 != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 10;
  for(unsigned int i=0;i<num;++i) {
    // prepare number of samples:
    unsigned int size = random_int(50,100);

    double* src = new double[size];

    double pred = 0.0;

    for(unsigned int j=0; j<size; ++j) {
      src[j] = random_double(-10.0,10.0);
      pred += src[j]*src[j];
    }

    double len = length2(src,size);

    BOOST_CHECK_MESSAGE(abs(len-pred)<1e-10,"Mismatch on length2 value: "<<len<<"!="<<pred); 

    delete [] src;
  }

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_dot_method )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // now load the train method:
  typedef double (*Func)(double* src1, double* src2, unsigned int size);

  Func dot = (Func) GetProcAddress(h, "compute_dot");
  BOOST_CHECK(dot != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 10;
  for(unsigned int i=0;i<num;++i) {
    // prepare number of samples:
    unsigned int size = random_int(50,100);

    double* src = new double[size];
    double* src2 = new double[size];

    double pred = 0.0;

    for(unsigned int j=0; j<size; ++j) {
      src[j] = random_double(-10.0,10.0);
      src2[j] = random_double(-10.0,10.0);
      pred += src[j]*src2[j];
    }

    double len = dot(src,src2,size);

    BOOST_CHECK_MESSAGE(abs(len-pred)<1e-10,"Mismatch on dot value: "<<len<<"!="<<pred); 

    delete [] src;
    delete [] src2;
  }

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_costfunc_performances )
{
  cudaProfilerStart();

  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  typedef void (*CostFunc)(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
  double* nn_params, double* X, double* yy, double lambda, double& J, double* gradients, double* deltas, double* inputs);

  CostFunc costfunc = (CostFunc) GetProcAddress(h, "costFunc");
  BOOST_CHECK(costfunc != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 1;
  for(unsigned int i=0;i<num;++i) {
    // prepare number of samples:
    unsigned int nsamples = 2000; //random_int(50,100);

    // max number of iterations:
    unsigned int maxiter = 10;

    // Prepare the layer size vector:
    unsigned int nl = 3; //random_int(3,5);
    unsigned int nt = nl-1;

    // logDEBUG("Num samples: "<<nsamples<<", num layers: "<<nl);

    unsigned int* lsizes = new unsigned int[nl];
    lsizes[0] = 1441;
    lsizes[1] = 200;
    lsizes[2] = 3;

    // for(unsigned int j = 0; j<nl; ++j) {
    //   lsizes[j] = random_int(3,6);
    // }

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

    double J=0.0;
    unsigned int niter=10;
    for(unsigned int j=0;j<niter;++j) {
      costfunc(nl, lsizes, nsamples, init_params, X, yy, lambda,J, grads, NULL, NULL);
    }

    boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
    logDEBUG("GPU costfunc taking " << (sec.count()) << " seconds.");

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

BOOST_AUTO_TEST_CASE( test_cg_train_performances )
{
  cudaProfilerStart();

  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // now load the train method:
  typedef void (* TrainFunc)(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
    unsigned int* lsizes, double* X, double* yy, double* init_params, 
    double lambda, unsigned int maxiter, double* params);

  // TrainFunc cgtrain_cpu = (TrainFunc) GetProcAddress(h, "cgtrainCPU");
  // BOOST_CHECK(cgtrain_cpu != nullptr);

  TrainFunc cgtrain = (TrainFunc) GetProcAddress(h, "cgtrain");
  BOOST_CHECK(cgtrain != nullptr);

  // Prepare some test cases to check that the 2 methods are computing the same things:
  unsigned int num = 1;
  for(unsigned int i=0;i<num;++i) {
    // prepare number of samples:
    unsigned int nsamples = 2000; //random_int(50,100);

    // max number of iterations:
    unsigned int maxiter = 10;

    // Prepare the layer size vector:
    unsigned int nl = 3; //random_int(3,5);
    unsigned int nt = nl-1;

    // logDEBUG("Num samples: "<<nsamples<<", num layers: "<<nl);

    unsigned int* lsizes = new unsigned int[nl];
    lsizes[0] = 1441;
    lsizes[1] = 200;
    lsizes[2] = 3;

    // for(unsigned int j = 0; j<nl; ++j) {
    //   lsizes[j] = random_int(3,6);
    // }

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
    // double* pred_params = new double[nparams];
    // memset(pred_params,0,sizeof(double)*nparams);

    boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();

    // Compute the predictions:
    // cgtrain_cpu(nl, nsamples, nparams, lsizes, X, yy, init_params, lambda, maxiter, pred_params);
    cgtrain(nl, nsamples, nparams, lsizes, X, yy, init_params, lambda, maxiter, params);

    boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
    logDEBUG("GPU cgtrain taking " << (sec.count()) << " seconds.");

    // Also compare the delta arrays:
    // for(unsigned int j=0; j<nparams;++j) {
    //   double v1 = params[j];
    //   double v2 = pred_params[j];
    //   BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch on params element "<<j<<": "<<v1<<"!="<<v2);      
    // }

    delete [] lsizes;
    delete [] X;
    delete [] yy;
    delete [] init_params;
    delete [] params;
    // delete [] pred_params;
  }

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
 
  cudaProfilerStop();  
}

BOOST_AUTO_TEST_SUITE_END()
