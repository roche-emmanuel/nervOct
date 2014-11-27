#include <boost/test/unit_test.hpp>

#include <iostream>

#include <nervcuda.h>
#include <GradientDescentd.h>
#include <windows.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <boost/chrono.hpp>

using namespace nerv;

BOOST_AUTO_TEST_SUITE( gradient_descent_suite )

BOOST_AUTO_TEST_CASE( test_create_gd_traits )
{
  GradientDescentd::Traits traits;

  // check the default values:
  BOOST_CHECK(traits.nl()==0);
  BOOST_CHECK(traits.lsizes()==nullptr);
  BOOST_CHECK(traits.nsamples()==0);
  BOOST_CHECK(traits.nparams()==0);
  BOOST_CHECK(traits.X_train()==nullptr);
  BOOST_CHECK(traits.X_train_size()==0);
  BOOST_CHECK(traits.y_train()==nullptr);
  BOOST_CHECK(traits.y_train_size()==0);
  BOOST_CHECK(traits.params()==nullptr);
  BOOST_CHECK(traits.maxiter()==-1);
  BOOST_CHECK(traits.lambda()==0.0);
}

BOOST_AUTO_TEST_CASE( test_create_gd )
{
  typedef GradientDescentd::value_type value_t;
  GradientDescentd::Traits traits;

  // GradientDescentd gd(traits);

  // should throw when the traits are invalid:
  BOOST_CHECK_THROW( new GradientDescentd(traits), std::runtime_error);

  unsigned int sizes2[] = { 3, 4};
  traits.lsizes(sizes2,2);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  unsigned int sizes[] = { 3, 4, 1};
  traits.lsizes(sizes,3);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  unsigned int nsamples = 10;
  traits.nsamples(nsamples);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  value_t* params = nullptr;
  traits.params(params, 10);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  params = new value_t[21];
  traits.params(params,21);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  value_t* X = nullptr;
  traits.X_train(X,10);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  X = new value_t[nsamples*3];
  traits.X_train(X,nsamples*3);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  value_t* y = nullptr;
  traits.y_train(y,5);
  BOOST_CHECK_THROW( new GradientDescentd(traits),  std::runtime_error);

  y = new value_t[nsamples*1];
  traits.y_train(y,nsamples*1);

  // If we use this call here we will have a problem because of pinned memory registration.
  // BOOST_CHECK_NO_THROW( new GradientDescentd(traits) );

  // Check that we can build on stack:
  GradientDescentd gd(traits);

  delete [] y;
  delete [] X;
  delete [] params;
}

int random_int(int mini, int maxi) {
  return mini + (int)floor(0.5 + (maxi-mini)*(double)rand()/(double)RAND_MAX);
}

template <typename T>
T random_real(T mini, T maxi) {
  return mini + (maxi-mini)*(T)rand()/(T)RAND_MAX;
}

BOOST_AUTO_TEST_CASE( test_run_gd )
{
  typedef GradientDescentd::value_type value_t;
  
  // number of tests to run:
  unsigned int num = 5;
  
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

    value_t* ptr;

    // prepare the X matrix data:
    unsigned int nx = nsamples*lsizes[0];
    value_t* X = new value_t[nx];
    ptr = X;
    for(unsigned int j=0;j<nx;++j) {
      (*ptr++) = (value_t)(sin(j)*10.0);
      // (*ptr++) = random_value_t(-10.0,10.0);
    }

    // Prepare the y matrix:
    unsigned int ny = nsamples*lsizes[nl-1];
    value_t* y = new value_t[ny];
    ptr = y;
    for(unsigned int j=0;j<ny;++j) {
      (*ptr++) = (value_t)(abs(cos(j)));
      // (*ptr++) = random_value_t(-10.0,10.0);
    }

    // Prepare the current weights matrices:
    unsigned int np = 0;
    for(unsigned j=0;j<nt;++j) {
      np += lsizes[j+1]*(lsizes[j]+1);
    }

    value_t* params = new value_t[np];
    ptr = params;
    for(unsigned int j=0;j<np;++j) {
      (*ptr++) = (value_t)(sin(j+0.5));
    }    

    // prepare the lambda value:
    value_t lambda = random_real<value_t>(0.0,1.0);

    // Now we prepare the traits:
    GradientDescentd::Traits traits;
    traits.lsizes(lsizes,nl);
    traits.nsamples(nsamples);
    traits.params(params,np);
    traits.X_train(X,nx);
    traits.y_train(y,ny);
    traits.maxiter(maxiter);
    traits.lambda(lambda);

    // Check that we can build on stack:
    GradientDescentd gd(traits);

    // try to run the gradient descent:
    gd.run();

    delete [] y;
    delete [] X;
    delete [] params;
    delete [] lsizes;
  }

}

BOOST_AUTO_TEST_SUITE_END()
