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

BOOST_AUTO_TEST_SUITE_END()
