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

BOOST_AUTO_TEST_CASE( test_building_gd_traits )
{
  GradientDescentd::Traits traits;

  // check the default values:
  BOOST_CHECK(traits.nl()==0);
  BOOST_CHECK(traits.lsizes()==nullptr);
  BOOST_CHECK(traits.nsamples()==0);
  BOOST_CHECK(traits.nparams()==0);
  BOOST_CHECK(traits.X_train()==nullptr);
  BOOST_CHECK(traits.y_train()==nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
