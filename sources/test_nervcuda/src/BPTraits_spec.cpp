#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <GradientDescent.h>
#include <nerv/BPDeviceTraits.h>

#include <boost/chrono.hpp>

using namespace nerv;

BOOST_AUTO_TEST_SUITE( bptraits_suite )

BOOST_AUTO_TEST_CASE( test_copy_bptraits )
{
  typedef double value_type;

  BPTraits<value_type> t1;
  t1.nl = 5;
  t1.lsizes = new unsigned int[5];
  BOOST_CHECK(t1.debug == false);

  BPTraits<value_type> t2 = t1;

  BOOST_CHECK(t1.nl == t2.nl);
  BOOST_CHECK(t1.lsizes == t2.lsizes);
  BOOST_CHECK(nullptr == t2.hx);
  BOOST_CHECK(t2.debug == false);

  BPTraits<value_type> t3;

  t3 = t1;
  BOOST_CHECK(t1.nl == t3.nl);
  BOOST_CHECK(t1.lsizes == t3.lsizes);

  delete [] t1.lsizes;
}

BOOST_AUTO_TEST_CASE( test_bpdevicetraits )
{
  typedef double value_type;

  BPTraits<value_type> t1;
  t1.nl = 5;
  t1.lsizes = new unsigned int[5];
  t1.nsamples_train = 10;
  t1.bias = 0.5;
  t1.lambda = 0.5;
  t1.cost = 1.1;

  BPDeviceTraits<value_type> dt;
  dt = t1;

  BOOST_CHECK(t1.nl == dt.nl);
  BOOST_CHECK(t1.lsizes == dt.lsizes);
  BOOST_CHECK(dt.nsamples == t1.nsamples_train);
  BOOST_CHECK(dt.nsamples_train == t1.nsamples_train);
  BOOST_CHECK(dt.nsamples_cv == t1.nsamples_cv);
  BOOST_CHECK(dt.bias == t1.bias);
  BOOST_CHECK(dt.lambda == t1.lambda);
  BOOST_CHECK(dt.cost == t1.cost);
  BOOST_CHECK(dt.compute_cost == t1.compute_cost);
  BOOST_CHECK(dt.compute_grads == t1.compute_grads);
  BOOST_CHECK(dt.wmults == t1.wmults);
  BOOST_CHECK(dt.randStates == nullptr);  
  BOOST_CHECK(dt.wbias == nullptr);  

  BPDeviceTraits<value_type> dt2;
  BOOST_CHECK(dt2.X_train == nullptr);
  BOOST_CHECK(dt2.y_train == nullptr);

  delete [] t1.lsizes;
}

BOOST_AUTO_TEST_CASE( test_bpdevicetraits_dropouts )
{
  typedef double value_type;

  BPTraits<value_type> t1;
  t1.nl = 5;
  t1.lsizes = new unsigned int[5];
  t1.nsamples_train = 10;
  t1.bias = 0.5;
  t1.lambda = 0.5;
  t1.cost = 1.1;
  t1.dropouts = new value_type[4];

  BPDeviceTraits<value_type> dt;

  // RandState should not be initialized on construction:
  BOOST_CHECK(dt.randStates == nullptr);  
  BOOST_CHECK(dt.wbias == nullptr);  
  dt = t1;
  BOOST_CHECK(dt.randStates != nullptr);  
  BOOST_CHECK(dt.wbias != nullptr);  

  delete [] t1.lsizes;
  delete [] t1.dropouts;
}

BOOST_AUTO_TEST_SUITE_END()
