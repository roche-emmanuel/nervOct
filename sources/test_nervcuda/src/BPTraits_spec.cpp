#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <GradientDescent.h>

#include <boost/chrono.hpp>

using namespace nerv;

BOOST_AUTO_TEST_SUITE( bptraits_suite )

BOOST_AUTO_TEST_CASE( test_copy_bptraits )
{
  typedef double value_type;

  BPTraits<value_type> t1;
  t1.nl = 5;
  t1.lsizes = new unsigned int[5];

  BPTraits<value_type> t2 = t1;

  BOOST_CHECK(t1.nl == t2.nl);
  BOOST_CHECK(t1.lsizes == t2.lsizes);
  BOOST_CHECK(nullptr == t2.hx);

  BPTraits<value_type> t3;

  t3 = t1;
  BOOST_CHECK(t1.nl == t3.nl);
  BOOST_CHECK(t1.lsizes == t3.lsizes);

  delete [] t1.lsizes;
}

BOOST_AUTO_TEST_SUITE_END()
