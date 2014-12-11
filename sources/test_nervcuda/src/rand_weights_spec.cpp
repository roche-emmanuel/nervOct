#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

#include <nervcuda.h>
#include <nerv/TrainingSet.h>
#include <nerv/BPTraits.h>
#include <GradientDescent.h>
#include <windows.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <boost/chrono.hpp>

using namespace nerv;

BOOST_AUTO_TEST_SUITE( rand_weights_suite )

BOOST_AUTO_TEST_CASE( test_rand_weights )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*Func)(value_type * weights, value_type threshold, unsigned int size, value_type value);

  // We should be able to retrieve the train function:
  Func rand_weights = (Func) GetProcAddress(h, "rand_weights");
  BOOST_CHECK(rand_weights != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    unsigned int n = random_uint(1000, 1500);

    value_type *weights = new value_type[n];
    value_type val = random_real((value_type)0.0, (value_type)1.0);

    // generate the weights:
    rand_weights(weights, 1.0, n, val);

    // Now compate the hx arrays:
    for (unsigned int j = 0; j < n; ++j)
    {
      value_type v1 = weights[j];
      BOOST_CHECK_MESSAGE(abs(v1 - val) <= epsilon, "Mismatch (thres==1.0) on weight element " << j << ": " << v1 << "!=" << val);
    }

    // Now do the generation with threshold of 0:
    // generate the weights:
    rand_weights(weights, 0.0, n, val);

    // Now compate the hx arrays:
    for (unsigned int j = 0; j < n; ++j)
    {
      value_type v1 = weights[j];
      BOOST_CHECK_MESSAGE(abs(v1) <= epsilon, "Mismatch (thres==0.0) on weight element " << j << ": " << v1 << "!= 0.0");
    }

    delete [] weights;
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_rand_weights_ratio )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*Func)(value_type * weights, value_type threshold, unsigned int size, value_type value);

  // We should be able to retrieve the train function:
  Func rand_weights = (Func) GetProcAddress(h, "rand_weights");
  BOOST_CHECK(rand_weights != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    unsigned int n = random_uint(1000000, 1500000);
    BOOST_CHECK(n>=1000000);
    BOOST_CHECK(n<=1500000);

    value_type *weights = new value_type[n];
    value_type val = random_real((value_type)0.1, (value_type)1.0);
    value_type threshold = random_real((value_type)0.0, (value_type)1.0);

    // generate the weights:
    rand_weights(weights, threshold, n, val);

    // count the number of zero values:
    value_type zeros = 0.0;
    value_type nzeros = 0.0;

    for (unsigned int j = 0; j < n; ++j)
    {
      if (weights[j] == 0.0)
      {
        zeros += 1.0;
      }
      else {
        nzeros += 1.0;
      };
    }

    BOOST_CHECK(zeros+nzeros == (value_type)n);

    // Now estimate the ratio of non zeros value:
    value_type ratio = nzeros/(zeros+nzeros);
    BOOST_CHECK_MESSAGE(abs(ratio-threshold) <= 0.01, "Invalid non zeros ratio: " << ratio << "!="<<threshold<<" for n="<<n);

    delete [] weights;
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_rand_weights_debug )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*Func)(value_type * weights, value_type threshold, unsigned int size, value_type value);

  // We should be able to retrieve the train function:
  Func rand_weights = (Func) GetProcAddress(h, "rand_weights_debug");
  BOOST_CHECK(rand_weights != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    unsigned int n = random_uint(1000, 1500);

    value_type *weights = new value_type[n];
    value_type val = random_real((value_type)0.0, (value_type)1.0);
    value_type threshold = random_real((value_type)0.1, (value_type)0.9);

    // generate the weights:
    rand_weights(weights, threshold, n, val);

    value_type zeros = 0.0;
    value_type nzeros = 0.0;

    // Now compate the hx arrays:
    for (unsigned int j = 0; j < n; ++j)
    {
      if (weights[j] == 0.0)
      {
        zeros += 1.0;
      }
      else {
        nzeros += 1.0;
      };

      value_type v1 = weights[j];
      value_type v2 = (abs(sin(j))<=threshold) ? val : 0.0;

      BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch on weight element " << j << ": " << v1 << "!=" << v2);
    }

   BOOST_CHECK(zeros>0.0);
   BOOST_CHECK(nzeros>0.0);

    delete [] weights;
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
