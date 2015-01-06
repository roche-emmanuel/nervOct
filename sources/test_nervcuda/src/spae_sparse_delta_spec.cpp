#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

#include <nervcuda.h>
#include <windows.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <boost/chrono.hpp>

using namespace nerv;

BOOST_AUTO_TEST_SUITE( spae_sparse_delta_suite )

BOOST_AUTO_TEST_CASE( test_spae_sparse_delta )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*MultFunc)(value_type* delta, value_type* rho, value_type beta, value_type sp, unsigned int n);

  // We should be able to retrieve the train function:
  MultFunc spae_sparse_delta = (MultFunc) GetProcAddress(h, "spae_sparse_delta");
  BOOST_CHECK(spae_sparse_delta != nullptr);

  MultFunc spae_sparse_delta_cpu = (MultFunc) GetProcAddress(h, "spae_sparse_delta_cpu");
  BOOST_CHECK(spae_sparse_delta_cpu != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    // Parepare a matrix and a vector:
    unsigned int nrows = random_uint(10, 30);

    value_type beta = random_real(0.01,0.99);
    value_type sp = random_real(0.01,0.99);

    value_type *rho = new value_type[nrows];
    value_type *delta = new value_type[nrows];
    value_type *delta_pred = new value_type[nrows];

    for (unsigned int j = 0; j < nrows; ++j)
    {
      rho[j] = random_real(0.0, 1.0);
    }

    spae_sparse_delta(delta, rho, beta, sp, nrows);
    spae_sparse_delta_cpu(delta_pred, rho, beta, sp, nrows);

    // Compare the delta vectors:
    for (unsigned int j = 0; j < nrows; ++j)
    {
      value_type v1 = delta[j];
      value_type v2 = delta_pred[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 50 * epsilon, "Mismatch on delta element " << j << ": " << v1 << "!=" << v2);
    }

    delete [] rho;
    delete [] delta;
    delete [] delta_pred;
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
