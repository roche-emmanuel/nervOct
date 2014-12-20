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

BOOST_AUTO_TEST_SUITE( mat_vec_mult_suite )

BOOST_AUTO_TEST_CASE( test_mat_vec_mult )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*MultFunc)(unsigned int nrows, unsigned int ncols, value_type * A, value_type * x, value_type * y, bool tpA);

  // We should be able to retrieve the train function:
  MultFunc mat_mult = (MultFunc) GetProcAddress(h, "mat_vec_mult");
  BOOST_CHECK(mat_mult != nullptr);

  MultFunc mat_mult_cpu = (MultFunc) GetProcAddress(h, "mat_vec_mult_cpu");
  BOOST_CHECK(mat_mult_cpu != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    // Parepare a matrix and a vector:
    unsigned int nrows = random_uint(10, 30);
    unsigned int ncols = random_uint(10, 30);
    unsigned int dotp = random_uint(0, 1);

    unsigned int count = nrows * ncols;
    value_type *A = new value_type[count];
    value_type *x = new value_type[ncols];
    value_type *x_pred = new value_type[ncols];
    value_type *y = new value_type[nrows];
    value_type *y_pred = new value_type[nrows];

    for (unsigned int j = 0; j < count; ++j)
    {
      A[j] = random_real(0.0, 1.0);
    }
    for (unsigned int j = 0; j < ncols; ++j)
    {
      x[j] = random_real(0.0, 1.0);
    }
    for (unsigned int j = 0; j < nrows; ++j)
    {
      y[j] = random_real(0.0, 1.0);
    }

    if (dotp)
    {
      logDEBUG("Testing with transpose...");
      mat_mult(nrows, ncols, A, y, x, true);
      mat_mult_cpu(nrows, ncols, A, y, x_pred, true);
    }
    else
    {
      logDEBUG("Testing without transpose...");
      mat_mult(nrows, ncols, A, x, y, false);
      mat_mult_cpu(nrows, ncols, A, x, y_pred, false);
    }

    if (dotp)
    {
      // Compare the X vectors:
      for (unsigned int j = 0; j < ncols; ++j)
      {
        value_type v1 = x[j];
        value_type v2 = x_pred[j];
        BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 50 * epsilon, "Mismatch on x element " << j << ": " << v1 << "!=" << v2);
      }
    }
    else
    {
      // Compare the Y vectors:
      for (unsigned int j = 0; j < nrows; ++j)
      {
        value_type v1 = y[j];
        value_type v2 = y_pred[j];
        BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 50 * epsilon, "Mismatch on y element " << j << ": " << v1 << "!=" << v2);
      }
    }

    delete [] A;
    delete [] x;
    delete [] x_pred;
    delete [] y;
    delete [] y_pred;
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
