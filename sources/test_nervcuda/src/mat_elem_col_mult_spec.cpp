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

BOOST_AUTO_TEST_SUITE( mat_elem_col_mult_suite )

BOOST_AUTO_TEST_CASE( test_mat_elem_col_mult )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*MultFunc)(unsigned int nrows, unsigned int ncols, double * A, double * y, bool div);

  // We should be able to retrieve the train function:
  MultFunc mat_mult = (MultFunc) GetProcAddress(h, "mat_elem_col_mult");
  BOOST_CHECK(mat_mult != nullptr);

  MultFunc mat_mult_cpu = (MultFunc) GetProcAddress(h, "mat_elem_col_mult_cpu");
  BOOST_CHECK(mat_mult_cpu != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    // Parepare a matrix and a vector:
    unsigned int nrows = random_uint(10, 30);
    unsigned int ncols = random_uint(10, 30);

    unsigned int count = nrows * ncols;
    value_type *A = new value_type[count];
    value_type *A_pred = new value_type[count];
    value_type *y = new value_type[ncols];

    for (unsigned int j = 0; j < count; ++j)
    {
      A[j] = random_real(0.0, 1.0);
      A_pred[j] = A[j];
    }
    for (unsigned int j = 0; j < ncols; ++j)
    {
      y[j] = random_real(0.0, 1.0);
    }

    mat_mult(nrows, ncols, A, y, false);
    mat_mult_cpu(nrows, ncols, A_pred, y, false);

    // Compare the matrices:
    for (unsigned int r = 0; r < nrows; ++r)
    {
      for (unsigned int c = 0; c < ncols; ++c)
      {
        value_type v1 = A[nrows * c + r];
        value_type v2 = A_pred[nrows * c + r];
        BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 50 * epsilon, "Mismatch on A element (" << r << ", " << c << "): " << v1 << "!=" << v2);
      }
    }

    delete [] A;
    delete [] A_pred;
    delete [] y;
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_mat_elem_col_div )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*MultFunc)(unsigned int nrows, unsigned int ncols, double * A, double * y, bool div);

  // We should be able to retrieve the train function:
  MultFunc mat_mult = (MultFunc) GetProcAddress(h, "mat_elem_col_mult");
  BOOST_CHECK(mat_mult != nullptr);

  MultFunc mat_mult_cpu = (MultFunc) GetProcAddress(h, "mat_elem_col_mult_cpu");
  BOOST_CHECK(mat_mult_cpu != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    // Parepare a matrix and a vector:
    unsigned int nrows = random_uint(10, 30);
    unsigned int ncols = random_uint(10, 30);

    unsigned int count = nrows * ncols;
    value_type *A = new value_type[count];
    value_type *A_pred = new value_type[count];
    value_type *y = new value_type[ncols];

    for (unsigned int j = 0; j < count; ++j)
    {
      A[j] = random_real(0.0, 1.0);
      A_pred[j] = A[j];
    }
    for (unsigned int j = 0; j < ncols; ++j)
    {
      y[j] = random_real(0.001, 1.0);
    }

    mat_mult(nrows, ncols, A, y, true);
    mat_mult_cpu(nrows, ncols, A_pred, y, true);

    // Compare the matrices:
    for (unsigned int r = 0; r < nrows; ++r)
    {
      for (unsigned int c = 0; c < ncols; ++c)
      {
        value_type v1 = A[nrows * c + r];
        value_type v2 = A_pred[nrows * c + r];
        BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 50 * epsilon, "Mismatch on A element (" << r << ", " << c << "): " << v1 << "!=" << v2);
      }
    }

    delete [] A;
    delete [] A_pred;
    delete [] y;
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
