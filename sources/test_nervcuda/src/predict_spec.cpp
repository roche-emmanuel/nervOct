#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

#include <nervcuda.h>
#include <nerv/TrainingSet.h>
#include <GradientDescent.h>
#include <windows.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <boost/chrono.hpp>

using namespace nerv;


BOOST_AUTO_TEST_SUITE( gradient_descent_suite )

BOOST_AUTO_TEST_CASE( test_nn_predict )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*PredictFunc)(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, value_type* params, value_type* X, value_type* hx, value_type bias);

  // We should be able to retrieve the train function:
  PredictFunc nn_predict = (PredictFunc) GetProcAddress(h, "nn_predict");
  BOOST_CHECK(nn_predict != nullptr);

  PredictFunc nn_predict_cpu = (PredictFunc) GetProcAddress(h, "nn_predict_cpu");
  BOOST_CHECK(nn_predict_cpu != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    // Prepare a random training set:
    TrainingSet<value_type> tr(3, 5, 3, 6, 500, 1000);

    value_type bias = tr.random_real(0.0,1.0);

    // Prepare the matrices for hx and pred_hx:
    unsigned int ny = tr.y_train_size();
    value_type* hx = tr.createArray(ny);
    value_type* pred_hx = tr.createArray(ny);

    // Now compute the predictions:
    nn_predict(tr.nl(),tr.lsizes(),tr.nsamples(),tr.params(),tr.X_train(),hx,bias);
    nn_predict_cpu(tr.nl(),tr.lsizes(),tr.nsamples(),tr.params(),tr.X_train(),pred_hx,bias);

    // Now compate the hx arrays:
    for (unsigned int j = 0; j < ny; ++j)
    {
      value_type v1 = hx[j];
      value_type v2 = pred_hx[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 2*epsilon, "Mismatch on hx element " << j << ": " << v1 << "!=" << v2);
    }
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_nn_predict_float )
{
  typedef float value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*PredictFunc)(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, value_type* params, value_type* X, value_type* hx, value_type bias);

  // We should be able to retrieve the train function:
  PredictFunc nn_predict = (PredictFunc) GetProcAddress(h, "nn_predict_f");
  BOOST_CHECK(nn_predict != nullptr);

  PredictFunc nn_predict_cpu = (PredictFunc) GetProcAddress(h, "nn_predict_cpu_f");
  BOOST_CHECK(nn_predict_cpu != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    // Prepare a random training set:
    TrainingSet<value_type> tr(3, 5, 3, 6, 500, 1000);

    value_type bias = tr.random_real(0.0,1.0);

    // Prepare the matrices for hx and pred_hx:
    unsigned int ny = tr.y_train_size();
    value_type* hx = tr.createArray(ny);
    value_type* pred_hx = tr.createArray(ny);

    // Now compute the predictions:
    nn_predict(tr.nl(),tr.lsizes(),tr.nsamples(),tr.params(),tr.X_train(),hx,bias);
    nn_predict_cpu(tr.nl(),tr.lsizes(),tr.nsamples(),tr.params(),tr.X_train(),pred_hx,bias);

    // Now compate the hx arrays:
    for (unsigned int j = 0; j < ny; ++j)
    {
      value_type v1 = hx[j];
      value_type v2 = pred_hx[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 2*epsilon, "Mismatch on hx element " << j << ": " << v1 << "!=" << v2);
    }
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
