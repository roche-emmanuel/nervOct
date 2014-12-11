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

BOOST_AUTO_TEST_SUITE( gradient_descent_suite )

BOOST_AUTO_TEST_CASE( test_nn_predict )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*PredictFunc)(BPTraits<value_type>& traits);
  // typedef void (*PredictFuncCPU)(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
  //                             value_type * params, value_type * X, value_type * hx, value_type bias, value_type * wmults);

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

    value_type bias = tr.random_real(0.0, 1.0);

    // Prepare the weight multipliers:
    value_type* wmults = tr.createArray(tr.nt());
    for(unsigned int j=0;j<tr.nt();++j) {
      wmults[j] = tr.random_real(0.0,1.0);
    }

    // Prepare the matrices for hx and pred_hx:
    unsigned int ny = tr.y_train_size();
    value_type *hx = tr.createArray(ny);
    value_type *pred_hx = tr.createArray(ny);

    // Now compute the predictions:
    BPTraits<double> traits;
    traits.nl = tr.nl();
    traits.lsizes = tr.lsizes();
    traits.nsamples_train = tr.nsamples();
    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.hx = hx;
    traits.bias = bias;
    traits.wmults = wmults;

    // nn_predict(tr.nl(), tr.lsizes(), tr.nsamples(), tr.params(), tr.X_train(), hx, bias, wmults);
    // nn_predict_cpu(tr.nl(), tr.lsizes(), tr.nsamples(), tr.params(), tr.X_train(), pred_hx, bias, wmults);

    nn_predict(traits);

    traits.hx = pred_hx;
    nn_predict_cpu(traits);

    // Now compate the hx arrays:
    for (unsigned int j = 0; j < ny; ++j)
    {
      value_type v1 = hx[j];
      value_type v2 = pred_hx[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 2 * epsilon, "Mismatch on hx element " << j << ": " << v1 << "!=" << v2);
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

  typedef void (*PredictFunc)(BPTraits<value_type>& traits);
  // typedef void (*PredictFuncCPU)(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
  //                             value_type * params, value_type * X, value_type * hx, value_type bias, value_type * wmults);

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

    value_type bias = tr.random_real(0.0, 1.0);

    // Prepare the weight multipliers:
    value_type* wmults = tr.createArray(tr.nt());
    for(unsigned int j=0;j<tr.nt();++j) {
      wmults[j] = tr.random_real(0.0,1.0);
    }
    
    // Prepare the matrices for hx and pred_hx:
    unsigned int ny = tr.y_train_size();
    value_type *hx = tr.createArray(ny);
    value_type *pred_hx = tr.createArray(ny);

    // Now compute the predictions:
    // Now compute the predictions:
    BPTraits<value_type> traits;
    traits.nl = tr.nl();
    traits.lsizes = tr.lsizes();
    traits.nsamples_train = tr.nsamples();
    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.hx = hx;
    traits.bias = bias;
    traits.wmults = wmults;

    // nn_predict(tr.nl(), tr.lsizes(), tr.nsamples(), tr.params(), tr.X_train(), hx, bias, wmults);
    // nn_predict_cpu(tr.nl(), tr.lsizes(), tr.nsamples(), tr.params(), tr.X_train(), pred_hx, bias, wmults);
    
    nn_predict(traits);
    traits.hx = pred_hx;
    nn_predict_cpu(traits);

    // Now compate the hx arrays:
    for (unsigned int j = 0; j < ny; ++j)
    {
      value_type v1 = hx[j];
      value_type v2 = pred_hx[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 2 * epsilon, "Mismatch on hx element " << j << ": " << v1 << "!=" << v2);
    }
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_nn_predict_with_dropout )
{
  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*PredictFunc)(BPTraits<value_type>& traits);

  // We should be able to retrieve the train function:
  PredictFunc nn_predict = (PredictFunc) GetProcAddress(h, "nn_predict");
  BOOST_CHECK(nn_predict != nullptr);

  PredictFunc nn_predict_cpu = (PredictFunc) GetProcAddress(h, "nn_predict_cpu");
  BOOST_CHECK(nn_predict_cpu != nullptr);

  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {
    // Prepare a random training set:
    TrainingSet<value_type> tr(std::vector<unsigned int>{100,100,100,32},1000,TrainingSet<value_type>::TRAIN_DEBUG);
    // TrainingSet<value_type> tr(std::vector<unsigned int>{50,30,20,10},1000,TrainingSet<value_type>::TRAIN_DEBUG);
    // TrainingSet<value_type> tr(std::vector<unsigned int>{5,4,3},10,TrainingSet<value_type>::TRAIN_DEBUG);

    value_type bias = tr.random_real(0.0, 1.0);

    // Prepare the matrices for hx and pred_hx:
    unsigned int ny = tr.y_train_size();
    value_type *hx = tr.createArray(ny);
    value_type *pred_hx = tr.createArray(ny);

    // Prepare the weight multipliers:
    value_type* dropouts = tr.createArray(tr.nt());
    for(unsigned int j=0;j<tr.nt();++j) {
      dropouts[j] = tr.random_real(0.0,1.0);
    }

    // dropouts[0] = 1.0;
    // dropouts[1] = 0.9;

    // Now compute the predictions:
    BPTraits<double> traits;
    traits.nl = tr.nl();
    traits.lsizes = tr.lsizes();
    traits.nsamples_train = tr.nsamples();
    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.inputs = tr.createArray(traits.nd());
    traits.bias = bias;
    traits.dropouts = dropouts;
    traits.hx = hx;

    // Ensure that we perform prediction with debug mode activated:
    traits.debug = true;

    nn_predict(traits);

    // Now compute the cpu version:
    traits.hx = pred_hx;
    nn_predict_cpu(traits);

    // Now compare the hx arrays:
    for (unsigned int j = 0; j < ny; ++j)
    {
      value_type v1 = hx[j];
      value_type v2 = pred_hx[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) <= 100 * epsilon, "Mismatch (with dropout) on hx element " << j << ": " << v1 << "!=" << v2);
    }
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
