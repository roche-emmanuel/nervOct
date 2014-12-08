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

BOOST_AUTO_TEST_CASE( test_create_gd_traits )
{
  GDd::Traits traits;

  // check the default values:
  BOOST_CHECK(traits.nl == 0);
  BOOST_CHECK(traits.lsizes == nullptr);
  BOOST_CHECK(traits.nsamples_train == 0);
  BOOST_CHECK(traits.nparams == 0);
  BOOST_CHECK(traits.X == nullptr);
  BOOST_CHECK(traits.X_train_size == 0);
  BOOST_CHECK(traits.yy == nullptr);
  BOOST_CHECK(traits.y_train_size == 0);
  BOOST_CHECK(traits.params == nullptr);
  BOOST_CHECK(traits.maxiter == 0);
  BOOST_CHECK(traits.lambda == 0.0);
  BOOST_CHECK(traits.momentum == 0.0);
  BOOST_CHECK(traits.epsilon == 0.0);
}

BOOST_AUTO_TEST_CASE( test_create_gd )
{
  typedef GDd::value_type value_t;
  GDd::Traits traits;

  // GDd gd(traits);

  // should throw when the traits are invalid:
  BOOST_CHECK_THROW( new GDd(traits), std::runtime_error);

  unsigned int sizes2[] = { 3, 4};
  traits.lsizes = sizes2;
  traits.nl = 2;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  unsigned int sizes[] = { 3, 4, 1};
  traits.lsizes = sizes;
  traits.nl = 3;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  unsigned int nsamples = 10;
  traits.nsamples_train = nsamples;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  value_t *params = nullptr;
  traits.params = params;
  traits.nparams = 10;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  params = new value_t[21];
  traits.params = params;
  traits.nparams = 21;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  value_t *X = nullptr;
  traits.X = X;
  traits.X_train_size = 10;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  X = new value_t[nsamples * 3];
  traits.X = X;
  traits.X_train_size = nsamples * 3;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  value_t *y = nullptr;
  traits.yy = y;
  traits.y_train_size = 5;
  BOOST_CHECK_THROW( new GDd(traits),  std::runtime_error);

  y = new value_t[nsamples * 1];
  traits.yy = y;
  traits.y_train_size = nsamples * 1;

  // If we use this call here we will have a problem because of pinned memory registration.
  // BOOST_CHECK_NO_THROW( new GDd(traits) );

  // Check that we can build on stack:
  GDd gd(traits);

  delete [] y;
  delete [] X;
  delete [] params;
}

int random_int(int mini, int maxi)
{
  return mini + (int)floor(0.5 + (maxi - mini) * (double)rand() / (double)RAND_MAX);
}

template <typename T>
T random_real(T mini, T maxi)
{
  return mini + (maxi - mini) * (T)rand() / (T)RAND_MAX;
}

BOOST_AUTO_TEST_CASE( test_training_set_default )
{
  // by default the pointers should be null:
  TrainingSet<double> tr;
  BOOST_CHECK(tr.nl() == 0);
  BOOST_CHECK(tr.lsizes() == nullptr);
  BOOST_CHECK(tr.X_train() == nullptr);
  BOOST_CHECK(tr.y_train() == nullptr);
  BOOST_CHECK(tr.params() == nullptr);
}

BOOST_AUTO_TEST_CASE( test_training_set_build_from_vector )
{
  // by default the pointers should be null:
  TrainingSet<double> tr(std::vector<unsigned int> {3, 3, 1}, 10);
  BOOST_CHECK(tr.nl() == 3);
  BOOST_CHECK(tr.lsizes() != nullptr);
  BOOST_CHECK(tr.X_train() != nullptr);
  BOOST_CHECK(tr.y_train() != nullptr);
  BOOST_CHECK(tr.params() != nullptr);
  BOOST_CHECK(tr.X_train_size() == 30);
  BOOST_CHECK(tr.y_train_size() == 10);
  BOOST_CHECK(tr.np() == 16);
}

BOOST_AUTO_TEST_CASE( test_training_set_build_random )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  logDEBUG("Epsilon value is: " << epsilon)

  // by default the pointers should be null:
  TrainingSet<value_type> tr(3, 5, 4, 6, 10, 20);
  BOOST_CHECK(3 <= tr.nl() && tr.nl() <= 5);
  BOOST_CHECK(tr.lsizes() != nullptr);
  BOOST_CHECK(tr.X_train() != nullptr);
  BOOST_CHECK(tr.y_train() != nullptr);
  BOOST_CHECK(tr.params() != nullptr);
  BOOST_CHECK(tr.X_train_size() == tr.nsamples()*tr.lsizes()[0]);
  BOOST_CHECK(tr.y_train_size() == tr.nsamples()*tr.lsizes()[tr.nt()]);

  // We should have build debug matrices so we can check the values:
  for (unsigned int i = 0; i < 100; ++i)
  {
    unsigned int index = tr.random_uint(0, tr.X_train_size() - 1);
    value_type v1 = sin(index) * 10.0;
    value_type v2 = tr.X_train()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at X element " << index << ": " << v1 << "!=" << v2);

    index = tr.random_uint(0, tr.y_train_size() - 1);
    v1 = abs(cos(index));
    v2 = tr.y_train()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at y element " << index << ": " << v1 << "!=" << v2);

    index = tr.random_uint(0, tr.np() - 1);
    v1 = sin(index + 0.5);
    v2 = tr.params()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at params element " << index << ": " << v1 << "!=" << v2);
  }
}

BOOST_AUTO_TEST_CASE( test_training_set_build_random_float )
{
  srand((unsigned int)time(nullptr));

  typedef float value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  logDEBUG("Epsilon value is: " << epsilon)

  // by default the pointers should be null:
  TrainingSet<value_type> tr(3, 5, 4, 6, 10, 20);
  BOOST_CHECK(3 <= tr.nl() && tr.nl() <= 5);
  BOOST_CHECK(tr.lsizes() != nullptr);
  BOOST_CHECK(tr.X_train() != nullptr);
  BOOST_CHECK(tr.y_train() != nullptr);
  BOOST_CHECK(tr.params() != nullptr);
  BOOST_CHECK(tr.X_train_size() == tr.nsamples()*tr.lsizes()[0]);
  BOOST_CHECK(tr.y_train_size() == tr.nsamples()*tr.lsizes()[tr.nt()]);

  // We should have build debug matrices so we can check the values:
  for (unsigned int i = 0; i < 100; ++i)
  {
    unsigned int index = tr.random_uint(0, tr.X_train_size() - 1);
    value_type v1 = (value_type)(sin(index) * 10.0);
    value_type v2 = tr.X_train()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at X element " << index << ": " << v1 << "!=" << v2);

    index = tr.random_uint(0, tr.y_train_size() - 1);
    v1 = (value_type)abs(cos(index));
    v2 = tr.y_train()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at y element " << index << ": " << v1 << "!=" << v2);

    index = tr.random_uint(0, tr.np() - 1);
    v1 = (value_type)sin(index + 0.5);
    v2 = tr.params()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at params element " << index << ": " << v1 << "!=" << v2);
  }
}

BOOST_AUTO_TEST_CASE( test_run_gd )
{
  typedef GDd::value_type value_t;

  // number of tests to run:
  unsigned int num = 5;

  value_t epsilon = std::numeric_limits<value_t>::epsilon();

  for (unsigned int i = 0; i < num; ++i)
  {
    TrainingSet<value_t> tr(3, 5, 3, 6, 50, 100);
    tr.maxiter(10);

    GDd::Traits traits(tr);

    // Check that we can build on stack:
    GDd gd(traits);

    // try to run the gradient descent:
    gd.run();
  }

}

BOOST_AUTO_TEST_CASE( test_gd_errfunc )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  // typedef void (*CostFunc)(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
  //                          double * nn_params, double * X, double * yy, double lambda, double & J, double * gradients, double * deltas, double * inputs);
  typedef void (*CostFunc)(BPTraits<double>& traits);

  typedef void (*CostFuncCPU)(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
                              double * nn_params, double * X, double * yy, double lambda, double * activation, unsigned int ninputs, double * inputs, double & J, double * gradients, double * deltas);

  // We should be able to retrieve the train function:
  CostFunc costfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(costfunc != nullptr);
  CostFuncCPU costfunc_cpu = (CostFuncCPU) GetProcAddress(h, "costFuncCPU");
  BOOST_CHECK(costfunc_cpu != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {

    TrainingSet<double> tr(3, 5, 3, 6, 500, 1000);

    unsigned int np = tr.np();
    unsigned int *lsizes = tr.lsizes();
    unsigned int nsamples = tr.nsamples();
    double lambda = tr.lambda();
    unsigned int nl = tr.nl();
    unsigned int nt = tr.nt();

    // Prepare the input array:
    unsigned int input_size = 0;
    for (unsigned int j = 0; j < nt; ++j)
    {
      input_size += lsizes[j + 1];
    }
    input_size *= nsamples;

    // Prepare the activation array:
    unsigned int act_size = 0;
    for (unsigned int j = 0; j < nl; ++j)
    {
      act_size += lsizes[j] + 1;
    }
    act_size *= nsamples;

    // also prepare an array to hold the predictions for the delta matrices:
    unsigned int nd = 0;
    for (unsigned int i = 1; i < nl; ++i)
    {
      nd += lsizes[i] * nsamples;
    }

    // prepare the output gradient array:
    double *grads = tr.createArray(np);
    double *pred_grads = tr.createArray(np);
    double *inputs = tr.createArray(input_size);
    double *pred_act = tr.createArray(act_size);
    double *pred_input = tr.createArray(input_size);
    double *deltas = tr.createArray(nd);
    double *pred_deltas = tr.createArray(nd);

    cudaDeviceSynchronize();

    // Now we call the cost function method:
    double J = 0.0;
    BPTraits<double> traits;
    traits.nl = nl;
    traits.lsizes = lsizes;
    traits.nsamples_train = nsamples;
    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.yy = tr.y_train();
    traits.lambda = lambda;
    traits.grads = grads;
    traits.deltas = deltas;
    traits.inputs = inputs;
    traits.compute_cost = true; // Note that this is disabled by default.

    // costfunc(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, J, grads, deltas, inputs);
    costfunc(traits);
    J = traits.cost;

    // And we call the same on the CPU:
    double pred_J = 0.0;
    costfunc_cpu(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, pred_act, input_size, pred_input, pred_J, pred_grads, pred_deltas);

    BOOST_CHECK_MESSAGE(abs(J - pred_J) < 1e-10, "Mismatch in J value: " << J << "!=" << pred_J);

    // Also compare the delta arrays:
    for (unsigned int j = 0; j < nd; ++j)
    {
      double v1 = deltas[j];
      double v2 = pred_deltas[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on deltas element " << j << ": " << v1 << "!=" << v2);
    }

    // Compare the grads arrays:
    // logDEBUG("Number of parameters: "<<np);
    for (unsigned int j = 0; j < np; ++j)
    {
      double v1 = grads[j];
      double v2 = pred_grads[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on gradient element " << j << ": " << v1 << "!=" << v2);
    }

    // Compare the content of the input array:
    for (unsigned int j = 0; j < input_size; ++j)
    {
      double v1 = inputs[j];
      double v2 = pred_input[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on inputs element " << j << ": " << v1 << "!=" << v2);
    }
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_training_set_cv_data )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  TrainingSet<value_type> tr0;
  BOOST_CHECK(tr0.X_cv() == nullptr);
  BOOST_CHECK(tr0.y_cv() == nullptr);
  BOOST_CHECK(tr0.X_cv_size() == 0);
  BOOST_CHECK(tr0.y_cv_size() == 0);


  // by default the pointers should be null:
  TrainingSet<value_type> tr(3, 5, 4, 6, 10, 20);
  unsigned int nx = (unsigned int)ceil(tr.nsamples() * 0.25) * tr.lsizes()[0];
  unsigned int ny = (unsigned int)ceil(tr.nsamples() * 0.25) * tr.lsizes()[tr.nt()];

  BOOST_CHECK(tr.X_cv() != nullptr);
  BOOST_CHECK(tr.y_cv() != nullptr);
  BOOST_CHECK(tr.X_cv_size() == nx);
  BOOST_CHECK(tr.y_cv_size() == ny);

  // We should have build debug matrices so we can check the values:
  for (unsigned int i = 0; i < 100; ++i)
  {
    unsigned int index = tr.random_uint(0, tr.X_cv_size() - 1);
    value_type v1 = (value_type)(sin(index + 0.5) * 10.0);
    value_type v2 = tr.X_cv()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at X_cv element " << index << ": " << v1 << "!=" << v2);

    index = tr.random_uint(0, tr.y_cv_size() - 1);
    v1 = (value_type)abs(cos(index + 0.5));
    v2 = tr.y_cv()[index];
    BOOST_CHECK_MESSAGE(abs(v1 - v2) <= epsilon, "Mismatch at y_cv element " << index << ": " << v1 << "!=" << v2);
  }
}

BOOST_AUTO_TEST_CASE( test_train_cost_reduction )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 6, 10, 20);
  tr.maxiter(20);

  // Create traits from that trainingset:
  GDd::Traits traits(tr);
  traits.epsilon = 0.001;
  traits.momentum = 0.995;

  // create gradient descent and run:
  GDd gd(traits);

  // compute initial train cost:
  value_type Jtrain0 = gd.computeTrainCost();

  // try to run the gradient descent:
  gd.run();

  // compute the cost on train and cv datasets:
  value_type Jtrain1 = gd.computeTrainCost();
  logDEBUG("Reduced training cost from " << Jtrain0 << " to " << Jtrain1);

  BOOST_CHECK_MESSAGE(Jtrain1 < Jtrain0, "No improvement in training cost:" << Jtrain1 << ">=" << Jtrain0);
}


BOOST_AUTO_TEST_CASE( test_early_stopping )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 500, 600);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // Create traits from that trainingset:
  GDd::Traits traits(tr);
  traits.epsilon = 0.001;
  traits.momentum = 0.995;

  // enabled early stopping:
  traits.validationWindowSize =10;

  // create gradient descent and run:
  GDd gd(traits);

  // compute initial train cost:
  value_type Jcv0 = gd.computeCvCost();

  // try to run the gradient descent:
  gd.run();

  // compute the cost on train and cv datasets:
  value_type Jcv1 = gd.computeCvCost();
  logDEBUG("Final cv cost is " << Jcv1);
  BOOST_CHECK_MESSAGE(Jcv1 <= Jcv0, "No improvement in cv cost:" << Jcv1 << ">=" << Jcv0);
}

BOOST_AUTO_TEST_CASE( test_early_stopping_minibatch )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 500, 600);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // Create traits from that trainingset:
  GDd::Traits traits(tr);
  traits.epsilon = 0.001;
  traits.momentum = 0.995;

  // enabled early stopping:
  traits.validationWindowSize = 10;
  traits.miniBatchSize = 10;

  // create gradient descent and run:
  GDd gd(traits);

  // compute initial train cost:
  value_type Jcv0 = gd.computeCvCost();

  // try to run the gradient descent:
  gd.run();

  // compute the cost on train and cv datasets:
  value_type Jcv1 = gd.computeCvCost();
  logDEBUG("Final cv cost is " << Jcv1);
  BOOST_CHECK_MESSAGE(Jcv1 <= Jcv0, "No improvement in cv cost:" << Jcv1 << ">=" << Jcv0);
}

BOOST_AUTO_TEST_CASE( test_zero_param )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 500, 600);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // set a parameter to zero and check its evolution:
  tr.params()[0] = 0.0;

  // Create traits from that trainingset:
  GDd::Traits traits(tr);
  traits.epsilon = 0.001;
  traits.momentum = 0.995;

  // enabled early stopping:
  traits.validationWindowSize = 10;
  traits.miniBatchSize = 10;

  // create gradient descent and run:
  GDd gd(traits);

  // try to run the gradient descent:
  gd.run();

  // Actually a value of zero will evolve just like other values
  // So we should not expect it to be still zero.
  BOOST_CHECK_MESSAGE(tr.params()[0] != 0.0, "Invalid value for parameter 0:" << tr.params()[0]);
}

BOOST_AUTO_TEST_CASE( test_same_params )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 500, 600);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // set a parameter to zero and check its evolution:
  tr.params()[0] = 0.0;
  tr.params()[1] = 0.0;

  // Create traits from that trainingset:
  GDd::Traits traits(tr);
  traits.epsilon = 0.001;
  traits.momentum = 0.995;

  // enabled early stopping:
  traits.validationWindowSize = 10;
  traits.miniBatchSize = 10;

  // create gradient descent and run:
  GDd gd(traits);

  // try to run the gradient descent:
  gd.run();

  // Even here we cannot expect the parameter values to always be the same:
  BOOST_CHECK_MESSAGE(tr.params()[0] != tr.params()[1], "Match in parameter 0 and 1 values:" << tr.params()[0] << "==" << tr.params()[1]);
}

BOOST_AUTO_TEST_CASE( test_specify_bias )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 3, 6, 200, 300);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // set a parameter to zero and check its evolution:
  tr.params()[0] = 0.0;
  tr.params()[1] = 1.0;

  // Create traits from that trainingset:
  GDd::Traits traits(tr);
  traits.epsilon = 0.001;
  traits.momentum = 0.995;
  traits.bias = 0.0;

  // enabled early stopping:
  traits.validationWindowSize = 10;
  traits.miniBatchSize = 10;

  // create gradient descent and run:
  GDd gd(traits);

  // try to run the gradient descent:
  gd.run();

  // Here we expect the weights to never be updated because the bias is used as input for
  // those 2 parameters and it was set to 0.0
  BOOST_CHECK_MESSAGE(tr.params()[0] == 0.0, "Invalid value for parameter 0:" << tr.params()[0]);
  BOOST_CHECK_MESSAGE(tr.params()[1] == 1.0, "Invalid value for parameter 1:" << tr.params()[1]);
}

BOOST_AUTO_TEST_CASE( test_early_stopping_minibatch_float )
{
  srand((unsigned int)time(nullptr));

  typedef float value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 500, 600);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // Create traits from that trainingset:
  GDf::Traits traits(tr);
  traits.epsilon = 0.001f;
  traits.momentum = 0.995f;

  // enabled early stopping:
  traits.validationWindowSize = 10;
  traits.miniBatchSize = 32;

  // create gradient descent and run:
  GDf gd(traits);

  // compute initial train cost:
  value_type Jcv0 = gd.computeCvCost();

  // try to run the gradient descent:
  gd.run();

  // compute the cost on train and cv datasets:
  value_type Jcv1 = gd.computeCvCost();
  logDEBUG("Final cv cost is " << Jcv1);
  BOOST_CHECK_MESSAGE(Jcv1 <= Jcv0, "No improvement in cv cost:" << Jcv1 << ">=" << Jcv0);
}


BOOST_AUTO_TEST_SUITE_END()
