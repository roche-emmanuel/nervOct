#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <iomanip>

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
  typedef void (*CostFunc)(BPTraits<double> &traits);

  // typedef void (*CostFuncCPU)(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
  //                             double * nn_params, double * X, double * yy, double lambda, double * activation, unsigned int ninputs, double * inputs, double & J, double * gradients, double * deltas);

  // We should be able to retrieve the train function:
  CostFunc costfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(costfunc != nullptr);
  // CostFuncCPU costfunc_cpu = (CostFuncCPU) GetProcAddress(h, "gd_errfunc_cpu");
  CostFunc costfunc_cpu = (CostFunc) GetProcAddress(h, "gd_errfunc_cpu");
  BOOST_CHECK(costfunc_cpu != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {

    TrainingSet<double> tr(3, 5, 3, 6, 500, 1000);

    // unsigned int np = tr.np();
    unsigned int *lsizes = tr.lsizes();
    unsigned int nsamples = tr.nsamples();
    double lambda = tr.lambda();
    unsigned int nl = tr.nl();
    unsigned int nt = tr.nt();

    BPTraits<double> traits;
    traits.nl = nl;
    traits.lsizes = lsizes;
    traits.nsamples_train = nsamples;

    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.yy = tr.y_train();
    traits.lambda = lambda;
    traits.compute_cost = true; // Note that this is disabled by default.

    unsigned int np = traits.np();
    unsigned int nd = traits.nd();

    // prepare the output gradient array:
    double *grads = tr.createArray(np);
    double *pred_grads = tr.createArray(np);

    double *inputs = tr.createArray(nd);
    double *pred_input = tr.createArray(nd);

    double *deltas = tr.createArray(nd);
    double *pred_deltas = tr.createArray(nd);

    cudaDeviceSynchronize();

    // Now we call the cost function method:
    traits.grads = grads;
    traits.deltas = deltas;
    traits.inputs = inputs;

    // costfunc(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, J, grads, deltas, inputs);
    costfunc(traits);
    double J = traits.cost;

    // And we call the same on the CPU:
    traits.deltas = pred_deltas;
    traits.inputs = pred_input;
    traits.grads = pred_grads;

    // costfunc_cpu(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, pred_act, input_size, pred_input, pred_J, pred_grads, pred_deltas);
    costfunc_cpu(traits);
    double pred_J = traits.cost;

    BOOST_CHECK_MESSAGE(abs(J - pred_J) < 1e-10, "Mismatch in J value: " << J << "!=" << pred_J);

    // Compare the content of the input array:
    for (unsigned int j = 0; j < nd; ++j)
    {
      double v1 = inputs[j];
      double v2 = pred_input[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on inputs element " << j << ": " << v1 << "!=" << v2);
    }

    // Also compare the delta arrays:
    for (unsigned int j = 0; j < nd; ++j)
    {
      double v1 = deltas[j];
      double v2 = pred_deltas[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on deltas element " << j << ": " << v1 << "!=" << v2);
    }

    // Compare the grads arrays:
    for (unsigned int j = 0; j < np; ++j)
    {
      double v1 = grads[j];
      double v2 = pred_grads[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on gradient element " << j << ": " << v1 << "!=" << v2);
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
  TrainingSet<value_type> tr(3, 5, 4, 8, 70, 100);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // Create traits from that trainingset:
  GDd::Traits traits(tr);
  traits.epsilon = 0.001;
  traits.momentum = 0.995;

  // enabled early stopping:
  traits.validationWindowSize = 10;

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

BOOST_AUTO_TEST_CASE( test_compute_train_cost )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*CostFunc)(BPTraits<value_type> &traits);

  // We should be able to retrieve the train function:
  CostFunc errfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(errfunc != nullptr);

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 70, 100);
  // tr.maxiter(-1); // no limit on maximum number of iterations.

  // Create traits from that trainingset:
  GDd::Traits gtraits(tr);
  gtraits.lambda = 0.1;
  // traits.epsilon = 0.001;
  // traits.momentum = 0.995;

  // enabled early stopping:
  // traits.validationWindowSize=10;

  // create gradient descent and run:
  GDd gd(gtraits);

  // compute train cost:
  value_type Jtrain = gd.computeTrainCost();
  // gtraits.compute_cost = true;
  // gtraits.compute_grads = false;
  // errfunc(gtraits);
  // value_type Jtrain = gtraits.cost;

  // Compute the prediction:
  BPTraits<value_type> traits;
  traits.nl = tr.nl();
  traits.lsizes = tr.lsizes();
  traits.nsamples_train = tr.nsamples();
  traits.params = tr.params();
  traits.X = tr.X_train();
  traits.yy = tr.y_train();
  traits.lambda = 0.0;
  traits.compute_cost = true;
  traits.compute_grads = false;

  errfunc(traits);

  BOOST_CHECK_MESSAGE(abs(traits.cost - Jtrain) <= epsilon, "Mismatch in J train values:" << traits.cost << "!=" << Jtrain);
  BOOST_CHECK(FreeLibrary(h));

}

BOOST_AUTO_TEST_CASE( test_compute_cv_cost )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*CostFunc)(BPTraits<value_type> &traits);

  // We should be able to retrieve the train function:
  CostFunc errfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(errfunc != nullptr);

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 70, 100);
  // tr.maxiter(-1); // no limit on maximum number of iterations.

  // Create traits from that trainingset:
  GDd::Traits gtraits(tr);
  gtraits.lambda = 0.1;
  // traits.epsilon = 0.001;
  // traits.momentum = 0.995;

  // enabled early stopping:
  // traits.validationWindowSize=10;

  // create gradient descent and run:
  GDd gd(gtraits);

  // compute train cost:
  value_type J = gd.computeCvCost();
  // gtraits.compute_cost = true;
  // gtraits.compute_grads = false;
  // errfunc(gtraits);
  // value_type J = gtraits.cost;

  // Compute the prediction:
  BPTraits<value_type> traits;
  traits.nl = tr.nl();
  traits.lsizes = tr.lsizes();
  traits.nsamples_train = tr.X_cv_size() / tr.lsizes()[0];
  traits.params = tr.params();
  traits.X = tr.X_cv();
  traits.yy = tr.y_cv();
  traits.lambda = 0.0;
  traits.compute_cost = true;
  traits.compute_grads = false;

  errfunc(traits);

  BOOST_CHECK_MESSAGE(abs(traits.cost - J) <= epsilon, "Mismatch in J cv values:" << traits.cost << "!=" << J);
  BOOST_CHECK(FreeLibrary(h));

}

BOOST_AUTO_TEST_CASE( test_gd_errfunc_dropout )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  typedef void (*CostFunc)(BPTraits<value_type> &traits);

  // We should be able to retrieve the train function:
  CostFunc costfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(costfunc != nullptr);
  // CostFuncCPU costfunc_cpu = (CostFuncCPU) GetProcAddress(h, "gd_errfunc_cpu");
  CostFunc costfunc_cpu = (CostFunc) GetProcAddress(h, "gd_errfunc_cpu");
  BOOST_CHECK(costfunc_cpu != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {

    TrainingSet<value_type> tr(3, 5, 3, 6, 500, 1000);

    // unsigned int np = tr.np();
    unsigned int *lsizes = tr.lsizes();
    unsigned int nsamples = tr.nsamples();
    value_type lambda = tr.lambda();
    unsigned int nl = tr.nl();
    unsigned int nt = tr.nt();

    BPTraits<value_type> traits;
    traits.nl = nl;
    traits.lsizes = lsizes;
    traits.nsamples_train = nsamples;

    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.yy = tr.y_train();
    traits.lambda = lambda;
    traits.compute_cost = true; // Note that this is disabled by default.

    // Also inject some random bias:
    traits.bias = tr.random_real(0.0, 1.0);

    unsigned int np = traits.np();
    unsigned int nd = traits.nd();

    // Prepare an array to contain the dropout values:
    value_type *dropouts = tr.createArray(nt);
    for (unsigned int j = 0; j < tr.nt(); ++j)
    {
      dropouts[j] = tr.random_real(0.0, 1.0);
    }

    // dropouts[0] = 0.5;
    // dropouts[1] = 0.9;

    traits.dropouts = dropouts;

    // ensure that we use debug mode for the dropout computation:
    traits.debug = true;

    // prepare the output gradient array:
    value_type *grads = tr.createArray(np);
    value_type *pred_grads = tr.createArray(np);

    value_type *inputs = tr.createArray(nd);
    value_type *pred_input = tr.createArray(nd);

    value_type *deltas = tr.createArray(nd);
    value_type *pred_deltas = tr.createArray(nd);

    cudaDeviceSynchronize();

    // Now we call the cost function method:
    traits.grads = grads;
    traits.deltas = deltas;
    traits.inputs = inputs;

    // costfunc(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, J, grads, deltas, inputs);
    costfunc(traits);
    value_type J = traits.cost;

    // And we call the same on the CPU:
    traits.deltas = pred_deltas;
    traits.inputs = pred_input;
    traits.grads = pred_grads;

    // costfunc_cpu(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, pred_act, input_size, pred_input, pred_J, pred_grads, pred_deltas);
    costfunc_cpu(traits);
    value_type pred_J = traits.cost;

    BOOST_CHECK_MESSAGE(abs(J - pred_J) < 1e-10, "Mismatch in J value: " << std::setprecision(16)  << J << "!=" << pred_J);

    // Compare the content of the input array:
    for (unsigned int j = 0; j < nd; ++j)
    {
      value_type v1 = inputs[j];
      value_type v2 = pred_input[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 10 * epsilon, "Mismatch on inputs element " << j << ": " << v1 << "!=" << v2);
    }

    // Also compare the delta arrays:
    for (unsigned int j = 0; j < nd; ++j)
    {
      value_type v1 = deltas[j];
      value_type v2 = pred_deltas[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 10 * epsilon, "Mismatch on deltas element " << j << ": " << v1 << "!=" << v2);
    }

    // Compare the grads arrays:
    for (unsigned int j = 0; j < np; ++j)
    {
      value_type v1 = grads[j];
      value_type v2 = pred_grads[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 10 * epsilon, "Mismatch on gradient element " << j << ": " << v1 << "!=" << v2);
    }
  }

  BOOST_CHECK(FreeLibrary(h));
}


BOOST_AUTO_TEST_CASE( test_gd_with_dropout )
{
  srand((unsigned int)time(nullptr));

  typedef float value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 100, 150, 1000, 2000, 3, 128);

  tr.maxiter(-1); // no limit on maximum number of iterations.

  // We make a copy of the initial parameters
  // So that we can test with the same initial settings the
  // behavior with and without dropout:
  unsigned int np = tr.np();
  value_type *params2 = tr.createArray(np);
  memcpy(params2, tr.params(), np * sizeof(value_type));
  value_type *params3 = tr.createArray(np);
  memcpy(params3, tr.params(), np * sizeof(value_type));

  // Create traits from that trainingset:
  GDf::Traits traits(tr);
  traits.epsilon = (value_type)0.001;
  traits.momentum = (value_type)0.995;

  // enabled early stopping:
  traits.validationWindowSize = 100;
  traits.miniBatchSize = 32;

  // create gradient descent and run:
  GDf gd(traits);

  // try to run the gradient descent:
  gd.run();

  // compute the cost on train and cv datasets:
  value_type Jcv1 = gd.computeCvCost();
  logDEBUG("Final cv cost is " << Jcv1);

  // Now sue the same process but with dropout:
  value_type *dropouts = tr.createArray(tr.nt());
  for (unsigned int j = 0; j < tr.nt(); ++j)
  {
    dropouts[j] = 1.0;
  }

  traits.dropouts = dropouts;
  traits.params = params2;

  // create another gradient descent and run:
  GDf gd_ndrop(traits);

  // try to run the gradient descent:
  gd_ndrop.run();

  value_type Jcv2 = gd_ndrop.computeCvCost();
  logDEBUG("Final cv cost without dropout is " << Jcv2);

  // There was no actual dropout applied so we expect exactly the same result:
  BOOST_CHECK_MESSAGE(abs(Jcv2 - Jcv1) <= epsilon, "Mismatch in cv cost when using trivial dropout:" << Jcv2 << "!=" << Jcv1);

  for (unsigned int j = 0; j < tr.nt(); ++j)
  {
    dropouts[j] = 0.5;
  }

  dropouts[0] = (value_type)0.8; // 20 % dropouts on inputs.

  traits.dropouts = dropouts;
  traits.params = params3;

  // create another gradient descent and run:
  GDf gd_drop(traits);

  // try to run the gradient descent:
  gd_drop.run();

  value_type Jcv3 = gd_drop.computeCvCost();
  logDEBUG("Final cv cost with dropout is " << Jcv3);

  BOOST_CHECK_MESSAGE(Jcv3 < Jcv1, "No improvement in cv cost when using dropout:" << Jcv3 << ">=" << Jcv1);
}

BOOST_AUTO_TEST_CASE( test_run_gradient_descent )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  typedef int (*RunFunc)(BPTraits<value_type> &traits);

  // We should be able to retrieve the train function:
  RunFunc runfunc = (RunFunc) GetProcAddress(h, "run_gradient_descent");
  BOOST_CHECK(runfunc != nullptr);

  typedef void (*CostFunc)(BPTraits<value_type> &traits);

  // We should be able to retrieve the train function:
  CostFunc errfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(errfunc != nullptr);

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 3, 6, 200, 300);
  tr.maxiter(-1); // no limit on maximum number of iterations.

  // Create traits from that trainingset:
  GDTraits<value_type> traits(tr);

  traits.epsilon = 0.001;
  traits.momentum = 0.995;
  traits.bias = 0.0;

  // enabled early stopping:
  traits.validationWindowSize = 10;
  traits.miniBatchSize = 10;

  traits.compute_cost = true;
  traits.compute_grads = false;
  errfunc(traits);
  value_type cost1 = traits.cost;

  int res = run_gradient_descent(traits);
  BOOST_CHECK(res == GD_SUCCESS);

  // Check that the gradient descent was run properly:
  traits.compute_cost = true;
  traits.compute_grads = false;
  errfunc(traits);
  value_type cost2 = traits.cost;

  BOOST_CHECK_MESSAGE(cost2 < cost1, "No improvement in train cost:" << cost2 << ">=" << cost1);

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_compute_cv_cost_final )
{
  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 70, 100);
  // tr.maxiter(-1); // no limit on maximum number of iterations.

  value_type *dropouts = tr.createArray(tr.nt());
  for (unsigned int j = 0; j < tr.nt(); ++j)
  {
    dropouts[j] = tr.random_real(0.0, 1.0);
  }

  // dropouts[0] = 0.5;
  // dropouts[1] = 0.9;

  // Create traits from that trainingset:
  GDd::Traits gtraits(tr);
  gtraits.lambda = 0.1;
  gtraits.epsilon = 0.001;
  gtraits.momentum = 0.995;

  gtraits.dropouts = dropouts;

  // enabled early stopping:
  gtraits.validationWindowSize = 10;

  // ask the gradient descent to compute the final Jcv value:
  // gtraits.compute_cost = true;

  // create gradient descent and run:
  GDd gd(gtraits);
  value_type cost = gd.run();

  // compute cv cost:
  value_type J = gd.computeCvCost();

  BOOST_CHECK_MESSAGE(abs(cost - J) <= 1e-10, "Mismatch in J cv values:" << std::setprecision(16) << cost << "!=" << J);
}

BOOST_AUTO_TEST_CASE( test_gd_errfunc_vs_costfunc )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*OldCostFunc)(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
                              double * nn_params, double * X, double * yy, double lambda, double & J, double * gradients, double * deltas, double * inputs);

  OldCostFunc old_costfunc = (OldCostFunc) GetProcAddress(h, "costFunc");
  BOOST_CHECK(old_costfunc != nullptr);

  typedef void (*CostFunc)(BPTraits<double> &traits);

  // We should be able to retrieve the train function:
  CostFunc costfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(costfunc != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {

    TrainingSet<double> tr(3, 5, 3, 6, 500, 1000);

    // unsigned int np = tr.np();
    unsigned int *lsizes = tr.lsizes();
    unsigned int nsamples = tr.nsamples();
    double lambda = tr.lambda();
    unsigned int nl = tr.nl();
    unsigned int nt = tr.nt();

    BPTraits<double> traits;
    traits.nl = nl;
    traits.lsizes = lsizes;
    traits.nsamples_train = nsamples;

    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.yy = tr.y_train();
    traits.lambda = lambda;
    traits.compute_cost = true; // Note that this is disabled by default.

    unsigned int np = traits.np();
    unsigned int nd = traits.nd();

    // prepare the output gradient array:
    double *grads = tr.createArray(np);
    double *pred_grads = tr.createArray(np);

    double *inputs = tr.createArray(nd);
    double *pred_input = tr.createArray(nd);

    double *deltas = tr.createArray(nd);
    double *pred_deltas = tr.createArray(nd);

    cudaDeviceSynchronize();

    // Now we call the cost function method:
    traits.grads = grads;
    traits.deltas = deltas;
    traits.inputs = inputs;

    // costfunc(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, J, grads, deltas, inputs);
    costfunc(traits);
    double J = traits.cost;

    // And we call the same on the CPU:
    traits.deltas = pred_deltas;
    traits.inputs = pred_input;
    traits.grads = pred_grads;

    double pred_J = 0.0;
    old_costfunc(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, pred_J, pred_grads, pred_deltas, pred_input);
    // costfunc_cpu(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, pred_act, input_size, pred_input, pred_J, pred_grads, pred_deltas);
    // costfunc_cpu(traits);

    BOOST_CHECK_MESSAGE(abs(J - pred_J) < 1e-10, "Mismatch in J value: " << J << "!=" << pred_J);

    // Compare the content of the input array:
    for (unsigned int j = 0; j < nd; ++j)
    {
      double v1 = inputs[j];
      double v2 = pred_input[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on inputs element " << j << ": " << v1 << "!=" << v2);
    }

    // Also compare the delta arrays:
    for (unsigned int j = 0; j < nd; ++j)
    {
      double v1 = deltas[j];
      double v2 = pred_deltas[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on deltas element " << j << ": " << v1 << "!=" << v2);
    }

    // Compare the grads arrays:
    for (unsigned int j = 0; j < np; ++j)
    {
      double v1 = grads[j];
      double v2 = pred_grads[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 1e-10, "Mismatch on gradient element " << j << ": " << v1 << "!=" << v2);
    }
  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_final_cv_cost_consistency )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef void (*CostFunc)(BPTraits<double> &traits);

  // We should be able to retrieve the train function:
  CostFunc costfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(costfunc != nullptr);


  srand((unsigned int)time(nullptr));

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();
  // logDEBUG("Epsilon value is: "<<epsilon)

  // prepare a dataset:
  TrainingSet<value_type> tr(3, 5, 4, 8, 70, 100);
  // tr.maxiter(-1); // no limit on maximum number of iterations.

  value_type *dropouts = tr.createArray(tr.nt());
  for (unsigned int j = 0; j < tr.nt(); ++j)
  {
    dropouts[j] = tr.random_real(0.0, 1.0);
  }

  // dropouts[0] = 0.5;
  // dropouts[1] = 0.9;

  // Create traits from that trainingset:
  GDd::Traits gtraits(tr);
  gtraits.lambda = 0.1;
  gtraits.epsilon = 0.001;
  gtraits.momentum = 0.995;

  // gtraits.dropouts = dropouts;

  // enabled early stopping:
  gtraits.validationWindowSize = 10;

  // ask the gradient descent to compute the final Jcv value:
  // gtraits.compute_cost = true;

  // create gradient descent and run:
  GDd gd(gtraits);
  value_type cost = gd.run();


  // Recompute the cost with the final parameters:
  // unsigned int np = tr.np();
  unsigned int *lsizes = tr.lsizes();
  unsigned int nsamples = tr.nsamples_cv();
  value_type lambda = tr.lambda();
  unsigned int nl = tr.nl();
  unsigned int nt = tr.nt();

  BPTraits<value_type> traits;
  traits.nl = nl;
  traits.lsizes = lsizes;
  traits.nsamples_train = nsamples;

  traits.params = tr.params();
  traits.X = tr.X_cv();
  traits.yy = tr.y_cv();
  traits.lambda = 0.0;
  traits.compute_cost = true; // Note that this is disabled by default.

  unsigned int np = traits.np();
  unsigned int nd = traits.nd();

  // prepare the output gradient array:
  traits.grads = tr.createArray(np);
  traits.inputs = tr.createArray(nd);
  traits.deltas = tr.createArray(nd);

  costfunc(traits);
  value_type J = traits.cost;

  BOOST_CHECK_MESSAGE(abs(cost - J) <= 1e-10, "Mismatch in J cv values:" << std::setprecision(16) << cost << "!=" << J);
}

BOOST_AUTO_TEST_CASE( test_gd_errfunc_dropout_with_softmax )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");
  BOOST_CHECK(h != nullptr);

  typedef double value_type;
  value_type epsilon = std::numeric_limits<value_type>::epsilon();

  typedef void (*CostFunc)(BPTraits<value_type> &traits);

  // We should be able to retrieve the train function:
  CostFunc costfunc = (CostFunc) GetProcAddress(h, "gd_errfunc");
  BOOST_CHECK(costfunc != nullptr);
  // CostFuncCPU costfunc_cpu = (CostFuncCPU) GetProcAddress(h, "gd_errfunc_cpu");
  CostFunc costfunc_cpu = (CostFunc) GetProcAddress(h, "gd_errfunc_cpu");
  BOOST_CHECK(costfunc_cpu != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  for (unsigned int i = 0; i < num; ++i)
  {

    TrainingSet<value_type> tr(3, 5, 3, 6, 500, 1000);

    // unsigned int np = tr.np();
    unsigned int *lsizes = tr.lsizes();
    unsigned int nsamples = tr.nsamples();
    value_type lambda = tr.lambda();
    unsigned int nl = tr.nl();
    unsigned int nt = tr.nt();

    BPTraits<value_type> traits;
    traits.nl = nl;
    traits.lsizes = lsizes;
    traits.nsamples_train = nsamples;

    traits.params = tr.params();
    traits.X = tr.X_train();
    traits.yy = tr.y_train();
    traits.lambda = lambda;
    traits.use_softmax = true;
    traits.compute_cost = true; // Note that this is disabled by default.

    // Also inject some random bias:
    traits.bias = tr.random_real(0.0, 1.0);

    unsigned int np = traits.np();
    unsigned int nd = traits.nd();

    // Prepare an array to contain the dropout values:
    value_type *dropouts = tr.createArray(nt);
    for (unsigned int j = 0; j < tr.nt(); ++j)
    {
      dropouts[j] = tr.random_real(0.0, 1.0);
    }

    // dropouts[0] = 0.5;
    // dropouts[1] = 0.9;

    traits.dropouts = dropouts;

    // ensure that we use debug mode for the dropout computation:
    traits.debug = true;

    // prepare the output gradient array:
    value_type *grads = tr.createArray(np);
    value_type *pred_grads = tr.createArray(np);

    value_type *inputs = tr.createArray(nd);
    value_type *pred_input = tr.createArray(nd);

    value_type *deltas = tr.createArray(nd);
    value_type *pred_deltas = tr.createArray(nd);

    cudaDeviceSynchronize();

    // Now we call the cost function method:
    traits.grads = grads;
    traits.deltas = deltas;
    traits.inputs = inputs;

    // costfunc(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, J, grads, deltas, inputs);
    costfunc(traits);
    value_type J = traits.cost;

    // And we call the same on the CPU:
    traits.deltas = pred_deltas;
    traits.inputs = pred_input;
    traits.grads = pred_grads;

    // costfunc_cpu(nl, lsizes, nsamples, tr.params(), tr.X_train(), tr.y_train(), lambda, pred_act, input_size, pred_input, pred_J, pred_grads, pred_deltas);
    costfunc_cpu(traits);
    value_type pred_J = traits.cost;

    BOOST_CHECK_MESSAGE(abs(J - pred_J) < 1e-10, "Mismatch in J value: " << std::setprecision(16)  << J << "!=" << pred_J);

    // Compare the content of the input array:
    for (unsigned int j = 0; j < nd; ++j)
    {
      value_type v1 = inputs[j];
      value_type v2 = pred_input[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 10 * epsilon, "Mismatch on inputs element " << j << ": " << v1 << "!=" << v2);
    }

    // Also compare the delta arrays:
    for (unsigned int j = 0; j < nd; ++j)
    {
      value_type v1 = deltas[j];
      value_type v2 = pred_deltas[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 10 * epsilon, "Mismatch on deltas element " << j << ": " << v1 << "!=" << v2);
    }

    // Compare the grads arrays:
    for (unsigned int j = 0; j < np; ++j)
    {
      value_type v1 = grads[j];
      value_type v2 = pred_grads[j];
      BOOST_CHECK_MESSAGE(abs(v1 - v2) < 10 * epsilon, "Mismatch on gradient element " << j << ": " << v1 << "!=" << v2);
    }

    // Check that for each column in hx the sum of the predictions is close to 1.0:
    unsigned int nout = traits.lsizes[traits.nl-1];
    unsigned int input_offset = nd - nsamples*nout;
    value_type* hx = inputs + input_offset;

    value_type tval;
    for (unsigned int j = 1; j < traits.nsamples_train; ++j)
    {
      tval = 0.0;
      for(unsigned int r=0;r<nout;++r) {
        tval += hx[nout*j+r];
      }
      BOOST_CHECK_MESSAGE(abs(tval - 1.0) <= 10 * epsilon, "Invalid probability sum for sample " << j << ": " << tval << "!=" << 1.0);
    }    

  }

  BOOST_CHECK(FreeLibrary(h));
}


BOOST_AUTO_TEST_SUITE_END()
