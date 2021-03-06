#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <sstream>
#include <windows.h>
#include <iomanip>

#include <nerv/GDTraits.h>

// #define WITH_COST_TEST

#define CHECK(cond, msg) if(!(cond)) { \
    std::ostringstream os; \
    os << msg; \
    error(os.str().c_str()); \
    return; \
  }

#define CHECK_RET(cond, msg) if(!(cond)) { \
    std::ostringstream os; \
    os << msg; \
    error(os.str().c_str()); \
    return result;\
  }

#define logDEBUG(msg) octave_stdout << msg << std::endl;

using namespace nerv;

typedef std::map<unsigned int, double> CostMap;

class NERVManager
{
protected:
  typedef int (* RunGDFunc)(GDTraits<double> &traits);

#ifdef WITH_COST_TEST
  typedef void (*CostFunc)(unsigned int nl, unsigned int *lsizes, unsigned int nsamples,
                           double *nn_params, double *X, double *yy, double lambda, double &J, double *gradients, double *deltas, double *inputs);
#endif

public:
  NERVManager()
  {
    logDEBUG("Loading nervCUDA...");
    _h = LoadLibrary("nervCUDA.dll");
    CHECK(_h, "ERROR: cannot load nervCUDA library! err=" << GetLastError());

    // Try loading the functions of interest:
    _run_gd = (RunGDFunc) GetProcAddress(_h, "run_gradient_descent");
    CHECK(_run_gd, "ERROR: cannot find run_gradient_descent method! err=" << GetLastError());

#ifdef WITH_COST_TEST
    _costFunc = (CostFunc) GetProcAddress(_h, "costFunc");
    CHECK(_costFunc, "ERROR: cannot find costFunc method! err=" << GetLastError());
#endif

  }

  ~NERVManager()
  {
    logDEBUG("Unloading nervCUDA module...")
    BOOL res = FreeLibrary(_h);
    CHECK(res, "ERROR: cannot free library! err=" << GetLastError());
  }

  inline void run_gradient_descent(const Matrix &lsizes_mat, const Matrix &X_train, const Matrix &y_train,
                                   const Matrix &params, octave_scalar_map &desc, CostMap &costMap, double &cvCost)
  {
    // Here we can already check that the feature matrix dimensions match
    // the lsizes description:
    GDTraits<double> traits;

    // Assign the nl value:
    unsigned int nl = lsizes_mat.numel();
    unsigned int nt = nl - 1;
    traits.nl = nl;
    // logDEBUG("traits.nl = " << traits.nl)

    // Assign the lsizes value:
    unsigned int *lsizes = new unsigned int[nl];
    for (unsigned int i = 0; i < nl; ++i)
    {
      lsizes[i] = lsizes_mat(i);
    }
    traits.lsizes = lsizes;

    // Retrieve the number of train samples from the X_train matrix.
    unsigned int nsamples_train = X_train.dim1();
    traits.nsamples_train = nsamples_train;

    // Check if we have the proper number of parameters:
    // logDEBUG("Setting nparams to: "<<params.numel());
    traits.nparams = params.numel();
    traits.params = (double *)params.data();

    // Assign X train data:
    // Note that here we need to transpose the X matrix before assigning it to the traits
    traits.X_train_size = X_train.numel();
    Matrix X_train_t = X_train.transpose();
    traits.X = (double *)X_train_t.data();

    // Assign the y_train data:
    traits.y_train_size = y_train.numel();
    traits.yy = (double *)y_train.data();

    // We should also read a value for momentum if available:
    octave_value val = desc.contents("momentum");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: momentum is not a double type");
      traits.momentum = val.double_value();
      CHECK(traits.momentum >= 0.0 && traits.momentum < 1.0, "nn_gradient_descent: invalid value for momentum: " << traits.momentum);
    }

    // We should also read a value for learning rate if available:
    val = desc.contents("epsilon");
    CHECK(val.is_defined(), "nn_gradient_descent: epsilon value is not defined");
    CHECK(val.is_double_type(), "nn_gradient_descent: epsilon is not a double type");
    traits.epsilon = val.double_value();
    CHECK(traits.epsilon > 0.0 && traits.epsilon < 1.0, "nn_gradient_descent: invalid value for epsilon: " << traits.epsilon);

    val = desc.contents("miniBatchSize");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: miniBatchSize is not a double type");
      traits.miniBatchSize = (unsigned int)val.double_value();
    }

    val = desc.contents("pingFrequency");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: pingFrequency is not a double type");
      traits.pingFrequency = (unsigned int)val.double_value();
    }

    val = desc.contents("learningDecay");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: learningDecay is not a double type");
      traits.learningDecay = val.double_value();
    }

    val = desc.contents("spaeBeta");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: spaeBeta is not a double type");
      CHECK(0.0 < val.double_value(), "nn_gradient_descent: out of range spaeBeta value");
      traits.spae_beta = val.double_value();
    }

    val = desc.contents("spaeSparsity");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: spaeSparsity is not a double type");
      CHECK(0.0 < val.double_value() && val.double_value() < 1.0, "nn_gradient_descent: out of range spaeSparsity value");
      traits.spae_sparsity = val.double_value();
    }

    val = desc.contents("minCostDecrease");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: minCostDecrease is not a double type");
      traits.minCostDecrease = val.double_value();
    }

    val = desc.contents("verbose");
    if (val.is_defined())
    {
      CHECK(val.is_bool_type(), "nn_gradient_descent: verbose is not a bool type");
      traits.verbose = val.bool_value();
    }

    val = desc.contents("useSoftmax");
    if (val.is_defined())
    {
      CHECK(val.is_bool_type(), "nn_gradient_descent: useSoftmax is not a bool type");
      traits.use_softmax = val.bool_value();
    }

    val = desc.contents("validationWindowSize");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: validationWindowSize is not a double type");
      traits.validationWindowSize = (unsigned int)val.double_value();
    }

    val = desc.contents("maxiter");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: maxiter is not a double type");
      traits.maxiter = (unsigned int)val.double_value();
    }

    val = desc.contents("evalFrequency");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: evalFrequency is not a double type");
      traits.evalFrequency = (unsigned int)val.double_value();
    }

    val = desc.contents("debug");
    if (val.is_defined())
    {
      CHECK(val.is_bool_type(), "nn_gradient_descent: debug is not a bool type");
      traits.debug = val.bool_value();
    }

    val = desc.contents("bias");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: bias is not a double type");
      traits.bias = val.double_value();
    }

    val = desc.contents("lambda");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_gradient_descent: lambda is not a double type");
      traits.lambda = val.double_value();
    }

    double *dropouts = nullptr;
    val = desc.contents("dropouts");
    if (val.is_defined())
    {
      CHECK(val.is_matrix_type(), "nn_gradient_descent: dropouts is not a matrix type");
      Matrix drop_mat = val.matrix_value();
      CHECK(drop_mat.numel() == nt, "nn_gradient_descent: invalid size for dropout matrix size: " << drop_mat.numel() << "!=" << nt);

      dropouts = new double[nt];
      for (unsigned int i = 0; i < nt; ++i)
      {
        dropouts[i] = drop_mat(i);
        CHECK(dropouts[i] >= 0.0 && dropouts[i] <= 1.0, "nn_gradient_descent: dropout for layer " << i << " is out of range");
      }

      traits.dropouts = dropouts;
    }

    Matrix X_cv_t; // Keep a reference on the X_cv transpose matrix here.

    // If we requested a validationWindow, then we also msut retrieve the cv datasets:
    if (traits.validationWindowSize > 0)
    {
      val = desc.contents("X_cv");
      CHECK(val.is_defined(), "nn_gradient_descent: X_cv value is not defined");
      CHECK(val.is_matrix_type(), "nn_gradient_descent: X_cv is not a matrix type");

      Matrix X_cv = val.matrix_value();
      // Check that this matrix matches the lsizes:
      CHECK(X_cv.dim2() == lsizes[0], "nn_gradient_descent: X_cv size doesn't match lsizes: " << X_cv.dim2() << "!=" << lsizes[0]);

      traits.nsamples_cv = X_cv.dim1();

      // Note that we need to transpose the X_cv matrix before assigning it to the traits:
      traits.X_cv_size = X_cv.numel();
      X_cv_t = X_cv.transpose();
      traits.X_cv = (double *)X_cv_t.data();

      val = desc.contents("y_cv");
      CHECK(val.is_defined(), "nn_gradient_descent: y_cv value is not defined");
      CHECK(val.is_matrix_type(), "nn_gradient_descent: y_cv is not a matrix type");

      Matrix y_cv = val.matrix_value();
      // Check that this matrix matches the lsizes:
      CHECK(y_cv.dim1() == lsizes[nt], "nn_gradient_descent: y_cv size doesn't match lsizes: " << y_cv.dim1() << "!=" << lsizes[nt]);

      // Check that X/Y cv do match:
      CHECK(y_cv.dim2() == X_cv.dim1(), "nn_gradient_descent: X_cv and y_cv sizes do not match: " << X_cv.dim1() << "!=" << y_cv.dim2());

      traits.y_cv_size = y_cv.numel();
      traits.y_cv = (double *)y_cv.data();
    }

    // Prepare a couple of vectors to hold the cv cost and iteration numbers:
    // Assign a callback to read the cv costs:
    auto costCB = [] (double cost, unsigned int iter, void *userdata)
    {
      // logDEBUG("Received cost value "<<cost<<" at iter "<<iter);
      CostMap &themap = *(CostMap *)(userdata);
      themap[iter] = cost;
    };

    traits.userdata = (void *)&costMap;
    traits.cvCostCB = costCB;

    // Specify that we want to retrieve the cost value:
    traits.compute_cost = true;
    traits.compute_grads = true;

    // perform actual gradient descent:
    // logDEBUG("Before call, nparams="<<traits.nparams);
    int res = _run_gd(traits);
    // logDEBUG("After call, nparams="<<traits.nparams);

    // retrieve the cv cost:
    cvCost = traits.cost;

    // release the resources:
    delete [] lsizes;
    delete [] dropouts;

    CHECK(res == GD_SUCCESS, "ERROR: exception occured in gradient descent.")
  }

#ifdef WITH_COST_TEST
  inline void costFunc(const Matrix &lsizes_mat, const Matrix &nn_params, const Matrix &X, const Matrix &yy, double lambda, double &J, Matrix &grads)
  {
    unsigned int nl = lsizes_mat.numel();
    unsigned int *lsizes = new unsigned int[nl];
    for (unsigned int i = 0; i < nl; ++i)
    {
      lsizes[i] = lsizes_mat(i);
    }

    unsigned int np = 0;
    unsigned int nt = nl - 1; // number of matrices evolved.

    for (unsigned int i = 0; i < nt; ++i)
    {
      np += lsizes[i + 1] * (lsizes[i] + 1);
    }

    if (nn_params.numel() != np)
    {
      error("Invalid number of parameters: %d!=%d", np, nn_params.numel());
    }

    _costFunc(nl, lsizes, X.dim1(), (double *)nn_params.data(), (double *)X.data(), (double *)yy.data(), lambda, J, (double *)grads.data(), NULL, NULL); // memcpy((double*)grads.data(),gradients,sizeof(double)*nn_params.numel());

    delete [] lsizes;
  }
#endif
  
protected:
  HMODULE _h;
  RunGDFunc _run_gd;

#ifdef WITH_COST_TEST
  CostFunc _costFunc;
#endif
};

NERVManager g_nerv;

DEFUN_DLD (nn_gradient_descent, args, nargout,
           "nn_gradient_descent function providing C++ implementation of Gradient Descent")
{
  octave_value_list result;

  // we expect to receive 1 arguments:
  int nargin = args.length();
  CHECK_RET(nargin == 1, "nn_gradient_descent: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK_RET(args(0).is_map(), "nn_gradient_descent: desc (arg 0) should be a structure type");

  // Try retrieving the structure:
  octave_scalar_map desc = args(0).scalar_map_value();

  // The desc structure should contain an lsizes element.
  octave_value lsizes_val = desc.contents("lsizes");
  CHECK_RET(lsizes_val.is_defined(), "nn_gradient_descent: lsizes value is not defined");
  CHECK_RET(lsizes_val.is_matrix_type(), "nn_gradient_descent: lsizes is not a matrix type");

  Matrix lsizes = lsizes_val.matrix_value();

  // The desc structure should contain an X_train element.
  octave_value X_train_val = desc.contents("X_train");
  CHECK_RET(X_train_val.is_defined(), "nn_gradient_descent: X_train value is not defined");
  CHECK_RET(X_train_val.is_matrix_type(), "nn_gradient_descent: X_train is not a matrix type");

  Matrix X_train = X_train_val.matrix_value();

  // The desc structure should contain an params element.
  octave_value params_val = desc.contents("params");
  CHECK_RET(params_val.is_defined(), "nn_gradient_descent: params value is not defined");
  CHECK_RET(params_val.is_matrix_type(), "nn_gradient_descent: params is not a matrix type");

  // Matrix params = params_val.matrix_value();
  // params.make_unique();

  Matrix params_orig = params_val.matrix_value();
  unsigned int np2 = params_orig.numel();
  Matrix params = Matrix(np2,1);
  memcpy((void*)params.data(),params_orig.data(),np2*sizeof(double));

  // The desc structure should contain an y_train element.
  octave_value y_train_val = desc.contents("y_train");
  CHECK_RET(y_train_val.is_defined(), "nn_gradient_descent: y_train value is not defined");
  CHECK_RET(y_train_val.is_matrix_type(), "nn_gradient_descent: y_train is not a matrix type");

  Matrix y_train = y_train_val.matrix_value();

  CHECK_RET(X_train.dim2() == lsizes(0), "nn_gradient_descent: Feature matrix doesn't match lsizes: " << X_train.dim2() << "!=" << lsizes(0));

  unsigned int np = 0;
  unsigned int nt = lsizes.numel() - 1;

  for (unsigned int i = 0; i < nt; ++i)
  {
    np += (lsizes(i) + 1) * lsizes(i + 1);
  }

  CHECK_RET(params.numel() == np, "nn_gradient_descent: params doesn't match expected size: " << params.numel() << "!=" << np);

  // Note that her we expect the matrix y_train to be transposed compared to X_train:
  CHECK_RET(X_train.dim1() == y_train.dim2(), "nn_gradient_descent: mismatch in nsamples_train: " << X_train.dim1() << "!=" << y_train.dim2());
  CHECK_RET(y_train.dim1() == lsizes(nt), "nn_gradient_descent: y_train doesn't match lsizes: " << y_train.dim1() << "!=" << lsizes(nt));

  // Prepare the cost map:
  CostMap costsmap;

  double Jcv = 0.0;

  // Call the gradient descent method:
  g_nerv.run_gradient_descent(lsizes, X_train, y_train, params, desc, costsmap, Jcv);

  // build the matrices to hold the cost data:
  size_t n = costsmap.size();

  Matrix costs = Matrix(n, 1);
  Matrix iters = Matrix(n, 1);
  double *cptr = (double *)costs.data();
  double *iptr = (double *)iters.data();

  for (CostMap::iterator it = costsmap.begin(); it != costsmap.end(); ++it)
  {
    (*iptr++) = (double)(it->first);
    (*cptr++) = it->second;
  }

  result.append(params);
  result.append(costs);
  result.append(iters);
  result.append(Jcv);

#ifdef WITH_COST_TEST
  double J = 0.0;
  Matrix grads = Matrix(params.numel(), 1);
  Matrix X_cv = desc.contents("X_cv").matrix_value();
  Matrix y_cv = desc.contents("y_cv").matrix_value();

  g_nerv.costFunc(lsizes, params, X_cv, y_cv, 0.0, J, grads);
  CHECK_RET(abs(J-Jcv)<1e-10,"Mismatch in cv cost computation: "<<J<<"!="<<Jcv);
#endif

  return result;
}

