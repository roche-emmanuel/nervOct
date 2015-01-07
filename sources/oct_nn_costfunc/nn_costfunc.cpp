#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <sstream>
#include <windows.h>
#include <iomanip>

#include <nerv/BPTraits.h>

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

class NERVManager
{
protected:
  typedef void (* CostFunc)(BPTraits<double> &traits);

public:
  NERVManager()
  {
    logDEBUG("Loading nervCUDA...");
    _h = LoadLibrary("nervCUDA.dll");
    CHECK(_h, "ERROR: cannot load nervCUDA library! err=" << GetLastError());

    // Try loading the functions of interest:
    _costfunc = (CostFunc) GetProcAddress(_h, "gd_errfunc_cpu");
    CHECK(_costfunc, "ERROR: cannot find gd_errfunc_cpu method! err=" << GetLastError());
  }

  ~NERVManager()
  {
    logDEBUG("Unloading nervCUDA module...")
    BOOL res = FreeLibrary(_h);
    CHECK(res, "ERROR: cannot free library! err=" << GetLastError());
  }

  inline void compute_cost(const Matrix &lsizes_mat, const Matrix &X_train, const Matrix &y_train,
                           const Matrix &params, octave_scalar_map &desc, double &cost)
  {
    // Here we can already check that the feature matrix dimensions match
    // the lsizes description:
    BPTraits<double> traits;

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
    // Note that we expect the X matrix to contain one sample per column
    // thus the format should be nfeatures x nsamples
    unsigned int nsamples_train = X_train.dim2();
    traits.nsamples_train = nsamples_train;

    // We already ensured that we have the proper number of parameters.
    traits.params = (double *)params.data();

    // Assign X train data:
    // Note that here we need to transpose the X matrix before assigning it to the traits
    traits.X = (double *)X_train.data();

    // Assign the y_train data:
    // Note that we also expect the y_train matrix to contain one sample per column.
    traits.yy = (double *)y_train.data();

    // Read the optional values:
    octave_value val = desc.contents("spaeBeta");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_costfunc: spaeBeta is not a double type");
      CHECK(0.0 < val.double_value(), "nn_costfunc: out of range spaeBeta value");
      traits.spae_beta = val.double_value();
    }

    val = desc.contents("spaeSparsity");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_costfunc: spaeSparsity is not a double type");
      CHECK(0.0 < val.double_value() && val.double_value() < 1.0, "nn_costfunc: out of range spaeSparsity value");
      traits.spae_sparsity = val.double_value();
    }

    val = desc.contents("useSoftmax");
    if (val.is_defined())
    {
      CHECK(val.is_bool_type(), "nn_costfunc: useSoftmax is not a bool type");
      traits.use_softmax = val.bool_value();
    }

    val = desc.contents("debug");
    if (val.is_defined())
    {
      CHECK(val.is_bool_type(), "nn_costfunc: debug is not a bool type");
      traits.debug = val.bool_value();
    }

    val = desc.contents("bias");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_costfunc: bias is not a double type");
      traits.bias = val.double_value();
    }

    val = desc.contents("lambda");
    if (val.is_defined())
    {
      CHECK(val.is_double_type(), "nn_costfunc: lambda is not a double type");
      traits.lambda = val.double_value();
    }

    double *dropouts = nullptr;
    val = desc.contents("dropouts");
    if (val.is_defined())
    {
      CHECK(val.is_matrix_type(), "nn_costfunc: dropouts is not a matrix type");
      Matrix drop_mat = val.matrix_value();
      CHECK(drop_mat.numel() == nt, "nn_costfunc: invalid size for dropout matrix size: " << drop_mat.numel() << "!=" << nt);

      dropouts = new double[nt];
      for (unsigned int i = 0; i < nt; ++i)
      {
        dropouts[i] = drop_mat(i);
        CHECK(dropouts[i] >= 0.0 && dropouts[i] <= 1.0, "nn_costfunc: dropout for layer " << i << " is out of range");
      }

      traits.dropouts = dropouts;
    }

    // Specify that we want to retrieve the cost value:
    traits.compute_cost = true;
    traits.compute_grads = true;

    // perform actual gradient descent:
    _costfunc(traits);

    // retrieve the cv cost:
    cost = traits.cost;

    // release the resources:
    delete [] lsizes;
    delete [] dropouts;
    // CHECK(res == GD_SUCCESS, "ERROR: exception occured in costfunc.")
  }

protected:
  HMODULE _h;
  CostFunc _costfunc;
};

NERVManager g_nerv;

DEFUN_DLD (nn_costfunc, args, nargout,
           "nn_costfunc function providing C++ implementation of neural network cost function")
{
  octave_value_list result;

  // we expect to receive 1 arguments:
  int nargin = args.length();
  CHECK_RET(nargin == 1, "nn_costfunc: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK_RET(args(0).is_map(), "nn_costfunc: desc (arg 0) should be a structure type");

  // Try retrieving the structure:
  octave_scalar_map desc = args(0).scalar_map_value();

  // The desc structure should contain an lsizes element.
  octave_value lsizes_val = desc.contents("lsizes");
  CHECK_RET(lsizes_val.is_defined(), "nn_costfunc: lsizes value is not defined");
  CHECK_RET(lsizes_val.is_matrix_type(), "nn_costfunc: lsizes is not a matrix type");

  Matrix lsizes = lsizes_val.matrix_value();

  // The desc structure should contain an X_train element.
  octave_value X_train_val = desc.contents("X_train");
  CHECK_RET(X_train_val.is_defined(), "nn_costfunc: X_train value is not defined");
  CHECK_RET(X_train_val.is_matrix_type(), "nn_costfunc: X_train is not a matrix type");

  Matrix X_train = X_train_val.matrix_value();

  // The desc structure should contain an params element.
  octave_value params_val = desc.contents("params");
  CHECK_RET(params_val.is_defined(), "nn_costfunc: params value is not defined");
  CHECK_RET(params_val.is_matrix_type(), "nn_costfunc: params is not a matrix type");

  // Matrix params = params_val.matrix_value();
  // params.make_unique();

  Matrix params_orig = params_val.matrix_value();
  unsigned int np2 = params_orig.numel();
  Matrix params = Matrix(np2, 1);
  memcpy((void *)params.data(), params_orig.data(), np2 * sizeof(double));

  // The desc structure should contain an y_train element.
  octave_value y_train_val = desc.contents("y_train");
  CHECK_RET(y_train_val.is_defined(), "nn_costfunc: y_train value is not defined");
  CHECK_RET(y_train_val.is_matrix_type(), "nn_costfunc: y_train is not a matrix type");

  Matrix y_train = y_train_val.matrix_value();

  CHECK_RET(X_train.dim1() == lsizes(0), "nn_costfunc: Feature matrix doesn't match lsizes: " << X_train.dim1() << "!=" << lsizes(0));

  unsigned int np = 0;
  unsigned int nt = lsizes.numel() - 1;

  for (unsigned int i = 0; i < nt; ++i)
  {
    np += (lsizes(i) + 1) * lsizes(i + 1);
  }

  CHECK_RET(params.numel() == np, "nn_costfunc: params doesn't match expected size: " << params.numel() << "!=" << np);

  // Note that her we expect the matrix y_train to be transposed compared to X_train:
  CHECK_RET(X_train.dim2() == y_train.dim2(), "nn_costfunc: mismatch in nsamples_train: " << X_train.dim1() << "!=" << y_train.dim2());
  CHECK_RET(y_train.dim1() == lsizes(nt), "nn_costfunc: y_train doesn't match lsizes: " << y_train.dim1() << "!=" << lsizes(nt));

  double Jcv = 0.0;

  // Call the gradient descent method:
  g_nerv.compute_cost(lsizes, X_train, y_train, params, desc, Jcv);

  result.append(Jcv);
  result.append(params);

  return result;
}

