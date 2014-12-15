#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <sstream>
#include <windows.h>

#include <nerv/GDTraits.h>

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
  typedef int (* RunGDFunc)(GDTraits<double> &traits);

public:
  NERVManager()
  {
    logDEBUG("Loading nervCUDA...");
    _h = LoadLibrary("nervCUDA.dll");
    CHECK(_h, "ERROR: cannot load nervCUDA library! err=" << GetLastError());

    // Try loading the functions of interest:
    _run_gd = (RunGDFunc) GetProcAddress(_h, "run_gradient_descent");
    CHECK(_run_gd, "ERROR: cannot find run_gradient_descent method! err=" << GetLastError());
  }

  ~NERVManager()
  {
    logDEBUG("Unloading nervCUDA module...")
    BOOL res = FreeLibrary(_h);
    CHECK(res, "ERROR: cannot free library! err=" << GetLastError());
  }

  inline void run_gradient_descent(const Matrix &lsizes_mat, const Matrix &X_train, const Matrix &y_train,
                                   const Matrix &params)
  {
    // Here we can already check that the feature matrix dimensions match
    // the lsizes description:
    CHECK(X_train.dim2() == lsizes_mat(0), "nn_gradient_descent: Feature matrix doesn't match lsizes: " << X_train.dim2() << "!=" << lsizes_mat(0));
    // TODO: also check here that we have the same number of samples.

    GDTraits<double> traits;

    // Assign the nl value:
    unsigned int nl = lsizes_mat.numel();
    traits.nl = nl;
    logDEBUG("traits.nl = " << traits.nl)

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
    CHECK(params.numel() == traits.np(), "nn_gradient_descent: params doesn't match expected size: " << params.numel() << "!=" << traits.np());
    traits.nparams = params.numel();
    traits.params = (double *)params.data();

    // Assign X train data:
    traits.X_train_size = X_train.numel();
    traits.X = (double *)X_train.data();

    // Assign the y_train data:
    CHECK(X_train.dim1() == y_train.dim1(), "nn_gradient_descent: mismatch in nsamples_train: " << X_train.dim1() << "!=" << y_train.dim1());
    CHECK(y_train.dim2() == lsizes_mat(nl-1), "nn_gradient_descent: y_train doesn't match lsizes: " << y_train.dim2() << "!=" << lsizes_mat(nl-1));
    traits.y_train_size = y_train.numel();
    traits.yy = (double*)y_train.data();

    // perform actual gradient descent:
    int res = _run_gd(traits);

    // release the resources:
    delete [] lsizes;

    CHECK(res == GD_SUCCESS, "ERROR: exception occured in gradient descent.")
  }

protected:
  HMODULE _h;
  RunGDFunc _run_gd;
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

  Matrix params = params_val.matrix_value();

  // The desc structure should contain an y_train element.
  octave_value y_train_val = desc.contents("y_train");
  CHECK_RET(y_train_val.is_defined(), "nn_gradient_descent: y_train value is not defined");
  CHECK_RET(y_train_val.is_matrix_type(), "nn_gradient_descent: y_train is not a matrix type");

  Matrix y_train = y_train_val.matrix_value();

  // Call the gradient descent method:
  g_nerv.run_gradient_descent(lsizes, X_train, y_train, params);

  return result;
}

