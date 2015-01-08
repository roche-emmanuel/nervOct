#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <sstream>
#include <windows.h>
#include <iomanip>

#include <nerv/BPTraitsInterface.h>
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

BPTraitsInterface g_intf;

DEFUN_DLD (nn_create_traits, args, nargout,
           "nn_create_traits function providing C++ implementation of neural network cost function")
{
  octave_value_list result;

  // we expect to receive 1 arguments:
  int nargin = args.length();
  CHECK_RET(nargin == 1, "nn_create_traits: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK_RET(args(0).is_map(), "nn_create_traits: desc (arg 0) should be a structure type");

  // Try retrieving the structure:
  octave_scalar_map desc = args(0).scalar_map_value();

  BPTraits<double> traits;

  // The desc structure should contain an lsizes element.
  octave_value lsizes_val = desc.contents("lsizes");
  CHECK_RET(lsizes_val.is_defined(), "nn_create_traits: lsizes value is not defined");
  CHECK_RET(lsizes_val.is_matrix_type(), "nn_create_traits: lsizes is not a matrix type");

  // Assign the nl value:
  Matrix lsizes_mat = lsizes_val.matrix_value();
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



  // The desc structure should contain an X_train element.
  octave_value X_train_val = desc.contents("X_train");
  CHECK_RET(X_train_val.is_defined(), "nn_create_traits: X_train value is not defined");
  CHECK_RET(X_train_val.is_matrix_type(), "nn_create_traits: X_train is not a matrix type");

  Matrix X_train = X_train_val.matrix_value();

  CHECK_RET(X_train.dim1() == lsizes(0), "nn_create_traits: Feature matrix doesn't match lsizes: " << X_train.dim1() << "!=" << lsizes(0));

  // Retrieve the number of train samples from the X_train matrix.
  // Note that we expect the X matrix to contain one sample per column
  // thus the format should be nfeatures x nsamples
  unsigned int nsamples_train = X_train.dim2();
  traits.nsamples_train = nsamples_train;

  // Assign X train data:
  // Note that here we need to transpose the X matrix before assigning it to the traits
  traits.X = (double *)X_train.data();



  // The desc structure should contain an params element.
  octave_value params_val = desc.contents("params");
  CHECK_RET(params_val.is_defined(), "nn_create_traits: params value is not defined");
  CHECK_RET(params_val.is_matrix_type(), "nn_create_traits: params is not a matrix type");

  Matrix params = params_val.matrix_value();

  unsigned int np = 0;
  for (unsigned int i = 0; i < nt; ++i)
  {
    np += (lsizes(i) + 1) * lsizes(i + 1);
  }

  CHECK_RET(params.numel() == np, "nn_create_traits: params doesn't match expected size: " << params.numel() << "!=" << np);

  // We already ensured that we have the proper number of parameters.
  traits.params = (double *)params.data();


  // The desc structure should contain an y_train element.
  octave_value y_train_val = desc.contents("y_train");
  CHECK_RET(y_train_val.is_defined(), "nn_create_traits: y_train value is not defined");
  CHECK_RET(y_train_val.is_matrix_type(), "nn_create_traits: y_train is not a matrix type");

  Matrix y_train = y_train_val.matrix_value();

  // Note that her we expect the matrix y_train to be transposed compared to X_train:
  CHECK_RET(X_train.dim2() == y_train.dim2(), "nn_create_traits: mismatch in nsamples_train: " << X_train.dim1() << "!=" << y_train.dim2());
  CHECK_RET(y_train.dim1() == lsizes(nt), "nn_create_traits: y_train doesn't match lsizes: " << y_train.dim1() << "!=" << lsizes(nt));

  // Assign the y_train data:
  // Note that we also expect the y_train matrix to contain one sample per column.
  traits.yy = (double *)y_train.data();


  // Read the optional values:
  octave_value val = desc.contents("spaeBeta");
  if (val.is_defined())
  {
    CHECK(val.is_double_type(), "nn_create_traits: spaeBeta is not a double type");
    CHECK(0.0 <= val.double_value(), "nn_create_traits: out of range spaeBeta value");
    traits.spae_beta = val.double_value();
  }

  val = desc.contents("spaeSparsity");
  if (val.is_defined())
  {
    CHECK(val.is_double_type(), "nn_create_traits: spaeSparsity is not a double type");
    CHECK(0.0 < val.double_value() && val.double_value() < 1.0, "nn_create_traits: out of range spaeSparsity value");
    traits.spae_sparsity = val.double_value();
  }

  val = desc.contents("useSoftmax");
  if (val.is_defined())
  {
    CHECK(val.is_bool_type(), "nn_create_traits: useSoftmax is not a bool type");
    traits.use_softmax = val.bool_value();
  }

  val = desc.contents("costMode");
  if (val.is_defined())
  {
    CHECK(val.is_double_type(), "nn_create_traits: costMode is not a double type");
    traits.cost_mode = (unsigned int)val.double_value();
  }

  val = desc.contents("debug");
  if (val.is_defined())
  {
    CHECK(val.is_bool_type(), "nn_create_traits: debug is not a bool type");
    traits.debug = val.bool_value();
  }

  val = desc.contents("bias");
  if (val.is_defined())
  {
    CHECK(val.is_double_type(), "nn_create_traits: bias is not a double type");
    traits.bias = val.double_value();
  }

  val = desc.contents("lambda");
  if (val.is_defined())
  {
    CHECK(val.is_double_type(), "nn_create_traits: lambda is not a double type");
    traits.lambda = val.double_value();
  }

  double *dropouts = nullptr;
  val = desc.contents("dropouts");
  if (val.is_defined())
  {
    CHECK(val.is_matrix_type(), "nn_create_traits: dropouts is not a matrix type");
    Matrix drop_mat = val.matrix_value();
    CHECK(drop_mat.numel() == nt, "nn_create_traits: invalid size for dropout matrix size: " << drop_mat.numel() << "!=" << nt);

    dropouts = new double[nt];
    for (unsigned int i = 0; i < nt; ++i)
    {
      dropouts[i] = drop_mat(i);
      CHECK(dropouts[i] >= 0.0 && dropouts[i] <= 1.0, "nn_create_traits: dropout for layer " << i << " is out of range");
    }

    traits.dropouts = dropouts;
  }

  // Specify that we want to retrieve the cost value:
  traits.compute_cost = true;
  traits.compute_grads = true;

  // Call the gradient descent method:
  Matrix grads = Matrix(params.numel(), 1);
  // add the gradient matrix:
  traits.grads = (double *)grads.data();

  int id = g_intf.create_device_traits(traits);

  // release the resources:
  delete [] lsizes;
  delete [] dropouts;
  
  CHECK(id>0, "ERROR: exception occured in nn_create_traits.")

  result.append(octave_uint32(id));

  return result;
}

