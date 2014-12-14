#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <sstream>
#include <windows.h>

#define CHECK(cond, msg) if(!(cond)) { \
    std::ostringstream os; \
    os << msg; \
    error(os.str().c_str()); \
    return result; \
  }

#define logDEBUG(msg) octave_stdout << msg << std::endl;


DEFUN_DLD (nn_gradient_descent, args, nargout,
           "nn_gradient_descent function providing C++ implementation of Gradient Descent")
{
  octave_value_list result;

  // we expect to receive 1 arguments:
  int nargin = args.length();
  CHECK(nargin == 1, "nn_gradient_descent: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK(args(0).is_map(), "nn_gradient_descent: desc (arg 0) should be a structure type");

  // Try retrieving the structure:
  octave_scalar_map desc = args(0).scalar_map_value();

  // The desc structure should contain an lsizes element.
  octave_value lsizes_val = desc.contents("lsizes");
  CHECK(lsizes_val.is_defined(), "nn_gradient_descent: lsizes value is not defined");

  return result;
}

