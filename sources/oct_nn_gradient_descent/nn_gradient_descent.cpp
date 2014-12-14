#include <octave/oct.h>
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
  CHECK(nargin==5,"nn_gradient_descent: Invalid number of arguments: " << nargin);

  // Check the argument types:
  // CHECK(args(0).is_matrix_type(),"nn_gradient_descent: nn_params (arg 0) should be a matrix type");

  return result;
}
