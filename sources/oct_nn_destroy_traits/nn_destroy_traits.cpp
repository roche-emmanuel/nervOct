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

DEFUN_DLD (nn_destroy_traits, args, nargout,
           "nn_destroy_traits function providing C++ implementation of neural network cost function")
{
  octave_value_list result;

  // we expect to receive 1 arguments:
  int nargin = args.length();
  CHECK_RET(nargin == 1, "nn_destroy_traits: Invalid number of arguments: " << nargin);

  CHECK_RET(args(0).is_uint32_type(), "trade_strategy: invalid strategy id");
  unsigned int sid = (unsigned int)(args(0).uint32_scalar_value());
  CHECK_RET(sid > 0, "nn_destroy_traits: invalud strategy id(==0)");

  int res = g_intf.destroy_device_traits(sid);
  
  CHECK_RET(res == TRAITS_SUCCESS, "ERROR: exception occured in nn_destroy_traits.")

  return result;
}

