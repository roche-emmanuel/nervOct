#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <sstream>
#include <windows.h>
#include <iomanip>

#include <nerv/StrategyInterface.h>

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
public:
};

NERVManager g_nerv;

DEFUN_DLD (trade_strategy, args, nargout,
           "trade_strategy function providing C++ implementation of Trading Strategy management")
{
  octave_value_list result;

  // we expect to receive at least 2 arguments:
  int nargin = args.length();
  CHECK_RET(nargin >= 2, "trade_strategy: Invalid number of arguments: " << nargin);

  // // Check the argument types:
  // CHECK_RET(args(0).is_map(), "trade_strategy: desc (arg 0) should be a structure type");

  // // Try retrieving the structure:
  // octave_scalar_map desc = args(0).scalar_map_value();

  // // The desc structure should contain an lsizes element.
  // octave_value lsizes_val = desc.contents("lsizes");
  // CHECK_RET(lsizes_val.is_defined(), "trade_strategy: lsizes value is not defined");
  // CHECK_RET(lsizes_val.is_matrix_type(), "trade_strategy: lsizes is not a matrix type");

  // Matrix lsizes = lsizes_val.matrix_value();


  return result;
}

