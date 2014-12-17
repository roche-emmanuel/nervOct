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

class NERVManager
{
public:
};

NERVManager g_nerv;

DEFUN_DLD (trade_strategy, args, nargout,
           "trade_strategy function providing C++ implementation of Trading Strategy management")
{
  octave_value_list result;

  // logDEBUG("Calling trade_strategy function.")

  // we expect to receive at least 2 arguments:
  int nargin = args.length();

  CHECK_RET(nargin == 3, "trade_strategy: Invalid number of arguments: " << nargin);

  // First we retrieve the command from the first argument (should be a string)
  CHECK_RET(args(0).is_string(), "trade_strategy: command is not a string");
  std::string cmd = args(0).string_value();

  // We can already read the strategy id and the desc structure:
  CHECK_RET(args(1).is_uint32_type(), "trade_strategy: invalid strategy id");
  unsigned int sid = (unsigned int)(args(1).uint32_scalar_value());
  CHECK_RET(cmd=="create" || sid > 0, "trade_strategy: invalud strategy id(==0)");

  CHECK_RET(args(2).is_map(), "trade_strategy: desc should be a structure type");
  octave_scalar_map desc = args(2).scalar_map_value();


  if (cmd == "create")
  {
    CHECK_RET(sid==0,"Strategy ID should be zero when creating new strategy");

  }
  else
  {
    CHECK_RET(false, "trade_strategy: unknown command name: " << cmd);
  }

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

