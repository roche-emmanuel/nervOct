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
    throw std::runtime_error(os.str()); \
  }

#define logDEBUG(msg) octave_stdout << msg << std::endl;

using namespace nerv;

StrategyInterface g_intf;


template<typename T>
T read_desc_value(octave_scalar_map &desc, std::string key, bool optional = false, T def = T())
{
  octave_value val = desc.contents(key);
  CHECK(val.is_defined() || optional, "trade_strategy: " << key << " value is not defined.");
  if (!val.is_defined())
    return def;

  CHECK(val.is_double_type(), "trade_strategy: " << key << " should be a double");
  return (T)val.double_value();
}

template <>
Matrix read_desc_value<Matrix>(octave_scalar_map &desc, std::string key, bool optional, Matrix def)
{
  octave_value val = desc.contents(key);
  CHECK(val.is_defined() || optional, "trade_strategy: " << key << " value is not defined.");
  if (!val.is_defined())
    return def;

  CHECK(val.is_matrix_type(), "trade_strategy: " << key << " should be a matrix");
  return val.matrix_value();
}

template <>
std::string read_desc_value<std::string>(octave_scalar_map &desc, std::string key, bool optional, std::string def)
{
  octave_value val = desc.contents(key);
  CHECK(val.is_defined() || optional, "trade_strategy: " << key << " value is not defined.");
  if (!val.is_defined())
    return def;

  CHECK(val.is_string(), "trade_strategy: " << key << " should be a string");
  return val.string_value();
}

unsigned int read_uint(octave_scalar_map &desc, std::string key, bool optional = false, unsigned int def = 0)
{
  return read_desc_value<unsigned int>(desc, key, optional, def);
}

double read_double(octave_scalar_map &desc, std::string key, bool optional = false, double def = 0)
{
  return read_desc_value<double>(desc, key, optional, def);
}

Matrix read_matrix(octave_scalar_map &desc, std::string key, bool optional = false, Matrix def = Matrix())
{
  return read_desc_value<Matrix>(desc, key, optional, def);
}

std::string read_string(octave_scalar_map &desc, std::string key, bool optional = false, std::string def = std::string())
{
  return read_desc_value<std::string>(desc, key, optional, def);
}

typedef std::map<std::string,int> TypeMap;

DEFUN_DLD (trade_strategy, args, nargout,
           "trade_strategy function providing C++ implementation of Trading Strategy management")
{
  octave_value_list result;

  try
  {
    // logDEBUG("Calling trade_strategy function.")

    // we expect to receive at least 2 arguments:
    int nargin = args.length();

    CHECK(nargin >= 2, "trade_strategy: Invalid number of arguments: " << nargin);

    // First we retrieve the command from the first argument (should be a string)
    CHECK(args(0).is_string(), "trade_strategy: command is not a string");
    std::string cmd = args(0).string_value();

    if (cmd == "create")
    {
      CHECK(args(1).is_map(), "trade_strategy: desc should be a structure type");
      octave_scalar_map desc = args(1).scalar_map_value();
      // Prepare the creation traits:
      Strategy::CreationTraits traits;
      traits.target_symbol = read_uint(desc, "target_symbol");

      // Create the strategy:
      int id = g_intf.create_strategy(traits);
      CHECK(id > 0, "trade_strategy: cannot create strategy.");

      result.append(octave_uint32(id));

      return result;
    }

    // We can already read the strategy id and the desc structure:
    CHECK(args(1).is_uint32_type(), "trade_strategy: invalid strategy id");
    unsigned int sid = (unsigned int)(args(1).uint32_scalar_value());
    CHECK(sid > 0, "trade_strategy: invalud strategy id(==0)");

    if (cmd == "destroy")
    {
      CHECK(g_intf.destroy_strategy(sid)==ST_SUCCESS,"Cannot destroy strategy "<<sid);
      return result;
    }

    // For the other commands we expect 3 arguments:
    CHECK(nargin == 3, "trade_strategy: Invalid final number of arguments: " << nargin);

    CHECK(args(2).is_map(), "trade_strategy: desc should be a structure type");
    octave_scalar_map desc = args(2).scalar_map_value();

    if (cmd == "evaluate")
    {
      Strategy::EvalTraits traits;

      Matrix balance = read_matrix(desc,"balance",true);
      
      if(balance.numel()>0) {
        traits.balance = (double*)balance.data();
      }

      Matrix inputs = read_matrix(desc,"inputs");
      traits.inputs = (double*)inputs.data();
      traits.inputs_nrows = inputs.dim1();
      traits.inputs_ncols = inputs.dim2();

      // Retrieve the prices to use here:
      Matrix prices = read_matrix(desc,"prices");
      traits.prices = (double*)prices.data();
      traits.prices_nrows = prices.dim1();
      traits.prices_ncols = prices.dim2();

      traits.mean_spread = read_double(desc,"mean_spread",true,8.0);
      traits.max_lost = read_double(desc,"max_lost",true,4.0);
      traits.lot_multiplier = read_double(desc,"lot_multiplier",true,1.0);
      

      CHECK(g_intf.evaluate_strategy(sid,traits)==ST_SUCCESS,"Could not evaluate strategy.");
    }
    else if (cmd == "add_model")
    {
      TypeMap model_type_map{
        {"nls_network", MODEL_NLS_NETWORK}
      };

      std::string tname = read_string(desc,"type");
      CHECK(model_type_map.find(tname)!=model_type_map.end(),"Invalid model type name "<<tname)

      Matrix lsizes_mat = read_matrix(desc,"lsizes");
      unsigned int nl = lsizes_mat.numel();
      unsigned int nt = nl-1;
      unsigned int* lsizes = new unsigned int[nl];
      for(unsigned int i=0;i<nl;++i) {
        lsizes[i] = lsizes_mat(i);
      }

      Matrix params = read_matrix(desc,"params");
      
      Matrix mu = read_matrix(desc,"mu");
      Matrix sigma = read_matrix(desc,"sigma");
      CHECK(mu.dim1()==lsizes[0] && mu.dim2()==1,"Invalid mu dimensions "<<mu.dim1()<<"x"<<mu.dim2());
      CHECK(sigma.dim1()==lsizes[0] && sigma.dim2()==1,"Invalid sigma dimensions "<<sigma.dim1()<<"x"<<sigma.dim2());

      // Check that the params size match the lsizes:
      unsigned int np = 0;
      for(unsigned int i=0;i<nt;++i) {
        np += (lsizes[i]+1)*lsizes[i+1];
      }
      CHECK(np==params.numel(),"Invalid number of parameters "<<params.numel()<<"!="<<np);

      Model::CreationTraits traits;
      traits.type = model_type_map[tname];
      traits.nl = nl;
      traits.lsizes = lsizes;
      traits.params = (double*)params.data();
      traits.mu = (double*)mu.data();
      traits.sigma = (double*)sigma.data();

      CHECK(g_intf.add_strategy_model(sid,traits)==ST_SUCCESS,"Could not add strategy model.");

      delete [] lsizes;
    }
    else
    {
      CHECK(false, "trade_strategy: unknown command name: " << cmd);
    }
  }
  catch (...)
  {
    // Do nothing special.
  }

  return result;
}

