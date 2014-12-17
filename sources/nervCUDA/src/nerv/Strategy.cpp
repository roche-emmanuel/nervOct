#include <nervCUDA.h>
#include <nerv/Strategy.h>
#include <sgtCore.h>

using namespace nerv;

Strategy::Strategy(const CreationTraits &traits)
{
  // Assign an unique ID to this strategy;
  static int counter = 0;
  _id = ++counter;

  logDEBUG("Creating Strategy " << _id << ".");
}

Strategy::~Strategy()
{
  logDEBUG("Destroying Strategy " << _id << ".");
}

void Strategy::evaluate(EvalTraits &traits)
{
  // Retrieve the input data:
  value_type *inputs = traits.inputs;
  int nfeatures = traits.inputs_nrows;
  int nsamples = traits.inputs_ncols;

  THROW_IF(!inputs || nfeatures <= 0 || nsamples <= 0, "Invalid input data");

  // Initial position should be NONE:
  int cur_pos = POS_NONE;
  

}
