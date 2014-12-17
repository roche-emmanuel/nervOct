#include <nervCUDA.h>
#include <nerv/Strategy.h>
#include <sgtCore.h>

using namespace nerv;

Strategy::Strategy(const CreationTraits &traits)
{
  // Assign an unique ID to this strategy;
  static int counter = 0;
  _id = ++counter;

  _target_symbol = traits.target_symbol;
  THROW_IF(_target_symbol <=0 || _target_symbol>=SYMBOL_LAST,"Invalid target symbol");

  logDEBUG("Creating Strategy " << _id << ".");
}

Strategy::~Strategy()
{
  logDEBUG("Destroying Strategy " << _id << ".");
}

void Strategy::evaluate(EvalTraits &traits) const
{
  // Retrieve the input data:
  value_type *iptr = traits.inputs;
  int nfeatures = traits.inputs_nrows;
  int nsamples = traits.inputs_ncols;

  THROW_IF(!iptr || nfeatures <= 0 || nsamples <= 0, "Invalid input data");

  // Initial position should be NONE:
  int cur_pos = POS_NONE;

  for(int i=0;i<nsamples;++i) {
  	// extract the current price from the input data:
  	value_type cur_price = getPrice(iptr, PRICE_CLOSE);
  }
}

Strategy::value_type Strategy::getPrice(value_type *iptr, int type, int symbol) const
{
	if(symbol<=0)
		symbol = _target_symbol;

	int index = 4*(symbol-1) + type;
	return iptr[index];
}
