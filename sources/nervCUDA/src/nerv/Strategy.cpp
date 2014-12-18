#include <nervCUDA.h>
#include <nerv/Strategy.h>
#include <sgtCore.h>
#include <nerv/NLSNetworkModel.h>

using namespace nerv;

Strategy::Strategy(const CreationTraits &traits)
{
  // Assign an unique ID to this strategy;
  static int counter = 0;
  _id = ++counter;

  _target_symbol = traits.target_symbol;
  THROW_IF(_target_symbol <= 0 || _target_symbol >= SYMBOL_LAST, "Invalid target symbol");

  logDEBUG("Creating Strategy " << _id << ".");
}

Strategy::~Strategy()
{
  logDEBUG("Destroying Strategy " << _id << ".");
  destroyAllModels();
}

void Strategy::evaluate(EvalTraits &traits)
{
  // Retrieve the input data:
  value_type *iptr = traits.inputs;
  int nfeatures = traits.inputs_nrows;
  int nsamples = traits.inputs_ncols;

  value_type *balance_ptr = traits.balance;

  THROW_IF(!iptr || nfeatures <= 0 || nsamples <= 0, "Invalid input data");

  value_type stop_lost = -1.0;
  value_type ref_price = -1.0;
  value_type num_lots = 0.0;

  value_type mean_spread = 0.00008;
  value_type max_lost = mean_spread * 0.5;

  value_type gross_profit = 0.0;
  value_type gross_lost = 0.0;

  value_type balance = 3000.0; // initial balance in euros.

  // Initial position should be NONE:
  int cur_pos = POS_NONE;

  int num_transactions = 0;

  DigestTraits dtraits;
  dtraits.input_size = nfeatures;

  value_type cur_price, cur_low, cur_high, gain, new_stop, profit;


  for (int i = 0; i < nsamples; ++i)
  {
    // logDEBUG("Evaluating input " << i << "...");

    // Digest this input:
    dtraits.input = iptr;
    digest(dtraits);

    // extract the current price from the input data:
    cur_price = getPrice(iptr, PRICE_CLOSE);

    // Check the ref price and stop lost values:
    if (cur_pos != POS_NONE)
    {
      THROW_IF(ref_price < 0 || stop_lost < 0, "Invalid ref price or stop lost");
    }

    if (cur_pos == POS_LONG)
    {
      // Retrieve the latest low price:
      cur_low = getPrice(iptr, PRICE_LOW);

      if (cur_low <= stop_lost)
      {
        // The price went under the stop lost at some point in the elapsed minute.
        // So the transaction was closed in the process.

        // Now we just need to compute the actual gain:
        gain = stop_lost - ref_price;
        if (gain > 0)
        {
          gross_profit += gain;
        }
        else
        {
          gross_lost += -gain; // take the negative value here.
        }

        // We assume here that we are trading the EURO symbol on our account.
        // Then the profit depends on the lot size:
        profit = num_lots * 100000.0 * (stop_lost / ref_price - 1.0);
        balance += profit;
        logDEBUG("New Balance value: "<<balance);

        // Leave the current position:
        ref_price = -1.0;
        stop_lost = -1.0;
        cur_pos = POS_NONE;
      }
      else
      {
        // we didn't reach any stop lost yet.
        // So here we check if we are still in the "fluctuation area"
        // or if we should move to the "take highest profit" mode:
        if (cur_price > ref_price)
        {
          // update the stop lost:
          new_stop = cur_price - 0.5 * std::min(cur_price - ref_price, mean_spread);

          // Only update the stop_lost if we are raising its value:
          stop_lost = std::max(stop_lost, new_stop);
        }
        else
        {
          // The price is still fluctuating between ref_price and the initial stop_lost, so we take no action here.
        }
      }
    }


    if (cur_pos == POS_SHORT)
    {
      cur_high = getPrice(iptr, PRICE_HIGH);

      // First check if at some point in this minute we went above the stop_lost value:
      if (cur_high >= stop_lost)
      {
        // We have to close this transaction.
        gain = ref_price - stop_lost;
        if (gain > 0)
        {
          gross_profit += gain;
        }
        else
        {
          gross_lost += -gain;
        }

        // We assume here that we are trading the EURO symbol on our account.
        // Then the profit depends on the lot size:
        profit = num_lots * 100000.0 * (ref_price / stop_lost - 1.0);
        balance += profit;
        logDEBUG("New Balance value: "<<balance);

        // Leave the current position:
        ref_price = -1.0;
        stop_lost = -1.0;
        cur_pos = POS_NONE;
      }
      else
      {
        // we didn't reach any stop lost yet.
        // So here we check if we are still in the "fluctuation area"
        // or if we should move to the "take highest profit" mode:
        if (cur_price < ref_price)
        {
          // update the stop lost:
          new_stop = cur_price + 0.5 * std::min(ref_price - cur_price, mean_spread);

          // Only update the stop_lost if we are lowering its value:
          stop_lost = std::min(stop_lost, new_stop);
        }
        else
        {
          // The price is still fluctuating between ref_price and the initial stop_lost, so we take no action here.
        }
      }
    }

    if (cur_pos == POS_NONE)
    {
      // here we should use the digest result if applicable
      // to select the next position to consider:
      if (dtraits.position != POS_UNKNOWN)
      {
        cur_pos = dtraits.position;
      }

      if(cur_pos != POS_NONE) {
        num_transactions++;
        num_lots = dtraits.confidence;
        num_lots = floor(num_lots*100.0)/100.0;
        logDEBUG("Performing transaction with lot size: "<<num_lots<<" confidence="<<dtraits.confidence)
      }

      if (cur_pos == POS_LONG)
      {
        // We are either buying or selling, so we record the current close price:
        // When buying we have to add the mean spread to the reference price to get the ask price
        ref_price = cur_price + mean_spread;

        // Stop the looses if the price goes down too much:
        stop_lost = cur_price - max_lost;
      }

      if (cur_pos == POS_SHORT)
      {
        // We are either buying or selling, so we record the current close price:
        ref_price = cur_price;

        // Stop the looses if the price goes high too much:
        stop_lost = cur_price + max_lost;
      }
    }

    // Save the current balance after each minute if requested:
    if (balance_ptr)
      balance_ptr[i] = balance;

    // Move to the next minute:
    iptr += nfeatures;
  }

  logDEBUG("Final balance is: " << balance);
}

Strategy::value_type Strategy::getPrice(value_type *iptr, int type, int symbol) const
{
  if (symbol <= 0)
    symbol = _target_symbol;

  int index = 4 * (symbol - 1) + type;
  return iptr[index];
}

void Strategy::digest(DigestTraits &traits)
{
  // if we have no model, then just return an unknown position.
  if (_models.empty())
  {
    traits.position = POS_UNKNOWN;
    return;
  }

  // Once we have a list of model, we can digest the input on each of them
  // and retrieve they decision:
  Model::DigestTraits mt;
  mt.input = traits.input;
  mt.input_size = traits.input_size;

  for (ModelVector::iterator it = _models.begin(); it != _models.end(); ++it)
  {
    (*it)->digest(mt);
  }

  // logDEBUG("Got NLS Probabilities: ["<<mt.field[MD_NONE_PROB]<<", "<<mt.field[MD_LONG_PROB]<<", "<<mt.field[MD_SHORT_PROB]<<"]");

  // select long or short position if appropriate:
  if (mt.field[MD_LONG_PROB] > 0.5)
  {
    traits.position = POS_LONG;
    traits.confidence = mt.field[MD_LONG_CONFIDENCE];
  }
  else if (mt.field[MD_SHORT_PROB] > 0.5)
  {
    traits.position = POS_SHORT;
    traits.confidence = mt.field[MD_SHORT_CONFIDENCE];
  }
}

void Strategy::destroyAllModels()
{
  for (ModelVector::iterator it = _models.begin(); it != _models.end(); ++it)
  {
    delete (*it);
  }
  _models.clear();
}

void Strategy::createModel(Model::CreationTraits &traits)
{
  Model* m = nullptr;
  if(traits.type == MODEL_NLS_NETWORK) {
    m = new NLSNetworkModel(traits);
  }

  THROW_IF(!m,"Invalid model type: "<< traits.type);

  // add the new model to the list:
  _models.push_back(m);
}
