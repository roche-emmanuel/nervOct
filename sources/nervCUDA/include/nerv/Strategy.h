
#ifndef NERV_STRATEGY_H_
#define NERV_STRATEGY_H_

#include <nerv/Model.h>

namespace nerv
{

enum StrategyErrorCode
{
  ST_SUCCESS = 0,
  ST_EXCEPTION_OCCURED
};

enum StrategyTradeSymbol
{
  SYMBOL_EURAUD = 1,
  SYMBOL_EURCAD,
  SYMBOL_EURCHF,
  SYMBOL_EURGBP,
  SYMBOL_EURJPY,
  SYMBOL_EURUSD,
  SYMBOL_LAST
};

enum StrategyPriceType
{
  PRICE_OPEN = 1,
  PRICE_LOW,
  PRICE_HIGH,
  PRICE_CLOSE
};

enum StrategeyTradePosition
{
  POS_UNKNOWN, // used when system is not ready.
  POS_NONE,
  POS_LONG, // when buying
  POS_SHORT // when selling
};

class NERVCUDA_EXPORT Strategy
{
public:
  typedef double value_type;

  struct CreationTraits
  {
    CreationTraits()
      : num_features(0),
        target_symbol(0) {}

    int num_features;
    int target_symbol;
  };

  struct EvalTraits
  {
    EvalTraits()
      : inputs(nullptr), inputs_nrows(0), inputs_ncols(0),
      balance(nullptr) {}

    // input array containing 1 minute of data per column.
    // Each column contains the minute id, following by the
    // The format of the matrix should thus be inputs_nrows*inputs_ncols
    // Each column should contain: nmins s1_open s1_low s1_high s1_close s2_open s2_low ... sn_high sn_close
    // => We have for values for each symbol.
    value_type *inputs;
    int inputs_nrows;
    int inputs_ncols;

    // Buffer that may be provided to hold the actual balance value for each minute.
    // This buffer may also be NULL.
    value_type* balance;
  };

  struct DigestTraits
  {
    DigestTraits()
      : input(nullptr), input_size(0), 
      position(POS_UNKNOWN), confidence(0.0) {}

    value_type *input;
    int input_size;

    // Suggested position given the input.
    int position;

    // The confidence value should be between 0 and 1 and
    // should be a measure of how much we trust the decision that was taken.
    value_type confidence;
  };

public:
  Strategy(const CreationTraits &traits);
  virtual ~Strategy();

  inline int getID()
  {
    return _id;
  }

  void evaluate(EvalTraits &traits);

  void digest(DigestTraits& traits);

  void createModel(Model::CreationTraits& traits);
  void destroyAllModels();

protected:
  typedef std::vector<Model*> ModelVector;
  // List of all models used for this strategy:
  ModelVector _models; 

  // Retrieve the current price for a given symbol.
  // if the default value -1 is used for symbol, then the current
  // target symbol is used.
  value_type getPrice(value_type *iptr, int type, int symbol = 0) const;

  int _id;

  // The symbol that we want to trade:
  // The symbol value should be taken from the
  // StrategyTradeSymbol enumeration.
  int _target_symbol;
};

};

#endif
