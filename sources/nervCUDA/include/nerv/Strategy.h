
#ifndef NERV_STRATEGY_H_
#define NERV_STRATEGY_H_

namespace nerv
{

enum StrategyErrorCode {
	ST_SUCCESS = 0,
	ST_EXCEPTION_OCCURED
};

class Strategy
{
public:
  typedef double value_type;

  enum TradePosition {
  	POS_UNKNONW, // used when system is not ready.
  	POS_NONE,
  	POS_LONG, // when buying
  	POS_SHORT // when selling
  };

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
      : inputs(nullptr), inputs_nrows(0), inputs_ncols(0) {}

    // input array containing 1 minute of data per column.
    // Each column contains the minute id, following by the
    // The format of the matrix should thus be inputs_nrows*inputs_ncols
    // Each column should contain: nmins s1_open s1_low s1_high s1_close s2_open s2_low ... sn_high sn_close
    // => We have for values for each symbol.
    value_type *inputs;
    int inputs_nrows;
    int inputs_ncols;
  };

public:
  Strategy(const CreationTraits &traits);
  ~Strategy();

  inline int getID()
  {
    return _id;
  }

  void evaluate(EvalTraits &traits);

protected:
  int _id;
};

};

#endif
