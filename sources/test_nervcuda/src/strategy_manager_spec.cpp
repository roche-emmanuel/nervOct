#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

#include <nervcuda.h>
#include <nerv/StrategyInterface.h>
#include <nerv/TrainingSet.h>


#include <boost/chrono.hpp>

using namespace nerv;

BOOST_AUTO_TEST_SUITE( strategy_manager_suite )

// BOOST_AUTO_TEST_CASE( test_get_strategy_manager )
// {
//   StrategyInterface intf;

//   StrategyManager& sm = intf.get_strategy_manager();
// }

BOOST_AUTO_TEST_CASE( test_strategy_creation )
{
  StrategyInterface intf;
  Strategy::CreationTraits traits;
  traits.num_features = 1441;
  traits.target_symbol = 6;

  int id = intf.create_strategy(traits);

  BOOST_CHECK(id == 1);
  BOOST_CHECK(intf.destroy_strategy(id) == ST_SUCCESS);
}

BOOST_AUTO_TEST_CASE( test_strategy_evaluate )
{
  StrategyInterface intf;
  Strategy::CreationTraits traits;
  traits.num_features = 1441;
  traits.target_symbol = SYMBOL_EURUSD;

  int id = intf.create_strategy(traits);

  Strategy::EvalTraits evt;

  evt.inputs_nrows = random_int(20, 30);
  evt.inputs_ncols = random_int(20, 30);
  evt.prices_nrows = 4;
  evt.prices_ncols = evt.inputs_ncols;

  int count = evt.inputs_ncols * evt.inputs_nrows;
  evt.inputs = new Strategy::value_type[count];
  evt.prices = new Strategy::value_type[count];

  for (int c = 0; c < evt.inputs_ncols; ++c)
  {
    for (int r = 0; r < evt.inputs_nrows; ++r)
    {
      // if we are on the first row, then we should use the value of c as
      // an indication of the current minute in the week:
      evt.inputs[evt.inputs_nrows * c + r] = r == 0 ? c : random_real(0.0, 10.0);
    }
    for (int r = 0; r < evt.prices_nrows; ++r)
    {
      evt.prices[evt.prices_nrows * c + r] = random_real(0.0, 10.0);
    }    
  }

  BOOST_CHECK(intf.evaluate_strategy(id, evt) == ST_SUCCESS);

  delete [] evt.inputs;
  delete [] evt.prices;

  BOOST_CHECK(intf.destroy_strategy(id) == ST_SUCCESS);
}

class TestStrategy : public Strategy
{
public:
  TestStrategy(const Strategy::CreationTraits &traits) : Strategy(traits) {}

  Strategy::value_type test_getPrice(value_type *iptr, int type) const
  {
    return getPrice(iptr, type);
  }
};

BOOST_AUTO_TEST_CASE( test_strategy_get_price )
{
  Strategy::CreationTraits traits;
  traits.target_symbol = SYMBOL_EURAUD;

  TestStrategy *s = new TestStrategy(traits);

  // test the getPrice Method:
  int count = 4;
  typedef Strategy::value_type value_t;

  value_t *iptr = new value_t[count];
  for (int i = 0; i < count; ++i)
  {
    iptr[i] = (value_t)(i);
  }

  value_t pred = 0.0;
  value_t val;
  for (int pt = 0; pt < 4; ++pt)
  {
    val = s->test_getPrice(iptr, pt);
    BOOST_CHECK_MESSAGE(abs(val - pred) <= 1e-10, "Mismatch in price value for ptype=" << pt << ": " << val << "!=" << pred);
    pred += 1.0;
  }

  delete [] iptr;
  delete s;
}

BOOST_AUTO_TEST_CASE( test_strategy_add_nls_network )
{
  StrategyInterface intf;
  Strategy::CreationTraits traits;
  traits.num_features = 1441;
  traits.target_symbol = SYMBOL_EURUSD;

  TrainingSet<double> tr(3, 5, 3, 6, 50, 100, 3);

  int id = intf.create_strategy(traits);

  Model::CreationTraits mt;
  mt.type = MODEL_NLS_NETWORK;

  unsigned int nl = 3;
  unsigned int nt = 2;

  mt.nl = tr.nl();
  mt.lsizes = tr.lsizes();

  // parameters are missing for this call:
  BOOST_CHECK(intf.add_strategy_model(id, mt) == ST_EXCEPTION_OCCURED);

  mt.params = tr.params();
  unsigned int nf = mt.lsizes[0];
  mt.mu = tr.createArray(nf);
  mt.sigma = tr.createArray(nf);
  for (unsigned int i = 0; i < nf; ++i)
  {
    mt.mu[i] = random_real(0.0, 1.0);
    mt.sigma[i] = random_real(0.01, 10.0);
  }

  BOOST_CHECK(intf.add_strategy_model(id, mt) == ST_SUCCESS);

  BOOST_CHECK(intf.destroy_strategy(id) == ST_SUCCESS);
}

BOOST_AUTO_TEST_CASE( test_strategy_eval_with_model )
{
  srand((unsigned int)time(nullptr));

  StrategyInterface intf;
  Strategy::CreationTraits traits;
  traits.num_features = 1441;
  traits.target_symbol = SYMBOL_EURUSD;

  Strategy::EvalTraits evt;

  evt.inputs_nrows = random_int(20, 30);
  evt.inputs_ncols = random_int(20, 30);
  evt.prices_nrows = 4;
  evt.prices_ncols = evt.inputs_ncols;

  TrainingSet<double> tr(3, 5, 3, 6, 50, 100, 3, evt.inputs_nrows);

  int id = intf.create_strategy(traits);

  Model::CreationTraits mt;
  mt.type = MODEL_NLS_NETWORK;

  unsigned int nl = 3;
  unsigned int nt = 2;

  mt.nl = tr.nl();
  mt.lsizes = tr.lsizes();

  // parameters are missing for this call:
  BOOST_CHECK(intf.add_strategy_model(id, mt) == ST_EXCEPTION_OCCURED);

  mt.params = tr.params();
  unsigned int nf = mt.lsizes[0];
  mt.mu = tr.createArray(nf);
  mt.sigma = tr.createArray(nf);
  for (unsigned int i = 0; i < nf; ++i)
  {
    mt.mu[i] = random_real(0.0, 1.0);
    mt.sigma[i] = random_real(0.01, 10.0);
  }

  BOOST_CHECK(intf.add_strategy_model(id, mt) == ST_SUCCESS);

  int count = evt.inputs_ncols * evt.inputs_nrows;
  evt.inputs = new Strategy::value_type[count];
  evt.prices = new Strategy::value_type[count];

  for (int c = 0; c < evt.inputs_ncols; ++c)
  {
    for (int r = 0; r < evt.inputs_nrows; ++r)
    {
      // if we are on the first row, then we should use the value of c as
      // an indication of the current minute in the week:
      evt.inputs[evt.inputs_nrows * c + r] = r == 0 ? c : random_real(0.0, 10.0);
    }
    for (int r = 0; r < evt.prices_nrows; ++r)
    {
      evt.prices[evt.prices_nrows * c + r] = random_real(0.0, 10.0);
    }
  }

  BOOST_CHECK(intf.evaluate_strategy(id, evt) == ST_SUCCESS);

  delete [] evt.inputs;
  delete [] evt.prices;

  BOOST_CHECK(intf.destroy_strategy(id) == ST_SUCCESS);
}

BOOST_AUTO_TEST_SUITE_END()
