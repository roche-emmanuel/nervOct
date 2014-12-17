#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

#include <nervcuda.h>
#include <nerv/StrategyInterface.h>


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
  traits.target_symbol = 6;

  int id = intf.create_strategy(traits);

  Strategy::EvalTraits evt;

  evt.inputs_nrows = random_int(20, 30);
  evt.inputs_ncols = random_int(20, 30);

  int count = evt.inputs_ncols * evt.inputs_nrows;
  evt.inputs = new Strategy::value_type[count];
  for (int c = 0; c < evt.inputs_ncols; ++c)
  {
    for (int r = 0; r < evt.inputs_nrows; ++r)
    {
      // if we are on the first row, then we should use the value of c as
      // an indication of the current minute in the week:
      evt.inputs[evt.inputs_nrows*c+r] = r==0 ? c : random_real(0.0, 10.0);
    }
  }

  BOOST_CHECK(intf.evaluate_strategy(id, evt) == ST_SUCCESS);

  delete [] evt.inputs;
  
  BOOST_CHECK(intf.destroy_strategy(id) == ST_SUCCESS);
}

BOOST_AUTO_TEST_SUITE_END()
