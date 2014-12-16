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

BOOST_AUTO_TEST_CASE( test_get_strategy_manager )
{
  StrategyInterface intf;

  StrategyManager& sm = intf.get_strategy_manager();
}

BOOST_AUTO_TEST_CASE( test_strategy_creation )
{
  StrategyInterface intf;
  int id = intf.create_strategy();
  BOOST_CHECK(id==1);
  intf.destroy_strategy(id);
}

BOOST_AUTO_TEST_SUITE_END()
