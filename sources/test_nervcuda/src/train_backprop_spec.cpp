#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>

#include <boost/chrono.hpp>

#define logDEBUG(msg) std::cout << msg << std::endl;

BOOST_AUTO_TEST_SUITE( train_backprop )


BOOST_AUTO_TEST_CASE( test_sanity_loading_module )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_retrieve_train_method )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}


BOOST_AUTO_TEST_SUITE_END()
