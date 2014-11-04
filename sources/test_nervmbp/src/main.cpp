// each test module could contain no more then one 'main' file with init function defined
// alternatively you could define init function yourself
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "NervMBP tests"

#include <boost/test/unit_test.hpp>

#include <nervmbp.h>
#include <windows.h>

BOOST_AUTO_TEST_SUITE( basic_suite )

BOOST_AUTO_TEST_CASE( test_sanity )
{
  // Dummy sanity check test: 
  BOOST_CHECK( 1 == 1 );
}

BOOST_AUTO_TEST_CASE( test_loading_module )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervMBP.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_retrieving_cuda_supported_function )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* IsCudaSupportedFunc)();

  // We should be able to retrieve the train function:
  IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, "isCudaSupported");
  BOOST_CHECK(isCudaSupported != nullptr);

  // Check that CUDA is supported:
  // BOOST_CHECK(isCudaSupported() == true);

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_check_cuda_is_supported )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* IsCudaSupportedFunc)();

  // We should be able to retrieve the train function:
  IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, "isCudaSupported");
  BOOST_CHECK(isCudaSupported != nullptr);

  // Check that CUDA is supported:
  BOOST_CHECK(isCudaSupported() == true);

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( should_show_cuda_infos )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* ShowInfoFunc)();

  // We should be able to retrieve the train function:
  ShowInfoFunc showInfo = (ShowInfoFunc) GetProcAddress(h, "showCudaInfo");
  BOOST_CHECK(showInfo != nullptr);

  // showInfo();

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( should_be_able_to_call_trainBP )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* TrainFunc)(const std::vector<int>& lsizes, 
    int num_inputs, double* inputs,
    int num_outputs, double* outputs,
    int num_weights, double* weights);

  // We should be able to retrieve the train function:
  TrainFunc trainBP = (TrainFunc) GetProcAddress(h, "trainBP");
  BOOST_CHECK(trainBP != nullptr);

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
