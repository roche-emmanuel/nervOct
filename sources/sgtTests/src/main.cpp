// each test module could contain no more then one 'main' file with init function defined
// alternatively you could define init function yourself
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "sgtCore Unit tests"

#include <boost/test/unit_test.hpp>

#include <sgtcore.h>
#include <sgt/Object.h>
#include <sgt/log/FileLogger.h>

class LoggerInit {
public:
  LoggerInit() {
    using namespace sgt;

    LogManager::instance().setDefaultLevelFlags(LogManager::TIME_STAMP);
    LogManager::instance().setDefaultTraceFlags(LogManager::TIME_STAMP);
    LogManager::instance().addLevelFlags(LogManager::SEV_FATAL,LogManager::FILE_NAME|LogManager::LINE_NUMBER);
    LogManager::instance().addLevelFlags(LogManager::SEV_ERROR,LogManager::FILE_NAME|LogManager::LINE_NUMBER);
    LogManager::instance().addLevelFlags(LogManager::SEV_WARNING,LogManager::FILE_NAME|LogManager::LINE_NUMBER);

    LogManager::instance().setVerbose(true);
    LogManager::instance().setNotifyLevel(LogManager::SEV_DEBUG0); // Log until DEBUG0 level only.

    LogManager::instance().addSink(new FileLogger("test.log",false,"main_log_file_sink"));
    
    // Remove the default console sink:
    // Since this one was added at some point during the static init of the sgtMX library.
    LogManager::instance().removeSink("default_console_sink");


    logDEBUG("Logging system initialized.");
  };

  ~LoggerInit() {
    logDEBUG("Logging system uninitialized.");
  };
};

LoggerInit logger_init;

BOOST_AUTO_TEST_SUITE( my_suite )

BOOST_AUTO_TEST_CASE( test_sanity )
{
	// Dummy sanity check test:	
  BOOST_CHECK( 1 == 1 );
}

BOOST_AUTO_TEST_CASE( test_referenced )
{
  sgt::Referenced* ref = new sgt::Referenced;

  BOOST_CHECK_EQUAL( ref->referenceCount(), 0 );

  BOOST_CHECK_EQUAL( ref->ref(), 1 );
  BOOST_CHECK_EQUAL( ref->unref(), 0 ); // will delete the object.

}

BOOST_AUTO_TEST_CASE( test_refptr )
{
	sgt::RefPtr<sgt::Referenced> ref = new sgt::Referenced;

  BOOST_CHECK_EQUAL( ref->referenceCount(), 1 );

  // Reset the ref pointer:
  ref = NULL;

  BOOST_CHECK( ref == NULL );

  sgt::RefPtr<sgt::Object> obj = new sgt::Object();

  BOOST_CHECK( obj.get() != NULL );  
}

BOOST_AUTO_TEST_CASE( test_observer )
{
  sgt::RefPtr<sgt::Referenced> ref = new sgt::Referenced;

  BOOST_CHECK_EQUAL( ref->referenceCount(), 1 );

  sgt::ObserverPtr<sgt::Referenced> obs = ref.get();

  BOOST_CHECK( obs != NULL );

  // Reset the ref pointer:
  ref = NULL;

  BOOST_CHECK( ref == NULL );
  BOOST_CHECK( obs.valid() == false );
}

BOOST_AUTO_TEST_CASE( test_log )
{
  trDEBUG("MyObject","This is an trace message.");
  logINFO("This is an info message.");
  logWARN("This is a warning message.");
  logERROR("This is an error message.");
  logFATAL("This is a fatal message.");
}

BOOST_AUTO_TEST_CASE( test_lambda_func )
{
  auto func = [] { logDEBUG("Hello world from lamda"); };
  func(); // now call the function
}

BOOST_AUTO_TEST_SUITE_END()
