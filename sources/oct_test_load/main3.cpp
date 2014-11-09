#include <octave/oct.h>
#include <sstream>
#include <windows.h>

#define CHECK(cond, msg) if(!(cond)) { \
  std::ostringstream os; \
  os << msg; \
  error(os.str().c_str()); \
  return result; \
}

#define logDEBUG(msg) std::cout << msg << std::endl;

// #endif

DEFUN_DLD (test_load, args, nargout,
           "test_load function using nervMBP module")
{
  int nargin = args.length ();

  octave_value_list result;

  HMODULE h = LoadLibrary("nervMBPProxy.dll");  //W:\\Cloud\\Projects\\nervtech\\bin\\x86\\
  CHECK(h != NULL,"Cannot load nervMBPProxy.dll module.");

  // typedef void (* ShowInfoFunc)();

  // // We should be able to retrieve the train function:
  // ShowInfoFunc showInfo = (ShowInfoFunc) GetProcAddress(h, "showCudaInfo");

  
  typedef bool (* IsCudaSupportedFunc)();

  // // We should be able to retrieve the train function:
  // // IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, MAKEINTRESOURCE(1)); //"isCudaSupported");
  // // IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, (LPCSTR)(1)); //"isCudaSupported");
  IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, "isCudaSupported");
  // if(isCudaSupported==NULL) {
  //   logDEBUG("Last error code is: "<<GetLastError());
  // }

  CHECK(isCudaSupported != NULL,"Cannot find isCudaSupported function");

  // Check that CUDA is supported:
  if(isCudaSupported()) {
    logDEBUG("CUDA is supported.");
  }
  else {
    logDEBUG("CUDA is not supported.");
  }

  // logDEBUG("Previous error: " << GetLastError());
  // CHECK(FreeLibrary(h),"Cannot free nervMBP library.");  
  if(!FreeLibrary(h)) {
    logDEBUG("FreeLibrary error: " << GetLastError());  
  }
  
 
  octave_stdout << "test_load has "
                << nargin << " input arguments and "
                << nargout << " output arguments.\n";

  return result;
}
