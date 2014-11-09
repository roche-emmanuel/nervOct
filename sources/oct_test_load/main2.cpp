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

#if 0

class NervManager {
public:
  NervManager() {
    logDEBUG("Loading NervMBP module...")
    _h = LoadLibrary("sgtCore.dll");
    if(!_h) {
      logDEBUG("ERROR: cannot load sgtcore library! err="<<GetLastError())
    }
    _h = LoadLibrary("CUDART32_65.dll");
    if(!_h) {
      logDEBUG("ERROR: cannot load cudart library! err="<<GetLastError())
    }
    // _h = LoadLibrary("nervMBP2.dll");
    // if(!_h) {
    //   logDEBUG("ERROR: cannot load nervmbp library! err="<<GetLastError())
    // }
  }

  ~NervManager() {
    logDEBUG("Unloading NervMBP module...")
    BOOL res = FreeLibrary(_h);
    if(!res) {
      logDEBUG("ERROR: cannot free library! err="<<GetLastError())
    }
  }

protected:
  HMODULE _h;
};

NervManager g_manager;

#endif

DEFUN_DLD (test_load, args, nargout,
           "test_load function using nervMBP module")
{
  int nargin = args.length ();

  octave_value_list result;

  logDEBUG("Trying to load sgtCore...")
  HMODULE h0 = LoadLibrary("sgtCore.dll");  //W:\\Cloud\\Projects\\nervtech\\bin\\x86\\
  CHECK(h0 != NULL,"Cannot load sgtCore.dll module.");

  logDEBUG("Trying to load nervMBP...")
  HMODULE h = LoadLibrary("nervMBP.dll");  //W:\\Cloud\\Projects\\nervtech\\bin\\x86\\
  CHECK(h != NULL,"Cannot load nervMBP.dll module.");

  // typedef void (* ShowInfoFunc)();

  // // We should be able to retrieve the train function:
  // ShowInfoFunc showInfo = (ShowInfoFunc) GetProcAddress(h, "showCudaInfo");

  
  typedef bool (* IsCudaSupportedFunc)();

  // // We should be able to retrieve the train function:
  // // IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, MAKEINTRESOURCE(1)); //"isCudaSupported");
  // // IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, (LPCSTR)(1)); //"isCudaSupported");
  IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, "isCudaSupported");
  // // if(isCudaSupported==NULL) {
  // //   logDEBUG("Last error code is: "<<GetLastError());
  // // }

  CHECK(isCudaSupported != NULL,"Cannot find isCudaSupported function");

  // // Check that CUDA is supported:
  if(isCudaSupported()) {
    logDEBUG("CUDA is supported.");
  }
  else {
    logDEBUG("CUDA is not supported.");
  }

  // logDEBUG("Previous error: " << GetLastError());
  CHECK(FreeLibrary(h),"Cannot free nervMBP library.");  
  CHECK(FreeLibrary(h0),"Cannot free sgtCore library.");  
  // if(FreeLibrary(h)==FALSE) {
  //   logDEBUG("FreeLibrary error: " << GetLastError());  
  // }
  
 
  octave_stdout << "test_load has "
                << nargin << " input arguments and "
                << nargout << " output arguments.\n";

  return result;
}
