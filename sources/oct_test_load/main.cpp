#include <octave/oct.h>
#include <sstream>
#include <windows.h>

#include <iostream>
#include <sstream>
// #include <windows.h>

#define CHECK(cond, msg) if(!(cond)) { \
  std::ostringstream os; \
  os << msg; \
  std::cout << os.str().c_str() <<std::endl; \
  return 0; \
}

#define logDEBUG(msg) std::cout << msg << std::endl;

int main() {
#if 0
  std::cout << "Hello World!" << std::endl;
  return 0;
#endif

  HMODULE h = LoadLibrary("nervMBP.dll");  //W:\\Cloud\\Projects\\nervtech\\bin\\x86\\
  CHECK(h != NULL,"Cannot load nervMBP.dll module.");

  // typedef void (* ShowInfoFunc)();

  // // We should be able to retrieve the train function:
  // ShowInfoFunc showInfo = (ShowInfoFunc) GetProcAddress(h, "showCudaInfo");

  
  typedef bool (* IsCudaSupportedFunc)();

  // We should be able to retrieve the train function:
  // IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, MAKEINTRESOURCE(1)); //"isCudaSupported");
  // IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, (LPCSTR)(1)); //"isCudaSupported");
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

  CHECK(FreeLibrary(h),"Cannot free nervMBP library.");  
}
