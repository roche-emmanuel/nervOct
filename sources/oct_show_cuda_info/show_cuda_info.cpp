#include <octave/oct.h>
#include <sstream>
#include <windows.h>

#define CHECK(cond, msg) if(!(cond)) { \
  std::ostringstream os; \
  os << msg; \
  error(os.str().c_str()); \
  return result; \
}

#define logDEBUG(msg) octave_stdout << msg << std::endl;


DEFUN_DLD (show_cuda_info, args, nargout,
           "show_cuda_info function using nervMBP module")
{
  octave_value_list result;
  // Now we need to perform the actual computation.
  // So we need to load the nervMBP library:
  HMODULE h = LoadLibrary("nervMBP.dll"); 
  CHECK(h != NULL,"show_cuda_info: Cannot load nervMBP.dll module.");

  typedef void (* ShowInfoFunc)();

  // We should be able to retrieve the train function:
  ShowInfoFunc showInfo = (ShowInfoFunc) GetProcAddress(h, "showCudaInfo");
  CHECK(showInfo != NULL,"Cannot find showCudaInfo function");

  showInfo();

  CHECK(FreeLibrary(h),"show_cuda_info: Cannot free nervMBP library.");

  return result;
}
