#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <sstream>
#include <windows.h>

#define CHECK(cond, msg) if(!(cond)) { \
    std::ostringstream os; \
    os << msg; \
    error(os.str().c_str()); \
    return result; \
  }

#define logDEBUG(msg) octave_stdout << msg << std::endl;

class NERVManager {
protected:
  // typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
  //   unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

public:
  NERVManager() {
    logDEBUG("Loading nervCUDA...");
    _h = LoadLibrary("nervCUDA.dll");
    if(!_h) {
      error("ERROR: cannot load nervCUDA library! err=%d",GetLastError());
    }

    // // Try loading the functions of interest:
    // _multMat = (MultMatFunc) GetProcAddress(_h, "matmult");
    // if(!_multMat) {
    //   error("ERROR: cannot find matmult method! err=%d",GetLastError());
    // }
  }

  ~NERVManager() {
    logDEBUG("Unloading nervCUDA module...")
    BOOL res = FreeLibrary(_h);
    if(!res) {
      error("ERROR: cannot free library! err=%d",GetLastError());
    }
  }

  // inline void multMat(const Matrix& A, const Matrix& B, Matrix& C, bool tpA = false, bool tpB = false) {
  //   if(tpA && tpB) {
  //     error("Dual transpose in multMat not supported yet.");
  //   }

  //   _multMat(A.dim1(),A.dim2(),A.data(),B.dim1(),B.dim2(),B.data(),(double*)C.data(),tpA,tpB);
  // }

protected:
  HMODULE _h;
  // MultMatFunc _multMat;
};

NERVManager g_nerv;

DEFUN_DLD (nn_gradient_descent, args, nargout,
           "nn_gradient_descent function providing C++ implementation of Gradient Descent")
{
  octave_value_list result;

  // we expect to receive 1 arguments:
  int nargin = args.length();
  CHECK(nargin == 1, "nn_gradient_descent: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK(args(0).is_map(), "nn_gradient_descent: desc (arg 0) should be a structure type");

  // Try retrieving the structure:
  octave_scalar_map desc = args(0).scalar_map_value();

  // The desc structure should contain an lsizes element.
  octave_value lsizes_val = desc.contents("lsizes");
  CHECK(lsizes_val.is_defined(), "nn_gradient_descent: lsizes value is not defined");

  // Prepare a BPTraits object:

  return result;
}

