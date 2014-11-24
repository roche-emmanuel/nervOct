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


class NERVManager {
protected:
  typedef void (* TrainFunc)(unsigned int nl, unsigned int nsamples, unsigned int nparams, 
    unsigned int* lsizes, double* X, double* yy, double* init_params, 
    double lambda, unsigned int maxiter, double* params);

  typedef std::vector<double> CostList;

public:
  NERVManager() {
    logDEBUG("Loading nervCUDA...");
    _h = LoadLibrary("nervCUDA.dll");
    if(!_h) {
      error("ERROR: cannot load nervCUDA library! err=%d",GetLastError());
    }

    // Try loading the functions of interest:
    // _cgtrain = (TrainFunc) GetProcAddress(_h, "cgtrainCPU");
    _cgtrain = (TrainFunc) GetProcAddress(_h, "cgtrain");
    if(!_cgtrain) {
      error("ERROR: cannot find cgtrain method! err=%d",GetLastError());
    }
  }

  ~NERVManager() {
    logDEBUG("Unloading nervCUDA module...")
    BOOL res = FreeLibrary(_h);
    if(!res) {
      error("ERROR: cannot free library! err=%d",GetLastError());
    }
  }

  inline void cgtrain(const Matrix& lsizes_mat, const Matrix& X, const Matrix& yy, const Matrix& init_params, double lambda, unsigned int maxiter,
    Matrix& params) {

    unsigned int nl = lsizes_mat.numel();
    unsigned int* lsizes = new unsigned int[nl];
    for(unsigned int i=0;i<nl;++i) {
      lsizes[i] = lsizes_mat(i);
    }

    unsigned int np = 0;
    unsigned int nt = nl-1; // number of matrices evolved.

    for(unsigned int i=0;i<nt;++i) {
      np += lsizes[i+1]*(lsizes[i]+1);
    }

    if(init_params.numel()!=np || params.numel()!=np) {
      error("Invalid number of parameters: %d!=%d",np,init_params.numel());
    }

    unsigned int nsamples = X.dim1();
    unsigned int nparams = np;

    // CostList clist;

    _cgtrain(nl, nsamples, nparams, lsizes, (double*)X.data(), (double*)yy.data(), (double*)init_params.data(), lambda, maxiter, (double*)params.data()); //,clist

    // if(clist.empty()) {
    //   error("Cost list is empty after cgtrain call.");
    // }

    // unsigned int num = clist.size();
    // Matrix costs = Matrix(num,1);
    // std::copy(clist.begin(),clist.end(),(double*)costs.data());

    delete [] lsizes;

    // return costs;
  }

protected:
  HMODULE _h;
  TrainFunc _cgtrain;
};

NERVManager g_nerv;

DEFUN_DLD (nn_cg_train, args, nargout,
           "nn_cg_train function providing C++ implementation of nn_cg_train")
{
  octave_value_list result;

  // we ecpect to receive 5 arguments:
  int nargin = args.length();
  CHECK(nargin==6,"nn_cg_train: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK(args(0).is_matrix_type(),"nn_cg_train: layer_sizes (arg 0) should be a matrix type");
  CHECK(args(1).is_matrix_type(),"nn_cg_train: X (arg 1) should be a matrix type");
  CHECK(args(2).is_matrix_type(),"nn_cg_train: yy (arg 2) should be a matrix type");
  CHECK(args(3).is_matrix_type(),"nn_cg_train: init_params (arg 3) should be a matrix type");
  CHECK(args(4).is_double_type(),"nn_cg_train: lambda (arg 4) should be a double");
  CHECK(args(5).is_double_type(),"nn_cg_train: maxiter (arg 5) should be a double");

  Matrix layer_sizes = args(0).matrix_value();
  Matrix X = args(1).matrix_value();
  Matrix yy = args(2).matrix_value();
  Matrix init_params = args(3).matrix_value();
  double lambda = args(4).double_value();
  int maxiter = (int)args(5).double_value();

  // logDEBUG("Received number of maxiter: "<<maxiter);

  // Prepare the result matrices:  
  Matrix params = Matrix(init_params.numel(),1);
  // Matrix costs;

  g_nerv.cgtrain(layer_sizes, X, yy, init_params, lambda, maxiter, params);

  result.append(params);
  // result.append(costs);

  return result;
}
