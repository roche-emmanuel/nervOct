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

class CUDAManager {
protected:
  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

  typedef void (*CostFuncCPU)(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
    double* nn_params, double* X, double* yy, double lambda, double* activation, double* inputs, double& J, double* gradients, double* deltas);

  typedef void (*CostFunc)(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
    double* nn_params, double* X, double* yy, double lambda, double* inputs, double& J, double* gradients, double* deltas);

public:
  CUDAManager() {
    logDEBUG("Loading nervCUDA...");
    _h = LoadLibrary("nervCUDA.dll");
    if(!_h) {
      error("ERROR: cannot load nervCUDA library! err=%d",GetLastError());
    }

    // Try loading the functions of interest:
    _multMat = (MultMatFunc) GetProcAddress(_h, "multiplyMatrices");
    if(!_multMat) {
      error("ERROR: cannot find multiplyMatrices method! err=%d",GetLastError());
    }

    _costFuncCPU = (CostFuncCPU) GetProcAddress(_h, "costFuncCPU");
    if(!_costFuncCPU) {
      error("ERROR: cannot find costFuncCPU method! err=%d",GetLastError());
    }

    _costFunc = (CostFunc) GetProcAddress(_h, "costFunc");
    if(!_costFunc) {
      error("ERROR: cannot find costFunc method! err=%d",GetLastError());
    }
  }

  ~CUDAManager() {
    logDEBUG("Unloading nervCUDA module...")
    BOOL res = FreeLibrary(_h);
    if(!res) {
      error("ERROR: cannot free library! err=%d",GetLastError());
    }
  }

  inline void multMat(const Matrix& A, const Matrix& B, Matrix& C, bool tpA = false, bool tpB = false) {
    if(tpA && tpB) {
      error("Dual transpose in multMat not supported yet.");
    }

    _multMat(A.dim1(),A.dim2(),A.data(),B.dim1(),B.dim2(),B.data(),(double*)C.data(),tpA,tpB);
  }

  inline Matrix multMat(const Matrix& A, const Matrix& B, bool tpA = false, bool tpB = false) {
    if(tpA && tpB) {
      error("Dual transpose in multMat not supported yet.");
    }

    Matrix C = Matrix(tpA ? A.dim2() : A.dim1(),tpB ? B.dim1() : B.dim2());
    _multMat(A.dim1(),A.dim2(),A.data(),B.dim1(),B.dim2(),B.data(),(double*)C.data(),tpA,tpB);
    return C;
  }

#if 0
  inline void costFuncCPU(const Matrix& lsizes_mat, const Matrix& nn_params, const Matrix& X, const Matrix& yy, double lambda, Matrix& activation, Matrix& inputs) {
    unsigned int nl = lsizes_mat.numel();
    unsigned int* lsizes = new unsigned int[nl];
    for(unsigned int i=0;i<nl;++i) {
      lsizes[i] = lsizes_mat(i);
    }

    double J = 0.0;
    // TODO : this method call needs update grads and deltas parameters are missing!
    _costFuncCPU(nl, lsizes, X.dim1(), (double*)nn_params.data(), (double*)X.data(), (double*)yy.data(), lambda, (double*)activation.data(), (double*)inputs.data(),J);

    delete [] lsizes;
  }
#endif

  inline Matrix costFunc(const Matrix& lsizes_mat, const Matrix& nn_params, const Matrix& X, const Matrix& yy, double lambda, Matrix& inputs, double& J) {
    unsigned int nl = lsizes_mat.numel();
    unsigned int* lsizes = new unsigned int[nl];
    for(unsigned int i=0;i<nl;++i) {
      lsizes[i] = lsizes_mat(i);
    }

    Matrix grads = Matrix(nn_params.numel(),1);
    _costFunc(nl, lsizes, X.dim1(), (double*)nn_params.data(), (double*)X.data(), (double*)yy.data(), lambda, (double*)inputs.data(), J, (double*)grads.data(), NULL);    // memcpy((double*)grads.data(),gradients,sizeof(double)*nn_params.numel());

    delete [] lsizes;
    return grads;
  }

protected:
  HMODULE _h;
  MultMatFunc _multMat;
  CostFuncCPU _costFuncCPU;
  CostFunc _costFunc;
};

CUDAManager g_cuda;

#define USE_GPU_INPUTS

DEFUN_DLD (nn_cost_function_cuda, args, nargout,
           "nn_cost_function_cuda function providing C++ implementation of nnCostFunction")
{
  octave_value_list result;

  // we ecpect to receive 5 arguments:
  int nargin = args.length();
  CHECK(nargin==5,"nn_cost_function_cuda: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK(args(0).is_matrix_type(),"nn_cost_function_cuda: nn_params (arg 0) should be a matrix type");
  CHECK(args(1).is_matrix_type(),"nn_cost_function_cuda: layer_sizes (arg 1) should be a matrix type");
  CHECK(args(2).is_matrix_type(),"nn_cost_function_cuda: X (arg 2) should be a matrix type");
  CHECK(args(3).is_matrix_type(),"nn_cost_function_cuda: yy (arg 3) should be a matrix type");
  CHECK(args(4).is_double_type(),"nn_cost_function_cuda: lambda (arg 4) should be a double");

  Matrix nn_params = args(0).matrix_value();
  Matrix layer_sizes = args(1).matrix_value();
  Matrix X = args(2).matrix_value();
  Matrix yy = args(3).matrix_value();
  double lambda = args(4).double_value();

  // Check how may layers we have:
  octave_idx_type nl = layer_sizes.numel();

  // when we have nl layers, we have nl-1 weights matrices.
  // That we should rebuild here:
  octave_idx_type nt = nl-1;

  // Number of samples:
  octave_idx_type nsamples = X.dim1();

  // compute expected activation and input lengths:
  octave_idx_type dbg_act_count_exp = 0;
  octave_idx_type dbg_input_count_exp = 0;
  for(octave_idx_type i=0;i<nl;++i) {
    dbg_act_count_exp += layer_sizes(i)+1;
  }

  for(octave_idx_type i=0;i<nt;++i) {
    dbg_input_count_exp += layer_sizes(i+1);
  }

  dbg_act_count_exp = nsamples*dbg_act_count_exp;
  dbg_input_count_exp = nsamples*dbg_input_count_exp;

  Matrix input_array = Matrix(dbg_input_count_exp,1);
  memset((void*)input_array.data(),0,sizeof(double)*dbg_input_count_exp);

  // compute the expected values:
  double Jcuda = 0.0;
  Matrix grads_cuda = g_cuda.costFunc(layer_sizes, nn_params, X, yy, lambda, input_array, Jcuda);


  // Reshape nn_params back into the parameters Thetas i, the weight matrices for our neural network:
  typedef std::vector<Matrix> MatrixVector;
  MatrixVector Thetas;
  MatrixVector Activation;
  MatrixVector Deltas(nl);
  MatrixVector Inputs;

  // First we need to compute h(x) for each example.
  Matrix a = Matrix(nsamples,X.dim2()+1,1.0);
  double* aptr = ((double*)a.data())+nsamples;
  octave_idx_type num = X.numel();
  memcpy((void*)aptr,(void*)X.data(),sizeof(double)*num);
  //a = a.transpose();  //DISCARDED

  Activation.push_back(a);

  // add a dummy value in the input vector:
  Inputs.push_back(Matrix());

  const double* ptr = (double*)nn_params.data();
  octave_idx_type pos = 0;

  octave_idx_type dbg_act_count = a.numel();
  octave_idx_type dbg_input_count = 0;
  
  octave_idx_type input_offset = 0;
  const double* input_ptr = input_array.data();

  for(octave_idx_type i=0; i<nt;++i) {
    octave_idx_type n = layer_sizes.elem(i+1);
    octave_idx_type m = layer_sizes.elem(i)+1;
    octave_idx_type count = n*m;

    Thetas.push_back(Matrix(n,m));

    Matrix& theta = Thetas[i];    
    memcpy((void*)theta.data(),ptr,sizeof(double)*n*m);
    ptr += count;
    pos += count;

    // Once the theta matrix is ready we can compute the activation:

    // Retrieve the current activation value:
    a = Activation[i];

    // Then we can compute the total input for the next layer:
    CHECK(theta.dim2()==a.dim2(),"Mismatch on forward propagation on level "<<i<<", "<< theta.dim2()<<"!="<<a.dim2())

    // We have to store the z input for the backpropagation later:

    // We use the inputs from the GPU computation instead:
    // Note that the Z matrix here already contain the sigmoid computation.
    Matrix z = Matrix(theta.dim1(),nsamples);
    memcpy((void*)z.data(),input_ptr,sizeof(double)*z.numel());
    input_ptr += z.numel();

    Inputs.push_back(z);
    a = z;

    dbg_input_count += z.numel();

    Matrix ac = Matrix(nsamples,1,1.0);
    ac = ac.append(a.transpose());
    //ac = ac.transpose(); //DISCARDED

    // Also save the activation value:
    Activation.push_back(ac);

    dbg_act_count += ac.numel();  
  }

  // Compare the observed counts with the expected values:
  CHECK(dbg_act_count==dbg_act_count_exp,"Mismatch in act count: "<<dbg_act_count<<"!="<<dbg_act_count_exp);
  CHECK(dbg_input_count==dbg_input_count_exp,"Mismatch in input count: "<<dbg_input_count<<"!="<<dbg_input_count_exp);

  // note that the pos variable is now on the next index available:
  CHECK(pos== nn_params.dim1(),"Mismatch in unrolled vector size " << pos <<"!="<< nn_params.dim1());

  // Now the final result is stored in a, the activation vector (eg. output) of the last layer.
  // Thus we can rename this output as hx:
  Matrix hx = a;

  // Now we build the cost "matrix":
  // cmat = yy .* log(hx) + (1-yy) .* log(1-hx);

  // Then we perform the summation on all classes and on all examples:
  double* hptr = (double*)hx.data();
  double* yptr = (double*)yy.data();

  result.append(Jcuda);

  // Part 2: 
  // now implementing back propagation.

  // We start with the latest delta value:
  Deltas[nl-1] = Matrix(hx.dim1(),hx.dim2());
  num = hx.numel();
  double* dest = (double*)Deltas[nl-1].data();
  hptr = (double*)hx.data();
  yptr = (double*)yy.data();

  for(octave_idx_type i=0;i<num;++i) {
    (*dest++) = (*hptr++) - (*yptr++);    
  }

  Matrix delta = Deltas[nl-1];

  for(octave_idx_type i=nt-1;i>0;i--) {
    // We have to accumulate the correction for each sample
    // Matrix mat = Thetas[i].transpose() * delta;
    Matrix mat = g_cuda.multMat(Thetas[i],delta,true);

    // We need to drop the first ROW of this matrix,
    // So we take the transpose, remove the first col,
    // then transpose again:
    mat = mat.transpose();

    const double* dptr = mat.data();

    // Then we have to discard the first row, so we just offset the pointer:
    dptr += mat.dim1();

    Matrix mat2 = Matrix(mat.dim1(),mat.dim2()-1);
    double* dest = (double*)mat2.data();
    memcpy(dest,dptr,sizeof(double)*mat2.dim1()*mat2.dim2());

    // transpose back:
    mat = mat2.transpose();
    dptr = (double*)mat.data();

    // And we multiply by the sigmoid gradient of the previous activation value:
    Matrix& z = Inputs[i];
    const double* zptr = z.data();

    // Prepare the final detal matrix:
    delta = Matrix(z.dim1(),z.dim2());
    dest = (double*)delta.data();
    num = z.numel();
    double sig;
    for(octave_idx_type j=0;j<num;++j) {
      sig = (*zptr++);
      (*dest++) = (*dptr++)*sig*(1.0-sig);
    }

    // Save that delta value:
    Deltas[i] = delta;
  }

  // result.append(Activation[nt-1]);
  // result.append(Deltas[nt]);

  // Now we can compute the theta grad values:
  octave_idx_type np = nn_params.dim1();
  Matrix grad = Matrix(np,1,0.0);
  double* gptr = (double*)grad.data();

  for(octave_idx_type i=0;i<nt;++i) {
    // Also add the regularization term at the same time:
    Matrix& Theta = Thetas[i];
    octave_idx_type n1 = Theta.dim1();
    octave_idx_type n2 = Theta.dim2();
    octave_idx_type count = n1*n2;

    Matrix reg = Matrix(n1,n2);
    memcpy((void*)reg.data(),(void*)Theta.data(),sizeof(double)*count);
    double* rptr = (double*)reg.data();
    for(octave_idx_type j=0;j<n1;++j) {
      // Reset the first column to zero:
      (*rptr++) = 0.0;
    }

    Matrix mat = (g_cuda.multMat(Deltas[i+1],Activation[i]) + lambda * reg)/nsamples; 

    memcpy((void*)gptr,(void*)mat.data(),sizeof(double)*count);
    gptr += count;
  }

  // Compare the gradients with the CUDA version:
  // octave_idx_type count = grads_cuda.dim1();
  // CHECK(count==grad.dim1(),"Mismatch in gradients dimensions.");

  // for(octave_idx_type j=0;j<count;++j) {
  //   double v1 = grads_cuda(j);
  //   double v2 = grad(j);
  //   CHECK(abs(v1-v2)<1e-10,"Mismatch in gradient at index "<<j<<": "<<v1<<"!="<<v2);
  // }

  result.append(grad);

  return result;
}
