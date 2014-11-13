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


DEFUN_DLD (nn_cost_function, args, nargout,
           "nn_cost_function function providing C++ implementation of nnCostFunction")
{
  octave_value_list result;

  // we ecpect to receive 5 arguments:
  int nargin = args.length();
  CHECK(nargin==5,"nn_cost_function: Invalid number of arguments: " << nargin);

  // Check the argument types:
  CHECK(args(0).is_matrix_type(),"nn_cost_function: nn_params (arg 0) should be a matrix type");
  CHECK(args(1).is_matrix_type(),"nn_cost_function: layer_sizes (arg 1) should be a matrix type");
  CHECK(args(2).is_matrix_type(),"nn_cost_function: X (arg 2) should be a matrix type");
  CHECK(args(3).is_matrix_type(),"nn_cost_function: yy (arg 3) should be a matrix type");
  CHECK(args(4).is_double_type(),"nn_cost_function: lambda (arg 4) should be a double");

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

  // Reshape nn_params back into the parameters Thetas i, the weight matrices for our neural network:
  typedef std::vector<Matrix> MatrixVector;
  MatrixVector Thetas;
  MatrixVector Activation;
  MatrixVector Deltas;
  MatrixVector Inputs;

  const double* ptr = (double*)nn_params.data();
  octave_idx_type pos = 0;

  for(octave_idx_type i=0; i<nt;++i) {
    octave_idx_type n = layer_sizes.elem(i+1);
    octave_idx_type m = layer_sizes.elem(i)+1;
    octave_idx_type count = n*m;

    Thetas.push_back(Matrix(n,m));

    Matrix& mat = Thetas[i];    
    memcpy((void*)mat.data(),ptr,sizeof(double)*n*m);
    ptr += count;
    pos += count;
  }

  // note that the pos variable is now on the next index available:
  CHECK(pos== nn_params.dim1(),"Mismatch in unrolled vector size " << pos <<"!="<< nn_params.dim1());

  // Number of samples:
  octave_idx_type m = X.dim1();

  // First we need to compute h(x) for each example.
  Matrix a = Matrix(m,1,1.0); 
  a = a.append(X); // we can only append columns !
  a = a.transpose(); 

  Activation.push_back(a);

  // add a dummy value in the input vector:
  Inputs.push_back(Matrix());

  for(octave_idx_type i=0;i<nt;++i) {
    // Retrieve the current activation value:
    a = Activation[i];

    // Then we can compute the total input for the next layer:
    CHECK(Thetas[i].dim2()==a.dim1(),"Mismatch on forward propagation on level "<<i<<", "<< Thetas[i].dim2()<<"!="<<a.dim1())

    Matrix z = Thetas[i] * a;

    // We have to store the z input for the backpropagation later:
    Inputs.push_back(z);

    // Compute the output of the next layer:
    a = Matrix(z.dim1(),z.dim2());
    double* src = (double*)z.data();
    double* dest = (double*)a.data();
    octave_idx_type num = z.numel();
    for(octave_idx_type j=0;j<num;++j) {
      *dest++ = 1.0/(1.0 + exp(- *src++));
    }

    Matrix ac = Matrix(m,1,1.0);
    ac = ac.append(a.transpose());
    ac = ac.transpose();

    // Also save the activation value:
    Activation.push_back(ac);
  }


  // Now the final result is stored in a, the activation vector (eg. output) of the last layer.
  // Thus we can rename this output as hx:
  Matrix hx = a;

  // Now we build the cost "matrix":
  // cmat = yy .* log(hx) + (1-yy) .* log(1-hx);

  // Then we perform the summation on all classes and on all examples:
  double J = 0;
  double* hptr = (double*)hx.data();
  double* yptr = (double*)yy.data();
  octave_idx_type num = hx.numel();
  CHECK(num==yy.numel(),"Mismatch in hx and yy elem count: "<<num<<"!="<<yy.numel());
  for(octave_idx_type i=0;i<num;++i) {
    J -= (*yptr) * log(*hptr);
    J -= (1.0 - *yptr++) * log(1.0 - *hptr++);
  }

  J /= m;

  // Now we add the regularization terms:
  double b = lambda/(2*m);

  double reg = 0;
  ptr = nn_params.data();
  for(octave_idx_type i=0;i<nt;++i) {
    // get the dimension of the theta matrix:
    octave_idx_type nrow = layer_sizes.elem(i+1);
    octave_idx_type ncol = layer_sizes.elem(i)+1;
    
    // We do NOT count the first row in that matrix:
    ptr += nrow;
    octave_idx_type count = (ncol-1)*nrow;
    for(octave_idx_type j=0;j<count;++j) {
      // then we count all the other values:
      reg += (*ptr)*(*ptr);
      ptr++;
    }
  }

  J += b * reg;

  result.append(J);

  return result;
}
