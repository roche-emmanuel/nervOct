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


DEFUN_DLD (train_bp, args, nargout,
           "train_bp function using nervMBP module")
{
  int nargin = args.length ();

  octave_value_list result;
  CHECK(nargin>=5,"train_bp: should receive more than " << nargin <<" arguments");

  // octave_stdout << "train_bp has "
  //               << nargin << " input arguments and "
  //               << nargout << " output arguments.\n";

  // 1. First we should retrieve the layer size vector;
  const octave_value& lsize_val = args(0);
  
  // check that this is a matrix type:
  CHECK(lsize_val.is_matrix_type(),"train_bp: lsize (arg 0) should be a matrix type")

  // Prepare the lsize vector:
  std::vector<int> lsizes;
  Array<double> lsizes_mat = lsize_val.matrix_value().as_column();
  octave_idx_type num = lsizes_mat.numel();
  
  // We must have at least 3 layers:
  CHECK(num>=3,"train_bp: Invalid number of layers: "<<num);

  for(octave_idx_type i=0;i<num;++i) {
    // logDEBUG("Adding layer size "<< lsizes_mat(i));
    lsizes.push_back(lsizes_mat(i));
  }

  // 2. Retrieve the input matrix:
  const octave_value& input_val = args(1);

  // check that this is a matrix type:
  CHECK(input_val.is_matrix_type(),"train_bp: inputs (arg 1) should be a matrix type")

  // Prepare the input array:
  Matrix inputs_mat = input_val.matrix_value();
  Matrix inputs_mat_tp = inputs_mat.transpose(); // we need to transpose to get the data row by row. (because matrices are columns major in octave apparently.)
  double* inputs = (double*)inputs_mat_tp.data(); // loosing const qualifier.

  // Now that we have the input matrix we should be able to check its dimensions:
  // The number of rows is the number of samples;
  // The number of columns is the number of features:
  octave_idx_type num_samples = inputs_mat.dim1();
  octave_idx_type num_features = inputs_mat.dim2();

  // The number of features should match the size of the first layer:
  CHECK(lsizes[0]==num_features,"train_bp: Mismatch between num features and layer 0 size: "<<lsizes[0]<<"!="<<num_features);

#if 0
  // Test section to check ordering of data:
  for(octave_idx_type r=0;r<num_samples;++r) {
    for(octave_idx_type c=0; c<num_features;++c) {
      CHECK(inputs[num_features*r+c]==inputs_mat(r,c),"Mismatch in input data at index ("<<r<<", "<<c<<")");
    }
  }
#endif

  // 3. Retrieve the output matrix:
  const octave_value& output_val = args(2);

  // check that this is a matrix type:
  CHECK(output_val.is_matrix_type(),"train_bp: outputs (arg 2) should be a matrix type")

  // Prepare the output array:
  Matrix outputs_mat = output_val.matrix_value();
  Matrix outputs_mat_tp = outputs_mat.transpose();
  double* outputs = (double*)outputs_mat_tp.data(); // loosing const qualifier.

  // Again we can check the dimensions of the output matrix:
  // The number of rows is the number of samples;
  // The number of columns is the number of labels:
  octave_idx_type num_samples2 = outputs_mat.dim1();
  octave_idx_type num_outputs = outputs_mat.dim2();

  // We should have the same number of samples:
  CHECK(num_samples==num_samples2,"train_bp: Mismatch in number of samples: "<<num_samples<<"!="<<num_samples2);

  // We should match the last layer size:
  CHECK(lsizes[lsizes.size()-1]==num_outputs,"train_bp: Mismatch between num outputs and layer "<<(lsizes.size()-1)<<" size: "<<lsizes[lsizes.size()-1]<<"!="<<num_outputs);

  // 4. Retrieve the desired rms_stop:
 const octave_value& rms_val = args(3);

  // check that this is a double type:
  CHECK(rms_val.is_scalar_type(),"train_bp: outputs (arg 3) should be a scalar type")

  double rms_stop = rms_val.scalar_value();

  // 5. Retrieve the desired max iter:
 const octave_value& maxiter_val = args(4);

  // check that this is a double type:
  CHECK(maxiter_val.is_scalar_type(),"train_bp: outputs (arg 4) should be a scalar type")

  int max_iter = (int)maxiter_val.scalar_value();


  // 6. prepare the weight vector:
  // Compute the expected number of weights:
  octave_idx_type expnum = 0;
  int nl = lsizes.size()-1;

  for(int i=0;i<nl;++i) {
    expnum += (lsizes[i]+1)*lsizes[i+1];
  }
  
  // now create the vector that will hold the data:
  Matrix weights_mat = Matrix(expnum,1,0.0);

  // Now we need to perform the actual computation.
  // So we need to load the nervMBP library:


  HMODULE h = LoadLibrary("nervMBP.dll");  //W:\\Cloud\\Projects\\nervtech\\bin\\x86\\
  CHECK(h != NULL,"Cannot load nervMBP.dll module.");

  typedef bool (* IsCudaSupportedFunc)();

  // // We should be able to retrieve the train function:
  IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, "isCudaSupported");

  CHECK(isCudaSupported != NULL,"Cannot find isCudaSupported function");

  // // Check that CUDA is supported:
  CHECK(isCudaSupported(),"CUDA is not supported.");
  typedef void (* ShowInfoFunc)();

#if 0
  // We should be able to retrieve the train function:
  ShowInfoFunc showInfo = (ShowInfoFunc) GetProcAddress(h, "showCudaInfo");
  CHECK(showInfo != NULL,"Cannot find showCudaInfo function");

  showInfo();
#endif

  CHECK(FreeLibrary(h),"Cannot free nervMBP library.");

  // Add the weight matrix to the results:
  result.append(weights_mat);

  return result;
}
