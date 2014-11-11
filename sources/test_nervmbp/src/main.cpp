// each test module could contain no more then one 'main' file with init function defined
// alternatively you could define init function yourself
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "NervMBP tests"

#include <boost/test/unit_test.hpp>

#include <nervmbp.h>
#include <windows.h>

BOOST_AUTO_TEST_SUITE( basic_suite )

BOOST_AUTO_TEST_CASE( test_sanity )
{
  // Dummy sanity check test: 
  BOOST_CHECK( 1 == 1 );
}

BOOST_AUTO_TEST_CASE( test_loading_module )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervMBP.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_retrieving_cuda_supported_function )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* IsCudaSupportedFunc)();

  // We should be able to retrieve the train function:
  IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, "isCudaSupported");
  BOOST_CHECK(isCudaSupported != nullptr);

  // Check that CUDA is supported:
  // BOOST_CHECK(isCudaSupported() == true);

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_check_cuda_is_supported )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* IsCudaSupportedFunc)();

  // We should be able to retrieve the train function:
  IsCudaSupportedFunc isCudaSupported = (IsCudaSupportedFunc) GetProcAddress(h, "isCudaSupported");
  BOOST_CHECK(isCudaSupported != nullptr);

  // Check that CUDA is supported:
  BOOST_CHECK(isCudaSupported() == true);

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( should_show_cuda_infos )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* ShowInfoFunc)();

  // We should be able to retrieve the train function:
  ShowInfoFunc showInfo = (ShowInfoFunc) GetProcAddress(h, "showCudaInfo");
  BOOST_CHECK(showInfo != nullptr);

  // showInfo();

  BOOST_CHECK(FreeLibrary(h));
}

// Compute the sigmoid of a value:
double sigmoid(double z)
{
  return 1.0/ (1.0 + exp(-z));
}

double predict_xor(double x1, double x2, double* weights)
{
  // Computing the values on the hidden layer:
  double a1 = sigmoid(1*weights[0] + x1*weights[1] + x2*weights[2]);
  double a2 = sigmoid(1*weights[3] + x1*weights[4] + x2*weights[5]);
  double a3 = sigmoid(1*weights[6] + a1*weights[7] + a2*weights[8]);

  return a3;
}

BOOST_AUTO_TEST_CASE( should_be_able_to_call_trainBP )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* TrainFunc)(const std::vector<int>& lsizes, 
    int num_inputs, double* inputs,
    int num_outputs, double* outputs,
    int num_weights, double* weights,
    double& rms_stop, int max_iter, bool use_weights);

  // We should be able to retrieve the train function:
  TrainFunc trainBP = (TrainFunc) GetProcAddress(h, "trainBP");
  BOOST_CHECK(trainBP != nullptr);

  // Here we try to train a simple XOR function:
  std::vector<int> lsizes;
  lsizes.push_back(2); // 2 inputs
  lsizes.push_back(2); // 2 hidden units in one hidden layer.
  lsizes.push_back(1); // 1 output

  double inputs[] = { 
    0, 0,
    1, 0,
    0, 1,
    1, 1
  };

  double outputs[] = { 1, 0, 0, 1};

  double* weights = new double[9];

  double rms = 0.002;
  bool res = trainBP(lsizes,8,inputs,4,outputs,9,weights,rms,10000,false);
  BOOST_CHECK(res==true);

  std::cout<<"The weights are:"<<std::endl;
  for(int i=0;i<9;++i) {
    std::cout<<"Weight "<<i<<": "<<weights[i]<<std::endl;
  }

  // ensure that the actua RMS value is lower that the value requested:
  BOOST_CHECK(rms <= 0.002);

  // Now that we have the weights, we can compute the actual values we can observe:
  double p1 = predict_xor(0, 0, weights);
  std::cout<<"Pred for (0,0) is: "<<p1<<std::endl;
  double p2 = predict_xor(1, 0, weights);
  std::cout<<"Pred for (1,0) is: "<<p2<<std::endl;
  double p3 = predict_xor(0, 1, weights);
  std::cout<<"Pred for (0,1) is: "<<p3<<std::endl;
  double p4 = predict_xor(1, 1, weights);
  std::cout<<"Pred for (1,1) is: "<<p4<<std::endl;

  // Compute the actual RMS we have:
  // No idea why, but the GPUMLib use a division by 2.0 when computing the RMS, so we need to reflect this here:
  double actual_rms = sqrt( ((p1-1.0)*(p1-1.0)+(p2-0.0)*(p2-0.0)+(p3-0.0)*(p3-0.0)+(p4-1.0)*(p4-1.0))/4.0 ) / 2.0;
  std::cout<<"Actual RMS is: "<<actual_rms<<std::endl;

  BOOST_CHECK_CLOSE(rms, actual_rms, 1e-10);

  delete [] weights;

  BOOST_CHECK(FreeLibrary(h));
}

double predict_x1(double x1, double x2, double* weights)
{
  // Computing the values on the hidden layer:
  double a1 = sigmoid(1*weights[0] + x1*weights[1] + x2*weights[2]);
  double a2 = sigmoid(1*weights[3] + x1*weights[4] + x2*weights[5]);
  double a3 = sigmoid(1*weights[6] + a1*weights[7] + a2*weights[8]);

  return a3;
}

BOOST_AUTO_TEST_CASE( should_be_able_to_predict_x1 )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* TrainFunc)(const std::vector<int>& lsizes, 
    int num_inputs, double* inputs,
    int num_outputs, double* outputs,
    int num_weights, double* weights,
    double& rms_stop, int max_iter, bool use_weights);

  // We should be able to retrieve the train function:
  TrainFunc trainBP = (TrainFunc) GetProcAddress(h, "trainBP");
  BOOST_CHECK(trainBP != nullptr);

  // Here we try to train a simple XOR function:
  std::vector<int> lsizes;
  lsizes.push_back(2); // 2 inputs
  lsizes.push_back(2); // 2 hidden units in one hidden layer.
  lsizes.push_back(1); // 1 output

  double inputs[] = { 
    0, 0,
    1, 0,
    0, 1,
    1, 1
  };

  double outputs[] = { 0, 1, 0, 1};

  double* weights = new double[9];

  double rms = 0.002;
  bool res = trainBP(lsizes,8,inputs,4,outputs,9,weights,rms,10000,false);
  BOOST_CHECK(res==true);

  std::cout<<"The weights are:"<<std::endl;
  for(int i=0;i<9;++i) {
    std::cout<<"Weight "<<i<<": "<<weights[i]<<std::endl;
  }

  // ensure that the actua RMS value is lower that the value requested:
  BOOST_CHECK(rms <= 0.002);

  // Now that we have the weights, we can compute the actual values we can observe:
  double p1 = predict_x1(0, 0, weights);
  std::cout<<"Pred for (0,0) is: "<<p1<<std::endl;
  double p2 = predict_x1(1, 0, weights);
  std::cout<<"Pred for (1,0) is: "<<p2<<std::endl;
  double p3 = predict_x1(0, 1, weights);
  std::cout<<"Pred for (0,1) is: "<<p3<<std::endl;
  double p4 = predict_x1(1, 1, weights);
  std::cout<<"Pred for (1,1) is: "<<p4<<std::endl;

  // Compute the actual RMS we have:
  // No idea why, but the GPUMLib use a division by 2.0 when computing the RMS, so we need to reflect this here:
  double actual_rms = sqrt( ((p1-0.0)*(p1-0.0)+(p2-1.0)*(p2-1.0)+(p3-0.0)*(p3-0.0)+(p4-1.0)*(p4-1.0))/4.0 ) / 2.0;
  std::cout<<"Actual RMS for predict X1 is: "<<actual_rms<<std::endl;

  BOOST_CHECK_CLOSE(rms, actual_rms, 1e-10);

  delete [] weights;

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( should_be_able_to_predict_x1_or_x2 )
{
  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* TrainFunc)(const std::vector<int>& lsizes, 
    int num_inputs, double* inputs,
    int num_outputs, double* outputs,
    int num_weights, double* weights,
    double& rms_stop, int max_iter, bool use_weights);

  // We should be able to retrieve the train function:
  TrainFunc trainBP = (TrainFunc) GetProcAddress(h, "trainBP");
  BOOST_CHECK(trainBP != nullptr);

  // Here we try to train a simple XOR function:
  std::vector<int> lsizes;
  lsizes.push_back(2); // 2 inputs
  lsizes.push_back(2); // 2 hidden units in one hidden layer.
  lsizes.push_back(1); // 1 output

  double inputs[] = { 
    0, 0,
    1, 0,
    0, 1,
    1, 1
  };

  double outputs[] = { 0, 1, 1, 1};

  double* weights = new double[9];

  double rms = 0.002;
  bool res = trainBP(lsizes,8,inputs,4,outputs,9,weights,rms,10000,false);
  BOOST_CHECK(res==true);

  std::cout<<"The weights are:"<<std::endl;
  for(int i=0;i<9;++i) {
    std::cout<<"Weight "<<i<<": "<<weights[i]<<std::endl;
  }

  // ensure that the actua RMS value is lower that the value requested:
  BOOST_CHECK(rms <= 0.002);

  // Now that we have the weights, we can compute the actual values we can observe:
  double p1 = predict_x1(0, 0, weights);
  std::cout<<"Pred for (0,0) is: "<<p1<<std::endl;
  double p2 = predict_x1(1, 0, weights);
  std::cout<<"Pred for (1,0) is: "<<p2<<std::endl;
  double p3 = predict_x1(0, 1, weights);
  std::cout<<"Pred for (0,1) is: "<<p3<<std::endl;
  double p4 = predict_x1(1, 1, weights);
  std::cout<<"Pred for (1,1) is: "<<p4<<std::endl;

  // Compute the actual RMS we have:
  // No idea why, but the GPUMLib use a division by 2.0 when computing the RMS, so we need to reflect this here:
  double actual_rms = sqrt( ((p1-0.0)*(p1-0.0)+(p2-1.0)*(p2-1.0)+(p3-1.0)*(p3-1.0)+(p4-1.0)*(p4-1.0))/4.0 ) / 2.0;
  std::cout<<"Actual RMS for predict X1 or X2 is: "<<actual_rms<<std::endl;

  BOOST_CHECK_CLOSE(rms, actual_rms, 1e-10);

  delete [] weights;

  BOOST_CHECK(FreeLibrary(h));
}


double random(double mini=0.0, double maxi=1.0)
{
  return mini + (maxi-mini)*(double)rand()/(double)RAND_MAX;
}

void predict_sincos_simple(double x1, double x2, double* weights, double& r1, double& r2, double& r3)
{
  double* ptr = weights;

  // first we compute the activation values for each neuron in the first hidden layer:
  int nh1 = 30;
  double* a1 = new double[nh1];
  double cur;
  for(int i=0;i<nh1; ++i) {
    cur = (*ptr++);
    cur += x1*(*ptr++);
    cur += x2*(*ptr++);
    a1[i] = sigmoid(cur);
  }

  // finally we compute the value on the output layer:
  int nout = 3;
  double* a2 = new double[nout];

  for(int i=0;i<nout;++i) {
    // init the computation with the bias value:
    cur = 1*(*ptr++);
    for(int j=0; j<nh1; ++j) {
      cur += a1[j]*(*ptr++);
    }

    // then we take the sigmoid:
    a2[i] = sigmoid(cur);
  }

  // retrieve the compute values:
  r1 = a2[0];
  r2 = a2[1];
  r3 = a2[2];

  delete [] a1;
  delete [] a2;
}

BOOST_AUTO_TEST_CASE( should_train_on_sin_cos_functions_simple )
{
  srand ((unsigned int)time(NULL));

  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* TrainFunc)(const std::vector<int>& lsizes, 
    int num_inputs, double* inputs,
    int num_outputs, double* outputs,
    int num_weights, double* weights,
    double& rms_stop, int max_iter, bool use_weights);

  // We should be able to retrieve the train function:
  TrainFunc trainBP = (TrainFunc) GetProcAddress(h, "trainBP");
  BOOST_CHECK(trainBP != nullptr);

  // Here we build a dataset where we try to simulate 3 outputs from 2 entries:
  // we have x1 and x2 and we compute sin(x1*x2), cos(x2), cos(x1)*sin(x1)
  // We will use 2 hidden layers.

  int nf = 2; // number of features:
  int nh1 = 30; // number of neurons in hidden layer 1
  int nout = 3; // number of outputs
  int m = 3000; // number of samples.

  std::vector<int> lsizes;
  lsizes.push_back(nf); // 2 inputs
  lsizes.push_back(nh1); 
  lsizes.push_back(nout); // 3 outputs

  // dataset arrays:
  double *inputs = new double[2*m];
  double *outputs = new double[3*m];

// #define DEBUG_SINCOS_NN

  // populate the input dataset:
  double* ptr = inputs;
  for(int i = 0; i<m; ++i) {
#ifdef DEBUG_SINCOS_NN
    *ptr++ = sin(2*i+1); //(double)(2*i+1)/(double)(2*m); //random(); //*10.0-5.0; //0.5; //random(); // 
    *ptr++ = sin(2*i+2); //(double)(2*i+2)/(double)(2*m); //random(); //*10.0-5.0; //0.5; //random(); //
#else
    *ptr++ = random(); //*10.0-5.0;
    *ptr++ = random(); //*10.0-5.0;
#endif
  }

  // Populate the output matrix:
  ptr = outputs;
  double* iptr = inputs;
  for(int i=0; i<m; ++i) {
    *ptr++ = abs(sin(iptr[0])); //iptr[0]; // //sin(iptr[0]*iptr[1]);
    *ptr++ = abs(cos(iptr[1])); //iptr[1]; //
    *ptr++ = abs(cos(iptr[0])*sin(iptr[1])); //iptr[0]+iptr[1]; //sin(iptr[0]) + cos(iptr[1]); //
    iptr += 2;
  }

  // weight array:
  int nw = nh1*(nf+1)+nout*(nh1+1);
  std::cout << "Number of weights: " << nw << std::endl;

  double* weights = new double[nw];
  // menset((void*)weights,0,sizeof(double)*nw);

#ifdef DEBUG_SINCOS_NN
  double rms = 0.004;
#else
  double rms = 0.002;
#endif

  bool res = trainBP(lsizes,2*m,inputs,3*m,outputs,nw,weights,rms,1000,false);
  BOOST_CHECK(res==true);

  // std::cout<<"The weights are:"<<std::endl;
  // for(int i=0;i<9;++i) {
  //   std::cout<<"Weight "<<i<<": "<<weights[i]<<std::endl;
  // }

  // ensure that the actua RMS value is lower that the value requested:
// #ifndef DEBUG_SINCOS_NN
//   BOOST_CHECK(rms <= 0.002);
// #endif

  double r1,r2,r3,x1,x2;
  double actual_rms = 0;
  iptr = inputs;
  for(int i=0;i<m;++i) {
    x1 = iptr[0];
    x2 = iptr[1];

    predict_sincos_simple(x1,x2,weights,r1,r2,r3);
    // std::cout<<"Predicted values are: "<<r1<<", "<<r2<<", "<<r3<<std::endl;

    iptr += 2;
    r1 = abs(sin(x1))-r1;
    r2 = abs(cos(x2))-r2;
    r3 = abs(cos(x1)*sin(x2))-r3;

    actual_rms += r1*r1 + r2*r2 + r3*r3;
  }

  // once we have summed all the actual_rms from the samples, we take the square root:
  // also we have to divide that by 2.0 as previsouly mentioned:
  actual_rms = sqrt(actual_rms/(3.0*m))/2.0;
  std::cout<<"Actual RMS is: "<<actual_rms<<std::endl;

  BOOST_CHECK_CLOSE(rms, actual_rms, 1e-10);

  delete [] weights;

  BOOST_CHECK(FreeLibrary(h));
}

void predict_sincos(double x1, double x2, double* weights, double& r1, double& r2, double& r3)
{
  double* ptr = weights;

  // first we compute the activation values for each neuron in the first hidden layer:
  int nh1 = 30;
  double* a1 = new double[nh1];
  double cur;

  for(int i=0;i<nh1; ++i) {
    cur = (*ptr++);
    cur += x1*(*ptr++);
    cur += x2*(*ptr++);
    a1[i] = sigmoid(cur);    
  }

  // Now we compute the activation from the second hidden layer:
  int nh2 = 10;
  double* a2 = new double[nh2];

  for(int i=0;i<nh2;++i) {
    // init the computation with the bias value:
    cur = 1*(*ptr++);
    for(int j=0; j<nh1; ++j) {
      cur += a1[j]*(*ptr++);
    }

    // then we take the sigmoid:
    a2[i] = sigmoid(cur);
  }

  // finally we compute the value on the output layer:
  int nout = 3;
  double* a3 = new double[nout];

  for(int i=0;i<nout;++i) {
    // init the computation with the bias value:
    cur = 1*(*ptr++);
    for(int j=0; j<nh2; ++j) {
      cur += a2[j]*(*ptr++);
    }

    // then we take the sigmoid:
    a3[i] = sigmoid(cur);
  }

  // retrieve the compute values:
  r1 = a3[0];
  r2 = a3[1];
  r3 = a3[2];

  delete [] a1;
  delete [] a2;
  delete [] a3;
}

BOOST_AUTO_TEST_CASE( should_train_on_sin_cos_functions )
{
  srand ((unsigned int)time(NULL));

  HMODULE h = LoadLibrary("nervMBP.dll");  
  BOOST_CHECK(h != nullptr);

  typedef bool (* TrainFunc)(const std::vector<int>& lsizes, 
    int num_inputs, double* inputs,
    int num_outputs, double* outputs,
    int num_weights, double* weights,
    double& rms_stop, int max_iter, bool use_weights);

  // We should be able to retrieve the train function:
  TrainFunc trainBP = (TrainFunc) GetProcAddress(h, "trainBP");
  BOOST_CHECK(trainBP != nullptr);

  // Here we build a dataset where we try to simulate 3 outputs from 2 entries:
  // we have x1 and x2 and we compute sin(x1*x2), cos(x2), cos(x1)*sin(x1)
  // We will use 2 hidden layers.

  int nf = 2; // number of features:
  int nh1 = 30; // number of neurons in hidden layer 1
  int nh2 = 10; // number of neurons in hidden layer 2
  int nout = 3; // number of outputs
  int m = 3000; // number of samples.

  std::vector<int> lsizes;
  lsizes.push_back(nf); // 2 inputs
  lsizes.push_back(nh1); 
  lsizes.push_back(nh2);
  lsizes.push_back(nout); // 3 outputs

  // dataset arrays:
  double *inputs = new double[2*m];
  double *outputs = new double[3*m];

// #define DEBUG_SINCOS_NN

  // populate the input dataset:
  double* ptr = inputs;
  for(int i = 0; i<m; ++i) {
#ifdef DEBUG_SINCOS_NN
    *ptr++ = sin(2*i+1); //(double)(2*i+1)/(double)(2*m); //random(); //*10.0-5.0; //0.5; //random(); // 
    *ptr++ = sin(2*i+2); //(double)(2*i+2)/(double)(2*m); //random(); //*10.0-5.0; //0.5; //random(); //
#else
    *ptr++ = random(); //*10.0-5.0;
    *ptr++ = random(); //*10.0-5.0;
#endif
  }

  // Populate the output matrix:
  ptr = outputs;
  double* iptr = inputs;
  for(int i=0; i<m; ++i) {
    *ptr++ = abs(sin(iptr[0])); //iptr[0]; // //sin(iptr[0]*iptr[1]);
    *ptr++ = abs(cos(iptr[1])); //iptr[1]; //
    *ptr++ = abs(cos(iptr[0])*sin(iptr[1])); //iptr[0]+iptr[1]; //sin(iptr[0]) + cos(iptr[1]); //
    iptr += 2;
  }

  // weight array:
  int nw = nh1*(nf+1)+nh2*(nh1+1)+nout*(nh2+1);
  std::cout << "Number of weights: " << nw << std::endl;

  double* weights = new double[nw];
  // menset((void*)weights,0,sizeof(double)*nw);

#ifdef DEBUG_SINCOS_NN
  double rms = 0.004;
#else
  double rms = 0.002;
#endif

  bool res = trainBP(lsizes,2*m,inputs,3*m,outputs,nw,weights,rms,1000,false);
  BOOST_CHECK(res==true);

  // std::cout<<"The weights are:"<<std::endl;
  // for(int i=0;i<9;++i) {
  //   std::cout<<"Weight "<<i<<": "<<weights[i]<<std::endl;
  // }

  // ensure that the actua RMS value is lower that the value requested:
// #ifndef DEBUG_SINCOS_NN
//   BOOST_CHECK(rms <= 0.002);
// #endif

  double r1,r2,r3,x1,x2;
  double actual_rms = 0;
  iptr = inputs;
  for(int i=0;i<m;++i) {
    x1 = iptr[0];
    x2 = iptr[1];

    predict_sincos(x1,x2,weights,r1,r2,r3);
    // std::cout<<"Predicted values are: "<<r1<<", "<<r2<<", "<<r3<<std::endl;

    iptr += 2;
    r1 = abs(sin(x1))-r1;
    r2 = abs(cos(x2))-r2;
    r3 = abs(cos(x1)*sin(x2))-r3;

    actual_rms += r1*r1 + r2*r2 + r3*r3;
  }

  // once we have summed all the actual_rms from the samples, we take the square root:
  // also we have to divide that by 2.0 as previsouly mentioned:
  actual_rms = sqrt(actual_rms/(3.0*m))/2.0;
  std::cout<<"Actual RMS is: "<<actual_rms<<std::endl;

  BOOST_CHECK_CLOSE(rms, actual_rms, 1e-10);

  delete [] weights;

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_SUITE_END()
