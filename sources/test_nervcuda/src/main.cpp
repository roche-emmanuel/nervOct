// each test module could contain no more then one 'main' file with init function defined
// alternatively you could define init function yourself
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "NervCUDA tests"

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>
#include <cuda_runtime.h>

#include <boost/chrono.hpp>

#define logDEBUG(msg) std::cout << msg << std::endl;

BOOST_AUTO_TEST_SUITE( basic_suite )


BOOST_AUTO_TEST_CASE( test_loading_module )
{
  // For this test we try to load/unload the NervMBP library.
  HMODULE h = LoadLibrary("nervCUDA.dll");
  
  // The pointer should not be null:
  BOOST_CHECK(h != nullptr);

  // Should be able to free the library:
  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_retrieving_mult_mat )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

  // We should be able to retrieve the train function:
  MultMatFunc mult_mat = (MultMatFunc) GetProcAddress(h, "multiplyMatrices");
  BOOST_CHECK(mult_mat != nullptr);

  BOOST_CHECK(FreeLibrary(h));
}

int random_int(int mini, int maxi) {
  return mini + (int)floor(0.5 + (maxi-mini)*(double)rand()/(double)RAND_MAX);
}

double random_double(double mini, double maxi) {
  return mini + (maxi-mini)*(double)rand()/(double)RAND_MAX;
}

BOOST_AUTO_TEST_CASE( test_mult_mat )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

  // We should be able to retrieve the train function:
  MultMatFunc mult_mat = (MultMatFunc) GetProcAddress(h, "multiplyMatrices");
  BOOST_CHECK(mult_mat != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 100; // number of tests to perform.
  for(unsigned int i = 0;i<num;++i) {
    unsigned int nrowA = random_int(10,100);
    unsigned int ncolA = random_int(10,100);
    unsigned int nrowB = ncolA;
    unsigned int ncolB = random_int(10,100);

    // prepare the matrix data:
    unsigned int count = nrowA*ncolA;
    double* ptr;
    double* A = new double[count];
    ptr = A;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = random_double(-10.0,10.0);
    }

    count = nrowB*ncolB;
    double* B = new double[count];
    ptr = B;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = random_double(-10.0,10.0);
    }

    count = nrowA*ncolB;
    double* C = new double[count];
    memset((void*)C,0,sizeof(double)*count);

    double* predC = new double[count];
    for(unsigned int row=0;row<nrowA;++row) {
      for(unsigned int col=0;col<ncolB;++col) {
        // compute the value C(row,col):
        double val = 0.0;
        for(unsigned int n = 0;n<ncolA;++n) {
          // val += A(row,n)*B(n,col);
          val += A[n*nrowA+row]*B[col*nrowB+n];
        }
        predC[nrowA*col+row] = val;
      }
    }

    // Now compute the value using the CUDA kernel:
    // logDEBUG("Testing wih A: "<<nrowA<<" x "<<ncolA<<" and B: "<<nrowB<<" x "<<ncolB);

    mult_mat(nrowA, ncolA, A, nrowB, ncolB, B, C, false, false);

    // finally we need to compare the computed matrices value by value:
    for(unsigned int row=0;row<nrowA;++row) {
      for(unsigned int col=0;col<ncolB;++col) {
        double v1 = C[nrowA*col+row];
        double v2 = predC[nrowA*col+row];
        BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch at element ("<<row<<", "<<col<<"): "<<v1<<"!="<<v2);
      }
    }

  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_mult_mat_tp_a )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

  // We should be able to retrieve the train function:
  MultMatFunc mult_mat = (MultMatFunc) GetProcAddress(h, "multiplyMatrices");
  BOOST_CHECK(mult_mat != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 100; // number of tests to perform.
  for(unsigned int i = 0;i<num;++i) {
    unsigned int nrowA = random_int(10,100);
    unsigned int ncolA = random_int(10,100);
    unsigned int nrowB = nrowA;
    unsigned int ncolB = random_int(10,100);

    // prepare the matrix data:
    unsigned int count = nrowA*ncolA;
    double* ptr;
    double* A = new double[count];
    ptr = A;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = random_double(-10.0,10.0);
    }

    count = nrowB*ncolB;
    double* B = new double[count];
    ptr = B;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = random_double(-10.0,10.0);
    }

    count = ncolA*ncolB;
    double* C = new double[count];
    memset((void*)C,0,sizeof(double)*count);

    double* predC = new double[count];
    for(unsigned int row=0;row<ncolA;++row) {
      for(unsigned int col=0;col<ncolB;++col) {
        // compute the value C(row,col):
        double val = 0.0;
        for(unsigned int n = 0;n<nrowA;++n) {
          // val += A(row,n)*B(n,col);
          val += A[row*nrowA+n]*B[col*nrowB+n];
        }
        predC[ncolA*col+row] = val;
      }
    }

    // Now compute the value using the CUDA kernel:
    // logDEBUG("Testing wih A: "<<nrowA<<" x "<<ncolA<<" and B: "<<nrowB<<" x "<<ncolB);

    mult_mat(nrowA, ncolA, A, nrowB, ncolB, B, C, true, false);

    // finally we need to compare the computed matrices value by value:
    for(unsigned int row=0;row<ncolA;++row) {
      for(unsigned int col=0;col<ncolB;++col) {
        double v1 = C[ncolA*col+row];
        double v2 = predC[ncolA*col+row];
        BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch at element ("<<row<<", "<<col<<"): "<<v1<<"!="<<v2);
      }
    }

  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_mult_mat_tp_b )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

  // We should be able to retrieve the train function:
  MultMatFunc mult_mat = (MultMatFunc) GetProcAddress(h, "multiplyMatrices");
  BOOST_CHECK(mult_mat != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 100; // number of tests to perform.
  for(unsigned int i = 0;i<num;++i) {
    unsigned int nrowA = random_int(10,100);
    unsigned int ncolA = random_int(10,100);
    unsigned int nrowB = random_int(10,100);
    unsigned int ncolB = ncolA;

    // prepare the matrix data:
    unsigned int count = nrowA*ncolA;
    double* ptr;
    double* A = new double[count];
    ptr = A;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = random_double(-10.0,10.0);
    }

    count = nrowB*ncolB;
    double* B = new double[count];
    ptr = B;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = random_double(-10.0,10.0);
    }

    count = nrowA*nrowB;
    double* C = new double[count];
    memset((void*)C,0,sizeof(double)*count);

    double* predC = new double[count];
    for(unsigned int row=0;row<nrowA;++row) {
      for(unsigned int col=0;col<nrowB;++col) {
        // compute the value C(row,col):
        double val = 0.0;
        for(unsigned int n = 0;n<ncolA;++n) {
          // val += A(row,n)*B(n,col);
          val += A[n*nrowA+row]*B[n*nrowB+col];
        }
        predC[nrowA*col+row] = val;
      }
    }

    // Now compute the value using the CUDA kernel:
    // logDEBUG("Testing wih A: "<<nrowA<<" x "<<ncolA<<" and B: "<<nrowB<<" x "<<ncolB);

    mult_mat(nrowA, ncolA, A, nrowB, ncolB, B, C, false, true);

    // finally we need to compare the computed matrices value by value:
    for(unsigned int row=0;row<nrowA;++row) {
      for(unsigned int col=0;col<nrowB;++col) {
        double v1 = C[nrowA*col+row];
        double v2 = predC[nrowA*col+row];
        BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch at element ("<<row<<", "<<col<<"): "<<v1<<"!="<<v2);
      }
    }

  }

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_mult_mat_performances )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (* MultMatFunc)(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C, bool tpA, bool tpB);

  // We should be able to retrieve the train function:
  MultMatFunc mult_mat = (MultMatFunc) GetProcAddress(h, "multiplyMatrices");
  BOOST_CHECK(mult_mat != nullptr);

  unsigned int nrowA = 500;
  unsigned int ncolA = 500;
  unsigned int nrowB = ncolA;
  unsigned int ncolB = 500;

  // prepare the matrix data:
  unsigned int count = nrowA*ncolA;
  double* ptr;
  double* A = new double[count];
  ptr = A;
  for(unsigned int j=0;j<count;++j) {
    // (*ptr++) = random_double(-10.0,10.0);
    (*ptr++) = sin(j)*10.0;
  }

  count = nrowB*ncolB;
  double* B = new double[count];
  ptr = B;
  for(unsigned int j=0;j<count;++j) {
    // (*ptr++) = random_double(-10.0,10.0);
    (*ptr++) = cos(j)*10.0;
  }

  count = nrowA*ncolB;
  double* C = new double[count];
  memset((void*)C,0,sizeof(double)*count);

  double* predC = new double[count];


  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.
  
  // Compute the matrix on the CPU:
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();

  for(unsigned int i = 0;i<num;++i) {
    for(unsigned int row=0;row<nrowA;++row) {
      for(unsigned int col=0;col<ncolB;++col) {
        // compute the value C(row,col):
        double val = 0.0;
        for(unsigned int n = 0;n<ncolA;++n) {
          // val += A(row,n)*B(n,col);
          val += A[n*nrowA+row]*B[col*nrowB+n];
        }
        predC[nrowA*col+row] = val;
      }
    }
  }
  
  boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
  logDEBUG("CPU matrix mult taking " << (sec.count()/num) << " seconds.");

  // Compute the matrix onthe GPU:
  start = boost::chrono::system_clock::now();
  
  for(unsigned int i = 0;i<num;++i) {
    mult_mat(nrowA, ncolA, A, nrowB, ncolB, B, C, false, false);
  }

  sec = boost::chrono::system_clock::now() - start;
  logDEBUG("GPU matrix mult taking " << (sec.count()/num) << " seconds.");

  BOOST_CHECK(FreeLibrary(h));
}

BOOST_AUTO_TEST_CASE( test_cost_function )
{
  HMODULE h = LoadLibrary("nervCUDA.dll");  
  BOOST_CHECK(h != nullptr);

  typedef void (*CostFunc)(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
  double* nn_params, double* X, double* yy, double lambda, double* activation, double* inputs);

  // We should be able to retrieve the train function:
  CostFunc costfunc = (CostFunc) GetProcAddress(h, "costFunc");
  BOOST_CHECK(costfunc != nullptr);
  CostFunc costfunc_cpu = (CostFunc) GetProcAddress(h, "costFuncCPU");
  BOOST_CHECK(costfunc_cpu != nullptr);

  // Now we use the mult mat method to compute a few matrices multiplication:
  unsigned int num = 10; // number of tests to perform.

  for(unsigned int i = 0;i<num;++i) {
    // prepare number of samples:
    unsigned int nsamples = random_int(10,100);

    // Prepare the layer size vector:
    unsigned int nl = random_int(3,5);
    unsigned int nt = nl-1;

    unsigned int* lsizes = new unsigned int[nl];

    for(unsigned int j = 0; j<nl; ++j) {
      lsizes[j] = random_int(3,6);
    }

    // prepare the X matrix data:
    unsigned int count = nsamples*lsizes[0];
    double* ptr;
    double* X = new double[count];
    ptr = X;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = sin(j)*10.0;
      // (*ptr++) = random_double(-10.0,10.0);
    }

    // Prepare the yy matrix:
    count = nsamples*lsizes[nl-1];
    double* yy = new double[count];
    ptr = yy;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = cos(j)*10.0;
      // (*ptr++) = random_double(-10.0,10.0);
    }

    // Prepare the current weights matrices:
    count = 0;
    for(unsigned j=0;j<nt;++j) {
      count += lsizes[j+1]*(lsizes[j]+1);
    }

    // logDEBUG("Allocating "<<count<<" bytes...")
    double* params = new double[count];
    ptr = params;
    for(unsigned int j=0;j<count;++j) {
      (*ptr++) = sin(j+0.5);
    }    

    // prepare the lambda value:
    double lambda = random_double(0.0,1.0);

    // Prepare the activation array:
    unsigned int act_size = 0;
    for(unsigned int j=0;j<nl;++j) {
      act_size += lsizes[j]+1;
    }
    act_size *= nsamples;

    double* activation = new double[act_size];
    memset(activation,0,sizeof(double)*act_size);

    // Prepare the input array:
    unsigned int input_size = 0;
    for(unsigned int j=0;j<nt;++j) {
      input_size += lsizes[j+1];
    }
    input_size *= nsamples;

    double* inputs = new double[input_size];
    memset(inputs,0,sizeof(double)*input_size);


    // Now we call the cost function method:
    costfunc(nl, lsizes, nsamples, params, X, yy, lambda, activation, inputs);

    // Now we should manually compute the activation/input values:
    double* pred_act = new double[act_size];
    double* pred_input = new double[input_size];
    memset(pred_act,0,sizeof(double)*act_size);
    memset(pred_input,0,sizeof(double)*input_size);

    costfunc_cpu(nl, lsizes, nsamples, params, X, yy, lambda, pred_act, pred_input);

#if 0
    // prepare the prediction data:
    // First we need to add the a0 data:
    ptr = pred_act;
    ptr += nsamples;

    // for(unsigned int j=0;j<nsamples;++j) {
    //   *ptr++ = 0.0;
    // }

    // inject the X matrix in a row major version:
    double* xptr = X;
    unsigned int nrows = nsamples;
    unsigned int ncols = lsizes[0];
    memcpy(ptr,xptr,sizeof(double)*nrows*ncols);

    // make the prediction for the other layer:
    unsigned int theta_offset = 0;
    unsigned int act_offset = 0;
    unsigned int next_act_offset = nsamples*(lsizes[0]+1);
    unsigned int input_offset = 0;

    for(unsigned int i=0; i<nt;++i) {

      // compute the matrix z_i = theta_i * a_i^T
      unsigned int nrows = lsizes[i+1];
      unsigned int ncols = nsamples;

      double* z = new double[nsamples*lsizes[i+1]];
      memset(z,0,sizeof(double)*nsamples*lsizes[i+1]);
      
      unsigned int num = lsizes[i]+1;

      for(unsigned int c=0;c<ncols;++c) {
        for(unsigned int r=0;r<nrows;++r) {
          // compute the value of z_i(r,c):
          double val = 0;
          for(unsigned int n=0;n<num;++n) {
            // val += theta_i(r,n)*a_i(c,n); // note that we transpose a_i here.
            val += params[theta_offset+nrows*n+r]*pred_act[act_offset+nsamples*n+c]; 
          }

          // We have compute the total value of the element z_i(r,c), but we still need to take the sigmoid:
          val = 1.0 / (1.0 + exp(-val));

          // Now we store this as the new value computed for the input:
          pred_input[input_offset+nrows*c+r] = val;

          // finally we also set the new activation matrix value
          // the value of z_i(r,c) is stored as a_(i+1)(c,r+1):
          pred_act[next_act_offset + nsamples*(r+1) + c] = val;
        }
      }

      // update the offsets:
      theta_offset += lsizes[i+1]*(lsizes[i]+1);
      act_offset = next_act_offset;
      next_act_offset += nsamples*(lsizes[i+1]+1);
      input_offset += nsamples*lsizes[i+1];
    }
#endif


    // Compare the content of the activation array:
    // for(unsigned int j=0;j<act_size;++j) {
    //   double v1 = activation[j];
    //   double v2 = pred_act[j];
    //   BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch on activation element "<<j<<": "<<v1<<"!="<<v2);
    // }

    // Compare the content of the input array:
    for(unsigned int j=0;j<input_size;++j) {
      double v1 = inputs[j];
      double v2 = pred_input[j];
      BOOST_CHECK_MESSAGE(abs(v1-v2)<1e-10,"Mismatch on inputs element "<<j<<": "<<v1<<"!="<<v2);
    }

    delete [] lsizes;
    delete [] X;
    delete [] yy;
    delete [] params;
    delete [] activation;
    delete [] inputs;
    delete [] pred_act;
    delete [] pred_input;
  }

  BOOST_CHECK(FreeLibrary(h));

  cudaDeviceReset();
}


BOOST_AUTO_TEST_SUITE_END()
