// each test module could contain no more then one 'main' file with init function defined
// alternatively you could define init function yourself
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "NervCUDA tests"

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <nervcuda.h>
#include <windows.h>

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


BOOST_AUTO_TEST_SUITE_END()
