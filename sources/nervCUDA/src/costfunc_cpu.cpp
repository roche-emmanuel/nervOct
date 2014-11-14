#include <nervCUDA.h>

#include <iostream>

#define logDEBUG(msg) std::cout << msg << std::endl;

extern "C" {

void costFuncCPU(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* params, double* X, double* yy, double lambda,
	double* activation, double* inputs)
{
  // prepare the prediction data:
  // First we need to add the a0 data:
  double* ptr = activation;
  for(unsigned int j=0;j<nsamples;++j) {
  	(*ptr++) = 1.0;
  }

  // ptr += nsamples;

  unsigned int nt = nl-1;

  unsigned int act_size = 0;
  for(unsigned int j=0;j<nl;++j) {
    act_size += lsizes[j]+1;
  }
  act_size *= nsamples;

  unsigned int input_size = 0;
  for(unsigned int j=0;j<nt;++j) {
    input_size += lsizes[j+1];
  }
  input_size *= nsamples;

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
          val += params[theta_offset+nrows*n+r]*activation[act_offset+nsamples*n+c]; 
        }

        if(val>1e10) {
        	logDEBUG("Value is very big: val="<<val<<" is that OK ?");
        }
        
        // We have compute the total value of the element z_i(r,c), but we still need to take the sigmoid:
        val = 1.0 / (1.0 + exp(-val));

        // Now we store this as the new value computed for the input:
        inputs[input_offset+nrows*c+r] = val;

        // finally we also set the new activation matrix value
        // the value of z_i(r,c) is stored as a_(i+1)(c,r+1):
        activation[next_act_offset + nsamples*(r+1) + c] = val;
      }
    }

    // update the offsets:
    theta_offset += lsizes[i+1]*(lsizes[i]+1);
    act_offset = next_act_offset;
	  
	  // set the activation values for the first row:
	  ptr = activation+act_offset ;
	  for(unsigned int j=0;j<nsamples;++j) {
	  	(*ptr++) = 1.0;
	  }

    next_act_offset += nsamples*(lsizes[i+1]+1);
    input_offset += nsamples*lsizes[i+1];
  }
}

}
