#include <nervCUDA.h>

#include <iostream>

#define logDEBUG(msg) std::cout << msg << std::endl;

extern "C" {

void costFuncCPU(unsigned int nl, unsigned int* lsizes, unsigned int nsamples, 
	double* params, double* X, double* yy, double lambda,
	double* activation, unsigned int ninputs, double* inputs, double& J, double* gradients, double* deltas)
{
  // prepare the prediction data:
  // First we need to add the a0 data:
  double* ptr = activation;
  for(unsigned int j=0;j<nsamples;++j) {
  	(*ptr++) = 1.0;
  }

  // ptr += nsamples;

  unsigned int nt = nl-1;

  unsigned int np = 0;
  for(unsigned int i=0;i<nt;++i) {
    np += lsizes[i+1]*(lsizes[i]+1);
  }

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

  if(ninputs!=input_size) {
    logDEBUG("ERROR: mismatch in input array size: "<<ninputs<<"!="<<input_size);
    return; // stop processing.
  }

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
  int input_offset = 0;

  for(unsigned int i=0; i<nt;++i) {

    // compute the matrix z_i = theta_i * a_i^T
    unsigned int nrows = lsizes[i+1];
    unsigned int ncols = nsamples;

    // double* z = new double[nsamples*lsizes[i+1]];
    // memset(z,0,sizeof(double)*nsamples*lsizes[i+1]);
    
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

  // Compute the value of J on the cpu:
  J = 0.0;
  input_offset -= nsamples*lsizes[nt];

  double* hx = inputs+input_offset;
  // for(unsigned int j=0;j<nt-1;++j) {
  //   hx += nsamples*lsizes[j+1];
  // }

  unsigned int count = nsamples*lsizes[nt];
  
  for(unsigned int j=0;j<count;++j) {
    J -= yy[j] * log(hx[j]) + (1.0 - yy[j]) * log(1.0 - hx[j]);
  }

  J /= (double)nsamples;

  // Add the regularisation:
  ptr = params;
  double Jreg = 0.0;
  for(unsigned int j=0;j<nt;++j) {
    ptr += lsizes[j+1];
    count = lsizes[j+1]*(lsizes[j]);
    for(unsigned int k=0;k<count;++k) {
      double val = (*ptr++);
      Jreg += val*val;
    }
  }

  J += Jreg*lambda/(2.0*nsamples);

  // we will now compute the delta vectors:
  // Offset to use when reading the delta matrix in the current iteration
  // except when next_delta_offset is 0, in that case we read the hx and yy matrices.
  unsigned int delta_offset = 0;

  // Offset to use when writing the delta matrix in the current iteration
  unsigned int next_delta_offset = 0;

  // remove the last theta matrix size from the theta offset so that we can use
  // that offset to retrieve the proper theta matrix:
  theta_offset -= lsizes[nt]*(lsizes[nt-1]+1);

  // initially the input_offset is pointing on the hx matrix which is z(nt-1) with our convention (eg. z(0) is not in the array.)
  // But the first one we will need is actually the one before that: z(nt-2)
  // So we need to update the offset, and remove the size of the matrix z(nt-2) ! (pointer is at the beginning of z(nt-1))
  // Note that this is now done inside the loop.
  // input_offset -= lsizes[nt-1]*nsamples;

  // Prepare the offset for the gradient array:
  // keep in mind we start with the latest theta matrix:
  unsigned int grad_offset = np - lsizes[nt]*(lsizes[nt-1]+1);

  ptr = deltas;

  for(unsigned int i=nt;i>0;--i) {
    unsigned int nrows = lsizes[i];
    unsigned int ncols = nsamples;
    unsigned int niter = lsizes[i+1];
    unsigned int count = nrows*ncols;

    if(i==nt) {
      // We just write the difference of hx and yy in the deltas array:
      for(unsigned int j=0;j<count;++j) {
        (*ptr++) = hx[j] - yy[j];
      }
    }
    else {
      for(unsigned int c=0;c<ncols;++c) {
        for(unsigned int r=0;r<nrows;++r) {
          // we want to compute the value delta(r,c);
          double val = 0.0;
          for(unsigned int n=0;n<niter;++n) {
            // val += theta_T(r+1,n)*delta_prev(n,c);
            // val += theta(n,r+1)*delta_prev(n,c);
            val += params[theta_offset+niter*(r+1)+n]*deltas[delta_offset+niter*c+n];
          }

          // Then we multiply by the sigmoid gradient at z(r,c):
          double sig = inputs[input_offset + nrows*c + r];
          // deltas[next_delta_offset + nrows*c + r] = next_delta_offset + nrows*c + r;
          deltas[next_delta_offset + nrows*c + r] = val * sig *(1.0-sig);
        }
      }

      // once the computation is done for that layer we move to the previous layer:
      theta_offset -= lsizes[i]*(lsizes[i-1]+1);
    }

    delta_offset = next_delta_offset;
    next_delta_offset += count;

    // At this point we have the previous theta matrix (eg. theta(i-1) pointed by theta_offset. (both when i=nt and i<nt).
    // and thats the matrix we need to compute the gradient values.
    // the gradient mat has the same size as the current theta matrix.
    // similarly, the input_offset is pointing on z(i-2) which is the one we need to perform the computation too.
    // and delta_offset points to the delta matrix we just wrote (eg. delta(i)).
    nrows = lsizes[i];
    ncols = lsizes[i-1]+1;
    niter = nsamples;
    count = nrows*ncols;

    input_offset -= lsizes[i-1]*nsamples; // we remove the size of the next delta matrix to be computed. which is also the size of the next z matrix we will use.
    // logDEBUG("CPU: Gradient at i="<<i<<" of size "<< nrows <<" x " << ncols<<", offset="<<grad_offset<<", input_offset="<<input_offset);

    // Compute the gradient:
    for(unsigned int c=0;c<ncols;++c) {
      for(unsigned int r=0;r<nrows;++r) {
        // we want to compute the value of the gradient matrix mat(r,c)
        // with mat_i = delta_i * act_i-1.
        double val = 0.0;
        for(unsigned int n=0;n<nsamples;++n) {
          // val += delta(r,n)*act(n,c);
          // if c==0 then act[i-1](n,c)==1 otherwise act[i-1](n,c)=z[i-2]_T(n,c-1)=z[i-2](c-1,n)
          // val += deltas[delta_offset + nrows*n + r]; //*(c==0 ? 1.0 : inputs[input_offset + (ncols-1)*n + c-1 ]);
          if(i==1) {
            // Here we have to use the X matrix instead of the z_T.
            // we still want to write the value act(n,c)=x(n,c-1) if c>0
            val += deltas[delta_offset + nrows*n + r]*(c==0 ? 1.0 : X[niter*(c-1) + n]);
          }
          else {
            if(c==0) {
              val += deltas[delta_offset + nrows*n + r];
            }
            else {
              int index = input_offset + (ncols-1)*n + c-1;
              if(index>=(int)ninputs || index < 0) {
                logDEBUG("ERROR: out of range input access: "<<index<<">="<<ninputs);
                // return;
              }

              val += deltas[delta_offset + nrows*n + r]*(c==0 ? 1.0 : inputs[index]);            
            }
          }
          // val += 1.0; //(c==0 ? 1.0 : inputs[input_offset + (ncols-1)*n + c-1 ]);
        }

        // Here we also need to add the regularization from the theta matrix:
        double reg = (c==0 ? 0.0 : params[theta_offset + nrows*c+r]);
        val += lambda*reg;

        gradients[grad_offset + nrows*c + r] = val/niter; //grad_offset + nrows*c + r; //val/niter;
      }
    }

    // update the gradient offset by removing the size of the next gradient matrix to be computed:
    // except for the last iteration where the value is not available:
    if(i>1) {
      grad_offset -= lsizes[i-1]*(lsizes[i-2]+1);
    }
  }

}

}
