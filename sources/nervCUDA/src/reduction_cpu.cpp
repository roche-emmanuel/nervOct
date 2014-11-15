#include <nervCUDA.h>

#include <iostream>

#define logDEBUG(msg) std::cout << msg << std::endl;

extern "C" {

void reductionCPU(double* inputs, unsigned int n, double& output)
{
  // Compute the sum of all elements in the input vector:
  double res = 0;
  for(unsigned int i=0;i<n;++i) {
    res += inputs[i];
  }
  output = res;
}

}
