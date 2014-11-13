#ifndef NERV_KERNELS_H_
#define NERV_KERNELS_H_

#define BLOCK_SIZE 32

__global__ void MatMulKernel(unsigned int nrowA, unsigned int ncolA, const double* A,
    unsigned int nrowB, unsigned int ncolB, const double* B, double* C);

#endif
