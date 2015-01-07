#include <nervCUDA.h>
#include <nerv_kernels.h>

template<typename T, unsigned int blockSize>
void gd_errfunc_device(BPDeviceTraits<T> &d_traits)
{
  unsigned int nl = d_traits.nl;
  unsigned int np = d_traits.np();
  unsigned int nt = nl - 1; // number of matrices evolved.
  unsigned int nsamples = d_traits.nsamples;
  unsigned int *lsizes = d_traits.lsizes;
  cudaStream_t stream = d_traits.stream;

  // we must call nn_activation_device before constructing
  // any BPComputeTraits since this method is responsible
  // for assigning the proper value to the wX buffer that
  // will be accessed by the ComputeTraits.
  unsigned int input_offset = nn_activation_device(d_traits);

  BPComputeTraits<T> traits;
  traits = d_traits;
  traits.input_offset = input_offset;

  T *d_hx = d_traits.inputs + traits.input_offset;

  // Here we should compute the rho vector if it is needed.
  // eg. if we are computing a sparse autoencoder layer.
  if (d_traits.spae_beta > 0)
  {
    // the a2 activation values should be the first matrix reported in the inputs.
    // So we should use an input offset of 0.
    // The dimensions of the matrix should be: lsizes[1]*nsamples
    // eg. each column in the matrix is the activation vector for a given sample.
    // So we need to sum on each row by multiplying by a vector of ones with nsamples elements.
    // The result rho should be divided by nsamples and will have lsizes[1] elements.

    T alpha = (T)1.0 / (T)nsamples;
    // logDEBUG("Computing rho vector with alpha=" << alpha);
    mat_vec_mult_device(d_traits.handle, CUBLAS_OP_N, lsizes[1], nsamples, d_traits.inputs, d_traits.spae_ones, d_traits.spae_rho, alpha);
  }

  // Here we can compute the cost now:
  // but only if requested:
  if (d_traits.compute_cost)
  {
    // The hx matrix is mapped to the last z matrix. eg at i=nt-1
    // So its dimensions are lsizes[nt-1+1] * nsamples = lsizes[nl-1] * nsamples
    // same dimensions for the yy matrix, and we want to perform reduction other those 2 matrices
    T J = 0.0;
    unsigned int count = nsamples * lsizes[nt];
    reduce_cost_device(d_hx, d_traits.yy, count, J, stream, d_traits.cost_mode);

    J /= (T)nsamples;

    if (d_traits.lambda > 0)
    {
      T Jreg = 0.0;
      reduce_cost_reg_device(d_traits.params, d_traits.regw, np, Jreg, stream);

      J += (T)((Jreg * d_traits.lambda) / (2.0 * nsamples));
    }

    if (d_traits.spae_beta > 0)
    {
      // If we are computing a sparse auto encoder we should add here the Kullback-Leibler (KL)
      // divergence component to the cost:
      spae_kl_divergence_device(d_traits.spae_kl, d_traits.spae_rho, d_traits.spae_sparsity, lsizes[1], stream);

      // Now reduce the KL vector to its sum:
      T Jsp = 0.0;
      reduce_sum_device(d_traits.spae_kl, lsizes[1], Jsp, stream);
      J += d_traits.spae_beta * Jsp;
      // logDEBUG("Computed sparsity penalty: Jsp=" << d_traits.spae_beta * Jsp);
    }

    d_traits.cost = J;
  }

  if (!d_traits.compute_grads)
  {
    // we don't need to compute the gradients.
    return;
  }

  // Prepare the computation of the delta matrices in reverse order:

  // remove the last theta matrix size from the theta offset so that we can use
  // that offset to retrieve the proper theta matrix:
  // theta_offset -= lsizes[nt]*(lsizes[nt-1]+1);
  traits.theta_offset = np - lsizes[nt] * (lsizes[nt - 1] + 1);

  // initially the input_offset is pointing on the hx matrix which is z(nt-1) with our convention (eg. z(0) is not in the array.)
  // But the first one we will need is actually the one before that: z(nt-2)
  // So we need to update the offset, and remove the size of the matrix z(nt-2) ! (pointer is at the beginning of z(nt-1))
  // Note: This is now done inside the loop:
  // input_offset -= lsizes[nt-1]*nsamples;

  // Prepare the offset for the gradient array:
  // keep in mind we start with the latest theta matrix:
  traits.grad_offset = np - lsizes[nt] * (lsizes[nt - 1] + 1);

  for (unsigned int i = nt; i > 0; --i)
  {
    traits.nrows = lsizes[i];
    traits.ncols = nsamples;
    traits.niter = lsizes[i + 1];
    unsigned int count = traits.nrows * traits.ncols;

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((blockSize + traits.ncols - 1) / blockSize, (blockSize + traits.nrows - 1) / blockSize);

    if (i == nt)
    {
      if (d_traits.cost_mode == COST_RMS)
      {
        // We want to take the sigmoid derivative into account for the last delta too:
        // Note that this implementation should NOT be used when we use a cross entropy cost function
        // or a softmax final layer (which is a generalization of cross entropy)
        // So for now we actually only use this implementation when building a sparse auto encoder layer.
        InitLastDeltaDeriv <<< dimGrid, dimBlock, 0, stream>>>(traits);
      }
      else
      {
        // we should just copy the difference of hx and yy into the z matrix.
        InitLastDelta <<< dimGrid, dimBlock, 0, stream>>>(traits);
      }
    }
    else
    {
      if (d_traits.spae_beta > 0 && i == (nt - 1))
      {
        // We are about to compute the hidden layer deltas
        // So first we need to prepare the sparse vector:
        spae_sparse_delta_device(d_traits.spae_delta, d_traits.spae_rho, d_traits.spae_beta, d_traits.spae_sparsity, lsizes[1], stream);
        ComputeDelta<T, true> <<< dimGrid, dimBlock, 0, stream>>>(traits);
      }
      else
      {
        // We compute the delta from the previous delta:
        // We start this computation for delta(nt-1). this matrix is build from theta(nt-1) and delta(nt).
        // also in the process we use the input matrix z(nt-2)
        ComputeDelta <<< dimGrid, dimBlock, 0, stream>>>(traits);
      }

      // once the computation is done for that layer we move to the previous layer:
      traits.theta_offset -= lsizes[i] * (lsizes[i - 1] + 1);
    }

    traits.delta_offset = traits.next_delta_offset;
    traits.next_delta_offset += count;

    // At this point we have the previous theta matrix (eg. theta(i-1) pointed by theta_offset. (both when i=nt and i<nt).
    // and thats the matrix we need to compute the gradient values.
    // the gradient mat has the same size as the current theta matrix.
    // similarly, the input_offset is pointing on z(i-2) which is the one we need to perform the computation too.
    // and delta_offset points to the delta matrix we just wrote (eg. delta(i)).
    traits.nrows = lsizes[i];
    traits.ncols = lsizes[i - 1] + 1;
    traits.niter = nsamples;
    count = traits.nrows * traits.ncols;

    // Also setup the wbias offset to ensure that we start with the correct offset:
    // basically, we have nsamples bias values for each layer, so we just need to multiply
    // nsamples by the desired number of layers offset.
    // When computing the gradients at index i (1<=i<=nt) we need to use the bias from layer (i-1),
    // thus the offset is:
    traits.wbias_offset = (i - 1) * nsamples;

    // Compute the gradient:
    dimBlock = dim3(blockSize, blockSize);
    dimGrid = dim3((blockSize + traits.ncols - 1) / blockSize, (blockSize + traits.nrows - 1) / blockSize);

    traits.input_offset -= lsizes[i - 1] * nsamples; // we remove the size of the next delta matrix to be computed. which is also the size of the next z matrix we will use.
    // logDEBUG("GPU: Gradient at i="<<i<<" of size "<< nrows <<" x " << ncols<<", offset="<<grad_offset<<", input_offset="<<input_offset<<", nsamples="<<nsamples);

    // We just need to check the wbias buffer to decide if we are using dropout or not.
    if (traits.wbias)
    {
      ComputeGradient<T, true> <<< dimGrid, dimBlock, 0, stream>>>(traits);
    }
    else
    {
      ComputeGradient <<< dimGrid, dimBlock, 0, stream>>>(traits);
    }

    // update the gradient offset by removing the size of the next gradient matrix to be computed:
    // except for the last iteration where the value is not available:
    if (i > 1)
    {
      traits.grad_offset -= lsizes[i - 1] * (lsizes[i - 2] + 1);
    }
  }
}

template <typename T>
void _gd_errfunc(BPTraits<T> &traits)
{
  // BPDeviceTraits<T> d_traits(traits);
  BPDeviceTraits<T> d_traits(false);
  d_traits = traits;

  // Compute the total number of delta coefficients:
  unsigned int nd = traits.nd();
  unsigned int np = traits.np();

  // Call the actual method to perform the computations:
  gd_errfunc_device<T>(d_traits);

  if (traits.compute_cost)
  {
    traits.cost = d_traits.cost;
  }

  // Here we should also read back the gradient values:
  if (traits.compute_grads)
  {
    copyFromDevice(traits.grads, d_traits.grads, np);
  }

  // Read inputs from device memory
  if (traits.inputs)
  {
    copyFromDevice(traits.inputs, d_traits.inputs, nd);
  }

  if (traits.deltas)
  {
    copyFromDevice(traits.deltas, d_traits.deltas, nd); // only retrieve the deltas if requested.
  }
}

template <typename T>
void _gd_errfunc_cpu(BPTraits<T> &traits)
// unsigned int nl, unsigned int* lsizes, unsigned int nsamples,
// double* params, double* X, double* yy, double lambda,
// double* activation, unsigned int ninputs, double* inputs, double& J, double* gradients, double* deltas)
{
  unsigned int nl = traits.nl;
  unsigned int *lsizes = traits.lsizes;
  unsigned int nsamples = traits.nsamples_train;
  unsigned int nt = nl - 1;

  bool owned_inputs = false;
  bool owned_deltas = false;
  bool owned_grads = false;

  if (!traits.inputs)
  {
    owned_inputs = true;
    traits.inputs = new T[traits.nd()];
  }

  if (!traits.deltas)
  {
    owned_deltas = true;
    traits.deltas = new T[traits.nd()];
  }

  if (!traits.grads)
  {
    owned_grads = true;
    traits.grads = new T[traits.np()];
  }

  T *X = traits.X;
  T *yy = traits.yy;
  T *inputs = traits.inputs;
  T *deltas = traits.deltas;
  T *params = traits.params;
  T *gradients = traits.grads;
  T *rho = nullptr;
  T *ptr;

  // Here we need to select the proper cost mode:
  unsigned int cmode = traits.cost_mode;
  if (cmode == COST_DEFAULT)
  {
    if (traits.use_softmax)
    {
      // logDEBUG("Updating cost mode to sotfmax cost");
      cmode = COST_SOFTMAX;
    }
    else if (traits.spae_beta > 0.0)
    {
      // logDEBUG("Updating cost mode to SPAE cost");
      cmode = COST_RMS;
    }
    else
    {
      // logDEBUG("Updating cost mode to cross entropy cost");
      cmode = COST_CROSS_ENTROPY;
    }
  }

  // First step is to compute the predictions, inside the input array.
  nn_predict_cpu(traits);

  unsigned int nh = lsizes[1];
  T sp = traits.spae_sparsity;

  // if spae_beta is positive then we need to prepare the rho vector:
  if (traits.spae_beta > 0.0)
  {
    rho = new T[nh];

    // Now use the activation values to fill the rho vector:
    T val;
    for (unsigned int r = 0; r < nh; ++r)
    {
      val = 0.0;
      for (unsigned int c = 0; c < nsamples; ++c)
      {
        val += inputs[nh * c + r];
      }

      rho[r] = val / (T)nsamples;
    }
  }

  // Compute the value of J on the cpu:

  // Place the input offset at the proper location:
  unsigned int input_offset = 0;
  for (unsigned int i = 1; i < nt; ++i)
  {
    // Add the size of the layer i multiplied by the number of samples:
    input_offset += lsizes[i] * nsamples;
  }

  T *hx = inputs + input_offset;

  if (traits.compute_cost)
  {
    T J = 0.0;

    unsigned int count = nsamples * lsizes[nt];
    if (cmode == COST_SOFTMAX)
    {
      for (unsigned int j = 0; j < count; ++j)
      {
        J -= yy[j] * log(hx[j]);
      }
    }
    else if (cmode == COST_CROSS_ENTROPY)
    {
      for (unsigned int j = 0; j < count; ++j)
      {
        J -= yy[j] * log(hx[j]) + (1.0 - yy[j]) * log(1.0 - hx[j]);
      }
    }
    else // in RMS mode
    {
      T dif;
      for (unsigned int j = 0; j < count; ++j)
      {
        dif = hx[j] - yy[j];
        J += 0.5 * dif * dif;
      }
    }

    J /= (double)nsamples;

    // Add the regularisation:
    ptr = params;

    double Jreg = 0.0;
    for (unsigned int j = 0; j < nt; ++j)
    {
      ptr += lsizes[j + 1];
      count = lsizes[j + 1] * (lsizes[j]);
      for (unsigned int k = 0; k < count; ++k)
      {
        double val = (*ptr++);
        Jreg += val * val;
      }
    }

    J += Jreg * traits.lambda / (2.0 * nsamples);

    // Also check here if we need to apply the spare penalty:
    if (traits.spae_beta > 0.0)
    {
      // sum over the KL divergence values:
      T Jsp = 0.0;
      T r;
      for (unsigned int j = 0; j < nh; ++j)
      {
        r = rho[j];
        Jsp += sp * log(sp / r) + (1.0 - sp) * log((1.0 - sp) / (1.0 - r));
      }

      J += traits.spae_beta * Jsp;
    }

    traits.cost = J;
  }

  if (!traits.compute_grads)
  {
    // we don't need to compute the gradients.
    // TODO: should release the previously allocated arrays here.
    return;
  }

  // we will now compute the delta vectors:
  // Offset to use when reading the delta matrix in the current iteration
  // except when next_delta_offset is 0, in that case we read the hx and yy matrices.
  unsigned int delta_offset = 0;

  // Offset to use when writing the delta matrix in the current iteration
  unsigned int next_delta_offset = 0;

  // remove the last theta matrix size from the theta offset so that we can use
  // that offset to retrieve the proper theta matrix:
  unsigned int theta_offset = traits.np() - lsizes[nt] * (lsizes[nt - 1] + 1);

  // initially the input_offset is pointing on the hx matrix which is z(nt-1) with our convention (eg. z(0) is not in the array.)
  // But the first one we will need is actually the one before that: z(nt-2)
  // So we need to update the offset, and remove the size of the matrix z(nt-2) ! (pointer is at the beginning of z(nt-1))
  // Note that this is now done inside the loop.
  // input_offset -= lsizes[nt-1]*nsamples;

  // Prepare the offset for the gradient array:
  // keep in mind we start with the latest theta matrix:
  unsigned int grad_offset = traits.np() - lsizes[nt] * (lsizes[nt - 1] + 1);

  ptr = traits.deltas;

  T rval;

  for (unsigned int i = nt; i > 0; --i)
  {
    unsigned int nrows = lsizes[i];
    unsigned int ncols = nsamples;
    unsigned int niter = lsizes[i + 1];
    unsigned int count = nrows * ncols;

    if (i == nt)
    {
      if (cmode == COST_RMS)
      {
        // Also take into account the final derivative:
        // We just write the difference of hx and yy in the deltas array:
        T hval;
        for (unsigned int j = 0; j < count; ++j)
        {
          hval = hx[j];
          (*ptr++) = (hval - yy[j]) * hval * (1.0 - hval);
        }
      }
      else
      {
        // We just write the difference of hx and yy in the deltas array:
        for (unsigned int j = 0; j < count; ++j)
        {
          (*ptr++) = hx[j] - yy[j];
        }
      }
    }
    else
    {
      for (unsigned int c = 0; c < ncols; ++c)
      {
        for (unsigned int r = 0; r < nrows; ++r)
        {
          // we want to compute the value delta(r,c);
          double val = 0.0;
          for (unsigned int n = 0; n < niter; ++n)
          {
            // val += theta_T(r+1,n)*delta_prev(n,c);
            // val += theta(n,r+1)*delta_prev(n,c);
            val += params[theta_offset + niter * (r + 1) + n] * deltas[delta_offset + niter * c + n];
          }

          // Then we multiply by the sigmoid gradient at z(r,c):
          double sig = inputs[input_offset + nrows * c + r];
          // deltas[next_delta_offset + nrows*c + r] = next_delta_offset + nrows*c + r;

          // Check here if we should apply the sparse penalty:
          if (i == (nt - 1) && traits.spae_beta > 0.0)
          {
            rval = rho[r];
            val += traits.spae_beta * ( -sp / rval + (1.0 - sp) / (1.0 - rval));
          }

          deltas[next_delta_offset + nrows * c + r] = val * sig * (1.0 - sig);
        }
      }

      // once the computation is done for that layer we move to the previous layer:
      theta_offset -= lsizes[i] * (lsizes[i - 1] + 1);
    }

    delta_offset = next_delta_offset;
    next_delta_offset += count;

    // At this point we have the previous theta matrix (eg. theta(i-1) pointed by theta_offset. (both when i=nt and i<nt).
    // and thats the matrix we need to compute the gradient values.
    // the gradient mat has the same size as the current theta matrix.
    // similarly, the input_offset is pointing on z(i-2) which is the one we need to perform the computation too.
    // and delta_offset points to the delta matrix we just wrote (eg. delta(i)).
    nrows = lsizes[i];
    ncols = lsizes[i - 1] + 1;
    niter = nsamples;
    count = nrows * ncols;

    input_offset -= lsizes[i - 1] * nsamples; // we remove the size of the next delta matrix to be computed. which is also the size of the next z matrix we will use.
    // logDEBUG("CPU: Gradient at i="<<i<<" of size "<< nrows <<" x " << ncols<<", offset="<<grad_offset<<", input_offset="<<input_offset);

    double bias;
    double xw;

    // Compute the gradient:
    for (unsigned int c = 0; c < ncols; ++c)
    {
      for (unsigned int r = 0; r < nrows; ++r)
      {
        // we want to compute the value of the gradient matrix mat(r,c)
        // with mat_i = delta_i * act_i-1.
        double val = 0.0;
        for (unsigned int n = 0; n < nsamples; ++n)
        {
          bias = traits.bias;
          if (traits.dropouts && (abs(sin(n)) > traits.dropouts[i - 1]))
          {
            // Flags the bias value with 0.0 if we should ignore that unit:
            bias = 0.0;
          }

          xw = 1.0;
          // if (traits.dropouts && (abs(sin(niter * (c - 1) + n))) > traits.dropouts[0])
          if (traits.dropouts && (abs(sin((ncols - 1) * n + (c - 1)))) > traits.dropouts[0])
          {
            xw = 0.0;
          }

          // val += delta(r,n)*act(n,c);
          // if c==0 then act[i-1](n,c)==1 otherwise act[i-1](n,c)=z[i-2]_T(n,c-1)=z[i-2](c-1,n)
          // val += deltas[delta_offset + nrows*n + r]; //*(c==0 ? 1.0 : inputs[input_offset + (ncols-1)*n + c-1 ]);
          if (i == 1)
          {
            // Here we have to use the X matrix instead of the z_T.
            // we still want to write the value act(n,c)=x(n,c-1) if c>0
            // Here we also need to check if we want to use the X value in case dropout is enabled:
            // val += deltas[delta_offset + nrows * n + r] * (c == 0 ? bias : X[niter * (c - 1) + n] * xw);
            val += deltas[delta_offset + nrows * n + r] * (c == 0 ? bias : X[(ncols - 1) * n + (c - 1)] * xw);
          }
          else
          {
            int index = input_offset + (ncols - 1) * n + c - 1;
            val += deltas[delta_offset + nrows * n + r] * (c == 0 ? bias : inputs[index]);
          }
          // val += 1.0; //(c==0 ? 1.0 : inputs[input_offset + (ncols-1)*n + c-1 ]);
        }

        // Here we also need to add the regularization from the theta matrix:
        double reg = (c == 0 ? 0.0 : params[theta_offset + nrows * c + r]);
        val += traits.lambda * reg;

        gradients[grad_offset + nrows * c + r] = val / niter; //grad_offset + nrows*c + r; //val/niter;
      }
    }

    // update the gradient offset by removing the size of the next gradient matrix to be computed:
    // except for the last iteration where the value is not available:
    if (i > 1)
    {
      grad_offset -= lsizes[i - 1] * (lsizes[i - 2] + 1);
    }
  }

  // release local resources:
  if (owned_inputs)
  {
    delete [] traits.inputs;
    traits.inputs = nullptr;
  }

  if (owned_deltas)
  {
    delete [] traits.deltas;
    traits.deltas = nullptr;
  }

  if (owned_grads)
  {
    delete [] traits.grads;
    traits.grads = nullptr;
  }

  if (rho)
  {
    delete [] rho;
  }
}

extern "C" {

  void gd_errfunc(BPTraits<double> &traits)
  {
    _gd_errfunc(traits);
  }

  void gd_errfunc_f(BPTraits<float> &traits)
  {
    _gd_errfunc(traits);
  }

  void gd_errfunc_cpu(BPTraits<double> &traits)
  {
    _gd_errfunc_cpu(traits);
  }

}
