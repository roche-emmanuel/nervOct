
#include <sgtcore.h>

#include <nervCUDA.h>
#include <GradientDescent.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>


#define LOG2 0.6931471805599453

using namespace nerv;

#define THIS "GradientDescent"

template <typename T>
GDTraits<T>::GDTraits(const TrainingSet<T> &tr) : BPTraits<T>()
{
  init();

  nl = tr.nl();
  nsamples_train = tr.nsamples();
  lsizes = tr.lsizes();

  X = tr.X_train();
  X_train_size = tr.X_train_size();

  yy = tr.y_train();
  y_train_size = tr.y_train_size();

  X_cv = tr.X_cv();
  X_cv_size = tr.X_cv_size();

  y_cv = tr.y_cv();
  y_cv_size = tr.y_cv_size();

  nsamples_cv = tr.X_cv_size() / lsizes[0];

  params = tr.params();
  nparams = tr.np();

  maxiter = tr.maxiter();
  lambda = tr.lambda();
}

template <typename T>
void GDTraits<T>::validate() const
{
  // ensure that the traits are usable:
  THROW_IF(nl < 3, "Invalid nl value: " << nl);
  THROW_IF(!lsizes, "Invalid lsizes value.");
  THROW_IF(!nsamples_train, "Invalid nsamples value.")
  THROW_IF(nparams != np(), "Invalid nparams value: " << nparams << "!=" << np());
  THROW_IF(!params, "Invalid params value.");
  THROW_IF(X_train_size != nx(), "Invalid size for X: " << X_train_size << "!=" << nx());
  THROW_IF(!X, "Invalid X_train value.");
  THROW_IF(y_train_size != ny(), "Invalid size for y: " << y_train_size << "!=" << ny());
  THROW_IF(!yy, "Invalid y_train value.");

  unsigned int ns_cv = y_cv_size / lsizes[nl - 1];
  THROW_IF(nsamples_cv != ns_cv, "Mismatch in computation of nsamples_cv" << nsamples_cv << "!=" << ns_cv)
  THROW_IF(miniBatchSize > nsamples_train, "mini-batch size is too big: " << miniBatchSize << ">" << (nsamples_train));
  THROW_IF(validationWindowSize > 0 && (!X_cv || !y_cv), "Invalid cv datasets.");

  if (dropouts)
  {
    for (unsigned int i = 0; i < nl - 1; ++i)
    {
      THROW_IF(dropouts[i] < 0.0 || dropouts[i] > 1.0, "Invalid value for dropout " << i << ":" << dropouts[i]);
    }
  }

  if (wmults)
  {
    for (unsigned int i = 0; i < nl - 1; ++i)
    {
      THROW_IF(wmults[i] < 0.0 || wmults[i] > 1.0, "Invalid value for wmult " << i << ":" << wmults[i]);
    }
  }
}

template <typename T>
GradientDescent<T>::GradientDescent(const GDTraits<T> &traits)
{
  traits.validate();

  _d_traits.allocateStream();
  THROW_IF(_d_traits.stream == 0, "Invalid stream.");

  _bestIter = 0;

  // Assign the max number of iteration:
  _maxiter = traits.maxiter;

  _mumax = traits.momentum;

  _miniBatchSize = traits.miniBatchSize;
  _validationWindowSize = traits.validationWindowSize;

  _nt = traits.nl - 1;

  _lsizes = traits.lsizes;

  // Compute the number of parameters that are expected:
  _np = traits.np();

  // keep a copy of the traits:
  _traits = traits;

  // Now that we checked that the input data is valid, we should allocate the GPU resources:
  // Upload all the buffers on the device:
  _d_traits = traits;

  // velocity vector used to store the NAG velocity for each cycle:
  d_vel = _d_traits.createDeviceBuffer(_np);
  d_vel_bak = _d_traits.createDeviceBuffer(_np);

  // Theta is the array containing the computed network weights at each cycle:
  d_theta = _d_traits.createDeviceBuffer(_np, traits.params);
  d_theta_bak = _d_traits.createDeviceBuffer(_np);

  // keep a reference on the device parameters buffer:
  d_params = _d_traits.params;
}

template <typename T>
T GradientDescent<T>::run()
{
  int iter = 0;
  value_type *current_cost = NULL;

  value_type *X_train_ptr = _d_traits.X_train;
  value_type *y_train_ptr = _d_traits.y_train;

  unsigned int mbOffset = 0;

  // Check if we should use all samples or not:
  unsigned int ns = _d_traits.nsamples_train;
  if (_miniBatchSize > 0)
  {
    ns = _miniBatchSize;
    if (_traits.verbose)
    {
      trDEBUG(THIS, "Using mini batch size: " << _miniBatchSize);
    }
  }

  if(_traits.dropouts) {
    if (_traits.verbose)
    {
      trDEBUG(THIS, "Using dropouts: [" << _traits.dropouts[0]<<" "<<_traits.dropouts[1]<<" ...]");
    }
  }

  bool earlyStopping = false;
  if (_validationWindowSize > 0)
  {
    if (_traits.verbose)
    {
      trDEBUG(THIS, "Using early stopping with window size: " << _validationWindowSize)
    }

    earlyStopping = true;
  }

  WindowedMean<value_type> mean(_validationWindowSize);
  value_type cur_mean = 0.0;
  value_type bestCvCost = std::numeric_limits<value_type>::max();

  // number of cycles after which to evaluation is performed on the cross validation set
  // when using early stopping:
  unsigned int evalFrequency = _traits.evalFrequency;

  // current number of invalid value for Jcv when using early stopping:
  unsigned int invalid_count = 0;
  unsigned int max_invalid_count = 5;


  value_type learning_rate = _traits.epsilon;
  value_type minCvCostDecrease = _traits.minCostDecrease;

  unsigned int ping = _traits.pingFrequency;

  if (_traits.verbose)
  {
    trDEBUG(THIS, "Using learning rate: " << _traits.epsilon);
    trDEBUG(THIS, "Using learning decay: " << _traits.learningDecay);
    trDEBUG(THIS, "Using max momentum: " << _mumax);
    trDEBUG(THIS, "Using lambda: " << _traits.lambda);
    trDEBUG(THIS, "Using minCvCostDecrease: " << _traits.minCostDecrease);
    trDEBUG(THIS, "Using spae_beta: " << _traits.spae_beta);
    trDEBUG(THIS, "Using spae_sparsity: " << _traits.spae_sparsity);
  }

  value_type mu; // current momentum.

  // Run the iteration loop:
  while (_maxiter <= 0 || iter < _maxiter)
  {
    // logDEBUG("Performing iteration "<<iter<<"...");

    // Here we apply the Nesterov Accelerated gradient (NAG) method described in
    // "On the importance of initialization and momentum in deep learning.pdf" [1]
    // so we compute the speed:
    // v(t+1) = mu * v(t) - epsilon * DeltaF(theta(t) + mu * v(t))
    // then theta(t+1) = theta(t)+v(t+1)
    // Where t is the current iteration number (or current optimization time).
    // fist we need to do the partial theta update:
    // d_params = d_theta + mu * v(t);

    // 1. We need to compute the desired value of mu for this cycle:
    // formula from [1]
    mu = (value_type)std::min(1.0 - pow(2.0, -1.0 - log(floor((double)iter / 250.0) + 1) / LOG2), (double)_mumax);

    // 2. prepare the parameter vector:
    mix_vectors_device(_d_traits.params, d_theta, d_vel, (value_type)1.0, mu, _np, _d_traits.stream);

    // 3. Once we have the parameter vector, we compute the gradient at that location:
    _d_traits.X = X_train_ptr;
    _d_traits.yy = y_train_ptr;
    _d_traits.nsamples = ns;

    _d_traits.compute_cost = false;
    _d_traits.compute_grads = true;

    _d_traits.lambda = _traits.lambda;
    _d_traits.dropouts = _traits.dropouts;
    _d_traits.wmults = nullptr;
    _d_traits.params = d_params;

    gd_errfunc_device(_d_traits);

    // logDEBUG("Performing iteration "<<iter<<", Jtrain="<<current_cost);

    // 4. With the gradient we update the velocity vector:
    mix_vectors_device(d_vel, d_vel, _d_traits.grads, mu, -learning_rate, _np, _d_traits.stream);

    // 5. Now that we have the new velocity, we can compute the new value for the theta vector:
    mix_vectors_device(d_theta, d_theta, d_vel, (value_type)1.0, (value_type)1.0, _np, _d_traits.stream);

    // update the value of the learning rate:
    learning_rate *= _traits.learningDecay;

    if (_miniBatchSize > 0)
    {
      // we should move to the next mini batch lot.
      mbOffset += _miniBatchSize;
      // logDEBUG("Using mini-batch offset: "<<mbOffset);

      // check if we have enough samples left or if we should reset to the beginning:
      if ((mbOffset + _miniBatchSize) > _d_traits.nsamples_train)
        mbOffset = 0;

      // logDEBUG("Using mbOffset: "<< mbOffset);

      // update the pointers:
      X_train_ptr = _d_traits.X_train + _lsizes[0] * mbOffset;
      y_train_ptr = _d_traits.y_train + _lsizes[_nt] * mbOffset;
    }

    if (earlyStopping && (iter % evalFrequency == 0))
    {
      // perform evaluation of the current theta value:
      // logDEBUG("Performing cv evaluation at iteration "<<iter<<"...");
      value_type J = computeCvCost();
      if (_traits.cvCostCB)
      {
        _traits.cvCostCB(J, iter, _traits.userdata);
      }

      if (J < bestCvCost)
      {
        trDEBUG_V(THIS, "Updating best Cv cost to " << J << " at iteration " << iter);
        // Here we need to save the best cv cost achieved so far and also
        // save all the relevant parameters in case we need to restart from this point:
        bestCvCost = J;
        saveState(iter, mean);
      }
      // else {
      //  logDEBUG("Discarding worst cv cost: "<<J)
      // }

      // push the new cost on the window mean:
      value_type new_mean = mean.push(J);
      // logDEBUG("New mean value is: "<<new_mean);

      // if we have enough cycles already then check if something is going wrong
      // with the cv cost:
      if (mean.size() == _validationWindowSize)
      {
        value_type dec = (cur_mean - new_mean) / cur_mean;
        if (dec < minCvCostDecrease)
        {
          trDEBUG_V(THIS, "Invalid mean cv cost decrease ratio of: " << dec); //new_mean<<">"<<cur_mean);
          // We count this as an error:
          invalid_count++;
          if (invalid_count > max_invalid_count)
          {
            iter = restoreState(mean);
            // logDEBUG("Max number of invalid Jcv count reached. Resetting the latest best state from iteration "<<iter);
            // value_type J2 = computeCvCost();
            // if (abs(J2 - bestCvCost) > 1e-6)
            // {
            //   trDEBUG(THIS, "Mismatch while restoring best cvcost " << std::setprecision(16) << bestCvCost << "!=" << J2);
            // }

            if (evalFrequency > 1)
            {
              // we reduce the evaluation frequency (assuming it is a power of 2)
              evalFrequency /= 2;
              evalFrequency = evalFrequency < 1 ? 1 : evalFrequency; // ensure the value stays valid.

              invalid_count = 0;
              trDEBUG_V(THIS, "Resetting from iteration " << iter << " with eval frequency of " << evalFrequency)
            }
            else
            {
              trDEBUG(THIS, "Early stopping training with cv cost " << bestCvCost << " from iteration " << iter);
              break;
            }
          }
        }
        else
        {
          // logDEBUG("Current cv cost is: "<<J);

          // Reset the invalid count otherwise:
          // logDEBUG("Resetting invalid count.");
          invalid_count = 0;
        }
      }

      cur_mean = mean.getMean();
    }

    // Finally move to the next cycle:
    iter++;

    if(ping && (iter%ping==0)) {
      trDEBUG(THIS,"On iteration "<<iter<<"...");
    }
  }

  // Download the parameters from the theta buffer on the GPU:
  copyFromDevice(_traits.params, d_theta, _np);

  return bestCvCost;
}

template <typename T>
T GradientDescent<T>::computeTrainCost()
{
  // compute the cost at d_theta location, on complete training set, and not accounting for regularization:
  _d_traits.X = _d_traits.X_train;
  _d_traits.yy = _d_traits.y_train;
  _d_traits.nsamples = _d_traits.nsamples_train;

  _d_traits.compute_cost = true;
  _d_traits.compute_grads = false;

  _d_traits.lambda = 0.0;
  _d_traits.dropouts = nullptr;
  _d_traits.wmults = _traits.dropouts;

  _d_traits.params = d_theta;

  gd_errfunc_device(_d_traits);

  return _d_traits.cost;
}

template <typename T>
T GradientDescent<T>::computeCvCost()
{
  _d_traits.X = _d_traits.X_cv;
  _d_traits.yy = _d_traits.y_cv;
  _d_traits.nsamples = _d_traits.nsamples_cv;

  _d_traits.compute_cost = true;
  _d_traits.compute_grads = false;

  _d_traits.lambda = 0.0;
  _d_traits.dropouts = nullptr;
  _d_traits.wmults = _traits.dropouts;

  _d_traits.params = d_theta;

  // checkCudaErrors(cudaMemset(_d_traits.grads, 0, _d_traits.np()*sizeof(T)));
  // checkCudaErrors(cudaMemset(_d_traits.inputs, 0, _d_traits.nd()*sizeof(T)));
  // checkCudaErrors(cudaMemset(_d_traits.deltas, 0, _d_traits.nd()*sizeof(T)));

  gd_errfunc_device(_d_traits);

  return _d_traits.cost;
}

template <typename T>
void GradientDescent<T>::saveState(unsigned int iter, const WindowedMean<T> &mean)
{
  _bestIter = iter;
  _bestMean = mean;
  // logDEBUG("Saved mean is: "<<_bestMean.getMean());
  copy_vector_device(d_theta_bak, d_theta, _np);
  copy_vector_device(d_vel_bak, d_vel, _np);
}

template <typename T>
unsigned int GradientDescent<T>::restoreState(WindowedMean<T> &mean)
{
  copy_vector_device(d_theta, d_theta_bak, _np);
  copy_vector_device(d_vel, d_vel_bak, _np);
  mean = _bestMean;
  // logDEBUG("Restored mean is: "<<mean.getMean());
  return _bestIter;
}


extern "C" {

  int run_gradient_descent(GDTraits<double> &traits)
  {
    try
    {
      GradientDescent<double> gd(traits);
      gd.run();
      if (traits.compute_cost && traits.validationWindowSize > 0)
      {
        traits.cost = gd.computeCvCost();
        // logDEBUG("Computed final Jcv value: " << traits.cost);
      }
    }
    catch (...) //std::runtime_error &e)
    {
      // error("Exception occured: %s", e.what());
      return GD_EXCEPTION_OCCURED;
    }

    return GD_SUCCESS;
  }

  int run_gradient_descent_f(GDTraits<float> &traits)
  {
    try
    {
      GradientDescent<float> gd(traits);
      gd.run();
      if (traits.compute_cost && traits.validationWindowSize > 0)
      {
        traits.cost = gd.computeCvCost();
        // logDEBUG("Computed final Jcv value: " << traits.cost);
      }
    }
    catch (...) //std::runtime_error &e)
    {
      // error("Exception occured: %s", e.what());
      return GD_EXCEPTION_OCCURED;
    }

    return GD_SUCCESS;
  }

}