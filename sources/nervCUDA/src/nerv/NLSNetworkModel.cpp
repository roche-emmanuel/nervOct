#include <nervCUDA.h>
#include <nerv/NLSNetworkModel.h>
#include <sgtCore.h>

using namespace nerv;

NLSNetworkModel::NLSNetworkModel(const CreationTraits &traits)
  : _lsizes(nullptr), _hx(nullptr), Model(traits)
{
  logDEBUG("Creating NLSNetworkModel.");

  THROW_IF(traits.type != MODEL_NLS_NETWORK, "Invalid model type for NLSNetwork.");
  THROW_IF(traits.nl < 3, "Invalid number of layers.")
  THROW_IF(!traits.lsizes, "Invalid lsizes pointer.")
  THROW_IF(!traits.params, "Invalid parameters pointer.")
  THROW_IF(!traits.mu, "Invalid mu pointer.")
  THROW_IF(!traits.sigma, "Invalid sigma pointer.")

  unsigned int nout = traits.lsizes[traits.nl - 1];
  THROW_IF(nout != 3, "Invalid number of outputs: " << nout);

  // We have to keep our own copy of CPU buffers here as we don't know when the parameter buffers will go
  // out of scope.
  // Note that we dont have to do that for the params buffer as this one is only copied on the GPU once anyway.
  _lsizes = new unsigned int[traits.nl];
  memcpy((void *)_lsizes, traits.lsizes, traits.nl * sizeof(unsigned int));

  _hx = new value_type[nout];

  _nf = _lsizes[0];
  _mu = new value_type[_nf];
  memcpy((void*)_mu, traits.mu, _nf*sizeof(value_type));

  _sigma = new value_type[_nf];
  memcpy((void*)_sigma, traits.sigma, _nf*sizeof(value_type));

  _input = new value_type[_nf];

  // Build BPTraits from the creation traits:
  BPTraits<value_type> btraits;
  btraits.nl = traits.nl;
  btraits.lsizes = _lsizes;
  btraits.nsamples_train = 1;
  btraits.params = traits.params;
  // No need to assign the X matrix here, it will only be created on the GPU.

  // Now prepare the device traits:
  _dtraits = btraits;
}

NLSNetworkModel::~NLSNetworkModel()
{
  logDEBUG("Destroying NLSNetworkModel.");
  delete [] _lsizes;
  delete [] _hx;
  delete [] _mu;
  delete [] _sigma;
  delete [] _input;
}

void NLSNetworkModel::predict(DigestTraits &traits, value_type &long_prob, value_type &short_prob, value_type &none_prob)
{
  // first we upload the input buffer onthe GPU.
  THROW_IF(traits.input_size != _dtraits.nx(), "Mismatch in number of features: " << traits.input_size << "!=" << _dtraits.nx())
  
  // first we copy on our internal buffer to be able to apply normalization:  
  memcpy((void*)_input, traits.input, traits.input_size*sizeof(value_type));

  // Apply normalization:
  
  for(unsigned int i=0;i<_nf;++i) {
    _input[i] = (_input[i] - _mu[i])/_sigma[i];
  }

  // for(unsigned int i=0;i<10;++i) {
  //   logDEBUG("Input "<<i<<": "<<traits.input[i]);
  // }
  copyToDevice(_dtraits.X, _input, _nf);

  int input_offset = nn_activation_device(_dtraits);

  copyFromDevice(_hx, _dtraits.inputs + input_offset, _dtraits.ny());

  logDEBUG("Hx values: "<<_hx[0]<<", "<<_hx[1]<<", "<<_hx[2]<<" (ny="<<_dtraits.ny()<<")");

  // Retrieve the computed probabilities:
  value_type total = _hx[0] + _hx[1] + _hx[2];
  none_prob = _hx[0] / total;
  long_prob = _hx[1] / total;
  short_prob = _hx[2] / total;
}

