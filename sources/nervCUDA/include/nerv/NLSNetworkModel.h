
#ifndef NERV_NLSNETWORKMODEL_H_
#define NERV_NLSNETWORKMODEL_H_

#include <nerv/Model.h>
#include <nerv/BPDeviceTraits.h>

namespace nerv
{

class NERVCUDA_EXPORT NLSNetworkModel : public Model
{
public:
  NLSNetworkModel(const CreationTraits &traits);
  virtual ~NLSNetworkModel();

  virtual void predict(DigestTraits &traits, value_type &long_prob, value_type &short_prob, value_type &none_prob);

protected:
  BPDeviceTraits<value_type> _dtraits;
  unsigned int *_lsizes;
  value_type *_hx;
};

};

#endif
