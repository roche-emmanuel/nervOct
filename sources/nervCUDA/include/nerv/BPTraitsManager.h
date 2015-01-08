
#ifndef NERV_BPTRAITSMANAGER_H_
#define NERV_BPTRAITSMANAGER_H_

#include <nerv/BPTraits.h>

namespace nerv
{

template<typename T>
struct BPDeviceTraits;

class NERVCUDA_EXPORT BPTraitsManager
{
public:
	typedef double value_type;

  typedef std::vector< BPDeviceTraits<value_type> *> DeviceTraitsVector;

public:
  BPTraitsManager();
  ~BPTraitsManager();

  int createDeviceTraits(const BPTraits<value_type> &traits);
  void mergeDeviceTraits(int id, BPDeviceTraits<value_type>& traits, const BPTraits<value_type>& override);
  void destroyDeviceTraits(int id);

  void destroyAllDeviceTraits();
  
  static BPTraitsManager &instance();

protected:
  DeviceTraitsVector _devTraits;
};

};


#endif
