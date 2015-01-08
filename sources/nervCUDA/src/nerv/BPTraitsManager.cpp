#include <nervCUDA.h>
#include <nerv/BPDeviceTraits.h>

#include <nerv/BPTraitsManager.h>

using namespace nerv;

BPTraitsManager::BPTraitsManager()
{
  logDEBUG("Creating BPTraitsManager.");
}

BPTraitsManager::~BPTraitsManager()
{
  logDEBUG("Destroying BPTraitsManager.");

  // Ensure all registered strategies are removed:
  destroyAllDeviceTraits();
}

int BPTraitsManager::createDeviceTraits(const BPTraits<value_type> &traits)
{
  static int counter = 0;
  int id = ++counter;

  BPDeviceTraits<value_type>* devtraits = new BPDeviceTraits<value_type>();

  // Here we should create our own copy of the lsizes and dropouts if applicable:
  unsigned int* lsizes = new unsigned int[traits.nl];
  memcpy(lsizes,traits.lsizes,traits.nl*sizeof(unsigned int));

  value_type* dropouts = nullptr;
  if(traits.dropouts) {
    dropouts = new value_type[traits.nl-1];
    memcpy(dropouts,traits.dropouts,(traits.nl-1)*sizeof(value_type));
  }

  // Allocate the GPU resources:
  *devtraits = traits;

  // Override the lsizes and dropouts:
  devtraits->lsizes = lsizes;
  devtraits->dropouts = dropouts;

  // Assign the ID:
  devtraits->id = id;

  _devTraits.push_back(devtraits);
  return id;
}

void BPTraitsManager::destroyDeviceTraits(int id)
{
  for (DeviceTraitsVector::iterator it = _devTraits.begin(); it != _devTraits.end(); ++it)
  {
    if ((*it)->id == id)
    {
      logDEBUG("Removing DevTraits with id " << id);
      
      // delete the lsizes and dropout arrays:
      BPDeviceTraits<value_type>* devtraits = (*it);
      
      delete [] devtraits->lsizes;
      delete [] devtraits->dropouts;

      delete devtraits;

      _devTraits.erase(it);
      return;
    }
  }

  THROW("Cannot find device traits with ID " << id);
}

void BPTraitsManager::mergeDeviceTraits(int id, BPDeviceTraits<value_type>& traits, const BPTraits<value_type>& override)
{
  for (DeviceTraitsVector::iterator it = _devTraits.begin(); it != _devTraits.end(); ++it)
  {
    if ((*it)->id == id)
    {
      logDEBUG("Merging with DevTraits with id " << id);
      
      BPDeviceTraits<value_type>* devtraits = (*it);
      // copy the traits:
      traits = *devtraits;

      if(override.params) {
        // upload the new parameters on the GPU:
        copyToDevice(traits.params,override.params,traits.np());
      }
    }
  }
}

void BPTraitsManager::destroyAllDeviceTraits()
{
  for (DeviceTraitsVector::iterator it = _devTraits.begin(); it != _devTraits.end(); ++it)
  {
    delete (*it);
  }

  _devTraits.clear();
}

BPTraitsManager& BPTraitsManager::instance()
{
    static BPTraitsManager singleton;
    return singleton;  
}

extern "C"
{
  int create_device_traits(const BPTraits<double> &traits)
  {
    try
    {
      return BPTraitsManager::instance().createDeviceTraits(traits);
    }
    catch (...)
    {
      logERROR("Exception occured in create_device_traits.")
      return 0;
    }
  }

  int destroy_device_traits(int id)
  {
    try
    {
      BPTraitsManager::instance().destroyDeviceTraits(id);
      return TRAITS_SUCCESS;
    }
    catch (...)
    {
      logERROR("Exception occured in destroy_device_traits.")
      return TRAITS_EXCEPTION_OCCURED;
    }
  }

  int compute_costfunc_device(int id, BPTraits<double> &over)
  {
    try
    {

      BPDeviceTraits<BPTraitsManager::value_type> dtraits;

      BPTraitsManager::instance().mergeDeviceTraits(id,dtraits,over);

      // Now that we have valid device traits we should call the errfunc method:
      gd_errfunc_device(dtraits);

      // Copy the results back into the over traits:
      over.cost = dtraits.cost;
      copyFromDevice(over.grads,dtraits.grads,dtraits.np());

      return TRAITS_SUCCESS;
    }
    catch (...)
    {
      logERROR("Exception occured in compute_costfunc_device.")
      return TRAITS_EXCEPTION_OCCURED;
    }
  }

}

