#include <nervCUDA.h>
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

  // Allocate the GPU resources:
  *devtraits = traits;

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
      delete (*it);
      _devTraits.erase(it);
      break;
    }
  }
}

void BPTraitsManager::mergeDeviceTraits(int id, BPDeviceTraits<value_type>& traits, const BPTraits<value_type>& override)
{
  for (DeviceTraitsVector::iterator it = _devTraits.begin(); it != _devTraits.end(); ++it)
  {
    if ((*it)->id == id)
    {
      logDEBUG("Merging with DevTraits with id " << id);

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

#if 0
extern "C"
{
  BPTraitsManager &get_strategy_manager()
  {
    return BPTraitsManager::instance();
  }
  
  int create_strategy(const Strategy::CreationTraits& traits)
  {
    try
    {
      return get_strategy_manager().createStrategy(traits)->getID();
    }
    catch (...)
    {
      logERROR("Exception occured in create_strategy.")
      return 0;
    }
  }

  int destroy_strategy(int id)
  {
    try
    {
      BPTraitsManager &sm = get_strategy_manager();
      Strategy *s = sm.getStrategy(id);
      THROW_IF(!s, "Cannot find strategy with ID " << id);
      sm.destroyStrategy(s);
      return ST_SUCCESS;
    }
    catch (...)
    {
      logERROR("Exception occured in destroy_strategy.")
      return ST_EXCEPTION_OCCURED;
    }
  }
}

#endif
