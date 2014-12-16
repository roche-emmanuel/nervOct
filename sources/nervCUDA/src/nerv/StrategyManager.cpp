#include <nervCUDA.h>
#include <nerv/StrategyManager.h>

using namespace nerv;

StrategyManager::StrategyManager()
{
  logDEBUG("Creating StrategyManager.");
}

StrategyManager::~StrategyManager()
{
  logDEBUG("Destroying StrategyManager.");
}

Strategy *StrategyManager::createStrategy()
{
  Strategy *s = new Strategy();
  _strategies.push_back(s);
  return s;
}

void StrategyManager::destroyStrategy(Strategy *s)
{
  THROW_IF(!s, "Invalid Strategy argument.");
  for (StrategyVector::iterator it = _strategies.begin(); it != _strategies.end(); ++it)
  {
    if ((*it) == s)
    {
      // logDEBUG("Removing strategy with id " << s->getID());
      _strategies.erase(it);
      break;
    }
  }

  delete s;
}

Strategy *StrategyManager::getStrategy(int id)
{
  for (StrategyVector::iterator it = _strategies.begin(); it != _strategies.end(); ++it)
  {
    if ((*it)->getID() == id)
    {
      return (*it);
    }
  }

  return nullptr;
}

extern "C"
{
  StrategyManager &get_strategy_manager()
  {
    static StrategyManager singleton;
    return singleton;
  }

  int create_strategy()
  {
    try
    {
      return get_strategy_manager().createStrategy()->getID();
    }
    catch (...)
    {
      logERROR("Exception occured in destroy_strategy.")
      return 0;
    }
  }

  void destroy_strategy(int id)
  {
    try
    {
      StrategyManager &sm = get_strategy_manager();
      Strategy *s = sm.getStrategy(id);
      THROW_IF(!s, "Cannot find strategy with ID " << id);
      sm.destroyStrategy(s);
    }
    catch (...)
    {
      logERROR("Exception occured in destroy_strategy.")
    }
  }
}

