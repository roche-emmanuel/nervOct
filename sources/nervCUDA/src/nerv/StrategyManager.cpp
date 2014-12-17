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

Strategy *StrategyManager::createStrategy(const Strategy::CreationTraits& traits) 
{
  Strategy *s = new Strategy(traits);
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

StrategyManager& StrategyManager::instance()
{
    static StrategyManager singleton;
    return singleton;  
}

extern "C"
{
  StrategyManager &get_strategy_manager()
  {
    return StrategyManager::instance();
  }
  
  int create_strategy(const Strategy::CreationTraits& traits)
  {
    try
    {
      return get_strategy_manager().createStrategy(traits)->getID();
    }
    catch (...)
    {
      logERROR("Exception occured in destroy_strategy.")
      return 0;
    }
  }

  int destroy_strategy(int id)
  {
    try
    {
      StrategyManager &sm = get_strategy_manager();
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

  int evaluate_strategy(int id, Strategy::EvalTraits& traits)
  {
    try
    {
      StrategyManager &sm = get_strategy_manager();
      Strategy *s = sm.getStrategy(id);
      THROW_IF(!s, "Cannot find strategy with ID " << id);
      s->evaluate(traits);
      return ST_SUCCESS;
    }
    catch (...)
    {
      logERROR("Exception occured in evaluate_strategy.")
      return ST_EXCEPTION_OCCURED;
    }

  }
}

