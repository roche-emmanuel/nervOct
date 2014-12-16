
#ifndef NERV_STRATEGYMANAGER_H_
#define NERV_STRATEGYMANAGER_H_

#include <nerv/Strategy.h>

namespace nerv
{

class StrategyManager
{
public:
  typedef std::vector<Strategy *> StrategyVector;

public:
  StrategyManager();
  ~StrategyManager();

  Strategy *createStrategy();
  Strategy* getStrategy(int id);
  void destroyStrategy(Strategy *s);

protected:
  StrategyVector _strategies;
};

};

#endif
