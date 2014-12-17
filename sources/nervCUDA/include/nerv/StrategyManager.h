
#ifndef NERV_STRATEGYMANAGER_H_
#define NERV_STRATEGYMANAGER_H_

#include <nerv/Strategy.h>

namespace nerv
{

class NERVCUDA_EXPORT StrategyManager
{
public:
  typedef std::vector<Strategy *> StrategyVector;

public:
  StrategyManager();
  ~StrategyManager();

  Strategy *createStrategy(const Strategy::CreationTraits &traits);
  Strategy *getStrategy(int id);
  void destroyStrategy(Strategy *s);

  void destroyAllStrategies();
  
  static StrategyManager &instance();

protected:
  StrategyVector _strategies;
};

};


#endif
