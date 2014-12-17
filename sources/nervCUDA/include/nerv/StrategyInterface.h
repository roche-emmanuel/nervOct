
#ifndef NERV_STRATEGYINTERFACE_H_
#define NERV_STRATEGYINTERFACE_H_

#include <windows.h>
#include <sstream>
#include <stdio.h>
#include <iostream>

#include <nerv/StrategyManager.h>

#define SLOG(msg) std::cout << msg << std::endl;
#define STHROW(msg) { std::ostringstream os; os << msg; SLOG("[ERROR] Throwing exception: " << msg); throw std::runtime_error(os.str()); }
#define SCHECK(cond,msg) if(!(cond)) STHROW(msg);

#define DECLARE_PROC(pname) pname##_t pname;
#define GET_PROC(ptype) ptype = (ptype##_t)GetProcAddress(_h,#ptype); \
	SCHECK(ptype##,"Invalid " #ptype " pointer.")

namespace nerv
{

class StrategyInterface
{
public:
  typedef StrategyManager &(*get_strategy_manager_t)();
  typedef int (*create_strategy_t)(const Strategy::CreationTraits& traits);
  typedef int (*destroy_strategy_t)(int id);
  typedef int (*evaluate_strategy_t)(int id, const Strategy::EvalTraits& traits);

public:
  StrategyInterface()
  {
  	// Load the library:
	  _h = LoadLibrary("nervCUDA.dll");
	  SCHECK(_h,"Invalid handle for nervCUDA library.")

	  // retrieve the methods:
	  GET_PROC(get_strategy_manager)
	  GET_PROC(create_strategy)
	  GET_PROC(destroy_strategy)
	  GET_PROC(evaluate_strategy)
  };

  ~StrategyInterface() {
  	// Release the library:
  	SCHECK(FreeLibrary(_h),"Cannot release nervCUDA library.");
  }

	DECLARE_PROC(get_strategy_manager)
	DECLARE_PROC(create_strategy)
	DECLARE_PROC(destroy_strategy)
	DECLARE_PROC(evaluate_strategy)

protected:
	HMODULE _h;
};

};

#endif
