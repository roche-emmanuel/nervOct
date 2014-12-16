#include <nervCUDA.h>
#include <nerv/Strategy.h>

using namespace nerv;

Strategy::Strategy()
{	
	// Assign an unique ID to this strategy;
	static int counter = 0;
	_id = ++counter;
	
  logDEBUG("Creating Strategy "<<_id<<".");
}

Strategy::~Strategy()
{
  logDEBUG("Destroying Strategy "<<_id<<".");
}

