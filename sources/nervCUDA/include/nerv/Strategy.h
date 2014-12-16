
#ifndef NERV_STRATEGY_H_
#define NERV_STRATEGY_H_

namespace nerv {

class Strategy {
public: 
  Strategy();
  ~Strategy();

  inline int getID() {
  	return _id;
  }
  
protected:
	int _id;
};

};

#endif
