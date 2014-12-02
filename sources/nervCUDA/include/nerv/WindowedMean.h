
#ifndef NERV_WINDOWEDMEAN_H_
#define NERV_WINDOWEDMEAN_H_

#include <nervCUDA.h>
#include <deque>

namespace nerv {

template<typename T>
class WindowedMean {
public: 
  WindowedMean(unsigned int maxSize = 0) : _maxSize(maxSize), _totalValue(0.0) {};

  WindowedMean(const WindowedMean& rhs) {
    _totalValue = rhs._totalValue;
    _stack = rhs._stack;
    _maxSize = rhs._maxSize;
    }

    WindowedMean& operator=(const WindowedMean& rhs) {
    _totalValue = rhs._totalValue;
    _stack = rhs._stack;
    _maxSize = rhs._maxSize;
    return *this;
    }

    inline unsigned int size() { return (unsigned int)_stack.size(); }
    inline T getMean() { return (T)(_stack.size()==0.0 ? 0.0 : _totalValue/_stack.size()); }
    T push(T val);

protected:
   T _totalValue;
   std::deque<T> _stack;
   unsigned int _maxSize;
};

template<typename T>
T WindowedMean<T>::push(T val) {
    _stack.push_back(val);
    _totalValue += val;

    if(_stack.size() > _maxSize) {
        // remove the initial value
        T pval = _stack.front();
        _stack.pop_front();
        _totalValue -= pval;
    }

    return _totalValue/_stack.size();
}

};

#endif
