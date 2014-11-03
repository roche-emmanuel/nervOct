#ifndef SGT_EVENTGENERATOR
#define SGT_EVENTGENERATOR 1

#include <sgtmx.h>
#include <sgt/Object.h>
#include <sgt/utils/Variant.h>

namespace sgt {


/** We implement here  various type of functions that can be called when an
event is triggered. All the supported function signatures are passed as argument.
Note that a corresponding boost visitor should also be created to support the
functions call. */
template<typename Parent, typename VariantType> //typename... Arguments
class EventGenerator : public Parent
{
public:
  // typedef sgt::Variant<Arguments...> argument_t;
  typedef VariantType argument_t;

  typedef std::function<void(const argument_t&, const argument_t&)> function2_t;
  typedef std::function<void(const argument_t&)> function1_t;
  typedef std::function<void()> function0_t;

  typedef boost::signals2::connection connection_t;

public:
  void triggerEvent(const std::string& eventName, const argument_t& arg1, const argument_t& arg2) const {
    auto it = _signals.find(eventName);
    if(it != _signals.end())
      it->second->operator()(arg1,arg2);
  }

  void triggerEvent(const std::string& eventName, const argument_t& arg1) const {
    auto it = _signals.find(eventName);
    if(it != _signals.end())
      it->second->operator()(arg1,_emptyArg);
  }

  void triggerEvent(const std::string& eventName) const {
    auto it = _signals.find(eventName);
    if(it != _signals.end())
      it->second->operator()(_emptyArg,_emptyArg);    
  }

  /** Add a new event listener to the list. and return a connection object.
  To disconnect, just call connection.disconnect().*/
  connection_t addEventListener2(const std::string& eventName, function2_t func) {
    auto it = _signals.find(eventName);
    if(it == _signals.end())
      _signals[eventName] = std::make_shared<SignalType>();

    return _signals[eventName]->connect(func);
  }

  connection_t addEventListener1(const std::string& eventName, function1_t func) {
    auto it = _signals.find(eventName);
    if(it == _signals.end())
      _signals[eventName] = std::make_shared<SignalType>();

    return _signals[eventName]->connect([=] (const argument_t& arg1, const argument_t& arg2) {func(arg1);});
  }

  connection_t addEventListener0(const std::string& eventName, function0_t func) {
    auto it = _signals.find(eventName);
    if(it == _signals.end())
      _signals[eventName] = std::make_shared<SignalType>();

    return _signals[eventName]->connect([=] (const argument_t& arg1, const argument_t& arg2) {func();});    
  }

protected:
  typedef boost::signals2::signal<void(const argument_t&, const argument_t& arg2)> SignalType;
  typedef std::map<std::string, std::shared_ptr<SignalType> > SignalMap;

  SignalMap _signals;
  argument_t _emptyArg;
};

};

#endif
