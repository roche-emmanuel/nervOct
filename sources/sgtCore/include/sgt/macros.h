#ifndef SGTCORE_MACROS_
#define SGTCORE_MACROS_

#define DECLARE_PIMPL() public: \
    struct PImpl;\
private:\
  inline PImpl* self() { return (PImpl*)_pimpl.get(); } \
  inline PImpl const* self() const { return (PImpl const*)_pimpl.get(); }

#define DECLARE_PIMPL_BASE() protected: \
    sgt::RefPtr<sgt::Object> _pimpl; \
    DECLARE_PIMPL()

#define DECLARE_EVENT_EMITTER(argtype) public: \
  typedef std::function<void(const argtype&, const argtype&)> function2_t; \
  typedef std::function<void(const argtype&)> function1_t; \
  typedef std::function<void()> function0_t; \
  typedef boost::signals2::connection connection_t; \
  void triggerEvent(const std::string& eventName, const argtype& arg1, const argtype& arg2) const; \
  void triggerEvent(const std::string& eventName, const argtype& arg1) const; \
  void triggerEvent(const std::string& eventName) const; \
  connection_t addEventListener2(const std::string& eventName, function2_t func); \
  connection_t addEventListener1(const std::string& eventName, function1_t func); \
  connection_t addEventListener0(const std::string& eventName, function0_t func);

#define IMPLEMENT_EVENT_EMITTER_PIMPL(cname,argtype,emitter) \
void cname::triggerEvent(const std::string& eventName, const argtype& arg1, const argtype& arg2) const \
{ \
  self()->emitter->triggerEvent(eventName,arg1,arg2); \
} \
void cname::triggerEvent(const std::string& eventName, const argtype& arg1) const \
{ \
  self()->emitter->triggerEvent(eventName,arg1); \
} \
void cname::triggerEvent(const std::string& eventName) const \
{ \
  self()->emitter->triggerEvent(eventName); \
} \
cname::connection_t cname::addEventListener2(const std::string& eventName, function2_t func) \
{ \
  return self()->emitter->addEventListener2(eventName,func); \
} \
cname::connection_t cname::addEventListener1(const std::string& eventName, function1_t func) \
{ \
  return self()->emitter->addEventListener1(eventName,func); \
} \
cname::connection_t cname::addEventListener0(const std::string& eventName, function0_t func) \
{ \
  return self()->emitter->addEventListener0(eventName,func); \
}

#endif
