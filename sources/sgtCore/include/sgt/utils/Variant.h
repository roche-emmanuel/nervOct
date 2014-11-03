#ifndef SGT_VARIANT
#define SGT_VARIANT 1

#include <sgtcore.h>
#include <sgt/Object.h>
#include <sgt/utils/VariantUtils.h>

namespace sgt {

typedef std::vector< sgt::RefPtr<sgt::Object> > ObjectVector;

struct EmptyType {

  bool operator==(const EmptyType& rhs) const {
    return true;
  }
};

template<typename T, bool val = std::is_pointer<T>::value >
struct GetReturn {
  typedef T& type; 
};

template<typename T>
struct GetReturn<T,true> {
  typedef T type; 
};

template<typename... Arguments>
class Variant
{
public:
  typedef boost::variant<EmptyType,Arguments...> container_t;

public:
  Variant() : _cont(EmptyType()) {};

  // Copy constructor:
  Variant(const Variant& rhs): _cont(rhs._cont) {};

  // Assignment operator:
  Variant& operator=(const Variant& rhs) { _cont = rhs._cont; return *this;}

  // Comparaison operator:
  template <typename T>
  bool operator==(const T& rhs) const {
    T* val = const_cast<T*>(boost::get<T>(&_cont));
    if(!val)
      return false;

    return *val==rhs;
  }

  bool operator==(const Variant& rhs) const {
    return _cont==rhs._cont;
  }

  template <typename T>
  bool operator!=(const T& rhs) const {
    return !(*this==rhs);
  }

  bool operator!=(const Variant& rhs) const {
    return !(_cont==rhs._cont);
  }

  template<typename T>
  Variant(T& v) : _cont(v) {};

  template<typename T>
  Variant(const T& v) : _cont(v) {};

  template<typename T>
  Variant(const sgt::RefPtr<T>& v) : _cont(sgt::RefPtr<sgt::Object>(v.get())) {};

  template<typename T>
  Variant(sgt::RefPtr<T>& v) : _cont(sgt::RefPtr<sgt::Object>(v.get())) {};

  template<typename T>
  Variant(const T* v) : _cont(sgt::RefPtr<sgt::Object>(v)) {};

  template<typename T>
  Variant(T* v) : _cont(sgt::RefPtr<sgt::Object>(v)) {};
  
  Variant(const char* v) : _cont(std::string(v)) {};

  inline bool empty() { return _cont.which()==0; } // The EmptyType is the first element in the list.

  inline void clear() { _cont = EmptyType(); }

  container_t &operator *() {return _cont;}
  const container_t &operator *() const {return _cont;}

  template <typename T>
  typename GetReturn<T>::type get() const {
    return VariantConverter<T,container_t>::convert(_cont);
  }

  template <typename T>
  typename GetReturn<T>::type get(const T& def) const {
    return VariantConverter<T,container_t>::convert(_cont,def);
  }

  template <typename T>
  typename GetReturn<T>::type get(const T* def) const {
    return VariantConverter<T*,container_t>::convert(_cont,def);
  }

protected:
  container_t _cont;
};

};

#endif
