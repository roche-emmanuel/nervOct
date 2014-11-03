#ifndef SGT_VARIANTUTILS
#define SGT_VARIANTUTILS 1

#include <sgtcore.h>

namespace sgt {

template <typename T, typename Container, bool val = std::is_pointer<T>::value >
struct VariantConverter {
  static inline T& convert(const std::string& key, const Container& var) {
    T* val = const_cast<T*>(boost::get<T>(&var));
    THROW_IF(val==nullptr,"Cannot extract the desired value type from variant container at key="<<key);
    return *val;
  };

  static inline T& convert(const Container& var) {
    T* val = const_cast<T*>(boost::get<T>(&var));
    THROW_IF(val==nullptr,"Cannot extract the desired value type from variant container.");
    return *val;
  };

  static inline T& convert(const Container& var, const T& def) {
    T* val = const_cast<T*>(boost::get<T>(&var));
    if(!val)
      return const_cast<T&>(def);
    return *val;
  };

  static inline bool try_convert(const Container& var, T& target) {
    T* val = const_cast<T*>(boost::get<T>(&var));
    if(!val)
      return false;
    target = *val;
    return true;
  };
};

template <typename T, typename Container>
struct VariantConverter<T,Container,true> {
  static inline T convert(const std::string& key, const Container& var) {
    sgt::RefPtr<sgt::Object>* obj =  const_cast< sgt::RefPtr<sgt::Object>* >(boost::get< sgt::RefPtr<sgt::Object> >(&var));
    THROW_IF(obj==nullptr,"Cannot extract the desired value type from variant container at key="<<key);

    return dynamic_cast<T>(obj->get());
  };

  static inline T convert(const Container& var) {
    sgt::RefPtr<sgt::Object>* obj =  const_cast< sgt::RefPtr<sgt::Object>* >(boost::get< sgt::RefPtr<sgt::Object> >(&var));
    THROW_IF(obj==nullptr,"Cannot extract the desired value type from variant container.");

    return dynamic_cast<T>(obj->get());
  };

  static inline T convert(const Container& var, const T def) {
    sgt::RefPtr<sgt::Object>* obj =  const_cast< sgt::RefPtr<sgt::Object>* >(boost::get< sgt::RefPtr<sgt::Object> >(&var));
    if(!obj)
      return def;

    return dynamic_cast<T>(obj->get());
  };  

  static inline bool try_convert(const Container& var, T& target) {
    sgt::RefPtr<sgt::Object>* obj =  const_cast< sgt::RefPtr<sgt::Object>* >(boost::get< sgt::RefPtr<sgt::Object> >(&var));
    if(!obj)
      return false;

    target = dynamic_cast<T>(obj->get());
    return true;
  };

};

};

#endif
