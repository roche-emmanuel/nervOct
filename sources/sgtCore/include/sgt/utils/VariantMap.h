#ifndef SGT_VARIANTMAP
#define SGT_VARIANTMAP 1

#include <sgtcore.h>
#include <sgt/Object.h>
#include <sgt/utils/Variant.h>

using namespace sgt;

namespace sgt {

template<typename Parent, typename ThisClass, typename... Arguments>
class VariantMap : public Parent
{
public:
  typedef sgt::Variant<Arguments...> data_t;
  typedef typename data_t::container_t variant_t;

  typedef std::map<std::string,data_t> map_t;
  typedef typename map_t::iterator iterator_t;
  typedef typename map_t::const_iterator const_iterator_t;

public:
  VariantMap() {};
  virtual ~VariantMap() {};  // Not protected to allow stack creation.

  virtual void setValue(const std::string& key, const data_t& val) {
    _data[key] = val;
  };

  virtual data_t& getValue(const std::string& key) {
    auto it = _data.find(key);
    THROW_IF(it == _data.end(),"Cannot find the key " << key);

    return it->second;
  };

  VariantMap(std::initializer_list< std::pair<std::string,data_t> > data) {
    FOREACH(data) {
      setValue(it->first,it->second);
    }
  }

  inline iterator_t begin() { return _data.begin(); }
  inline iterator_t end() { return _data.end(); }

  inline const_iterator_t begin() const { return _data.begin(); }
  inline const_iterator_t end() const { return _data.end(); }

  inline size_t size() const { return _data.size(); }
  
  inline sgt::RefPtr<ThisClass> clone() const {
    sgt::RefPtr<ThisClass> result = new ThisClass();
    FOREACH(_data) {
      result->setValue(it->first,it->second);
    }
    return result;
  }

  inline bool hasKey(const std::string& key) const {
  	return _data.find(key)!= _data.end();
  };

  void set(const std::string& key, const data_t& value) {
    setValue(key,value);
  }

  /** Erase a key if it exists.*/
  bool unset(const std::string& key) {
    auto it = _data.find(key);
    if(it!=_data.end()) {
      _data.erase(it);
      return true;
    }

    return false;    
  }

  template <typename T>
  typename GetReturn<T>::type get(const std::string& key) const {
    auto it = _data.find(key);
    THROW_IF(it == _data.end(),"Cannot find the key " << key);

    // Check if we can extract the value:
    // return it->second.get<T>();
    return VariantConverter<T,variant_t>::convert(key,*(it->second)); // const_cast<T*>(boost::get<T>(&(it->second)));
  }

  template <typename T>
  T& get(const std::string& key, const T& def) const {
    auto it = _data.find(key);
    if(it==_data.end())
      return const_cast<T&>(def);

    // Check if we can extract the value:
    // return it->second.get<T>(def);
    return VariantConverter<T,variant_t>::convert(*(it->second),def); // const_cast<T*>(boost::get<T>(&(it->second)));
  }

  template <typename T>
  T* get(const std::string& key, const T* def) const {
    auto it = _data.find(key);
    if(it==_data.end())
      return const_cast<T*>(def);

    // Check if we can extract the value:
    return VariantConverter<T*,variant_t>::convert(*(it->second),def); // const_cast<T*>(boost::get<T>(&(it->second)));
  }

  template <typename T>
  typename GetReturn<T>::type getOrCreate(const std::string& key) {
    auto it = _data.find(key);
    if(it==_data.end()) {
      set(key,T());
      it = _data.find(key);
    }

    // Check if we can extract the value:
    return VariantConverter<T,variant_t>::convert(key, *(it->second)); // const_cast<T*>(boost::get<T>(&(it->second)));
  }

  template <typename T >
  bool get_optional(const std::string& key, T& target) const {
    auto it = _data.find(key);
    if(it == _data.end())
      return false;

    // Check if we can extract the value:
    return VariantConverter<T,variant_t>::try_convert(*(it->second),target); // boost::get<T>(&(it->second));
  }

  template <typename T>
	T& pick(const T& def, const std::string& key1, const std::string& key2 = "", const std::string& key3 = "") const {
		T& val = const_cast<T&>(def);

		if(get_optional<T>(key1,val))
			return val;
		if(get_optional<T>(key2,val))
			return val;

		get_optional<T>(key3,val);
		return val;
	}

  template <typename T>
	typename GetReturn<T>::type fetch(const std::string& key1, const std::string& key2 = "", const std::string& key3 = "") const {
    if(hasKey(key1))
      return get<T>(key1);
    if(hasKey(key2))
      return get<T>(key2);
    if(hasKey(key3))
      return get<T>(key3);

    // We could not fetch from any of the keys, so we trigger an error:
    if(!key3.empty()) {
      THROW("Cannot fetch data from the keys: " << key1 << ", " << key2 << ", "<<key3);  
    }
    else if(!key2.empty()) {
      THROW("Cannot fetch data from the keys: " << key1 << ", " << key2);        
    }
    else {
      THROW("Cannot fetch data from the key: " << key1);         
    }    
	}

  /** Merge with another DataMap object. */
  void merge(const VariantMap& vmap) {
    CHECK(&vmap != this,"Merging a VariantMap with itself?");

    // Override all the values provided in the new data map:
    FOREACH(vmap) {
      // logDEBUG("Merging key " << it->first);
      // this->operator[](it->first) = it->second;
      setValue(it->first,it->second);
    }

  }

protected:
  map_t _data;
};

};

#endif
