
#ifndef SGT_CORE_H_
#define SGT_CORE_H_

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
    #  if defined( SGTCORE_LIBRARY_STATIC )
    #    define SGTCORE_EXPORT
    #  elif defined( SGTCORE_LIBRARY )
    #    define SGTCORE_EXPORT   __declspec(dllexport)
    #  else
    #    define SGTCORE_EXPORT   __declspec(dllimport)
    #  endif
#else
    #  define SGTCORE_EXPORT
#endif

#if defined(_MSC_VER)
    // #pragma warning( disable : 4244 )
    #pragma warning( disable : 4251 )
    #pragma warning( disable : 4275 )
#endif

#include <map>
#include <set>
#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <typeinfo>
#include <memory>

#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <boost/filesystem.hpp>
#include <boost/signals2/signal.hpp>
#include <boost/property_tree/ptree.hpp>

// Configuration elements:
// #define SGT_LOG_RELEASE_MODE
// #define SGT_USE_REF_PTR_IMPLICIT_OUTPUT_CONVERSION

#include "sgt/RefPtr.h"
#include "sgt/ObserverPtr.h"
#include "sgt/Referenced.h"

#include "sgt/logging.h"

namespace sgt {

SGTCORE_EXPORT bool writeFile(const std::string& filename, void* data, unsigned long long len);
}

struct no_pimpl {};

#include "sgt/macros.h"

#define SCOPEDLOCK(m) boost::mutex::scoped_lock lock(m);

#define FOREACH_EX(cont, it) for(auto it = (cont).begin(); it != (cont).end(); ++it)
#define FOREACH(cont) FOREACH_EX(cont, it)
#define FOREACH_CONST_EX(cont, it) for(auto it = (cont).begin(); it != (cont).end(); ++it)
#define FOREACH_CONST(cont) FOREACH_CONST_EX(cont, it)


#define EACH(it,v) \
      for(typeof(v.begin()) it = v.begin();it != v.end(); ++it)
#ifdef WIN32
#define DEBUG_MSG(msg) { std::ostringstream os; os << msg; MessageBox(NULL,os.str().c_str(),"DEBUG",MB_OK); }
#else
#define DEBUG_MSG(msg) logDEBUG(msg)
#endif

#define THROW(msg) { std::ostringstream os; os << msg; logERROR("Throwing exception: " << msg); throw std::runtime_error(os.str()); }
#define THROW_IF(cond,msg) if(cond) THROW(msg)
#define CHECK_EQ(val,expected,msg) if((val)!=(expected)) { logERROR(msg << " (Expected: " << (expected) << " and got: " << (val) << ")"); return; }
#define CHECK_EQ_RET(val,expected,ret,msg) if((val)!=(expected)) { logERROR(msg << " (Expected: " << (expected) << " and got: " << (val) << ")"); return ret; }

#define CHECK(val,msg) if(!(val)) { logERROR(msg); return; }
#define CHECK_RET(val,ret,msg) if(!(val)) { logERROR(msg); return ret; }
#define TRY(code,msg) try { code; } catch(std::exception& e) { logERROR(msg << " - Exception occured: " << e.what()); }
#define TRY_RET(code,ret,msg) try { code; } catch(std::exception& e) { logERROR(msg << " - Exception occured: " << e.what()); return ret; }


#define DEPRECATED(msg) { logWARN("Deprecated: " << msg); }

#define WRITEFILE(filename, content) {std::ostringstream os; os << content; std::string text = os.str(); CHECK(sgt::writeFile(filename,(void*)text.data(),text.size()),"Could not write file " << filename); }

#endif
