
#ifndef NERV_MBP_H_
#define NERV_MBP_H_

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
    #  if defined( NERVMBP_LIBRARY_STATIC )
    #    define NERVMBP_EXPORT
    #  elif defined( NERVMBP_LIBRARY )
    #    define NERVMBP_EXPORT   __declspec(dllexport)
    #  else
    #    define NERVMBP_EXPORT   __declspec(dllimport)
    #  endif
#else
    #  define NERVMBP_EXPORT
#endif

#include <sgtcore.h>

#endif
