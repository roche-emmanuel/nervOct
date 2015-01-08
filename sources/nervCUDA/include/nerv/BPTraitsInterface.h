
#ifndef NERV_BPTRAITSINTERFACE_H_
#define NERV_BPTRAITSINTERFACE_H_

#ifndef NERVCUDA_EXPORT
#define NERVCUDA_EXPORT
#endif

#include <windows.h>
#include <sstream>
#include <stdio.h>
#include <iostream>

#include <nerv/BPTraitsManager.h>

#define SLOG(msg) std::cout << msg << std::endl;
#define STHROW(msg) { std::ostringstream os; os << msg; SLOG("[ERROR] Throwing exception: " << msg); throw std::runtime_error(os.str()); }
#define SCHECK(cond,msg) if(!(cond)) STHROW(msg);

#define DECLARE_PROC(pname) pname##_t pname;
#define GET_PROC(ptype) ptype = (ptype##_t)GetProcAddress(_h,#ptype); \
	SCHECK(ptype,"Invalid " #ptype " pointer.")

namespace nerv
{

class BPTraitsInterface
{
public:
  typedef int (*create_device_traits_t)(const BPTraits<double> &traits);
  typedef int (*destroy_device_traits_t)(int id);
  typedef int (*compute_costfunc_device_t)(int id, BPTraits<double> &over);
public:
  BPTraitsInterface()
  {
  	// Load the library:
	  _h = LoadLibrary("nervCUDA.dll");
	  SCHECK(_h,"Invalid handle for nervCUDA library.")

	  // retrieve the methods:
	  GET_PROC(create_device_traits)
	  GET_PROC(destroy_device_traits)
	  GET_PROC(compute_costfunc_device)
  };

  ~BPTraitsInterface() {
  	// Release the library:
  	SCHECK(FreeLibrary(_h),"Cannot release nervCUDA library.");
  }

	DECLARE_PROC(create_device_traits)
	DECLARE_PROC(destroy_device_traits)
	DECLARE_PROC(compute_costfunc_device)
	
protected:
	HMODULE _h;
};

};

#undef GET_PROC
#undef DECLARE_PROC
#undef SCHECK
#undef STHROW
#undef SLOG

#endif
