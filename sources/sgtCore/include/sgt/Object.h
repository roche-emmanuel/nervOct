#ifndef SGT_OBJECT
#define SGT_OBJECT 1

#include <sgtcore.h>

namespace sgt {

/** Base class for providing reference counted objects.*/
class SGTCORE_EXPORT Object : public sgt::Referenced
{
public:
    Object();

protected:
    virtual ~Object();
};

};

#endif
