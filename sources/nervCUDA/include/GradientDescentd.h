
#ifndef NERV_GRADIENTDESCENTD_H_
#define NERV_GRADIENTDESCENTD_H_

#ifdef GradientDescentClass
#undef GradientDescentClass
#endif

#ifdef GradientDescentValueType
#undef GradientDescentValueType
#endif

#define GradientDescentClass GradientDescentd
#define GradientDescentValueType double

#include "GradientDescent.h"

#endif
