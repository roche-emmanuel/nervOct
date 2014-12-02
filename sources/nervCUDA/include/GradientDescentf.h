
#ifndef NERV_GRADIENTDESCENTF_H_
#define NERV_GRADIENTDESCENTF_H_

#ifdef GradientDescentClass
#undef GradientDescentClass
#endif

#ifdef GradientDescentValueType
#undef GradientDescentValueType
#endif

#define GradientDescentClass GradientDescentf
#define GradientDescentValueType float

#include "GradientDescent.h"

#endif
