#include "expr.h"
#include "obj.h"

using Signal = double;

struct VCO : public Object
{
    void process();

    std::unique_ptr<Expression> fm, pm;
    Signal base_freq, base_phase;
    Signal sin, sqr, saw, tri;
};