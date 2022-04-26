#include "obj.h"
#include "vco.h"

Object* makeVCO(double base_freq)
{
    return new VCO(base_freq);
}