#include <vector>
#include <cmath>
#include <numbers>

#include "obj.h"

using std::abs, std::fmod, std::pow, std::sin;
using std::numbers::pi;

VCO::VCO(Signal f, Signal p) : base_freq(f), base_phase(p) 
{
  // implicit cast to unique_ptr<Expression>
  fm = std::make_unique<Literal>(Literal(0.0));
}

void VCO::process() 
{  
  auto fm_val = fm->eval();
  base_phase += base_freq * pow(2.0, abs(fm_val)) / 44100.0;
  base_phase = fmod(base_phase, 1.0);
  auto tz = fm_val > 0.0 ? base_phase : 1.0 - base_phase;
  outputs[SAW] = (2.0 * tz) - 1.0;
  outputs[SQR] = saw > 0.0 ? 1.0 : -1.0;
  outputs[TRI] = 2.0 * abs(saw) - 1.0;
  outputs[SIN] = std::sin(tri * pi / 2.0);
}

void VCO::update()
{
  outputs[SAW] = this->saw;
  outputs[SQR] = this->sqr;
  outputs[TRI] = this->tri;
  outputs[SIN] = this->sin;
}

// Evaluate expression and store result
void Variable::process()
{
  this->value = expr->eval();
}

// Update reference outputs
void Variable::update()
{
  this->outputs[0] = this->value;
}

extern "C"
{
  Object* makeVCO(double base_freq, double base_phase)
  {
    return new VCO(base_freq, base_phase);
  }
}

// Returned object takes ownership of expression e
extern "C"
{
  Object* makeVariable(Expression* e) 
  {
    return new Variable(e);
  }
}