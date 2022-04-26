#include <vector>
#include <cmath>
#include <numbers>
#include "vco.h"

VCO::VCO(Signal f) : base_freq(f), base_phase(0.0) {}

VCO::VCO(Signal f, Signal p) : base_freq(f), base_phase(p) {}

void VCO::process() 
{   
  auto fm_val = fm->eval();
  base_phase += base_freq * std::pow(2.0, std::abs(fm_val)) / 44100.0;
  base_phase = std::fmod(base_phase, 1.0);
  auto tz = fm_val > 0.0 ? base_phase : 1.0 - base_phase;
  saw = (2.0 * tz) - 1.0;
  sqr = saw > 0.0 ? 1.0 : -1.0;
  tri = 2.0 * std::abs(saw) - 1.0;
  sin = std::sin(tri * std::numbers::pi / 2.0);
}
