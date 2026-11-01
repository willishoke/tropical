// The clock rail's format: τ and every clock are signed i64 with
// kClockFracBits fraction bits (Q48.16). Mirrors
// lean/Tropical/Ir/ClockFormat.lean; the kernel manifest names the format
// as τ's element type, kTauElem, and the loader refuses any other.
#pragma once

#include <cstdint>

namespace tropical_clock
{

inline constexpr int kClockFracBits = 16;
inline constexpr const char * kTauElem = "i64q16";

// Whole sample n as a coordinate on the rail. The shift wraps modulo 2⁶⁴
// past the horizon (2⁴⁷ samples), as the kernel's own clock arithmetic does.
constexpr int64_t clock_of_sample(uint64_t n)
{
  return static_cast<int64_t>(n << kClockFracBits);
}

} // namespace tropical_clock
