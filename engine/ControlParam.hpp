#pragma once

#include <atomic>

namespace tropical_expr
{

// ControlParam: lock-free parameter for control-rate values.
// Written from UI/control thread, read per-sample by the DSP evaluator.
// One-pole lowpass smoothing (time_const in seconds) is applied automatically.
struct ControlParam
{
  std::atomic<double> value;
  double time_const;

  ControlParam(double init, double tc) : value(init), time_const(tc) {}

  // Non-copyable (std::atomic is non-copyable)
  ControlParam(const ControlParam &) = delete;
  ControlParam & operator=(const ControlParam &) = delete;
};

} // namespace tropical_expr
