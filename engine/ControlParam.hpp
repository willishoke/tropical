#pragma once

#include <atomic>

namespace tropical_expr
{

// ControlParam: lock-free parameter for control-rate values.
// Written from UI/control thread, read per-sample by the DSP evaluator.
// One-pole lowpass smoothing (time_const in seconds) is applied automatically.
// frame_value: written once per frame before kernel execution (used by TriggerParam).
struct ControlParam
{
  std::atomic<double> value;
  double time_const;
  std::atomic<double> frame_value{0.0};

  ControlParam(double init, double tc) : value(init), time_const(tc) {}

  // Non-copyable (std::atomic is non-copyable)
  ControlParam(const ControlParam &) = delete;
  ControlParam & operator=(const ControlParam &) = delete;
};

} // namespace tropical_expr
