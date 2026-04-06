#pragma once

/**
 * FlatRuntime.hpp — Minimal plan executor for JIT-compiled audio.
 *
 * Receives a flat set of expression trees (outputs + registers) via JSON plan,
 * JIT-compiles them into a single native kernel, and runs it per sample.
 * No module boundaries, no graph, no orchestration.
 */

#include "graph/GraphTypes.hpp"
#include "jit/OrcJitEngine.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace egress_runtime
{

struct KernelState
{
  egress_jit::NumericKernelFn kernel = nullptr;

  // Flat buffers passed to kernel (matches NumericKernelFn signature)
  std::vector<int64_t> registers;
  std::vector<int64_t> temps;
  std::vector<std::vector<int64_t>> array_storage;
  std::vector<int64_t *> array_ptrs;
  std::vector<uint64_t> array_sizes;
  std::vector<uint64_t> param_ptrs;

  // Register names for state transfer on hot-swap
  std::vector<std::string> register_names;
  // Array slot names for array state transfer on hot-swap
  std::vector<std::string> array_names;

  // Output extraction
  uint32_t output_count = 0;

  // Trigger params (need per-frame snapshot)
  std::vector<egress_expr::ControlParam *> trigger_params;

  double sample_rate = 44100.0;
  uint64_t sample_index = 0;
};


class FlatRuntime
{
public:
  explicit FlatRuntime(unsigned int buffer_length)
    : buffer_length_(buffer_length),
      outputBuffer(buffer_length, 0.0)
  {
  }



  /**
   * Load a plan JSON string, compile to a single kernel, and publish atomically.
   *
   * Plan schema (egress_plan_3):
   * {
   *   "schema": "egress_plan_3",
   *   "config": { "sample_rate": 44100 },
   *   "state_init": [0.0, ...],
   *   "register_names": ["VCO1_phase", ...],
   *   "outputs": [0, 2, ...],
   *   "instructions": [...],
   *   "register_count": N,
   *   "array_slot_sizes": [...],
   *   "output_targets": [...],
   *   "register_targets": [...]
   * }
   */
  bool load_plan(const std::string & plan_json);

  /**
   * Process one buffer of audio. Called from the audio thread.
   * Fills outputBuffer with buffer_length_ samples.
   */
  void process()
  {

    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    audio_state_index_.store(state_idx, std::memory_order_release);
    audio_processing_.store(true, std::memory_order_release);

    KernelState & state = states_[state_idx];

    if (!state.kernel)
    {
      std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0);
      audio_processing_.store(false, std::memory_order_release);
      return;
    }

    // Snapshot trigger params once per buffer
    for (auto * p : state.trigger_params)
    {
      p->frame_value.store(
        p->value.exchange(0.0, std::memory_order_acq_rel),
        std::memory_order_relaxed);
    }

    // Single kernel call processes the entire buffer
    state.kernel(
      nullptr,                       // no inputs (all embedded in expressions)
      state.registers.data(),
      state.array_ptrs.data(),
      state.array_sizes.data(),
      state.temps.data(),
      state.sample_rate,
      state.sample_index,
      state.param_ptrs.data(),
      outputBuffer.data(),
      buffer_length_);

    state.sample_index += buffer_length_;

    // Apply fade envelope
    {
      int fi = fade_in_remaining_.load(std::memory_order_relaxed);
      int fo = fade_out_remaining_.load(std::memory_order_relaxed);
      if (fi > 0 || fo != -1)
      {
        for (unsigned int s = 0; s < buffer_length_; ++s)
        {
          if (fi > 0)
          {
            const double t = 1.0 - static_cast<double>(fi) / kFadeSamples_;
            outputBuffer[s] *= t * t * (3.0 - 2.0 * t);
            --fi;
          }
          if (fo != -1)
          {
            if (fo > 0)
            {
              const double t = static_cast<double>(fo) / kFadeSamples_;
              outputBuffer[s] *= t * t * (3.0 - 2.0 * t);
              --fo;
            }
            else
            {
              outputBuffer[s] = 0.0;
            }
          }
        }
        fade_in_remaining_.store(fi, std::memory_order_relaxed);
        fade_out_remaining_.store(fo, std::memory_order_relaxed);
      }
    }

    audio_processing_.store(false, std::memory_order_release);
  }

  std::vector<double> outputBuffer;

  unsigned int getBufferLength() const { return buffer_length_; }

  void begin_fade_in(int samples = 2048)
  {
    fade_out_remaining_.store(-1, std::memory_order_relaxed);
    fade_in_remaining_.store(samples, std::memory_order_relaxed);
  }

  void begin_fade_out(int samples = 2048)
  {
    fade_out_remaining_.store(samples, std::memory_order_release);
  }

  bool is_fade_out_complete() const
  {
    return fade_out_remaining_.load(std::memory_order_acquire) == 0;
  }

private:
  void wait_for_state_available(uint32_t state_index) const
  {
    while (audio_processing_.load(std::memory_order_acquire) &&
           audio_state_index_.load(std::memory_order_acquire) == state_index)
    {
      std::this_thread::yield();
    }
  }

  // Compute scalar register mapping for hot-swap state transfer
  static std::vector<std::pair<uint32_t, uint32_t>> compute_register_mapping(
    const KernelState & old_state,
    const KernelState & new_state)
  {
    std::vector<std::pair<uint32_t, uint32_t>> mapping;
    for (uint32_t new_idx = 0; new_idx < new_state.register_names.size(); ++new_idx)
    {
      const auto & name = new_state.register_names[new_idx];
      if (name.empty()) continue;
      for (uint32_t old_idx = 0; old_idx < old_state.register_names.size(); ++old_idx)
      {
        if (old_state.register_names[old_idx] == name)
        {
          mapping.push_back({old_idx, new_idx});
          break;
        }
      }
    }
    return mapping;
  }

  // Compute array slot mapping for hot-swap state transfer.
  // Only transfers slots where name and size both match (size change = reset to zero).
  static std::vector<std::pair<uint32_t, uint32_t>> compute_array_mapping(
    const KernelState & old_state,
    const KernelState & new_state)
  {
    std::vector<std::pair<uint32_t, uint32_t>> mapping;
    for (uint32_t ni = 0; ni < new_state.array_names.size(); ++ni)
    {
      const auto & name = new_state.array_names[ni];
      if (name.empty()) continue;
      for (uint32_t oi = 0; oi < old_state.array_names.size(); ++oi)
      {
        if (old_state.array_names[oi] == name &&
            oi < old_state.array_storage.size() &&
            ni < new_state.array_storage.size() &&
            old_state.array_storage[oi].size() == new_state.array_storage[ni].size())
        {
          mapping.push_back({oi, ni});
          break;
        }
      }
    }
    return mapping;
  }

  unsigned int buffer_length_;
  std::array<KernelState, 2> states_;
  std::atomic<uint32_t> active_state_{0};
  std::atomic<uint32_t> audio_state_index_{0};
  std::atomic<bool> audio_processing_{false};

  mutable std::mutex build_mutex_;

  static constexpr int kFadeSamples_ = 2048;
  std::atomic<int> fade_in_remaining_{0};
  std::atomic<int> fade_out_remaining_{-1};
};

} // namespace egress_runtime
