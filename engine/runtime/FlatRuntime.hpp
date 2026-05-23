#pragma once

/**
 * FlatRuntime.hpp — Minimal plan executor for JIT-compiled audio.
 *
 * Receives a flat set of expression trees (outputs + registers) via JSON plan,
 * JIT-compiles them into a single native kernel, and runs it per sample.
 * No module boundaries, no graph, no orchestration.
 */

#include "ControlParam.hpp"
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

namespace tropical_runtime
{

struct KernelState
{
  // ── Execution mode + kernel handles ──────────────────────────────────────
  // Fused mode populates `kernel` (single NumericKernelFn). Microkernel
  // mode populates `microkernels` (preamble, N instance kernels,
  // state_evolution, postamble_mix); `kernel` is nullptr in that case.
  // The audio-thread process() loop branches on `mode`.
  tropical_jit::CompilationMode      mode   = tropical_jit::CompilationMode::Fused;
  tropical_jit::NumericKernelFn      kernel = nullptr;
  tropical_jit::MicrokernelKernels   microkernels{};

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

  // ── Slot model state (M5) ────────────────────────────────────────────────
  // Inter-module slot array. M5 uses it for control-plane writes only
  // (via tropical_runtime_set_slot); M6 introduces JIT codegen that
  // reads from it via 'slot' operands. Slots survive hot-swap by name
  // so control values set from outside persist; this transfer becomes
  // unnecessary in M8 once the kernel rewrites slots every sample.
  std::vector<double>      slots;
  std::vector<std::string> slot_names;

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
   * Plan schema (tropical_plan_4):
   * {
   *   "schema": "tropical_plan_4",
   *   "config": { "sampleRate": 44100 },
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

    // No active kernel — emit silence (covers both fused and microkernel
    // modes; the "no kernel" predicate is mode-specific).
    const bool fused_ready = state.mode == tropical_jit::CompilationMode::Fused
                          && state.kernel != nullptr;
    const bool mk_ready    = state.mode == tropical_jit::CompilationMode::Microkernel
                          && state.microkernels.preamble != nullptr;
    if (!fused_ready && !mk_ready)
    {
      std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0);
      audio_processing_.store(false, std::memory_order_release);
      return;
    }

    if (state.mode == tropical_jit::CompilationMode::Fused)
    {
      // Single kernel call processes the entire buffer.
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
        buffer_length_,
        state.slots.data());          // M6: shared inter-module slot array
    }
    else
    {
      // Microkernel: outer sample loop in C++, dispatching the N+3
      // per-sample functions in scheduler order. This is the
      // architectural shape the scoped-lifetime / per-voice
      // microkernel roadmap needs; the spike measures whether the
      // dispatch cost is acceptable for realtime synthesis.
      const auto & mk = state.microkernels;
      int64_t *      regs        = state.registers.data();
      int64_t **     arrays      = state.array_ptrs.data();
      const uint64_t * arr_sizes = state.array_sizes.data();
      int64_t *      temps       = state.temps.data();
      const double   sr          = state.sample_rate;
      const uint64_t * params    = state.param_ptrs.data();
      double *       slots       = state.slots.data();
      double *       out         = outputBuffer.data();
      const uint64_t start_idx   = state.sample_index;
      for (unsigned int i = 0; i < buffer_length_; ++i)
      {
        const uint64_t s = start_idx + i;
        mk.preamble(regs, arrays, arr_sizes, temps, sr, s, params, slots);
        for (auto fn : mk.instances)
          fn(regs, arrays, arr_sizes, temps, sr, s, params, slots);
        mk.state_evolution(regs, arrays, arr_sizes, temps, sr, s, params, slots);
        mk.postamble_mix(regs, arrays, arr_sizes, temps, sr, s, params, slots, out, i);
      }
    }

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

  // ── Slot model accessors (M5) ──────────────────────────────────────────────
  // Lookup a slot index by name — returns the index or UINT32_MAX if no
  // slot with that name exists in the active plan. Wired by the C API
  // helper so external controllers can resolve a name once and write
  // by integer index thereafter.
  uint32_t slot_index(const std::string & name) const
  {
    const uint32_t idx = active_state_.load(std::memory_order_acquire);
    const KernelState & state = states_[idx];
    for (uint32_t i = 0; i < state.slot_names.size(); ++i)
      if (state.slot_names[i] == name) return i;
    return UINT32_MAX;
  }

  // Write a slot value. Safe to call from any thread; the audio thread
  // reads slots via the kernel (M6+) and tolerates one-buffer races on
  // double writes. Out-of-range indices are no-ops.
  void set_slot(uint32_t idx, double value)
  {
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    KernelState & state = states_[state_idx];
    if (idx < state.slots.size()) state.slots[idx] = value;
  }

  // Read a slot value. Useful for tests and introspection. Returns 0.0
  // if the index is out of range.
  double get_slot(uint32_t idx) const
  {
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    const KernelState & state = states_[state_idx];
    if (idx >= state.slots.size()) return 0.0;
    return state.slots[idx];
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

  // M5: name-based slot transfer on hot-swap. Control-plane writes via
  // tropical_runtime_set_slot must persist across recompiles even though
  // the slot's integer index may change in the new plan. Drops in M8
  // once the kernel rewrites slots every sample.
  static std::vector<std::pair<uint32_t, uint32_t>> compute_slot_mapping(
    const KernelState & old_state,
    const KernelState & new_state)
  {
    std::vector<std::pair<uint32_t, uint32_t>> mapping;
    for (uint32_t ni = 0; ni < new_state.slot_names.size(); ++ni)
    {
      const auto & name = new_state.slot_names[ni];
      if (name.empty()) continue;
      for (uint32_t oi = 0; oi < old_state.slot_names.size(); ++oi)
      {
        if (old_state.slot_names[oi] == name)
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

} // namespace tropical_runtime
