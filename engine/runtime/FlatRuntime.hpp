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

namespace tropical_plan5 { struct ParsedPlan5; }

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

  // ── Slot model state ─────────────────────────────────────────────────────
  // Inter-module slot array. Control-plane writes via
  // tropical_runtime_set_slot; JIT codegen reads via 'slot' operands and
  // writes via WriteSlot instructions (parent→child input wiring,
  // MCP wire auto-delay, etc.). Slots survive hot-swap by name so
  // control values set from outside persist across kernel rebuilds.
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
   * Load a kernel from textual LLVM IR plus a metadata manifest.
   *
   * `manifest_json` is a tropical_plan_5 JSON whose instruction graph is
   * ignored — codegen comes from `ir_text` (JIT-compiled via
   * OrcJitEngine::compile_ir_text). The manifest's metadata (state_init,
   * register/array/slot names + types + sizes, sample_rate, slot
   * defaults) drives runtime state setup exactly as load_plan does, so
   * hot-swap state transfer and slot wiring behave identically. Always
   * fused. This is the seam for the Lean-emits-IR migration: the same
   * metadata Lean already holds in FlatPlan, plus the IR it will emit.
   */
  bool load_ir(const std::string & ir_text, const std::string & manifest_json);

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

    // No active kernel — emit silence (covers fused, microkernel, and
    // microkernel-deep modes; the "no kernel" predicate is mode-
    // specific).
    const bool fused_ready = state.mode == tropical_jit::CompilationMode::Fused
                          && state.kernel != nullptr;
    const bool mk_ready    =
      (state.mode == tropical_jit::CompilationMode::Microkernel
       || state.mode == tropical_jit::CompilationMode::MicrokernelDeep)
      && state.microkernels.postamble_mix != nullptr;
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
      // Microkernel: outer sample loop in C++, dispatching the N+1
      // per-sample functions (N instances, then the sink mix). This is
      // the architectural shape the scoped-lifetime / per-voice
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
        for (auto fn : mk.instances)
          fn(regs, arrays, arr_sizes, temps, sr, s, params, slots);
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

  // ── Control-plane data path (socket endpoint) ──────────────────────────────
  // Resolve a slot by name and write it atomically against the live kernel,
  // serialized with hot-swap (publish_state) via build_mutex_. Unlike the
  // lock-free set_slot, this is safe to call from a control thread running
  // concurrently with load_ir: the lock excludes the inactive-state rebuild,
  // and resolving + writing under one lock means a recompile can never land a
  // value in a stale slot index. build_mutex_ is held by publish_state only
  // for the cheap by-name transfer + flip (microseconds) — never the LLVM
  // compile — so this contends only during the swap itself. Returns false if
  // no slot of that name exists in the active plan.
  bool set_slot_by_name_sync(const std::string & name, double value)
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    KernelState & state = states_[state_idx];
    for (uint32_t i = 0; i < state.slot_names.size(); ++i)
      if (state.slot_names[i] == name) { state.slots[i] = value; return true; }
    return false;
  }

  // ── Random-access render (scope / slave consumers) ─────────────────────────
  // Render `count` samples starting at an arbitrary sample index, snapshotting
  // the requested slots per sample. For a STATELESS (register-free) patch this
  // is exact and safe to run concurrently with the audio thread: it renders
  // into private scratch temps/slots so it never disturbs the live state, and
  // holds build_mutex_ so a hot-swap can't rebuild the active state mid-render.
  // `out` is slot-major: out[k*count + i] = value of slot_ids[k] at sample
  // start_index+i. Fused mode only; returns false otherwise or on a bad slot id.
  bool render_window(uint64_t start_index, uint32_t count,
                     const uint32_t * slot_ids, uint32_t n_slots,
                     double * out)
  {
    if (!out || (n_slots > 0 && !slot_ids)) return false;
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    KernelState & active = states_[state_idx];
    if (active.mode != tropical_jit::CompilationMode::Fused || active.kernel == nullptr)
      return false;
    for (uint32_t k = 0; k < n_slots; ++k)
      if (slot_ids[k] >= active.slots.size()) return false;

    // Private scratch, isolated from the audio thread. Holding build_mutex_
    // excludes the inactive-state rebuild, so these copies can't tear
    // structurally (no concurrent resize); a register-free patch never writes
    // registers/arrays, so the audio thread and this render don't fight.
    std::vector<int64_t>              registers = active.registers;
    std::vector<int64_t>              temps     = active.temps;
    std::vector<double>               slots     = active.slots;
    std::vector<std::vector<int64_t>> arrays    = active.array_storage;
    std::vector<int64_t *>            array_ptrs(arrays.size());
    for (size_t a = 0; a < arrays.size(); ++a) array_ptrs[a] = arrays[a].data();

    double scratch_out = 0.0;
    for (uint32_t i = 0; i < count; ++i)
    {
      active.kernel(
        nullptr,
        registers.data(),
        array_ptrs.data(),
        active.array_sizes.data(),
        temps.data(),
        active.sample_rate,
        start_index + i,
        active.param_ptrs.data(),
        &scratch_out,
        1,
        slots.data());
      for (uint32_t k = 0; k < n_slots; ++k)
        out[static_cast<size_t>(k) * count + i] = slots[slot_ids[k]];
    }
    return true;
  }

  // Monotonic counter bumped on every successful hot-swap (publish_state).
  // Telemetry clients poll it to detect that the session topology changed.
  uint64_t recompile_version() const
  {
    return recompile_version_.load(std::memory_order_acquire);
  }

  // Slot count of the currently-active kernel (telemetry).
  uint32_t slot_count() const
  {
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    return static_cast<uint32_t>(states_[state_idx].slots.size());
  }

  // The current sample index of the active kernel — the count of samples the
  // audio thread has processed (advances one buffer per process() call, paced
  // by the device clock when the DAC is running). The master "now" a slave
  // consumer renders a window ending at; static when audio isn't running.
  // Benign stale-by-a-buffer race on read.
  uint64_t current_sample_index() const
  {
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    return states_[state_idx].sample_index;
  }

private:
  // build_kernel_state maps a parsed plan's *metadata* (everything except
  // the kernel handle) into a fresh KernelState; publish_state runs the
  // by-name hot-swap state transfer and the atomic double-buffer flip.
  // load_ir fills the kernel handle (via compile_ir_text) between them.
  KernelState build_kernel_state(const tropical_plan5::ParsedPlan5 & parsed);
  bool publish_state(KernelState && new_state);

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
  std::atomic<uint64_t> recompile_version_{0};

  mutable std::mutex build_mutex_;

  static constexpr int kFadeSamples_ = 2048;
  std::atomic<int> fade_in_remaining_{0};
  std::atomic<int> fade_out_remaining_{-1};
};

} // namespace tropical_runtime
