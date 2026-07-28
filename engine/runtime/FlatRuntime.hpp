#pragma once

/**
 * FlatRuntime.hpp — publication and execution for emitted kernels.
 *
 * Receives Lean-emitted LLVM (and optionally MSL/coefficient LLVM) plus a
 * tropical_plan_5 metadata manifest, compiles/loads the artifacts, constructs
 * their backing regions, and atomically publishes the inactive KernelState.
 * The plan instruction graph is diagnostic metadata here; C++ does not turn
 * expression trees into executable code.
 */

#include "ControlParam.hpp"
#include "jit/OrcJitEngine.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef TROPICAL_METAL
#include "metal/MetalKernel.hpp"
#endif

namespace tropical_plan5 { struct ParsedPlan5; }
namespace tropical_metal { struct MetalKernel; }

namespace tropical_runtime
{

// Host-contract param write discipline (design/host-param-dispatch.md),
// carried by the plan manifest exactly like slot_names / slot_defaults.
// Runtime copy of ParsedPlan5::ParamDiscipline so this header stays free
// of the parser header.
struct ParamDiscipline
{
  std::string              name;             // base param ("sf.depth"; slot "param:<name>")
  std::string              discipline;       // raw | glide | anchor | velocity
  double                   glide_dur_sec = 0.0;
  std::vector<std::string> companions;       // #v0/#v1/#t0, #phase, tau_base sibling
};

struct ParamDispatchResult
{
  bool        ok = false;
  std::string error;
};

// Explicit ownership for storage that is mutated off the audio thread.
// These atomics deliberately live outside KernelState: KernelState is movable
// during hot-swap, while ownership must remain at a stable address.
enum class StorageOwner : uint8_t { Free, Audio, Writer };
static_assert(std::atomic<StorageOwner>::is_always_lock_free);

// Deterministic concurrency-test seam. Production leaves this null. Tests may
// pause a clock request while odd or the audio callback after it owns a state
// or generation, forcing boundary/publication/reuse interleavings without
// relying on scheduler timing.
struct RuntimeOwnershipTestSeam
{
  std::atomic<bool> pause_after_clock_payload{false};
  std::atomic<bool> clock_payload_stored{false};
  std::atomic<bool> release_clock{false};
  std::atomic<bool> pause_after_state_ownership{false};
  std::atomic<bool> state_owned{false};
  std::atomic<bool> release_state{false};
  std::atomic<bool> writer_waiting_for_state{false};
  std::atomic<bool> pause_after_generation_ownership{false};
  std::atomic<bool> generation_owned{false};
  std::atomic<bool> release_generation{false};
};

struct KernelState
{
  // ── Metal execution backend (TROPICAL_METAL builds) ─────────────────────
  // When set, process() dispatches this instead of the JIT fn-ptr — the
  // kernel runs on the GPU, one thread per sample. Dual-loaded next to the
  // JIT kernel (which keeps serving render_window / the scope), and swapped
  // by the same publish/flip as everything else in this state. shared_ptr
  // over an incomplete type: non-Metal builds never construct one.
  std::shared_ptr<tropical_metal::MetalKernel> metal;

  // ── Execution mode + kernel handles ──────────────────────────────────────
  // Fused mode populates `kernel` (single NumericKernelFn). Microkernel
  // mode populates `microkernels` (preamble, N instance kernels,
  // state_evolution, postamble_mix); `kernel` is nullptr in that case.
  // The audio-thread process() loop branches on `mode`.
  tropical_jit::CompilationMode      mode   = tropical_jit::CompilationMode::Fused;
  tropical_jit::NumericKernelFn      kernel = nullptr;
  tropical_jit::MicrokernelKernels   microkernels{};

  // Stage-0 coefficient kernel (optional). A second single-function module
  // holding the plan's τ-independent instructions; run once at load (before
  // publish) and after every control-plane slot write, materializing
  // coefficient values into `coef:<n>` module slots the audio kernel reads.
  // Same NumericKernelFn signature, invoked with buffer_length = 1. Both
  // kernels keep temps as SSA (never store `%temps`). Control and audio use
  // disjoint slot views, so coefficient materialization never races the
  // in-flight audio kernel.
  //
  // BANKS-AS-DATA (per-array staging): the coefficient kernel also fills shared
  // COEFFICIENT COLUMNS — `array_ptrs` slots whose fills classified s0 — that the
  // audio kernel's in-loop `Index` reads. Those slots (the plan advertises them
  // as `coeff_array_slots`) are GENERATION-BUFFERED together with the entire
  // scalar slot set. A control transaction materializes both into one free
  // generation and release-publishes one generation word. process() captures
  // that generation once per buffer, so ordinary slots, scalar coefficients,
  // and every coefficient column are mutually coherent.
  tropical_jit::NumericKernelFn      coeff_kernel = nullptr;

  // The coefficient-column slots (indices into array_storage/array_ptrs), the
  // three storage generations ([gen][j] backs slot coeff_array_slots[j]), the
  // coefficient kernel's own pointer view (column entries repointed at the
  // write generation per run; non-column entries alias array_storage).
  // slot_generations are immutable once published until the control thread
  // proves a generation is neither published nor captured by audio.
  // audio_slots is private per-state scratch: audio copies the captured slot
  // generation at the boundary, then kernels may write it freely.
  // control_published_gen covers BOTH slot and column generations. atomic_ref
  // is used at access sites so KernelState remains movable.
  std::vector<uint32_t> coeff_array_slots;
  std::array<std::vector<std::vector<int64_t>>, 3> coeff_generations;
  std::vector<int64_t *> coeff_array_ptrs;
  std::array<std::vector<double>, 3> slot_generations;
  std::vector<double> audio_slots;
  uint32_t control_published_gen = 0;

  // Metal upload staging for the coefficient columns: the packed f32 image
  // of the captured generation, rebuilt once per process() (audio thread,
  // no allocation) and copied into the GPU column buffer per dispatch —
  // at encode on the sync path, at enqueue on the pipelined path. Packed
  // in `coeff_array_slots` order, each column at full capacity — the SAME
  // layout EmitMsl bakes as compile-time offsets into the kernel text.
  // Column storage is int64 bit-punned f64 (the JIT loads i64 + bitcast to
  // double — see EmitLlvm.emitIndex), so the pack is bit_cast<double> then
  // the same f64→f32 narrowing slots get. Sized at build; empty when the
  // plan hoists no columns.
  std::vector<float> metal_column_staging;

  // Flat buffers passed to kernel (matches NumericKernelFn signature)
  std::vector<int64_t> registers;
  std::vector<int64_t> temps;
  // Control-only scratch for the coefficient kernel. Even though emitted
  // stage-0 code currently keeps temporaries in SSA, using disjoint backing
  // buffers makes that codegen property unnecessary for thread safety.
  std::vector<int64_t> coeff_registers;
  std::vector<int64_t> coeff_temps;
  std::vector<std::vector<int64_t>> array_storage;
  std::vector<int64_t *> array_ptrs;
  std::vector<uint64_t> array_sizes;
  std::vector<uint64_t> param_ptrs;

  // Array slot names from the manifest (diagnostic/compatibility metadata).
  std::vector<std::string> array_names;

  // Output extraction
  uint32_t output_count = 0;

  // ── Slot model state ─────────────────────────────────────────────────────
  // Inter-module slot array. Control-plane writes via
  // tropical_runtime_set_slot; emitted kernels read/write these slots.
  // A fresh load initializes them from its own manifest. FlatRuntime does not
  // copy slot values between kernels; the session/control host supplies
  // current parameter values.
  std::vector<double>      slots;
  std::vector<std::string> slot_names;

  // Host-contract dispatch table: which write discipline each base param
  // carries. Lives here so a hot-swap replaces it together with the plan
  // it describes (like slot_names); an empty table means every write is raw.
  std::vector<ParamDiscipline> param_disciplines;

  // Discipline lookup by base param name. Returns nullptr when the name is
  // absent from the loaded plan's table (callers treat that as raw). The
  // pointer is into this state — only valid while the state is held (e.g.
  // under FlatRuntime::with_active_state_sync).
  const ParamDiscipline * find_discipline(const std::string & name) const
  {
    for (const auto & d : param_disciplines)
      if (d.name == name) return &d;
    return nullptr;
  }

  double sample_rate = 44100.0;
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
   * OrcJitEngine::compile_ir_text). The manifest's metadata (temp pool
   * size, array/slot names + sizes, sample_rate, slot defaults) drives
   * runtime scratch + slot setup. CF-only: there is no per-sample state,
   * so no state-register metadata or by-name transfer. Always fused. This
   * is the seam for the Lean-emits-IR migration: the same metadata Lean
   * already holds in FlatPlan, plus the IR it will emit.
   */
  bool load_ir(const std::string & ir_text, const std::string & manifest_json);

  /**
   * Load a kernel DUAL: LLVM IR → JIT (always — render_window, the scope,
   * and the goldens stay on the CPU kernel) and MSL → Metal pipeline (the
   * audio path, when built with TROPICAL_METAL; throws otherwise when
   * `msl_source` is non-empty). Same manifest contract as load_ir; the
   * same publish/flip hot-swap covers both artifacts.
   */
  bool load_ir_msl(const std::string & ir_text, const std::string & msl_source,
                   const std::string & manifest_json);

  /**
   * Load a kernel with an optional stage-0 coefficient kernel. `msl_source`
   * may be empty (JIT-only); `coeff_ir` may be empty (no stage-0 split).
   * When present, the coefficient kernel is compiled as its own module,
   * run once against the new state before publish (initial knob values
   * exist — a first buffer reading zero-initialized coefficient slots
   * would be a plain bug, not graceful exclusion), and re-run after every
   * control-plane slot write.
   */
  bool load_ir_staged(const std::string & ir_text, const std::string & msl_source,
                      const std::string & coeff_ir, const std::string & manifest_json);

  /**
   * Control-plane/test-only: reposition the active kernel's sample clock
   * (the `--start` of render verbs — long-τ gates render windows at
   * arbitrary positions). Serialized against hot-swap.
   */
  void set_sample_index(uint64_t idx)
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    sample_index_request_seq_.fetch_add(1, std::memory_order_acq_rel);
    requested_sample_index_.store(idx, std::memory_order_release);
    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire);
        seam && seam->pause_after_clock_payload.load(std::memory_order_acquire))
    {
      seam->clock_payload_stored.store(true, std::memory_order_release);
      while (!seam->release_clock.load(std::memory_order_acquire))
        std::this_thread::yield();
    }
    sample_index_request_seq_.fetch_add(1, std::memory_order_release);
  }

  /**
   * Process one buffer of audio. Called from the audio thread.
   * Fills outputBuffer with buffer_length_ samples.
   */
  void process()
  {
    // Apply a control-plane clock reposition exactly at this buffer boundary.
    // The plain audio_* clock fields are owned solely by this thread.
    const uint64_t request_seq_before =
      sample_index_request_seq_.load(std::memory_order_acquire);
    if ((request_seq_before & 1U) == 0
        && request_seq_before != audio_applied_sample_index_seq_)
    {
      const uint64_t requested =
        requested_sample_index_.load(std::memory_order_acquire);
      const uint64_t request_seq_after =
        sample_index_request_seq_.load(std::memory_order_acquire);
      if (request_seq_before == request_seq_after)
      {
        audio_sample_index_ = requested;
        audio_applied_sample_index_seq_ = request_seq_after;
      }
    }
    const uint64_t start_sample_index = audio_sample_index_;

    auto finish_silence = [&] {
      std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0);
      audio_sample_index_ += buffer_length_;
      published_sample_index_.store(
        audio_sample_index_, std::memory_order_release);
    };

    // Own a state before reading any of its movable storage, then revalidate
    // the publication. A control writer may wait/yield; the audio thread makes
    // only a small fixed number of attempts and fails safely to silence.
    uint32_t state_idx = 0;
    bool state_owned = false;
    for (unsigned int attempt = 0; attempt < kAudioCaptureAttempts_; ++attempt)
    {
      state_idx = active_state_.load(std::memory_order_acquire);
      StorageOwner expected = StorageOwner::Free;
      if (!state_owners_[state_idx].compare_exchange_strong(
            expected, StorageOwner::Audio,
            std::memory_order_acquire, std::memory_order_relaxed))
        continue;
      if (active_state_.load(std::memory_order_acquire) == state_idx)
      {
        state_owned = true;
        break;
      }
      state_owners_[state_idx].store(
        StorageOwner::Free, std::memory_order_release);
    }
    if (!state_owned)
    {
      ownership_failure_count_.fetch_add(1, std::memory_order_relaxed);
      finish_silence();
      return;
    }

    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire);
        seam && seam->pause_after_state_ownership.load(std::memory_order_acquire))
    {
      seam->state_owned.store(true, std::memory_order_release);
      while (!seam->release_state.load(std::memory_order_acquire))
        std::this_thread::yield();
    }

    KernelState & state = states_[state_idx];

    // Own the published slot/coefficient generation before copying or using
    // it. Revalidating after the CAS closes both the initial race and the ABA
    // reuse window: a writer cannot mutate an audio-owned generation.
    uint32_t control_gen = 0;
    bool generation_owned = false;
    for (unsigned int attempt = 0; attempt < kAudioCaptureAttempts_; ++attempt)
    {
      control_gen = std::atomic_ref(state.control_published_gen)
                      .load(std::memory_order_acquire);
      StorageOwner expected = StorageOwner::Free;
      if (!control_generation_owners_[state_idx][control_gen]
             .compare_exchange_strong(
               expected, StorageOwner::Audio,
               std::memory_order_acquire, std::memory_order_relaxed))
        continue;
      if (std::atomic_ref(state.control_published_gen)
            .load(std::memory_order_acquire) == control_gen)
      {
        generation_owned = true;
        break;
      }
      control_generation_owners_[state_idx][control_gen].store(
        StorageOwner::Free, std::memory_order_release);
    }
    if (!generation_owned)
    {
      ownership_failure_count_.fetch_add(1, std::memory_order_relaxed);
      state_owners_[state_idx].store(
        StorageOwner::Free, std::memory_order_release);
      finish_silence();
      return;
    }

    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire);
        seam
        && seam->pause_after_generation_ownership.load(
          std::memory_order_acquire))
    {
      seam->generation_owned.store(true, std::memory_order_release);
      while (!seam->release_generation.load(std::memory_order_acquire))
        std::this_thread::yield();
    }

    std::copy(state.slot_generations[control_gen].begin(),
              state.slot_generations[control_gen].end(),
              state.audio_slots.begin());

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
      audio_sample_index_ += buffer_length_;
      published_sample_index_.store(
        audio_sample_index_, std::memory_order_release);
      control_generation_owners_[state_idx][control_gen].store(
        StorageOwner::Free, std::memory_order_release);
      state_owners_[state_idx].store(
        StorageOwner::Free, std::memory_order_release);
      return;
    }

    // Point the audio view of every coefficient column at the captured
    // generation — one repoint per buffer, so every in-loop Index this buffer
    // executes reads the same generation.
    if (!state.coeff_array_slots.empty())
      for (std::size_t j = 0; j < state.coeff_array_slots.size(); ++j)
        state.array_ptrs[state.coeff_array_slots[j]] =
          state.coeff_generations[control_gen][j].data();

    if (state.mode == tropical_jit::CompilationMode::Fused)
    {
#ifdef TROPICAL_METAL
      if (state.metal)
      {
        // Pack the captured coefficient-column generation for the GPU.
        // The generation was captured and advertised once above, so this pack
        // (and the per-dispatch copy it feeds) inherits the whole-generation
        // guarantee: the GPU reads one
        // consistent generation of columns, no cross-column tear on a
        // live knob move. Value semantics match the JIT's array access
        // exactly: the int64 storage is bit-punned f64 (load i64 +
        // bitcast to double), then narrowed f64→f32 like the slot
        // snapshot.
        if (!state.coeff_array_slots.empty())
        {
          std::size_t off = 0;
          for (std::size_t j = 0; j < state.coeff_array_slots.size(); ++j)
          {
            const auto & col = state.coeff_generations[control_gen][j];
            for (std::size_t e = 0; e < col.size(); ++e)
              state.metal_column_staging[off + e] =
                static_cast<float>(std::bit_cast<double>(col[e]));
            off += col.size();
          }
        }
        // GPU path: one thread per sample, synchronous per-block dispatch
        // (v1). On a dispatch error, fall through to silence — the JIT
        // kernel stays authoritative for render_window either way.
        if (!tropical_metal::process_block(
              *state.metal,
              state.audio_slots.data(),
              static_cast<uint32_t>(state.audio_slots.size()),
              state.metal_column_staging.data(),
              static_cast<uint32_t>(state.metal_column_staging.size()),
              state.sample_rate, start_sample_index,
              outputBuffer.data(), buffer_length_))
          std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0);
      }
      else
#endif
      // Single kernel call processes the entire buffer.
      state.kernel(
        nullptr,                       // no inputs (all embedded in expressions)
        state.registers.data(),
        state.array_ptrs.data(),
        state.array_sizes.data(),
        state.temps.data(),
        state.sample_rate,
        start_sample_index,
        state.param_ptrs.data(),
        outputBuffer.data(),
        buffer_length_,
        state.audio_slots.data());   // private whole-buffer slot snapshot
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
      double *       slots       = state.audio_slots.data();
      double *       out         = outputBuffer.data();
      const uint64_t start_idx   = start_sample_index;
      for (unsigned int i = 0; i < buffer_length_; ++i)
      {
        const uint64_t s = start_idx + i;
        for (auto fn : mk.instances)
          fn(regs, arrays, arr_sizes, temps, sr, s, params, slots);
        mk.postamble_mix(regs, arrays, arr_sizes, temps, sr, s, params, slots, out, i);
      }
    }

    audio_sample_index_ += buffer_length_;

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

    // Publish the completed boundary only after all output processing for this
    // buffer (including fade) is complete.
    published_sample_index_.store(audio_sample_index_, std::memory_order_release);
    control_generation_owners_[state_idx][control_gen].store(
      StorageOwner::Free, std::memory_order_release);
    state_owners_[state_idx].store(
      StorageOwner::Free, std::memory_order_release);
  }

  std::vector<double> outputBuffer;

  unsigned int getBufferLength() const { return buffer_length_; }

  uint32_t metal_pipeline_depth() const
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
#ifdef TROPICAL_METAL
    const uint32_t idx = active_state_.load(std::memory_order_acquire);
    const auto & metal = states_[idx].metal;
    return metal ? tropical_metal::pipeline_depth(*metal) : 0;
#else
    return 0;
#endif
  }

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
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t idx = active_state_.load(std::memory_order_acquire);
    const KernelState & state = states_[idx];
    for (uint32_t i = 0; i < state.slot_names.size(); ++i)
      if (state.slot_names[i] == name) return i;
    return UINT32_MAX;
  }

  // Write a slot value from a control thread. All control writers serialize
  // with hot-swap, then publish one coherent snapshot for the next buffer.
  void set_slot(uint32_t idx, double value)
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    KernelState & state = states_[state_idx];
    if (idx < state.slots.size())
    {
      state.slots[idx] = value;
      publish_control_snapshot(state, state_idx);
    }
  }

  // Read a slot value. Useful for tests and introspection. Returns 0.0
  // if the index is out of range.
  double get_slot(uint32_t idx) const
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    const KernelState & state = states_[state_idx];
    if (idx >= state.slots.size()) return 0.0;
    return state.slots[idx];
  }

  // ── Control-plane data path (socket endpoint) ──────────────────────────────
  // Resolve a slot by name and write it atomically against the live kernel,
  // serialized with hot-swap (publish_state) via build_mutex_. This is safe to
  // call from a control thread running concurrently with load_ir: the lock
  // excludes the inactive-state rebuild,
  // and resolving + writing under one lock means a recompile can never land a
  // value in a stale slot index. build_mutex_ is held by publish_state only
  // for the cheap state publication + flip (microseconds) — never the LLVM
  // compile — so this contends only during the swap itself. Returns false if
  // no slot of that name exists in the active plan.
  bool set_slot_by_name_sync(const std::string & name, double value)
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    KernelState & state = states_[state_idx];
    for (uint32_t i = 0; i < state.slot_names.size(); ++i)
      if (state.slot_names[i] == name)
      {
        state.slots[i] = value;
        publish_control_snapshot(state, state_idx);
        return true;
      }
    return false;
  }

  // Run `fn` against the active KernelState under build_mutex_ — the
  // generalization of set_slot_by_name_sync for multi-slot writes. The
  // discipline dispatches (glide/anchor/velocity re-anchorings) must
  // resolve, read, and write several slots against ONE consistent plan;
  // without the lock a concurrent hot-swap could flip the active state
  // between resolve and write and land values at stale indices. Same
  // contention profile as set_slot_by_name_sync: publish_state holds the
  // lock only for the cheap sample-index publication + flip, never the LLVM
  // compile.
  // The coefficient re-run after `fn` covers any slot writes it made
  // (discipline dispatches write several slots per param event); it is
  // idempotent, so read-only callers pay one cheap one-sample kernel call.
  template <typename Fn>
  auto with_active_state_sync(Fn && fn)
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    KernelState & state = states_[state_idx];
    if constexpr (std::is_void_v<std::invoke_result_t<Fn &, KernelState &>>)
    {
      fn(state);
      publish_control_snapshot(state, state_idx);
    }
    else
    {
      auto result = fn(state);
      publish_control_snapshot(state, state_idx);
      return result;
    }
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
    // Coefficient columns live in the same immutable published generation as
    // the control slot snapshot.
    if (!active.coeff_array_slots.empty())
    {
      const uint32_t gen = std::atomic_ref(active.control_published_gen)
                             .load(std::memory_order_acquire);
      for (std::size_t j = 0; j < active.coeff_array_slots.size(); ++j)
        arrays[active.coeff_array_slots[j]] = active.coeff_generations[gen][j];
    }
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
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    return static_cast<uint32_t>(states_[state_idx].slots.size());
  }

  // The current sample index of the active kernel — the count of samples the
  // audio thread has processed (advances one buffer per process() call, paced
  // by the device clock when the DAC is running). The master "now" a slave
  // consumer renders a window ending at; static when audio isn't running.
  // Published at each completed audio buffer boundary.
  uint64_t current_sample_index() const
  {
    return published_sample_index_.load(std::memory_order_acquire);
  }

  // Number of callbacks that emitted silence because bounded ownership
  // acquisition could not obtain a coherent state/generation.
  uint64_t ownership_failure_count() const
  {
    return ownership_failure_count_.load(std::memory_order_acquire);
  }

  // Test-only: install/remove the deterministic ownership pause seam.
  void set_ownership_test_seam(RuntimeOwnershipTestSeam * seam)
  {
    ownership_test_seam_.store(seam, std::memory_order_release);
  }

  // The active kernel's sample rate — the same value the kernel reads for
  // `sampleRate()` (opRate). The control plane needs it to reproduce the
  // phasor's `inc = floor(freq*2^32/SR)` when phase-anchoring a freq change.
  double sample_rate() const
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    return states_[state_idx].sample_rate;
  }

  // Exact production host-param dispatch, shared by the socket and
  // qualification harness. Companion reads/writes publish atomically as one
  // control generation at the next audio buffer.
  ParamDispatchResult dispatch_param_sync(const std::string & name, double value);

private:
  // build_kernel_state maps a parsed plan's *metadata* (everything except
  // the kernel handle) into a fresh KernelState; publish_state carries the
  // sample coordinate and performs the atomic double-buffer flip.
  // load_ir fills the kernel handle (via compile_ir_text) between them.
  KernelState build_kernel_state(const tropical_plan5::ParsedPlan5 & parsed);
  bool publish_state(KernelState && new_state);

  // Materialize one complete control transaction into a FREE generation.
  // Scalar coefficient slots, coefficient columns, and ordinary slots become
  // visible together through one release-published generation word. Called
  // under build_mutex_, except while constructing an unpublished local state.
  void publish_control_snapshot(
    KernelState & state, uint32_t state_index = UINT32_MAX)
  {
    const uint32_t pub = std::atomic_ref(state.control_published_gen)
                           .load(std::memory_order_acquire);
    uint32_t target = 0;
    bool target_owned = false;
    if (state_index == UINT32_MAX)
    {
      // A newly built local state is not visible and needs no atomic owner.
      while (target == pub) ++target;
    }
    else
    {
      // Control writers are serialized by build_mutex_. They may wait/yield
      // off the real-time thread until a non-published generation is free.
      for (;;)
      {
        const uint32_t published =
          std::atomic_ref(state.control_published_gen)
            .load(std::memory_order_acquire);
        for (target = 0; target < control_generation_owners_[state_index].size();
             ++target)
        {
          if (target == published) continue;
          StorageOwner expected = StorageOwner::Free;
          if (control_generation_owners_[state_index][target]
                .compare_exchange_strong(
                  expected, StorageOwner::Writer,
                  std::memory_order_acquire, std::memory_order_relaxed))
          {
            target_owned = true;
            break;
          }
        }
        if (target_owned) break;
        std::this_thread::yield();
      }
    }

    int64_t ** arrays = state.array_ptrs.data();
    if (!state.coeff_array_slots.empty())
    {
      for (std::size_t j = 0; j < state.coeff_array_slots.size(); ++j)
        state.coeff_array_ptrs[state.coeff_array_slots[j]] =
          state.coeff_generations[target][j].data();
      arrays = state.coeff_array_ptrs.data();
    }
    if (state.coeff_kernel)
    {
      double scratch_out = 0.0;
      state.coeff_kernel(
        nullptr,
        state.coeff_registers.data(),
        arrays,
        state.array_sizes.data(),
        state.coeff_temps.data(),
        state.sample_rate,
        0,                        // tick-free coefficient stage
        state.param_ptrs.data(),
        &scratch_out,
        1,
        state.slots.data());
    }
    std::copy(state.slots.begin(), state.slots.end(),
              state.slot_generations[target].begin());
    if (target_owned)
      control_generation_owners_[state_index][target].store(
        StorageOwner::Free, std::memory_order_release);
    std::atomic_ref(state.control_published_gen)
      .store(target, std::memory_order_release);
  }

  unsigned int buffer_length_;
  std::array<KernelState, 2> states_;
  std::atomic<uint32_t> active_state_{0};
  std::array<std::atomic<StorageOwner>, 2> state_owners_{};
  std::array<std::array<std::atomic<StorageOwner>, 3>, 2>
    control_generation_owners_{};
  std::atomic<uint64_t> recompile_version_{0};
  std::atomic<uint64_t> ownership_failure_count_{0};
  std::atomic<RuntimeOwnershipTestSeam *> ownership_test_seam_{nullptr};

  // Buffer-boundary clock handoff. Only the audio thread touches plain
  // audio_* fields. Control publishes an odd/even seqlock around the atomic
  // value; audio never spins and simply defers an in-progress request.
  std::atomic<uint64_t> requested_sample_index_{0};
  std::atomic<uint64_t> sample_index_request_seq_{0};
  uint64_t audio_applied_sample_index_seq_ = 0;
  uint64_t audio_sample_index_ = 0;
  std::atomic<uint64_t> published_sample_index_{0};

  mutable std::mutex build_mutex_;

  static constexpr unsigned int kAudioCaptureAttempts_ = 4;
  static constexpr int kFadeSamples_ = 2048;
  std::atomic<int> fade_in_remaining_{0};
  std::atomic<int> fade_out_remaining_{-1};
};

} // namespace tropical_runtime
