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

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
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
  // kernels keep temps as SSA (never store `%temps`), so sharing the scratch
  // buffers with a concurrently-running audio kernel is read-only-safe;
  // coefficient slot writes are the same tolerated one-buffer race class as
  // set_slot itself.
  //
  // BANKS-AS-DATA (per-array staging): the coefficient kernel also fills shared
  // COEFFICIENT COLUMNS — `array_ptrs` slots whose fills classified s0 — that the
  // audio kernel's in-loop `Index` reads. Those slots (the plan advertises them
  // as `coeff_array_slots`) are GENERATION-BUFFERED: three storage generations
  // per column, one published-generation word. run_coeff fills a generation that
  // is neither published nor captured by the in-flight audio buffer, then
  // publishes it with one atomic store; process() captures the published
  // generation ONCE per buffer (before raising audio_processing_, so a
  // concurrent run_coeff can never pick the captured generation as its write
  // target) and repoints `array_ptrs` for the whole buffer. The audio thread
  // therefore always reads one whole, consistent generation of columns — no
  // cross-column tear on a live knob move, no waiting on either thread. Three
  // generations (not two) because a knob stream can run the coefficient kernel
  // twice within one audio buffer: with two buffers the second run would have to
  // rewrite the generation the in-flight buffer is still reading. Scalar
  // coefficient slots stay in the tolerated one-buffer race class of set_slot.
  tropical_jit::NumericKernelFn      coeff_kernel = nullptr;

  // The coefficient-column slots (indices into array_storage/array_ptrs), the
  // three storage generations ([gen][j] backs slot coeff_array_slots[j]), the
  // coefficient kernel's own pointer view (column entries repointed at the
  // write generation per run; non-column entries alias array_storage), and the
  // published-generation word (accessed via std::atomic_ref — kept a plain
  // member so KernelState stays movable; every concurrent access site holds
  // the publish/audio protocol described above).
  std::vector<uint32_t> coeff_array_slots;
  std::array<std::vector<std::vector<int64_t>>, 3> coeff_generations;
  std::vector<int64_t *> coeff_array_ptrs;
  uint32_t coeff_published_gen = 0;

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
    states_[active_state_.load(std::memory_order_acquire)].sample_index = idx;
  }

  /**
   * Process one buffer of audio. Called from the audio thread.
   * Fills outputBuffer with buffer_length_ samples.
   */
  void process()
  {

    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    KernelState & state = states_[state_idx];

    // Capture ONE coefficient-column generation for the whole buffer (banks-as-
    // data). Advertised BEFORE audio_processing_ goes true: a concurrent
    // run_coeff that observes audio_processing_ == true (acquire) is then
    // guaranteed to see this capture and will never pick it as a write target;
    // one that observes false avoids the published generation, which is what
    // this buffer read. Either way the columns this buffer indexes are one
    // whole generation.
    uint32_t coeff_gen = 0;
    if (!state.coeff_array_slots.empty())
    {
      coeff_gen = std::atomic_ref(state.coeff_published_gen)
                    .load(std::memory_order_acquire);
      audio_coeff_gen_.store(coeff_gen, std::memory_order_relaxed);
    }

    audio_state_index_.store(state_idx, std::memory_order_release);
    audio_processing_.store(true, std::memory_order_release);

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

    // Point the audio view of every coefficient column at the captured
    // generation — one repoint per buffer, so every in-loop Index this buffer
    // executes reads the same generation.
    if (!state.coeff_array_slots.empty())
      for (std::size_t j = 0; j < state.coeff_array_slots.size(); ++j)
        state.array_ptrs[state.coeff_array_slots[j]] =
          state.coeff_generations[coeff_gen][j].data();

    if (state.mode == tropical_jit::CompilationMode::Fused)
    {
#ifdef TROPICAL_METAL
      if (state.metal)
      {
        // Pack the captured coefficient-column generation for the GPU.
        // The generation was captured ONCE above, BEFORE audio_processing_
        // went true — so this pack (and the per-dispatch copy it feeds)
        // inherits the whole-generation guarantee: the GPU reads one
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
            const auto & col = state.coeff_generations[coeff_gen][j];
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
              state.slots.data(), static_cast<uint32_t>(state.slots.size()),
              state.metal_column_staging.data(),
              static_cast<uint32_t>(state.metal_column_staging.size()),
              state.sample_rate, state.sample_index,
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
    if (idx < state.slots.size())
    {
      state.slots[idx] = value;
      run_coeff(state);
    }
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
        run_coeff(state);
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
      run_coeff(state);
    }
    else
    {
      auto result = fn(state);
      run_coeff(state);
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
    // Coefficient columns live in the generation store, not array_storage —
    // copy the published generation in. (A run_coeff racing this copy via the
    // lock-free set_slot can retarget mid-copy; that transient is the same
    // control-plane class as a scope frame during a knob move, and strictly
    // narrower than the pre-generation tear.)
    if (!active.coeff_array_slots.empty())
    {
      const uint32_t gen = std::atomic_ref(active.coeff_published_gen)
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

  // The active kernel's sample rate — the same value the kernel reads for
  // `sampleRate()` (opRate). The control plane needs it to reproduce the
  // phasor's `inc = floor(freq*2^32/SR)` when phase-anchoring a freq change.
  double sample_rate() const
  {
    const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
    return states_[state_idx].sample_rate;
  }

private:
  // build_kernel_state maps a parsed plan's *metadata* (everything except
  // the kernel handle) into a fresh KernelState; publish_state carries the
  // sample coordinate and performs the atomic double-buffer flip.
  // load_ir fills the kernel handle (via compile_ir_text) between them.
  KernelState build_kernel_state(const tropical_plan5::ParsedPlan5 & parsed);
  bool publish_state(KernelState && new_state);

  // Evaluate the stage-0 coefficient kernel once (one "sample") against a
  // state's slots. No-op when the plan carried no coefficient kernel.
  // Coefficient columns are written into a FREE generation (neither published
  // nor captured by the in-flight audio buffer — with three generations one
  // always exists) via the coefficient kernel's own pointer view, then
  // published with one atomic store. The audio thread picks the new
  // generation up at its next buffer boundary; the control thread never
  // waits. Non-member state (audio_coeff_gen_) is why this isn't static.
  void run_coeff(KernelState & state)
  {
    if (!state.coeff_kernel) return;
    int64_t ** arrays = state.array_ptrs.data();
    uint32_t target = UINT32_MAX;
    if (!state.coeff_array_slots.empty())
    {
      const uint32_t pub = std::atomic_ref(state.coeff_published_gen)
                             .load(std::memory_order_acquire);
      uint32_t in_use = pub;
      if (audio_processing_.load(std::memory_order_acquire))
        in_use = audio_coeff_gen_.load(std::memory_order_acquire);
      target = 0;
      while (target == pub || target == in_use) ++target;
      for (std::size_t j = 0; j < state.coeff_array_slots.size(); ++j)
        state.coeff_array_ptrs[state.coeff_array_slots[j]] =
          state.coeff_generations[target][j].data();
      arrays = state.coeff_array_ptrs.data();
    }
    double scratch_out = 0.0;
    state.coeff_kernel(
      nullptr,
      state.registers.data(),
      arrays,
      state.array_sizes.data(),
      state.temps.data(),
      state.sample_rate,
      0,                          // start_sample_index — tick-free by construction
      state.param_ptrs.data(),
      &scratch_out,
      1,
      state.slots.data());
    if (target != UINT32_MAX)
      std::atomic_ref(state.coeff_published_gen)
        .store(target, std::memory_order_release);
  }

  void wait_for_state_available(uint32_t state_index) const
  {
    while (audio_processing_.load(std::memory_order_acquire) &&
           audio_state_index_.load(std::memory_order_acquire) == state_index)
    {
      std::this_thread::yield();
    }
  }

  unsigned int buffer_length_;
  std::array<KernelState, 2> states_;
  std::atomic<uint32_t> active_state_{0};
  std::atomic<uint32_t> audio_state_index_{0};
  std::atomic<bool> audio_processing_{false};
  // The coefficient-column generation captured by the in-flight audio buffer
  // (valid while audio_processing_; stored before it goes true, so an
  // acquire-load of audio_processing_ == true guarantees visibility).
  std::atomic<uint32_t> audio_coeff_gen_{0};
  std::atomic<uint64_t> recompile_version_{0};

  mutable std::mutex build_mutex_;

  static constexpr int kFadeSamples_ = 2048;
  std::atomic<int> fade_in_remaining_{0};
  std::atomic<int> fade_out_remaining_{-1};
};

} // namespace tropical_runtime
