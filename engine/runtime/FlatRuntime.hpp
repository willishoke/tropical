#pragma once

/**
 * FlatRuntime.hpp — publication and execution for emitted kernels.
 *
 * Receives Lean-emitted LLVM (and optionally MSL/coefficient LLVM) plus a
 * tropical_plan_6 metadata manifest, compiles/loads the artifacts, constructs
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
#include "metal/MetalRenderWorker.hpp"
#include "runtime/EpochTileQueue.hpp"
#endif

namespace tropical_plan6 { struct ParsedPlan6; struct ParsedKernel1; }
namespace tropical_metal {
struct MetalKernel;
}

namespace tropical_runtime
{

// Host-contract param write discipline (design/host-param-dispatch.md),
// carried by the plan manifest exactly like slot_names / slot_defaults.
// Runtime copy of ParsedPlan6::ParamDiscipline so this header stays free
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
  std::string discipline;
  // Last completed callback boundary visible when the control transaction
  // began. Diagnostic only: discipline math is stamped at the first output
  // sample that can actually contain the published generation.
  uint64_t    observed_sample_index = 0;
  uint64_t    effective_sample_index = 0;
  // The loaded manifest has no slot for the name: a client error, which the
  // engine reports as its unknown_param envelope rather than a host failure.
  bool        unknown_param = false;
};

// Exact graph generation created by one successful publication. Returning the
// pair from inside the publication critical section avoids a compile caller
// racing a subsequent control-only publication while sampling two counters.
struct PublishedGeneration
{
  uint64_t program_version = 0;
  uint64_t control_version = 0;
};

// Random-access observation retains an explicit cancellation outcome for
// latest-only/superseded callers. Snapshot rendering itself does not preempt;
// the status remains part of the wire contract so callers can discard partial
// work if a future observation scheduler adds cooperative cancellation.
enum class RenderWindowStatus : uint8_t
{
  Rendered,
  Preempted,
  InvalidArgument,
  Unsupported,
};

struct ObservationRenderResult
{
  RenderWindowStatus status = RenderWindowStatus::InvalidArgument;
  uint64_t program_version = 0;
  uint64_t control_version = 0;
  uint64_t effective_sample_index = 0;
  std::string unknown_slot;
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
  std::atomic<bool> pause_before_boundary_capture{false};
  std::atomic<bool> boundary_capture_pending{false};
  std::atomic<bool> release_boundary_capture{false};
  std::atomic<bool> pause_after_dispatch_materialization{false};
  std::atomic<bool> dispatch_materialized{false};
  std::atomic<bool> release_dispatch{false};
  std::atomic<bool> pause_after_observation_coordinate{false};
  std::atomic<bool> observation_coordinate_completed{false};
  std::atomic<bool> release_observation_coordinate{false};
};

// Immutable read-side program/control images for every random-access
// observer. A slow reader retains one heap image while audio and control keep
// publishing through their bounded ownership protocol; no pointer below
// aliases either movable KernelState slot.
struct ObservationProgramImage
{
  uint64_t program_version = 0;
  tropical_jit::KernelArgvFn kernel = nullptr;
  uint32_t frames = 0;
  uint32_t output_channel_count = 1;
  std::vector<uint32_t> input_slots;        // param input j → slot index
  std::vector<std::string> slot_names;
  std::vector<double> initial_slots;
  std::unordered_map<std::string, uint32_t> slot_lookup;
  std::vector<std::string> channel_names;
  std::unordered_map<std::string, uint32_t> channel_lookup;
};

struct ObservationControlImage
{
  uint64_t control_version = 0;
  uint64_t effective_sample_index = 0;
  std::vector<double> slots;
};

struct ObservationSnapshot
{
  std::shared_ptr<const ObservationProgramImage> program;
  std::shared_ptr<const ObservationControlImage> controls;
};

// Mutable scratch is serialized only with other observers. Keeping it outside
// the live state makes control publication and hot-swap independent of a slow
// room/scope/telemetry consumer, while retaining allocation-free steady-state
// frames.
struct ObservationRenderWorkspace
{
  uint64_t program_version = 0;
  std::vector<int64_t> tau;
  std::vector<double> slots;
  std::vector<double> out;
  std::vector<void *> argv;
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
  bool metal_sample_threadgroups = false;
  uint32_t metal_threadgroup_scratch_bytes = 0;

  // ── The kernel at the port ───────────────────────────────────────────────
  // `kernel` is the argv thunk of the loaded artifact: argv[0] = tau
  // (`frames` Q32.32 clock values), argv[1..k] = one double per param input
  // in port order (`input_slots[j]` is the slot the j-th param lives in),
  // argv[k+1] = the frame-major output. Everything else — the slot vector,
  // the arrays, the SSA scratch, the stage-0 prologue — is inside the
  // function. `frames` must equal the runtime's buffer length.
  tropical_jit::KernelArgvFn         kernel = nullptr;
  uint32_t                           frames = 0;
  std::vector<uint32_t>              input_slots;
  std::vector<std::string>           channel_names;
  std::vector<int64_t>               tau;
  std::vector<void *>                argv;

  // Metal pre-pass (TROPICAL_METAL artifacts only). The coefficient plan as a
  // host function over the slot vector and the coefficient columns: run once
  // at load (before publish) and after every control-plane slot write, so the
  // GPU kernel — which has no prologue of its own — reads a complete f64-
  // computed slot/column image. A JIT artifact carries none; its kernel does
  // this work inside itself.
  //
  // BANKS-AS-DATA (per-array staging): the pre-pass also fills the COEFFICIENT
  // COLUMNS — `array_ptrs` slots whose fills classified s0 — that the GPU
  // kernel's in-loop `Index` reads. Those slots (the plan advertises them as
  // `coeff_array_slots`) are GENERATION-BUFFERED together with the entire
  // scalar slot set. A control transaction materializes both into one free
  // generation and release-publishes one generation word.
  tropical_jit::Stage0Fn             stage0 = nullptr;
  uint32_t                           output_channel_count = 1;

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

  // Host storage for the Metal pre-pass: the array backing stores (the
  // coefficient columns among them) and its SSA scratch. Unused by a JIT
  // artifact, whose kernel allocates its own.
  std::vector<int64_t> coeff_temps;
  std::vector<std::vector<int64_t>> array_storage;
  std::vector<int64_t *> array_ptrs;
  std::vector<uint64_t> array_sizes;

  // Array slot names from the manifest (diagnostic metadata).
  std::vector<std::string> array_names;

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
  explicit FlatRuntime(unsigned int buffer_length);
  ~FlatRuntime();

  /**
   * Load one kernel artifact: the port IR (defining `tropical_kernel` and
   * its argv thunk), the MSL realization (empty for JIT-only), the Metal
   * pre-pass IR (empty unless MSL is present and the plan hoisted
   * anything), and the `tropical_kernel_1` manifest. The manifest's
   * `frames` must equal this runtime's buffer length. A failed load leaves
   * the previous kernel playing.
   */
  bool load(const std::string & ir_text, const std::string & msl_source,
            const std::string & stage0_ir, const std::string & manifest_json);

  /** Publish one artifact and return its exact initial generation pair. */
  PublishedGeneration load_generation(
    const std::string & ir_text, const std::string & msl_source,
    const std::string & stage0_ir, const std::string & manifest_json);

  /**
   * Atomically publish distinct audio and observation artifacts. The audio
   * artifact may own Metal execution; observation is JIT-only. Slot layouts
   * may differ: each audio control image is projected by exact manifest name
   * into the observation program's initial slot image.
   */
  PublishedGeneration load_with_observation_generation(
    const std::string & audio_ir, const std::string & audio_msl_source,
    const std::string & audio_stage0_ir,
    const std::string & audio_manifest_json,
    const std::string & observation_ir,
    const std::string & observation_manifest_json);

  /**
   * Control-plane/test-only: reposition the active kernel's sample clock
   * (the `--start` of render verbs — long-τ gates render windows at
   * arbitrary positions). Serialized against hot-swap.
   */
  void set_sample_index(uint64_t idx);

  /**
   * Process one buffer of audio. Called from the audio thread.
   * Fills the compact frame-major output image. `outputBuffer` remains the
   * channel-0 compatibility view used by historical render clients.
   */
  void process()
  {
    auto finish_silence = [&] {
      std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0);
      std::fill(interleaved_output_buffer_.begin(),
                interleaved_output_buffer_.end(), 0.0);
      rendered_output_channel_count_.store(1, std::memory_order_release);
      audio_sample_index_ += buffer_length_;
      audio_device_frame_ += buffer_length_;
      publish_audio_boundary();
    };

#ifdef TROPICAL_METAL
    // Metal's audio-facing path never captures KernelState, owns a control
    // generation, packs coefficients, submits work, or waits. The stable
    // queue and its tile storage live for the FlatRuntime's whole lifetime.
    if (metal_audio_enabled_.load(std::memory_order_acquire))
    {
      const uint32_t output_channels = metal_tiles_->output_channels();
      double * const rendered_output = output_channels == 1
        ? outputBuffer.data() : interleaved_output_buffer_.data();
      metal_tiles_->synchronize_audio_coordinates(
        audio_device_frame_, audio_sample_index_);
      tropical_metal::set_audio_callback_thread(true);
      metal_tiles_->consume(rendered_output, buffer_length_);
      tropical_metal::set_audio_callback_thread(false);
      audio_device_frame_ = metal_tiles_->published_device_frame();
      audio_sample_index_ = metal_tiles_->published_source_sample();
      apply_fade_to_output(rendered_output, output_channels);
      if (output_channels > 1)
        for (unsigned int frame = 0; frame < buffer_length_; ++frame)
          outputBuffer[frame] = rendered_output[
            static_cast<std::size_t>(frame) * output_channels];
      rendered_output_channel_count_.store(
        output_channels, std::memory_order_release);
      publish_audio_boundary();
      return;
    }
#endif

    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire);
        seam
        && seam->pause_before_boundary_capture.load(
          std::memory_order_acquire))
    {
      seam->boundary_capture_pending.store(true, std::memory_order_release);
      while (!seam->release_boundary_capture.load(std::memory_order_acquire))
        std::this_thread::yield();
    }

    // Wait-free boundary linearization. The returned even word contains the
    // exact state/generation captured by this callback. The first add marks
    // the word odd while audio updates the next-capture prediction; the
    // second makes it even and advances the epoch. A control publication CAS
    // therefore lands wholly before capture or loses and retries off-RT.
    const uint64_t captured_boundary_word =
      capture_boundary_word_.fetch_add(
        kCapturePhaseBit_, std::memory_order_acquire);
    const uint32_t captured_state_idx =
      static_cast<uint32_t>(
        (captured_boundary_word & kCaptureStateMask_) >> 2);
    const uint32_t captured_control_gen =
      static_cast<uint32_t>(
        captured_boundary_word & kCaptureGenerationMask_);

    // Apply a control-plane clock reposition exactly at this buffer boundary.
    // The plain audio_* clock fields are owned solely by this thread.
    const uint64_t request_seq_before =
      sample_index_request_seq_.load(std::memory_order_acquire);
    if ((request_seq_before & 1U) == 0
        && request_seq_before
             != audio_applied_sample_index_seq_.load(
                  std::memory_order_relaxed))
    {
      const uint64_t requested =
        requested_sample_index_.load(std::memory_order_acquire);
      const uint64_t request_seq_after =
        sample_index_request_seq_.load(std::memory_order_acquire);
      if (request_seq_before == request_seq_after)
      {
        audio_sample_index_ = requested;
        audio_applied_sample_index_seq_.store(
          request_seq_after, std::memory_order_relaxed);
      }
    }
    const uint64_t start_sample_index = audio_sample_index_;

    // Own a state before reading any of its movable storage, then revalidate
    // the publication. A control writer may wait/yield; the audio thread makes
    // only a small fixed number of attempts and fails safely to silence.
    uint32_t state_idx = captured_state_idx;
    bool state_owned = false;
    for (unsigned int attempt = 0; attempt < kAudioCaptureAttempts_; ++attempt)
    {
      StorageOwner expected = StorageOwner::Free;
      if (!state_owners_[state_idx].compare_exchange_strong(
            expected, StorageOwner::Audio,
            std::memory_order_acquire, std::memory_order_relaxed))
        continue;
      const uint64_t boundary_now =
        capture_boundary_word_.load(std::memory_order_relaxed);
      if ((boundary_now & kCaptureStateMask_)
          == (captured_boundary_word & kCaptureStateMask_))
      {
        state_owned = true;
        break;
      }
      state_owners_[state_idx].store(
        StorageOwner::Free, std::memory_order_release);
    }
    if (!state_owned)
    {
      next_capture_sample_index_.store(
        start_sample_index + buffer_length_, std::memory_order_relaxed);
      next_capture_effective_sample_index_.store(
        start_sample_index + buffer_length_, std::memory_order_relaxed);
      state_requires_reprime_[state_idx].store(
        true, std::memory_order_relaxed);
      capture_boundary_word_.fetch_add(
        kCapturePhaseBit_, std::memory_order_release);
      ownership_failure_count_.fetch_add(1, std::memory_order_relaxed);
      finish_silence();
      return;
    }

    KernelState & state = states_[state_idx];

    // Own the published slot/coefficient generation before copying or using
    // it. Revalidating after the CAS closes both the initial race and the ABA
    // reuse window: a writer cannot mutate an audio-owned generation.
    uint32_t control_gen = captured_control_gen;
    bool generation_owned = false;
    for (unsigned int attempt = 0; attempt < kAudioCaptureAttempts_; ++attempt)
    {
      StorageOwner expected = StorageOwner::Free;
      if (!control_generation_owners_[state_idx][control_gen]
             .compare_exchange_strong(
               expected, StorageOwner::Audio,
               std::memory_order_acquire, std::memory_order_relaxed))
        continue;
      const uint64_t boundary_now =
        capture_boundary_word_.load(std::memory_order_relaxed);
      if ((boundary_now
           & (kCaptureStateMask_ | kCaptureGenerationMask_))
          == (captured_boundary_word
              & (kCaptureStateMask_ | kCaptureGenerationMask_)))
      {
        generation_owned = true;
        break;
      }
      control_generation_owners_[state_idx][control_gen].store(
        StorageOwner::Free, std::memory_order_release);
    }
    if (!generation_owned)
    {
      next_capture_sample_index_.store(
        start_sample_index + buffer_length_, std::memory_order_relaxed);
      next_capture_effective_sample_index_.store(
        start_sample_index + buffer_length_, std::memory_order_relaxed);
      state_requires_reprime_[state_idx].store(
        true, std::memory_order_relaxed);
      capture_boundary_word_.fetch_add(
        kCapturePhaseBit_, std::memory_order_release);
      ownership_failure_count_.fetch_add(1, std::memory_order_relaxed);
      state_owners_[state_idx].store(
        StorageOwner::Free, std::memory_order_release);
      finish_silence();
      return;
    }

    // The generation capture above is the boundary linearization point.
    // JIT callbacks render the captured generation at this boundary. Metal
    // callbacks returned above after consuming a worker-prepared epoch tile.
    next_capture_sample_index_.store(
      start_sample_index + buffer_length_, std::memory_order_relaxed);
    next_capture_effective_sample_index_.store(
      start_sample_index + buffer_length_,
      std::memory_order_relaxed);
    state_requires_reprime_[state_idx].store(
      false, std::memory_order_relaxed);
    capture_boundary_word_.fetch_add(
      kCapturePhaseBit_, std::memory_order_release);

    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire);
        seam && seam->pause_after_state_ownership.load(std::memory_order_acquire))
    {
      seam->state_owned.store(true, std::memory_order_release);
      while (!seam->release_state.load(std::memory_order_acquire))
        std::this_thread::yield();
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

    // No active kernel — emit silence.
    if (state.kernel == nullptr)
    {
      std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0);
      const uint32_t output_channels = state.output_channel_count;
      std::fill_n(interleaved_output_buffer_.begin(),
        static_cast<std::size_t>(buffer_length_) * output_channels, 0.0);
      rendered_output_channel_count_.store(
        output_channels, std::memory_order_release);
      audio_sample_index_ += buffer_length_;
      audio_device_frame_ += buffer_length_;
      publish_audio_boundary();
      control_generation_owners_[state_idx][control_gen].store(
        StorageOwner::Free, std::memory_order_release);
      state_owners_[state_idx].store(
        StorageOwner::Free, std::memory_order_release);
      return;
    }

    const uint32_t output_channels = state.output_channel_count;
    double * const rendered_output = output_channels == 1
      ? outputBuffer.data() : interleaved_output_buffer_.data();
    // The emitter writes every channel of every frame (silent channels
    // included); clearing first keeps a refused call deterministic too.
    std::fill_n(rendered_output,
      static_cast<std::size_t>(buffer_length_) * output_channels, 0.0);

    // One call renders the whole buffer: tau is the Q32.32 clock of each
    // frame, the params point into this buffer's private slot snapshot.
    for (uint32_t s = 0; s < buffer_length_; ++s)
      state.tau[s] = static_cast<int64_t>(
        (start_sample_index + static_cast<uint64_t>(s)) << 32);
    state.argv[0] = state.tau.data();
    for (std::size_t j = 0; j < state.input_slots.size(); ++j)
      state.argv[1 + j] = &state.audio_slots[state.input_slots[j]];
    state.argv.back() = rendered_output;
    state.kernel(state.argv.data());

    audio_sample_index_ += buffer_length_;
    audio_device_frame_ += buffer_length_;
    apply_fade_to_output(rendered_output, output_channels);
    if (output_channels > 1)
      for (unsigned int frame = 0; frame < buffer_length_; ++frame)
        outputBuffer[frame] =
          rendered_output[static_cast<std::size_t>(frame) * output_channels];
    rendered_output_channel_count_.store(
      output_channels, std::memory_order_release);

    // Publish the completed boundary only after all output processing for this
    // buffer (including fade) is complete.
    publish_audio_boundary();
    control_generation_owners_[state_idx][control_gen].store(
      StorageOwner::Free, std::memory_order_release);
    state_owners_[state_idx].store(
      StorageOwner::Free, std::memory_order_release);
  }

  /**
   * Deterministic/offline renderer only. Metal waits off the callback until
   * the worker has prepared the next exact tile, then executes the ordinary
   * bounded callback path once. JIT is identical to process().
   */
  void process_offline();

  /**
   * DAC start/control thread only. JIT retains the legacy cache warm-up
   * process cycles. Metal does not consume prepared audio before the device
   * clock starts; it waits off-RT until the exact first callback tile is ready.
   */
  void prepare_realtime_start(unsigned int process_cycles);

  std::vector<double> outputBuffer;

  unsigned int getBufferLength() const { return buffer_length_; }

  /** Number of channels declared by the active published plan. */
  uint32_t getOutputChannelCount() const noexcept
  {
    return output_channel_count_.load(std::memory_order_acquire);
  }

  /** Compact frame-major output (`frame * channels + channel`). The pointer is
      stable for the runtime lifetime; callers consume only
      `buffer_length * getOutputChannelCount()` values. */
  const double * getInterleavedOutputBuffer() const noexcept
  {
    return getOutputChannelCount() == 1
      ? outputBuffer.data() : interleaved_output_buffer_.data();
  }

  /** Callback-local image access for the DAC. Publication can race immediately
      after a callback, so device mapping follows the layout actually rendered
      by that callback rather than a newly published plan's future layout. */
  uint32_t getRenderedOutputChannelCount() const noexcept
  {
    return rendered_output_channel_count_.load(std::memory_order_acquire);
  }

  const double * getRenderedInterleavedOutputBuffer() const noexcept
  {
    return getRenderedOutputChannelCount() == 1
      ? outputBuffer.data() : interleaved_output_buffer_.data();
  }

  uint32_t metal_render_tile_frames() const
  {
#ifdef TROPICAL_METAL
    return metal_runtime_loaded_.load(std::memory_order_acquire)
      ? metal_render_tile_frames_ : 0;
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
      const std::vector<double> previous = state.slots;
      state.slots[idx] = value;
#ifdef TROPICAL_METAL
      if (state.metal)
      {
        if (!publish_metal_control_snapshot_locked(state, state_idx).ok)
          state.slots = previous;
        return;
      }
#endif
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
        const std::vector<double> previous = state.slots;
        state.slots[i] = value;
#ifdef TROPICAL_METAL
        if (state.metal)
        {
          if (!publish_metal_control_snapshot_locked(
                state, state_idx).ok)
          {
            state.slots = previous;
            return false;
          }
          return true;
        }
#endif
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
    const std::vector<double> previous = state.slots;
    if constexpr (std::is_void_v<std::invoke_result_t<Fn &, KernelState &>>)
    {
      fn(state);
#ifdef TROPICAL_METAL
      if (state.metal)
      {
        if (!publish_metal_control_snapshot_locked(state, state_idx).ok)
          state.slots = previous;
        return;
      }
#endif
      publish_control_snapshot(state, state_idx);
    }
    else
    {
      auto result = fn(state);
#ifdef TROPICAL_METAL
      if (state.metal)
      {
        if (!publish_metal_control_snapshot_locked(state, state_idx).ok)
          state.slots = previous;
        return result;
      }
#endif
      publish_control_snapshot(state, state_idx);
      return result;
    }
  }

  // ── Random-access observation (room / scope / telemetry consumers) ─────────
  // Render `count` coordinates from one atomically acquired program/control
  // image. `out` is channel-major: out[k*count+i] is output channel
  // channel_ids[k] at start_index+i*stride. The scope's taps are output
  // channels of the observation artifact (design/typed-kernel.md). This path never borrows a live KernelState or takes
  // build_mutex_, so a paused observer cannot delay control or hot-swap.
  RenderWindowStatus render_window_observation(
    uint64_t start_index, uint32_t count, uint32_t stride,
    const uint32_t * channel_ids, uint32_t n_channels, double * out,
    std::shared_ptr<const ObservationSnapshot> snapshot = {})
  {
    if (!out || stride == 0 || (n_channels > 0 && !channel_ids))
      return RenderWindowStatus::InvalidArgument;
    if (!snapshot)
      snapshot = std::atomic_load_explicit(
        &observation_snapshot_, std::memory_order_acquire);
    if (!snapshot || !snapshot->program || !snapshot->controls)
      return RenderWindowStatus::Unsupported;
    const ObservationProgramImage & program = *snapshot->program;
    if (program.kernel == nullptr || program.frames == 0)
      return RenderWindowStatus::Unsupported;
    for (uint32_t k = 0; k < n_channels; ++k)
      if (channel_ids[k] >= program.output_channel_count)
        return RenderWindowStatus::InvalidArgument;

    std::lock_guard<std::mutex> observation_lock(observation_render_mutex_);
    ObservationRenderWorkspace & workspace = observation_workspace_;
    const uint32_t N = program.frames;
    const uint32_t C = program.output_channel_count;
    if (workspace.program_version != program.program_version)
    {
      workspace.tau.assign(N, 0);
      workspace.out.assign(static_cast<std::size_t>(N) * C, 0.0);
      workspace.argv.assign(program.input_slots.size() + 2, nullptr);
      workspace.program_version = program.program_version;
    }
    // The controls image is this read's param point; the kernel reads
    // params through pointers into it, so a copy keeps it immutable.
    workspace.slots = snapshot->controls->slots;
    workspace.argv[0] = workspace.tau.data();
    for (std::size_t j = 0; j < program.input_slots.size(); ++j)
      workspace.argv[1 + j] = &workspace.slots[program.input_slots[j]];
    workspace.argv.back() = workspace.out.data();

    // The kernel's shape is static: `count` coordinates render in chunks of
    // N calls, each over its own tau image (coordinates past `count` in the
    // last chunk are computed and discarded — a closed form has no cost
    // beyond the arithmetic).
    for (uint32_t chunk = 0; chunk < count; chunk += N)
    {
      for (uint32_t i = 0; i < N; ++i)
        workspace.tau[i] = static_cast<int64_t>(
          (start_index
           + static_cast<uint64_t>(chunk + i) * stride) << 32);
      program.kernel(workspace.argv.data());
      const uint32_t produced = std::min<uint32_t>(N, count - chunk);
      for (uint32_t k = 0; k < n_channels; ++k)
        for (uint32_t i = 0; i < produced; ++i)
          out[static_cast<size_t>(k) * count + chunk + i] =
            workspace.out[static_cast<std::size_t>(i) * C + channel_ids[k]];
      if (chunk == 0)
      {
        if (RuntimeOwnershipTestSeam * seam =
              ownership_test_seam_.load(std::memory_order_acquire);
            seam && seam->pause_after_observation_coordinate.load(
              std::memory_order_acquire))
        {
          seam->observation_coordinate_completed.store(
            true, std::memory_order_release);
          while (!seam->release_observation_coordinate.load(
            std::memory_order_acquire))
            std::this_thread::yield();
        }
      }
    }
    return RenderWindowStatus::Rendered;
  }

  // Resolve names and evaluate against the same pinned generation. A hot-swap
  // cannot reinterpret an old room/tap name through the new program's layout.
  ObservationRenderResult render_window_observation_by_name(
    uint64_t start_index, uint32_t count, uint32_t stride,
    const std::vector<std::string> & slot_names, double * out)
  {
    ObservationRenderResult result;
    const auto snapshot = std::atomic_load_explicit(
      &observation_snapshot_, std::memory_order_acquire);
    if (!snapshot || !snapshot->program || !snapshot->controls)
    {
      result.status = RenderWindowStatus::Unsupported;
      return result;
    }
    result.program_version = snapshot->program->program_version;
    result.control_version = snapshot->controls->control_version;
    result.effective_sample_index =
      snapshot->controls->effective_sample_index;
    // A name is an output channel of the observation kernel, or else a
    // control slot of the pinned control image (a param or one of its
    // discipline companions), whose trajectory is its constant value — the
    // data plane's way to read a control back without a second method.
    std::vector<uint32_t> channel_ids;
    std::vector<std::pair<std::size_t, uint32_t>> constant_slots; // (k, slot)
    std::vector<std::size_t> channel_positions;
    for (std::size_t k = 0; k < slot_names.size(); ++k)
    {
      const std::string & name = slot_names[k];
      const auto channel = snapshot->program->channel_lookup.find(name);
      if (channel != snapshot->program->channel_lookup.end())
      {
        channel_positions.push_back(k);
        channel_ids.push_back(channel->second);
        continue;
      }
      const auto slot = snapshot->program->slot_lookup.find(name);
      if (slot == snapshot->program->slot_lookup.end()
          || slot->second >= snapshot->controls->slots.size())
      {
        result.status = RenderWindowStatus::InvalidArgument;
        result.unknown_slot = name;
        return result;
      }
      constant_slots.emplace_back(k, slot->second);
    }
    std::vector<double> rendered(
      static_cast<std::size_t>(count) * channel_ids.size(), 0.0);
    // A read of controls only never runs the kernel.
    result.status = channel_ids.empty()
      ? RenderWindowStatus::Rendered
      : render_window_observation(
          start_index, count, stride,
          channel_ids.data(), static_cast<uint32_t>(channel_ids.size()),
          rendered.data(), snapshot);
    if (result.status != RenderWindowStatus::Rendered) return result;
    for (std::size_t j = 0; j < channel_positions.size(); ++j)
      std::copy_n(rendered.data() + j * count, count,
                  out + channel_positions[j] * count);
    for (const auto & [k, slot] : constant_slots)
      std::fill_n(out + k * count, count, snapshot->controls->slots[slot]);
    return result;
  }

  bool render_window(uint64_t start_index, uint32_t count,
                     const uint32_t * channel_ids, uint32_t n_channels,
                     double * out)
  {
    return render_window_observation(
      start_index, count, 1, channel_ids, n_channels, out)
      == RenderWindowStatus::Rendered;
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

  // DAC lifecycle signal for the asynchronous Metal scheduler. While a
  // callback is running, control epochs are serialized through activation
  // acknowledgement so glide projections remain continuous. With the DAC
  // stopped, unclaimed epochs may coalesce because no callback can claim them.
  void set_realtime_running(bool running) noexcept
  {
#ifdef TROPICAL_METAL
    realtime_running_.store(running, std::memory_order_release);
#else
    (void)running;
#endif
  }

  // Number of callbacks that emitted silence because bounded ownership
  // acquisition could not obtain a coherent state/generation.
  uint64_t ownership_failure_count() const
  {
    return ownership_failure_count_.load(std::memory_order_acquire);
  }

  // Sticky count of underlying Metal dispatch failures. One failed command
  // latches its kernel silent but increments this only once; a fresh kernel
  // restores execution without erasing the evidence. Qualification requires
  // zero.
  uint64_t metal_dispatch_failure_count() const
  {
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    return metal_worker_ ? metal_worker_->dispatch_failure_count() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_render_starvation_count() const
  {
#ifdef TROPICAL_METAL
    return metal_tiles_ ? metal_tiles_->starvation_count() : 0;
#else
    return 0;
#endif
  }

  std::array<uint64_t, 16> metal_first_starvation_snapshot() const
  {
#ifdef TROPICAL_METAL
    if (metal_tiles_)
    {
      const EpochTileStarvationSnapshot snapshot =
        metal_tiles_->first_starvation_snapshot();
      return {
        snapshot.valid ? 1ULL : 0ULL,
        snapshot.epoch_id,
        snapshot.device_frame,
        snapshot.source_sample,
        snapshot.expected_tile_device,
        snapshot.expected_tile_source,
        snapshot.last_published_epoch,
        snapshot.last_published_device_end,
        snapshot.last_published_source_end,
        snapshot.active_bank,
        snapshot.expected_tile_index,
        snapshot.observed_tile_state,
        snapshot.ready_mask,
        snapshot.free_mask,
        snapshot.rendering_mask,
        snapshot.reading_mask,
      };
    }
#endif
    return {};
  }

  uint64_t metal_epoch_tag_mismatch_count() const
  {
#ifdef TROPICAL_METAL
    return metal_tiles_ ? metal_tiles_->tag_mismatch_count() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_activation_retarget_count() const
  {
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    return metal_worker_ ? metal_worker_->activation_retarget_count() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_activation_failure_count() const
  {
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    return metal_worker_ ? metal_worker_->activation_failure_count() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_callback_thread_violation_count() const
  {
#ifdef TROPICAL_METAL
    return tropical_metal::callback_thread_violation_count();
#else
    return 0;
#endif
  }

  uint32_t metal_worker_capacity_frames() const
  {
#ifdef TROPICAL_METAL
    return metal_tiles_ ? metal_tiles_->capacity_frames() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_published_activation_epoch() const
  {
#ifdef TROPICAL_METAL
    return metal_tiles_ ? metal_tiles_->published_activation_epoch() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_acknowledged_activation_epoch() const
  {
#ifdef TROPICAL_METAL
    return metal_tiles_ ? metal_tiles_->activation_acknowledged() : 0;
#else
    return 0;
#endif
  }

  std::array<uint64_t, 8> metal_worker_stage_times() const
  {
    std::array<uint64_t, 8> result{};
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    if (metal_worker_)
    {
      const auto times = metal_worker_->stage_times();
      result = {
        times.request_received,
        times.activation_target_reserved,
        times.candidate_render_submitted,
        times.candidate_gpu_completion,
        times.candidate_window_ready,
        times.activation_published,
        times.activation_acknowledged,
        times.old_epoch_retired,
      };
    }
#endif
    return result;
  }

  std::array<uint64_t, 4> metal_activation_latency_stats() const
  {
    std::array<uint64_t, 4> result{};
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    if (metal_worker_)
    {
      const auto stats = metal_worker_->activation_latency_stats();
      result = {
        stats.count, stats.total_ns, stats.min_ns, stats.max_ns
      };
    }
#endif
    return result;
  }

  uint64_t metal_worker_cpu_time_ns() const
  {
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    return metal_worker_ ? metal_worker_->worker_cpu_time_ns() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_worker_wall_time_ns() const
  {
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    return metal_worker_ ? metal_worker_->worker_wall_time_ns() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_render_time_ns() const
  {
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    return metal_worker_ ? metal_worker_->render_time_ns() : 0;
#else
    return 0;
#endif
  }

  uint64_t metal_rendered_frame_count() const
  {
#ifdef TROPICAL_METAL
    std::lock_guard<std::mutex> lock(build_mutex_);
    return metal_worker_ ? metal_worker_->rendered_frame_count() : 0;
#else
    return 0;
#endif
  }

  // Test-only: install/remove the deterministic ownership pause seam.
  void set_ownership_test_seam(RuntimeOwnershipTestSeam * seam)
  {
    ownership_test_seam_.store(seam, std::memory_order_release);
  }

#ifdef TROPICAL_METAL
  void set_metal_worker_test_seam(
    tropical_metal::MetalRenderWorkerTestSeam * seam)
  {
    std::lock_guard<std::mutex> lock(build_mutex_);
    if (metal_worker_) metal_worker_->set_test_seam(seam);
  }
#endif

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

  // Qualification/test-only oracle entry point. It runs the exact production
  // discipline math at a caller-supplied first-audible sample index, allowing
  // a separate reference runtime to replay production's effective boundary.
  // It is intentionally not in the C API or socket protocol as a way to
  // control production time.
  ParamDispatchResult dispatch_param_sync_at_sample_index(
    const std::string & name, double value, uint64_t sample_index);

private:
  struct ControlGenerationReservation
  {
    uint32_t target = 0;
    bool owned = false;
  };

  ParamDispatchResult dispatch_param_sync_locked(
    KernelState & state, uint32_t state_idx, const std::string & name,
    double value, uint64_t sample_index);

  ControlGenerationReservation reserve_control_generation(
    KernelState & state, uint32_t state_index);
  void materialize_control_snapshot(
    KernelState & state, uint32_t target);
  void commit_control_snapshot(
    KernelState & state, uint32_t state_index,
    ControlGenerationReservation reservation,
    uint64_t effective_sample_index = UINT64_MAX);
  void release_control_snapshot_reservation(
    uint32_t state_index, ControlGenerationReservation reservation);
  void publish_observation_controls_locked(
    const KernelState & state,
    uint64_t effective_sample_index = UINT64_MAX);

  void apply_fade_to_output(double * output, uint32_t channels)
  {
    int fi = fade_in_remaining_.load(std::memory_order_relaxed);
    int fo = fade_out_remaining_.load(std::memory_order_relaxed);
    if (fi <= 0 && fo == -1) return;
    for (unsigned int s = 0; s < buffer_length_; ++s)
    {
      double gain = 1.0;
      if (fi > 0)
      {
        const double t = 1.0 - static_cast<double>(fi) / kFadeSamples_;
        gain *= t * t * (3.0 - 2.0 * t);
        --fi;
      }
      if (fo != -1)
      {
        if (fo > 0)
        {
          const double t = static_cast<double>(fo) / kFadeSamples_;
          gain *= t * t * (3.0 - 2.0 * t);
          --fo;
        }
        else
          gain = 0.0;
      }
      for (uint32_t channel = 0; channel < channels; ++channel)
        output[static_cast<std::size_t>(s) * channels + channel] *= gain;
    }
    fade_in_remaining_.store(fi, std::memory_order_relaxed);
    fade_out_remaining_.store(fo, std::memory_order_relaxed);
  }

  void publish_audio_boundary()
  {
    published_device_frame_.store(
      audio_device_frame_, std::memory_order_release);
    published_sample_index_.store(
      audio_sample_index_, std::memory_order_release);
#ifdef TROPICAL_METAL
    if (metal_queue_ready_.load(std::memory_order_acquire))
      metal_tiles_->synchronize_audio_coordinates(
        audio_device_frame_, audio_sample_index_);
#endif
  }

  // build_kernel_state maps a parsed plan's *metadata* (everything except
  // the kernel handle) into a fresh KernelState; publish_state carries the
  // sample coordinate and performs the atomic double-buffer flip.
  // load_ir fills the kernel handle (via compile_ir_text) between them.
  KernelState build_kernel_state(const tropical_plan6::ParsedKernel1 & parsed);
  PublishedGeneration publish_state(
    KernelState && new_state,
    const KernelState * observation_state = nullptr);

#ifdef TROPICAL_METAL
  tropical_metal::RenderEpochRequest make_metal_epoch_request(
    KernelState & state, uint64_t epoch_id,
    tropical_metal::EpochTransitionKind transition,
    uint64_t source_origin,
    uint32_t generation = UINT32_MAX) const;
  tropical_metal::EpochScheduleResult schedule_metal_epoch_locked(
    KernelState & state,
    tropical_metal::EpochTransitionKind transition,
    uint64_t requested_source,
    uint32_t generation = UINT32_MAX);
  tropical_metal::EpochScheduleResult
  publish_metal_control_snapshot_locked(
    KernelState & state, uint32_t state_index);
  void wait_for_next_metal_tile(const char * purpose);
  static uint32_t configure_metal_render_tile_frames(
    uint32_t device_frames);
#endif

  // Materialize one complete control transaction into a FREE generation.
  // Scalar coefficient slots, coefficient columns, and ordinary slots become
  // visible together through one release-published generation word. Called
  // under build_mutex_, except while constructing an unpublished local state.
  void publish_control_snapshot(
    KernelState & state, uint32_t state_index = UINT32_MAX,
    uint64_t effective_sample_index = UINT64_MAX)
  {
    auto reservation = reserve_control_generation(state, state_index);
    materialize_control_snapshot(state, reservation.target);
    commit_control_snapshot(
      state, state_index, reservation, effective_sample_index);
  }

  unsigned int buffer_length_;
  // Fixed-capacity callback storage: loading a wider plan never allocates or
  // resizes on the real-time thread. Mono kernels write the legacy view
  // directly; wider kernels write this compact frame-major image.
  std::vector<double> interleaved_output_buffer_;
  std::atomic<uint32_t> output_channel_count_{1};
  std::atomic<uint32_t> rendered_output_channel_count_{1};
  std::array<KernelState, 2> states_;
  std::atomic<uint32_t> active_state_{0};
  std::array<std::atomic<StorageOwner>, 2> state_owners_{};
  std::array<std::array<std::atomic<StorageOwner>, 3>, 2>
    control_generation_owners_{};
  std::atomic<uint64_t> recompile_version_{0};
  // C++ shared_ptr atomic free functions support the release SDK where the
  // C++20 atomic<shared_ptr> specialization is unavailable.
  std::shared_ptr<const ObservationSnapshot> observation_snapshot_;
  uint64_t next_observation_control_version_ = 1;
  std::mutex observation_render_mutex_;
  ObservationRenderWorkspace observation_workspace_;
  std::atomic<uint64_t> ownership_failure_count_{0};
  std::atomic<RuntimeOwnershipTestSeam *> ownership_test_seam_{nullptr};

  // Buffer-boundary clock handoff. Only the audio thread touches plain
  // audio_* fields. Control publishes an odd/even seqlock around the atomic
  // value; audio never spins and simply defers an in-progress request.
  std::atomic<uint64_t> requested_sample_index_{0};
  std::atomic<uint64_t> sample_index_request_seq_{0};
  std::atomic<uint64_t> audio_applied_sample_index_seq_{0};
  uint64_t audio_sample_index_ = 0;
  uint64_t audio_device_frame_ = 0;
  std::atomic<uint64_t> published_sample_index_{0};
  std::atomic<uint64_t> published_device_frame_{0};

#ifdef TROPICAL_METAL
  uint32_t metal_render_tile_frames_ = 0;
  std::unique_ptr<EpochTileQueue> metal_tiles_;
  std::atomic<bool> realtime_running_{false};
  std::unique_ptr<tropical_metal::MetalRenderWorker> metal_worker_;
  std::atomic<bool> metal_queue_ready_{false};
  std::atomic<bool> metal_runtime_loaded_{false};
  std::atomic<bool> metal_audio_enabled_{false};
  uint64_t next_metal_epoch_id_ = 1;
#endif

  // Atomic audio-boundary word:
  //   bits 0..1  published control generation (0..2)
  //   bit  2     active state index
  //   bit  3     audio capture phase (set while prediction is updated)
  //   bits 4..63 boundary epoch
  // The two phase-bit additions advance the epoch without carrying through
  // the isolated state/generation bits. Even at the impossible B=1 extreme,
  // 2^60 callbacks is about 829,000 years at 44.1 kHz, so wrap is outside
  // any runtime lifetime.
  static_assert(std::atomic<uint64_t>::is_always_lock_free);
  static constexpr uint64_t kCaptureGenerationMask_ = 0x3;
  static constexpr uint64_t kCaptureStateMask_ = 0x4;
  static constexpr uint64_t kCapturePhaseBit_ = 0x8;
  std::atomic<uint64_t> capture_boundary_word_{0};
  std::atomic<uint64_t> next_capture_sample_index_{0};
  std::atomic<uint64_t> next_capture_effective_sample_index_{0};
  std::array<std::atomic<bool>, 2> state_requires_reprime_{{true, true}};

  mutable std::mutex build_mutex_;

  static constexpr unsigned int kAudioCaptureAttempts_ = 4;
  static constexpr int kFadeSamples_ = 2048;
  std::atomic<int> fade_in_remaining_{0};
  std::atomic<int> fade_out_remaining_{-1};
};

} // namespace tropical_runtime
