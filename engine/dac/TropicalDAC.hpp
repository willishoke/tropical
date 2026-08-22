#pragma once

#include "../lib/rtaudio/RtAudio.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

static constexpr int kPrimeCycles = 4;
static constexpr uint64_t kCallbackHistogramBinNs = 1000;
static constexpr std::size_t kCallbackHistogramRegularBins = 20000;
static constexpr std::size_t kCallbackHistogramBins =
  kCallbackHistogramRegularBins + 1; // final bin is >= 20 ms

/**
 * C — the device-boundary output bound. The last thing between a computed
 * value and a speaker: `out = clamp(sample, ±C)`.
 *
 * This is the first SAFETY-class gate in the project. Every other gate proves
 * agreement (wasm ≡ JIT), preservation (goldens), or law-conformance (the seam
 * atoms); none of them bounds MAGNITUDE, and two backends can bit-agree on a
 * speaker-destroying burst. Stateless hard min/max is the whole mechanism —
 * closed-form, bit-transparent below C, no state, no smoothing. NOT a limiter
 * and NOT a softclip: the purity story is swap continuity, and a hard clamp
 * downstream of the kernel does not touch it.
 *
 * WHY HERE AND NOT IN THE KERNEL. The emitted kernel's output buffer is not
 * the device boundary — it is the value of `f(τ)`, and the whole system reads
 * it as a number: `diffcli render-bytes`, the frozen goldens, the wasm≡JIT
 * differential, and the numeric coverage gates all consume it. `bootstrap-exp`
 * legitimately renders exp(x) out to e^10 ≈ 22026 through the sink to check the
 * emitted polynomial against libm. Clamping in `emitSinks` would make the
 * compiler lie about the value of the closed-form function, and would bake an
 * audio-specific magnitude assumption into an evaluator that is meant to stay
 * nature-agnostic (video, control-rate). The DAC callback is where a value
 * becomes sound, so it is where a sound-safety bound belongs. `WasmKernel`
 * mirrors this exactly: `render()` (the equivalence surface) is untouched and
 * `process()` (the worklet feed) clamps — see `web/runtime/kernel.ts`.
 *
 * WHAT SETS C. Not the patch corpus. Peak |sample| over
 * `patches/{reverse_reverb,scrub_reverb}` and `web/patches/{pure-sine-440,
 * ring-mod,tz-flanger}` is only 1.807 (measured 2026-07-20, 16384 samples
 * each), and C = 4 shipped on that basis — but that corpus contains no modal
 * playground patches, and the modal vocabulary legitimately reaches far higher.
 * A resonant lowpass has Q gain at resonance: `filterPair` maps
 * `res ↦ Q = 0.55·80^res`, the composed modal coefficient is `|A| ≈ Q`, and the
 * rendered peak is `≈ Q · master_gain`. At the top of the resonance knob
 * (res = 1, Q = 44, master gain 3.7) that is ≈ 163 — spike-free, correct, and
 * exactly the physics of a driven high-Q resonator. C = 4 would have hard-clipped
 * the top of a shipped knob by 20×. C = 256 is the smallest power of two above
 * the vocabulary's legitimate reach.
 *
 * C IS A HAZARD BOUND, NOT A LEVEL CONTROL. This follows `defaultSinkGain`'s
 * own committed position — amplitude is a frontend concern; the backend invents
 * no headroom scale of its own. C exists to stop the unbounded (∞, NaN, a 1e9
 * excursion from a datapath defect), not to voice the instrument.
 *
 * AND IT CANNOT DISCRIMINATE THE MODAL RAIL — stated plainly because the
 * temptation to believe otherwise is the whole hazard. The i64 modal-datapath
 * incident (design/modal-datapath-rail.local.md) renders peaks of ≈ 234 on the
 * production path, which is BELOW this C and therefore passes the clamp
 * untouched. That is not a hole to close by lowering C: the incident's peak and
 * the vocabulary's legitimate peak are the same order (234 vs 163, a factor of
 * 1.4), because both are `≈ Q · gain` — no value of C admits the instrument and
 * rejects the burst. The rail is fixed at its source (per-bank landing exponent);
 * the clamp is defense in depth against the NEXT defect, whose magnitude is
 * unknown and may be unbounded. Do not re-tune C to chase a known rail.
 *
 * NaN maps to −C: the first comparison is false on NaN, so the low arm takes
 * −C, which then passes the high arm. Ordered-compare + select rather than
 * std::fmin/fmax, whose IEEE minNum semantics return the *other* operand on NaN
 * and leave signed zero unspecified. Exact for every non-NaN value in [−C, C],
 * including −0.0 and denormals.
 */
static constexpr double kDeviceOutputBound = 256.0;

static inline double clamp_to_device_bound(double x)
{
  const double lo = x > -kDeviceOutputBound ? x : -kDeviceOutputBound;
  return lo < kDeviceOutputBound ? lo : kDeviceOutputBound;
}

static inline void update_max(std::atomic<uint64_t>& cur, uint64_t val)
{
  uint64_t prev = cur.load(std::memory_order_relaxed);
  while (val > prev && !cur.compare_exchange_weak(prev, val,
      std::memory_order_relaxed, std::memory_order_relaxed))
  {}
}

/**
 * TropicalDACImpl — templated DAC driver.
 *
 * AudioSource must provide:
 *   void process();
 *   std::vector<double> outputBuffer;
 *   unsigned int getBufferLength() const;
 *   void begin_fade_in(int samples = 2048);
 *   void begin_fade_out(int samples = 2048);
 *   bool is_fade_out_complete() const;
 *   void set_realtime_running(bool) noexcept;  // optional async scheduler hint
 *   unsigned int getOutputChannelCount() const; // optional; defaults to mono
 *   const double* getInterleavedOutputBuffer() const; // optional with count
 *
 * Both Graph and FlatRuntime satisfy this.
 */
template <typename AudioSource>
struct TropicalDACImpl
{
  AudioSource* source;
  RtAudio      audio;
  unsigned int sample_rate;
  unsigned int channels;
  bool         running = false;

  std::atomic<uint64_t> callback_count_{0};
  std::atomic<uint64_t> total_callback_ns_{0};
  std::atomic<uint64_t> max_callback_ns_{0};
  std::atomic<uint64_t> underrun_count_{0};
  std::atomic<uint64_t> overrun_count_{0};
  std::array<std::array<std::atomic<uint64_t>, kCallbackHistogramBins>, 2>
    callback_histogram_{};
  std::atomic<unsigned int> active_histogram_{0};
  std::atomic<unsigned int> requested_histogram_{0};
  std::atomic<uint64_t> stats_epoch_{1};
  std::atomic<uint64_t> requested_stats_epoch_{1};

  // One qualification capture slot. Storage is allocated at construction.
  // The explicit state machine enforces exactly one requester and one reader:
  // callback writes happen only in Requested, reader copies only in Reading,
  // and another request cannot begin until the successful read returns Idle.
  enum class CaptureState : uint8_t { Idle, Requested, Ready, Reading };
  static_assert(std::atomic<CaptureState>::is_always_lock_free);
  std::vector<double> captured_output_;
  std::atomic<CaptureState> capture_state_{CaptureState::Idle};
  std::atomic<uint64_t> capture_sequence_{0};
  std::atomic<uint64_t> capture_requested_{0};
  std::atomic<uint64_t> capture_ready_{0};
  std::atomic<uint64_t> captured_start_index_{0};
  std::atomic<unsigned int> negotiated_buffer_frames_{0};

  std::atomic<bool>     device_disconnected_{false};
  std::atomic<bool>     rtaudio_disconnect_latched_{false};
  std::atomic<uint64_t> disconnect_count_{0};
  std::atomic<uint64_t> reconnect_success_count_{0};
  std::atomic<uint64_t> reconnect_failure_count_{0};
  std::atomic<bool>     watcher_shutdown_{false};
  std::thread           watcher_thread_;
  unsigned int          active_device_id_{0};

  TropicalDACImpl(AudioSource* s, unsigned int sr, unsigned int ch)
    : source(s), sample_rate(sr), channels(ch),
      captured_output_(s->getBufferLength(), 0.0)
  {
    audio.setErrorCallback([this](RtAudioErrorType type, const std::string&) {
      if (type == RTAUDIO_DEVICE_DISCONNECT)
      {
        // Count a physical RtAudio disconnect edge exactly once. Watcher
        // retries and default-route changes have separate reconnect counters.
        if (!rtaudio_disconnect_latched_.exchange(
              true, std::memory_order_acq_rel))
          disconnect_count_.fetch_add(1, std::memory_order_relaxed);
        device_disconnected_.store(true, std::memory_order_release);
      }
    });
  }

  ~TropicalDACImpl() { stop(); }

  /**
   * Control/start thread only. Sources with an asynchronous render-ahead
   * backend may replace the legacy process() warm-up with a readiness barrier.
   * This keeps DAC startup from consuming a finite prepared queue before the
   * device clock begins pacing callbacks.
   */
  void prepare_source_for_start()
  {
    if constexpr (requires { source->prime_numeric_jit(); })
      source->prime_numeric_jit();
    if constexpr (requires {
                    source->prepare_realtime_start(kPrimeCycles);
                  })
      source->prepare_realtime_start(kPrimeCycles);
    else
      for (int i = 0; i < kPrimeCycles; ++i)
        source->process();
  }

  /**
   * A device switch/reconnect resumes an already-warm source. Readiness-aware
   * asynchronous sources still need a non-consuming barrier before the new
   * stream starts; legacy sources need no additional process cycles.
   */
  void prepare_source_for_stream_restart()
  {
    if constexpr (requires {
                    source->prepare_realtime_start(0);
                  })
      source->prepare_realtime_start(0);
  }

  void set_source_realtime_running(bool value) noexcept
  {
    if constexpr (requires {
                    { source->set_realtime_running(value) } noexcept;
                  })
      source->set_realtime_running(value);
  }

  void start()
  {
    if (running)
      return;

    // Declare realtime intent before preparation/opening so a control request
    // racing startup cannot cancel a glide epoch whose projection the callback
    // is about to consume.
    set_source_realtime_running(true);
    try
    {
      prepare_source_for_start();

      source->begin_fade_in();

      callback_count_.store(0, std::memory_order_relaxed);
      total_callback_ns_.store(0, std::memory_order_relaxed);
      max_callback_ns_.store(0, std::memory_order_relaxed);
      underrun_count_.store(0, std::memory_order_relaxed);
      overrun_count_.store(0, std::memory_order_relaxed);
      for (auto & bank : callback_histogram_)
        for (auto & bin : bank)
          bin.store(0, std::memory_order_relaxed);
      active_histogram_.store(0, std::memory_order_relaxed);
      requested_histogram_.store(0, std::memory_order_relaxed);
      stats_epoch_.store(1, std::memory_order_relaxed);
      requested_stats_epoch_.store(1, std::memory_order_relaxed);
      capture_requested_.store(0, std::memory_order_relaxed);
      capture_ready_.store(0, std::memory_order_relaxed);
      capture_state_.store(CaptureState::Idle, std::memory_order_relaxed);
      disconnect_count_.store(0, std::memory_order_relaxed);
      reconnect_success_count_.store(0, std::memory_order_relaxed);
      reconnect_failure_count_.store(0, std::memory_order_relaxed);
      rtaudio_disconnect_latched_.store(false, std::memory_order_relaxed);

      open_stream();
      running = true;

      device_disconnected_.store(false, std::memory_order_relaxed);
      watcher_shutdown_.store(false, std::memory_order_relaxed);
      watcher_thread_ = std::thread(&TropicalDACImpl::watcher_loop, this);
    }
    catch (...)
    {
      try { close_stream(/*fade=*/false); } catch (...) {}
      running = false;
      set_source_realtime_running(false);
      throw;
    }
  }

  void stop()
  {
    watcher_shutdown_.store(true, std::memory_order_relaxed);
    if (watcher_thread_.joinable())
      watcher_thread_.join();

    close_stream(/*fade=*/true);
    running = false;
    // This is a lock-free scheduler signal, so it can release an in-flight
    // control request even after a disconnect removed the last callback.
    set_source_realtime_running(false);
  }

  bool is_reconnecting() const
  {
    return device_disconnected_.load(std::memory_order_relaxed);
  }

  uint64_t disconnect_count() const
  {
    return disconnect_count_.load(std::memory_order_acquire);
  }

  uint64_t reconnect_success_count() const
  {
    return reconnect_success_count_.load(std::memory_order_acquire);
  }

  uint64_t reconnect_failure_count() const
  {
    return reconnect_failure_count_.load(std::memory_order_acquire);
  }

  struct Stats {
    uint64_t callback_count;
    double   avg_callback_ms;
    double   max_callback_ms;
    uint64_t underrun_count;
    uint64_t overrun_count;
  };

  Stats stats() const
  {
    const uint64_t n   = callback_count_.load(std::memory_order_relaxed);
    const uint64_t tot = total_callback_ns_.load(std::memory_order_relaxed);
    const uint64_t mx  = max_callback_ns_.load(std::memory_order_relaxed);
    Stats s;
    s.callback_count  = n;
    s.avg_callback_ms = n > 0 ? (tot / static_cast<double>(n)) / 1e6 : 0.0;
    s.max_callback_ms = mx / 1e6;
    s.underrun_count  = underrun_count_.load(std::memory_order_relaxed);
    s.overrun_count   = overrun_count_.load(std::memory_order_relaxed);
    return s;
  }

  uint64_t stats_epoch() const noexcept
  {
    return stats_epoch_.load(std::memory_order_acquire);
  }

  uint64_t reset_stats_epoch()
  {
    const unsigned int next_bank =
      1U - active_histogram_.load(std::memory_order_acquire);
    for (auto & bin : callback_histogram_[next_bank])
      bin.store(0, std::memory_order_relaxed);

    const uint64_t next_epoch =
      requested_stats_epoch_.load(std::memory_order_relaxed) + 1;
    requested_histogram_.store(next_bank, std::memory_order_relaxed);
    requested_stats_epoch_.store(next_epoch, std::memory_order_release);

    if (!running)
    {
      callback_count_.store(0, std::memory_order_relaxed);
      total_callback_ns_.store(0, std::memory_order_relaxed);
      max_callback_ns_.store(0, std::memory_order_relaxed);
      underrun_count_.store(0, std::memory_order_relaxed);
      overrun_count_.store(0, std::memory_order_relaxed);
      active_histogram_.store(next_bank, std::memory_order_release);
      stats_epoch_.store(next_epoch, std::memory_order_release);
      return next_epoch;
    }

    // The callback applies the epoch at its next boundary. Waiting here makes
    // the measured window exact without putting a lock on the audio thread.
    const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (stats_epoch_.load(std::memory_order_acquire) != next_epoch
           && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return stats_epoch_.load(std::memory_order_acquire) == next_epoch
      ? next_epoch : 0;
  }

  void reset_stats()
  {
    (void)reset_stats_epoch();
  }

  std::size_t copy_callback_histogram(
    uint64_t * out, std::size_t capacity, uint64_t & epoch) const
  {
    if (!out || capacity < kCallbackHistogramBins) return 0;
    const uint64_t before = stats_epoch_.load(std::memory_order_acquire);
    const uint64_t count_before =
      callback_count_.load(std::memory_order_acquire);
    const unsigned int bank =
      active_histogram_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < kCallbackHistogramBins; ++i)
      out[i] = callback_histogram_[bank][i].load(std::memory_order_relaxed);
    const uint64_t count_after =
      callback_count_.load(std::memory_order_acquire);
    const uint64_t after = stats_epoch_.load(std::memory_order_acquire);
    if (before != after || count_before != count_after) return 0;
    epoch = after;
    return kCallbackHistogramBins;
  }

  uint64_t request_output_capture()
  {
    CaptureState expected = CaptureState::Idle;
    if (!capture_state_.compare_exchange_strong(
          expected, CaptureState::Requested,
          std::memory_order_acq_rel, std::memory_order_acquire))
      return 0;
    const uint64_t sequence =
      capture_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    capture_requested_.store(sequence, std::memory_order_release);
    return sequence;
  }

  bool read_output_capture(uint64_t sequence, uint64_t & start_index,
                           double * out, std::size_t capacity)
  {
    if (sequence == 0 || !out || capacity < captured_output_.size()
        || capture_ready_.load(std::memory_order_acquire) != sequence)
      return false;
    CaptureState expected = CaptureState::Ready;
    if (!capture_state_.compare_exchange_strong(
          expected, CaptureState::Reading,
          std::memory_order_acq_rel, std::memory_order_acquire))
      return false;
    start_index = captured_start_index_.load(std::memory_order_relaxed);
    std::copy(captured_output_.begin(), captured_output_.end(), out);
    const bool valid =
      capture_ready_.load(std::memory_order_acquire) == sequence;
    capture_requested_.store(0, std::memory_order_relaxed);
    capture_state_.store(CaptureState::Idle, std::memory_order_release);
    return valid;
  }

  unsigned int negotiated_buffer_frames() const noexcept
  {
    return negotiated_buffer_frames_.load(std::memory_order_acquire);
  }

  // ── Master clock (multi-device A/V sync) ───────────────────────────────────
  // The authoritative "audible now" sample index a slave consumer (scope,
  // video) renders against: seconds-of-stream-played × rate, minus the output
  // buffering latency. Buffer-granular; returns 0 when the stream isn't
  // running. getStreamTime/getStreamLatency are RtAudio queries — reading them
  // off the audio thread is a benign stale-by-a-buffer race, fine for a clock.
  uint64_t playback_position()
  {
    if (!running) return 0;
    const double secs = audio.getStreamTime();
    long lat = audio.getStreamLatency();
    if (lat < 0) lat = 0;
    const double audible =
      secs * static_cast<double>(sample_rate) - static_cast<double>(lat);
    return audible > 0.0 ? static_cast<uint64_t>(audible) : 0;
  }

  // ---------- Device query ----------

  unsigned int active_device() const noexcept { return active_device_id_; }

  std::vector<unsigned int> device_ids() { return audio.getDeviceIds(); }

  RtAudio::DeviceInfo device_info(unsigned int id)
  {
    return audio.getDeviceInfo(id);
  }

  unsigned int default_output_device()
  {
    return audio.getDefaultOutputDevice();
  }

  // ---------- Device switching ----------

  bool switch_device(unsigned int device_id)
  {
    if (!running)
      return false;

    const auto info = audio.getDeviceInfo(device_id);
    if (info.outputChannels == 0)
      return false;

    watcher_shutdown_.store(true, std::memory_order_relaxed);
    if (watcher_thread_.joinable())
      watcher_thread_.join();

    try
    {
      if (audio.isStreamRunning())
        audio.abortStream();
      if (audio.isStreamOpen())
        audio.closeStream();
    }
    catch (...) {}

    bool ok = true;
    try
    {
      prepare_source_for_stream_restart();
      source->begin_fade_in();
      open_stream(device_id);
    }
    catch (...)
    {
      ok = false;
    }

    device_disconnected_.store(!ok, std::memory_order_relaxed);
    if (ok)
      rtaudio_disconnect_latched_.store(false, std::memory_order_release);
    watcher_shutdown_.store(false, std::memory_order_relaxed);
    watcher_thread_ = std::thread(&TropicalDACImpl::watcher_loop, this);

    return ok;
  }

  static int fill_buffer(
    void*               output_buffer,
    void*,
    unsigned int        n_buffer_frames,
    double,
    RtAudioStreamStatus status,
    void*               user_data)
  {
    auto* self = static_cast<TropicalDACImpl*>(user_data);

    const uint64_t requested_epoch =
      self->requested_stats_epoch_.load(std::memory_order_acquire);
    if (requested_epoch !=
        self->stats_epoch_.load(std::memory_order_relaxed))
    {
      self->callback_count_.store(0, std::memory_order_relaxed);
      self->total_callback_ns_.store(0, std::memory_order_relaxed);
      self->max_callback_ns_.store(0, std::memory_order_relaxed);
      self->underrun_count_.store(0, std::memory_order_relaxed);
      self->overrun_count_.store(0, std::memory_order_relaxed);
      self->active_histogram_.store(
        self->requested_histogram_.load(std::memory_order_relaxed),
        std::memory_order_release);
      self->stats_epoch_.store(requested_epoch, std::memory_order_release);
    }

    const auto t0 = std::chrono::steady_clock::now();

    if (status)
      self->underrun_count_.fetch_add(1, std::memory_order_relaxed);

    self->source->process();
    const auto& buf = self->source->outputBuffer;

    if (self->capture_state_.load(std::memory_order_acquire)
        == CaptureState::Requested)
    {
      const uint64_t capture =
        self->capture_requested_.load(std::memory_order_acquire);
      if (capture != 0)
      {
        const std::size_t n =
          std::min<std::size_t>(buf.size(), self->captured_output_.size());
        std::copy_n(buf.begin(), n, self->captured_output_.begin());
        std::fill(self->captured_output_.begin() + n,
                  self->captured_output_.end(), 0.0);
        if constexpr (requires { self->source->current_sample_index(); })
        {
          const uint64_t next = self->source->current_sample_index();
          self->captured_start_index_.store(
            next >= self->captured_output_.size()
              ? next - self->captured_output_.size() : 0,
            std::memory_order_relaxed);
        }
        self->capture_ready_.store(capture, std::memory_order_release);
        self->capture_state_.store(
          CaptureState::Ready, std::memory_order_release);
      }
    }

    auto* out = static_cast<double*>(output_buffer);
    unsigned int source_channels = 1;
    const double * interleaved = buf.data();
    if constexpr (requires {
                    self->source->getRenderedOutputChannelCount();
                    self->source->getRenderedInterleavedOutputBuffer();
                  })
    {
      source_channels = std::max(
        1u, static_cast<unsigned int>(
          self->source->getRenderedOutputChannelCount()));
      interleaved = self->source->getRenderedInterleavedOutputBuffer();
    }
    else if constexpr (requires {
                    self->source->getOutputChannelCount();
                    self->source->getInterleavedOutputBuffer();
                  })
    {
      source_channels = std::max(
        1u, static_cast<unsigned int>(
          self->source->getOutputChannelCount()));
      interleaved = self->source->getInterleavedOutputBuffer();
    }

    // A multichannel plan cannot be folded down implicitly. Startup rejects a
    // too-narrow device; this callback guard covers a plan hot-swap that races
    // an already-open stream by failing safely to silence.
    if (source_channels > 1 && source_channels > self->channels)
    {
      std::fill_n(out,
        static_cast<std::size_t>(n_buffer_frames) * self->channels, 0.0);
    }
    else for (unsigned int i = 0; i < n_buffer_frames; ++i)
    {
      for (unsigned int c = 0; c < self->channels; ++c)
      {
        // Mono sources upmix to every device channel. Multichannel sources map
        // independently; surplus device channels are explicitly silent.
        const unsigned int source_channel = source_channels == 1 ? 0 : c;
        const bool in_range = i < buf.size()
                           && source_channel < source_channels
                           && interleaved != nullptr;
        const double sample = in_range
          ? interleaved[static_cast<std::size_t>(i) * source_channels
                        + source_channel]
          : 0.0;
        // The device-boundary clamp — the last thing before the speaker,
        // applied AFTER the runtime fade envelope.
        *out++ = clamp_to_device_bound(sample);
      }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    self->total_callback_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
    update_max(self->max_callback_ns_, elapsed_ns);
    const std::size_t histogram_bin =
      std::min<std::size_t>(
        elapsed_ns / kCallbackHistogramBinNs,
        kCallbackHistogramRegularBins);
    const unsigned int histogram =
      self->active_histogram_.load(std::memory_order_relaxed);
    self->callback_histogram_[histogram][histogram_bin].fetch_add(
      1, std::memory_order_relaxed);

    const uint64_t budget_ns =
        (static_cast<uint64_t>(n_buffer_frames) * 1000000000ULL) / self->sample_rate;
    if (elapsed_ns > budget_ns)
      self->overrun_count_.fetch_add(1, std::memory_order_relaxed);
    // Publish callback completion last so readers that acquire the count can
    // take a self-consistent histogram/counter snapshot.
    self->callback_count_.fetch_add(1, std::memory_order_release);

    return 0;
  }

private:
  void open_stream()
  {
    open_stream(audio.getDefaultOutputDevice());
  }

  void open_stream(unsigned int device_id)
  {
    if constexpr (requires { source->getOutputChannelCount(); })
    {
      const unsigned int source_channels = std::max(
        1u, static_cast<unsigned int>(source->getOutputChannelCount()));
      if (source_channels > 1 && source_channels > channels)
        throw std::runtime_error(
          "Tropical multichannel plan requires "
          + std::to_string(source_channels)
          + " device output channels, but the stream requested "
          + std::to_string(channels));
    }
    if (audio.getDeviceCount() < 1)
      throw std::runtime_error("No audio output devices found.");

    RtAudio::StreamParameters out_params;
    active_device_id_       = device_id;
    out_params.deviceId     = active_device_id_;
    out_params.nChannels    = channels;
    out_params.firstChannel = 0;

    unsigned int buffer_frames = source->getBufferLength();

    audio.openStream(
      &out_params,
      nullptr,
      RTAUDIO_FLOAT64,
      sample_rate,
      &buffer_frames,
      &TropicalDACImpl::fill_buffer,
      this);

    negotiated_buffer_frames_.store(buffer_frames, std::memory_order_release);
    if (buffer_frames != source->getBufferLength())
    {
      audio.closeStream();
      throw std::runtime_error(
        "RtAudio negotiated " + std::to_string(buffer_frames)
        + " frames, but Tropical runtime requires "
        + std::to_string(source->getBufferLength()));
    }
    audio.startStream();
  }

  void close_stream(bool fade)
  {
    if (!audio.isStreamOpen())
      return;
    if (audio.isStreamRunning())
    {
      if (fade)
      {
        source->begin_fade_out();
        const int buf_ms = (static_cast<int>(source->getBufferLength()) * 1000)
                           / static_cast<int>(sample_rate);
        const int timeout_ms = (2048 * 1000) / static_cast<int>(sample_rate)
                               + 4 * buf_ms + 50;
        const auto deadline = std::chrono::steady_clock::now()
                              + std::chrono::milliseconds(timeout_ms);
        while (!source->is_fade_out_complete()
               && std::chrono::steady_clock::now() < deadline)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(buf_ms + 5));
      }
      audio.stopStream();
    }
    audio.closeStream();
  }

  void watcher_loop()
  {
    while (!watcher_shutdown_.load(std::memory_order_relaxed))
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));

      const bool disconnected = device_disconnected_.load(std::memory_order_relaxed);
      const bool default_changed = !disconnected
                                   && audio.isStreamOpen()
                                   && audio.getDefaultOutputDevice() != active_device_id_;

      if (!disconnected && !default_changed)
        continue;

      try
      {
        if (audio.isStreamRunning())
          audio.abortStream();
        if (audio.isStreamOpen())
          audio.closeStream();
      }
      catch (...) {}

      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      if (watcher_shutdown_.load(std::memory_order_relaxed))
        break;

      device_disconnected_.store(false, std::memory_order_relaxed);

      try
      {
        prepare_source_for_stream_restart();
        source->begin_fade_in();
        open_stream();
        rtaudio_disconnect_latched_.store(false, std::memory_order_release);
        reconnect_success_count_.fetch_add(1, std::memory_order_relaxed);
      }
      catch (...)
      {
        reconnect_failure_count_.fetch_add(1, std::memory_order_relaxed);
        device_disconnected_.store(true, std::memory_order_relaxed);
      }
    }
  }
};
