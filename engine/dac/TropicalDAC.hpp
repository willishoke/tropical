#pragma once

#include "../lib/rtaudio/RtAudio.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

static constexpr int kPrimeCycles = 4;

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
 * MEASURED at landing (2026-07-20): peak |sample| over the whole patch corpus —
 * `patches/{reverse_reverb,scrub_reverb}` and `web/patches/{pure-sine-440,
 * ring-mod,tz-flanger}`, 16384 samples each — is 1.807, so C = 4 is the
 * smallest power of two comfortably above it, at 2.2× headroom. For scale: the
 * i64 modal-datapath rail incident (design/modal-datapath-rail.local.md)
 * produced peaks of 63.3, 16× past this bound. The clamp is what protects the
 * interval between "this rail is fixed" and "the next rail is discovered."
 *
 * NaN maps to −C: the first comparison is false on NaN, so the low arm takes
 * −C, which then passes the high arm. Ordered-compare + select rather than
 * std::fmin/fmax, whose IEEE minNum semantics return the *other* operand on NaN
 * and leave signed zero unspecified. Exact for every non-NaN value in [−C, C],
 * including −0.0 and denormals.
 */
static constexpr double kDeviceOutputBound = 4.0;

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

  std::atomic<bool>     device_disconnected_{false};
  std::atomic<bool>     watcher_shutdown_{false};
  std::thread           watcher_thread_;
  unsigned int          active_device_id_{0};

  TropicalDACImpl(AudioSource* s, unsigned int sr, unsigned int ch)
    : source(s), sample_rate(sr), channels(ch)
  {
    audio.setErrorCallback([this](RtAudioErrorType type, const std::string&) {
      if (type == RTAUDIO_DEVICE_DISCONNECT)
        device_disconnected_.store(true, std::memory_order_relaxed);
    });
  }

  ~TropicalDACImpl() { stop(); }

  void start()
  {
    if (running)
      return;

    if constexpr (requires { source->prime_numeric_jit(); })
      source->prime_numeric_jit();
    for (int i = 0; i < kPrimeCycles; ++i)
      source->process();

    source->begin_fade_in();

    callback_count_.store(0, std::memory_order_relaxed);
    total_callback_ns_.store(0, std::memory_order_relaxed);
    max_callback_ns_.store(0, std::memory_order_relaxed);
    underrun_count_.store(0, std::memory_order_relaxed);
    overrun_count_.store(0, std::memory_order_relaxed);

    open_stream();
    running = true;

    device_disconnected_.store(false, std::memory_order_relaxed);
    watcher_shutdown_.store(false, std::memory_order_relaxed);
    watcher_thread_ = std::thread(&TropicalDACImpl::watcher_loop, this);
  }

  void stop()
  {
    watcher_shutdown_.store(true, std::memory_order_relaxed);
    if (watcher_thread_.joinable())
      watcher_thread_.join();

    close_stream(/*fade=*/true);
    running = false;
  }

  bool is_reconnecting() const
  {
    return device_disconnected_.load(std::memory_order_relaxed);
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

  void reset_stats()
  {
    callback_count_.store(0, std::memory_order_relaxed);
    total_callback_ns_.store(0, std::memory_order_relaxed);
    max_callback_ns_.store(0, std::memory_order_relaxed);
    underrun_count_.store(0, std::memory_order_relaxed);
    overrun_count_.store(0, std::memory_order_relaxed);
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
      source->begin_fade_in();
      open_stream(device_id);
    }
    catch (...)
    {
      ok = false;
    }

    device_disconnected_.store(false, std::memory_order_relaxed);
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
    const auto t0 = std::chrono::steady_clock::now();

    auto* self = static_cast<TropicalDACImpl*>(user_data);

    if (status)
      self->underrun_count_.fetch_add(1, std::memory_order_relaxed);

    self->source->process();
    const auto& buf = self->source->outputBuffer;
    auto* out = static_cast<double*>(output_buffer);

    for (unsigned int i = 0; i < n_buffer_frames; ++i)
    {
      // The device-boundary clamp — the last thing before the speaker, applied
      // AFTER the runtime's fade envelope (which is in [0,1] and so can never
      // defeat it). See kDeviceOutputBound.
      const double sample =
          clamp_to_device_bound(i < buf.size() ? buf[i] : 0.0);
      for (unsigned int c = 0; c < self->channels; ++c)
        *out++ = sample;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    self->callback_count_.fetch_add(1, std::memory_order_relaxed);
    self->total_callback_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
    update_max(self->max_callback_ns_, elapsed_ns);

    const uint64_t budget_ns =
        (static_cast<uint64_t>(n_buffer_frames) * 1000000000ULL) / self->sample_rate;
    if (elapsed_ns > budget_ns)
      self->overrun_count_.fetch_add(1, std::memory_order_relaxed);

    return 0;
  }

private:
  void open_stream()
  {
    open_stream(audio.getDefaultOutputDevice());
  }

  void open_stream(unsigned int device_id)
  {
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
        source->begin_fade_in();
        open_stream();
      }
      catch (...)
      {
        device_disconnected_.store(true, std::memory_order_relaxed);
      }
    }
  }
};
