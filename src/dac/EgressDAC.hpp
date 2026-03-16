#pragma once

#include "graph/Graph.hpp"
#include "../lib/rtaudio/RtAudio.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

static constexpr int kPrimeCycles = 4;

static inline void update_max(std::atomic<uint64_t>& cur, uint64_t val)
{
  uint64_t prev = cur.load(std::memory_order_relaxed);
  while (val > prev && !cur.compare_exchange_weak(prev, val,
      std::memory_order_relaxed, std::memory_order_relaxed))
  {}
}

struct EgressDAC
{
  Graph*       graph;
  RtAudio      audio;
  unsigned int sample_rate;
  unsigned int channels;
  bool         running = false;

  std::atomic<uint64_t> callback_count_{0};
  std::atomic<uint64_t> total_callback_ns_{0};
  std::atomic<uint64_t> max_callback_ns_{0};
  std::atomic<uint64_t> underrun_count_{0};
  std::atomic<uint64_t> overrun_count_{0};

  EgressDAC(Graph* g, unsigned int sr, unsigned int ch)
    : graph(g), sample_rate(sr), channels(ch)
  {
  }

  ~EgressDAC() { stop(); }

  void start()
  {
    if (running)
      return;
    if (audio.getDeviceCount() < 1)
      throw std::runtime_error("No audio output devices found.");

    graph->prime_numeric_jit();
    for (int i = 0; i < kPrimeCycles; ++i)
      graph->process();

    graph->begin_fade_in();

    callback_count_.store(0, std::memory_order_relaxed);
    total_callback_ns_.store(0, std::memory_order_relaxed);
    max_callback_ns_.store(0, std::memory_order_relaxed);
    underrun_count_.store(0, std::memory_order_relaxed);
    overrun_count_.store(0, std::memory_order_relaxed);

    RtAudio::StreamParameters out_params;
    out_params.deviceId     = audio.getDefaultOutputDevice();
    out_params.nChannels    = channels;
    out_params.firstChannel = 0;

    unsigned int buffer_frames = graph->getBufferLength();

    audio.openStream(
      &out_params,
      nullptr,
      RTAUDIO_FLOAT64,
      sample_rate,
      &buffer_frames,
      &EgressDAC::fill_buffer,
      this);

    audio.startStream();
    running = true;
  }

  void stop()
  {
    if (!audio.isStreamOpen())
    {
      running = false;
      return;
    }
    if (audio.isStreamRunning())
    {
      graph->begin_fade_out();

      const int buf_ms = (static_cast<int>(graph->getBufferLength()) * 1000)
                         / static_cast<int>(sample_rate);
      const int timeout_ms = (2048 * 1000) / static_cast<int>(sample_rate)
                             + 4 * buf_ms + 50;
      const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::milliseconds(timeout_ms);
      while (!graph->is_fade_out_complete()
             && std::chrono::steady_clock::now() < deadline)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(buf_ms + 5));
      audio.stopStream();
    }
    audio.closeStream();
    running = false;
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

  static int fill_buffer(
    void*               output_buffer,
    void*,
    unsigned int        n_buffer_frames,
    double,
    RtAudioStreamStatus status,
    void*               user_data)
  {
    const auto t0 = std::chrono::steady_clock::now();

    auto* self = static_cast<EgressDAC*>(user_data);

    if (status)
      self->underrun_count_.fetch_add(1, std::memory_order_relaxed);

    self->graph->process();
    const auto& source = self->graph->outputBuffer;
    auto* out = static_cast<double*>(output_buffer);

    for (unsigned int i = 0; i < n_buffer_frames; ++i)
    {
      const double sample = i < source.size() ? source[i] : 0.0;
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
};
