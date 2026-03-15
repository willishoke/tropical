#pragma once

class PythonDAC
{
  public:
    explicit PythonDAC(unsigned int sample_rate = 44100, unsigned int channels = 2)
      : PythonDAC(default_graph(), sample_rate, channels)
    {
    }

  private:
    explicit PythonDAC(PythonGraph & graph, unsigned int sample_rate, unsigned int channels)
      : graph_(graph), sample_rate_(sample_rate), channels_(channels), running_(false)
    {
    }

  public:
    ~PythonDAC()
    {
      stop();
    }

    void start()
    {
      if (running_)
      {
        return;
      }

      if (audio_.getDeviceCount() < 1)
      {
        throw std::runtime_error("No audio output devices found.");
      }

      RtAudio::StreamParameters out_params;
      out_params.deviceId = audio_.getDefaultOutputDevice();
      out_params.nChannels = channels_;
      out_params.firstChannel = 0;

      graph_.graph().prime_numeric_jit();
      unsigned int buffer_frames = graph_.graph().getBufferLength();

      audio_.openStream(
        &out_params,
        nullptr,
        RTAUDIO_FLOAT64,
        sample_rate_,
        &buffer_frames,
        &PythonDAC::fill_buffer,
        this);

      audio_.startStream();
      running_ = true;
    }

    void stop()
    {
      if (!audio_.isStreamOpen())
      {
        running_ = false;
        return;
      }

      if (audio_.isStreamRunning())
      {
        audio_.stopStream();
      }
      audio_.closeStream();
      running_ = false;
    }

    bool is_running() const
    {
      return running_;
    }

    py::dict callback_timing_stats() const
    {
#ifdef EGRESS_PROFILE
      const uint64_t callbacks = callback_count_.load(std::memory_order_relaxed);
      const uint64_t total_ns = total_callback_ns_.load(std::memory_order_relaxed);
      const uint64_t max_ns = max_callback_ns_.load(std::memory_order_relaxed);
      const uint64_t overruns = overrun_count_.load(std::memory_order_relaxed);

      py::dict stats;
      stats["enabled"] = true;
      stats["callback_count"] = callbacks;
      stats["avg_callback_ms"] = callbacks == 0 ? 0.0 : (static_cast<double>(total_ns) / static_cast<double>(callbacks)) / 1e6;
      stats["max_callback_ms"] = static_cast<double>(max_ns) / 1e6;
      stats["overrun_count"] = overruns;
      return stats;
#else
      py::dict stats;
      stats["enabled"] = false;
      stats["callback_count"] = 0;
      stats["avg_callback_ms"] = 0.0;
      stats["max_callback_ms"] = 0.0;
      stats["overrun_count"] = 0;
      return stats;
#endif
    }

    void reset_callback_timing_stats()
    {
#ifdef EGRESS_PROFILE
      callback_count_.store(0, std::memory_order_relaxed);
      total_callback_ns_.store(0, std::memory_order_relaxed);
      max_callback_ns_.store(0, std::memory_order_relaxed);
      overrun_count_.store(0, std::memory_order_relaxed);
#endif
    }

  private:
#ifdef EGRESS_PROFILE
    static void update_max_callback_ns(std::atomic<uint64_t> & dst, uint64_t candidate)
    {
      uint64_t current = dst.load(std::memory_order_relaxed);
      while (current < candidate && !dst.compare_exchange_weak(current, candidate, std::memory_order_relaxed))
      {
      }
    }
#endif

    static int fill_buffer(
      void * output_buffer,
      void *,
      unsigned int n_buffer_frames,
      double,
      RtAudioStreamStatus status,
      void * user_data)
    {
      auto * self = static_cast<PythonDAC *>(user_data);
      auto * out = static_cast<double *>(output_buffer);

    #ifdef EGRESS_PROFILE
      const auto callback_start = std::chrono::steady_clock::now();
    #endif

      self->graph_.graph().process();
      const auto & source = self->graph_.graph().outputBuffer;

      for (unsigned int i = 0; i < n_buffer_frames; ++i)
      {
        const double sample = i < source.size() ? source[i] : 0.0;
        for (unsigned int c = 0; c < self->channels_; ++c)
        {
          *out++ = sample;
        }
      }

#ifdef EGRESS_PROFILE
      const auto callback_end = std::chrono::steady_clock::now();
      const uint64_t elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(callback_end - callback_start).count());

      self->callback_count_.fetch_add(1, std::memory_order_relaxed);
      self->total_callback_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
      update_max_callback_ns(self->max_callback_ns_, elapsed_ns);

      const uint64_t budget_ns = static_cast<uint64_t>(
        (static_cast<double>(n_buffer_frames) * 1e9) / static_cast<double>(self->sample_rate_));
      if (elapsed_ns > budget_ns || status != 0)
      {
        self->overrun_count_.fetch_add(1, std::memory_order_relaxed);
      }
#else
      (void)status;
#endif

      return 0;
    }

    PythonGraph & graph_;
    RtAudio audio_;
    unsigned int sample_rate_;
    unsigned int channels_;
    bool running_;
  #ifdef EGRESS_PROFILE
    std::atomic<uint64_t> callback_count_{0};
    std::atomic<uint64_t> total_callback_ns_{0};
    std::atomic<uint64_t> max_callback_ns_{0};
    std::atomic<uint64_t> overrun_count_{0};
  #endif
};
