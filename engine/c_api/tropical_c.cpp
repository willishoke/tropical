#include "c_api/tropical_c.h"

#include "runtime/FlatRuntime.hpp"
#include "dac/TropicalDAC.hpp"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ---------- Thread-local error string ----------

static thread_local std::string tls_last_error;

static void set_error(const std::string & msg) { tls_last_error = msg; }

extern "C" const char* tropical_last_error(void)
{
  return tls_last_error.c_str();
}

// ---------- DAC wrapper (RuntimeDAC only) ----------

using RuntimeDAC = TropicalDACImpl<tropical_runtime::FlatRuntime>;

// ---------- Opaque wrapper types ----------

struct TropicalParam
{
  expr::ControlParam * param;
  TropicalParam(double init, double tc) : param(new expr::ControlParam(init, tc)) {}
  ~TropicalParam() { delete param; }
  // Non-copyable
  TropicalParam(const TropicalParam &) = delete;
  TropicalParam & operator=(const TropicalParam &) = delete;
};

// ============================================================
// C API implementation
// ============================================================

extern "C" {

// ---------- ControlParam API ----------

tropical_param_t tropical_param_new(double init_value, double time_const)
{
  try { return new TropicalParam(init_value, time_const); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void tropical_param_free(tropical_param_t p)
{
  delete static_cast<TropicalParam *>(p);
}

void tropical_param_set(tropical_param_t p, double value)
{
  if (p)
  {
    static_cast<TropicalParam *>(p)->param->value.store(value, std::memory_order_relaxed);
  }
}

double tropical_param_get(tropical_param_t p)
{
  if (!p) return 0.0;
  return static_cast<TropicalParam *>(p)->param->value.load(std::memory_order_relaxed);
}

tropical_param_t tropical_param_new_trigger(void)
{
  try { return new TropicalParam(0.0, 0.0); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

// ---------- Device enumeration ----------

unsigned int tropical_audio_device_count(void)
{
  RtAudio tmp;
  return tmp.getDeviceCount();
}

void tropical_audio_get_device_ids(unsigned int* out, unsigned int count)
{
  if (!out) return;
  RtAudio tmp;
  const auto ids = tmp.getDeviceIds();
  const unsigned int n = std::min(static_cast<unsigned int>(ids.size()), count);
  for (unsigned int i = 0; i < n; ++i)
    out[i] = ids[i];
}

bool tropical_audio_get_device_info(unsigned int device_id, tropical_device_info_t* out)
{
  if (!out) return false;
  try
  {
    RtAudio tmp;
    const RtAudio::DeviceInfo info = tmp.getDeviceInfo(device_id);
    out->id                  = info.ID;
    std::strncpy(out->name, info.name.c_str(), sizeof(out->name) - 1);
    out->name[sizeof(out->name) - 1] = '\0';
    out->output_channels     = info.outputChannels;
    out->input_channels      = info.inputChannels;
    out->is_default_output   = info.isDefaultOutput;
    out->preferred_sample_rate = info.preferredSampleRate;
    const unsigned int n = std::min(static_cast<unsigned int>(info.sampleRates.size()),
                                    static_cast<unsigned int>(32));
    out->sample_rate_count = n;
    for (unsigned int i = 0; i < n; ++i)
      out->sample_rates[i] = info.sampleRates[i];
    return true;
  }
  catch (...) { return false; }
}

unsigned int tropical_audio_default_output_device(void)
{
  RtAudio tmp;
  return tmp.getDefaultOutputDevice();
}

// ---------- DAC API ----------

tropical_dac_t tropical_dac_new_runtime(tropical_runtime_t r, unsigned int sample_rate, unsigned int channels)
{
  try {
    auto * rt = static_cast<tropical_runtime::FlatRuntime*>(r);
    auto * dac = new RuntimeDAC(rt, sample_rate, channels);
    return static_cast<void*>(dac);
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void tropical_dac_free(tropical_dac_t d)
{
  delete static_cast<RuntimeDAC*>(d);
}

void tropical_dac_start(tropical_dac_t d)
{
  try { static_cast<RuntimeDAC*>(d)->start(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

void tropical_dac_stop(tropical_dac_t d)
{
  try { static_cast<RuntimeDAC*>(d)->stop(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

bool tropical_dac_is_running(tropical_dac_t d)
{
  return static_cast<RuntimeDAC*>(d)->running;
}

void tropical_dac_get_stats(tropical_dac_t d, tropical_dac_stats_t* out)
{
  if (!d || !out) return;
  const auto s = static_cast<RuntimeDAC*>(d)->stats();
  out->callback_count  = s.callback_count;
  out->avg_callback_ms = s.avg_callback_ms;
  out->max_callback_ms = s.max_callback_ms;
  out->underrun_count  = s.underrun_count;
  out->overrun_count   = s.overrun_count;
}

void tropical_dac_reset_stats(tropical_dac_t d)
{
  if (d) static_cast<RuntimeDAC*>(d)->reset_stats();
}

bool tropical_dac_is_reconnecting(tropical_dac_t d)
{
  return d && static_cast<RuntimeDAC*>(d)->is_reconnecting();
}

unsigned int tropical_dac_get_active_device(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->active_device() : 0;
}

bool tropical_dac_switch_device(tropical_dac_t d, unsigned int device_id)
{
  if (!d) return false;
  try {
    return static_cast<RuntimeDAC*>(d)->switch_device(device_id);
  }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

// ---------- FlatRuntime API ----------

tropical_runtime_t tropical_runtime_new(unsigned int buffer_length)
{
  try { return new tropical_runtime::FlatRuntime(buffer_length); }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
}

void tropical_runtime_free(tropical_runtime_t r)
{
  delete static_cast<tropical_runtime::FlatRuntime*>(r);
}

bool tropical_runtime_load_plan(tropical_runtime_t r, const char* plan_json, size_t len)
{
  if (!r || !plan_json) return false;
  try
  {
    return static_cast<tropical_runtime::FlatRuntime*>(r)->load_plan(std::string(plan_json, len));
  }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

void tropical_runtime_process(tropical_runtime_t r)
{
  if (r) static_cast<tropical_runtime::FlatRuntime*>(r)->process();
}

const double* tropical_runtime_output_buffer(tropical_runtime_t r)
{
  if (!r) return nullptr;
  return static_cast<tropical_runtime::FlatRuntime*>(r)->outputBuffer.data();
}

unsigned int tropical_runtime_get_buffer_length(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)->getBufferLength();
}

void tropical_runtime_begin_fade_in(tropical_runtime_t r)
{
  if (r) static_cast<tropical_runtime::FlatRuntime*>(r)->begin_fade_in();
}

void tropical_runtime_begin_fade_out(tropical_runtime_t r)
{
  if (r) static_cast<tropical_runtime::FlatRuntime*>(r)->begin_fade_out();
}

bool tropical_runtime_is_fade_out_complete(tropical_runtime_t r)
{
  if (!r) return true;
  return static_cast<tropical_runtime::FlatRuntime*>(r)->is_fade_out_complete();
}

} // extern "C"
