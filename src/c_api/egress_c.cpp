#include "c_api/egress_c.h"

#include "runtime/FlatRuntime.hpp"
#include "dac/EgressDAC.hpp"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ---------- Thread-local error string ----------

static thread_local std::string tls_last_error;

static void set_error(const std::string & msg) { tls_last_error = msg; }

extern "C" const char* egress_last_error(void)
{
  return tls_last_error.c_str();
}

// ---------- DAC wrapper (RuntimeDAC only) ----------

using RuntimeDAC = EgressDACImpl<egress_runtime::FlatRuntime>;

// ---------- Opaque wrapper types ----------

struct EgressParam
{
  expr::ControlParam * param;
  EgressParam(double init, double tc) : param(new expr::ControlParam(init, tc)) {}
  ~EgressParam() { delete param; }
  // Non-copyable
  EgressParam(const EgressParam &) = delete;
  EgressParam & operator=(const EgressParam &) = delete;
};

// ============================================================
// C API implementation
// ============================================================

extern "C" {

// ---------- ControlParam API ----------

egress_param_t egress_param_new(double init_value, double time_const)
{
  try { return new EgressParam(init_value, time_const); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void egress_param_free(egress_param_t p)
{
  delete static_cast<EgressParam *>(p);
}

void egress_param_set(egress_param_t p, double value)
{
  if (p)
  {
    static_cast<EgressParam *>(p)->param->value.store(value, std::memory_order_relaxed);
  }
}

double egress_param_get(egress_param_t p)
{
  if (!p) return 0.0;
  return static_cast<EgressParam *>(p)->param->value.load(std::memory_order_relaxed);
}

egress_param_t egress_param_new_trigger(void)
{
  try { return new EgressParam(0.0, 0.0); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

// ---------- Device enumeration ----------

unsigned int egress_audio_device_count(void)
{
  RtAudio tmp;
  return tmp.getDeviceCount();
}

void egress_audio_get_device_ids(unsigned int* out, unsigned int count)
{
  if (!out) return;
  RtAudio tmp;
  const auto ids = tmp.getDeviceIds();
  const unsigned int n = std::min(static_cast<unsigned int>(ids.size()), count);
  for (unsigned int i = 0; i < n; ++i)
    out[i] = ids[i];
}

bool egress_audio_get_device_info(unsigned int device_id, egress_device_info_t* out)
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

unsigned int egress_audio_default_output_device(void)
{
  RtAudio tmp;
  return tmp.getDefaultOutputDevice();
}

// ---------- DAC API ----------

egress_dac_t egress_dac_new_runtime(egress_runtime_t r, unsigned int sample_rate, unsigned int channels)
{
  try {
    auto * rt = static_cast<egress_runtime::FlatRuntime*>(r);
    auto * dac = new RuntimeDAC(rt, sample_rate, channels);
    return static_cast<void*>(dac);
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void egress_dac_free(egress_dac_t d)
{
  delete static_cast<RuntimeDAC*>(d);
}

void egress_dac_start(egress_dac_t d)
{
  try { static_cast<RuntimeDAC*>(d)->start(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

void egress_dac_stop(egress_dac_t d)
{
  try { static_cast<RuntimeDAC*>(d)->stop(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

bool egress_dac_is_running(egress_dac_t d)
{
  return static_cast<RuntimeDAC*>(d)->running;
}

void egress_dac_get_stats(egress_dac_t d, egress_dac_stats_t* out)
{
  if (!d || !out) return;
  const auto s = static_cast<RuntimeDAC*>(d)->stats();
  out->callback_count  = s.callback_count;
  out->avg_callback_ms = s.avg_callback_ms;
  out->max_callback_ms = s.max_callback_ms;
  out->underrun_count  = s.underrun_count;
  out->overrun_count   = s.overrun_count;
}

void egress_dac_reset_stats(egress_dac_t d)
{
  if (d) static_cast<RuntimeDAC*>(d)->reset_stats();
}

bool egress_dac_is_reconnecting(egress_dac_t d)
{
  return d && static_cast<RuntimeDAC*>(d)->is_reconnecting();
}

unsigned int egress_dac_get_active_device(egress_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->active_device() : 0;
}

bool egress_dac_switch_device(egress_dac_t d, unsigned int device_id)
{
  if (!d) return false;
  try {
    return static_cast<RuntimeDAC*>(d)->switch_device(device_id);
  }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

// ---------- FlatRuntime API ----------

egress_runtime_t egress_runtime_new(unsigned int buffer_length)
{
  try { return new egress_runtime::FlatRuntime(buffer_length); }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
}

void egress_runtime_free(egress_runtime_t r)
{
  delete static_cast<egress_runtime::FlatRuntime*>(r);
}

bool egress_runtime_load_plan(egress_runtime_t r, const char* plan_json, size_t len)
{
  if (!r || !plan_json) return false;
  try
  {
    return static_cast<egress_runtime::FlatRuntime*>(r)->load_plan(std::string(plan_json, len));
  }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

void egress_runtime_process(egress_runtime_t r)
{
  if (r) static_cast<egress_runtime::FlatRuntime*>(r)->process();
}

const double* egress_runtime_output_buffer(egress_runtime_t r)
{
  if (!r) return nullptr;
  return static_cast<egress_runtime::FlatRuntime*>(r)->outputBuffer.data();
}

unsigned int egress_runtime_get_buffer_length(egress_runtime_t r)
{
  if (!r) return 0;
  return static_cast<egress_runtime::FlatRuntime*>(r)->getBufferLength();
}

void egress_runtime_begin_fade_in(egress_runtime_t r)
{
  if (r) static_cast<egress_runtime::FlatRuntime*>(r)->begin_fade_in();
}

void egress_runtime_begin_fade_out(egress_runtime_t r)
{
  if (r) static_cast<egress_runtime::FlatRuntime*>(r)->begin_fade_out();
}

bool egress_runtime_is_fade_out_complete(egress_runtime_t r)
{
  if (!r) return true;
  return static_cast<egress_runtime::FlatRuntime*>(r)->is_fade_out_complete();
}

} // extern "C"
