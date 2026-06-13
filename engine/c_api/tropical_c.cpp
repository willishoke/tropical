#include "c_api/tropical_c.h"

#include "runtime/FlatRuntime.hpp"
#include "runtime/NumericProgramParser.hpp"
#include "dac/TropicalDAC.hpp"

#include <llvm/Support/Error.h>

#include <algorithm>
#include <cstdlib>
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

namespace expr = tropical_expr;

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

bool tropical_runtime_load_ir(tropical_runtime_t r, const char* ir_text, size_t ir_len,
                              const char* manifest_json, size_t manifest_len)
{
  if (!r || !ir_text || !manifest_json) return false;
  try
  {
    return static_cast<tropical_runtime::FlatRuntime*>(r)->load_ir(
      std::string(ir_text, ir_len), std::string(manifest_json, manifest_len));
  }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

char* tropical_compile_plan_to_ir(const char* plan_json, size_t len)
{
  if (!plan_json) return nullptr;
  try
  {
    const nlohmann::json plan = nlohmann::json::parse(std::string(plan_json, len));
    const std::string schema = plan.value("schema", std::string{});
    tropical_plan5::ParsedPlan5 parsed;
    if (schema == "tropical_plan_5")
      parsed = tropical_plan5::parse_plan5(plan);
    else if (schema == "tropical_plan_4")
      parsed = tropical_plan5::parse_plan4(plan);
    else { set_error("compile_plan_to_ir: unsupported schema '" + schema + "'"); return nullptr; }

    std::string ir;
    // emit_only returns a null kernel on success (the module is printed,
    // not JIT'd), so check the Expected's error state, not the pointer.
    auto kernel = tropical_jit::OrcJitEngine::instance().compile_flat_program(
      parsed.program, &ir, /*emit_only=*/true);
    if (auto e = kernel.takeError())
    {
      std::string err;
      llvm::handleAllErrors(std::move(e),
        [&err](const llvm::ErrorInfoBase & b) { err = b.message(); });
      set_error("compile_plan_to_ir: " + err);
      return nullptr;
    }

    char* out = static_cast<char*>(std::malloc(ir.size() + 1));
    if (!out) { set_error("compile_plan_to_ir: out of memory"); return nullptr; }
    std::memcpy(out, ir.c_str(), ir.size() + 1);
    return out;
  }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
}

void tropical_free_string(char* s) { std::free(s); }

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

// ── Slot model (M5+) ────────────────────────────────────────────────────────

unsigned int tropical_runtime_slot_index(tropical_runtime_t r, const char* name)
{
  if (!r || !name) return UINT32_MAX;
  return static_cast<tropical_runtime::FlatRuntime*>(r)->slot_index(std::string(name));
}

void tropical_runtime_set_slot(tropical_runtime_t r, unsigned int slot_index, double value)
{
  if (!r) return;
  static_cast<tropical_runtime::FlatRuntime*>(r)->set_slot(slot_index, value);
}

double tropical_runtime_get_slot(tropical_runtime_t r, unsigned int slot_index)
{
  if (!r) return 0.0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)->get_slot(slot_index);
}

} // extern "C"
