#include "c_api/tropical_c.h"

#include "c_api/tropical_socket.hpp"
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

#ifdef TROPICAL_WASM_EMIT
namespace tropical_jit { std::vector<uint8_t> compile_ir_to_wasm(const std::string &, std::string &); }
#endif

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

uint64_t tropical_dac_reset_stats_epoch(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->reset_stats_epoch() : 0;
}

uint64_t tropical_dac_get_stats_epoch(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->stats_epoch() : 0;
}

size_t tropical_dac_callback_histogram_bin_count(void)
{
  return kCallbackHistogramBins;
}

uint64_t tropical_dac_callback_histogram_bin_width_ns(void)
{
  return kCallbackHistogramBinNs;
}

uint64_t tropical_dac_callback_histogram_overflow_floor_ns(void)
{
  return kCallbackHistogramRegularBins * kCallbackHistogramBinNs;
}

size_t tropical_dac_copy_callback_histogram(
  tropical_dac_t d, uint64_t* out, size_t capacity, uint64_t* epoch)
{
  if (!d || !out || !epoch) return 0;
  return static_cast<RuntimeDAC*>(d)->copy_callback_histogram(
    out, capacity, *epoch);
}

uint64_t tropical_dac_request_output_capture(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->request_output_capture() : 0;
}

bool tropical_dac_read_output_capture(
  tropical_dac_t d, uint64_t sequence, uint64_t* start_index,
  double* out, size_t capacity)
{
  if (!d || !start_index) return false;
  return static_cast<RuntimeDAC*>(d)->read_output_capture(
    sequence, *start_index, out, capacity);
}

unsigned int tropical_dac_get_buffer_frames(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->negotiated_buffer_frames() : 0;
}

bool tropical_dac_is_reconnecting(tropical_dac_t d)
{
  return d && static_cast<RuntimeDAC*>(d)->is_reconnecting();
}

uint64_t tropical_dac_disconnect_count(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->disconnect_count() : 0;
}

uint64_t tropical_dac_reconnect_success_count(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->reconnect_success_count() : 0;
}

uint64_t tropical_dac_reconnect_failure_count(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->reconnect_failure_count() : 0;
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

uint64_t tropical_dac_playback_position(tropical_dac_t d)
{
  return d ? static_cast<RuntimeDAC*>(d)->playback_position() : 0;
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

bool tropical_runtime_load_ir_msl(tropical_runtime_t r, const char* ir_text, size_t ir_len,
                                  const char* msl_source, size_t msl_len,
                                  const char* manifest_json, size_t manifest_len)
{
  if (!r || !ir_text || !manifest_json) return false;
  try
  {
    return static_cast<tropical_runtime::FlatRuntime*>(r)->load_ir_msl(
      std::string(ir_text, ir_len),
      msl_source ? std::string(msl_source, msl_len) : std::string{},
      std::string(manifest_json, manifest_len));
  }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

bool tropical_runtime_load_ir_staged(tropical_runtime_t r, const char* ir_text, size_t ir_len,
                                     const char* msl_source, size_t msl_len,
                                     const char* coeff_ir, size_t coeff_len,
                                     const char* manifest_json, size_t manifest_len)
{
  if (!r || !ir_text || !manifest_json) return false;
  try
  {
    return static_cast<tropical_runtime::FlatRuntime*>(r)->load_ir_staged(
      std::string(ir_text, ir_len),
      msl_source ? std::string(msl_source, msl_len) : std::string{},
      coeff_ir ? std::string(coeff_ir, coeff_len) : std::string{},
      std::string(manifest_json, manifest_len));
  }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

void tropical_runtime_set_sample_index(tropical_runtime_t r, uint64_t idx)
{
  if (!r) return;
  static_cast<tropical_runtime::FlatRuntime*>(r)->set_sample_index(idx);
}

uint8_t* tropical_compile_ir_to_wasm(const char* ir_text, size_t ir_len, size_t* out_len)
{
#ifdef TROPICAL_WASM_EMIT
  if (!ir_text || !out_len) { set_error("compile_ir_to_wasm: null argument"); return nullptr; }
  try
  {
    std::string err;
    std::vector<uint8_t> bytes =
      tropical_jit::compile_ir_to_wasm(std::string(ir_text, ir_len), err);
    if (bytes.empty())
    {
      set_error(err.empty() ? "compile_ir_to_wasm: produced no output" : err);
      return nullptr;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (!buf) { set_error("compile_ir_to_wasm: out of memory"); return nullptr; }
    std::memcpy(buf, bytes.data(), bytes.size());
    *out_len = bytes.size();
    return buf;
  }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
#else
  (void)ir_text; (void)ir_len;
  if (out_len) *out_len = 0;
  set_error("libtropical was built without TROPICAL_WASM_EMIT");
  return nullptr;
#endif
}

void tropical_free_buffer(uint8_t* buf) { std::free(buf); }

void tropical_runtime_process(tropical_runtime_t r)
{
  if (r) static_cast<tropical_runtime::FlatRuntime*>(r)->process();
}

bool tropical_runtime_process_offline(tropical_runtime_t r)
{
  if (!r)
  {
    set_error("runtime process offline: null runtime");
    return false;
  }
  try
  {
    static_cast<tropical_runtime::FlatRuntime*>(r)->process_offline();
    return true;
  }
  catch (const std::exception & e)
  {
    set_error(e.what());
    return false;
  }
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

unsigned int tropical_runtime_metal_render_tile_frames(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_render_tile_frames();
}

unsigned int tropical_runtime_metal_worker_capacity_frames(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_worker_capacity_frames();
}

uint64_t tropical_runtime_metal_published_activation_epoch(
  tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_published_activation_epoch();
}

uint64_t tropical_runtime_metal_acknowledged_activation_epoch(
  tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_acknowledged_activation_epoch();
}

bool tropical_runtime_metal_worker_stage_times(
  tropical_runtime_t r, tropical_metal_worker_stage_times_t * out)
{
  if (!out) return false;
  *out = {};
  if (!r) return false;
  const auto times = static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_worker_stage_times();
  *out = {
    times[0], times[1], times[2], times[3],
    times[4], times[5], times[6], times[7],
  };
  return true;
}

bool tropical_runtime_metal_activation_latency_stats(
  tropical_runtime_t r, tropical_metal_activation_latency_stats_t * out)
{
  if (!out) return false;
  *out = {};
  if (!r) return false;
  const auto stats = static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_activation_latency_stats();
  *out = {stats[0], stats[1], stats[2], stats[3]};
  return true;
}

uint64_t tropical_runtime_metal_worker_cpu_time_ns(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_worker_cpu_time_ns();
}

uint64_t tropical_runtime_metal_worker_wall_time_ns(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_worker_wall_time_ns();
}

uint64_t tropical_runtime_ownership_failure_count(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->ownership_failure_count();
}

uint64_t tropical_runtime_metal_dispatch_failure_count(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_dispatch_failure_count();
}

uint64_t tropical_runtime_metal_render_starvation_count(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_render_starvation_count();
}

bool tropical_runtime_metal_first_starvation_snapshot(
  tropical_runtime_t r, tropical_metal_starvation_snapshot_t * out)
{
  if (!out) return false;
  *out = {};
  if (!r) return false;
  const auto snapshot = static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_first_starvation_snapshot();
  *out = {
    snapshot[0] != 0,
    snapshot[1],
    snapshot[2],
    snapshot[3],
    snapshot[4],
    snapshot[5],
    snapshot[6],
    snapshot[7],
    snapshot[8],
    static_cast<uint32_t>(snapshot[9]),
    static_cast<uint32_t>(snapshot[10]),
    static_cast<uint32_t>(snapshot[11]),
    static_cast<uint32_t>(snapshot[12]),
    static_cast<uint32_t>(snapshot[13]),
    static_cast<uint32_t>(snapshot[14]),
    static_cast<uint32_t>(snapshot[15]),
  };
  return out->valid;
}

uint64_t tropical_runtime_metal_epoch_tag_mismatch_count(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_epoch_tag_mismatch_count();
}

uint64_t tropical_runtime_metal_activation_retarget_count(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_activation_retarget_count();
}

uint64_t tropical_runtime_metal_activation_failure_count(tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_activation_failure_count();
}

uint64_t tropical_runtime_metal_callback_thread_violation_count(
  tropical_runtime_t r)
{
  if (!r) return 0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)
    ->metal_callback_thread_violation_count();
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

double tropical_runtime_current_sample_index(tropical_runtime_t r)
{
  if (!r) return 0.0;
  return static_cast<double>(
    static_cast<tropical_runtime::FlatRuntime*>(r)->current_sample_index());
}

double tropical_runtime_sample_rate(tropical_runtime_t r)
{
  if (!r) return 44100.0;
  return static_cast<tropical_runtime::FlatRuntime*>(r)->sample_rate();
}

bool tropical_runtime_render_window(tropical_runtime_t r, uint64_t start_index,
                                    unsigned int count, const unsigned int* slot_ids,
                                    unsigned int n_slots, double* out)
{
  if (!r) return false;
  return static_cast<tropical_runtime::FlatRuntime*>(r)->render_window(
    start_index, count,
    reinterpret_cast<const uint32_t*>(slot_ids), n_slots, out);
}

// ── Socket endpoint ─────────────────────────────────────────────────────────

tropical_socket_t tropical_socket_listen(tropical_runtime_t r, const char* addr)
{
  if (!r || !addr) { set_error("socket_listen: null argument"); return nullptr; }
  try
  {
    auto * rt = static_cast<tropical_runtime::FlatRuntime*>(r);
    auto * s = new tropical_socket::SocketServer(rt, std::string(addr));
    if (!s->start()) { set_error(s->error()); delete s; return nullptr; }
    return static_cast<void*>(s);
  }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
}

void tropical_socket_free(tropical_socket_t s)
{
  delete static_cast<tropical_socket::SocketServer*>(s);
}

bool tropical_socket_next_control(tropical_socket_t s, uint64_t* out_client_id,
                                  char** out_bytes, size_t* out_len)
{
  if (!s) return false;
  return static_cast<tropical_socket::SocketServer*>(s)->next_control(out_client_id, out_bytes, out_len);
}

void tropical_socket_send_response(tropical_socket_t s, uint64_t client_id,
                                   const char* bytes, size_t len)
{
  if (s) static_cast<tropical_socket::SocketServer*>(s)->send_response(client_id, bytes, len);
}

} // extern "C"
