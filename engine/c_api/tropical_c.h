#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Opaque handle types */
typedef void* tropical_dac_t;
typedef void* tropical_param_t;
typedef void* tropical_runtime_t;

/* Error handling — thread-local; valid until next call on this thread */
const char* tropical_last_error(void);

/* ---------- ControlParam API ---------- */
/* Create a smoothed parameter. init_value is the starting value; time_const is the
   one-pole lowpass time constant in seconds (e.g. 0.01 = ~10ms ramp). */
tropical_param_t tropical_param_new(double init_value, double time_const);
void           tropical_param_free(tropical_param_t);
/* Thread-safe write (atomic store) — call from UI/control thread */
void           tropical_param_set(tropical_param_t, double value);
/* Thread-safe read (atomic load) */
double         tropical_param_get(tropical_param_t);

/* ---------- Device enumeration (no DAC instance required) ---------- */

typedef struct {
  unsigned int id;
  char         name[256];
  unsigned int output_channels;
  unsigned int input_channels;
  bool         is_default_output;
  unsigned int preferred_sample_rate;
  unsigned int sample_rate_count;        /* number of valid entries in sample_rates */
  unsigned int sample_rates[32];
} tropical_device_info_t;

unsigned int tropical_audio_device_count(void);
/* Fills `out[0..count-1]` with device IDs.  Call tropical_audio_device_count() first. */
void         tropical_audio_get_device_ids(unsigned int* out, unsigned int count);
bool         tropical_audio_get_device_info(unsigned int device_id, tropical_device_info_t* out);
unsigned int tropical_audio_default_output_device(void);

/* ---------- DAC API ---------- */
tropical_dac_t tropical_dac_new_runtime(tropical_runtime_t, unsigned int sample_rate, unsigned int channels);
void         tropical_dac_free(tropical_dac_t);
void         tropical_dac_start(tropical_dac_t);
void         tropical_dac_stop(tropical_dac_t);
bool         tropical_dac_is_running(tropical_dac_t);

typedef struct {
  uint64_t callback_count;
  double   avg_callback_ms;
  double   max_callback_ms;
  uint64_t underrun_count;  /* non-zero RtAudioStreamStatus reported by driver */
  uint64_t overrun_count;   /* callbacks that exceeded their time budget */
} tropical_dac_stats_t;

void tropical_dac_get_stats(tropical_dac_t, tropical_dac_stats_t* out);
void tropical_dac_reset_stats(tropical_dac_t);
/* True while a device-disconnect has been detected and reconnection is in progress */
bool tropical_dac_is_reconnecting(tropical_dac_t);

/* Returns the device ID currently open for output (0 if not started) */
unsigned int tropical_dac_get_active_device(tropical_dac_t);
/* Switch the running DAC to a different output device.  Returns false on failure. */
bool         tropical_dac_switch_device(tropical_dac_t, unsigned int device_id);

/* Master clock: the "audible now" sample index — seconds of stream played ×
   rate, minus output latency. The authoritative position a slave consumer
   (scope, video) renders against for drift-free sync. 0 if not running. */
uint64_t     tropical_dac_playback_position(tropical_dac_t);

/* ---------- FlatRuntime API ---------- */

tropical_runtime_t tropical_runtime_new(unsigned int buffer_length);
void             tropical_runtime_free(tropical_runtime_t);
/* Load a kernel from textual LLVM IR plus a metadata manifest (a
   tropical_plan_5 JSON; its instruction graph is metadata only — codegen
   comes from ir_text, emitted by Lean's EmitLlvm). The sole load path:
   the C++ plan compiler was retired in Phase 2. Always fused. */
bool             tropical_runtime_load_ir(tropical_runtime_t, const char* ir_text, size_t ir_len, const char* manifest_json, size_t manifest_len);

/* ---------- Build-time IR→wasm (TROPICAL_WASM_EMIT builds only) ----------
   Lower Lean's LLVM IR to a complete wasm32 module, fully in-process (wasm32
   TargetMachine + lld-as-a-library — no subprocess). Returns a malloc'd buffer
   of length *out_len (free with tropical_free_buffer), or NULL on error (see
   tropical_last_error()). Returns NULL if libtropical was built without
   TROPICAL_WASM_EMIT. This is a build/CI tool — the shipped audio runtime does
   not need it. */
uint8_t*         tropical_compile_ir_to_wasm(const char* ir_text, size_t ir_len, size_t* out_len);
void             tropical_free_buffer(uint8_t* buf);
void             tropical_runtime_process(tropical_runtime_t);
const double*    tropical_runtime_output_buffer(tropical_runtime_t);
unsigned int     tropical_runtime_get_buffer_length(tropical_runtime_t);

/* Fade control (for DAC) */
void             tropical_runtime_begin_fade_in(tropical_runtime_t);
void             tropical_runtime_begin_fade_out(tropical_runtime_t);
bool             tropical_runtime_is_fade_out_complete(tropical_runtime_t);

/* ---------- Slot model ----------
 *
 * Slots are an inter-module shared array. Each output port of every
 * instance + each control-plane param has a dedicated slot index. The
 * JIT codegen reads slots via 'slot' operands and writes via WriteSlot
 * instructions (parent→child input wiring, MCP wire auto-delay,
 * session-level delay extraction). Control-plane writes from outside
 * the kernel go through `set_slot`.
 *
 * `slot_index` returns UINT32_MAX when no slot with that name exists
 * in the active plan — typical lookup pattern is to resolve the name
 * once at session start and write by integer index thereafter. */
unsigned int     tropical_runtime_slot_index(tropical_runtime_t, const char* name);
void             tropical_runtime_set_slot(tropical_runtime_t, unsigned int slot_index, double value);
double           tropical_runtime_get_slot(tropical_runtime_t, unsigned int slot_index);

/* Random-access render (scope / slave consumers): evaluate the active fused
   kernel over [start_index, start_index+count) and write each requested slot's
   per-sample trajectory into `out`, slot-major (out[k*count + i]). Exact and
   concurrency-safe with the audio thread for a register-free (stateless) patch.
   Returns false if the kernel isn't fused or a slot id is out of range. */
bool             tropical_runtime_render_window(tropical_runtime_t, uint64_t start_index,
                   unsigned int count, const unsigned int* slot_ids, unsigned int n_slots,
                   double* out);

/* ---------- Socket endpoint ----------
 *
 * A single Unix-domain socket serving many clients over newline-delimited
 * JSON-RPC, with an internal control/data plane split. The socket borrows the
 * runtime (which must outlive it). Data-plane methods (set_param, get_telemetry)
 * are handled inside C++; control-plane requests are queued for the Lean driver
 * thread, which pulls them with tropical_socket_next_control, dispatches them,
 * and replies with tropical_socket_send_response. The call direction across the
 * FFI is strictly Lean->C++ — Lean pulls and pushes; C++ never calls up. */

typedef void* tropical_socket_t;

/* Bind + listen on the Unix-domain socket at `addr` and spawn the accept
   thread. Returns NULL on failure (see tropical_last_error). */
tropical_socket_t tropical_socket_listen(tropical_runtime_t, const char* addr);
/* Stop the server (close the socket, wake the pull, join threads, unlink the
   socket file) and free it. */
void             tropical_socket_free(tropical_socket_t);

/* Control-plane pull, called only from the single Lean driver thread. Blocks
   until a control message arrives or the socket shuts down. Returns false on
   shutdown-with-empty-queue. On true, *out_client_id is set and the request
   line is copied into a malloc'd NUL-terminated buffer (*out_bytes, *out_len);
   free it with tropical_free_buffer. */
bool             tropical_socket_next_control(tropical_socket_t, uint64_t* out_client_id,
                                              char** out_bytes, size_t* out_len);

/* Control-plane push: send a response line (a newline is appended) to the
   client identified by client_id. No-op if the client has disconnected. */
void             tropical_socket_send_response(tropical_socket_t, uint64_t client_id,
                                               const char* bytes, size_t len);

#ifdef __cplusplus
}
#endif
