#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Opaque handle types */
typedef void* egress_dac_t;
typedef void* egress_param_t;
typedef void* egress_runtime_t;

/* Error handling — thread-local; valid until next call on this thread */
const char* egress_last_error(void);

/* ---------- ControlParam API ---------- */
/* Create a smoothed parameter. init_value is the starting value; time_const is the
   one-pole lowpass time constant in seconds (e.g. 0.01 = ~10ms ramp). */
egress_param_t egress_param_new(double init_value, double time_const);
void           egress_param_free(egress_param_t);
/* Thread-safe write (atomic store) — call from UI/control thread */
void           egress_param_set(egress_param_t, double value);
/* Thread-safe read (atomic load) */
double         egress_param_get(egress_param_t);

/* Create a trigger parameter. Fires once per frame: set value to 1.0 from the UI
   thread; the DSP evaluator reads and atomically clears it each frame.
   The Param must outlive all modules that reference the returned expression. */
egress_param_t egress_param_new_trigger(void);

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
} egress_device_info_t;

unsigned int egress_audio_device_count(void);
/* Fills `out[0..count-1]` with device IDs.  Call egress_audio_device_count() first. */
void         egress_audio_get_device_ids(unsigned int* out, unsigned int count);
bool         egress_audio_get_device_info(unsigned int device_id, egress_device_info_t* out);
unsigned int egress_audio_default_output_device(void);

/* ---------- DAC API ---------- */
egress_dac_t egress_dac_new_runtime(egress_runtime_t, unsigned int sample_rate, unsigned int channels);
void         egress_dac_free(egress_dac_t);
void         egress_dac_start(egress_dac_t);
void         egress_dac_stop(egress_dac_t);
bool         egress_dac_is_running(egress_dac_t);

typedef struct {
  uint64_t callback_count;
  double   avg_callback_ms;
  double   max_callback_ms;
  uint64_t underrun_count;  /* non-zero RtAudioStreamStatus reported by driver */
  uint64_t overrun_count;   /* callbacks that exceeded their time budget */
} egress_dac_stats_t;

void egress_dac_get_stats(egress_dac_t, egress_dac_stats_t* out);
void egress_dac_reset_stats(egress_dac_t);
/* True while a device-disconnect has been detected and reconnection is in progress */
bool egress_dac_is_reconnecting(egress_dac_t);

/* Returns the device ID currently open for output (0 if not started) */
unsigned int egress_dac_get_active_device(egress_dac_t);
/* Switch the running DAC to a different output device.  Returns false on failure. */
bool         egress_dac_switch_device(egress_dac_t, unsigned int device_id);

/* ---------- FlatRuntime API ---------- */

egress_runtime_t egress_runtime_new(unsigned int buffer_length);
void             egress_runtime_free(egress_runtime_t);
bool             egress_runtime_load_plan(egress_runtime_t, const char* plan_json, size_t len);
void             egress_runtime_process(egress_runtime_t);
const double*    egress_runtime_output_buffer(egress_runtime_t);
unsigned int     egress_runtime_get_buffer_length(egress_runtime_t);

/* Fade control (for DAC) */
void             egress_runtime_begin_fade_in(egress_runtime_t);
void             egress_runtime_begin_fade_out(egress_runtime_t);
bool             egress_runtime_is_fade_out_complete(egress_runtime_t);

#ifdef __cplusplus
}
#endif
