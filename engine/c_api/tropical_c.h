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

/* Create a trigger parameter. Fires once per frame: set value to 1.0 from the UI
   thread; the DSP evaluator reads and atomically clears it each frame.
   The Param must outlive all modules that reference the returned expression. */
tropical_param_t tropical_param_new_trigger(void);

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

/* ---------- FlatRuntime API ---------- */

tropical_runtime_t tropical_runtime_new(unsigned int buffer_length);
void             tropical_runtime_free(tropical_runtime_t);
bool             tropical_runtime_load_plan(tropical_runtime_t, const char* plan_json, size_t len);
void             tropical_runtime_process(tropical_runtime_t);
const double*    tropical_runtime_output_buffer(tropical_runtime_t);
unsigned int     tropical_runtime_get_buffer_length(tropical_runtime_t);

/* Fade control (for DAC) */
void             tropical_runtime_begin_fade_in(tropical_runtime_t);
void             tropical_runtime_begin_fade_out(tropical_runtime_t);
bool             tropical_runtime_is_fade_out_complete(tropical_runtime_t);

#ifdef __cplusplus
}
#endif
