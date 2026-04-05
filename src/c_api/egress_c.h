#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Opaque handle types */
typedef void* egress_graph_t;
typedef void* egress_expr_t;
typedef void* egress_value_t;
typedef void* egress_module_spec_t;
typedef void* egress_nested_spec_t;
typedef void* egress_dac_t;
typedef void* egress_param_t;

/* ExprKind integer constants — match ExprKind enum order in Expr.hpp */
#define EGRESS_EXPR_LITERAL       0
#define EGRESS_EXPR_REF           1
#define EGRESS_EXPR_INPUT         2
#define EGRESS_EXPR_REGISTER      3
#define EGRESS_EXPR_NESTED        4
#define EGRESS_EXPR_DELAY         5
#define EGRESS_EXPR_SAMPLE_RATE   6
#define EGRESS_EXPR_SAMPLE_INDEX  7
#define EGRESS_EXPR_FUNCTION      8
#define EGRESS_EXPR_CALL          9
#define EGRESS_EXPR_ARRAY_PACK    10
#define EGRESS_EXPR_INDEX         11
#define EGRESS_EXPR_ARRAY_SET     12
#define EGRESS_EXPR_NOT           13
#define EGRESS_EXPR_LESS          14
#define EGRESS_EXPR_LESS_EQUAL    15
#define EGRESS_EXPR_GREATER       16
#define EGRESS_EXPR_GREATER_EQUAL 17
#define EGRESS_EXPR_EQUAL         18
#define EGRESS_EXPR_NOT_EQUAL     19
#define EGRESS_EXPR_ADD           20
#define EGRESS_EXPR_SUB           21
#define EGRESS_EXPR_MUL           22
#define EGRESS_EXPR_DIV           23
#define EGRESS_EXPR_MATMUL        24
#define EGRESS_EXPR_POW           25
#define EGRESS_EXPR_MOD           26
#define EGRESS_EXPR_FLOOR_DIV     27
#define EGRESS_EXPR_BIT_AND       28
#define EGRESS_EXPR_BIT_OR        29
#define EGRESS_EXPR_BIT_XOR       30
#define EGRESS_EXPR_LSHIFT        31
#define EGRESS_EXPR_RSHIFT        32
#define EGRESS_EXPR_ABS           33
#define EGRESS_EXPR_CLAMP         34
#define EGRESS_EXPR_LOG           35
#define EGRESS_EXPR_SIN           36
#define EGRESS_EXPR_NEG           37
#define EGRESS_EXPR_BIT_NOT       38
#define EGRESS_EXPR_SMOOTHED_PARAM       39
#define EGRESS_EXPR_SELECT               40
#define EGRESS_EXPR_TRIGGER_PARAM        41
#define EGRESS_EXPR_CONSTRUCT_STRUCT     42
#define EGRESS_EXPR_FIELD_ACCESS         43
#define EGRESS_EXPR_CONSTRUCT_VARIANT    44
#define EGRESS_EXPR_MATCH_VARIANT        45

/* Error handling — thread-local; valid until next call on this thread */
const char* egress_last_error(void);

/* ---------- Value API ---------- */
egress_value_t egress_value_float(double v);
egress_value_t egress_value_int(int64_t v);
egress_value_t egress_value_bool(bool v);
egress_value_t egress_value_array(const egress_value_t* items, size_t n);
egress_value_t egress_value_matrix(const egress_value_t* items, size_t rows, size_t cols);
double         egress_value_to_float(egress_value_t);
int64_t        egress_value_to_int(egress_value_t);
void           egress_value_free(egress_value_t);

/* ---------- Expression factory API ---------- */
egress_expr_t egress_expr_literal_float(double v);
egress_expr_t egress_expr_literal_int(int64_t v);
egress_expr_t egress_expr_literal_bool(bool v);
egress_expr_t egress_expr_literal_value(egress_value_t v);
egress_expr_t egress_expr_input(unsigned int input_id);
egress_expr_t egress_expr_register(unsigned int reg_id);
egress_expr_t egress_expr_nested_output(unsigned int node_id, unsigned int output_id);
egress_expr_t egress_expr_delay_value(unsigned int node_id);
egress_expr_t egress_expr_ref(const char* module_name, unsigned int output_id);
egress_expr_t egress_expr_sample_rate(void);
egress_expr_t egress_expr_sample_index(void);
egress_expr_t egress_expr_unary(int kind, egress_expr_t operand);
egress_expr_t egress_expr_binary(int kind, egress_expr_t lhs, egress_expr_t rhs);
egress_expr_t egress_expr_clamp(egress_expr_t v, egress_expr_t lo, egress_expr_t hi);
egress_expr_t egress_expr_select(egress_expr_t cond, egress_expr_t then_expr, egress_expr_t else_expr);
egress_expr_t egress_expr_array_pack(const egress_expr_t* items, size_t n);
egress_expr_t egress_expr_index(egress_expr_t arr, egress_expr_t idx);
egress_expr_t egress_expr_array_set(egress_expr_t arr, egress_expr_t idx, egress_expr_t val);
egress_expr_t egress_expr_function(unsigned int param_count, egress_expr_t body);
egress_expr_t egress_expr_call(egress_expr_t callee, const egress_expr_t* args, size_t n);
void          egress_expr_free(egress_expr_t);

/* ---------- ADT expression constructors ---------- */
egress_expr_t egress_expr_construct_struct(
    const char* type_name, const egress_expr_t* field_exprs, size_t count);
egress_expr_t egress_expr_field_access(
    const char* type_name, egress_expr_t struct_expr, unsigned int field_index);
egress_expr_t egress_expr_construct_variant(
    const char* type_name, unsigned int variant_tag,
    const egress_expr_t* payload_exprs, size_t count);
egress_expr_t egress_expr_match_variant(
    const char* type_name, egress_expr_t scrutinee,
    const egress_expr_t* branch_exprs, size_t branch_count);

/* ---------- Type definition API (graph-scoped) ---------- */
bool egress_typedef_struct(egress_graph_t g, const char* name,
    const char** field_names, const int* field_scalar_types, size_t count);
bool egress_typedef_sum(egress_graph_t g, const char* name,
    const char** variant_names,
    const char** variant_field_names_flat,
    const int* variant_field_scalar_types_flat,
    const size_t* variant_field_counts,
    size_t variant_count);

/* ---------- Port type annotation API ---------- */
bool egress_module_declare_input_type(egress_graph_t g, const char* module_name,
    unsigned int input_index, const char* type_name);
bool egress_module_declare_output_type(egress_graph_t g, const char* module_name,
    unsigned int output_index, const char* type_name);
bool egress_module_declare_register_type(egress_graph_t g, const char* module_name,
    unsigned int register_index, const char* type_name);

/* ---------- ControlParam API ---------- */
/* Create a smoothed parameter. init_value is the starting value; time_const is the
   one-pole lowpass time constant in seconds (e.g. 0.01 = ~10ms ramp). */
egress_param_t egress_param_new(double init_value, double time_const);
void           egress_param_free(egress_param_t);
/* Thread-safe write (atomic store) — call from UI/control thread */
void           egress_param_set(egress_param_t, double value);
/* Thread-safe read (atomic load) */
double         egress_param_get(egress_param_t);
/* Create a SmoothedParam expression that can be used in module outputs/registers.
   The Param must outlive all modules that reference the returned expression. */
egress_expr_t  egress_expr_param(egress_param_t);

/* Create a trigger parameter. Fires once per frame: set value to 1.0 from the UI
   thread; the DSP evaluator reads and atomically clears it each frame.
   The Param must outlive all modules that reference the returned expression. */
egress_param_t egress_param_new_trigger(void);
egress_expr_t  egress_expr_trigger_param(egress_param_t);

/* ---------- Module spec builder API ---------- */
egress_module_spec_t egress_module_spec_new(unsigned int input_count, double sample_rate);
void         egress_module_spec_add_output(egress_module_spec_t, egress_expr_t);
void         egress_module_spec_add_register(egress_module_spec_t, egress_expr_t body, egress_value_t init);
/* Returns the node_id assigned to the delay state (for use in egress_expr_delay_value) */
unsigned int egress_module_spec_add_delay_state(egress_module_spec_t, egress_value_t init, egress_expr_t update_expr);
void         egress_module_spec_add_nested(egress_module_spec_t, egress_nested_spec_t);
void         egress_module_spec_set_composite_schedule(egress_module_spec_t, const unsigned int* schedule, size_t n);
void         egress_module_spec_set_output_boundary(egress_module_spec_t, unsigned int boundary_id);
void         egress_module_spec_free(egress_module_spec_t);

/* ---------- Nested module spec builder API ---------- */
/* node_id is auto-assigned from a global counter; retrieve with egress_nested_spec_node_id */
egress_nested_spec_t egress_nested_spec_new(unsigned int input_count, double sample_rate);
unsigned int egress_nested_spec_node_id(egress_nested_spec_t);
void         egress_nested_spec_add_input_expr(egress_nested_spec_t, egress_expr_t);
void         egress_nested_spec_add_output(egress_nested_spec_t, egress_expr_t);
void         egress_nested_spec_add_register(egress_nested_spec_t, egress_expr_t body, egress_value_t init);
unsigned int egress_nested_spec_add_delay_state(egress_nested_spec_t, egress_value_t init, egress_expr_t update_expr);
void         egress_nested_spec_add_nested(egress_nested_spec_t outer, egress_nested_spec_t inner);
void         egress_nested_spec_set_composite_schedule(egress_nested_spec_t, const unsigned int* schedule, size_t n);
void         egress_nested_spec_set_output_boundary(egress_nested_spec_t, unsigned int boundary_id);
void         egress_nested_spec_free(egress_nested_spec_t);

/* ---------- Graph API ---------- */
egress_graph_t  egress_graph_new(unsigned int buffer_length);
void            egress_graph_free(egress_graph_t);
bool            egress_graph_add_module(egress_graph_t, const char* name, egress_module_spec_t);
bool            egress_graph_remove_module(egress_graph_t, const char* name);
bool            egress_graph_connect(egress_graph_t, const char* src, unsigned int src_out,
                                     const char* dst, unsigned int dst_in);
bool            egress_graph_set_input_expr(egress_graph_t, const char* module,
                                            unsigned int input_id, egress_expr_t);
bool            egress_graph_add_output(egress_graph_t, const char* module, unsigned int output_id);
size_t          egress_graph_add_output_tap(egress_graph_t, const char* module, unsigned int output_id);
bool            egress_graph_remove_output_tap(egress_graph_t, size_t tap_id);
void            egress_graph_process(egress_graph_t);
void            egress_graph_prime_jit(egress_graph_t);
/* Clear all input expressions and output mix on the graph. */
void            egress_graph_clear_wiring(egress_graph_t);
/* Begin a batched update — addModule, set_input_expr, etc. defer rebuilds
   until egress_graph_end_update(). Calls may nest with load_plan. */
void            egress_graph_begin_update(egress_graph_t);
/* End a batched update — triggers a single rebuild for all deferred changes.
   Returns false on error. */
bool            egress_graph_end_update(egress_graph_t);
/* Load wiring and outputs from a plan JSON string. Modules must already exist.
   Joins an active batch if one exists, otherwise wraps its own begin/end_update. */
bool            egress_graph_load_plan(egress_graph_t, const char* plan_json, size_t len);
/* Pointer valid until next egress_graph_process() call */
const double*   egress_graph_output_buffer(egress_graph_t);
/* Pointer valid until next egress_graph_tap_buffer() call on this thread */
const double*   egress_graph_tap_buffer(egress_graph_t, size_t tap_id, size_t* out_len);
void            egress_graph_set_worker_count(egress_graph_t, unsigned int);
unsigned int    egress_graph_get_worker_count(egress_graph_t);
void            egress_graph_set_fusion_enabled(egress_graph_t, bool);
bool            egress_graph_get_fusion_enabled(egress_graph_t);
unsigned int    egress_graph_get_buffer_length(egress_graph_t);
/* Returns JSON-serialized ProfileStats. Pointer valid until next call on this thread. */
const char*     egress_graph_get_profile_stats_json(egress_graph_t);
void            egress_graph_reset_profile_stats(egress_graph_t);
/* Returns JSON array of build timing entries accumulated since the last begin_update().
   Each entry: {module_count, input_programs_ms, fused_jit_ms, total_ms}.
   Pointer valid until next call on this thread. */
const char*     egress_graph_get_build_timing_json(egress_graph_t);

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
egress_dac_t egress_dac_new(egress_graph_t, unsigned int sample_rate, unsigned int channels);
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
typedef void* egress_runtime_t;

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
