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
#define EGRESS_EXPR_SMOOTHED_PARAM 39
#define EGRESS_EXPR_SELECT        40

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

/* ---------- Module spec builder API ---------- */
egress_module_spec_t egress_module_spec_new(unsigned int input_count, double sample_rate);
void         egress_module_spec_add_output(egress_module_spec_t, egress_expr_t);
void         egress_module_spec_add_register(egress_module_spec_t, egress_expr_t body, egress_value_t init);
void         egress_module_spec_add_register_array(egress_module_spec_t, unsigned int source_input_id, egress_value_t init);
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
void         egress_nested_spec_add_register_array(egress_nested_spec_t, unsigned int source_input_id, egress_value_t init);
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
bool            egress_graph_disconnect(egress_graph_t, const char* src, unsigned int src_out,
                                        const char* dst, unsigned int dst_in);
bool            egress_graph_set_input_expr(egress_graph_t, const char* module,
                                            unsigned int input_id, egress_expr_t);
egress_expr_t   egress_graph_get_input_expr(egress_graph_t, const char* module,
                                            unsigned int input_id);
bool            egress_graph_add_output(egress_graph_t, const char* module, unsigned int output_id);
bool            egress_graph_add_output_expr(egress_graph_t, egress_expr_t);
size_t          egress_graph_add_output_tap(egress_graph_t, const char* module, unsigned int output_id);
bool            egress_graph_remove_output_tap(egress_graph_t, size_t tap_id);
void            egress_graph_process(egress_graph_t);
void            egress_graph_prime_jit(egress_graph_t);
/* Pointer valid until next egress_graph_process() call */
const double*   egress_graph_output_buffer(egress_graph_t);
/* Pointer valid until next egress_graph_tap_buffer() call on this thread */
const double*   egress_graph_tap_buffer(egress_graph_t, size_t tap_id, size_t* out_len);
void            egress_graph_set_worker_count(egress_graph_t, unsigned int);
unsigned int    egress_graph_get_worker_count(egress_graph_t);
void            egress_graph_set_fusion_enabled(egress_graph_t, bool);
bool            egress_graph_get_fusion_enabled(egress_graph_t);
unsigned int    egress_graph_get_buffer_length(egress_graph_t);

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

#ifdef __cplusplus
}
#endif
