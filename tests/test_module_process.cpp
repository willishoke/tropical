/**
 * test_module_process.cpp
 *
 * Instantiates modules via the C API and calls egress_graph_process() without
 * an audio device.  Covers the JIT code paths that the audio thread exercises,
 * catching crashes that would otherwise terminate the server process.
 *
 * Build with EGRESS_LLVM_ORC_JIT=ON to exercise JIT paths.
 */

#include "c_api/egress_c.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

// ---- tiny test harness -------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

static void run_test(const char* name, std::function<void()> fn)
{
  printf("  %-60s", name);
  fflush(stdout);
  fn();
  // If fn() didn't abort/exit we reach here
  printf("PASS\n");
  ++g_pass;
}

#define ASSERT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      printf("FAIL\n    assertion failed: %s  (line %d)\n", #cond, __LINE__);\
      const char* err = egress_last_error();                                  \
      if (err && err[0]) printf("    last error: %s\n", err);                \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

#define ASSERT_OK(call)                                                       \
  do {                                                                        \
    bool _ok = (call);                                                        \
    if (!_ok) {                                                               \
      const char* _err = egress_last_error();                                 \
      printf("FAIL\n    %s returned false\n    last error: %s\n",            \
             #call, _err ? _err : "(none)");                                  \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

// ---- helpers -----------------------------------------------------------------

// Builds the expression:  mod(add(mod(x, 1.0), 1.0), 1.0)
// This is _wrap01 inlined — the same wrapping the Clock module uses.
static egress_expr_t wrap01(egress_expr_t x)
{
  egress_expr_t one  = egress_expr_literal_float(1.0);
  egress_expr_t one2 = egress_expr_literal_float(1.0);
  egress_expr_t one3 = egress_expr_literal_float(1.0);
  egress_expr_t inner = egress_expr_binary(EGRESS_EXPR_MOD, x, one);
  egress_expr_t added = egress_expr_binary(EGRESS_EXPR_ADD, inner, one2);
  egress_expr_t outer = egress_expr_binary(EGRESS_EXPR_MOD, added, one3);
  egress_expr_free(one);
  egress_expr_free(one2);
  egress_expr_free(one3);
  return outer;
}

// ---- module spec builders ----------------------------------------------------

// Clock module: 2 inputs (freq scalar, ratios_in array), 2 outputs
//   output  = mul(lt(basePhase, 0.5), 1.0)
//   ratios_out = mul(lt(ratioPhase, 0.5), 1.0)   <- array output
// No registers, no delay states.  Mirrors module_library.ts clock().
static egress_module_spec_t build_clock_spec()
{
  egress_module_spec_t spec = egress_module_spec_new(2, 44100.0);

  egress_expr_t sr  = egress_expr_sample_rate();
  egress_expr_t idx = egress_expr_sample_index();
  egress_expr_t inp0 = egress_expr_input(0);  // freq (scalar)
  egress_expr_t inp1 = egress_expr_input(1);  // ratios_in (array)

  // basePhase = wrap01(sampleIndex * inp0 / sr)
  egress_expr_t phase_num  = egress_expr_binary(EGRESS_EXPR_MUL, egress_expr_sample_index(), egress_expr_input(0));
  egress_expr_t base_raw   = egress_expr_binary(EGRESS_EXPR_DIV, phase_num, egress_expr_sample_rate());
  egress_expr_t base_phase = wrap01(base_raw);

  // output = mul(lt(basePhase, 0.5), 1.0)
  egress_expr_t half0   = egress_expr_literal_float(0.5);
  egress_expr_t lt0     = egress_expr_binary(EGRESS_EXPR_LESS, base_phase, half0);
  egress_expr_t one0    = egress_expr_literal_float(1.0);
  egress_expr_t output  = egress_expr_binary(EGRESS_EXPR_MUL, lt0, one0);
  egress_module_spec_add_output(spec, output);
  egress_expr_free(half0);
  egress_expr_free(one0);
  egress_expr_free(output);

  // ratioPhase = wrap01(sampleIndex * inp0 * inp1 / sr)   <- inp1 is array
  egress_expr_t ri_inner = egress_expr_binary(EGRESS_EXPR_MUL, egress_expr_sample_index(), egress_expr_input(0));
  egress_expr_t ri_scaled = egress_expr_binary(EGRESS_EXPR_MUL, ri_inner, egress_expr_input(1));
  egress_expr_t ri_div    = egress_expr_binary(EGRESS_EXPR_DIV, ri_scaled, egress_expr_sample_rate());
  egress_expr_t ratio_phase = wrap01(ri_div);

  // ratios_out = mul(lt(ratioPhase, 0.5), 1.0)
  egress_expr_t half1     = egress_expr_literal_float(0.5);
  egress_expr_t lt1       = egress_expr_binary(EGRESS_EXPR_LESS, ratio_phase, half1);
  egress_expr_t one1      = egress_expr_literal_float(1.0);
  egress_expr_t ratios_out = egress_expr_binary(EGRESS_EXPR_MUL, lt1, one1);
  egress_module_spec_add_output(spec, ratios_out);
  egress_expr_free(half1);
  egress_expr_free(one1);
  egress_expr_free(ratios_out);

  // Free intermediate exprs (add_output keeps a reference)
  egress_expr_free(sr);
  egress_expr_free(idx);
  egress_expr_free(inp0);
  egress_expr_free(inp1);
  egress_expr_free(base_phase);
  egress_expr_free(lt0);
  egress_expr_free(ratio_phase);
  egress_expr_free(lt1);

  return spec;
}

// Simple scalar VCO-like module: 1 input (freq), 1 output (sin wave)
//   output = sin(2π * sampleIndex * freq / sr)
// No registers.
static egress_module_spec_t build_simple_scalar_spec()
{
  egress_module_spec_t spec = egress_module_spec_new(1, 44100.0);

  const double TWO_PI = 6.283185307179586;
  egress_expr_t two_pi = egress_expr_literal_float(TWO_PI);
  egress_expr_t idx    = egress_expr_sample_index();
  egress_expr_t freq   = egress_expr_input(0);
  egress_expr_t sr     = egress_expr_sample_rate();

  egress_expr_t phase  = egress_expr_binary(EGRESS_EXPR_MUL, idx, freq);
  egress_expr_t phase2 = egress_expr_binary(EGRESS_EXPR_DIV, phase, sr);
  egress_expr_t phase3 = egress_expr_binary(EGRESS_EXPR_MUL, two_pi, phase2);
  egress_expr_t output = egress_expr_unary(EGRESS_EXPR_SIN, phase3);

  egress_module_spec_add_output(spec, output);

  egress_expr_free(two_pi);
  egress_expr_free(idx);
  egress_expr_free(freq);
  egress_expr_free(sr);
  egress_expr_free(phase);
  egress_expr_free(phase2);
  egress_expr_free(phase3);
  egress_expr_free(output);

  return spec;
}

// IntSeq with edge detection: includes prev_trig register so a held gate
// only advances the index once (on the rising edge).
// 2 registers: 0=index (int, init 0), 1=prev_trig (int, init 0)
// index_next = select(bit_and(trigger, not(prev_trig)),
//                     mod(add(index, step), seq_len),
//                     index)
// prev_trig_next = trigger
static egress_module_spec_t build_intseq_edge_detect_spec()
{
  egress_module_spec_t spec = egress_module_spec_new(5, 44100.0);

  // output 0: value = sequence[index_reg]
  egress_expr_t value_out = egress_expr_binary(EGRESS_EXPR_INDEX,
    egress_expr_input(4), egress_expr_register(0));
  egress_module_spec_add_output(spec, value_out);
  egress_expr_free(value_out);

  // output 1: index = index_reg
  egress_expr_t idx_out = egress_expr_register(0);
  egress_module_spec_add_output(spec, idx_out);
  egress_expr_free(idx_out);

  // Edge detect: rising = bit_and(trigger, not(prev_trig))
  egress_expr_t trigger   = egress_expr_input(0);
  egress_expr_t prev_trig = egress_expr_register(1);
  egress_expr_t not_prev  = egress_expr_unary(EGRESS_EXPR_NOT, prev_trig);
  egress_expr_t rising    = egress_expr_binary(EGRESS_EXPR_BIT_AND, trigger, not_prev);

  // next_index = select(rising, mod(add(index, step), seq_len), index)
  egress_expr_t rev_step   = egress_expr_binary(EGRESS_EXPR_SUB, egress_expr_input(3), egress_expr_input(2));
  egress_expr_t fwd_step   = egress_expr_select(egress_expr_input(1), egress_expr_input(2), rev_step);
  egress_expr_t next_raw   = egress_expr_binary(EGRESS_EXPR_ADD, egress_expr_register(0), fwd_step);
  egress_expr_t next_mod   = egress_expr_binary(EGRESS_EXPR_MOD, next_raw, egress_expr_input(3));
  egress_expr_t next_index = egress_expr_select(rising, next_mod, egress_expr_register(0));

  // register 0: index, initial value int 0
  egress_value_t zero_idx = egress_value_int(0);
  egress_module_spec_add_register(spec, next_index, zero_idx);
  egress_value_free(zero_idx);
  egress_expr_free(next_index);

  // register 1: prev_trig = trigger, initial value int 0
  egress_expr_t prev_trig_next = egress_expr_input(0);
  egress_value_t zero_pt = egress_value_int(0);
  egress_module_spec_add_register(spec, prev_trig_next, zero_pt);
  egress_value_free(zero_pt);
  egress_expr_free(prev_trig_next);

  egress_expr_free(trigger);
  egress_expr_free(prev_trig);
  egress_expr_free(not_prev);
  egress_expr_free(rising);
  egress_expr_free(rev_step);
  egress_expr_free(fwd_step);
  egress_expr_free(next_raw);
  egress_expr_free(next_mod);

  return spec;
}

// ---- tests -------------------------------------------------------------------

static void test_scalar_module_processes()
{
  egress_graph_t g = egress_graph_new(256);
  ASSERT(g != nullptr);

  egress_module_spec_t spec = build_simple_scalar_spec();
  ASSERT_OK(egress_graph_add_module(g, "VCO1", spec));
  egress_module_spec_free(spec);

  // Set freq = 440.0
  egress_expr_t freq_expr = egress_expr_literal_float(440.0);
  ASSERT_OK(egress_graph_set_input_expr(g, "VCO1", 0, freq_expr));
  egress_expr_free(freq_expr);

  // Prime JIT then process several frames
  egress_graph_prime_jit(g);
  for (int i = 0; i < 32; ++i)
  {
    egress_graph_process(g);
  }

  egress_graph_free(g);
}

static void test_clock_module_default_array_input()
{
  // This test exercises the exact crash scenario: Clock has an array default
  // for ratios_in. On first process(), the JIT must re-initialize with the
  // array input and compile array-output kernels.
  egress_graph_t g = egress_graph_new(256);
  ASSERT(g != nullptr);

  egress_module_spec_t spec = build_clock_spec();
  ASSERT_OK(egress_graph_add_module(g, "Clock1", spec));
  egress_module_spec_free(spec);

  // freq = 2.0 Hz
  egress_expr_t freq_expr = egress_expr_literal_float(2.0);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 0, freq_expr));
  egress_expr_free(freq_expr);

  // ratios_in = [1.0]  (array with one element — default from module_library)
  egress_value_t item    = egress_value_float(1.0);
  egress_value_t arr_val = egress_value_array(&item, 1);
  egress_expr_t  arr_expr = egress_expr_literal_value(arr_val);
  egress_value_free(item);
  egress_value_free(arr_val);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 1, arr_expr));
  egress_expr_free(arr_expr);

  egress_graph_prime_jit(g);
  for (int i = 0; i < 32; ++i)
  {
    egress_graph_process(g);
  }

  egress_graph_free(g);
}

static void test_clock_module_multi_ratio_array()
{
  // Same as above but ratios_in has 3 elements.
  egress_graph_t g = egress_graph_new(256);
  ASSERT(g != nullptr);

  egress_module_spec_t spec = build_clock_spec();
  ASSERT_OK(egress_graph_add_module(g, "Clock1", spec));
  egress_module_spec_free(spec);

  egress_expr_t freq_expr = egress_expr_literal_float(1.0);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 0, freq_expr));
  egress_expr_free(freq_expr);

  egress_value_t items[3] = {
    egress_value_float(1.0),
    egress_value_float(2.0),
    egress_value_float(4.0),
  };
  egress_value_t arr_val  = egress_value_array(items, 3);
  egress_expr_t  arr_expr = egress_expr_literal_value(arr_val);
  for (int i = 0; i < 3; ++i) egress_value_free(items[i]);
  egress_value_free(arr_val);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 1, arr_expr));
  egress_expr_free(arr_expr);

  egress_graph_prime_jit(g);
  for (int i = 0; i < 32; ++i)
  {
    egress_graph_process(g);
  }

  egress_graph_free(g);
}

static void test_clock_wired_to_seq_trigger()
{
  // Clock1.output -> Seq1 (a simple passthrough for the trigger).
  // We can't instantiate IntSeq here without the full TS spec, but we can
  // verify Clock1's output is readable after processing.
  egress_graph_t g = egress_graph_new(256);
  ASSERT(g != nullptr);

  egress_module_spec_t spec = build_clock_spec();
  ASSERT_OK(egress_graph_add_module(g, "Clock1", spec));
  egress_module_spec_free(spec);

  // freq = 4.0, ratios_in = [1.0, 2.0]
  egress_expr_t freq_expr = egress_expr_literal_float(4.0);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 0, freq_expr));
  egress_expr_free(freq_expr);

  egress_value_t items[2] = { egress_value_float(1.0), egress_value_float(2.0) };
  egress_value_t arr_val  = egress_value_array(items, 2);
  egress_expr_t  arr_expr = egress_expr_literal_value(arr_val);
  egress_value_free(items[0]);
  egress_value_free(items[1]);
  egress_value_free(arr_val);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 1, arr_expr));
  egress_expr_free(arr_expr);

  // Add Clock1.output (output_id=0) as a tap so it is materialized
  ASSERT_OK(egress_graph_add_output(g, "Clock1", 0));

  egress_graph_prime_jit(g);
  for (int i = 0; i < 64; ++i)
  {
    egress_graph_process(g);
  }

  egress_graph_free(g);
}

// IntSeq-style module: 5 inputs (trigger, is_forward, step, seq_len, sequence/array)
// 2 outputs (value=sequence[index], index), 1 integer register (index=0)
// Mirrors the inline module_def from 31tet_otonal_seq.json.
static egress_module_spec_t build_intseq_spec()
{
  // 5 inputs, sample_rate=44100
  egress_module_spec_t spec = egress_module_spec_new(5, 44100.0);

  // inputs: 0=trigger, 1=is_forward, 2=step, 3=seq_len, 4=sequence(array)
  egress_expr_t trigger    = egress_expr_input(0);
  egress_expr_t is_forward = egress_expr_input(1);
  egress_expr_t step       = egress_expr_input(2);
  egress_expr_t seq_len    = egress_expr_input(3);
  egress_expr_t sequence   = egress_expr_input(4);
  egress_expr_t index_reg  = egress_expr_register(0);

  // output: value = sequence[index_reg]
  egress_expr_t value_out = egress_expr_binary(EGRESS_EXPR_INDEX, sequence, index_reg);
  egress_module_spec_add_output(spec, value_out);
  egress_expr_free(value_out);

  // output: index = index_reg
  egress_module_spec_add_output(spec, egress_expr_register(0));

  // next_index = select(trigger,
  //   mod(add(index_reg, select(is_forward, step, seq_len - step)), seq_len),
  //   index_reg)
  egress_expr_t rev_step   = egress_expr_binary(EGRESS_EXPR_SUB, egress_expr_input(3), egress_expr_input(2));
  egress_expr_t fwd_step   = egress_expr_select(egress_expr_input(1), egress_expr_input(2), rev_step);
  egress_expr_t next_raw   = egress_expr_binary(EGRESS_EXPR_ADD, egress_expr_register(0), fwd_step);
  egress_expr_t next_mod   = egress_expr_binary(EGRESS_EXPR_MOD, next_raw, egress_expr_input(3));
  egress_expr_t next_index = egress_expr_select(egress_expr_input(0), next_mod, egress_expr_register(0));

  // register 0: index, initial value int 0
  egress_value_t zero = egress_value_int(0);
  egress_module_spec_add_register(spec, next_index, zero);
  egress_value_free(zero);
  egress_expr_free(next_index);

  egress_expr_free(trigger);
  egress_expr_free(is_forward);
  egress_expr_free(step);
  egress_expr_free(seq_len);
  egress_expr_free(sequence);
  egress_expr_free(index_reg);
  egress_expr_free(rev_step);
  egress_expr_free(fwd_step);
  egress_expr_free(next_raw);
  egress_expr_free(next_mod);

  return spec;
}

static void test_intseq_with_clock()
{
  egress_graph_t g = egress_graph_new(256);
  ASSERT(g != nullptr);

  // Add Clock
  egress_module_spec_t clock_spec = build_clock_spec();
  ASSERT_OK(egress_graph_add_module(g, "Clock1", clock_spec));
  egress_module_spec_free(clock_spec);

  egress_expr_t freq_expr = egress_expr_literal_float(2.0);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 0, freq_expr));
  egress_expr_free(freq_expr);

  egress_value_t r_item  = egress_value_float(1.0);
  egress_value_t r_arr   = egress_value_array(&r_item, 1);
  egress_expr_t  r_expr  = egress_expr_literal_value(r_arr);
  egress_value_free(r_item);
  egress_value_free(r_arr);
  ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 1, r_expr));
  egress_expr_free(r_expr);

  // Add IntSeq
  egress_module_spec_t seq_spec = build_intseq_spec();
  ASSERT_OK(egress_graph_add_module(g, "Seq1", seq_spec));
  egress_module_spec_free(seq_spec);

  // Wire Clock1.output -> Seq1.trigger
  ASSERT_OK(egress_graph_connect(g, "Clock1", 0, "Seq1", 0));

  // Set remaining Seq1 inputs
  egress_expr_t fwd  = egress_expr_literal_bool(true);
  egress_expr_t stp  = egress_expr_literal_int(1);
  egress_expr_t slen = egress_expr_literal_int(8);
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 1, fwd));
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 2, stp));
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 3, slen));
  egress_expr_free(fwd);
  egress_expr_free(stp);
  egress_expr_free(slen);

  // sequence = [110, 137, 164, 192, 220, 192, 164, 137]
  double freqs[8] = {110.0, 137.56, 164.48, 192.38, 220.0, 192.38, 164.48, 137.56};
  egress_value_t seq_items[8];
  for (int i = 0; i < 8; i++) seq_items[i] = egress_value_float(freqs[i]);
  egress_value_t seq_arr  = egress_value_array(seq_items, 8);
  egress_expr_t  seq_expr = egress_expr_literal_value(seq_arr);
  for (int i = 0; i < 8; i++) egress_value_free(seq_items[i]);
  egress_value_free(seq_arr);
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 4, seq_expr));
  egress_expr_free(seq_expr);

  ASSERT_OK(egress_graph_add_output(g, "Seq1", 0));

  egress_graph_prime_jit(g);
  for (int i = 0; i < 128; ++i)
  {
    egress_graph_process(g);
  }

  egress_graph_free(g);
}

// Gate held high should only advance sequencer once (rising edge detection).
// Uses build_intseq_edge_detect_spec which has prev_trig register.
// Feed a constant 1.0 gate — index should go from 0→1 on the first frame
// and stay at 1 for all subsequent frames.
static void test_intseq_gate_triggers_once()
{
  const unsigned int BUF_LEN = 1; // 1 sample per process() call for precise control
  egress_graph_t g = egress_graph_new(BUF_LEN);
  ASSERT(g != nullptr);

  egress_module_spec_t spec = build_intseq_edge_detect_spec();
  ASSERT_OK(egress_graph_add_module(g, "Seq1", spec));
  egress_module_spec_free(spec);

  // trigger = constant 1.0 (gate held high — a Float, like Clock output)
  egress_expr_t trig = egress_expr_literal_float(1.0);
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 0, trig));
  egress_expr_free(trig);

  // is_forward = true, step = 1, seq_len = 4
  egress_expr_t fwd  = egress_expr_literal_bool(true);
  egress_expr_t stp  = egress_expr_literal_int(1);
  egress_expr_t slen = egress_expr_literal_int(4);
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 1, fwd));
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 2, stp));
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 3, slen));
  egress_expr_free(fwd);
  egress_expr_free(stp);
  egress_expr_free(slen);

  // sequence = [100, 200, 300, 400]
  egress_value_t items[4] = {
    egress_value_float(100.0), egress_value_float(200.0),
    egress_value_float(300.0), egress_value_float(400.0),
  };
  egress_value_t arr = egress_value_array(items, 4);
  egress_expr_t  seq = egress_expr_literal_value(arr);
  for (int i = 0; i < 4; i++) egress_value_free(items[i]);
  egress_value_free(arr);
  ASSERT_OK(egress_graph_set_input_expr(g, "Seq1", 4, seq));
  egress_expr_free(seq);

  // Tap the index output (output 1)
  size_t tap_id = egress_graph_add_output_tap(g, "Seq1", 1);
  ASSERT(tap_id != (size_t)-1);

  egress_graph_prime_jit(g);

  // Process 10 frames and collect index values
  double indices[10];
  for (int f = 0; f < 10; ++f)
  {
    egress_graph_process(g);
    size_t len = 0;
    const double * buf = egress_graph_tap_buffer(g, tap_id, &len);
    ASSERT(len == 1);
    indices[f] = buf[0];
  }

  // Frame 0: output reads initial index (0), rising edge advances it for next frame
  ASSERT(indices[0] == 0.0);
  // Frame 1+: index=1, gate held so no more advances
  for (int f = 1; f < 10; ++f)
  {
    ASSERT(indices[f] == 1.0);
  }

  egress_graph_free(g);
}

// Four IntSeqs + four scalar modules to mimic the 31tet_otonal_seq patch structure.
// Exercises primitive body fusion across multiple heterogeneous modules.
static void test_multi_module_fusion()
{
  egress_graph_t g = egress_graph_new(256);
  ASSERT(g != nullptr);

  // Clock1
  {
    egress_module_spec_t s = build_clock_spec();
    ASSERT_OK(egress_graph_add_module(g, "Clock1", s));
    egress_module_spec_free(s);
    egress_expr_t f = egress_expr_literal_float(1.5);
    ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 0, f));
    egress_expr_free(f);
    egress_value_t ri = egress_value_float(1.0);
    egress_value_t ra = egress_value_array(&ri, 1);
    egress_expr_t  re = egress_expr_literal_value(ra);
    egress_value_free(ri); egress_value_free(ra);
    ASSERT_OK(egress_graph_set_input_expr(g, "Clock1", 1, re));
    egress_expr_free(re);
  }

  double freqs[4][8] = {
    {110.00, 137.56, 164.48, 192.38, 220.00, 192.38, 164.48, 137.56},
    {137.56, 172.01, 205.66, 240.53, 275.12, 240.53, 205.66, 172.01},
    {164.48, 205.73, 245.93, 287.67, 328.97, 287.67, 245.93, 205.73},
    {192.38, 240.53, 287.67, 336.44, 384.75, 336.44, 287.67, 240.53},
  };

  const char* seq_names[4] = {"Seq1","Seq2","Seq3","Seq4"};
  const char* vco_names[4] = {"VCO1","VCO2","VCO3","VCO4"};

  for (int i = 0; i < 4; i++)
  {
    egress_module_spec_t s = build_intseq_spec();
    ASSERT_OK(egress_graph_add_module(g, seq_names[i], s));
    egress_module_spec_free(s);

    ASSERT_OK(egress_graph_connect(g, "Clock1", 0, seq_names[i], 0));

    egress_expr_t fwd  = egress_expr_literal_bool(true);
    egress_expr_t stp  = egress_expr_literal_int(1);
    egress_expr_t slen = egress_expr_literal_int(8);
    ASSERT_OK(egress_graph_set_input_expr(g, seq_names[i], 1, fwd));
    ASSERT_OK(egress_graph_set_input_expr(g, seq_names[i], 2, stp));
    ASSERT_OK(egress_graph_set_input_expr(g, seq_names[i], 3, slen));
    egress_expr_free(fwd); egress_expr_free(stp); egress_expr_free(slen);

    egress_value_t items[8];
    for (int j = 0; j < 8; j++) items[j] = egress_value_float(freqs[i][j]);
    egress_value_t arr = egress_value_array(items, 8);
    egress_expr_t  ex  = egress_expr_literal_value(arr);
    for (int j = 0; j < 8; j++) egress_value_free(items[j]);
    egress_value_free(arr);
    ASSERT_OK(egress_graph_set_input_expr(g, seq_names[i], 4, ex));
    egress_expr_free(ex);
  }

  // Four VCOs driven by sequencer output
  for (int i = 0; i < 4; i++)
  {
    egress_module_spec_t s = build_simple_scalar_spec();
    ASSERT_OK(egress_graph_add_module(g, vco_names[i], s));
    egress_module_spec_free(s);
    ASSERT_OK(egress_graph_connect(g, seq_names[i], 0, vco_names[i], 0));
    ASSERT_OK(egress_graph_add_output(g, vco_names[i], 0));
  }

  egress_graph_prime_jit(g);
  for (int i = 0; i < 128; ++i)
  {
    egress_graph_process(g);
  }

  egress_graph_free(g);
}

// ---- main --------------------------------------------------------------------

int main()
{
  printf("=== egress module process tests ===\n\n");

  run_test("scalar module: construct + 32 process() calls",
           test_scalar_module_processes);

  run_test("clock: array default input (ratios_in=[1.0]) + 32 process() calls",
           test_clock_module_default_array_input);

  run_test("clock: multi-ratio array input (ratios_in=[1,2,4]) + 32 process() calls",
           test_clock_module_multi_ratio_array);

  run_test("clock: two-ratio array + output tap + 64 process() calls",
           test_clock_wired_to_seq_trigger);

  run_test("intseq: integer-register module wired to clock + 128 process() calls",
           test_intseq_with_clock);

  run_test("intseq edge detect: held gate triggers only one index advance",
           test_intseq_gate_triggers_once);

  run_test("multi-module fusion: clock + 4 intseqs + 4 vcos + 128 frames",
           test_multi_module_fusion);

  printf("\n=== results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
