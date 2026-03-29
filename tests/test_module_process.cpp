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

  printf("\n=== results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
