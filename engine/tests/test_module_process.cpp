/**
 * test_module_process.cpp
 *
 * Exercises the current C++ engine with hand-written LLVM IR:
 * `tropical_runtime_load_ir` → compile_ir_text → JIT → run, plus the
 * device-boundary clamp. Lean's EmitLlvm owns production codegen;
 * arithmetic/op correctness lives in the Lean `tropicaltest` gates.
 * These tests cover the engine half only. They do not assert plan
 * instruction semantics or per-sample state.
 */

#include "c_api/tropical_c.h"
#include "dac/TropicalDAC.hpp"   // kDeviceOutputBound / clamp_to_device_bound
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

// Keep the existing public stats ABI frozen: new qualification telemetry uses
// separate accessors, never appended fields that older callers did not
// allocate.
static_assert(sizeof(tropical_dac_stats_t) == 40);
static_assert(offsetof(tropical_dac_stats_t, overrun_count) == 32);

static void run_test(const char* name, std::function<void()> fn)
{
  printf("  %-60s", name);
  fflush(stdout);
  fn();
  printf("PASS\n");
  ++g_pass;
}

#define ASSERT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      printf("FAIL\n    assertion failed: %s  (line %d)\n", #cond, __LINE__);\
      const char* err = tropical_last_error();                                \
      if (err && err[0]) printf("    last error: %s\n", err);                \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

#define ASSERT_OK(call)                                                       \
  do {                                                                        \
    bool _ok = (call);                                                        \
    if (!_ok) {                                                               \
      const char* _err = tropical_last_error();                               \
      printf("FAIL\n    %s returned false\n    last error: %s\n",            \
             #call, _err ? _err : "(none)");                                  \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

#define ASSERT_NEAR(val, expected, tol)                                       \
  do {                                                                        \
    double _v = (val), _e = (expected), _t = (tol);                           \
    if (std::fabs(_v - _e) > _t) {                                            \
      printf("FAIL\n    %s = %.10f, expected %.10f +/- %g  (line %d)\n",     \
             #val, _v, _e, _t, __LINE__);                                     \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

// The fused kernel ABI (must match NumericKernelFn). Lean's EmitLlvm emits
// exactly this signature; these hand kernels mirror it.
static const char* KERNEL_SIG =
  "ptr %inputs, ptr %registers, ptr %arrays, ptr %array_sizes, ptr %temps, "
  "double %sampleRate, i64 %start_sample_index, ptr %param_ptrs, "
  "ptr %output_buffer, i64 %buffer_length, ptr %slots";

static std::string wrap_loop(const std::string& body)
{
  return std::string("define void @kernel(") + KERNEL_SIG + ") {\n"
    "entry:\n  br label %lc\n"
    "lc:\n  %s = phi i64 [0, %entry], [%sn, %lb]\n"
    "  %c = icmp ult i64 %s, %buffer_length\n"
    "  br i1 %c, label %lb, label %le\n"
    "lb:\n" + body +
    "  %sn = add i64 %s, 1\n  br label %lc\n"
    "le:\n  ret void\n}\n";
}

/**
 * 1. Constant kernel via load_ir — proves textual IR parses, JITs, runs.
 */
static void test_ir_constant()
{
  const unsigned int buf = 64;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  std::string ir = wrap_loop(
    "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 5.000000e-01, ptr %p, align 8\n");
  std::string manifest = R"({"schema":"tropical_plan_5","config":{"sampleRate":44100},
    "register_count":0,"array_slot_count":0,"array_slot_sizes":[],
    "instance_functions":[],"sinks":[],"slot_count":0})";

  ASSERT_OK(tropical_runtime_load_ir(rt, ir.c_str(), ir.size(),
                                     manifest.c_str(), manifest.size()));
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.5, 1e-12);

  tropical_runtime_free(rt);
}

// A sample-index ramp via load_ir — closed-form, no state. Exercises the
// `%current_idx` (Tick source) argument and float<->i64 store path. CF-only
// removed per-sample registers, so the kernel is a pure function of the
// sample index.
static const char* RAMP_BODY =
  "  %idx = add i64 %start_sample_index, %s\n"
  "  %v = sitofp i64 %idx to double\n"
  "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
  "  store double %v, ptr %op, align 8\n";

static const char* RAMP_MANIFEST = R"({"schema":"tropical_plan_5",
  "config":{"sampleRate":44100},"register_count":0,
  "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
  "sinks":[],"slot_count":0})";

/**
 * 2. Closed-form ramp — output 0,1,2,… as a function of the sample index,
 *    continuing across process() because the sample counter advances.
 */
static void test_ir_index_ramp()
{
  const unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  std::string ir = wrap_loop(RAMP_BODY);
  ASSERT_OK(tropical_runtime_load_ir(rt, ir.c_str(), ir.size(),
                                     RAMP_MANIFEST, strlen(RAMP_MANIFEST)));

  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], (double)i, 1e-12);

  // Second buffer continues from buf (the sample index advances).
  tropical_runtime_process(rt);
  const double* out2 = tropical_runtime_output_buffer(rt);
  ASSERT_NEAR(out2[0], (double)buf, 1e-12);

  tropical_runtime_free(rt);
}

// ─────────────────────────────────────────────────────────────
// The device-boundary clamp (the safety-class gate)
// ─────────────────────────────────────────────────────────────

/**
 * `clamp_to_device_bound` is the last thing between a computed value and a
 * speaker (TropicalDAC's callback applies it per sample, after the fade). Two
 * properties, both load-bearing:
 *
 *   BOUNDEDNESS — nothing leaves the device boundary outside [−C, C], for any
 *   input at all, including gross excursions and non-finite values. This is the
 *   property the gate exists for.
 *
 *   BIT-TRANSPARENCY — every non-NaN value already inside [−C, C] passes
 *   through UNCHANGED, bit for bit: ±0.0 keep their signs, denormals survive,
 *   and the endpoints themselves are fixed points. This is what makes the
 *   clamp invisible to every existing golden.
 *
 * The transparency list deliberately includes 163.5 (the modal vocabulary's
 * legitimate peak at the top of the resonance knob, ≈ Q·master_gain) and 233.7
 * (the measured i64 modal-datapath rail incident). BOTH pass through untouched
 * at C = 256, and the second one is an asserted non-property, not an oversight:
 * the clamp does not and cannot discriminate that rail, because the burst and
 * the music are the same order of magnitude. See kDeviceOutputBound. The rail
 * is fixed at its source; this gate bounds the unbounded.
 *
 * MUTATION: delete the clamp from TropicalDAC::fill_buffer (or widen
 * kDeviceOutputBound past 1e9) and the boundedness assertions below fail.
 */
static void test_device_bound_clamp()
{
  const double C = kDeviceOutputBound;

  // Bit-transparency below C — compare BITS, not values, so signed zero counts.
  const double transparent[] = {
    0.0, -0.0, 1.0, -1.0, 1.807261 /* the corpus peak */, -1.807261,
    163.5 /* the modal vocabulary at res = 1: legitimate, must survive */,
    233.7 /* the rail incident: below C, passes through — see the docstring */,
    C, -C, 4.9406564584124654e-324 /* min denormal */, -2.2250738585072014e-308,
  };
  for (double v : transparent)
  {
    const double got = clamp_to_device_bound(v);
    uint64_t vb, gb;
    std::memcpy(&vb, &v, 8);
    std::memcpy(&gb, &got, 8);
    ASSERT(vb == gb);
  }

  // Boundedness above C.
  const double over[] = { 256.0000001, 257.0, 1e3, 1e9, 6.8719476736e10 /* 2^36,
                          the hard-wrap magnitude an unrecoverable i64 rail
                          produces */, std::numeric_limits<double>::infinity() };
  for (double v : over)
  {
    ASSERT(clamp_to_device_bound(v) == C);
    ASSERT(clamp_to_device_bound(-v) == -C);
  }

  // NaN is silenced to −C (ordered compare fails, so the low arm takes −C).
  // Crucially it does NOT pass through: a NaN reaching a DAC is a click at
  // best. This is the one input the clamp deliberately does not preserve.
  const double got_nan =
      clamp_to_device_bound(std::numeric_limits<double>::quiet_NaN());
  ASSERT(!std::isnan(got_nan));
  ASSERT(got_nan == -C);

  // The bound itself: a power of two, above the vocabulary's legitimate reach
  // (Q = 0.55·80^res peaks at 44, times the 3.7 master gain ≈ 163), and far
  // below a hard i64 wrap.
  ASSERT(C == 256.0);
  ASSERT(C > 163.5);
  ASSERT(C < 6.8719476736e10);
}

static void test_dac_histogram_contract()
{
  ASSERT(tropical_dac_callback_histogram_bin_count() == 20001);
  ASSERT(tropical_dac_callback_histogram_bin_width_ns() == 1000);
  ASSERT(tropical_dac_callback_histogram_overflow_floor_ns() == 20000000);
  ASSERT(tropical_dac_get_stats_epoch(nullptr) == 0);
}

int main()
{
  printf("test_module_process (load_ir engine)\n");

  run_test("constant kernel via load_ir",   test_ir_constant);
  run_test("closed-form index ramp",        test_ir_index_ramp);
  run_test("device-boundary clamp",         test_device_bound_clamp);
  run_test("fixed DAC histogram/ABI contract", test_dac_histogram_contract);

  printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
