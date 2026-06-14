/**
 * test_module_process.cpp
 *
 * Exercises the surviving C++ engine — `tropical_runtime_load_ir`
 * (compile_ir_text → JIT → run) and hot-swap state transfer — with
 * hand-written LLVM IR. The C++ plan compiler was retired in Phase 2
 * (Lean's EmitLlvm owns codegen); arithmetic/op correctness now lives in
 * the Lean `tropicaltest` gates. These tests cover the engine half: that
 * textual IR matching the kernel ABI parses, JITs, runs, and that named
 * register state transfers across a load_ir hot-swap.
 */

#include "c_api/tropical_c.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

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

// A counter kernel: output = registers[0] (float-encoded), then += 1.
// Exercises the i64-backed register encoding (bitcast double<->i64).
static const char* COUNTER_BODY =
  "  %rp = getelementptr inbounds i64, ptr %registers, i64 0\n"
  "  %raw = load i64, ptr %rp, align 8\n"
  "  %v = bitcast i64 %raw to double\n"
  "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
  "  store double %v, ptr %op, align 8\n"
  "  %v1 = fadd double %v, 1.000000e+00\n"
  "  %nb = bitcast double %v1 to i64\n"
  "  store i64 %nb, ptr %rp, align 8\n";

static const char* COUNTER_MANIFEST = R"({"schema":"tropical_plan_5",
  "config":{"sampleRate":44100},"register_count":1,"state_init":[0.0],
  "register_names":["counter"],"register_types":["float"],
  "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
  "sinks":[],"slot_count":0})";

/**
 * 2. Stateful counter — output 0,1,2,… and state persists across process().
 */
static void test_ir_counter_state()
{
  const unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  std::string ir = wrap_loop(COUNTER_BODY);
  ASSERT_OK(tropical_runtime_load_ir(rt, ir.c_str(), ir.size(),
                                     COUNTER_MANIFEST, strlen(COUNTER_MANIFEST)));

  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], (double)i, 1e-12);

  // Second buffer continues from buf (state held in registers[0]).
  tropical_runtime_process(rt);
  const double* out2 = tropical_runtime_output_buffer(rt);
  ASSERT_NEAR(out2[0], (double)buf, 1e-12);

  tropical_runtime_free(rt);
}

/**
 * 3. Hot-swap state transfer via load_ir — a second kernel sharing the
 *    register name "counter" sees the accumulated value, not 0.
 */
static void test_ir_hot_swap_state()
{
  const unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  std::string counter_ir = wrap_loop(COUNTER_BODY);
  ASSERT_OK(tropical_runtime_load_ir(rt, counter_ir.c_str(), counter_ir.size(),
                                     COUNTER_MANIFEST, strlen(COUNTER_MANIFEST)));
  tropical_runtime_process(rt);  // counter advances to `buf`

  // Reader kernel: outputs registers[0] without incrementing. Same
  // register name "counter", so hot-swap transfers the state by name.
  std::string reader_ir = wrap_loop(
    "  %rp = getelementptr inbounds i64, ptr %registers, i64 0\n"
    "  %raw = load i64, ptr %rp, align 8\n"
    "  %v = bitcast i64 %raw to double\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  ASSERT_OK(tropical_runtime_load_ir(rt, reader_ir.c_str(), reader_ir.size(),
                                     COUNTER_MANIFEST, strlen(COUNTER_MANIFEST)));
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  // State transferred by name: the reader sees the counter's value.
  ASSERT_NEAR(out[0], (double)buf, 1e-12);

  tropical_runtime_free(rt);
}

int main()
{
  printf("test_module_process (load_ir engine)\n");

  run_test("constant kernel via load_ir",        test_ir_constant);
  run_test("stateful counter + register encoding", test_ir_counter_state);
  run_test("hot-swap state transfer via load_ir", test_ir_hot_swap_state);

  printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
