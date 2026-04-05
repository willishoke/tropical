/**
 * test_module_process.cpp
 *
 * Exercises the FlatRuntime C API (egress_runtime_*) and JIT code paths
 * without an audio device.  Plans are specified as JSON strings, compiled
 * to native kernels, and processed in-memory.
 */

#include "c_api/egress_c.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

// ---- tiny test harness -------------------------------------------------------

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

// ---- tests ------------------------------------------------------------------

/**
 * 1. Sawtooth oscillator
 *
 * One register (phase), init=0.  Update: mod(add(reg(0), div(440.0, sample_rate)), 1.0)
 * Output 0: mul(sub(mul(reg(0), 2.0), 1.0), 10.0)  — maps [0,1) to [-10,10)
 * Outputs: [0]
 *
 * Audio = output / 20.0, so saw ranges [-0.5, 0.5).
 * After one sample, phase = 440/44100 ~= 0.00998.
 */
static void test_sawtooth()
{
  const unsigned int buf_len = 256;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [
      { "op": "mul", "args": [
        { "op": "sub", "args": [
          { "op": "mul", "args": [{ "op": "reg", "id": 0 }, 2.0] },
          1.0
        ]},
        10.0
      ]}
    ],
    "register_exprs": [
      { "op": "mod", "args": [
        { "op": "add", "args": [
          { "op": "reg", "id": 0 },
          { "op": "div", "args": [440.0, { "op": "sample_rate" }] }
        ]},
        1.0
      ]}
    ],
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan.c_str(), plan.size()));
  egress_runtime_process(rt);

  const double* buf = egress_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  // Sample 0: phase=0 at start, output = (0*2 - 1)*10 = -10, audio = -10/20 = -0.5
  ASSERT_NEAR(buf[0], -0.5, 1e-6);

  // Sample 1: phase = 440/44100, output = (phase*2 - 1)*10, audio = that/20
  double phase1 = 440.0 / 44100.0;
  double expected1 = (phase1 * 2.0 - 1.0) * 10.0 / 20.0;
  ASSERT_NEAR(buf[1], expected1, 1e-6);

  // Check several samples are increasing before the first phase wrap.
  // Phase wraps at sample ~44100/440 ~= 100, so check the first 50.
  for (unsigned int i = 1; i < 50; ++i) {
    ASSERT(buf[i] > buf[i - 1]);
  }

  egress_runtime_free(rt);
}

/**
 * 2. Two outputs with mix
 *
 * Output 0: constant 5.0
 * Output 1: constant -3.0
 * Outputs: [0, 1]
 *
 * Audio = (5.0 + -3.0) / 20.0 = 0.1
 */
static void test_two_outputs_mix()
{
  const unsigned int buf_len = 32;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [5.0, -3.0],
    "register_exprs": [],
    "state_init": [],
    "register_names": [],
    "outputs": [0, 1]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan.c_str(), plan.size()));
  egress_runtime_process(rt);

  const double* buf = egress_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  for (unsigned int i = 0; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 0.1, 1e-9);
  }

  egress_runtime_free(rt);
}

/**
 * 3. Hot-swap preserves state
 *
 * Load sawtooth plan, process a few buffers so phase accumulates,
 * then load a new plan with the same register name.  The phase
 * should carry over.
 */
static void test_hot_swap_preserves_state()
{
  const unsigned int buf_len = 64;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  // Plan A: sawtooth, phase register named "phase"
  std::string plan_a = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [{ "op": "reg", "id": 0 }],
    "register_exprs": [
      { "op": "mod", "args": [
        { "op": "add", "args": [
          { "op": "reg", "id": 0 },
          { "op": "div", "args": [440.0, { "op": "sample_rate" }] }
        ]},
        1.0
      ]}
    ],
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan_a.c_str(), plan_a.size()));

  // Process 4 buffers to accumulate phase
  for (int i = 0; i < 4; ++i) {
    egress_runtime_process(rt);
  }

  // Read last sample of last buffer — phase should be well above 0
  const double* buf = egress_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);
  double phase_before = buf[buf_len - 1];
  ASSERT(phase_before > 0.01);

  // Plan B: same structure, same register name "phase", but scale output differently
  // Output: mul(reg(0), 5.0)
  std::string plan_b = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [{ "op": "mul", "args": [{ "op": "reg", "id": 0 }, 5.0] }],
    "register_exprs": [
      { "op": "mod", "args": [
        { "op": "add", "args": [
          { "op": "reg", "id": 0 },
          { "op": "div", "args": [440.0, { "op": "sample_rate" }] }
        ]},
        1.0
      ]}
    ],
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan_b.c_str(), plan_b.size()));
  egress_runtime_process(rt);

  // First sample of new plan should use the transferred phase, not 0.
  // Output = phase * 5.0, audio = that / 20.0
  // The phase should be near where plan A left off (plus one increment).
  const double* buf2 = egress_runtime_output_buffer(rt);
  ASSERT(buf2 != nullptr);
  double first_output_audio = buf2[0];
  // If state was NOT preserved, output would be 0*5/20 = 0.
  // With preserved state, it should be noticeably positive.
  ASSERT(first_output_audio > 0.001);

  egress_runtime_free(rt);
}

/**
 * 4. Array literal in expression
 *
 * Register 0 (idx): init=0, update=identity (stays 0)
 * Register 1 (idx2): init=1, update=identity (stays 1)
 * Output 0: div(index([10.0, 20.0, 30.0, 40.0], reg(0)),
 *               index([10.0, 20.0, 30.0, 40.0], reg(1)))
 *         = 10.0 / 20.0 = 0.5
 *
 * Audio = 0.5 / 20.0 = 0.025
 */
static void test_array_literal()
{
  const unsigned int buf_len = 8;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [
      { "op": "div", "args": [
        { "op": "index", "args": [[10.0, 20.0, 30.0, 40.0], { "op": "reg", "id": 0 }] },
        { "op": "index", "args": [[10.0, 20.0, 30.0, 40.0], { "op": "reg", "id": 1 }] }
      ]}
    ],
    "register_exprs": [
      { "op": "reg", "id": 0 },
      { "op": "reg", "id": 1 }
    ],
    "state_init": [0.0, 1.0],
    "register_names": ["idx", "idx2"],
    "outputs": [0]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan.c_str(), plan.size()));
  egress_runtime_process(rt);

  const double* buf = egress_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  // 10.0 / 20.0 = 0.5, audio = 0.5 / 20.0 = 0.025
  for (unsigned int i = 0; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 0.025, 1e-9);
  }

  egress_runtime_free(rt);
}

/**
 * 5. Integer counter with modular wrap
 *
 * Register 0 (counter): init=0, update=mod(add(reg(0), 1), 8)
 * Output 0: reg(0)
 * Outputs: [0]
 *
 * Process 10 single-sample buffers.
 * Counter sequence: 0, 1, 2, 3, 4, 5, 6, 7, 0, 1
 *   (output reads register BEFORE update)
 *
 * Audio = counter / 20.0
 */
static void test_counter_wrap()
{
  const unsigned int buf_len = 1;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [{ "op": "reg", "id": 0 }],
    "register_exprs": [
      { "op": "mod", "args": [
        { "op": "add", "args": [{ "op": "reg", "id": 0 }, 1.0] },
        8.0
      ]}
    ],
    "state_init": [0.0],
    "register_names": ["counter"],
    "outputs": [0]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan.c_str(), plan.size()));

  double expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1};
  for (int i = 0; i < 10; ++i) {
    egress_runtime_process(rt);
    const double* buf = egress_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    ASSERT_NEAR(buf[0], expected[i] / 20.0, 1e-9);
  }

  egress_runtime_free(rt);
}

/**
 * 6. Select/conditional expression
 *
 * Register 0 (phase): init=0, update=add(reg(0), 1)
 * Output 0: select(gt(reg(0), 4), 1.0, 0.0)
 *   — gate goes high when phase > 4
 *
 * Process 8 single-sample buffers.
 * Phase sequence (at read time): 0, 1, 2, 3, 4, 5, 6, 7
 * Gate output:                   0, 0, 0, 0, 0, 1, 1, 1
 *
 * Audio = gate / 20.0
 */
static void test_select_conditional()
{
  const unsigned int buf_len = 1;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [
      { "op": "select", "args": [
        { "op": "gt", "args": [{ "op": "reg", "id": 0 }, 4.0] },
        1.0,
        0.0
      ]}
    ],
    "register_exprs": [
      { "op": "add", "args": [{ "op": "reg", "id": 0 }, 1.0] }
    ],
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan.c_str(), plan.size()));

  double expected[] = {0, 0, 0, 0, 0, 1, 1, 1};
  for (int i = 0; i < 8; ++i) {
    egress_runtime_process(rt);
    const double* buf = egress_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    ASSERT_NEAR(buf[0], expected[i] / 20.0, 1e-9);
  }

  egress_runtime_free(rt);
}

/**
 * 7. Multi-register interaction (clock-like)
 *
 * Register 0 (phase): init=0, update=mod(add(reg(0), div(1.0, sample_rate)), 1.0)
 * Register 1 (gate):  init=0, update=select(lt(reg(0), 0.5), 1.0, 0.0)
 * Output 0: reg(1)
 * Outputs: [0]
 *
 * The gate reads phase BEFORE its update this sample.
 * Phase ramps from 0 to 1 over 44100 samples.
 * For the first ~22050 samples, phase < 0.5 so gate update writes 1.0.
 * But gate output reads gate BEFORE update, so there is a 1-sample delay.
 *
 * Process one 256-sample buffer.  Gate should be ~1.0 for first half of a
 * 44100-sample cycle (we are well within that at 256 samples).
 *
 * Sample 0: gate reads init(0)=0, then gate update sets to 1.0 (since phase=0 < 0.5)
 * Sample 1+: gate reads 1.0 (set last sample), stays 1.0
 */
static void test_multi_register_clock()
{
  const unsigned int buf_len = 256;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [{ "op": "reg", "id": 1 }],
    "register_exprs": [
      { "op": "mod", "args": [
        { "op": "add", "args": [
          { "op": "reg", "id": 0 },
          { "op": "div", "args": [1.0, { "op": "sample_rate" }] }
        ]},
        1.0
      ]},
      { "op": "select", "args": [
        { "op": "lt", "args": [{ "op": "reg", "id": 0 }, 0.5] },
        1.0,
        0.0
      ]}
    ],
    "state_init": [0.0, 0.0],
    "register_names": ["phase", "gate"],
    "outputs": [0]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan.c_str(), plan.size()));
  egress_runtime_process(rt);

  const double* buf = egress_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  // Sample 0: gate=init(0)=0
  ASSERT_NEAR(buf[0], 0.0, 1e-9);

  // Samples 1..255: gate should be 1.0 (phase is still well under 0.5)
  // Audio = 1.0 / 20.0 = 0.05
  for (unsigned int i = 1; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 1.0 / 20.0, 1e-9);
  }

  egress_runtime_free(rt);
}

/**
 * 8. Multiple outputs summed
 *
 * Output 0: constant 3.0
 * Output 1: constant 7.0
 * Outputs: [0, 1]
 *
 * Audio = (3.0 / 20.0) + (7.0 / 20.0) = 10.0 / 20.0 = 0.5
 */
static void test_multiple_outputs_summed()
{
  const unsigned int buf_len = 16;
  egress_runtime_t rt = egress_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "egress_plan_2",
    "config": { "sample_rate": 44100.0 },
    "output_exprs": [3.0, 7.0],
    "register_exprs": [],
    "state_init": [],
    "register_names": [],
    "outputs": [0, 1]
  })";

  ASSERT_OK(egress_runtime_load_plan(rt, plan.c_str(), plan.size()));
  egress_runtime_process(rt);

  const double* buf = egress_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  for (unsigned int i = 0; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 0.5, 1e-9);
  }

  egress_runtime_free(rt);
}

// ---- main -------------------------------------------------------------------

int main()
{
  printf("test_module_process (FlatRuntime API)\n");

  run_test("sawtooth oscillator",            test_sawtooth);
  run_test("two outputs with mix",           test_two_outputs_mix);
  run_test("hot-swap preserves state",       test_hot_swap_preserves_state);
  run_test("array literal in expression",    test_array_literal);
  run_test("integer counter with mod wrap",  test_counter_wrap);
  run_test("select/conditional expression",  test_select_conditional);
  run_test("multi-register clock",           test_multi_register_clock);
  run_test("multiple outputs summed",        test_multiple_outputs_summed);

  printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
