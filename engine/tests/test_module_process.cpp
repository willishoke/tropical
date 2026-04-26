/**
 * test_module_process.cpp
 *
 * Exercises the FlatRuntime C API (tropical_runtime_*) and JIT code paths
 * without an audio device.  Plans are specified as tropical_plan_4 JSON strings,
 * compiled to native kernels via compile_flat_program, and processed in-memory.
 */

#include "c_api/tropical_c.h"
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
      const char* err = tropical_last_error();                                  \
      if (err && err[0]) printf("    last error: %s\n", err);                \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

#define ASSERT_OK(call)                                                       \
  do {                                                                        \
    bool _ok = (call);                                                        \
    if (!_ok) {                                                               \
      const char* _err = tropical_last_error();                                 \
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
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0],
    "instructions": [
      {"tag":"Mul","dst":0,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":2.0}],"loop_count":1,"strides":[]},
      {"tag":"Sub","dst":1,"args":[{"kind":"reg","slot":0},{"kind":"const","val":1.0}],"loop_count":1,"strides":[]},
      {"tag":"Mul","dst":2,"args":[{"kind":"reg","slot":1},{"kind":"const","val":10.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":3,"args":[{"kind":"reg","slot":2},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Div","dst":4,"args":[{"kind":"const","val":440.0},{"kind":"rate"}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":5,"args":[{"kind":"state_reg","slot":0},{"kind":"reg","slot":4}],"loop_count":1,"strides":[]},
      {"tag":"Mod","dst":6,"args":[{"kind":"reg","slot":5},{"kind":"const","val":1.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":7,"args":[{"kind":"reg","slot":6},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 8,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [3],
    "register_targets": [7]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));
  tropical_runtime_process(rt);

  const double* buf = tropical_runtime_output_buffer(rt);
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

  tropical_runtime_free(rt);
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
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [],
    "register_names": [],
    "outputs": [0, 1],
    "instructions": [
      {"tag":"Add","dst":0,"args":[{"kind":"const","val":5.0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":1,"args":[{"kind":"const","val":-3.0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 2,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [0, 1],
    "register_targets": []
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));
  tropical_runtime_process(rt);

  const double* buf = tropical_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  for (unsigned int i = 0; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 0.1, 1e-9);
  }

  tropical_runtime_free(rt);
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
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  // Plan A: output = reg(0), register update = mod(reg(0) + 440/sr, 1)
  std::string plan_a = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0],
    "instructions": [
      {"tag":"Add","dst":0,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Div","dst":1,"args":[{"kind":"const","val":440.0},{"kind":"rate"}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":2,"args":[{"kind":"state_reg","slot":0},{"kind":"reg","slot":1}],"loop_count":1,"strides":[]},
      {"tag":"Mod","dst":3,"args":[{"kind":"reg","slot":2},{"kind":"const","val":1.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":4,"args":[{"kind":"reg","slot":3},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 5,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [0],
    "register_targets": [4]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan_a.c_str(), plan_a.size()));

  // Process 4 buffers to accumulate phase
  for (int i = 0; i < 4; ++i) {
    tropical_runtime_process(rt);
  }

  // Read last sample of last buffer — phase should be well above 0
  const double* buf = tropical_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);
  double phase_before = buf[buf_len - 1];
  ASSERT(phase_before > 0.01);

  // Plan B: output = reg(0) * 5.0, same register name "phase"
  std::string plan_b = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0],
    "instructions": [
      {"tag":"Mul","dst":0,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":5.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":1,"args":[{"kind":"reg","slot":0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Div","dst":2,"args":[{"kind":"const","val":440.0},{"kind":"rate"}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":3,"args":[{"kind":"state_reg","slot":0},{"kind":"reg","slot":2}],"loop_count":1,"strides":[]},
      {"tag":"Mod","dst":4,"args":[{"kind":"reg","slot":3},{"kind":"const","val":1.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":5,"args":[{"kind":"reg","slot":4},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 6,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [1],
    "register_targets": [5]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan_b.c_str(), plan_b.size()));
  tropical_runtime_process(rt);

  // First sample of new plan should use the transferred phase, not 0.
  // Output = phase * 5.0, audio = that / 20.0
  const double* buf2 = tropical_runtime_output_buffer(rt);
  ASSERT(buf2 != nullptr);
  double first_output_audio = buf2[0];
  // If state was NOT preserved, output would be 0*5/20 = 0.
  // With preserved state, it should be noticeably positive.
  ASSERT(first_output_audio > 0.001);

  tropical_runtime_free(rt);
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
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0, 1.0],
    "register_names": ["idx", "idx2"],
    "outputs": [0],
    "instructions": [
      {"tag":"Pack","dst":0,"args":[{"kind":"const","val":10.0},{"kind":"const","val":20.0},{"kind":"const","val":30.0},{"kind":"const","val":40.0}],"loop_count":1,"strides":[]},
      {"tag":"Index","dst":0,"args":[{"kind":"array_reg","slot":0},{"kind":"state_reg","slot":0}],"loop_count":1,"strides":[]},
      {"tag":"Pack","dst":1,"args":[{"kind":"const","val":10.0},{"kind":"const","val":20.0},{"kind":"const","val":30.0},{"kind":"const","val":40.0}],"loop_count":1,"strides":[]},
      {"tag":"Index","dst":1,"args":[{"kind":"array_reg","slot":1},{"kind":"state_reg","slot":1}],"loop_count":1,"strides":[]},
      {"tag":"Div","dst":2,"args":[{"kind":"reg","slot":0},{"kind":"reg","slot":1}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":3,"args":[{"kind":"reg","slot":2},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":4,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":5,"args":[{"kind":"state_reg","slot":1},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 6,
    "array_slot_count": 2,
    "array_slot_sizes": [4, 4],
    "output_targets": [3],
    "register_targets": [4, 5]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));
  tropical_runtime_process(rt);

  const double* buf = tropical_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  // 10.0 / 20.0 = 0.5, audio = 0.5 / 20.0 = 0.025
  for (unsigned int i = 0; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 0.025, 1e-9);
  }

  tropical_runtime_free(rt);
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
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0],
    "register_names": ["counter"],
    "outputs": [0],
    "instructions": [
      {"tag":"Add","dst":0,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":1,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":1.0}],"loop_count":1,"strides":[]},
      {"tag":"Mod","dst":2,"args":[{"kind":"reg","slot":1},{"kind":"const","val":8.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":3,"args":[{"kind":"reg","slot":2},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 4,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [0],
    "register_targets": [3]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));

  double expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1};
  for (int i = 0; i < 10; ++i) {
    tropical_runtime_process(rt);
    const double* buf = tropical_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    ASSERT_NEAR(buf[0], expected[i] / 20.0, 1e-9);
  }

  tropical_runtime_free(rt);
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
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0],
    "register_names": ["phase"],
    "outputs": [0],
    "instructions": [
      {"tag":"Greater","dst":0,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":4.0}],"loop_count":1,"strides":[]},
      {"tag":"Select","dst":1,"args":[{"kind":"reg","slot":0},{"kind":"const","val":1.0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":2,"args":[{"kind":"reg","slot":1},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":3,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":1.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":4,"args":[{"kind":"reg","slot":3},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 5,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [2],
    "register_targets": [4]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));

  double expected[] = {0, 0, 0, 0, 0, 1, 1, 1};
  for (int i = 0; i < 8; ++i) {
    tropical_runtime_process(rt);
    const double* buf = tropical_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    ASSERT_NEAR(buf[0], expected[i] / 20.0, 1e-9);
  }

  tropical_runtime_free(rt);
}

/**
 * 7. Multi-register interaction (clock-like)
 *
 * Register 0 (phase): init=0, update=mod(add(reg(0), div(1.0, sample_rate)), 1.0)
 * Register 1 (gate):  init=0, update=select(lt(reg(0), 0.5), 1.0, 0.0)
 * Output 0: reg(1)
 * Outputs: [0]
 *
 * Sample 0: gate=init(0)=0, then gate update sets to 1.0 (since phase=0 < 0.5)
 * Sample 1+: gate reads 1.0 (set last sample), stays 1.0
 */
static void test_multi_register_clock()
{
  const unsigned int buf_len = 256;
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0, 0.0],
    "register_names": ["phase", "gate"],
    "outputs": [0],
    "instructions": [
      {"tag":"Add","dst":0,"args":[{"kind":"state_reg","slot":1},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Div","dst":1,"args":[{"kind":"const","val":1.0},{"kind":"rate"}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":2,"args":[{"kind":"state_reg","slot":0},{"kind":"reg","slot":1}],"loop_count":1,"strides":[]},
      {"tag":"Mod","dst":3,"args":[{"kind":"reg","slot":2},{"kind":"const","val":1.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":4,"args":[{"kind":"reg","slot":3},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Less","dst":5,"args":[{"kind":"state_reg","slot":0},{"kind":"const","val":0.5}],"loop_count":1,"strides":[]},
      {"tag":"Select","dst":6,"args":[{"kind":"reg","slot":5},{"kind":"const","val":1.0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":7,"args":[{"kind":"reg","slot":6},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 8,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [0],
    "register_targets": [4, 7]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));
  tropical_runtime_process(rt);

  const double* buf = tropical_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  // Sample 0: gate=init(0)=0
  ASSERT_NEAR(buf[0], 0.0, 1e-9);

  // Samples 1..255: gate should be 1.0 (phase is still well under 0.5)
  // Audio = 1.0 / 20.0 = 0.05
  for (unsigned int i = 1; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 1.0 / 20.0, 1e-9);
  }

  tropical_runtime_free(rt);
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
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [],
    "register_names": [],
    "outputs": [0, 1],
    "instructions": [
      {"tag":"Add","dst":0,"args":[{"kind":"const","val":3.0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]},
      {"tag":"Add","dst":1,"args":[{"kind":"const","val":7.0},{"kind":"const","val":0.0}],"loop_count":1,"strides":[]}
    ],
    "register_count": 2,
    "array_slot_count": 0,
    "array_slot_sizes": [],
    "output_targets": [0, 1],
    "register_targets": []
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));
  tropical_runtime_process(rt);

  const double* buf = tropical_runtime_output_buffer(rt);
  ASSERT(buf != nullptr);

  for (unsigned int i = 0; i < buf_len; ++i) {
    ASSERT_NEAR(buf[i], 0.5, 1e-9);
  }

  tropical_runtime_free(rt);
}

/**
 * 9. Typed integer bitwise ops (LFSR-style)
 *
 * State register 0 (seed): init=1, type=int
 * Update: BitXor(RShift(state, 1), BitAnd(Mul(BitAnd(state, 1), 0xB400), 0xFFFF))
 * This is a 16-bit LFSR step. With scalar_type annotations, the JIT should
 * emit native i64 ops (no FPToSI/SIToFP round-trips).
 *
 * Output 0: state value (as float / 20.0 via output mix)
 * Process 4 steps and verify LFSR sequence: 1, 0x5A00, 0x2D00, 0x1680
 */
static void test_typed_int_bitwise()
{
  const unsigned int buf_len = 1;
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  // LFSR: next = (state >> 1) ^ ((state & 1) * 0xB400 & 0xFFFF)
  // reg0 = state_reg(0)                               [int]
  // reg1 = BitAnd(reg0, 1)                             [int] — extract LSB
  // reg2 = Mul(reg1, 0xB400)                           [int] — feedback mask
  // reg3 = BitAnd(reg2, 0xFFFF)                        [int] — 16-bit mask
  // reg4 = RShift(reg0, 1)                             [int] — shift right
  // reg5 = BitXor(reg4, reg3)                          [int] — next state
  // reg6 = SIToFP equivalent via Mul(reg0, 1.0)        [float] — for output
  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [1.0],
    "register_names": ["lfsr_state"],
    "register_types": ["int"],
    "outputs": [0],
    "instructions": [
      {"tag":"Add","dst":0,"result_type":"int","args":[
        {"kind":"state_reg","slot":0,"scalar_type":"int"},
        {"kind":"const","val":0.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"BitAnd","dst":1,"result_type":"int","args":[
        {"kind":"reg","slot":0,"scalar_type":"int"},
        {"kind":"const","val":1.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"Mul","dst":2,"result_type":"int","args":[
        {"kind":"reg","slot":1,"scalar_type":"int"},
        {"kind":"const","val":46080.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"BitAnd","dst":3,"result_type":"int","args":[
        {"kind":"reg","slot":2,"scalar_type":"int"},
        {"kind":"const","val":65535.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"RShift","dst":4,"result_type":"int","args":[
        {"kind":"reg","slot":0,"scalar_type":"int"},
        {"kind":"const","val":1.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"BitXor","dst":5,"result_type":"int","args":[
        {"kind":"reg","slot":4,"scalar_type":"int"},
        {"kind":"reg","slot":3,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"Mul","dst":6,"result_type":"float","args":[
        {"kind":"reg","slot":0,"scalar_type":"int"},
        {"kind":"const","val":1.0,"scalar_type":"float"}
      ],"loop_count":1,"strides":[]}
    ],
    "register_count": 7,
    "array_slot_sizes": [],
    "output_targets": [6],
    "register_targets": [5]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));

  // LFSR sequence starting from seed=1:
  // step 0: state=1, output=1
  //   LSB=1, feedback=0xB400 & 0xFFFF=0xB400, shift=0, next=0^0xB400=0xB400
  //   Wait: 0xB400 = 46080
  // step 1: state=0xB400=46080
  //   LSB=0, feedback=0, shift=0xB400>>1=0x5A00=23040, next=0x5A00
  // step 2: state=0x5A00=23040
  //   LSB=0, feedback=0, shift=0x5A00>>1=0x2D00=11520, next=0x2D00
  // step 3: state=0x2D00=11520
  //   LSB=0, feedback=0, shift=0x2D00>>1=0x1680=5760, next=0x1680

  double expected[] = {1.0, 46080.0, 23040.0, 11520.0};
  for (int i = 0; i < 4; ++i) {
    tropical_runtime_process(rt);
    const double* buf = tropical_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    ASSERT_NEAR(buf[0], expected[i] / 20.0, 1e-6);
  }

  tropical_runtime_free(rt);
}

/**
 * 10. Bool-typed comparison chain with Select
 *
 * No state registers. Pure combinational:
 * reg0 = tick (sample index, int)
 * reg1 = Greater(reg0, 2)   → bool
 * reg2 = Less(reg0, 6)      → bool
 * reg3 = BitAnd(reg1, reg2) → int (bool promoted to int for bitwise)
 * reg4 = Select(reg3, 10.0, 0.0) → float
 *
 * Output: 10/20=0.5 when 2 < tick < 6, else 0.
 * Tick starts at 0 for first buffer, 1 for second, etc. (buf_len=1)
 */
static void test_typed_bool_select()
{
  const unsigned int buf_len = 1;
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [],
    "register_names": [],
    "outputs": [0],
    "instructions": [
      {"tag":"Add","dst":0,"result_type":"int","args":[
        {"kind":"tick","scalar_type":"int"},
        {"kind":"const","val":0.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"Greater","dst":1,"result_type":"bool","args":[
        {"kind":"reg","slot":0,"scalar_type":"int"},
        {"kind":"const","val":2.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"Less","dst":2,"result_type":"bool","args":[
        {"kind":"reg","slot":0,"scalar_type":"int"},
        {"kind":"const","val":6.0,"scalar_type":"int"}
      ],"loop_count":1,"strides":[]},
      {"tag":"BitAnd","dst":3,"result_type":"int","args":[
        {"kind":"reg","slot":1,"scalar_type":"bool"},
        {"kind":"reg","slot":2,"scalar_type":"bool"}
      ],"loop_count":1,"strides":[]},
      {"tag":"Select","dst":4,"result_type":"float","args":[
        {"kind":"reg","slot":3,"scalar_type":"int"},
        {"kind":"const","val":10.0,"scalar_type":"float"},
        {"kind":"const","val":0.0,"scalar_type":"float"}
      ],"loop_count":1,"strides":[]}
    ],
    "register_count": 5,
    "array_slot_sizes": [],
    "output_targets": [4],
    "register_targets": []
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));

  // tick: 0,1,2,3,4,5,6,7
  // >2:  F,F,F,T,T,T,T,T
  // <6:  T,T,T,T,T,T,F,F
  // AND: F,F,F,T,T,T,F,F
  // out: 0,0,0,0.5,0.5,0.5,0,0
  double expected[] = {0, 0, 0, 0.5, 0.5, 0.5, 0, 0};
  for (int i = 0; i < 8; ++i) {
    tropical_runtime_process(rt);
    const double* buf = tropical_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    ASSERT_NEAR(buf[0], expected[i], 1e-9);
  }

  tropical_runtime_free(rt);
}

/**
 * 11. Float temp writeback into int register (Phase 5 failing-first)
 *
 * A float-typed temp value is flagged as the register_target for an
 * int-typed state register. The JIT writeback must FPToSI-coerce the
 * float to int on store — NOT bitcast the f64 bit pattern into i64,
 * which would produce a massive garbage integer.
 *
 * Plan:
 *   register_types=["int"], state_init=[0]
 *   Mul(0.5, 3.0) → temp 0 (float=1.5)
 *   register_target[0] = 0        (float temp → int register)
 *   Output: Mul(state_reg:int, 1.0) → float, yields /20.0 per mix
 *
 * With a correct FPToSI writeback, step 0 output uses reg=0 (init), next
 * reg = trunc(1.5) = 1. Step 1 output uses reg=1 → 1/20 = 0.05.
 * With the current bitcast writeback, bits 0x3ff8000000000000 (~4.6e18)
 * flood the register and the step-1 output blows up past any sane range.
 */
static void test_float_to_int_register_writeback()
{
  const unsigned int buf_len = 1;
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  std::string plan = R"({
    "schema": "tropical_plan_4",
    "config": { "sampleRate": 44100.0 },
    "state_init": [0.0],
    "register_names": ["counter"],
    "register_types": ["int"],
    "outputs": [0],
    "instructions": [
      {"tag":"Mul","dst":0,"result_type":"float","args":[
        {"kind":"const","val":0.5,"scalar_type":"float"},
        {"kind":"const","val":3.0,"scalar_type":"float"}
      ],"loop_count":1,"strides":[]},
      {"tag":"Mul","dst":1,"result_type":"float","args":[
        {"kind":"state_reg","slot":0,"scalar_type":"int"},
        {"kind":"const","val":1.0,"scalar_type":"float"}
      ],"loop_count":1,"strides":[]}
    ],
    "register_count": 2,
    "array_slot_sizes": [],
    "output_targets": [1],
    "register_targets": [0]
  })";

  ASSERT_OK(tropical_runtime_load_plan(rt, plan.c_str(), plan.size()));

  // Step 0: reg starts at 0, output = 0/20 = 0.
  tropical_runtime_process(rt);
  const double* buf0 = tropical_runtime_output_buffer(rt);
  ASSERT(buf0 != nullptr);
  ASSERT_NEAR(buf0[0], 0.0, 1e-9);

  // Step 1: reg updated via FPToSI(1.5) = 1, output = 1/20 = 0.05.
  tropical_runtime_process(rt);
  const double* buf1 = tropical_runtime_output_buffer(rt);
  ASSERT(buf1 != nullptr);
  ASSERT_NEAR(buf1[0], 1.0 / 20.0, 1e-9);

  tropical_runtime_free(rt);
}

/**
 * 12. Cast ops: to_int / to_bool / to_float
 *
 * Verifies truncate-toward-zero (FPToSI) semantics of ToInt, not floor:
 *   to_int(-0.5) == 0  (not -1)
 *   to_int( 0.5) == 0
 *   to_int(-1.7) == -1 (not -2)
 *   to_int( 1.7) == 1
 * to_float/to_bool round-trip correctness.
 *
 * Single-step plan: sweeps a 4-case switch driven by sample_index.
 * Output: to_float(to_int(cand)) / 10 — so we can read back the cast int.
 */
static void test_cast_ops()
{
  const unsigned int buf_len = 1;
  tropical_runtime_t rt = tropical_runtime_new(buf_len);
  ASSERT(rt != nullptr);

  // One plan per value we want to test; keep it simple and iterate.
  struct Case { double in; double expected_int; };
  Case cases[] = {
    { -0.5,  0.0},   // FPToSI truncs toward zero
    {  0.5,  0.0},
    { -1.7, -1.0},
    {  1.7,  1.0},
    {  3.0,  3.0},
  };

  for (const auto & c : cases) {
    char plan_buf[2048];
    snprintf(plan_buf, sizeof(plan_buf), R"({
      "schema": "tropical_plan_4",
      "config": { "sampleRate": 44100.0 },
      "state_init": [],
      "register_names": [],
      "register_types": [],
      "outputs": [0],
      "instructions": [
        {"tag":"ToInt","dst":0,"result_type":"int","args":[
          {"kind":"const","val":%.6f,"scalar_type":"float"}
        ],"loop_count":1,"strides":[]},
        {"tag":"ToFloat","dst":1,"result_type":"float","args":[
          {"kind":"reg","slot":0,"scalar_type":"int"}
        ],"loop_count":1,"strides":[]}
      ],
      "register_count": 2,
      "array_slot_sizes": [],
      "output_targets": [1],
      "register_targets": []
    })", c.in);

    ASSERT_OK(tropical_runtime_load_plan(rt, plan_buf, strlen(plan_buf)));
    tropical_runtime_process(rt);
    const double* buf = tropical_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    // Output mix divides by 20 (standard mix).
    ASSERT_NEAR(buf[0], c.expected_int / 20.0, 1e-9);
  }

  // to_bool: any nonzero → 1, zero → 0. Then to_float brings it back.
  struct BoolCase { double in; double expected; };
  BoolCase bool_cases[] = {
    { 0.0, 0.0},
    {-3.2, 1.0},
    { 7.5, 1.0},
  };
  for (const auto & c : bool_cases) {
    char plan_buf[2048];
    snprintf(plan_buf, sizeof(plan_buf), R"({
      "schema": "tropical_plan_4",
      "config": { "sampleRate": 44100.0 },
      "state_init": [],
      "register_names": [],
      "register_types": [],
      "outputs": [0],
      "instructions": [
        {"tag":"ToBool","dst":0,"result_type":"bool","args":[
          {"kind":"const","val":%.6f,"scalar_type":"float"}
        ],"loop_count":1,"strides":[]},
        {"tag":"ToFloat","dst":1,"result_type":"float","args":[
          {"kind":"reg","slot":0,"scalar_type":"bool"}
        ],"loop_count":1,"strides":[]}
      ],
      "register_count": 2,
      "array_slot_sizes": [],
      "output_targets": [1],
      "register_targets": []
    })", c.in);

    ASSERT_OK(tropical_runtime_load_plan(rt, plan_buf, strlen(plan_buf)));
    tropical_runtime_process(rt);
    const double* buf = tropical_runtime_output_buffer(rt);
    ASSERT(buf != nullptr);
    ASSERT_NEAR(buf[0], c.expected / 20.0, 1e-9);
  }

  tropical_runtime_free(rt);
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
  run_test("typed int bitwise (LFSR)",       test_typed_int_bitwise);
  run_test("typed bool comparison + select", test_typed_bool_select);
  run_test("float→int register writeback coercion", test_float_to_int_register_writeback);
  run_test("cast ops (to_int/to_bool/to_float)", test_cast_ops);

  printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
