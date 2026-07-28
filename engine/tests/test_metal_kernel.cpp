/**
 * test_metal_kernel.cpp — Metal execution backend smoke (TROPICAL_METAL
 * builds; no audio device required, needs a Metal GPU).
 *
 * Exercises `tropical_runtime_load_ir_msl`: dual-load (JIT + MSL→PSO),
 * GPU per-block dispatch through tropical_runtime_process, sample-clock
 * continuity across blocks, hot-swap (double load) mid-stream, and the
 * set_sample_index test hook. Hand-written IR/MSL pairs mirror the two
 * emitters' ABIs exactly.
 */

#include "c_api/tropical_c.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int g_fail = 0;

#define ASSERT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      printf("FAIL\n    %s  (line %d): %s\n", #cond, __LINE__,               \
             tropical_last_error());                                          \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

#define ASSERT_NEAR(val, expect, tol)                                         \
  do {                                                                        \
    const double _v = (val), _e = (expect), _t = (tol);                       \
    if (std::fabs(_v - _e) > _t) {                                            \
      printf("FAIL\n    %s = %.10f, expected %.10f +/- %g  (line %d)\n",     \
             #val, _v, _e, _t, __LINE__);                                     \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

// The JIT side of the dual load — a constant kernel (never dispatched when
// the Metal side is live, but load_ir_msl compiles it regardless).
static const char* JIT_CONST_IR =
  "define void @kernel(ptr %inputs, ptr %registers, ptr %arrays, "
  "ptr %array_sizes, ptr %temps, double %sampleRate, i64 %start_sample_index, "
  "ptr %param_ptrs, ptr %output_buffer, i64 %buffer_length, ptr %slots) {\n"
  "entry:\n  br label %lc\n"
  "lc:\n  %s = phi i64 [0, %entry], [%sn, %lb]\n"
  "  %c = icmp ult i64 %s, %buffer_length\n"
  "  br i1 %c, label %lb, label %le\n"
  "lb:\n"
  "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
  "  store double 9.000000e-01, ptr %p, align 8\n"
  "  %sn = add i64 %s, 1\n  br label %lc\n"
  "le:\n  ret void\n}\n";

static const char* MANIFEST = R"({"schema":"tropical_plan_5",
  "config":{"sampleRate":44100},"register_count":0,
  "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
  "sinks":[],"slot_count":2,"slot_defaults":[0.25, 0]})";

// MSL ABI mirror of EmitMsl's header.
static std::string msl_kernel(const std::string& body)
{
  return std::string(
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct TropicalKernelConsts {\n"
    "    ulong start_sample_index;\n"
    "    float sample_rate;\n"
    "    uint  buffer_length;\n"
    "};\n"
    "kernel void tropical_kernel(\n"
    "    device float*                  output_buffer [[buffer(0)]],\n"
    "    constant float*                slots         [[buffer(1)]],\n"
    "    constant TropicalKernelConsts& k             [[buffer(2)]],\n"
    "    uint s [[thread_position_in_grid]])\n"
    "{\n"
    "    if (s >= k.buffer_length) { return; }\n"
    "    const long current_idx = long(k.start_sample_index) + long(s);\n")
    + body + "\n}\n";
}

/** 1. GPU ramp: output = current_idx, continuous across process() calls —
 *     proves the dispatch runs, the clock threads through, and blocks
 *     advance the sample index. */
static void test_metal_ramp()
{
  const unsigned int buf = 64;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  const std::string msl = msl_kernel(
    "    output_buffer[s] = float(current_idx);");
  ASSERT(tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                      msl.c_str(), msl.size(),
                                      MANIFEST, strlen(MANIFEST)));
  ASSERT(tropical_runtime_metal_pipeline_depth(rt) == 0);
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], (double)i, 1e-6);
  // Second block continues the clock (sample_index advanced by buf).
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], (double)(buf + i), 1e-6);
  // If the GPU were silently falling back to the JIT kernel, the output
  // would be the constant 0.9 — the ramp proves the Metal path ran.
  tropical_runtime_free(rt);
  printf("PASS  metal ramp + clock continuity\n");
}

/** 2. Slots reach the kernel (defaults from the manifest). */
static void test_metal_slots()
{
  const unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  const std::string msl = msl_kernel(
    "    output_buffer[s] = slots[0];");
  ASSERT(tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                      msl.c_str(), msl.size(),
                                      MANIFEST, strlen(MANIFEST)));
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.25, 1e-7);
  // Live slot write lands next block.
  tropical_runtime_set_slot(rt, 0, 0.75);
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.75, 1e-7);
  tropical_runtime_free(rt);
  printf("PASS  metal slot snapshot + live write\n");
}

/** 3. Hot-swap: load a second MSL kernel mid-stream; the sample clock
 *     carries over (the publish/flip contract). */
static void test_metal_hotswap()
{
  const unsigned int buf = 32;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  const std::string rampA = msl_kernel("    output_buffer[s] = float(current_idx);");
  ASSERT(tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                      rampA.c_str(), rampA.size(),
                                      MANIFEST, strlen(MANIFEST)));
  tropical_runtime_process(rt);   // clock now at 32

  const std::string rampB = msl_kernel(
    "    output_buffer[s] = float(current_idx) * as_type<float>(0x40000000u);"); // ×2
  ASSERT(tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                      rampB.c_str(), rampB.size(),
                                      MANIFEST, strlen(MANIFEST)));
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 2.0 * (buf + i), 1e-5);   // clock CONTINUED at 32
  tropical_runtime_free(rt);
  printf("PASS  metal hot-swap carries the sample clock\n");
}

// ── Banked coefficient columns (WS4: the Metal column crossing) ────────────
//
// A staged manifest advertising ONE coefficient column (array slot 0,
// capacity 4). The stage-0 coefficient kernel fills it host-side from
// slots[0] — column[k] = slots[0] + k, f64 bit-punned into the i64 storage
// exactly as EmitLlvm's Pack does — and the GPU kernel reads it from the
// packed `coeff_columns` buffer(3) binding.

static const char* STAGED_MANIFEST = R"({"schema":"tropical_plan_5",
  "config":{"sampleRate":44100},"register_count":0,
  "array_slot_count":1,"array_slot_sizes":[4],"array_slot_names":["col"],
  "coeff_array_slots":[0],
  "instance_functions":[],"sinks":[],"slot_count":2,"slot_defaults":[0.25, 0]})";

// Stage-0 coefficient kernel (run during control-snapshot publication with
// buffer_length = 1):
// arrays[0][k] = slots[0] + k for k = 0..3, stored as bitcast i64 — the
// same f64-punned view the JIT's Index loads.
static const char* COEFF_FILL_IR =
  "define void @kernel(ptr %inputs, ptr %registers, ptr %arrays, "
  "ptr %array_sizes, ptr %temps, double %sampleRate, i64 %start_sample_index, "
  "ptr %param_ptrs, ptr %output_buffer, i64 %buffer_length, ptr %slots) {\n"
  "entry:\n"
  "  %s0 = load double, ptr %slots, align 8\n"
  "  %a0p = getelementptr inbounds ptr, ptr %arrays, i64 0\n"
  "  %a0 = load ptr, ptr %a0p, align 8\n"
  "  %v0 = fadd double %s0, 0.000000e+00\n"
  "  %b0 = bitcast double %v0 to i64\n"
  "  %e0 = getelementptr inbounds i64, ptr %a0, i64 0\n"
  "  store i64 %b0, ptr %e0, align 8\n"
  "  %v1 = fadd double %s0, 1.000000e+00\n"
  "  %b1 = bitcast double %v1 to i64\n"
  "  %e1 = getelementptr inbounds i64, ptr %a0, i64 1\n"
  "  store i64 %b1, ptr %e1, align 8\n"
  "  %v2 = fadd double %s0, 2.000000e+00\n"
  "  %b2 = bitcast double %v2 to i64\n"
  "  %e2 = getelementptr inbounds i64, ptr %a0, i64 2\n"
  "  store i64 %b2, ptr %e2, align 8\n"
  "  %v3 = fadd double %s0, 3.000000e+00\n"
  "  %b3 = bitcast double %v3 to i64\n"
  "  %e3 = getelementptr inbounds i64, ptr %a0, i64 3\n"
  "  store i64 %b3, ptr %e3, align 8\n"
  "  ret void\n}\n";

// MSL ABI mirror of EmitMsl's COLUMN-BINDING header (buffer(3)).
static std::string msl_kernel_columns(const std::string& body)
{
  return std::string(
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct TropicalKernelConsts {\n"
    "    ulong start_sample_index;\n"
    "    float sample_rate;\n"
    "    uint  buffer_length;\n"
    "};\n"
    "kernel void tropical_kernel(\n"
    "    device float*                  output_buffer [[buffer(0)]],\n"
    "    constant float*                slots         [[buffer(1)]],\n"
    "    constant TropicalKernelConsts& k             [[buffer(2)]],\n"
    "    constant float*                coeff_columns [[buffer(3)]],\n"
    "    uint s [[thread_position_in_grid]])\n"
    "{\n"
    "    if (s >= k.buffer_length) { return; }\n"
    "    const long current_idx = long(k.start_sample_index) + long(s);\n")
    + body + "\n}\n";
}

/** 5. Banked columns, LIVE (sync path): the coefficient kernel fills the
 *     column host-side (run at load, then on every set_slot), process()
 *     uploads the captured generation, and the GPU reads real values from
 *     buffer(3). A slot write must land in the NEXT block's columns —
 *     exactly the scalar live-slot contract of test 2, but through the
 *     generation-buffered column crossing. */
static void test_metal_columns_live()
{
  const unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  const std::string msl = msl_kernel_columns(
    "    output_buffer[s] = coeff_columns[s % 4u];");
  ASSERT(tropical_runtime_load_ir_staged(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                         msl.c_str(), msl.size(),
                                         COEFF_FILL_IR, strlen(COEFF_FILL_IR),
                                         STAGED_MANIFEST, strlen(STAGED_MANIFEST)));
  // Load ran the coefficient kernel against the slot default (0.25):
  // column = [0.25, 1.25, 2.25, 3.25].
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 0.25 + (double)(i % 4), 1e-7);
  // Live knob: set_slot re-runs the coefficient kernel into a fresh
  // generation; the next block captures + uploads it.
  tropical_runtime_set_slot(rt, 0, 1.5);
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 1.5 + (double)(i % 4), 1e-7);
  tropical_runtime_free(rt);
  printf("PASS  metal coefficient columns reach the GPU + live knob refill\n");
}

/** 6. Banked columns, PIPELINED (TROPICAL_METAL_PIPELINE=1): columns ride
 *     per-ring-entry buffers copied at enqueue, so a knob move lands with
 *     the backwards-compatible D(=3)-block lag — never mid-flight, never
 *     torn. */
static void test_metal_columns_pipelined()
{
  setenv("TROPICAL_METAL_PIPELINE", "1", 1);
  const unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  if (rt == nullptr) { unsetenv("TROPICAL_METAL_PIPELINE"); ++g_fail; return; }

  const std::string msl = msl_kernel_columns(
    "    output_buffer[s] = coeff_columns[s % 4u];");
  const bool loaded =
    tropical_runtime_load_ir_staged(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                    msl.c_str(), msl.size(),
                                    COEFF_FILL_IR, strlen(COEFF_FILL_IR),
                                    STAGED_MANIFEST, strlen(STAGED_MANIFEST));
  unsetenv("TROPICAL_METAL_PIPELINE");   // create() read it at load
  if (!loaded)
  {
    printf("FAIL\n    pipelined staged load failed: %s\n", tropical_last_error());
    ++g_fail;
    tropical_runtime_free(rt);
    return;
  }
  ASSERT(tropical_runtime_metal_pipeline_depth(rt) == 3);
  const double* out = tropical_runtime_output_buffer(rt);
  // Block 1 primes the ring (3 futures @ old columns) and reads the first.
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 0.25 + (double)(i % 4), 1e-7);
  tropical_runtime_set_slot(rt, 0, 2.0);
  // Blocks 2..4 drain futures enqueued BEFORE the write (the D-block lag);
  // block 5 is the first enqueued after it.
  for (int b = 0; b < 3; ++b)
  {
    tropical_runtime_process(rt);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT_NEAR(out[i], 0.25 + (double)(i % 4), 1e-7);
  }
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 2.0 + (double)(i % 4), 1e-7);
  tropical_runtime_free(rt);
  printf("PASS  metal pipelined columns: per-ring upload, knob lands after the D-block lag\n");
}

/** 7. Qualification depth sweep. Every supported depth has the exact
 *     D-block slot-snapshot lag, while a clock jump drains queued futures
 *     and re-primes from the current slot snapshot immediately. */
static void test_metal_pipeline_depth_sweep()
{
  const unsigned int buf = 16;
  const std::string msl = msl_kernel("    output_buffer[s] = slots[0];");
  for (unsigned int depth = 1; depth <= 3; ++depth)
  {
    const std::string raw = std::to_string(depth);
    // The numeric seam is authoritative when both controls are present.
    // This exercises precedence at D=1 while the other rows exercise the
    // depth-only spelling.
    if (depth == 1) setenv("TROPICAL_METAL_PIPELINE", "1", 1);
    setenv("TROPICAL_METAL_PIPELINE_DEPTH", raw.c_str(), 1);
    tropical_runtime_t rt = tropical_runtime_new(buf);
    if (rt == nullptr)
    {
      unsetenv("TROPICAL_METAL_PIPELINE_DEPTH");
      ++g_fail;
      return;
    }
    const bool loaded =
      tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                   msl.c_str(), msl.size(),
                                   MANIFEST, strlen(MANIFEST));
    unsetenv("TROPICAL_METAL_PIPELINE_DEPTH");
    if (depth == 1) unsetenv("TROPICAL_METAL_PIPELINE");
    ASSERT(loaded);
    ASSERT(tropical_runtime_metal_pipeline_depth(rt) == depth);

    tropical_runtime_process(rt); // prime and consume the first old block
    tropical_runtime_set_slot(rt, 0, 0.75);
    for (unsigned int b = 0; b < depth; ++b)
    {
      tropical_runtime_process(rt);
      const double * out = tropical_runtime_output_buffer(rt);
      for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.25, 1e-7);
    }
    tropical_runtime_process(rt);
    const double * out = tropical_runtime_output_buffer(rt);
    for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.75, 1e-7);

    // A discontinuous clock move must discard every queued old snapshot.
    tropical_runtime_set_slot(rt, 0, 0.5);
    tropical_runtime_set_sample_index(rt, 1000000);
    tropical_runtime_process(rt);
    out = tropical_runtime_output_buffer(rt);
    for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.5, 1e-7);

    // Hot-swap builds a fresh ring at the carried coordinate. No completed
    // future from the old kernel may be emitted after publication.
    setenv("TROPICAL_METAL_PIPELINE_DEPTH", raw.c_str(), 1);
    const std::string replacement =
      msl_kernel("    output_buffer[s] = slots[0] + 1.0f;");
    const bool swapped =
      tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                   replacement.c_str(), replacement.size(),
                                   MANIFEST, strlen(MANIFEST));
    unsetenv("TROPICAL_METAL_PIPELINE_DEPTH");
    ASSERT(swapped);
    tropical_runtime_process(rt);
    out = tropical_runtime_output_buffer(rt);
    for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 1.25, 1e-7);
    tropical_runtime_free(rt);
  }
  printf("PASS  metal pipeline depths 1/2/3: lag + clock/hot-swap re-prime\n");
}

/** 8. Invalid qualification configuration refuses at Metal construction
 *     with a stable, actionable diagnostic. */
static void test_metal_pipeline_invalid_depth()
{
  setenv("TROPICAL_METAL_PIPELINE_DEPTH", "4", 1);
  tropical_runtime_t rt = tropical_runtime_new(16);
  ASSERT(rt != nullptr);
  const std::string msl = msl_kernel("    output_buffer[s] = slots[0];");
  const bool loaded =
    tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                 msl.c_str(), msl.size(),
                                 MANIFEST, strlen(MANIFEST));
  unsetenv("TROPICAL_METAL_PIPELINE_DEPTH");
  ASSERT(!loaded);
  ASSERT(std::strstr(tropical_last_error(),
                     "TROPICAL_METAL_PIPELINE_DEPTH must be an integer in [1,3]")
         != nullptr);
  tropical_runtime_free(rt);
  printf("PASS  metal invalid pipeline depth refuses clearly\n");
}

/** 9. A deterministic completion failure must reject the whole block before
 *     output copy, emit silence, increment sticky telemetry once, latch the
 *     failed kernel silent until replacement, and never strand either the
 *     synchronous or pipelined wait path. No DAC is opened. */
static void test_metal_dispatch_failure_fail_closed()
{
  using Clock = std::chrono::steady_clock;
  const unsigned int buf = 16;
  const std::string msl = msl_kernel(
    "    output_buffer[s] = slots[0];");

  for (const unsigned int depth : {0u, 3u})
  {
    setenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT", "2", 1);
    if (depth > 0)
      setenv("TROPICAL_METAL_PIPELINE_DEPTH", "3", 1);

    tropical_runtime_t rt = tropical_runtime_new(buf);
    if (!rt)
    {
      unsetenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT");
      unsetenv("TROPICAL_METAL_PIPELINE_DEPTH");
      ++g_fail;
      return;
    }
    const bool loaded =
      tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                   msl.c_str(), msl.size(),
                                   MANIFEST, strlen(MANIFEST));
    // create() captured both test controls; realtime dispatch never reads
    // the environment.
    unsetenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT");
    unsetenv("TROPICAL_METAL_PIPELINE_DEPTH");
    ASSERT(loaded);
    ASSERT(tropical_runtime_metal_pipeline_depth(rt) == depth);
    ASSERT(tropical_runtime_metal_dispatch_failure_count(rt) == 0);

    // Establish nonzero prior output so a failed command cannot pass by
    // leaving stale samples in FlatRuntime's output buffer.
    tropical_runtime_process(rt);
    const double * out = tropical_runtime_output_buffer(rt);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT_NEAR(out[i], 0.25, 1e-7);

    const auto failure_start = Clock::now();
    tropical_runtime_process(rt);
    const double failure_seconds =
      std::chrono::duration<double>(Clock::now() - failure_start).count();
    ASSERT(failure_seconds < 5.0);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT(out[i] == 0.0);
    ASSERT(tropical_runtime_metal_dispatch_failure_count(rt) == 1);

    // The failed kernel is latched: subsequent callbacks return silent without
    // another wait/dispatch and do not count one underlying command twice.
    const auto latched_start = Clock::now();
    tropical_runtime_process(rt);
    const double latched_seconds =
      std::chrono::duration<double>(Clock::now() - latched_start).count();
    ASSERT(latched_seconds < 1.0);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT(out[i] == 0.0);
    ASSERT(tropical_runtime_metal_dispatch_failure_count(rt) == 1);

    // A fresh MetalKernel clears the execution latch, while FlatRuntime's
    // monotonic evidence survives the hot-swap.
    if (depth > 0)
      setenv("TROPICAL_METAL_PIPELINE_DEPTH", "3", 1);
    const bool reloaded =
      tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                   msl.c_str(), msl.size(),
                                   MANIFEST, strlen(MANIFEST));
    unsetenv("TROPICAL_METAL_PIPELINE_DEPTH");
    ASSERT(reloaded);
    tropical_runtime_process(rt);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT_NEAR(out[i], 0.25, 1e-7);
    ASSERT(tropical_runtime_metal_dispatch_failure_count(rt) == 1);
    tropical_runtime_free(rt);
  }
  printf("PASS  metal failure latches silent until fresh kernel (sync and pipeline)\n");
}

/** 4. set_sample_index repositions the clock (the render --start hook). */
static void test_metal_set_index()
{
  const unsigned int buf = 8;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  const std::string msl = msl_kernel("    output_buffer[s] = float(current_idx);");
  ASSERT(tropical_runtime_load_ir_msl(rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
                                      msl.c_str(), msl.size(),
                                      MANIFEST, strlen(MANIFEST)));
  tropical_runtime_set_sample_index(rt, 1000000);
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 1000000.0 + i, 1e-1);   // f32 at 1e6: ulp ≈ 0.0625
  tropical_runtime_free(rt);
  printf("PASS  metal set_sample_index\n");
}

int main()
{
  test_metal_ramp();
  test_metal_slots();
  test_metal_hotswap();
  test_metal_set_index();
  test_metal_columns_live();
  test_metal_columns_pipelined();
  test_metal_pipeline_depth_sweep();
  test_metal_pipeline_invalid_depth();
  test_metal_dispatch_failure_fail_closed();
  if (g_fail == 0) printf("ALL METAL TESTS PASSED\n");
  return g_fail == 0 ? 0 : 1;
}
