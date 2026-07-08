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

#include <cmath>
#include <cstdio>
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
  if (g_fail == 0) printf("ALL METAL TESTS PASSED\n");
  return g_fail == 0 ? 0 : 1;
}
