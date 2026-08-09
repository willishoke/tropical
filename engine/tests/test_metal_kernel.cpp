/**
 * test_metal_kernel.cpp — Metal execution backend smoke (TROPICAL_METAL
 * builds; no audio device required, needs a Metal GPU).
 *
 * Exercises `tropical_runtime_load_ir_msl`: dual-load (JIT + MSL→PSO),
 * worker-prepared GPU epochs consumed by tropical_runtime_process, sample-clock
 * continuity across blocks, hot-swap (double load) mid-stream, and the
 * set_sample_index test hook. Hand-written IR/MSL pairs mirror the two
 * emitters' ABIs exactly.
 */

#include "c_api/tropical_c.h"
#include "metal/MetalKernel.hpp"
#include "metal/MetalRenderWorker.hpp"
#include "runtime/FlatRuntime.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;

static bool wait_for_true(
  const std::atomic<bool> & value,
  std::chrono::milliseconds timeout = std::chrono::milliseconds(2000))
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!value.load(std::memory_order_acquire)
         && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return value.load(std::memory_order_acquire);
}

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

static const char* JIT_SLOT0_IR =
  "define void @kernel(ptr %inputs, ptr %registers, ptr %arrays, "
  "ptr %array_sizes, ptr %temps, double %sampleRate, i64 %start_sample_index, "
  "ptr %param_ptrs, ptr %output_buffer, i64 %buffer_length, ptr %slots) {\n"
  "entry:\n"
  "  %value = load double, ptr %slots, align 8\n"
  "  br label %lc\n"
  "lc:\n  %s = phi i64 [0, %entry], [%sn, %lb]\n"
  "  %c = icmp ult i64 %s, %buffer_length\n"
  "  br i1 %c, label %lb, label %le\n"
  "lb:\n"
  "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
  "  store double %value, ptr %p, align 8\n"
  "  %sn = add i64 %s, 1\n  br label %lc\n"
  "le:\n  ret void\n}\n";

static const char* MANIFEST = R"({"schema":"tropical_plan_6",
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
  ASSERT(tropical_runtime_metal_render_tile_frames(rt) == 512);
  ASSERT(tropical_runtime_metal_worker_capacity_frames(rt) == 2048);
  tropical_metal::reset_callback_thread_violation_count();
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], (double)i, 1e-6);
  // Second block continues the clock (sample_index advanced by buf).
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], (double)(buf + i), 1e-6);
  ASSERT(tropical_metal::callback_thread_violation_count() == 0);
  // If the GPU were silently falling back to the JIT kernel, the output
  // would be the constant 0.9 — the ramp proves the Metal path ran.
  tropical_runtime_free(rt);
  printf("PASS  metal ramp + clock continuity\n");
}

static void test_offline_renderer_waits_for_worker_refill()
{
  const unsigned int buf = 512;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);
  const std::string msl = msl_kernel(
    "    output_buffer[s] = float(current_idx);");
  ASSERT(tropical_runtime_load_ir_msl(
    rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
    msl.c_str(), msl.size(), MANIFEST, strlen(MANIFEST)));

  const double * output = tropical_runtime_output_buffer(rt);
  constexpr uint64_t offline_blocks = 1000;
  for (uint64_t block = 0; block < offline_blocks; ++block)
  {
    ASSERT(tropical_runtime_process_offline(rt));
    for (uint32_t sample = 0; sample < buf; ++sample)
      ASSERT_NEAR(
        output[sample],
        static_cast<double>(block * buf + sample), 1e-5);
  }

  auto * runtime =
    static_cast<tropical_runtime::FlatRuntime *>(rt);
  const uint64_t before_start = runtime->current_sample_index();
  runtime->prepare_realtime_start(4);
  ASSERT(runtime->current_sample_index() == before_start);
  ASSERT(tropical_runtime_metal_render_starvation_count(rt) == 0);

  // The readiness barrier leaves the exact next tile intact for callback 1.
  tropical_runtime_process(rt);
  for (uint32_t sample = 0; sample < buf; ++sample)
    ASSERT_NEAR(
      output[sample],
      static_cast<double>(before_start + sample), 1e-5);
  ASSERT(tropical_runtime_metal_render_starvation_count(rt) == 0);
  ASSERT(tropical_runtime_metal_epoch_tag_mismatch_count(rt) == 0);
  tropical_runtime_free(rt);
  printf(
    "PASS  1,000 offline blocks refill and DAC readiness preserves callback 1\n");
}

static void test_callback_thread_provenance()
{
  const std::string msl = msl_kernel(
    "    output_buffer[s] = float(current_idx);");
  std::string error;
  auto kernel = tropical_metal::create(msl, 32, 0, 0, error);
  ASSERT(kernel != nullptr);
  std::array<double, 32> output{};

  tropical_metal::reset_callback_thread_violation_count();
  tropical_metal::set_audio_callback_thread(true);
  const bool callback_render = tropical_metal::render_tile(
    *kernel, nullptr, 0, nullptr, 0, 44100.0, 0,
    static_cast<uint32_t>(output.size()), output.data());
  tropical_metal::set_audio_callback_thread(false);
  ASSERT(!callback_render);
  ASSERT(tropical_metal::callback_thread_violation_count() == 1);

  ASSERT(tropical_metal::render_tile(
    *kernel, nullptr, 0, nullptr, 0, 44100.0, 0,
    static_cast<uint32_t>(output.size()), output.data()));
  for (uint32_t i = 0; i < output.size(); ++i)
    ASSERT_NEAR(output[i], static_cast<double>(i), 1e-6);
  tropical_metal::reset_callback_thread_violation_count();
  printf("PASS  callback-thread Metal provenance guard\n");
}

static void test_cooperative_dispatch_geometry()
{
  const std::string msl =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct TropicalKernelConsts { ulong start_sample_index; float sample_rate; uint buffer_length; };\n"
    "kernel void tropical_kernel(device float* out [[buffer(0)]], "
    "constant float* slots [[buffer(1)]], "
    "constant TropicalKernelConsts& k [[buffer(2)]], "
    "uint3 group [[threadgroup_position_in_grid]], "
    "uint lane [[thread_index_in_threadgroup]], "
    "uint3 width [[threads_per_threadgroup]]) {\n"
    "  threadgroup float shared_value;\n"
    "  const uint sample = group.x;\n"
    "  if (sample >= k.buffer_length) return;\n"
    "  if (lane == 0u) shared_value = float(k.start_sample_index + sample);\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (lane == width.x - 1u) out[sample] = shared_value;\n"
    "}\n";
  std::string error;
  auto kernel = tropical_metal::create(
    msl, 64, 0, 0, error,
    tropical_metal::DispatchKind::SampleThreadgroups, 64);
  ASSERT(kernel != nullptr);
  ASSERT(tropical_metal::dispatch_kind(*kernel)
         == tropical_metal::DispatchKind::SampleThreadgroups);
  ASSERT(tropical_metal::threadgroup_width(*kernel) > 0);
  ASSERT(tropical_metal::threadgroup_scratch_bytes(*kernel) == 64);
  std::array<double, 64> output{};
  ASSERT(tropical_metal::render_tile(
    *kernel, nullptr, 0, nullptr, 0, 44100.0, 100,
    static_cast<uint32_t>(output.size()), output.data()));
  for (uint32_t i = 0; i < output.size(); ++i)
    ASSERT_NEAR(output[i], static_cast<double>(100 + i), 1e-6);

  error.clear();
  auto over_cap = tropical_metal::create(
    msl, 64, 0, 0, error,
    tropical_metal::DispatchKind::SampleThreadgroups,
    std::numeric_limits<uint32_t>::max());
  ASSERT(over_cap == nullptr);
  ASSERT(error.find("scratch exceeds device publication limit")
         != std::string::npos);
  printf("PASS  cooperative threadgroup geometry + scratch refusal\n");
}

static void test_independent_device_and_render_quanta()
{
  setenv("TROPICAL_METAL_RENDER_TILE_FRAMES", "512", 1);
  const std::string msl = msl_kernel(
    "    output_buffer[s] = float(current_idx);");
  for (const unsigned int device_frames : {128U, 256U, 512U})
  {
    tropical_runtime_t rt = tropical_runtime_new(device_frames);
    ASSERT(rt != nullptr);
    ASSERT(tropical_runtime_metal_render_tile_frames(rt) == 0);
    ASSERT(tropical_runtime_load_ir_msl(
      rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
      msl.c_str(), msl.size(), MANIFEST, strlen(MANIFEST)));
    ASSERT(tropical_runtime_metal_render_tile_frames(rt) == 512);
    const double * output = tropical_runtime_output_buffer(rt);
    const unsigned callbacks = 512 / device_frames;
    for (unsigned callback = 0; callback < callbacks; ++callback)
    {
      tropical_runtime_process(rt);
      for (unsigned int i = 0; i < device_frames; ++i)
        ASSERT_NEAR(
          output[i],
          static_cast<double>(callback * device_frames + i), 1e-5);
    }
    tropical_runtime_free(rt);
  }
  unsetenv("TROPICAL_METAL_RENDER_TILE_FRAMES");
  printf("PASS  independent Bdev 128/256/512 and Rgpu 512\n");
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
  for (unsigned int block = 0; block < 512 / buf; ++block)
  {
    tropical_runtime_process(rt);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT_NEAR(out[i], 0.25, 1e-7);
  }
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
  const double* out = tropical_runtime_output_buffer(rt);
  for (unsigned int block = 0; block < 512 / buf; ++block)
  {
    tropical_runtime_process(rt);
    const uint64_t start = buf + static_cast<uint64_t>(block) * buf;
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT_NEAR(out[i], static_cast<double>(start + i), 1e-5);
  }
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 2.0 * (buf + 512 + i), 1e-5);
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

static const char* STAGED_MANIFEST = R"({"schema":"tropical_plan_6",
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

/** 5. Banked columns in immutable render-epoch snapshots: the coefficient
 *     kernel fills the column host-side at load and on every set_slot, then
 *     the worker uploads one coherent captured generation for buffer(3). */
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
  for (unsigned int block = 0; block < 512 / buf; ++block)
  {
    tropical_runtime_process(rt);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT_NEAR(out[i], 0.25 + (double)(i % 4), 1e-7);
  }
  tropical_runtime_process(rt);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 1.5 + (double)(i % 4), 1e-7);
  tropical_runtime_free(rt);
  printf("PASS  metal coefficient columns reach the GPU + live knob refill\n");
}

/** Exact E for all four parameter disciplines on the epoch worker path. */
static void test_metal_effective_dispatch_epochs()
{
  constexpr unsigned int buf = 16;
  constexpr double sr = 44100.0;
  const std::string msl = msl_kernel("    output_buffer[s] = slots[0];");
  const std::string manifest = R"({"schema":"tropical_plan_6",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":8,
    "slot_names":["param:bank.freq","param:canary.morph#v0",
      "param:canary.morph#v1","param:canary.morph#t0",
      "param:canary.freq","param:canary.freq#phase",
      "param:master.velocity","param:master.tau_base"],
    "slot_defaults":[180.0,0.0,0.0,0.0,55.0,0.0,1.0,0.0],
    "param_disciplines":[
      {"name":"bank.freq","discipline":"raw","companions":[]},
      {"name":"canary.morph","discipline":"glide",
       "glide_dur_sec":0.02,
       "companions":["canary.morph#v0","canary.morph#v1",
         "canary.morph#t0"]},
      {"name":"canary.freq","discipline":"anchor",
       "companions":["canary.freq#phase"]},
      {"name":"master.velocity","discipline":"velocity",
       "companions":["master.tau_base"]}
    ]})";
  tropical_runtime::FlatRuntime rt(buf);
  ASSERT(rt.load_ir_msl(JIT_CONST_IR, msl, manifest));
  rt.process();
  for (double sample : rt.outputBuffer) ASSERT_NEAR(sample, 180.0, 1e-5);

  auto advance_to_activation = [&](uint64_t effective) {
    while (rt.current_sample_index() < effective)
      rt.process();
    ASSERT(rt.current_sample_index() == effective);
    rt.process();
  };

  const auto raw = rt.dispatch_param_sync("bank.freq", 260.0);
  ASSERT(raw.ok);
  ASSERT(raw.observed_sample_index == buf);
  ASSERT(raw.effective_sample_index == buf + 512);
  while (rt.current_sample_index() < raw.effective_sample_index)
  {
    for (double sample : rt.outputBuffer) ASSERT_NEAR(sample, 180.0, 1e-5);
    rt.process();
  }
  rt.process();
  for (double sample : rt.outputBuffer) ASSERT_NEAR(sample, 260.0, 1e-5);

  const double old_v0 = rt.get_slot(1);
  const double old_v1 = rt.get_slot(2);
  const double old_t0 = rt.get_slot(3);
  const auto glide = rt.dispatch_param_sync("canary.morph", 0.5);
  ASSERT(glide.ok);
  const double r =
    (static_cast<double>(glide.effective_sample_index) - old_t0)
    / (0.02 * sr);
  const double s = std::clamp(r, 0.0, 1.0);
  ASSERT_NEAR(
    rt.get_slot(1),
    old_v0 + (old_v1 - old_v0) * (s * s * (3.0 - 2.0 * s)),
    1e-12);
  ASSERT_NEAR(rt.get_slot(3),
              static_cast<double>(glide.effective_sample_index), 1e-12);
  advance_to_activation(glide.effective_sample_index);

  const double old_freq = rt.get_slot(4);
  const double old_phase = rt.get_slot(5);
  const auto anchor = rt.dispatch_param_sync("canary.freq", 65.0);
  ASSERT(anchor.ok);
  const double inc0 = std::floor(old_freq * 4294967296.0 / sr);
  const double inc1 = std::floor(65.0 * 4294967296.0 / sr);
  const double ae = static_cast<double>(anchor.effective_sample_index);
  const auto frac = [](double x) { return x - std::floor(x); };
  ASSERT_NEAR(
    frac(old_phase + inc0 * ae / 4294967296.0),
    frac(rt.get_slot(5) + inc1 * ae / 4294967296.0), 1e-12);
  advance_to_activation(anchor.effective_sample_index);

  const double old_velocity = rt.get_slot(6);
  const double old_tau = rt.get_slot(7);
  const auto velocity =
    rt.dispatch_param_sync("master.velocity", 0.75);
  ASSERT(velocity.ok);
  const double ve = static_cast<double>(velocity.effective_sample_index);
  ASSERT_NEAR(
    old_tau * sr + old_velocity * ve,
    rt.get_slot(7) * sr + rt.get_slot(6) * ve, 1e-9);
  advance_to_activation(velocity.effective_sample_index);
  ASSERT(rt.ownership_failure_count() == 0);
  printf("PASS  Metal exact-E raw/glide/anchor/velocity epochs\n");
}

static void test_metal_retarget_recomputes_companions()
{
  constexpr unsigned int buf = 16;
  constexpr double sr = 44100.0;
  const std::string msl = msl_kernel("    output_buffer[s] = slots[0];");
  const std::string manifest = R"({"schema":"tropical_plan_6",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":4,
    "slot_names":["param:bank.freq","param:canary.morph#v0",
      "param:canary.morph#v1","param:canary.morph#t0"],
    "slot_defaults":[180.0,0.0,1.0,0.0],
    "param_disciplines":[
      {"name":"bank.freq","discipline":"raw","companions":[]},
      {"name":"canary.morph","discipline":"glide",
       "glide_dur_sec":0.02,
       "companions":["canary.morph#v0","canary.morph#v1",
         "canary.morph#t0"]}
    ]})";
  tropical_runtime::FlatRuntime rt(buf);
  ASSERT(rt.load_ir_msl(JIT_CONST_IR, msl, manifest));
  rt.process();

  const uint64_t provisional = rt.current_sample_index() + 512;
  const double old_v0 = rt.get_slot(1);
  const double old_v1 = rt.get_slot(2);
  const double old_t0 = rt.get_slot(3);
  tropical_metal::MetalRenderWorkerTestSeam seam;
  seam.pause_after_target_reserved.store(true, std::memory_order_release);
  rt.set_metal_worker_test_seam(&seam);

  tropical_runtime::ParamDispatchResult result;
  std::jthread control([&] {
    result = rt.dispatch_param_sync("canary.morph", 0.5);
  });
  ASSERT(wait_for_true(seam.target_reserved));
  for (unsigned int block = 0; block < 512 / buf; ++block)
    rt.process();
  seam.pause_after_target_reserved.store(false, std::memory_order_release);
  seam.release_target.store(true, std::memory_order_release);
  control.join();
  rt.set_metal_worker_test_seam(nullptr);

  ASSERT(result.ok);
  ASSERT(result.effective_sample_index > provisional);
  const double r =
    (static_cast<double>(result.effective_sample_index) - old_t0)
    / (0.02 * sr);
  const double s = std::clamp(r, 0.0, 1.0);
  ASSERT_NEAR(
    rt.get_slot(1),
    old_v0 + (old_v1 - old_v0) * (s * s * (3.0 - 2.0 * s)),
    1e-12);
  ASSERT_NEAR(
    rt.get_slot(3),
    static_cast<double>(result.effective_sample_index), 1e-12);
  printf("PASS  missed Metal activation recomputes companions at new E\n");
}

/** A candidate render failure refuses activation. A failure after activation
 *  drains already-prepared audio, then latches the active epoch silent with
 *  sticky diagnostics until a fresh epoch activates. No DAC is opened. */
static void test_metal_dispatch_failure_fail_closed()
{
  const unsigned int buf = 16;
  const std::string msl = msl_kernel(
    "    output_buffer[s] = slots[0];");

  setenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT", "2", 1);
  tropical_runtime_t refused = tropical_runtime_new(buf);
  ASSERT(refused != nullptr);
  const bool loaded =
    tropical_runtime_load_ir_msl(
      refused, JIT_CONST_IR, strlen(JIT_CONST_IR),
      msl.c_str(), msl.size(), MANIFEST, strlen(MANIFEST));
  unsetenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT");
  ASSERT(!loaded);
  ASSERT(std::strstr(
           tropical_last_error(), "initial Metal epoch failed") != nullptr);
  ASSERT(tropical_runtime_metal_dispatch_failure_count(refused) == 1);
  ASSERT(tropical_runtime_metal_activation_failure_count(refused) == 1);
  tropical_runtime_free(refused);

  setenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT", "5", 1);
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);
  ASSERT(tropical_runtime_load_ir_msl(
    rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
    msl.c_str(), msl.size(), MANIFEST, strlen(MANIFEST)));
  unsetenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT");
  ASSERT(tropical_runtime_metal_dispatch_failure_count(rt) == 0);
  ASSERT(tropical_runtime_metal_render_starvation_count(rt) == 0);

  const double * out = tropical_runtime_output_buffer(rt);
  for (unsigned int block = 0; block < 32; ++block)
    tropical_runtime_process(rt);
  bool failed = false;
  const auto failure_deadline =
    std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < failure_deadline)
  {
    if (tropical_runtime_metal_dispatch_failure_count(rt) == 1)
    {
      failed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT(failed);

  bool starved = false;
  for (unsigned int block = 0; block < 512; ++block)
  {
    tropical_runtime_process(rt);
    if (tropical_runtime_metal_render_starvation_count(rt) == 1)
    {
      starved = true;
      break;
    }
    std::this_thread::yield();
  }
  ASSERT(starved);
  for (unsigned int i = 0; i < buf; ++i) ASSERT(out[i] == 0.0);
  ASSERT(tropical_runtime_metal_dispatch_failure_count(rt) == 1);
  ASSERT(tropical_runtime_metal_epoch_tag_mismatch_count(rt) == 0);

  ASSERT(tropical_runtime_load_ir_msl(
    rt, JIT_CONST_IR, strlen(JIT_CONST_IR),
    msl.c_str(), msl.size(), MANIFEST, strlen(MANIFEST)));
  const uint64_t replacement_epoch =
    tropical_runtime_metal_published_activation_epoch(rt);
  for (unsigned int block = 0; block < 64
       && tropical_runtime_metal_acknowledged_activation_epoch(rt)
            < replacement_epoch; ++block)
    tropical_runtime_process(rt);
  ASSERT(tropical_runtime_metal_acknowledged_activation_epoch(rt)
         == replacement_epoch);
  for (unsigned int i = 0; i < buf; ++i)
    ASSERT_NEAR(out[i], 0.25, 1e-7);
  ASSERT(tropical_runtime_metal_dispatch_failure_count(rt) == 1);
  tropical_runtime_free(rt);
  printf("PASS  Metal failures refuse activation or latch active epochs silent\n");
}

static void test_metal_clock_jump_and_precompiled_swap_stress()
{
  constexpr uint32_t frames = 512;
  constexpr uint64_t transitions = 10000;
  const std::string body_a =
    "    output_buffer[s] = float(current_idx & 4095l);";
  const std::string body_b =
    "    output_buffer[s] = float(current_idx & 4095l) + 0.25f;";
  std::string error;
  auto kernel_a = tropical_metal::create(
    msl_kernel(body_a), frames, 0, 0, error);
  ASSERT(kernel_a != nullptr);
  auto kernel_b = tropical_metal::create(
    msl_kernel(body_b), frames, 0, 0, error);
  ASSERT(kernel_b != nullptr);

  tropical_runtime::EpochTileQueue queue(frames, frames);
  tropical_metal::MetalRenderWorker worker(queue);
  auto make_request = [](uint64_t epoch,
                         tropical_metal::MetalKernelPtr kernel,
                         tropical_metal::EpochTransitionKind transition,
                         uint64_t source) {
    tropical_metal::RenderEpochRequest request;
    request.epoch_id = epoch;
    request.kernel = std::move(kernel);
    request.sample_rate = 44100.0;
    request.source_origin = source;
    request.transition = transition;
    return request;
  };
  auto expected = [](uint64_t source, uint32_t offset, double marker) {
    return static_cast<double>(
      static_cast<float>((source + offset) & 4095ULL)) + marker;
  };
  std::array<double, frames> output{};
  auto consume = [&] {
    tropical_metal::set_audio_callback_thread(true);
    const auto result = queue.consume(output.data(), frames);
    tropical_metal::set_audio_callback_thread(false);
    return result;
  };

  tropical_metal::reset_callback_thread_violation_count();
  ASSERT(worker.schedule(make_request(
    1, kernel_a, tropical_metal::EpochTransitionKind::Fresh, 0)).ok);
  auto active = consume();
  ASSERT(active.status == tropical_runtime::TileConsumeStatus::Audio);
  double active_marker = 0.0;
  uint64_t active_source = frames;

  for (uint64_t i = 0; i < transitions; ++i)
  {
    const uint64_t source = (i & 1ULL)
      ? ((1ULL << 40) + i * frames)
      : (i * 3ULL * frames);
    const bool use_b = (i & 1ULL) != 0;
    const auto scheduled = worker.schedule(make_request(
      i + 2, use_b ? kernel_b : kernel_a,
      tropical_metal::EpochTransitionKind::ClockJump, source));
    ASSERT(scheduled.ok);

    const auto old = consume();
    ASSERT(old.status == tropical_runtime::TileConsumeStatus::Audio);
    ASSERT(!old.activated);
    ASSERT(old.source_start == active_source);
    for (uint32_t sample = 0; sample < frames; ++sample)
      ASSERT(output[sample]
             == expected(active_source, sample, active_marker));

    const auto switched = consume();
    ASSERT(switched.status == tropical_runtime::TileConsumeStatus::Audio);
    ASSERT(switched.activated);
    ASSERT(switched.source_start == source);
    const double marker = use_b ? 0.25 : 0.0;
    for (uint32_t sample = 0; sample < frames; ++sample)
      ASSERT(output[sample] == expected(source, sample, marker));
    active_source = source + frames;
    active_marker = marker;
  }

  ASSERT(worker.dispatch_failure_count() == 0);
  ASSERT(worker.activation_failure_count() == 0);
  ASSERT(worker.stale_completion_count() == 0);
  ASSERT(queue.starvation_count() == 0);
  ASSERT(queue.tag_mismatch_count() == 0);
  ASSERT(tropical_metal::callback_thread_violation_count() == 0);
  const auto latency = worker.activation_latency_stats();
  ASSERT(latency.count >= transitions);
  ASSERT(latency.min_ns > 0);
  ASSERT(worker.worker_wall_time_ns() > 0);
  printf("PASS  10,000 clock jumps + precompiled A/B swaps under Metal\n");
}

static void test_metal_control_transition_stress()
{
  constexpr unsigned int frames = 512;
  constexpr uint64_t transitions = 10000;
  constexpr double sr = 44100.0;
  const std::string msl = msl_kernel(
    "    output_buffer[s] = slots[0];");
  const std::string manifest = R"({"schema":"tropical_plan_6",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":8,
    "slot_names":["param:bank.freq","param:canary.morph#v0",
      "param:canary.morph#v1","param:canary.morph#t0",
      "param:canary.freq","param:canary.freq#phase",
      "param:master.velocity","param:master.tau_base"],
    "slot_defaults":[180.0,0.0,0.0,0.0,55.0,0.0,1.0,0.0],
    "param_disciplines":[
      {"name":"bank.freq","discipline":"raw","companions":[]},
      {"name":"canary.morph","discipline":"glide",
       "glide_dur_sec":0.02,
       "companions":["canary.morph#v0","canary.morph#v1",
         "canary.morph#t0"]},
      {"name":"canary.freq","discipline":"anchor",
       "companions":["canary.freq#phase"]},
      {"name":"master.velocity","discipline":"velocity",
       "companions":["master.tau_base"]}
    ]})";
  tropical_runtime::FlatRuntime rt(frames);
  ASSERT(rt.load_ir_msl(JIT_CONST_IR, msl, manifest));
  rt.process();
  tropical_runtime::FlatRuntime jit_reference(frames);
  ASSERT(jit_reference.load_ir(JIT_SLOT0_IR, manifest));

  std::array<std::jthread, 2> contention{
    std::jthread([](std::stop_token stop) {
      volatile uint64_t value = 1;
      while (!stop.stop_requested())
        value = value * 6364136223846793005ULL + 1;
    }),
    std::jthread([](std::stop_token stop) {
      volatile uint64_t value = 7;
      while (!stop.stop_requested())
        value = value * 2862933555777941757ULL + 3037000493ULL;
    }),
  };

  tropical_metal::reset_callback_thread_violation_count();
  std::vector<double> reference(frames);
  const uint32_t slot_zero = 0;
  for (uint64_t i = 0; i < transitions; ++i)
  {
    tropical_runtime::ParamDispatchResult result;
    std::string name;
    double value = 0.0;
    switch (i % 4)
    {
      case 0:
        name = "bank.freq";
        value = 180.0 + static_cast<double>(i % 97);
        result = rt.dispatch_param_sync(name, value);
        break;
      case 1:
      {
        const double old_v0 = rt.get_slot(1);
        const double old_v1 = rt.get_slot(2);
        const double old_t0 = rt.get_slot(3);
        name = "canary.morph";
        value = static_cast<double>(i % 101) / 100.0;
        result = rt.dispatch_param_sync(name, value);
        ASSERT(result.ok);
        const double r =
          (static_cast<double>(result.effective_sample_index) - old_t0)
          / (0.02 * sr);
        const double s = std::clamp(r, 0.0, 1.0);
        ASSERT_NEAR(
          rt.get_slot(1),
          old_v0 + (old_v1 - old_v0) * (s * s * (3.0 - 2.0 * s)),
          1e-9);
        ASSERT_NEAR(
          rt.get_slot(3),
          static_cast<double>(result.effective_sample_index), 1e-9);
        break;
      }
      case 2:
      {
        const double old_freq = rt.get_slot(4);
        const double old_phase = rt.get_slot(5);
        const double next_freq = 45.0 + static_cast<double>(i % 41);
        name = "canary.freq";
        value = next_freq;
        result = rt.dispatch_param_sync(name, value);
        ASSERT(result.ok);
        const double e =
          static_cast<double>(result.effective_sample_index);
        const double inc0 = std::floor(old_freq * 4294967296.0 / sr);
        const double inc1 = std::floor(next_freq * 4294967296.0 / sr);
        const auto frac = [](double x) { return x - std::floor(x); };
        ASSERT_NEAR(
          frac(old_phase + inc0 * e / 4294967296.0),
          frac(rt.get_slot(5) + inc1 * e / 4294967296.0), 1e-9);
        break;
      }
      default:
      {
        const double old_velocity = rt.get_slot(6);
        const double old_tau = rt.get_slot(7);
        const double next_velocity =
          0.5 + static_cast<double>(i % 31) / 100.0;
        name = "master.velocity";
        value = next_velocity;
        result = rt.dispatch_param_sync(name, value);
        ASSERT(result.ok);
        const double e =
          static_cast<double>(result.effective_sample_index);
        ASSERT_NEAR(
          old_tau * sr + old_velocity * e,
          rt.get_slot(7) * sr + rt.get_slot(6) * e, 1e-6);
        break;
      }
    }
    ASSERT(result.ok);
    const auto replay =
      jit_reference.dispatch_param_sync_at_sample_index(
        name, value, result.effective_sample_index);
    ASSERT(replay.ok);
    ASSERT(replay.effective_sample_index
           == result.effective_sample_index);
    const uint64_t epoch = rt.metal_published_activation_epoch();
    while (rt.metal_acknowledged_activation_epoch() < epoch)
      rt.process();
    ASSERT(rt.current_sample_index()
           == result.effective_sample_index + frames);
    ASSERT(rt.render_window(
      result.effective_sample_index, frames,
      &slot_zero, 1, reference.data()));
    jit_reference.set_sample_index(result.effective_sample_index);
    jit_reference.process();
    for (uint32_t sample = 0; sample < frames; ++sample)
    {
      ASSERT_NEAR(rt.outputBuffer[sample], reference[sample], 1e-4);
      ASSERT_NEAR(
        rt.outputBuffer[sample],
        jit_reference.outputBuffer[sample], 1e-4);
    }
  }

  for (auto & thread : contention) thread.request_stop();
  ASSERT(rt.ownership_failure_count() == 0);
  ASSERT(rt.metal_dispatch_failure_count() == 0);
  ASSERT(rt.metal_render_starvation_count() == 0);
  ASSERT(rt.metal_epoch_tag_mismatch_count() == 0);
  ASSERT(rt.metal_activation_failure_count() == 0);
  ASSERT(rt.metal_callback_thread_violation_count() == 0);
  auto stages = rt.metal_worker_stage_times();
  const auto observation_deadline =
    std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (stages[6] < stages[5]
         && std::chrono::steady_clock::now() < observation_deadline)
  {
    std::this_thread::yield();
    stages = rt.metal_worker_stage_times();
  }
  ASSERT(stages[0] > 0);
  ASSERT(stages[6] >= stages[5]);
  const auto latency = rt.metal_activation_latency_stats();
  ASSERT(latency[0] >= transitions);
  ASSERT(rt.metal_worker_wall_time_ns() > 0);
  printf("PASS  10,000 raw/glide/anchor/velocity epochs under contention\n");
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
  test_callback_thread_provenance();
  test_cooperative_dispatch_geometry();
  test_independent_device_and_render_quanta();
  test_metal_ramp();
  test_offline_renderer_waits_for_worker_refill();
  test_metal_slots();
  test_metal_hotswap();
  test_metal_set_index();
  test_metal_columns_live();
  test_metal_effective_dispatch_epochs();
  test_metal_retarget_recomputes_companions();
  test_metal_dispatch_failure_fail_closed();
  test_metal_clock_jump_and_precompiled_swap_stress();
  test_metal_control_transition_stress();
  if (g_fail == 0) printf("ALL METAL TESTS PASSED\n");
  return g_fail == 0 ? 0 : 1;
}
