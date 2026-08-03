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
#include "runtime/FlatRuntime.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

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

struct TestAssetFile
{
  TestAssetFile(std::string name, const std::vector<uint8_t>& bytes)
    : path(std::move(name))
  {
    std::filesystem::remove(path);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed to write test asset");
  }
  ~TestAssetFile() { std::filesystem::remove(path); }
  std::filesystem::path path;
};

static std::string plan6_manifest(
  const std::string& path,
  const std::string& sha =
    "8edfe9d801000ee043eee7a06dcfe1f77a1ec553a4fa4f7990ad5770981ff890",
  uint64_t byte_count = 16,
  uint32_t sample_rate = 44100,
  uint64_t byte_offset = 0,
  uint64_t element_count = 4,
  const std::string& scalar_format = "float32")
{
  return std::string(R"({"schema":"tropical_plan_6","config":{"sampleRate":44100},
    "register_count":0,"array_slot_count":1,"array_slot_sizes":[4],
    "array_slot_names":["frozen"],"instance_functions":[],"sinks":[],
    "slot_count":0,"immutable_assets":[{"path":")") + path
    + R"(","byte_count":)" + std::to_string(byte_count)
    + R"(,"sha256":")" + sha
    + R"(","sample_rate":)" + std::to_string(sample_rate)
    + R"(,"array_slots":[{"slot":0,"byte_offset":)"
    + std::to_string(byte_offset)
    + R"(,"element_count":)" + std::to_string(element_count)
    + R"(,"scalar_format":")" + scalar_format + R"("}]}]})";
}

static std::string immutable_index_ir()
{
  return wrap_loop(
    "  %a0p = getelementptr inbounds ptr, ptr %arrays, i64 0\n"
    "  %a0 = load ptr, ptr %a0p, align 8\n"
    "  %idx = urem i64 %s, 4\n"
    "  %ep = getelementptr inbounds i64, ptr %a0, i64 %idx\n"
    "  %raw = load i64, ptr %ep, align 8\n"
    "  %v = bitcast i64 %raw to double\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
}

static bool wait_for_true(
  const std::atomic<bool> & value,
  std::chrono::milliseconds timeout = std::chrono::seconds(10))
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!value.load(std::memory_order_acquire)
         && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return value.load(std::memory_order_acquire);
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

static bool load_fails_with(tropical_runtime_t runtime, const std::string& ir,
                            const char* manifest, const char* expected)
{
  if (tropical_runtime_load_ir(runtime, ir.c_str(), ir.size(),
                               manifest, std::strlen(manifest)))
    return false;
  const char* error = tropical_last_error();
  return error && std::strstr(error, expected);
}

/**
 * 2. The native boundary accepts exactly tropical_plan_5. Plan 4, missing or
 *    unknown schemas, retired state/output carriers, legacy operands, and
 *    missing destination namespaces fail before caller-supplied IR is loaded.
 */
static void test_plan5_only_manifest_boundary()
{
  tropical_runtime_t rt = tropical_runtime_new(16);
  ASSERT(rt != nullptr);
  const std::string ir = wrap_loop(RAMP_BODY);

  ASSERT(load_fails_with(
    rt, ir, R"({"schema":"tropical_plan_4"})",
    "unsupported manifest schema 'tropical_plan_4'; expected 'tropical_plan_5'"));
  ASSERT(load_fails_with(
    rt, ir, R"({})",
    "unsupported manifest schema ''; expected 'tropical_plan_5'"));
  ASSERT(load_fails_with(
    rt, ir, R"({"schema":"tropical_plan_99"})",
    "unsupported manifest schema 'tropical_plan_99'; expected 'tropical_plan_5'"));
  ASSERT(load_fails_with(
    rt, ir, R"({"schema":"tropical_plan_5","state_init":[1]})",
    "retired field 'state_init' is not valid in tropical_plan_5"));
  ASSERT(load_fails_with(
    rt, ir,
    R"({"schema":"tropical_plan_5","output_targets":[],"outputs":[]})",
    "retired field 'output_targets' is not valid in tropical_plan_5"));
  ASSERT(load_fails_with(
    rt, ir,
    R"({"schema":"tropical_plan_5","instance_functions":[{"instructions":[{"tag":"Add","dst":0,"args":[],"result_type":"float"}]}]})",
    "plan-5 instruction missing required dst_kind"));
  ASSERT(load_fails_with(
    rt, ir,
    R"({"schema":"tropical_plan_5","instance_functions":[{"instructions":[{"tag":"Add","dst":0,"dst_kind":"temp","args":[{"kind":"rate","scalar_type":"float"}],"result_type":"float"}]}]})",
    "unknown operand kind 'rate'"));
  ASSERT(load_fails_with(
    rt, ir, R"({"schema":"tropical_plan_6"})",
    "tropical_plan_6 requires nonempty immutable_assets"));
  ASSERT(load_fails_with(
    rt, ir,
    R"({"schema":"tropical_plan_5","immutable_assets":[{}]})",
    "tropical_plan_5 cannot carry immutable_assets"));

  tropical_runtime_free(rt);
}

static void test_plan6_immutable_asset_jit()
{
  const std::vector<uint8_t> bytes = {
    0x00, 0x00, 0x80, 0x3e, // 0.25f
    0x00, 0x00, 0xc0, 0xbf, // -1.5f
    0x00, 0x00, 0x30, 0x40, // 2.75f
    0x00, 0x00, 0x80, 0x40, // 4.0f
  };
  TestAssetFile asset("tropical-plan6-jit-test.f32", bytes);
  const std::string ir = immutable_index_ir();
  const std::string manifest = plan6_manifest(asset.path.string());
  tropical_runtime_t rt = tropical_runtime_new(16);
  ASSERT(rt != nullptr);
  ASSERT_OK(tropical_runtime_load_ir(
    rt, ir.c_str(), ir.size(), manifest.c_str(), manifest.size()));

  // Removing the source after load proves process() performs no path lookup,
  // I/O, hash, conversion, or allocation and that state ownership retains the
  // converted immutable arrays.
  std::filesystem::remove(asset.path);
  tropical_runtime_process(rt);
  const double expected[] = {0.25, -1.5, 2.75, 4.0};
  const double* out = tropical_runtime_output_buffer(rt);
  for (unsigned int i = 0; i < 16; ++i)
    ASSERT_NEAR(out[i], expected[i % 4], 0.0);

  // A normal Plan-5 hot-swap retires the owning state without carrying its
  // immutable pointers into the replacement.
  const std::string ramp = wrap_loop(RAMP_BODY);
  ASSERT_OK(tropical_runtime_load_ir(
    rt, ramp.c_str(), ramp.size(), RAMP_MANIFEST, std::strlen(RAMP_MANIFEST)));
  tropical_runtime_process(rt);
  out = tropical_runtime_output_buffer(rt);
  for (unsigned int i = 0; i < 16; ++i)
    ASSERT_NEAR(out[i], static_cast<double>(16 + i), 0.0);
  tropical_runtime_free(rt);
}

// A paused reader retains the old Plan-6 pack while two hot-swaps reuse both
// bounded KernelState slots. The source file is already gone, so the completed
// old frame can only succeed through the scope image's shared asset ownership.
static void test_scope_snapshot_retains_plan6_asset()
{
  const std::vector<uint8_t> bytes = {
    0x00, 0x00, 0x80, 0x3e, // 0.25f
    0x00, 0x00, 0xc0, 0xbf, // -1.5f
    0x00, 0x00, 0x30, 0x40, // 2.75f
    0x00, 0x00, 0x80, 0x40, // 4.0f
  };
  TestAssetFile asset("tropical-plan6-scope-snapshot.f32", bytes);
  const std::string ir = wrap_loop(
    "  %a0p = getelementptr inbounds ptr, ptr %arrays, i64 0\n"
    "  %a0 = load ptr, ptr %a0p, align 8\n"
    "  %idx = urem i64 %s, 4\n"
    "  %ep = getelementptr inbounds i64, ptr %a0, i64 %idx\n"
    "  %raw = load i64, ptr %ep, align 8\n"
    "  %v = bitcast i64 %raw to double\n"
    "  %sp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  store double %v, ptr %sp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  std::string manifest = plan6_manifest(asset.path.string());
  const std::string empty_slots = R"("slot_count":0)";
  const std::string tap_slot =
    R"("slot_count":1,"slot_names":["tap"],"slot_defaults":[0.0])";
  const auto slot_pos = manifest.find(empty_slots);
  ASSERT(slot_pos != std::string::npos);
  manifest.replace(slot_pos, empty_slots.size(), tap_slot);

  tropical_runtime::FlatRuntime rt(16);
  ASSERT(rt.load_ir(ir, manifest));
  std::filesystem::remove(asset.path);

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_render_coordinate.store(true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  const uint32_t slot = 0;
  std::array<double, 16> values{};
  tropical_runtime::RenderWindowStatus status =
    tropical_runtime::RenderWindowStatus::InvalidArgument;
  std::jthread scope([&] {
    status = rt.render_window_low_priority(
      0, static_cast<uint32_t>(values.size()), 1,
      &slot, 1, values.data());
  });
  ASSERT(wait_for_true(seam.render_coordinate_completed));

  const std::string ramp = wrap_loop(RAMP_BODY);
  ASSERT(rt.load_ir(ramp, RAMP_MANIFEST));
  ASSERT(rt.load_ir(ramp, RAMP_MANIFEST));

  seam.release_render_coordinate.store(true, std::memory_order_release);
  scope.join();
  rt.set_ownership_test_seam(nullptr);
  ASSERT(status == tropical_runtime::RenderWindowStatus::Rendered);
  // render_window invokes the one-sample kernel once per coordinate, so this
  // fixture's buffer-local `%s` index remains zero on every invocation.
  for (double value : values) ASSERT_NEAR(value, 0.25, 0.0);
}

static void test_plan6_immutable_asset_refusals()
{
  const std::vector<uint8_t> bytes = {
    0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0xc0, 0xbf,
    0x00, 0x00, 0x30, 0x40, 0x00, 0x00, 0x80, 0x40,
  };
  TestAssetFile asset("tropical-plan6-refusal-test.f32", bytes);
  const std::string ir = immutable_index_ir();
  tropical_runtime_t rt = tropical_runtime_new(4);
  ASSERT(rt != nullptr);
  auto fails = [&](const std::string& manifest, const char* expected) {
    return load_fails_with(rt, ir, manifest.c_str(), expected);
  };

  ASSERT(fails(plan6_manifest("/tmp/absolute.f32"), "must be package-relative"));
  ASSERT(fails(plan6_manifest("../escape.f32"), "may not contain '..'"));
  ASSERT(fails(plan6_manifest("missing-plan6-asset.f32"), "missing or unreadable"));
  ASSERT(fails(plan6_manifest(asset.path.string(), std::string(64, '0')),
               "SHA-256 mismatch"));
  ASSERT(fails(plan6_manifest(asset.path.string(),
               "8edfe9d801000ee043eee7a06dcfe1f77a1ec553a4fa4f7990ad5770981ff890",
               20), "byte_count mismatch"));
  ASSERT(fails(plan6_manifest(asset.path.string(),
               "8edfe9d801000ee043eee7a06dcfe1f77a1ec553a4fa4f7990ad5770981ff890",
               16, 32000), "sample_rate mismatch"));
  ASSERT(fails(plan6_manifest(asset.path.string(),
               "8edfe9d801000ee043eee7a06dcfe1f77a1ec553a4fa4f7990ad5770981ff890",
               16, 44100, 4, 4), "byte range is invalid"));
  ASSERT(fails(plan6_manifest(asset.path.string(),
               "8edfe9d801000ee043eee7a06dcfe1f77a1ec553a4fa4f7990ad5770981ff890",
               16, 44100, 0, 3), "element_count does not match"));
  ASSERT(fails(plan6_manifest(asset.path.string(),
               "8edfe9d801000ee043eee7a06dcfe1f77a1ec553a4fa4f7990ad5770981ff890",
               16, 44100, 0, 4, "float64"), "scalar_format must be 'float32'"));

  TestAssetFile nonfinite(
    "tropical-plan6-nonfinite-test.f32",
    {0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0x00, 0x00,
     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
  ASSERT(fails(plan6_manifest(
    nonfinite.path.string(),
    "2ffb157a90d8b58f6114acbdf9474af21dee3cfabfe507dc0f3c29b02170423f",
    16, 44100, 0, 4), "contains a non-finite value"));

  tropical_runtime_free(rt);
}

/**
 * 3. Closed-form ramp — output 0,1,2,… as a function of the sample index,
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

// Concurrent clock requests are applied only at process boundaries. Every
// rendered block must therefore be internally contiguous and its first sample
// must be either the previous block's successor or one whole requested jump.
// This is intentionally TSAN-suitable: output observations stay on the audio
// thread while the control thread touches only the public request API.
static void test_clock_boundary_handoff()
{
  constexpr unsigned int buf = 16;
  constexpr uint64_t base = 1000000;
  constexpr uint64_t stride = 1000;
  constexpr uint64_t requests = 10000;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);
  const std::string ir = wrap_loop(RAMP_BODY);
  ASSERT_OK(tropical_runtime_load_ir(
    rt, ir.c_str(), ir.size(), RAMP_MANIFEST, strlen(RAMP_MANIFEST)));

  std::atomic<bool> go{false};
  std::atomic<bool> done{false};
  std::jthread control([&] {
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    for (uint64_t i = 0; i < requests; ++i)
    {
      tropical_runtime_set_sample_index(rt, base + i * stride);
      if ((i & 7U) == 0) std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  });

  go.store(true, std::memory_order_release);
  std::vector<uint64_t> starts;
  for (uint64_t block = 0;
       block < 2000 || !done.load(std::memory_order_acquire); ++block)
  {
    tropical_runtime_process(rt);
    const double * out = tropical_runtime_output_buffer(rt);
    ASSERT(out != nullptr);
    const uint64_t start = static_cast<uint64_t>(out[0]);
    for (unsigned int i = 0; i < buf; ++i)
      ASSERT_NEAR(out[i], static_cast<double>(start + i), 0.0);
    starts.push_back(start);
  }
  control.join();
  for (int i = 0; i < 4; ++i)
  {
    tropical_runtime_process(rt);
    starts.push_back(static_cast<uint64_t>(
      tropical_runtime_output_buffer(rt)[0]));
  }

  for (std::size_t i = 1; i < starts.size(); ++i)
  {
    const uint64_t start = starts[i];
    const bool continuation = start == starts[i - 1] + buf;
    const bool whole_request =
      start >= base && (start - base) % stride == 0
      && (start - base) / stride < requests;
    ASSERT(continuation || whole_request);
  }

  // Every observed non-continuation is one unique request, applied once and
  // in publication order. A duplicated request application or a future value
  // paired with an older even sequence fails these assertions.
  std::unordered_set<uint64_t> applied_requests;
  uint64_t previous_request = 0;
  bool have_request = false;
  for (std::size_t i = 1; i < starts.size(); ++i)
  {
    if (starts[i] == starts[i - 1] + buf) continue;
    ASSERT(applied_requests.insert(starts[i]).second);
    if (have_request) ASSERT(starts[i] > previous_request);
    previous_request = starts[i];
    have_request = true;
  }
  ASSERT(static_cast<uint64_t>(tropical_runtime_current_sample_index(rt))
         == starts.back() + buf);
  ASSERT(tropical_runtime_current_sample_index_u64(rt)
         == starts.back() + buf);
  constexpr uint64_t far_exact = (uint64_t{1} << 54) + 123;
  tropical_runtime_set_sample_index(rt, far_exact);
  tropical_runtime_process(rt);
  ASSERT(tropical_runtime_current_sample_index_u64(rt)
         == far_exact + buf);
  ASSERT(tropical_runtime_ownership_failure_count(rt) == 0);
  tropical_runtime_free(rt);
}

// Deterministically hold a clock request after its release-stored payload but
// while its sequence is odd. Audio must defer it, then apply the completed
// request at exactly one boundary and continue from there.
static void test_clock_request_barrier()
{
  constexpr unsigned int buf = 16;
  constexpr uint64_t requested = 1000000;
  tropical_runtime::FlatRuntime rt(buf);
  const std::string ir = wrap_loop(RAMP_BODY);
  ASSERT(rt.load_ir(ir, RAMP_MANIFEST));
  rt.process();
  ASSERT(rt.outputBuffer[0] == 0.0);

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_clock_payload.store(true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  std::jthread control([&] { rt.set_sample_index(requested); });
  const bool payload_stored = wait_for_true(seam.clock_payload_stored);
  if (!payload_stored)
  {
    seam.release_clock.store(true, std::memory_order_release);
    control.join();
    ASSERT(payload_stored);
  }

  rt.process();
  const uint64_t deferred_start = static_cast<uint64_t>(rt.outputBuffer[0]);
  seam.release_clock.store(true, std::memory_order_release);
  control.join();
  rt.set_ownership_test_seam(nullptr);

  rt.process();
  const uint64_t applied_start = static_cast<uint64_t>(rt.outputBuffer[0]);
  rt.process();
  const uint64_t continued_start = static_cast<uint64_t>(rt.outputBuffer[0]);
  ASSERT(deferred_start == buf);
  ASSERT(applied_start == requested);
  ASSERT(continued_start == requested + buf);
  ASSERT(rt.ownership_failure_count() == 0);
}

// A reference oracle must replay each production dispatch at the first output
// sample where its generation is audible. Freezing one batch-start boundary
// can accumulate a visible phase/origin error when the callback advances
// between the four discipline writes.
static void test_param_dispatch_exact_sample_replay()
{
  constexpr unsigned int buf = 512;
  constexpr uint64_t base = uint64_t{1} << 40;
  tropical_runtime::FlatRuntime live(buf);
  tropical_runtime::FlatRuntime replay(buf);
  tropical_runtime::FlatRuntime stale_batch_oracle(buf);
  const std::string ir = wrap_loop(
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 0.000000e+00, ptr %op, align 8\n");
  const std::string manifest = R"({"schema":"tropical_plan_5",
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
  ASSERT(live.load_ir(ir, manifest));
  ASSERT(replay.load_ir(ir, manifest));
  ASSERT(stale_batch_oracle.load_ir(ir, manifest));

  const std::array<std::string, 4> names = {
    "bank.freq", "canary.morph", "canary.freq", "master.velocity"
  };
  const std::array<std::string, 4> disciplines = {
    "raw", "glide", "anchor", "velocity"
  };
  const std::array<double, 4> lows = {180.0, 0.0, 55.0, 1.0};
  const std::array<double, 4> highs = {260.0, 0.5, 65.0, 0.75};

  // High/low/high: the anchor crosses one callback after the batch snapshot
  // on each high write. The stale oracle therefore accumulates exactly two
  // buffers of 10 Hz phase-reanchoring error.
  for (uint64_t round = 0; round < 3; ++round)
  {
    const uint64_t batch_start = base + (round * 86 + 1) * buf;
    live.set_sample_index(batch_start - buf);
    live.process();
    const bool high = (round & 1U) == 0;
    const auto & values = high ? highs : lows;

    for (std::size_t i = 0; i < names.size(); ++i)
    {
      if (i == 2 && high)
      {
        live.set_sample_index(batch_start);
        live.process();
      }
      const auto production =
        live.dispatch_param_sync(names[i], values[i]);
      ASSERT(production.ok);
      ASSERT(production.discipline == disciplines[i]);
      const uint64_t expected_index =
        i >= 2 && high ? batch_start + buf : batch_start;
      ASSERT(production.observed_sample_index == expected_index);
      ASSERT(production.effective_sample_index == expected_index);

      const auto exact = replay.dispatch_param_sync_at_sample_index(
        names[i], values[i], production.effective_sample_index);
      ASSERT(exact.ok);
      ASSERT(exact.discipline == disciplines[i]);
      ASSERT(exact.effective_sample_index
             == production.effective_sample_index);

      const auto stale =
        stale_batch_oracle.dispatch_param_sync_at_sample_index(
          names[i], values[i], batch_start);
      ASSERT(stale.ok);
      ASSERT(stale.effective_sample_index == batch_start);
    }
  }

  for (uint32_t slot = 0; slot < 8; ++slot)
    ASSERT_NEAR(live.get_slot(slot), replay.get_slot(slot), 0.0);

  const uint32_t phase_slot = live.slot_index("param:canary.freq#phase");
  ASSERT(phase_slot != UINT32_MAX);
  const double live_phase = live.get_slot(phase_slot);
  const double stale_phase = stale_batch_oracle.get_slot(phase_slot);
  double phase_delta = std::fabs(live_phase - stale_phase);
  phase_delta = std::min(phase_delta, 1.0 - phase_delta);
  const double inc55 = std::floor(55.0 * 4294967296.0 / 44100.0);
  const double inc65 = std::floor(65.0 * 4294967296.0 / 44100.0);
  const double expected_phase_delta =
    std::fabs((inc55 - inc65) * (2.0 * buf) / 4294967296.0);
  ASSERT_NEAR(phase_delta, expected_phase_delta, 1e-15);

  // This is the incident-scale output error for a 1e-12 canary: 1.333e-12,
  // matching the blocked row's 1.351e-12 without implicating the 1e-8 bank.
  const double predicted_max_error =
    2e-12 * std::sin(3.14159265358979323846 * expected_phase_delta);
  ASSERT_NEAR(predicted_max_error, 1.332958724784292e-12, 1e-18);
}

// Audio owns the boundary word: it never waits for control. These seams cover
// both sides of capture and force a publication CAS loss for every discipline.
// A retry must restore the pre-transaction slots before re-evaluating at E.
static void test_param_dispatch_effective_boundary_races()
{
  constexpr unsigned int buf = 16;
  constexpr double sr = 44100.0;
  tropical_runtime::FlatRuntime rt(buf);
  const std::string ir = wrap_loop(
    "  %sp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  %v = load double, ptr %sp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  const std::string manifest = R"({"schema":"tropical_plan_5",
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
  ASSERT(rt.load_ir(ir, manifest));

  // Forced before-capture on a fresh state: C=E=0 for all four disciplines.
  tropical_runtime::RuntimeOwnershipTestSeam before;
  before.pause_before_boundary_capture.store(true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&before);
  std::jthread first_audio([&] { rt.process(); });
  ASSERT(wait_for_true(before.boundary_capture_pending));
  const auto raw0 = rt.dispatch_param_sync("bank.freq", 260.0);
  const auto glide0 = rt.dispatch_param_sync("canary.morph", 0.5);
  const auto anchor0 = rt.dispatch_param_sync("canary.freq", 65.0);
  const auto velocity0 = rt.dispatch_param_sync("master.velocity", 0.75);
  for (const auto * result : {&raw0, &glide0, &anchor0, &velocity0})
  {
    ASSERT(result->ok);
    ASSERT(result->observed_sample_index == 0);
    ASSERT(result->effective_sample_index == 0);
  }
  before.release_boundary_capture.store(true, std::memory_order_release);
  first_audio.join();
  rt.set_ownership_test_seam(nullptr);
  for (double sample : rt.outputBuffer) ASSERT(sample == 260.0);

  auto force_loss = [&](const std::string & name, double value) {
    tropical_runtime::RuntimeOwnershipTestSeam seam;
    seam.pause_after_dispatch_materialization.store(
      true, std::memory_order_relaxed);
    rt.set_ownership_test_seam(&seam);
    const uint64_t crossing_start = rt.current_sample_index();
    tropical_runtime::ParamDispatchResult result;
    std::jthread control([&] {
      result = rt.dispatch_param_sync(name, value);
    });
    const bool materialized = wait_for_true(seam.dispatch_materialized);
    if (!materialized)
    {
      seam.release_dispatch.store(true, std::memory_order_release);
      control.join();
      rt.set_ownership_test_seam(nullptr);
      return tropical_runtime::ParamDispatchResult{};
    }
    rt.process();
    seam.pause_after_dispatch_materialization.store(
      false, std::memory_order_release);
    seam.release_dispatch.store(true, std::memory_order_release);
    control.join();
    rt.set_ownership_test_seam(nullptr);
    if (result.observed_sample_index != crossing_start
        || result.effective_sample_index != crossing_start + buf)
      result.ok = false;
    return result;
  };

  // Raw is old strictly before E and new at E.
  const auto raw = force_loss("bank.freq", 300.0);
  ASSERT(raw.ok);
  for (double sample : rt.outputBuffer) ASSERT(sample == 260.0);
  rt.process();
  for (double sample : rt.outputBuffer) ASSERT(sample == 300.0);

  // Glide starts at the old curve's exact value at E.
  const double old_v0 = rt.get_slot(1);
  const double old_v1 = rt.get_slot(2);
  const double old_t0 = rt.get_slot(3);
  const auto glide = force_loss("canary.morph", 0.0);
  ASSERT(glide.ok);
  const double r =
    (static_cast<double>(glide.effective_sample_index) - old_t0)
    / (0.02 * sr);
  const double s = std::clamp(r, 0.0, 1.0);
  const double old_at_e =
    old_v0 + (old_v1 - old_v0) * (s * s * (3.0 - 2.0 * s));
  ASSERT_NEAR(rt.get_slot(1), old_at_e, 1e-12);
  ASSERT_NEAR(rt.get_slot(2), 0.0, 1e-12);
  ASSERT_NEAR(rt.get_slot(3),
              static_cast<double>(glide.effective_sample_index), 1e-12);
  rt.process();

  // Quantized phase agrees on both sides at E.
  const double old_freq = rt.get_slot(4);
  const double old_phase = rt.get_slot(5);
  const auto anchor = force_loss("canary.freq", 55.0);
  ASSERT(anchor.ok);
  const double inc0 = std::floor(old_freq * 4294967296.0 / sr);
  const double inc1 = std::floor(55.0 * 4294967296.0 / sr);
  const double ae = static_cast<double>(anchor.effective_sample_index);
  const auto frac = [](double x) { return x - std::floor(x); };
  ASSERT_NEAR(
    frac(old_phase + inc0 * ae / 4294967296.0),
    frac(rt.get_slot(5) + inc1 * ae / 4294967296.0),
    1e-12);
  rt.process();

  // tau_base*SR + velocity*n agrees on both sides at E.
  const double old_velocity = rt.get_slot(6);
  const double old_tau = rt.get_slot(7);
  const auto velocity = force_loss("master.velocity", 1.0);
  ASSERT(velocity.ok);
  const double ve = static_cast<double>(velocity.effective_sample_index);
  ASSERT_NEAR(
    old_tau * sr + old_velocity * ve,
    rt.get_slot(7) * sr + rt.get_slot(6) * ve,
    1e-9);
  rt.process();
  ASSERT(rt.ownership_failure_count() == 0);
}

// A scope owns one immutable program/control image for its whole frame. A
// control publication completes while that reader is deliberately paused, and
// the old frame remains coherent instead of tearing to the new slot value.
static void test_scope_snapshot_does_not_block_control()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string ir = wrap_loop(
    "  %sp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  %v = load double, ptr %sp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  const std::string manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":1,"slot_names":["param:x"],
    "slot_defaults":[1.0],"param_disciplines":[
      {"name":"x","discipline":"raw","companions":[]}
    ]})";
  ASSERT(rt.load_ir(ir, manifest));

  const uint32_t slot = 0;
  std::array<double, 3> strided{};
  ASSERT(rt.render_window_low_priority(
    10, static_cast<uint32_t>(strided.size()), 3,
    &slot, 1, strided.data())
    == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : strided) ASSERT(value == 1.0);

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_render_coordinate.store(true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  std::array<double, 128> scope_output{};
  tropical_runtime::RenderWindowStatus scope_status =
    tropical_runtime::RenderWindowStatus::InvalidArgument;
  std::jthread scope([&] {
    scope_status = rt.render_window_low_priority(
      0, static_cast<uint32_t>(scope_output.size()), 1,
      &slot, 1, scope_output.data());
  });
  ASSERT(wait_for_true(seam.render_coordinate_completed));

  tropical_runtime::ParamDispatchResult control_result;
  std::atomic<bool> control_done{false};
  std::jthread control([&] {
    control_result = rt.dispatch_param_sync("x", 0.5);
    control_done.store(true, std::memory_order_release);
  });
  ASSERT(wait_for_true(control_done));
  ASSERT(control_result.ok);

  seam.release_render_coordinate.store(true, std::memory_order_release);
  scope.join();
  control.join();
  rt.set_ownership_test_seam(nullptr);

  ASSERT(scope_status == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : scope_output) ASSERT(value == 1.0);
  std::array<double, 4> next{};
  ASSERT(rt.render_window_low_priority(
    0, static_cast<uint32_t>(next.size()), 1,
    &slot, 1, next.data())
    == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : next) ASSERT(value == 0.5);
  ASSERT(rt.scope_preemption_count() == 0);
  ASSERT(rt.control_waiter_count() == 0);
}

// Socket callers name taps rather than passing numeric slots. Name resolution
// and evaluation must pin one program image so a hot-swap cannot reinterpret
// an old tap name using the new program's slot layout.
static void test_scope_named_render_pins_program()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string ir = wrap_loop(
    "  %sp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  %v = load double, ptr %sp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  const auto manifest = [](const char * name, double value) {
    return std::string(R"({"schema":"tropical_plan_5",
      "config":{"sampleRate":44100},"register_count":0,
      "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
      "sinks":[],"slot_count":1,"slot_names":[")") + name
      + R"("],"slot_defaults":[)" + std::to_string(value) + R"(],
      "param_disciplines":[]})";
  };
  ASSERT(rt.load_ir(ir, manifest("tap:old", 1.0)));

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_render_coordinate.store(true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  std::array<double, 64> old_values{};
  tropical_runtime::ScopeRenderResult old_result;
  std::jthread old_scope([&] {
    old_result = rt.render_window_low_priority_by_name(
      0, static_cast<uint32_t>(old_values.size()), 1,
      {"tap:old"}, old_values.data());
  });
  ASSERT(wait_for_true(seam.render_coordinate_completed));

  ASSERT(rt.load_ir(ir, manifest("tap:new", 2.0)));
  seam.release_render_coordinate.store(true, std::memory_order_release);
  old_scope.join();
  rt.set_ownership_test_seam(nullptr);

  ASSERT(old_result.status == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : old_values) ASSERT(value == 1.0);

  std::array<double, 4> next_values{};
  const auto missing = rt.render_window_low_priority_by_name(
    0, static_cast<uint32_t>(next_values.size()), 1,
    {"tap:old"}, next_values.data());
  ASSERT(missing.status == tropical_runtime::RenderWindowStatus::InvalidArgument);
  ASSERT(missing.unknown_slot == "tap:old");
  const auto next = rt.render_window_low_priority_by_name(
    0, static_cast<uint32_t>(next_values.size()), 1,
    {"tap:new"}, next_values.data());
  ASSERT(next.status == tropical_runtime::RenderWindowStatus::Rendered);
  ASSERT(next.program_version > old_result.program_version);
  for (double value : next_values) ASSERT(value == 2.0);
}

// Audio and visualization may be different compiled programs. The scope JIT
// owns its slot layout and workspace; named controls are projected from the
// audio state into that layout on each immutable control publication.
static void test_separate_scope_artifact()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string audio_ir = wrap_loop(
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 2.500000e-01, ptr %op, align 8\n");
  const std::string scope_ir = wrap_loop(
    "  %xp = getelementptr inbounds double, ptr %slots, i64 1\n"
    "  %x = load double, ptr %xp, align 8\n"
    "  %tp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  store double %x, ptr %tp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %x, ptr %op, align 8\n");
  const std::string audio_manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":1,"slot_names":["param:x"],
    "slot_defaults":[3.0],"param_disciplines":[
      {"name":"x","discipline":"raw","companions":[]}
    ]})";
  const std::string scope_manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":2,
    "slot_names":["__root__.tap:mode","param:x"],
    "slot_defaults":[0.0,-1.0],"param_disciplines":[]})";

  ASSERT(rt.load_ir_staged_with_scope(
    audio_ir, "", "", audio_manifest,
    scope_ir, "", scope_manifest));
  rt.process();
  for (double value : rt.outputBuffer) ASSERT(value == 0.25);

  std::array<double, 4> values{};
  auto scope = rt.render_window_low_priority_by_name(
    0, static_cast<uint32_t>(values.size()), 1,
    {"__root__.tap:mode"}, values.data());
  ASSERT(scope.status == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : values) ASSERT(value == 3.0);

  const auto update = rt.dispatch_param_sync("x", 7.0);
  ASSERT(update.ok);
  scope = rt.render_window_low_priority_by_name(
    0, static_cast<uint32_t>(values.size()), 1,
    {"__root__.tap:mode"}, values.data());
  ASSERT(scope.status == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : values) ASSERT(value == 7.0);
  rt.process();
  for (double value : rt.outputBuffer) ASSERT(value == 0.25);
}

// A scope control image contains projected raw controls, not the scalar
// coefficients privately materialized from them. Two reads of one immutable
// control version must therefore reuse the materialized scalar slot image;
// resetting to the raw image without rerunning the coefficient kernel makes
// every read after the first render a different program.
static void test_scope_reuses_materialized_scalar_coefficients()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string audio_ir = wrap_loop(
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 0.000000e+00, ptr %op, align 8\n");
  const std::string scope_ir = wrap_loop(
    "  %cp = getelementptr inbounds double, ptr %slots, i64 2\n"
    "  %coef = load double, ptr %cp, align 8\n"
    "  %tp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  store double %coef, ptr %tp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %coef, ptr %op, align 8\n");
  const std::string scope_coeff_ir = std::string("define void @kernel(")
    + KERNEL_SIG + ") {\n"
      "entry:\n"
      "  %xp = getelementptr inbounds double, ptr %slots, i64 1\n"
      "  %x = load double, ptr %xp, align 8\n"
      "  %c = fmul double %x, 2.000000e+00\n"
      "  %cp = getelementptr inbounds double, ptr %slots, i64 2\n"
      "  store double %c, ptr %cp, align 8\n"
      "  ret void\n}\n";
  const std::string audio_manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":1,"slot_names":["param:x"],
    "slot_defaults":[3.0],"param_disciplines":[
      {"name":"x","discipline":"raw","companions":[]}
    ]})";
  const std::string scope_manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":3,
    "slot_names":["__root__.tap:mode","param:x","coeff:x2"],
    "slot_defaults":[0.0,-1.0,-99.0],"param_disciplines":[]})";

  ASSERT(rt.load_ir_staged_with_scope(
    audio_ir, "", "", audio_manifest,
    scope_ir, scope_coeff_ir, scope_manifest));

  const auto check_twice = [&](double expected) {
    std::array<double, 4> values{};
    for (unsigned int read = 0; read < 2; ++read)
    {
      const auto scope = rt.render_window_low_priority_by_name(
        0, static_cast<uint32_t>(values.size()), 1,
        {"__root__.tap:mode"}, values.data());
      ASSERT(scope.status == tropical_runtime::RenderWindowStatus::Rendered);
      for (double value : values) ASSERT(value == expected);
    }
  };
  check_twice(6.0);
  const auto update = rt.dispatch_param_sync("x", 7.0);
  ASSERT(update.ok);
  check_twice(14.0);
}

// Scope-side coefficient materialization is private to the immutable control
// image: a column update becomes visible coherently without borrowing any of
// the audio state's three mutable generations.
static void test_scope_snapshot_materializes_coefficients()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string ir = wrap_loop(
    "  %a0p = getelementptr inbounds ptr, ptr %arrays, i64 0\n"
    "  %a0 = load ptr, ptr %a0p, align 8\n"
    "  %raw = load i64, ptr %a0, align 8\n"
    "  %v = bitcast i64 %raw to double\n"
    "  %sp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  store double %v, ptr %sp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  const std::string coeff_ir = std::string("define void @kernel(")
    + KERNEL_SIG + ") {\n"
      "entry:\n"
      "  %sp = getelementptr inbounds double, ptr %slots, i64 0\n"
      "  %x = load double, ptr %sp, align 8\n"
      "  %v = fmul double %x, 2.000000e+00\n"
      "  %raw = bitcast double %v to i64\n"
      "  %a0p = getelementptr inbounds ptr, ptr %arrays, i64 0\n"
      "  %a0 = load ptr, ptr %a0p, align 8\n"
      "  store i64 %raw, ptr %a0, align 8\n"
      "  ret void\n}\n";
  const std::string manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":1,"array_slot_sizes":[1],
    "array_slot_names":["coef"],"coeff_array_slots":[0],
    "instance_functions":[],"sinks":[],"slot_count":1,
    "slot_names":["param:x"],"slot_defaults":[1.0],
    "param_disciplines":[
      {"name":"x","discipline":"raw","companions":[]}
    ]})";
  ASSERT(rt.load_ir_staged(ir, "", coeff_ir, manifest));

  const uint32_t slot = 0;
  std::array<double, 4> values{};
  ASSERT(rt.render_window_low_priority(
    0, static_cast<uint32_t>(values.size()), 1,
    &slot, 1, values.data())
    == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : values) ASSERT(value == 2.0);

  const auto update = rt.dispatch_param_sync("x", 3.0);
  ASSERT(update.ok);
  ASSERT(rt.render_window_low_priority(
    0, static_cast<uint32_t>(values.size()), 1,
    &slot, 1, values.data())
    == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : values) ASSERT(value == 6.0);
}

// A coefficient transaction writes slot[1] = 2*slot[0], while the audio
// kernel outputs slot[1] - 2*slot[0]. Old and new generations both produce
// exact zero; any scalar/coeff publication tear is immediately nonzero.
static void test_control_generation_coherence()
{
  constexpr unsigned int buf = 32;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);
  const std::string audio_ir = wrap_loop(
    "  %xp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  %x = load double, ptr %xp, align 8\n"
    "  %cp = getelementptr inbounds double, ptr %slots, i64 1\n"
    "  %coefv = load double, ptr %cp, align 8\n"
    "  %twox = fmul double %x, 2.000000e+00\n"
    "  %v = fsub double %coefv, %twox\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  const std::string coeff_ir = std::string("define void @kernel(")
    + KERNEL_SIG + ") {\n"
      "entry:\n"
      "  %xp = getelementptr inbounds double, ptr %slots, i64 0\n"
      "  %x = load double, ptr %xp, align 8\n"
      "  %v = fmul double %x, 2.000000e+00\n"
      "  %cp = getelementptr inbounds double, ptr %slots, i64 1\n"
      "  store double %v, ptr %cp, align 8\n"
      "  ret void\n}\n";
  const std::string manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":2,"slot_names":["param:x","coef:y"],
    "slot_defaults":[1.0,0.0]})";
  ASSERT_OK(tropical_runtime_load_ir_staged(
    rt, audio_ir.data(), audio_ir.size(), nullptr, 0,
    coeff_ir.data(), coeff_ir.size(), manifest.data(), manifest.size()));

  std::atomic<bool> go{false};
  std::atomic<bool> done{false};
  std::jthread control([&] {
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    for (uint64_t i = 0; i < 20000; ++i)
    {
      tropical_runtime_set_slot(rt, 0, 0.25 + (i % 97) * 0.01);
      if ((i & 7U) == 0) std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  });

  go.store(true, std::memory_order_release);
  for (uint64_t block = 0;
       block < 3000 || !done.load(std::memory_order_acquire); ++block)
  {
    tropical_runtime_process(rt);
    const double * out = tropical_runtime_output_buffer(rt);
    for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.0, 0.0);
  }
  control.join();
  tropical_runtime_free(rt);
}

// Rapid A/B/A publications exercise the state-index advertise/verify
// handshake. Each callback must bind one complete state for its whole block.
static void test_hot_swap_state_handoff()
{
  constexpr unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);
  const std::string one = wrap_loop(
    "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 1.000000e+00, ptr %p, align 8\n");
  const std::string two = wrap_loop(
    "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 2.000000e+00, ptr %p, align 8\n");
  ASSERT_OK(tropical_runtime_load_ir(
    rt, one.data(), one.size(), RAMP_MANIFEST, strlen(RAMP_MANIFEST)));

  std::atomic<bool> go{false};
  std::atomic<bool> done{false};
  std::jthread control([&] {
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    for (unsigned int i = 0; i < 80; ++i)
    {
      const std::string & next = (i & 1U) ? one : two;
      if (!tropical_runtime_load_ir(
            rt, next.data(), next.size(),
            RAMP_MANIFEST, strlen(RAMP_MANIFEST)))
        std::abort();
    }
    done.store(true, std::memory_order_release);
  });

  go.store(true, std::memory_order_release);
  for (uint64_t block = 0;
       block < 1000 || !done.load(std::memory_order_acquire); ++block)
  {
    tropical_runtime_process(rt);
    const double * out = tropical_runtime_output_buffer(rt);
    ASSERT(out[0] == 1.0 || out[0] == 2.0);
    for (unsigned int i = 1; i < buf; ++i) ASSERT(out[i] == out[0]);
  }
  control.join();
  tropical_runtime_free(rt);
}

// Pause after audio owns the published generation. Two complete control
// publications must proceed through the other generations while the captured
// generation remains immutable; the paused callback still renders its old
// value, and the next callback sees the newest value.
static void test_generation_ownership_barrier()
{
  constexpr unsigned int buf = 16;
  tropical_runtime::FlatRuntime rt(buf);
  const std::string ir = wrap_loop(
    "  %xp = getelementptr inbounds double, ptr %slots, i64 0\n"
    "  %x = load double, ptr %xp, align 8\n"
    "  %op = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double %x, ptr %op, align 8\n");
  const std::string manifest = R"({"schema":"tropical_plan_5",
    "config":{"sampleRate":44100},"register_count":0,
    "array_slot_count":0,"array_slot_sizes":[],"instance_functions":[],
    "sinks":[],"slot_count":1,"slot_names":["param:x"],
    "slot_defaults":[1.0]})";
  ASSERT(rt.load_ir(ir, manifest));

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_generation_ownership.store(true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  std::jthread audio([&] { rt.process(); });
  const bool acquired = wait_for_true(seam.generation_owned);
  if (!acquired)
  {
    seam.release_generation.store(true, std::memory_order_release);
    audio.join();
    ASSERT(acquired);
  }

  rt.set_slot(0, 2.0);
  rt.set_slot(0, 3.0);
  seam.release_generation.store(true, std::memory_order_release);
  audio.join();
  rt.set_ownership_test_seam(nullptr);

  for (double sample : rt.outputBuffer) ASSERT(sample == 1.0);
  rt.process();
  for (double sample : rt.outputBuffer) ASSERT(sample == 3.0);
  ASSERT(rt.ownership_failure_count() == 0);
}

// Pause after audio owns the active state. One swap may publish into the
// inactive state; the next swap's attempt to reuse the audio-owned old state
// must block until the callback releases it.
static void test_state_ownership_barrier()
{
  constexpr unsigned int buf = 16;
  tropical_runtime::FlatRuntime rt(buf);
  const std::string one = wrap_loop(
    "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 1.000000e+00, ptr %p, align 8\n");
  const std::string two = wrap_loop(
    "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 2.000000e+00, ptr %p, align 8\n");
  const std::string three = wrap_loop(
    "  %p = getelementptr inbounds double, ptr %output_buffer, i64 %s\n"
    "  store double 3.000000e+00, ptr %p, align 8\n");
  ASSERT(rt.load_ir(one, RAMP_MANIFEST));
  const uint64_t initial_version = rt.recompile_version();

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_state_ownership.store(true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  std::jthread audio([&] { rt.process(); });
  const bool acquired = wait_for_true(seam.state_owned);
  if (!acquired)
  {
    seam.release_state.store(true, std::memory_order_release);
    audio.join();
    ASSERT(acquired);
  }

  std::atomic<bool> loads_ok{true};
  std::atomic<bool> control_done{false};
  std::jthread control([&] {
    if (!rt.load_ir(two, RAMP_MANIFEST)) loads_ok.store(false);
    if (!rt.load_ir(three, RAMP_MANIFEST)) loads_ok.store(false);
    control_done.store(true, std::memory_order_release);
  });

  const bool reuse_blocked = wait_for_true(seam.writer_waiting_for_state);
  const uint64_t version_while_blocked = rt.recompile_version();
  const bool completed_while_blocked =
    control_done.load(std::memory_order_acquire);

  seam.release_state.store(true, std::memory_order_release);
  audio.join();
  control.join();
  rt.set_ownership_test_seam(nullptr);

  ASSERT(reuse_blocked);
  ASSERT(version_while_blocked == initial_version + 1);
  ASSERT(!completed_while_blocked);
  ASSERT(loads_ok.load(std::memory_order_acquire));
  ASSERT(control_done.load(std::memory_order_acquire));
  for (double sample : rt.outputBuffer) ASSERT(sample == 1.0);
  rt.process();
  for (double sample : rt.outputBuffer) ASSERT(sample == 3.0);
  ASSERT(rt.ownership_failure_count() == 0);
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
  ASSERT(tropical_dac_disconnect_count(nullptr) == 0);
  ASSERT(tropical_dac_reconnect_success_count(nullptr) == 0);
  ASSERT(tropical_dac_reconnect_failure_count(nullptr) == 0);
  ASSERT(tropical_runtime_metal_dispatch_failure_count(nullptr) == 0);
  ASSERT(tropical_runtime_metal_render_tile_frames(nullptr) == 0);
  ASSERT(tropical_runtime_ownership_failure_count(nullptr) == 0);
}

struct CaptureTestSource
{
  explicit CaptureTestSource(unsigned int n) : outputBuffer(n, 0.0) {}
  void process()
  {
    ++process_calls;
    for (unsigned int i = 0; i < outputBuffer.size(); ++i)
      outputBuffer[i] = static_cast<double>(sample_index + i);
    sample_index += outputBuffer.size();
  }
  unsigned int getBufferLength() const
  {
    return static_cast<unsigned int>(outputBuffer.size());
  }
  uint64_t current_sample_index() const { return sample_index; }
  void begin_fade_in(int = 2048) {}
  void begin_fade_out(int = 2048) {}
  bool is_fade_out_complete() const { return true; }

  std::vector<double> outputBuffer;
  uint64_t sample_index = 0;
  unsigned int process_calls = 0;
};

struct ReadyAwareStartSource : CaptureTestSource
{
  using CaptureTestSource::CaptureTestSource;

  void prepare_realtime_start(unsigned int process_cycles)
  {
    ++prepare_calls;
    requested_process_cycles = process_cycles;
  }

  unsigned int prepare_calls = 0;
  unsigned int requested_process_cycles = 0;
};

static void test_dac_start_source_preparation()
{
  constexpr unsigned int frames = 8;
  CaptureTestSource legacy(frames);
  TropicalDACImpl<CaptureTestSource> legacy_dac(&legacy, 44100, 1);
  legacy_dac.prepare_source_for_start();
  ASSERT(legacy.process_calls == kPrimeCycles);
  ASSERT(legacy.sample_index == frames * kPrimeCycles);

  ReadyAwareStartSource ready_aware(frames);
  TropicalDACImpl<ReadyAwareStartSource> ready_dac(
    &ready_aware, 44100, 1);
  ready_dac.prepare_source_for_start();
  ASSERT(ready_aware.prepare_calls == 1);
  ASSERT(ready_aware.requested_process_cycles == kPrimeCycles);
  ASSERT(ready_aware.process_calls == 0);
  ASSERT(ready_aware.sample_index == 0);

  ready_dac.prepare_source_for_stream_restart();
  ASSERT(ready_aware.prepare_calls == 2);
  ASSERT(ready_aware.requested_process_cycles == 0);
  ASSERT(ready_aware.process_calls == 0);

  legacy_dac.prepare_source_for_stream_restart();
  ASSERT(legacy.process_calls == kPrimeCycles);
}

static void test_capture_serial_state_machine()
{
  constexpr unsigned int frames = 8;
  CaptureTestSource source(frames);
  TropicalDACImpl<CaptureTestSource> dac(&source, 44100, 1);

  const uint64_t sequence = dac.request_output_capture();
  ASSERT(sequence != 0);
  ASSERT(dac.request_output_capture() == 0);
  double device[frames]{};
  ASSERT(TropicalDACImpl<CaptureTestSource>::fill_buffer(
           device, nullptr, frames, 0.0, 0, &dac) == 0);

  uint64_t start = UINT64_MAX;
  double captured[frames]{};
  ASSERT(!dac.read_output_capture(
    sequence + 1, start, captured, frames));
  ASSERT(dac.read_output_capture(sequence, start, captured, frames));
  ASSERT(start == 0);
  for (unsigned int i = 0; i < frames; ++i)
    ASSERT(captured[i] == static_cast<double>(i));
  ASSERT(!dac.read_output_capture(sequence, start, captured, frames));
  ASSERT(dac.request_output_capture() != 0);
}

int main()
{
  printf("test_module_process (load_ir engine)\n");

  run_test("constant kernel via load_ir",   test_ir_constant);
  run_test("plan-5/6 manifest boundary", test_plan5_only_manifest_boundary);
  run_test("Plan-6 immutable asset JIT/lifetime", test_plan6_immutable_asset_jit);
  run_test("scope snapshot retains Plan-6 asset through state reuse",
           test_scope_snapshot_retains_plan6_asset);
  run_test("Plan-6 immutable asset refusals", test_plan6_immutable_asset_refusals);
  run_test("closed-form index ramp",        test_ir_index_ramp);
  run_test("clock request boundary handoff", test_clock_boundary_handoff);
  run_test("clock odd-sequence barrier", test_clock_request_barrier);
  run_test("param dispatch exact-sample replay",
           test_param_dispatch_exact_sample_replay);
  run_test("param dispatch effective-boundary races",
           test_param_dispatch_effective_boundary_races);
  run_test("scope snapshot never blocks control publication",
           test_scope_snapshot_does_not_block_control);
  run_test("named scope render pins one hot-swap program",
           test_scope_named_render_pins_program);
  run_test("separate scope artifact projects controls by name",
           test_separate_scope_artifact);
  run_test("scope reuses materialized scalar coefficients",
           test_scope_reuses_materialized_scalar_coefficients);
  run_test("scope snapshot materializes coefficient columns privately",
           test_scope_snapshot_materializes_coefficients);
  run_test("coherent slot/coefficient generation", test_control_generation_coherence);
  run_test("hot-swap state handoff", test_hot_swap_state_handoff);
  run_test("owned generation survives two publications", test_generation_ownership_barrier);
  run_test("owned state blocks ABA reuse", test_state_ownership_barrier);
  run_test("device-boundary clamp",         test_device_bound_clamp);
  run_test("fixed DAC histogram/ABI contract", test_dac_histogram_contract);
  run_test("DAC startup honors source readiness barrier",
           test_dac_start_source_preparation);
  run_test("serial output-capture state machine", test_capture_serial_state_machine);

  printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
