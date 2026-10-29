/**
 * test_module_process.cpp
 *
 * Exercises the current C++ engine with hand-written LLVM IR at the port
 * (design/typed-kernel.md): `tropical_runtime_load` → compile_ir_text →
 * JIT → run, plus the device-boundary clamp. Lean's EmitLlvm owns production codegen;
 * arithmetic/op correctness lives in the Lean `tropicaltest` gates.
 * These tests cover the engine half only. They do not assert plan
 * instruction semantics or per-sample state.
 */

#include "c_api/tropical_c.h"
#include "c_api/tropical_socket.hpp"
#include "dac/TropicalDAC.hpp"   // kDeviceOutputBound / clamp_to_device_bound
#include "runtime/FlatRuntime.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

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

// The port (design/typed-kernel.md): `tropical_kernel(ptr tau, ptr in_1..k,
// ptr out)` over a static frame count, plus the argv thunk compile_ir_text
// looks up. Lean's EmitLlvm emits exactly this shape; these hand kernels
// mirror it. Inside the loop `%s` is the frame, `%tau_s` the frame's Q32.32
// coordinate, `%idx` its integer part (`tau[s] >> 32`, which these hand
// kernels ramp on), `%in_<j>` the j-th param input, `%out` the output.
static std::string wrap_kernel(unsigned int frames, unsigned int params,
                               const std::string& body)
{
  std::string args = "ptr %tau";
  for (unsigned int j = 1; j <= params; ++j)
    args += ", ptr %in_" + std::to_string(j);
  args += ", ptr %out";
  std::string loads, call_args;
  for (unsigned int i = 0; i < params + 2; ++i)
  {
    loads += "  %ap" + std::to_string(i)
      + " = getelementptr inbounds ptr, ptr %argv, i64 " + std::to_string(i) + "\n"
      "  %a" + std::to_string(i) + " = load ptr, ptr %ap" + std::to_string(i) + ", align 8\n";
    call_args += (i ? ", ptr %a" : "ptr %a") + std::to_string(i);
  }
  return "define void @tropical_kernel(" + args + ") {\n"
    "entry:\n  br label %lc\n"
    "lc:\n  %s = phi i64 [0, %entry], [%sn, %lb]\n"
    "  %c = icmp ult i64 %s, " + std::to_string(frames) + "\n"
    "  br i1 %c, label %lb, label %le\n"
    "lb:\n"
    "  %tau_p = getelementptr inbounds i64, ptr %tau, i64 %s\n"
    "  %tau_s = load i64, ptr %tau_p, align 8\n"
    "  %idx = ashr i64 %tau_s, 32\n"
    + body +
    "  %sn = add i64 %s, 1\n  br label %lc\n"
    "le:\n  ret void\n}\n\n"
    "define void @tropical_kernel_argv(ptr %argv) {\n"
    "entry:\n" + loads +
    "  call void @tropical_kernel(" + call_args + ")\n"
    "  ret void\n}\n";
}

// A tropical_kernel_1 manifest: tau[frames], one f64 param input per name,
// out[frames, channels] with the given channel names, and the nested plan
// (slot table + disciplines) a host still reads.
static std::string kernel_manifest(unsigned int frames,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& channels,
                                   const std::string& plan_json)
{
  nlohmann::json inputs = nlohmann::json::array();
  nlohmann::json tau;
  tau["name"] = "tau"; tau["elem"] = "i64q32";
  tau["shape"] = nlohmann::json::array({frames}); tau["role"] = "signal";
  inputs.push_back(tau);
  for (const auto& name : params)
  {
    nlohmann::json port;
    port["name"] = name; port["elem"] = "f64";
    port["shape"] = nlohmann::json::array(); port["role"] = "param";
    inputs.push_back(port);
  }
  nlohmann::json out;
  out["name"] = "out"; out["elem"] = "f64";
  out["shape"] = nlohmann::json::array(
    {frames, static_cast<unsigned int>(channels.size())});
  out["channels"] = channels;
  nlohmann::json manifest;
  manifest["schema"] = "tropical_kernel_1";
  manifest["rate"] = 44100;
  manifest["inputs"] = inputs;
  manifest["outputs"] = nlohmann::json::array({out});
  manifest["realization"] = {{"mode", "fused"}, {"register_count", 0}};
  manifest["plan"] = nlohmann::json::parse(plan_json);
  return manifest.dump();
}

static std::string plan_json(unsigned int channels,
                             const std::vector<std::string>& slot_names,
                             const std::vector<double>& slot_defaults,
                             const std::string& extra = "")
{
  nlohmann::json plan;
  plan["schema"] = "tropical_plan_6";
  plan["config"] = {{"sampleRate", 44100}};
  plan["output_channel_count"] = channels;
  plan["register_count"] = 0;
  plan["array_slot_count"] = 0;
  plan["array_slot_sizes"] = nlohmann::json::array();
  plan["instance_functions"] = nlohmann::json::array();
  plan["slot_count"] = static_cast<unsigned int>(slot_names.size());
  plan["slot_names"] = slot_names;
  plan["slot_defaults"] = slot_defaults;
  if (!extra.empty())
    plan.merge_patch(nlohmann::json::parse(extra));
  return plan.dump();
}

// A frames-wide constant kernel with no params and one channel.
static std::string constant_manifest(unsigned int frames, unsigned int channels = 1)
{
  std::vector<std::string> channel_names(channels, "");
  channel_names[0] = "out";
  return kernel_manifest(frames, {}, channel_names, plan_json(channels, {}, {}));
}

static std::string stereo_constant_ir(unsigned int frames)
{
  return wrap_kernel(frames, 0,
    "  %frame_base = mul i64 %s, 2\n"
    "  %left = getelementptr inbounds double, ptr %out, i64 %frame_base\n"
    "  store double 2.500000e-01, ptr %left, align 8\n"
    "  %right_index = add i64 %frame_base, 1\n"
    "  %right = getelementptr inbounds double, ptr %out, i64 %right_index\n"
    "  store double -5.000000e-01, ptr %right, align 8\n");
}

static bool wait_for_true(
  const std::atomic<bool> & value,
  std::chrono::milliseconds timeout = std::chrono::seconds(10))
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!value.load(std::memory_order_acquire))
  {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

static bool socket_request(const std::string & path,
                           const std::string & request,
                           std::string & response)
{
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return false;
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
  if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0)
  {
    ::close(fd);
    return false;
  }
  const std::string line = request + "\n";
  if (::write(fd, line.data(), line.size())
      != static_cast<ssize_t>(line.size()))
  {
    ::close(fd);
    return false;
  }
  response.clear();
  char buffer[4096];
  while (response.find('\n') == std::string::npos)
  {
    const ssize_t got = ::read(fd, buffer, sizeof(buffer));
    if (got <= 0) break;
    response.append(buffer, static_cast<size_t>(got));
  }
  ::close(fd);
  response.resize(response.find('\n'));
  return true;
}

static bool load_c(tropical_runtime_t rt, const std::string& ir,
                   const std::string& manifest)
{
  return tropical_runtime_load(rt, ir.c_str(), ir.size(), nullptr, 0,
                               nullptr, 0, manifest.c_str(), manifest.size());
}

/**
 * 1. Constant kernel via load — proves textual IR parses, JITs, runs.
 */
static void test_ir_constant()
{
  const unsigned int buf = 64;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);

  const std::string ir = wrap_kernel(buf, 0,
    "  %p = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double 5.000000e-01, ptr %p, align 8\n");
  ASSERT_OK(load_c(rt, ir, constant_manifest(buf)));
  tropical_runtime_process(rt);
  const double* out = tropical_runtime_output_buffer(rt);
  ASSERT(out != nullptr);
  ASSERT(tropical_runtime_output_channel_count(rt) == 1);
  ASSERT(tropical_runtime_interleaved_output_buffer(rt) == out);
  for (unsigned int i = 0; i < buf; ++i) ASSERT_NEAR(out[i], 0.5, 1e-12);

  tropical_runtime_free(rt);
}

static void test_interleaved_native_output()
{
  constexpr unsigned int frames = 8;
  tropical_runtime_t rt = tropical_runtime_new(frames);
  ASSERT(rt != nullptr);
  const std::string stereo_ir = stereo_constant_ir(frames);
  const std::string stereo_manifest = constant_manifest(frames, 2);
  ASSERT_OK(load_c(rt, stereo_ir, stereo_manifest));
  ASSERT(tropical_runtime_output_channel_count(rt) == 2);
  ASSERT(tropical_runtime_interleaved_output_buffer(rt)
         != tropical_runtime_output_buffer(rt));
  tropical_runtime_process(rt);

  ASSERT(tropical_runtime_output_channel_count(rt) == 2);
  const double * compact =
    tropical_runtime_interleaved_output_buffer(rt);
  const double * mono = tropical_runtime_output_buffer(rt);
  ASSERT(compact != nullptr);
  ASSERT(mono != nullptr);
  ASSERT(compact != mono);
  for (unsigned int frame = 0; frame < frames; ++frame)
  {
    ASSERT_NEAR(compact[frame * 2], 0.25, 0.0);
    ASSERT_NEAR(compact[frame * 2 + 1], -0.5, 0.0);
    ASSERT_NEAR(mono[frame], compact[frame * 2], 0.0);
  }

  // A kernel that writes nothing relies on the runtime's full B*C pre-clear.
  // Hot-swap from nonzero stereo proves this is not allocator-zero luck. The
  // maximum supported width uses the same fixed callback scratch.
  const std::string silent_ir = wrap_kernel(frames, 0, "");
  ASSERT_OK(load_c(rt, silent_ir, constant_manifest(frames, 64)));
  tropical_runtime_process(rt);
  ASSERT(tropical_runtime_output_channel_count(rt) == 64);
  compact = tropical_runtime_interleaved_output_buffer(rt);
  for (std::size_t i = 0; i < frames * 64; ++i)
    ASSERT_NEAR(compact[i], 0.0, 0.0);
  for (unsigned int frame = 0; frame < frames; ++frame)
    ASSERT_NEAR(tropical_runtime_output_buffer(rt)[frame], 0.0, 0.0);

  // One frame gain is shared across every channel: a fade must preserve the
  // stereo relationship, and the compatibility view follows faded channel 0.
  tropical_runtime::FlatRuntime faded(frames);
  ASSERT(faded.load(stereo_ir, "", "", stereo_manifest));
  faded.begin_fade_in();
  faded.process();
  const double * faded_compact = faded.getInterleavedOutputBuffer();
  ASSERT(faded.getOutputChannelCount() == 2);
  ASSERT_NEAR(faded_compact[0], 0.0, 0.0);
  ASSERT_NEAR(faded_compact[1], 0.0, 0.0);
  for (unsigned int frame = 1; frame < frames; ++frame)
  {
    ASSERT_NEAR(faded_compact[frame * 2 + 1],
                -2.0 * faded_compact[frame * 2], 1e-15);
    ASSERT_NEAR(faded.outputBuffer[frame], faded_compact[frame * 2], 0.0);
  }
}

// A sample-index ramp — closed-form, no state. Exercises the tau input
// (`%idx = tau[s] >> 32`) and the output store. The kernel is a pure
// function of the clock it is handed.
static const char* RAMP_BODY =
  "  %v = sitofp i64 %idx to double\n"
  "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
  "  store double %v, ptr %op, align 8\n";

static bool load_fails_with(tropical_runtime_t runtime, const std::string& ir,
                            const std::string& manifest, const char* expected)
{
  if (load_c(runtime, ir, manifest)) return false;
  const char* error = tropical_last_error();
  return error && std::strstr(error, expected);
}

/**
 * 2. The native boundary accepts exactly tropical_kernel_1 over a nested
 *    tropical_plan_6. Other schemas, a missing plan, a malformed port, a
 *    frame count other than the runtime's, and every retired plan carrier
 *    fail before caller-supplied IR is loaded.
 */
static void test_manifest_boundary()
{
  tropical_runtime_t rt = tropical_runtime_new(16);
  ASSERT(rt != nullptr);
  const std::string ir = wrap_kernel(16, 0, RAMP_BODY);
  const auto with_plan = [](const char* plan) {
    return kernel_manifest(16, {}, {"out"}, plan);
  };

  ASSERT(load_fails_with(
    rt, ir, R"({"schema":"tropical_plan_6"})",
    "unsupported manifest schema 'tropical_plan_6'; expected 'tropical_kernel_1'"));
  ASSERT(load_fails_with(
    rt, ir, R"({})",
    "unsupported manifest schema ''; expected 'tropical_kernel_1'"));
  ASSERT(load_fails_with(
    rt, ir, R"({"schema":"tropical_kernel_1"})",
    "kernel manifest has no plan"));
  ASSERT(load_fails_with(
    rt, ir, with_plan(R"({"schema":"tropical_plan_4"})"),
    "unsupported manifest schema 'tropical_plan_4'; expected 'tropical_plan_6'"));
  ASSERT(load_fails_with(
    rt, ir, with_plan(R"({"schema":"tropical_plan_6","state_init":[1]})"),
    "retired field 'state_init' is not valid in tropical_plan_6"));
  ASSERT(load_fails_with(
    rt, ir,
    with_plan(R"({"schema":"tropical_plan_6","output_targets":[],"outputs":[]})"),
    "retired field 'output_targets' is not valid in tropical_plan_6"));
  ASSERT(load_fails_with(
    rt, ir,
    with_plan(R"({"schema":"tropical_plan_6","instance_functions":[{"instructions":[{"tag":"Add","dst":0,"args":[],"result_type":"float"}]}]})"),
    "plan-6 instruction missing required dst_kind"));
  ASSERT(load_fails_with(
    rt, ir,
    with_plan(R"({"schema":"tropical_plan_6","instance_functions":[{"instructions":[{"tag":"Add","dst":0,"dst_kind":"temp","args":[{"kind":"rate","scalar_type":"float"}],"result_type":"float"}]}]})"),
    "unknown operand kind 'rate'"));
  ASSERT(load_fails_with(
    rt, ir, with_plan(R"({"schema":"tropical_plan_6","output_channel_count":0})"),
    "output_channel_count must be in [1,64]"));
  ASSERT(load_fails_with(
    rt, ir,
    with_plan(R"({"schema":"tropical_plan_6","output_channel_count":2,"sinks":[{"target":2}]})"),
    "sink target 2 is out of range"));
  ASSERT(load_fails_with(
    rt, ir,
    with_plan(R"({"schema":"tropical_plan_6","output_channel_count":2,"sinks":[{"target":0},{"target":0}]})"),
    "duplicate sink target 0"));
  ASSERT(load_fails_with(
    rt, ir, with_plan(R"({"schema":"tropical_plan_6","compilation_mode":"microkernel"})"),
    "realization 'microkernel' is not a host-visible mode"));
  // The port itself: tau first, a frame count that is the runtime's, one
  // output whose width matches the nested plan, and an entry the IR defines.
  ASSERT(load_fails_with(
    rt, ir, kernel_manifest(32, {}, {"out"}, plan_json(1, {}, {})),
    "artifact compiled for 32 frames per call; this runtime renders 16"));
  ASSERT(load_fails_with(
    rt, ir, kernel_manifest(16, {}, {"out", "right"}, plan_json(1, {}, {})),
    "kernel output width disagrees with the nested plan"));
  ASSERT(load_fails_with(
    rt, ir, kernel_manifest(16, {"param:missing"}, {"out"}, plan_json(1, {}, {})),
    "kernel input 'param:missing' names no slot of the nested plan"));
  ASSERT(load_fails_with(
    rt, "define void @not_the_port() {\nentry:\n  ret void\n}\n",
    constant_manifest(16),
    "IR does not define 'tropical_kernel_argv'"));

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

  const std::string ir = wrap_kernel(buf, 0, RAMP_BODY);
  ASSERT_OK(load_c(rt, ir, constant_manifest(buf)));

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
  const std::string ir = wrap_kernel(buf, 0, RAMP_BODY);
  ASSERT_OK(load_c(rt, ir, constant_manifest(buf)));

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
  // The Q32.32 rail carries 2^31 integer sample indices exactly; this is
  // far past any session but inside the rail.
  constexpr uint64_t far_exact = (uint64_t{1} << 30) + 123;
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
  const std::string ir = wrap_kernel(buf, 0, RAMP_BODY);
  ASSERT(rt.load(ir, "", "", constant_manifest(buf)));
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

// The four-discipline control slot table the dispatch tests share.
static const std::vector<std::string> DISCIPLINE_SLOTS = {
  "param:bank.freq", "param:canary.morph#v0", "param:canary.morph#v1",
  "param:canary.morph#t0", "param:canary.freq", "param:canary.freq#phase",
  "param:master.velocity", "param:master.tau_base"};
static const char* DISCIPLINE_TABLE = R"({"param_disciplines":[
  {"name":"bank.freq","discipline":"raw","companions":[]},
  {"name":"canary.morph","discipline":"glide","glide_dur_sec":0.02,
   "companions":["canary.morph#v0","canary.morph#v1","canary.morph#t0"]},
  {"name":"canary.freq","discipline":"anchor","companions":["canary.freq#phase"]},
  {"name":"master.velocity","discipline":"velocity","companions":["master.tau_base"]}
]})";

static std::string discipline_manifest(unsigned int frames)
{
  return kernel_manifest(frames, DISCIPLINE_SLOTS, {"out"},
    plan_json(1, DISCIPLINE_SLOTS, {180.0, 0.0, 0.0, 0.0, 55.0, 0.0, 1.0, 0.0},
              DISCIPLINE_TABLE));
}

// A reference oracle must replay each production dispatch at the first output
// sample where its generation is audible. Freezing one batch-start boundary
// can accumulate a visible phase/origin error when the callback advances
// between the four discipline writes.
static void test_param_dispatch_exact_sample_replay()
{
  constexpr unsigned int buf = 512;
  constexpr uint64_t base = uint64_t{1} << 30;
  tropical_runtime::FlatRuntime live(buf);
  tropical_runtime::FlatRuntime replay(buf);
  tropical_runtime::FlatRuntime stale_batch_oracle(buf);
  const std::string ir = wrap_kernel(buf, 8,
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double 0.000000e+00, ptr %op, align 8\n");
  const std::string manifest = discipline_manifest(buf);
  ASSERT(live.load(ir, "", "", manifest));
  ASSERT(replay.load(ir, "", "", manifest));
  ASSERT(stale_batch_oracle.load(ir, "", "", manifest));

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
  const std::string ir = wrap_kernel(buf, 8,
    "  %v = load double, ptr %in_1, align 8\n"
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  ASSERT(rt.load(ir, "", "", discipline_manifest(buf)));

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

// An observation kernel: one param `param:x` and one output channel named
// `tap:x` carrying `f(x)`. The scope reads output channels by name.
static std::string tap_manifest(unsigned int frames, const char* tap,
                                double x_default)
{
  return kernel_manifest(frames, {"param:x"}, {tap},
    plan_json(1, {"param:x"}, {x_default},
      R"({"param_disciplines":[{"name":"x","discipline":"raw","companions":[]}]})"));
}
static const char* PASS_X_BODY =
  "  %x = load double, ptr %in_1, align 8\n"
  "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
  "  store double %x, ptr %op, align 8\n";

// One immutable observation generation remains coherent while a raw control
// publication completes. The paused reader must neither block control nor tear
// to the newly published slot image.
static void test_observation_snapshot_does_not_block_control()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string ir = wrap_kernel(16, 1, PASS_X_BODY);
  ASSERT(rt.load(ir, "", "", tap_manifest(16, "tap:x", 1.0)));

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_observation_coordinate.store(
    true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  std::array<double, 64> old_values{};
  tropical_runtime::ObservationRenderResult old_result;
  std::jthread observer([&] {
    old_result = rt.render_window_observation_by_name(
      0, static_cast<uint32_t>(old_values.size()), 1,
      {"tap:x"}, old_values.data());
  });
  const bool paused = wait_for_true(seam.observation_coordinate_completed);
  if (!paused)
  {
    seam.release_observation_coordinate.store(
      true, std::memory_order_release);
    observer.join();
    rt.set_ownership_test_seam(nullptr);
    ASSERT(paused);
  }

  tropical_runtime::ParamDispatchResult control_result;
  std::atomic<bool> control_done{false};
  std::jthread control([&] {
    control_result = rt.dispatch_param_sync("x", 0.5);
    control_done.store(true, std::memory_order_release);
  });
  const bool completed_while_paused = wait_for_true(control_done);
  seam.release_observation_coordinate.store(true, std::memory_order_release);
  observer.join();
  control.join();
  rt.set_ownership_test_seam(nullptr);

  ASSERT(completed_while_paused);
  ASSERT(control_result.ok);
  ASSERT(old_result.status == tropical_runtime::RenderWindowStatus::Rendered);
  for (double value : old_values) ASSERT(value == 1.0);

  std::array<double, 4> next_values{};
  const auto next = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(next_values.size()), 1,
    {"tap:x"}, next_values.data());
  ASSERT(next.status == tropical_runtime::RenderWindowStatus::Rendered);
  ASSERT(next.program_version == old_result.program_version);
  ASSERT(next.control_version > old_result.control_version);
  ASSERT(next.effective_sample_index ==
         control_result.effective_sample_index);
  for (double value : next_values) ASSERT(value == 0.5);
}

// Name resolution, strided coordinates, and evaluation all pin one program
// image. A hot-swap while the old observer is paused cannot reinterpret the
// old name through the new channel layout. The window (64 coordinates) is
// wider than the kernel's frame count (16): the chunked evaluation is exact.
static void test_named_observation_pins_hot_swap_program()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string coordinate_ir = wrap_kernel(16, 1,
    "  %v = uitofp i64 %idx to double\n"
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  const std::string value_ir = wrap_kernel(16, 1, PASS_X_BODY);
  ASSERT(rt.load(coordinate_ir, "", "", tap_manifest(16, "tap:old", 0.0)));

  tropical_runtime::RuntimeOwnershipTestSeam seam;
  seam.pause_after_observation_coordinate.store(
    true, std::memory_order_relaxed);
  rt.set_ownership_test_seam(&seam);
  std::array<double, 40> old_values{};
  tropical_runtime::ObservationRenderResult old_result;
  std::jthread old_observer([&] {
    old_result = rt.render_window_observation_by_name(
      10, static_cast<uint32_t>(old_values.size()), 3,
      {"tap:old"}, old_values.data());
  });
  const bool paused = wait_for_true(seam.observation_coordinate_completed);
  if (!paused)
  {
    seam.release_observation_coordinate.store(
      true, std::memory_order_release);
    old_observer.join();
    rt.set_ownership_test_seam(nullptr);
    ASSERT(paused);
  }

  ASSERT(rt.load(value_ir, "", "", tap_manifest(16, "tap:new", 2.0)));
  seam.release_observation_coordinate.store(true, std::memory_order_release);
  old_observer.join();
  rt.set_ownership_test_seam(nullptr);

  ASSERT(old_result.status == tropical_runtime::RenderWindowStatus::Rendered);
  for (uint32_t i = 0; i < old_values.size(); ++i)
    ASSERT(old_values[i] == 10.0 + 3.0 * i);

  std::array<double, 4> next_values{};
  const auto missing = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(next_values.size()), 1,
    {"tap:old"}, next_values.data());
  ASSERT(missing.status ==
         tropical_runtime::RenderWindowStatus::InvalidArgument);
  ASSERT(missing.unknown_slot == "tap:old");
  const auto next = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(next_values.size()), 1,
    {"tap:new"}, next_values.data());
  ASSERT(next.status == tropical_runtime::RenderWindowStatus::Rendered);
  ASSERT(next.program_version > old_result.program_version);
  for (double value : next_values) ASSERT(value == 2.0);
}

// Audio and observation may be different compiled programs with unrelated
// slot indices. One publication installs both; controls project by exact name
// and the observation kernel derives its own coefficients inside itself.
static void test_separate_observation_artifact_projects_controls()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string audio_ir = wrap_kernel(16, 1,
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double 2.500000e-01, ptr %op, align 8\n");
  const std::string observation_ir = wrap_kernel(16, 1,
    "  %x = load double, ptr %in_1, align 8\n"
    "  %coef = fmul double %x, 2.000000e+00\n"
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double %coef, ptr %op, align 8\n");
  const std::string audio_manifest = kernel_manifest(16, {"param:x"}, {"dac"},
    plan_json(1, {"param:x"}, {3.0},
      R"({"param_disciplines":[{"name":"x","discipline":"raw","companions":[]}]})"));
  const std::string observation_manifest = tap_manifest(16, "tap:mode", -1.0);

  const auto generation =
    rt.load_with_observation_generation(
      audio_ir, "", "", audio_manifest,
      observation_ir, observation_manifest);
  ASSERT(generation.program_version == rt.recompile_version());
  rt.process();
  for (double value : rt.outputBuffer) ASSERT(value == 0.25);

  std::array<double, 4> values{};
  auto observation = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(values.size()), 1,
    {"tap:mode"}, values.data());
  ASSERT(observation.status == tropical_runtime::RenderWindowStatus::Rendered);
  ASSERT(observation.program_version == generation.program_version);
  ASSERT(observation.control_version == generation.control_version);
  for (double value : values) ASSERT(value == 6.0);

  const auto update = rt.dispatch_param_sync("x", 7.0);
  ASSERT(update.ok);
  observation = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(values.size()), 1,
    {"tap:mode"}, values.data());
  ASSERT(observation.program_version == generation.program_version);
  ASSERT(observation.control_version > generation.control_version);
  for (double value : values) ASSERT(value == 14.0);
  rt.process();
  for (double value : rt.outputBuffer) ASSERT(value == 0.25);
}

// C generation APIs return the exact initial pair from their publication,
// while a failed second-artifact compile publishes neither half and zeros its
// out-param. Existing single-artifact behavior remains the first case here.
static void test_publication_generation_tokens_and_atomic_failure()
{
  tropical_runtime_t handle = tropical_runtime_new(16);
  ASSERT(handle != nullptr);
  auto & rt = *static_cast<tropical_runtime::FlatRuntime *>(handle);
  const std::string single_ir = wrap_kernel(16, 1, PASS_X_BODY);
  const std::string single_manifest = tap_manifest(16, "tap:single", 1.25);
  tropical_runtime_generation_t first{};
  ASSERT(tropical_runtime_load_generation(
    handle, single_ir.data(), single_ir.size(), nullptr, 0, nullptr, 0,
    single_manifest.data(), single_manifest.size(), &first));
  std::array<double, 2> values{};
  auto rendered = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(values.size()), 1,
    {"tap:single"}, values.data());
  ASSERT(rendered.program_version == first.program_version);
  ASSERT(rendered.control_version == first.control_version);
  for (double value : values) ASSERT(value == 1.25);

  const std::string audio_ir = wrap_kernel(16, 1,
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double 2.500000e-01, ptr %op, align 8\n");
  const std::string observation_ir = wrap_kernel(16, 1, PASS_X_BODY);
  const std::string audio_manifest = kernel_manifest(16, {"param:x"}, {"dac"},
    plan_json(1, {"param:x"}, {3.0},
      R"({"param_disciplines":[{"name":"x","discipline":"raw","companions":[]}]})"));
  const std::string observation_manifest = tap_manifest(16, "tap:projected", -1.0);
  tropical_runtime_generation_t second{};
  ASSERT(tropical_runtime_load_with_observation_generation(
    handle,
    audio_ir.data(), audio_ir.size(), nullptr, 0, nullptr, 0,
    audio_manifest.data(), audio_manifest.size(),
    observation_ir.data(), observation_ir.size(),
    observation_manifest.data(), observation_manifest.size(), &second));
  ASSERT(second.program_version == first.program_version + 1);
  ASSERT(second.control_version > first.control_version);
  rendered = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(values.size()), 1,
    {"tap:projected"}, values.data());
  ASSERT(rendered.program_version == second.program_version);
  ASSERT(rendered.control_version == second.control_version);
  for (double value : values) ASSERT(value == 3.0);

  tropical_runtime_generation_t failed{99, 101};
  const std::string replacement_audio = wrap_kernel(16, 1,
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double 7.500000e-01, ptr %op, align 8\n");
  const std::string invalid_observation = "not LLVM IR";
  ASSERT(!tropical_runtime_load_with_observation_generation(
    handle,
    replacement_audio.data(), replacement_audio.size(), nullptr, 0,
    nullptr, 0, audio_manifest.data(), audio_manifest.size(),
    invalid_observation.data(), invalid_observation.size(),
    observation_manifest.data(), observation_manifest.size(), &failed));
  ASSERT(failed.program_version == 0);
  ASSERT(failed.control_version == 0);
  ASSERT(rt.recompile_version() == second.program_version);
  rt.process();
  for (double value : rt.outputBuffer) ASSERT(value == 0.25);
  rendered = rt.render_window_observation_by_name(
    0, static_cast<uint32_t>(values.size()), 1,
    {"tap:projected"}, values.data());
  ASSERT(rendered.program_version == second.program_version);
  ASSERT(rendered.control_version == second.control_version);
  for (double value : values) ASSERT(value == 3.0);
  tropical_runtime_free(handle);
}

// The playback-anchored socket form captures the authoritative runtime
// position and returns it with one version-pinned, point-budgeted frame. The
// explicit-start spelling remains accepted for existing callers.
static void test_playback_anchored_observation_socket_frame()
{
  tropical_runtime::FlatRuntime rt(16);
  const std::string ir = wrap_kernel(16, 0,
    "  %v = uitofp i64 %idx to double\n"
    "  %op = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double %v, ptr %op, align 8\n");
  ASSERT(rt.load(ir, "", "", kernel_manifest(16, {}, {"tap:ramp"}, plan_json(1, {}, {}))));
  rt.process();
  rt.process();
  ASSERT(rt.current_sample_index() == 32);

  const std::string socket_path =
    "/tmp/tropical-observation-" + std::to_string(::getpid()) + ".sock";
  tropical_socket::SocketServer server(&rt, socket_path);
  ASSERT(server.start());

  nlohmann::json request = {
    {"jsonrpc", "2.0"}, {"id", 1}, {"method", "render_window"},
    {"params", {
      {"anchor", "playback"}, {"count", 6}, {"point_budget", 3},
      {"slots", {"tap:ramp"}},
    }},
  };
  std::string response_text;
  ASSERT(socket_request(socket_path, request.dump(), response_text));
  const auto response = nlohmann::json::parse(response_text);
  const auto & result = response.at("result");
  ASSERT(result.at("anchor") == "playback");
  ASSERT(result.at("playback_position") == 32);
  ASSERT(result.at("start") == 26);
  ASSERT(result.at("span") == 6);
  ASSERT(result.at("stride") == 2);
  ASSERT(result.at("count") == 3);
  ASSERT(!result.at("preempted").get<bool>());
  ASSERT(result.at("program_version").get<uint64_t>() > 0);
  ASSERT(result.at("control_version").get<uint64_t>() > 0);
  ASSERT(result.at("values")[0][0] == 26.0);
  ASSERT(result.at("values")[0][1] == 28.0);
  ASSERT(result.at("values")[0][2] == 30.0);

  request["id"] = 2;
  request["params"] = {
    {"start", 5}, {"count", 3}, {"slots", {"tap:ramp"}},
  };
  ASSERT(socket_request(socket_path, request.dump(), response_text));
  const auto explicit_response = nlohmann::json::parse(response_text);
  const auto & explicit_result = explicit_response.at("result");
  ASSERT(explicit_result.at("start") == 5);
  ASSERT(!explicit_result.contains("playback_position"));
  ASSERT(explicit_result.at("values")[0][0] == 5.0);
  ASSERT(explicit_result.at("values")[0][1] == 6.0);
  ASSERT(explicit_result.at("values")[0][2] == 7.0);

  // The source-sample span may exceed the response budget. Only the bounded
  // number of strided coordinates is evaluated and returned.
  request["id"] = 3;
  request["params"] = {
    {"start", 5}, {"count", 100000}, {"point_budget", 3},
    {"slots", {"tap:ramp"}},
  };
  ASSERT(socket_request(socket_path, request.dump(), response_text));
  const auto long_response = nlohmann::json::parse(response_text);
  const auto & long_result = long_response.at("result");
  ASSERT(long_result.at("span") == 100000);
  ASSERT(long_result.at("stride") == 33334);
  ASSERT(long_result.at("count") == 3);
  ASSERT(long_result.at("values")[0][0] == 5.0);
  ASSERT(long_result.at("values")[0][1] == 33339.0);
  ASSERT(long_result.at("values")[0][2] == 66673.0);
  server.stop();
}

// Rapid A/B/A publications exercise the state-index advertise/verify
// handshake. Each callback must bind one complete state for its whole block.
static void test_hot_swap_state_handoff()
{
  constexpr unsigned int buf = 16;
  tropical_runtime_t rt = tropical_runtime_new(buf);
  ASSERT(rt != nullptr);
  const std::string one = wrap_kernel(buf, 0,
    "  %p = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double 1.000000e+00, ptr %p, align 8\n");
  const std::string two = wrap_kernel(buf, 0,
    "  %p = getelementptr inbounds double, ptr %out, i64 %s\n"
    "  store double 2.000000e+00, ptr %p, align 8\n");
  const std::string manifest = constant_manifest(buf);
  ASSERT_OK(load_c(rt, one, manifest));

  std::atomic<bool> go{false};
  std::atomic<bool> done{false};
  std::jthread control([&] {
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    for (unsigned int i = 0; i < 80; ++i)
    {
      const std::string & next = (i & 1U) ? one : two;
      if (!load_c(rt, next, manifest))
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
  const std::string ir = wrap_kernel(buf, 1, PASS_X_BODY);
  ASSERT(rt.load(ir, "", "", tap_manifest(buf, "out", 1.0)));

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
  const auto constant = [](double value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.6e", value);
    return wrap_kernel(buf, 0,
      "  %p = getelementptr inbounds double, ptr %out, i64 %s\n"
      "  store double " + std::string(text) + ", ptr %p, align 8\n");
  };
  const std::string one = constant(1.0);
  const std::string two = constant(2.0);
  const std::string three = constant(3.0);
  const std::string manifest = constant_manifest(buf);
  ASSERT(rt.load(one, "", "", manifest));
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
    if (!rt.load(two, "", "", manifest)) loads_ok.store(false);
    if (!rt.load(three, "", "", manifest)) loads_ok.store(false);
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

struct MultichannelTestSource
{
  explicit MultichannelTestSource(unsigned int frames)
    : outputBuffer(frames, 0.0), compact(frames * 2, 0.0) {}

  void process()
  {
    for (unsigned int frame = 0; frame < outputBuffer.size(); ++frame)
    {
      compact[frame * 2] = 0.25 + frame;
      compact[frame * 2 + 1] = -0.5 - frame;
      outputBuffer[frame] = compact[frame * 2];
    }
  }
  unsigned int getBufferLength() const
  {
    return static_cast<unsigned int>(outputBuffer.size());
  }
  unsigned int getOutputChannelCount() const { return 2; }
  const double * getInterleavedOutputBuffer() const { return compact.data(); }
  void begin_fade_in(int = 2048) {}
  void begin_fade_out(int = 2048) {}
  bool is_fade_out_complete() const { return true; }

  std::vector<double> outputBuffer;
  std::vector<double> compact;
};

static void test_dac_multichannel_mapping()
{
  constexpr unsigned int frames = 4;
  MultichannelTestSource stereo(frames);
  TropicalDACImpl<MultichannelTestSource> stereo_dac(
    &stereo, 44100, 3);
  double three_channel_device[frames * 3]{};
  ASSERT(TropicalDACImpl<MultichannelTestSource>::fill_buffer(
           three_channel_device, nullptr, frames, 0.0, 0,
           &stereo_dac) == 0);
  for (unsigned int frame = 0; frame < frames; ++frame)
  {
    ASSERT_NEAR(three_channel_device[frame * 3], 0.25 + frame, 0.0);
    ASSERT_NEAR(three_channel_device[frame * 3 + 1], -0.5 - frame, 0.0);
    ASSERT_NEAR(three_channel_device[frame * 3 + 2], 0.0, 0.0);
  }

  // An already-open stream may become too narrow after a plan hot-swap. The
  // callback never downmixes implicitly; it fails the whole device image to
  // silence. open_stream rejects the same policy before startup.
  TropicalDACImpl<MultichannelTestSource> narrow_dac(
    &stereo, 44100, 1);
  double narrow_device[frames];
  std::fill_n(narrow_device, frames, 1.0);
  ASSERT(TropicalDACImpl<MultichannelTestSource>::fill_buffer(
           narrow_device, nullptr, frames, 0.0, 0, &narrow_dac) == 0);
  for (double sample : narrow_device)
    ASSERT_NEAR(sample, 0.0, 0.0);

  CaptureTestSource mono(frames);
  TropicalDACImpl<CaptureTestSource> mono_dac(&mono, 44100, 2);
  double upmixed_device[frames * 2]{};
  ASSERT(TropicalDACImpl<CaptureTestSource>::fill_buffer(
           upmixed_device, nullptr, frames, 0.0, 0, &mono_dac) == 0);
  for (unsigned int frame = 0; frame < frames; ++frame)
  {
    ASSERT_NEAR(upmixed_device[frame * 2], frame, 0.0);
    ASSERT_NEAR(upmixed_device[frame * 2 + 1], frame, 0.0);
  }
}

struct ReadyAwareStartSource : CaptureTestSource
{
  using CaptureTestSource::CaptureTestSource;

  void prepare_realtime_start(unsigned int process_cycles)
  {
    ++prepare_calls;
    requested_process_cycles = process_cycles;
  }

  void set_realtime_running(bool value) noexcept
  {
    realtime_running = value;
    ++realtime_signals;
  }

  unsigned int prepare_calls = 0;
  unsigned int requested_process_cycles = 0;
  bool realtime_running = false;
  unsigned int realtime_signals = 0;
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

  ready_dac.set_source_realtime_running(true);
  ASSERT(ready_aware.realtime_running);
  ready_dac.set_source_realtime_running(false);
  ASSERT(!ready_aware.realtime_running);
  ASSERT(ready_aware.realtime_signals == 2);

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
  printf("test_module_process (the port: typed kernel load)\n");

  run_test("constant kernel via load",      test_ir_constant);
  run_test("compact interleaved native output", test_interleaved_native_output);
  run_test("kernel manifest boundary",      test_manifest_boundary);
  run_test("closed-form index ramp",        test_ir_index_ramp);
  run_test("clock request boundary handoff", test_clock_boundary_handoff);
  run_test("clock odd-sequence barrier", test_clock_request_barrier);
  run_test("param dispatch exact-sample replay",
           test_param_dispatch_exact_sample_replay);
  run_test("param dispatch effective-boundary races",
           test_param_dispatch_effective_boundary_races);
  run_test("observation snapshot never blocks control publication",
           test_observation_snapshot_does_not_block_control);
  run_test("named observation pins one strided hot-swap program",
           test_named_observation_pins_hot_swap_program);
  run_test("separate observation artifact projects named controls",
           test_separate_observation_artifact_projects_controls);
  run_test("publication generation tokens and atomic dual failure",
           test_publication_generation_tokens_and_atomic_failure);
  run_test("playback-anchored observation socket frame",
           test_playback_anchored_observation_socket_frame);
  run_test("hot-swap state handoff", test_hot_swap_state_handoff);
  run_test("owned generation survives two publications", test_generation_ownership_barrier);
  run_test("owned state blocks ABA reuse", test_state_ownership_barrier);
  run_test("device-boundary clamp",         test_device_bound_clamp);
  run_test("fixed DAC histogram/ABI contract", test_dac_histogram_contract);
  run_test("DAC startup honors source readiness barrier",
           test_dac_start_source_preparation);
  run_test("DAC multichannel mapping and narrow-device silence",
           test_dac_multichannel_mapping);
  run_test("serial output-capture state machine", test_capture_serial_state_machine);

  printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
