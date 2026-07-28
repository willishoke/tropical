#include "runtime/FlatRuntime.hpp"
#include "runtime/NumericProgramParser.hpp"

#ifdef TROPICAL_METAL
#include "metal/MetalKernel.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace tropical_runtime
{

// Map a parsed plan's *metadata* (the manifest) into a fresh KernelState —
// everything except the kernel handle, which load_ir fills (via
// compile_ir_text) between this and publish_state. The plan's instruction
// graph is metadata-only now; codegen is Lean's (EmitLlvm).
KernelState FlatRuntime::build_kernel_state(const tropical_plan5::ParsedPlan5 & parsed)
{
  KernelState new_state;
  new_state.mode           = parsed.compilation_mode;
  new_state.sample_rate    = parsed.sample_rate;
  new_state.array_names    = parsed.array_slot_names;

  // Slot array
  new_state.slots.assign(parsed.slot_count, 0.0);
  new_state.slot_names = parsed.slot_names;
  for (std::size_t i = 0;
       i < parsed.slot_defaults.size() && i < new_state.slots.size();
       ++i)
  {
    new_state.slots[i] = parsed.slot_defaults[i];
  }
  for (auto & generation : new_state.slot_generations)
    generation = new_state.slots;
  new_state.audio_slots = new_state.slots;

  // Host-contract dispatch table — swaps with the plan it describes, like
  // slot_names. Field-copied because the runtime keeps its own struct (the
  // header stays free of the parser header).
  new_state.param_disciplines.reserve(parsed.param_disciplines.size());
  for (const auto & d : parsed.param_disciplines)
    new_state.param_disciplines.push_back(
      { d.name, d.discipline, d.glide_dur_sec, d.companions });

  // CF-only: there are no per-sample state registers. The kernel's
  // `%registers` argument survives in the calling convention but is never
  // read or written, so the backing buffer is empty. The temp pool
  // (`register_count`) is the SSA scratch the kernel actually uses.
  new_state.registers.clear();
  new_state.temps.assign(parsed.program.register_count, 0);
  new_state.coeff_registers = new_state.registers;
  new_state.coeff_temps.assign(parsed.program.register_count, 0);

  const auto & sizes = parsed.program.array_slot_sizes;
  new_state.array_storage.resize(sizes.size());
  for (std::size_t i = 0; i < sizes.size(); ++i)
    new_state.array_storage[i].assign(static_cast<std::size_t>(sizes[i]), 0);
  new_state.array_ptrs.resize(new_state.array_storage.size());
  new_state.array_sizes.resize(new_state.array_storage.size());
  for (std::size_t i = 0; i < new_state.array_storage.size(); ++i)
  {
    new_state.array_ptrs[i]  = new_state.array_storage[i].data();
    new_state.array_sizes[i] = new_state.array_storage[i].size();
  }

  // BANKS-AS-DATA: generation-buffer the coefficient columns the plan
  // advertises (see the KernelState comment for the protocol). Three
  // generations per column, sized like the slot's base storage; the
  // coefficient kernel's pointer view starts as a copy of the audio view
  // (publish_control_snapshot repoints column entries to its target generation).
  for (uint32_t s : parsed.coeff_array_slots)
    if (s < new_state.array_storage.size())
      new_state.coeff_array_slots.push_back(s);
  if (!new_state.coeff_array_slots.empty())
  {
    for (auto & gen : new_state.coeff_generations)
    {
      gen.resize(new_state.coeff_array_slots.size());
      for (std::size_t j = 0; j < new_state.coeff_array_slots.size(); ++j)
        gen[j].assign(
          new_state.array_storage[new_state.coeff_array_slots[j]].size(), 0);
    }
    new_state.coeff_array_ptrs = new_state.array_ptrs;
    // Metal upload staging: one f32 per column element, packed in
    // coeff_array_slots order (EmitMsl's compile-time offset layout).
    // Sized here so a hot-swap re-derives it with the plan; harmless
    // (a few bytes) on JIT-only loads.
    std::size_t column_floats = 0;
    for (uint32_t s : new_state.coeff_array_slots)
      column_floats += new_state.array_storage[s].size();
    new_state.metal_column_staging.assign(column_floats, 0.0f);
  }

  return new_state;
}

// Atomic double-buffer flip (CF-only: no by-name state transfer — see below).
// Assumes new_state already carries a populated kernel handle.
bool FlatRuntime::publish_state(KernelState && new_state)
{
  std::lock_guard<std::mutex> lock(build_mutex_);
  const uint32_t active   = active_state_.load(std::memory_order_acquire);
  const uint32_t inactive = 1U - active;
  for (;;)
  {
    StorageOwner expected = StorageOwner::Free;
    if (state_owners_[inactive].compare_exchange_strong(
          expected, StorageOwner::Writer,
          std::memory_order_acquire, std::memory_order_relaxed))
      break;
    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire))
      seam->writer_waiting_for_state.store(true, std::memory_order_release);
    std::this_thread::yield();
  }

  // CF-only: no by-name state transfer on hot-swap. Registers/arrays/slots
  // zero-init from the fresh kernel. The audio-owned clock is runtime-global,
  // so a state flip cannot repeat, skip, or race its sample position.

  states_[inactive] = std::move(new_state);
  state_owners_[inactive].store(
    StorageOwner::Free, std::memory_order_release);
  active_state_.store(inactive, std::memory_order_release);
  recompile_version_.fetch_add(1, std::memory_order_release);
  return true;
}

bool FlatRuntime::load_ir(const std::string & ir_text, const std::string & manifest_json)
{
  return load_ir_staged(ir_text, {}, {}, manifest_json);
}

bool FlatRuntime::load_ir_msl(const std::string & ir_text,
                              const std::string & msl_source,
                              const std::string & manifest_json)
{
  return load_ir_staged(ir_text, msl_source, {}, manifest_json);
}

bool FlatRuntime::load_ir_staged(const std::string & ir_text,
                                 const std::string & msl_source,
                                 const std::string & coeff_ir,
                                 const std::string & manifest_json)
{
  using json = nlohmann::json;

  const json manifest = json::parse(manifest_json);
  const std::string schema = manifest.value("schema", std::string{});

  if (schema != "tropical_plan_5")
    throw std::runtime_error(
      "FlatRuntime: load_ir unsupported manifest schema '" + schema +
      "'; expected 'tropical_plan_5'");
  tropical_plan5::ParsedPlan5 parsed = tropical_plan5::parse_plan5(manifest);

  KernelState new_state = build_kernel_state(parsed);
  // The IR path is always fused — Lean emits a single fused kernel.
  new_state.mode = tropical_jit::CompilationMode::Fused;

  // The JIT kernel loads ALWAYS — it keeps serving render_window (the
  // scope) and the correctness reference even when audio runs on Metal.
  auto kernel_result = tropical_jit::OrcJitEngine::instance().compile_ir_text(ir_text);
  if (!kernel_result)
  {
    std::string err;
    llvm::handleAllErrors(kernel_result.takeError(),
      [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
    throw std::runtime_error("FlatRuntime: IR JIT compilation failed: " + err);
  }
  new_state.kernel = *kernel_result;

  if (!coeff_ir.empty())
  {
    // O0: the coefficient kernel runs once per control write — codegen
    // quality is irrelevant, and O2 on its thousands-of-flops block is
    // exactly the compile wall the stage-0 split removes.
    auto coeff_result = tropical_jit::OrcJitEngine::instance().compile_ir_text(coeff_ir, true);
    if (!coeff_result)
    {
      std::string err;
      llvm::handleAllErrors(coeff_result.takeError(),
        [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
      throw std::runtime_error("FlatRuntime: coefficient IR JIT compilation failed: " + err);
    }
    new_state.coeff_kernel = *coeff_result;
  }

  if (!msl_source.empty())
  {
#ifdef TROPICAL_METAL
    std::string err;
    new_state.metal = tropical_metal::create(
      msl_source, buffer_length_,
      static_cast<uint32_t>(new_state.slots.size()),
      static_cast<uint32_t>(new_state.metal_column_staging.size()), err);
    if (!new_state.metal)
      throw std::runtime_error("FlatRuntime: " + err);
#else
    throw std::runtime_error(
      "FlatRuntime: MSL source supplied but the engine was built without TROPICAL_METAL");
#endif
  }

  // Materialize coefficient slots before the state becomes visible to the
  // audio thread or render_window — initial knob values exist, so a first
  // buffer reading zero-initialized coefficient slots would be a bug.
  publish_control_snapshot(new_state);

  return publish_state(std::move(new_state));
}

ParamDispatchResult FlatRuntime::dispatch_param_sync(
  const std::string & name, double value)
{
  if (!std::isfinite(value))
    return {
      false, "set_param: value must be finite", {},
      published_sample_index_.load(std::memory_order_acquire)
    };

  std::lock_guard<std::mutex> lock(build_mutex_);
  const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
  KernelState & state = states_[state_idx];
  const uint64_t sample_index =
    published_sample_index_.load(std::memory_order_acquire);
  return dispatch_param_sync_locked(
    state, state_idx, name, value, sample_index);
}

ParamDispatchResult FlatRuntime::dispatch_param_sync_at_sample_index(
  const std::string & name, double value, uint64_t sample_index)
{
  if (!std::isfinite(value))
    return {
      false, "set_param: value must be finite", {}, sample_index
    };

  std::lock_guard<std::mutex> lock(build_mutex_);
  const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
  KernelState & state = states_[state_idx];
  return dispatch_param_sync_locked(
    state, state_idx, name, value, sample_index);
}

ParamDispatchResult FlatRuntime::dispatch_param_sync_locked(
  KernelState & state, uint32_t state_idx, const std::string & name,
  double value, uint64_t sample_index)
{
  auto slot_of = [&state](const std::string & slot_name) -> uint32_t {
    for (uint32_t i = 0; i < state.slot_names.size(); ++i)
      if (state.slot_names[i] == slot_name) return i;
    return UINT32_MAX;
  };
  auto read = [&state](uint32_t i) {
    return i < state.slots.size() ? state.slots[i] : 0.0;
  };
  auto write = [&state](uint32_t i, double v) {
    if (i < state.slots.size()) state.slots[i] = v;
  };
  const ParamDiscipline * pd = state.find_discipline(name);
  const std::string discipline = pd ? pd->discipline : std::string{"raw"};
  auto fail = [&](std::string error) {
    return ParamDispatchResult{
      false, std::move(error), discipline, sample_index
    };
  };
  auto commit = [&]() {
    publish_control_snapshot(state, state_idx);
    return ParamDispatchResult{
      true, {}, discipline, sample_index
    };
  };
  const double now = static_cast<double>(sample_index);

  if (discipline == "glide")
  {
    const uint32_t v0i = slot_of("param:" + name + "#v0");
    const uint32_t v1i = slot_of("param:" + name + "#v1");
    const uint32_t t0i = slot_of("param:" + name + "#t0");
    if (v0i == UINT32_MAX)
      return fail("set_param: no glide slots for '" + name + "'");
    const double dur_sec = pd->glide_dur_sec > 0.0
      ? pd->glide_dur_sec : 0.02;
    const double dur = state.sample_rate * dur_sec;
    const double v0 = read(v0i);
    const double v1 = read(v1i);
    const double t0 = read(t0i);
    const double r = (now - t0) / dur;
    const double s = r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
    write(v0i, v0 + (v1 - v0) * (s * s * (3.0 - 2.0 * s)));
    write(v1i, value);
    write(t0i, now);
    return commit();
  }

  if (discipline == "anchor")
  {
    const uint32_t fi = slot_of("param:" + name);
    if (fi == UINT32_MAX)
      return fail("set_param: unknown param '" + name + "'");
    const uint32_t pi = slot_of("param:" + name + "#phase");
    if (pi == UINT32_MAX)
    {
      write(fi, value);
      return commit();
    }
    const double inc0 =
      std::floor(read(fi) * 4294967296.0 / state.sample_rate);
    const double inc1 =
      std::floor(value * 4294967296.0 / state.sample_rate);
    const double dcyc = ((inc0 - inc1) * now) / 4294967296.0;
    const double phase = read(pi) + dcyc;
    write(pi, phase - std::floor(phase));
    write(fi, value);
    return commit();
  }

  if (discipline == "velocity")
  {
    const uint32_t vi = slot_of("param:" + name);
    if (vi == UINT32_MAX)
      return fail("set_param: unknown param '" + name + "'");
    std::string tau;
    if (!pd->companions.empty()) tau = pd->companions.front();
    else
    {
      tau = name;
      const std::size_t pos = tau.find("velocity");
      if (pos != std::string::npos) tau.replace(pos, 8, "tau_base");
    }
    const uint32_t ti = slot_of("param:" + tau);
    if (ti == UINT32_MAX)
      return fail("set_param: no origin slot '" + tau + "'");
    write(ti, read(ti) + (read(vi) - value) * now / state.sample_rate);
    write(vi, value);
    return commit();
  }

  const uint32_t bi = slot_of("param:" + name);
  if (bi == UINT32_MAX)
    return fail("set_param: unknown param '" + name + "'");
  write(bi, value);
  return commit();
}

} // namespace tropical_runtime
