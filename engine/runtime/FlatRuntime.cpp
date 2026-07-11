#include "runtime/FlatRuntime.hpp"
#include "runtime/NumericProgramParser.hpp"

#ifdef TROPICAL_METAL
#include "metal/MetalKernel.hpp"
#endif

#include <algorithm>
#include <bit>
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
  new_state.output_count   = static_cast<uint32_t>(parsed.program.output_targets.size());
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
  // (run_coeff repoints the column entries at its write generation per run).
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

// By-name hot-swap state transfer + atomic double-buffer flip. Assumes
// new_state already carries a populated kernel handle.
bool FlatRuntime::publish_state(KernelState && new_state)
{
  std::lock_guard<std::mutex> lock(build_mutex_);
  const uint32_t active   = active_state_.load(std::memory_order_acquire);
  const uint32_t inactive = 1U - active;
  wait_for_state_available(inactive);

  const auto & old_state = states_[active];

  new_state.sample_index = old_state.sample_index;
  // CF-only: no by-name state transfer on hot-swap. Registers/arrays/slots
  // zero-init from the fresh kernel; only the sample index carries over.

  states_[inactive] = std::move(new_state);
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

  tropical_plan5::ParsedPlan5 parsed;
  if (schema == "tropical_plan_5")
    parsed = tropical_plan5::parse_plan5(manifest);
  else if (schema == "tropical_plan_4")
    parsed = tropical_plan5::parse_plan4(manifest);
  else
    throw std::runtime_error("FlatRuntime: load_ir unsupported manifest schema '" + schema + "'");

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
  run_coeff(new_state);

  return publish_state(std::move(new_state));
}

} // namespace tropical_runtime
