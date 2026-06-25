#include "runtime/FlatRuntime.hpp"
#include "runtime/NumericProgramParser.hpp"

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
  new_state.register_names = parsed.register_names;
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

  // Scalar state registers (type-aware initialization)
  new_state.registers.resize(parsed.state_init.size(), 0);
  for (std::size_t i = 0; i < parsed.state_init.size(); ++i)
  {
    const auto ty = (i < parsed.register_types.size())
      ? parsed.register_types[i]
      : tropical_jit::JitScalarType::Float;
    if (ty == tropical_jit::JitScalarType::Int || ty == tropical_jit::JitScalarType::Bool)
      new_state.registers[i] = static_cast<int64_t>(parsed.state_init[i]);
    else
      new_state.registers[i] = std::bit_cast<int64_t>(parsed.state_init[i]);
  }
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
  // Hot-swap state transfer runs whenever the old state has a populated
  // kernel of EITHER mode. Mode may change between loads (e.g. control
  // surface flips a session between fused and microkernel for A/B
  // testing); state transfer by name is mode-agnostic.
  const bool old_has_kernel =
    (old_state.mode == tropical_jit::CompilationMode::Fused          && old_state.kernel != nullptr) ||
    (old_state.mode == tropical_jit::CompilationMode::Microkernel    && old_state.microkernels.postamble_mix != nullptr) ||
    (old_state.mode == tropical_jit::CompilationMode::MicrokernelDeep && old_state.microkernels.postamble_mix != nullptr);
  if (old_has_kernel)
  {
    const auto mapping       = compute_register_mapping(old_state, new_state);
    const auto array_mapping = compute_array_mapping(old_state, new_state);
    const auto slot_mapping  = compute_slot_mapping(old_state, new_state);
    for (const auto & [si, di] : mapping)
      if (si < old_state.registers.size() && di < new_state.registers.size())
        new_state.registers[di] = old_state.registers[si];
    for (const auto & [si, di] : array_mapping)
      if (si < old_state.array_storage.size() && di < new_state.array_storage.size() &&
          old_state.array_storage[si].size() == new_state.array_storage[di].size())
        new_state.array_storage[di] = old_state.array_storage[si];
    for (const auto & [si, di] : slot_mapping)
      if (si < old_state.slots.size() && di < new_state.slots.size())
        new_state.slots[di] = old_state.slots[si];
  }

  states_[inactive] = std::move(new_state);
  active_state_.store(inactive, std::memory_order_release);
  recompile_version_.fetch_add(1, std::memory_order_release);
  return true;
}

bool FlatRuntime::load_ir(const std::string & ir_text, const std::string & manifest_json)
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

  auto kernel_result = tropical_jit::OrcJitEngine::instance().compile_ir_text(ir_text);
  if (!kernel_result)
  {
    std::string err;
    llvm::handleAllErrors(kernel_result.takeError(),
      [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
    throw std::runtime_error("FlatRuntime: IR JIT compilation failed: " + err);
  }
  new_state.kernel = *kernel_result;

  return publish_state(std::move(new_state));
}

} // namespace tropical_runtime
