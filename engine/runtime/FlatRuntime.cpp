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

  // Scalar state registers: sized but zero-initialized. CF-only removed the
  // by-name hot-swap transfer; any surviving reg (e.g. Delay.md) zero-inits
  // on each fresh kernel.
  new_state.registers.assign(parsed.state_init.size(), 0);
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
  // CF-only: no by-name state transfer on hot-swap. Registers/arrays/slots
  // zero-init from the fresh kernel; only the sample index carries over.

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
