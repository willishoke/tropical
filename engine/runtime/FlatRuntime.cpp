#include "runtime/FlatRuntime.hpp"
#include "runtime/NumericProgramParser.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace tropical_runtime
{

bool FlatRuntime::load_plan(const std::string & plan_json)
{
  using json = nlohmann::json;

  const json plan = json::parse(plan_json);

  const std::string schema = plan.value("schema", std::string{});

  // ── tropical_plan_4: compiled flat instruction stream with typed operands ──
  if (schema == "tropical_plan_4")
  {
    const auto parsed = tropical_plan4::parse_plan4(plan);

    auto kernel_result = tropical_jit::OrcJitEngine::instance().compile_flat_program(parsed.program);
    if (!kernel_result)
    {
      std::string err;
      llvm::handleAllErrors(kernel_result.takeError(),
        [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
      throw std::runtime_error("FlatRuntime: JIT compilation failed: " + err);
    }

    KernelState new_state;
    new_state.kernel         = *kernel_result;
    new_state.sample_rate    = parsed.sample_rate;
    new_state.output_count   = static_cast<uint32_t>(parsed.program.output_targets.size());
    new_state.register_names = parsed.register_names;
    new_state.array_names    = parsed.array_slot_names;

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
    // Scalar temps
    new_state.temps.assign(parsed.program.register_count, 0);

    // Array slot storage
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

    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t active   = active_state_.load(std::memory_order_acquire);
    const uint32_t inactive = 1U - active;
    wait_for_state_available(inactive);

    const auto & old_state = states_[active];

    // Apply state transfer directly into new_state BEFORE the atomic swap.
    // This eliminates the race window that existed when PendingTransfer was used:
    // the audio thread would run one buffer with zeroed registers between
    // active_state_.store() and pending_transfer_.exchange(), causing a pop.
    // Reading old_state concurrently with the audio thread is a benign data
    // race — at most one sample stale, completely inaudible.
    new_state.sample_index = old_state.sample_index;
    if (old_state.kernel != nullptr)
    {
      const auto mapping       = compute_register_mapping(old_state, new_state);
      const auto array_mapping = compute_array_mapping(old_state, new_state);
      for (const auto & [si, di] : mapping)
        if (si < old_state.registers.size() && di < new_state.registers.size())
          new_state.registers[di] = old_state.registers[si];
      for (const auto & [si, di] : array_mapping)
        if (si < old_state.array_storage.size() && di < new_state.array_storage.size() &&
            old_state.array_storage[si].size() == new_state.array_storage[di].size())
          new_state.array_storage[di] = old_state.array_storage[si];
    }

    states_[inactive] = std::move(new_state);
    active_state_.store(inactive, std::memory_order_release);
    return true;
  }

  throw std::runtime_error("FlatRuntime: unsupported schema '" + schema + "'");
}

} // namespace tropical_runtime
