#include "runtime/FlatRuntime.hpp"
#include "runtime/NumericProgramParser.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace egress_runtime
{

bool FlatRuntime::load_plan(const std::string & plan_json)
{
  using json = nlohmann::json;

  const json plan = json::parse(plan_json);

  const std::string schema = plan.value("schema", std::string{});

  // ── egress_plan_3: compiled flat instruction stream ──
  if (schema == "egress_plan_3")
  {
    const auto parsed = egress_plan3::parse_plan3(plan);

    auto kernel_result = egress_jit::OrcJitEngine::instance().compile_flat_program(parsed.program);
    if (!kernel_result)
    {
      std::string err;
      llvm::handleAllErrors(kernel_result.takeError(),
        [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
      throw std::runtime_error("FlatRuntime: JIT compilation failed: " + err);
    }

    KernelState new_state;
    new_state.kernel       = *kernel_result;
    new_state.sample_rate  = parsed.sample_rate;
    new_state.output_count = static_cast<uint32_t>(parsed.program.output_targets.size());
    new_state.mix_indices  = parsed.mix_indices;
    new_state.register_names = parsed.register_names;

    // Scalar state registers
    new_state.registers.resize(parsed.state_init.size(), 0);
    for (std::size_t i = 0; i < parsed.state_init.size(); ++i)
      new_state.registers[i] = std::bit_cast<int64_t>(parsed.state_init[i]);
    new_state.register_scalar_mask.assign(parsed.state_init.size(), true);

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

    new_state.output_temp_indices.assign(
      parsed.program.output_targets.begin(), parsed.program.output_targets.end());
    new_state.register_temp_indices.assign(
      parsed.program.register_targets.begin(), parsed.program.register_targets.end());

    std::lock_guard<std::mutex> lock(build_mutex_);
    const uint32_t active   = active_state_.load(std::memory_order_acquire);
    const uint32_t inactive = 1U - active;
    wait_for_state_available(inactive);

    const auto & old_state = states_[active];
    auto mapping = compute_register_mapping(old_state, new_state);
    new_state.sample_index = old_state.sample_index;

    states_[inactive] = std::move(new_state);
    active_state_.store(inactive, std::memory_order_release);

    if (!mapping.empty() && old_state.kernel != nullptr)
    {
      auto * pt = new PendingTransfer{std::move(mapping), active};
      delete pending_transfer_.exchange(pt, std::memory_order_acq_rel);
    }
    return true;
  }

  // ── egress_plan_2: expression tree path ──
  if (schema != "egress_plan_2")
    throw std::runtime_error("FlatRuntime: unsupported schema '" + schema + "'");

  // ── Parse config ──
  const double sample_rate = plan.value("config", json::object()).value("sample_rate", 44100.0);

  // ── Parse expression trees ──
  std::vector<ExprSpecPtr> output_exprs;
  if (plan.contains("output_exprs"))
  {
    for (const auto & node : plan["output_exprs"])
      output_exprs.push_back(egress_plan::parse_expr(node));
  }

  std::vector<ExprSpecPtr> register_exprs;
  if (plan.contains("register_exprs"))
  {
    for (const auto & node : plan["register_exprs"])
      register_exprs.push_back(egress_plan::parse_expr(node));
  }

  // ── Parse initial register values ──
  std::vector<Value> initial_registers;
  if (plan.contains("state_init"))
  {
    for (const auto & v : plan["state_init"])
    {
      if (v.is_number_float() || v.is_number_integer())
        initial_registers.push_back(egress_expr::float_value(v.get<double>()));
      else if (v.is_boolean())
        initial_registers.push_back(egress_expr::bool_value(v.get<bool>()));
      else if (v.is_array())
      {
        std::vector<Value> items;
        items.reserve(v.size());
        for (const auto & elem : v)
          items.push_back(egress_expr::float_value(elem.get<double>()));
        initial_registers.push_back(egress_expr::array_value(std::move(items)));
      }
      else
        initial_registers.push_back(egress_expr::float_value(0.0));
    }
  }

  // ── Parse register names ──
  std::vector<std::string> register_names;
  if (plan.contains("register_names"))
  {
    for (const auto & n : plan["register_names"])
      register_names.push_back(n.get<std::string>());
  }

  // ── Parse output mix indices ──
  std::vector<uint32_t> mix_indices;
  if (plan.contains("outputs"))
  {
    for (const auto & o : plan["outputs"])
      mix_indices.push_back(o.get<uint32_t>());
  }

  // ── Walk expressions for SmoothedParam/TriggerParam ──
  std::unordered_map<egress_expr::ControlParam *, uint32_t> param_anon_reg_map;
  std::unordered_set<egress_expr::ControlParam *> trigger_params;
  uint32_t next_anon_idx = 0;
  const uint32_t user_register_count = static_cast<uint32_t>(initial_registers.size());

  for (const auto & e : output_exprs)
    walk_expr_for_params(e, param_anon_reg_map, next_anon_idx);
  for (const auto & e : register_exprs)
    walk_expr_for_params(e, param_anon_reg_map, next_anon_idx);

  for (const auto & e : output_exprs)
    collect_trigger_params(e, trigger_params);
  for (const auto & e : register_exprs)
    collect_trigger_params(e, trigger_params);

  // Append anonymous registers for smoothed params
  if (!param_anon_reg_map.empty())
  {
    std::vector<std::pair<uint32_t, egress_expr::ControlParam *>> sorted_anon;
    sorted_anon.reserve(param_anon_reg_map.size());
    for (const auto & kv : param_anon_reg_map)
      sorted_anon.push_back({kv.second, kv.first});
    std::sort(sorted_anon.begin(), sorted_anon.end());
    for (const auto & [idx, p] : sorted_anon)
    {
      const double init = p->value.load(std::memory_order_relaxed);
      initial_registers.push_back(egress_expr::float_value(init));
      register_names.push_back("");  // anonymous
    }
  }

  // ── Compile ExprSpec trees → CompiledProgram ──
  const auto compiled = compile_expr_program(
    output_exprs, register_exprs, param_anon_reg_map, user_register_count);

  // ── Build NumericProgram from CompiledProgram ──
  // Use Module's static build_numeric_program_impl (made public in Phase 1)
  egress_runtime::NumericJitState jit_state;
  egress_jit::NumericProgram numeric_program;

  // The flat program has zero inputs — everything is embedded in expressions
  const std::vector<Value> empty_inputs;

  if (!egress_runtime::build_numeric_program(
        compiled, initial_registers, sample_rate,
        empty_inputs, numeric_program, jit_state, nullptr))
  {
    throw std::runtime_error(
      "FlatRuntime: failed to build numeric program ("
      "outputs=" + std::to_string(output_exprs.size()) +
      " regs=" + std::to_string(register_exprs.size()) +
      " instrs=" + std::to_string(compiled.instructions.size()) +
      " reg_count=" + std::to_string(compiled.register_count) +
      " init_regs=" + std::to_string(initial_registers.size()) + ")");
  }

  // ── JIT compile to native kernel ──
  auto kernel_result = egress_jit::OrcJitEngine::instance().compile_numeric_program(numeric_program);
  if (!kernel_result)
  {
    std::string err;
    llvm::handleAllErrors(kernel_result.takeError(),
      [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
    throw std::runtime_error("FlatRuntime: JIT compilation failed: " + err);
  }

  // ── Build KernelState ──
  KernelState new_state;
  new_state.kernel = *kernel_result;
  new_state.sample_rate = sample_rate;
  new_state.output_count = static_cast<uint32_t>(output_exprs.size());
  new_state.mix_indices = std::move(mix_indices);
  new_state.register_names = std::move(register_names);

  // Initialize registers from initial values (bitcast float→int64)
  // Array registers store their data in array_storage, not in the scalar registers array.
  new_state.registers.resize(initial_registers.size(), 0);
  for (std::size_t i = 0; i < initial_registers.size(); ++i)
  {
    if (initial_registers[i].type != egress_expr::ValueType::Array &&
        initial_registers[i].type != egress_expr::ValueType::Matrix)
    {
      new_state.registers[i] = std::bit_cast<int64_t>(egress_expr::to_float64(initial_registers[i]));
    }
  }

  // Temps
  new_state.temps.assign(numeric_program.register_count, 0);

  // Array storage from JIT state
  new_state.array_storage = std::move(jit_state.array_storage);
  new_state.array_ptrs.resize(new_state.array_storage.size());
  new_state.array_sizes.resize(new_state.array_storage.size());
  for (std::size_t i = 0; i < new_state.array_storage.size(); ++i)
  {
    new_state.array_ptrs[i] = new_state.array_storage[i].data();
    new_state.array_sizes[i] = new_state.array_storage[i].size();
  }

  // Param pointers (SmoothedParam addresses for atomic reads in kernel)
  new_state.param_ptrs.clear();
  {
    // Build sorted param list matching the order used by build_numeric_program_impl
    std::vector<std::pair<uint32_t, egress_expr::ControlParam *>> sorted_params;
    sorted_params.reserve(param_anon_reg_map.size());
    for (const auto & kv : param_anon_reg_map)
      sorted_params.push_back({kv.second, kv.first});
    std::sort(sorted_params.begin(), sorted_params.end());
    for (const auto & [idx, p] : sorted_params)
      new_state.param_ptrs.push_back(reinterpret_cast<uint64_t>(p));
  }

  // Output temp indices: output i is stored in compiled.output_targets[i]
  new_state.output_temp_indices.resize(compiled.output_targets.size());
  for (std::size_t i = 0; i < compiled.output_targets.size(); ++i)
  {
    new_state.output_temp_indices[i] = compiled.output_targets[i];
  }

  // Register writeback targets: register i gets new value from temps[register_targets[i]]
  new_state.register_temp_indices.resize(compiled.register_targets.size());
  for (std::size_t i = 0; i < compiled.register_targets.size(); ++i)
  {
    new_state.register_temp_indices[i] = compiled.register_targets[i];
  }
  new_state.register_scalar_mask = std::move(jit_state.register_scalar_mask);

  // Trigger params
  new_state.trigger_params.assign(trigger_params.begin(), trigger_params.end());

  // ── Publish atomically via double-buffer ──
  std::lock_guard<std::mutex> lock(build_mutex_);

  const uint32_t active = active_state_.load(std::memory_order_acquire);
  const uint32_t inactive = 1U - active;

  wait_for_state_available(inactive);

  // Compute register mapping for state transfer
  const auto & old_state = states_[active];
  auto mapping = compute_register_mapping(old_state, new_state);

  // Preserve sample_index continuity
  new_state.sample_index = old_state.sample_index;

  // Write new state to inactive slot
  states_[inactive] = std::move(new_state);

  // Swap active state
  active_state_.store(inactive, std::memory_order_release);

  // Schedule register transfer (audio thread will apply it)
  if (!mapping.empty() && old_state.kernel != nullptr)
  {
    auto * pt = new PendingTransfer{std::move(mapping), active};
    delete pending_transfer_.exchange(pt, std::memory_order_acq_rel);
  }

  return true;
}

} // namespace egress_runtime
