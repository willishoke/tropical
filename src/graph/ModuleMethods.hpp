#pragma once

void Module::process(const std::vector<bool> * output_materialize_mask)
{
  resize_array_registers_to_inputs();
  const bool use_composite_programs = has_nested_modules_ || has_delay_states_;
#ifdef EGRESS_LLVM_ORC_JIT
  if (!has_dynamic_registers_)
  {
    ensure_numeric_jit_current();
  }

  if (!use_composite_programs && jit_kernel_)
  {
    if (!sync_numeric_inputs_from_values())
    {
      jit_status_ = "numeric input sync failed";
      jit_kernel_ = nullptr;
    }
    else
    {
      jit_kernel_(
        numeric_inputs_.data(),
        numeric_registers_.data(),
        numeric_array_ptrs_.data(),
        numeric_array_sizes_.data(),
        numeric_temps_.data(),
        sample_rate_,
        sample_index_);

#ifdef EGRESS_PROFILE
      const auto materialize_start = std::chrono::steady_clock::now();
      uint64_t materialized_scalar_outputs = 0;
      uint64_t materialized_array_outputs = 0;
      uint64_t materialized_matrix_outputs = 0;
#endif
      for (unsigned int output_id = 0; output_id < program_.output_targets.size(); ++output_id)
      {
        if (output_materialize_mask != nullptr &&
            output_id < output_materialize_mask->size() &&
            !(*output_materialize_mask)[output_id] &&
            output_id < numeric_output_info_.size())
        {
          const NumericValueKind output_kind = static_cast<NumericValueKind>(numeric_output_info_[output_id].kind);
          if (output_kind == NumericValueKind::Array || output_kind == NumericValueKind::Matrix)
          {
            continue;
          }
        }
#ifdef EGRESS_PROFILE
        switch (static_cast<NumericValueKind>(numeric_output_info_[output_id].kind))
        {
          case NumericValueKind::Scalar:
            ++materialized_scalar_outputs;
            break;
          case NumericValueKind::Array:
            ++materialized_array_outputs;
            break;
          case NumericValueKind::Matrix:
            ++materialized_matrix_outputs;
            break;
        }
#endif
        assign_numeric_value_to(
          outputs[output_id],
          numeric_output_info_[output_id],
          program_.output_targets[output_id],
          numeric_temps_,
          numeric_array_storage_);
      }
#ifdef EGRESS_PROFILE
      record_numeric_output_materialize_profile(
        static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - materialize_start)
            .count()),
        materialized_scalar_outputs,
        materialized_array_outputs,
        materialized_matrix_outputs);
#endif

      for (unsigned int register_id = 0; register_id < program_.register_targets.size(); ++register_id)
      {
        const int32_t target = program_.register_targets[register_id];
        if (register_scalar_mask_[register_id])
        {
          if (target >= 0)
          {
            numeric_next_registers_[register_id] = numeric_temps_[static_cast<std::size_t>(target)];
          }
          else
          {
            numeric_next_registers_[register_id] = numeric_registers_[register_id];
          }
        }
        else
        {
          numeric_next_registers_[register_id] = numeric_registers_[register_id];
        }
      }

      numeric_registers_.swap(numeric_next_registers_);

      for (unsigned int register_id = 0; register_id < array_register_targets_.size(); ++register_id)
      {
        if (register_scalar_mask_[register_id])
        {
          continue;
        }
        const int32_t src_slot = array_register_targets_[register_id];
        if (src_slot < 0)
        {
          continue;
        }
        const uint32_t dst_slot = register_array_slot_[register_id];
        if (dst_slot >= numeric_array_storage_.size() ||
            static_cast<std::size_t>(src_slot) >= numeric_array_storage_.size())
        {
          continue;
        }
        if (dst_slot == static_cast<uint32_t>(src_slot))
        {
          continue;
        }
        auto & dst = numeric_array_storage_[dst_slot];
        auto & src = numeric_array_storage_[static_cast<std::size_t>(src_slot)];
        if (dst.size() != src.size())
        {
          continue;
        }
        if (register_id < array_register_can_swap_.size() && array_register_can_swap_[register_id])
        {
          dst.swap(src);
          if (dst_slot < numeric_array_ptrs_.size())
          {
            numeric_array_ptrs_[dst_slot] = dst.empty() ? nullptr : dst.data();
            numeric_array_sizes_[dst_slot] = static_cast<uint64_t>(dst.size());
          }
          if (static_cast<std::size_t>(src_slot) < numeric_array_ptrs_.size())
          {
            numeric_array_ptrs_[static_cast<std::size_t>(src_slot)] = src.empty() ? nullptr : src.data();
            numeric_array_sizes_[static_cast<std::size_t>(src_slot)] = static_cast<uint64_t>(src.size());
          }
          continue;
        }
        std::copy(src.begin(), src.end(), dst.begin());
      }

      postprocess();
      return;
    }
  }
#endif
  bool composite_outputs_materialized = false;
  bool used_composite_body_jit = false;
  for (uint32_t node_id : composite_schedule_)
  {
    if (use_composite_programs && node_id == composite_output_boundary_id_)
    {
 #ifdef EGRESS_LLVM_ORC_JIT
      if (!has_dynamic_registers_)
      {
        ensure_composite_body_jit_current();
        if (run_composite_body_jit(output_materialize_mask))
        {
          composite_outputs_materialized = true;
          used_composite_body_jit = true;
          continue;
        }
      }
      bool used_jit_outputs = false;
      if (!has_dynamic_registers_)
      {
        ensure_numeric_jit_state_current(
          composite_output_jit_,
          composite_output_program_,
          inputs,
          static_cast<unsigned int>(inputs.size()),
          "egress_udm_composite_output");
        if (composite_output_jit_.kernel != nullptr &&
            run_numeric_jit_state(composite_output_jit_, inputs))
        {
          materialize_numeric_outputs(composite_output_jit_, composite_output_program_, outputs, output_materialize_mask);
          used_jit_outputs = true;
        }
      }
      if (!used_jit_outputs)
 #endif
      {
        eval_program(composite_output_program_, temps_);
        for (unsigned int output_id = 0; output_id < composite_output_program_.output_targets.size(); ++output_id)
        {
          outputs[output_id] = temps_[composite_output_program_.output_targets[output_id]];
        }
      }
      composite_outputs_materialized = true;
      continue;
    }

    auto nested_it = nested_module_lookup_.find(node_id);
    if (nested_it == nested_module_lookup_.end())
    {
      continue;
    }

    NestedModuleRuntime & nested = nested_modules_[nested_it->second];
#ifdef EGRESS_LLVM_ORC_JIT
    bool used_jit_inputs = false;
    if (!has_dynamic_registers_ &&
        nested_it->second < nested_input_jit_states_.size() &&
        nested_input_jit_states_[nested_it->second])
    {
      NumericJitState & input_jit = *nested_input_jit_states_[nested_it->second];
      ensure_numeric_jit_state_current(
        input_jit,
        nested.input_program,
        inputs,
        static_cast<unsigned int>(inputs.size()),
        "egress_udm_nested_input");
      if (run_numeric_jit_state(input_jit, inputs))
      {
        materialize_numeric_outputs(input_jit, nested.input_program, nested.module->inputs);
        used_jit_inputs = true;
      }
    }
    if (!used_jit_inputs)
#endif
    {
    eval_program(nested.input_program, nested.input_temps);
    for (unsigned int input_id = 0; input_id < nested.input_program.output_targets.size(); ++input_id)
    {
      nested.module->inputs[input_id] = nested.input_temps[nested.input_program.output_targets[input_id]];
    }
    }
    nested.module->process();
  }

  const CompiledProgram & output_program =
    use_composite_programs ? composite_output_program_ : program_;
  const CompiledProgram & register_program =
    use_composite_programs ? composite_register_program_ : program_;

  if (!use_composite_programs)
  {
    eval_program(output_program, temps_);
    for (unsigned int output_id = 0; output_id < output_program.output_targets.size(); ++output_id)
    {
      outputs[output_id] = temps_[output_program.output_targets[output_id]];
    }
  }
  else if (!composite_outputs_materialized)
  {
    throw std::invalid_argument("Composite module schedule did not materialize the output boundary.");
  }

#ifdef EGRESS_LLVM_ORC_JIT
  if (!used_composite_body_jit &&
      use_composite_programs &&
      !has_dynamic_registers_ &&
      (ensure_numeric_jit_state_current(
         composite_register_jit_,
         composite_register_program_,
         inputs,
         static_cast<unsigned int>(inputs.size()),
         "egress_udm_composite_register"),
       composite_register_jit_.kernel != nullptr) &&
      run_numeric_jit_state(composite_register_jit_, inputs))
  {
    apply_numeric_register_targets(composite_register_jit_, register_program);
  }
  else
#endif
  {
    eval_program(register_program, temps_);
    for (unsigned int register_id = 0; register_id < register_program.register_targets.size(); ++register_id)
    {
      const int32_t target = register_program.register_targets[register_id];
      if (target >= 0)
      {
        next_registers_[register_id] = temps_[static_cast<std::size_t>(target)];
      }
      else
      {
        next_registers_[register_id] = registers_[register_id];
      }
    }
    registers_.swap(next_registers_);
  }

  if (!delay_update_program_.output_targets.empty())
  {
    if (used_composite_body_jit)
    {
      // Delay states were already advanced by the fused composite body kernel.
    }
    else
    {
#ifdef EGRESS_LLVM_ORC_JIT
      if (!has_dynamic_registers_ &&
        (ensure_numeric_jit_state_current(
           delay_update_jit_,
           delay_update_program_,
           inputs,
           static_cast<unsigned int>(inputs.size()),
           "egress_udm_delay_update"),
         delay_update_jit_.kernel != nullptr) &&
        run_numeric_jit_state(delay_update_jit_, inputs))
    {
      next_delay_states_ = delay_states_;
      materialize_numeric_outputs(delay_update_jit_, delay_update_program_, next_delay_states_);
      delay_states_.swap(next_delay_states_);
    }
      else
#endif
      {
        eval_program(delay_update_program_, temps_);
        next_delay_states_ = delay_states_;
        for (unsigned int delay_id = 0; delay_id < delay_update_program_.output_targets.size(); ++delay_id)
        {
          next_delay_states_[delay_id] = temps_[delay_update_program_.output_targets[delay_id]];
        }
        delay_states_.swap(next_delay_states_);
      }
    }
  }

  if (!use_composite_programs || has_dynamic_registers_)
  {
    registers_.swap(next_registers_);
  }
  postprocess();
}

void Module::advance_sample_index_tree()
{
  ++sample_index_;
  for (auto & nested : nested_modules_)
  {
    if (nested.module)
    {
      nested.module->advance_sample_index_tree();
    }
  }
}

unsigned int Module::input_count() const
{
  return input_count_;
}

unsigned int Module::output_count() const
{
  return static_cast<unsigned int>(outputs.size());
}

unsigned int Module::register_count() const
{
  return static_cast<unsigned int>(registers_.size());
}

#ifdef EGRESS_PROFILE
Module::CompileStats Module::compile_stats() const
{
  CompileStats stats;
  if (has_nested_modules_ || has_delay_states_)
  {
    stats.instruction_count = static_cast<uint64_t>(composite_output_program_.instructions.size()) +
                              static_cast<uint64_t>(composite_register_program_.instructions.size());
    stats.register_count = static_cast<uint64_t>(
      std::max(
        composite_output_program_.register_count,
        std::max(composite_register_program_.register_count, delay_update_program_.register_count)));
  }
  else
  {
    stats.instruction_count = static_cast<uint64_t>(program_.instructions.size());
    stats.register_count = static_cast<uint64_t>(std::max(program_.register_count, delay_update_program_.register_count));
  }
  stats.instruction_count += static_cast<uint64_t>(delay_update_program_.instructions.size());
  stats.nested_module_count = static_cast<uint64_t>(nested_modules_.size());
#ifdef EGRESS_LLVM_ORC_JIT
  stats.numeric_jit_instruction_count = numeric_jit_instruction_count_;
  if (composite_body_jit_.state.kernel != nullptr)
  {
    stats.numeric_jit_instruction_count += composite_body_jit_.instruction_count;
  }
  if (composite_output_jit_.kernel != nullptr)
  {
    stats.numeric_jit_instruction_count += composite_output_jit_.instruction_count;
  }
  if (composite_register_jit_.kernel != nullptr)
  {
    stats.numeric_jit_instruction_count += composite_register_jit_.instruction_count;
  }
  if (delay_update_jit_.kernel != nullptr)
  {
    stats.numeric_jit_instruction_count += delay_update_jit_.instruction_count;
  }
  for (const auto & state : nested_input_jit_states_)
  {
    if (state && state->kernel != nullptr)
    {
      stats.numeric_jit_instruction_count += state->instruction_count;
    }
  }
  stats.jit_status = jit_status_;
  #else
  stats.jit_status = "LLVM ORC JIT disabled";
#endif
  return stats;
}

void Module::update_profile_max(std::atomic<uint64_t> & dst, uint64_t candidate)
{
  uint64_t current = dst.load(std::memory_order_relaxed);
  while (current < candidate &&
         !dst.compare_exchange_weak(current, candidate, std::memory_order_relaxed))
  {
  }
}

void Module::record_numeric_input_sync_profile(uint64_t elapsed_ns)
{
  profile_numeric_input_sync_call_count_.fetch_add(1, std::memory_order_relaxed);
  profile_numeric_input_sync_total_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
  update_profile_max(profile_numeric_input_sync_max_ns_, elapsed_ns);
}

void Module::record_numeric_output_materialize_profile(
  uint64_t elapsed_ns,
  uint64_t scalar_count,
  uint64_t array_count,
  uint64_t matrix_count)
{
  profile_numeric_output_materialize_call_count_.fetch_add(1, std::memory_order_relaxed);
  profile_numeric_output_materialize_total_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
  update_profile_max(profile_numeric_output_materialize_max_ns_, elapsed_ns);
  profile_materialized_scalar_outputs_.fetch_add(scalar_count, std::memory_order_relaxed);
  profile_materialized_array_outputs_.fetch_add(array_count, std::memory_order_relaxed);
  profile_materialized_matrix_outputs_.fetch_add(matrix_count, std::memory_order_relaxed);
}

void Module::record_numeric_register_sync_profile(
  uint64_t elapsed_ns,
  uint64_t scalar_count,
  uint64_t array_count)
{
  profile_numeric_register_sync_call_count_.fetch_add(1, std::memory_order_relaxed);
  profile_numeric_register_sync_total_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
  update_profile_max(profile_numeric_register_sync_max_ns_, elapsed_ns);
  profile_materialized_scalar_registers_.fetch_add(scalar_count, std::memory_order_relaxed);
  profile_materialized_array_registers_.fetch_add(array_count, std::memory_order_relaxed);
}

Module::RuntimeStats Module::runtime_stats() const
{
  RuntimeStats stats;
  stats.numeric_input_sync_call_count = profile_numeric_input_sync_call_count_.load(std::memory_order_relaxed);
  stats.numeric_input_sync_total_ns = profile_numeric_input_sync_total_ns_.load(std::memory_order_relaxed);
  stats.numeric_input_sync_max_ns = profile_numeric_input_sync_max_ns_.load(std::memory_order_relaxed);
  stats.numeric_output_materialize_call_count =
    profile_numeric_output_materialize_call_count_.load(std::memory_order_relaxed);
  stats.numeric_output_materialize_total_ns =
    profile_numeric_output_materialize_total_ns_.load(std::memory_order_relaxed);
  stats.numeric_output_materialize_max_ns =
    profile_numeric_output_materialize_max_ns_.load(std::memory_order_relaxed);
  stats.materialized_scalar_outputs = profile_materialized_scalar_outputs_.load(std::memory_order_relaxed);
  stats.materialized_array_outputs = profile_materialized_array_outputs_.load(std::memory_order_relaxed);
  stats.materialized_matrix_outputs = profile_materialized_matrix_outputs_.load(std::memory_order_relaxed);
  stats.numeric_register_sync_call_count = profile_numeric_register_sync_call_count_.load(std::memory_order_relaxed);
  stats.numeric_register_sync_total_ns = profile_numeric_register_sync_total_ns_.load(std::memory_order_relaxed);
  stats.numeric_register_sync_max_ns = profile_numeric_register_sync_max_ns_.load(std::memory_order_relaxed);
  stats.materialized_scalar_registers = profile_materialized_scalar_registers_.load(std::memory_order_relaxed);
  stats.materialized_array_registers = profile_materialized_array_registers_.load(std::memory_order_relaxed);
  return stats;
}

void Module::reset_runtime_stats()
{
  profile_numeric_input_sync_call_count_.store(0, std::memory_order_relaxed);
  profile_numeric_input_sync_total_ns_.store(0, std::memory_order_relaxed);
  profile_numeric_input_sync_max_ns_.store(0, std::memory_order_relaxed);
  profile_numeric_output_materialize_call_count_.store(0, std::memory_order_relaxed);
  profile_numeric_output_materialize_total_ns_.store(0, std::memory_order_relaxed);
  profile_numeric_output_materialize_max_ns_.store(0, std::memory_order_relaxed);
  profile_materialized_scalar_outputs_.store(0, std::memory_order_relaxed);
  profile_materialized_array_outputs_.store(0, std::memory_order_relaxed);
  profile_materialized_matrix_outputs_.store(0, std::memory_order_relaxed);
  profile_numeric_register_sync_call_count_.store(0, std::memory_order_relaxed);
  profile_numeric_register_sync_total_ns_.store(0, std::memory_order_relaxed);
  profile_numeric_register_sync_max_ns_.store(0, std::memory_order_relaxed);
  profile_materialized_scalar_registers_.store(0, std::memory_order_relaxed);
  profile_materialized_array_registers_.store(0, std::memory_order_relaxed);
}
#endif

void Module::reset_inputs_after_process()
{
  for (auto & in : inputs)
  {
    in = expr::float_value(0.0);
  }
}

void Module::postprocess()
{
  reset_inputs_after_process();

  for (auto & out : outputs)
  {
    clamp_output_value(out);
  }
}

double Module::clamp_output_scalar(double value)
{
  return std::fmax(-10.0, std::fmin(10.0, value));
}

void Module::clamp_output_value(Value & value)
{
  if (value.type == ValueType::Array)
  {
    for (auto & item : value.array_items)
    {
      clamp_output_value(item);
    }
    return;
  }
  if (value.type == ValueType::Matrix)
  {
    for (auto & item : value.matrix_items)
    {
      clamp_output_value(item);
    }
    return;
  }
  const double clamped = clamp_output_scalar(expr::to_float64(value));
  value = expr::float_value(clamped);
}

void Module::resize_array_registers_to_inputs()
{
  if (!has_dynamic_registers_)
  {
    return;
  }

  for (unsigned int reg_id = 0; reg_id < register_array_specs_.size(); ++reg_id)
  {
    const RegisterArraySpec & spec = register_array_specs_[reg_id];
    if (!spec.enabled)
    {
      continue;
    }

    if (spec.source_input_id >= inputs.size())
    {
      throw std::invalid_argument("Array register spec references unknown input id.");
    }

    const Value & input = inputs[spec.source_input_id];
    if (input.type != ValueType::Array)
    {
      throw std::invalid_argument("Array register requires array-valued input.");
    }

    const std::size_t desired = input.array_items.size();
    Value & reg = registers_[reg_id];
    bool resized = false;

    if (reg.type != ValueType::Array)
    {
      std::vector<Value> items(desired, spec.init_value);
      reg = expr::array_value(std::move(items));
      resized = true;
    }
    else if (reg.array_items.size() != desired)
    {
      std::vector<Value> items = reg.array_items;
      items.resize(desired, spec.init_value);
      reg = expr::array_value(std::move(items));
      resized = true;
    }

    if (resized)
    {
      next_registers_[reg_id] = reg;
    }
  }
}

Module::CompiledProgram Module::compile_program(
  const std::vector<ExprSpecPtr> & output_exprs,
  const std::vector<ExprSpecPtr> & register_exprs)
{
  CompiledProgram compiled;
  compiled.output_targets.reserve(output_exprs.size());
  compiled.register_targets.assign(register_exprs.size(), -1);

  std::unordered_map<std::size_t, std::vector<std::pair<ExprSpecPtr, uint32_t>>> memo;
  std::unordered_map<const ExprSpec *, std::size_t> hash_cache;
  for (const auto & expr : output_exprs)
  {
    ExprSpecPtr inlined = egress_expr_inline::inline_functions(expr);
    compiled.output_targets.push_back(compile_expr_node(inlined, compiled, memo, hash_cache));
  }
  for (unsigned int i = 0; i < register_exprs.size(); ++i)
  {
    if (register_exprs[i])
    {
      ExprSpecPtr inlined = egress_expr_inline::inline_functions(register_exprs[i]);
      compiled.register_targets[i] = static_cast<int32_t>(compile_expr_node(inlined, compiled, memo, hash_cache));
    }
  }
  return compiled;
}

uint32_t Module::compile_expr_node(
  const ExprSpecPtr & expr,
  CompiledProgram & compiled,
  std::unordered_map<std::size_t, std::vector<std::pair<ExprSpecPtr, uint32_t>>> & memo,
  std::unordered_map<const ExprSpec *, std::size_t> & hash_cache)
{
  const std::size_t hash = egress_expr_inline::structural_hash(expr, hash_cache);
  auto memo_it = memo.find(hash);
  if (memo_it != memo.end())
  {
    for (const auto & candidate : memo_it->second)
    {
      if (egress_expr_inline::structural_equal(expr, candidate.first))
      {
        return candidate.second;
      }
    }
  }

  if (!expr)
  {
    Instr instr;
    instr.kind = ExprKind::Literal;
    instr.dst = compiled.register_count++;
    instr.literal = expr::float_value(0.0);
    compiled.instructions.push_back(std::move(instr));
    memo[hash].push_back(std::make_pair(expr, compiled.instructions.back().dst));
    return compiled.instructions.back().dst;
  }

  Instr instr;
  instr.kind = expr->kind;
  instr.dst = compiled.register_count++;

  switch (expr->kind)
  {
    case ExprKind::Literal:
      instr.literal = expr->literal;
      break;
    case ExprKind::InputValue:
    case ExprKind::RegisterValue:
      instr.slot_id = expr->slot_id;
      break;
    case ExprKind::NestedValue:
      instr.slot_id = expr->slot_id;
      instr.output_id = expr->output_id;
      break;
    case ExprKind::DelayValue:
      instr.slot_id = expr->slot_id;
      break;
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
      break;
    case ExprKind::ArrayPack:
      instr.args.reserve(expr->args.size());
      for (const auto & arg : expr->args)
      {
        instr.args.push_back(compile_expr_node(arg, compiled, memo, hash_cache));
      }
      break;
    default:
      if (egress_module_detail::is_local_unary(expr->kind))
      {
        instr.src_a = compile_expr_node(expr->lhs, compiled, memo, hash_cache);
      }
      else if (egress_module_detail::is_local_ternary(expr->kind))
      {
        instr.src_a = compile_expr_node(expr->lhs, compiled, memo, hash_cache);
        instr.src_b = compile_expr_node(expr->rhs, compiled, memo, hash_cache);
        instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, memo, hash_cache);
      }
      else if (egress_module_detail::is_local_binary(expr->kind))
      {
        instr.src_a = compile_expr_node(expr->lhs, compiled, memo, hash_cache);
        instr.src_b = compile_expr_node(expr->rhs, compiled, memo, hash_cache);
      }
      else
      {
        throw std::invalid_argument("Unsupported module expression kind.");
      }
      break;
  }

  compiled.instructions.push_back(std::move(instr));
  memo[hash].push_back(std::make_pair(expr, compiled.instructions.back().dst));
  return compiled.instructions.back().dst;
}

void Module::eval_program(const CompiledProgram & expr, std::vector<Value> & temps) const
{
  if (expr.instructions.empty())
  {
    return;
  }

  if (temps.size() < expr.register_count)
  {
    temps.resize(expr.register_count, expr::float_value(0.0));
  }

  for (const Instr & instr : expr.instructions)
  {
    switch (instr.kind)
    {
      case ExprKind::Function:
      case ExprKind::Call:
        throw std::invalid_argument("Function values must be inlined before evaluation.");
      case ExprKind::Literal:
        temps[instr.dst] = instr.literal;
        break;
      case ExprKind::InputValue:
        temps[instr.dst] = instr.slot_id < inputs.size()
                             ? inputs[instr.slot_id]
                             : expr::float_value(0.0);
        break;
      case ExprKind::RegisterValue:
        temps[instr.dst] = instr.slot_id < registers_.size()
                             ? registers_[instr.slot_id]
                             : expr::float_value(0.0);
        break;
      case ExprKind::NestedValue:
      {
        const auto nested_it = nested_module_lookup_.find(instr.slot_id);
        if (nested_it == nested_module_lookup_.end())
        {
          temps[instr.dst] = expr::float_value(0.0);
          break;
        }
        const NestedModuleRuntime & nested = nested_modules_[nested_it->second];
        temps[instr.dst] = instr.output_id < nested.module->outputs.size()
                             ? nested.module->outputs[instr.output_id]
                             : expr::float_value(0.0);
        break;
      }
      case ExprKind::DelayValue:
      {
        const auto delay_it = delay_state_lookup_.find(instr.slot_id);
        temps[instr.dst] = delay_it != delay_state_lookup_.end()
                             ? delay_states_[delay_it->second]
                             : expr::float_value(0.0);
        break;
      }
      case ExprKind::SampleRate:
        temps[instr.dst] = expr::float_value(sample_rate_);
        break;
      case ExprKind::SampleIndex:
        temps[instr.dst] = expr::int_value(static_cast<int64_t>(sample_index_));
        break;
      case ExprKind::ArrayPack:
      {
        std::vector<Value> items;
        items.reserve(instr.args.size());
        for (uint32_t src : instr.args)
        {
          if (expr::is_array(temps[src]) || expr::is_matrix(temps[src]))
          {
            throw std::invalid_argument("Nested arrays are not supported.");
          }
          items.push_back(temps[src]);
        }
        temps[instr.dst] = expr::array_value(std::move(items));
        break;
      }
      case ExprKind::Index:
      {
        const Value & array_value = temps[instr.src_a];
        const int64_t index = expr::to_int64(temps[instr.src_b]);
        if (index < 0)
        {
          throw std::out_of_range("Array index out of range.");
        }
        if (expr::is_array(array_value))
        {
          if (static_cast<std::size_t>(index) >= array_value.array_items.size())
          {
            throw std::out_of_range("Array index out of range.");
          }
          temps[instr.dst] = array_value.array_items[static_cast<std::size_t>(index)];
          break;
        }
        if (expr::is_matrix(array_value))
        {
          temps[instr.dst] = expr::array_from_matrix_row(array_value, static_cast<std::size_t>(index));
          break;
        }
        temps[instr.dst] = expr::float_value(0.0);
        break;
      }
      case ExprKind::ArraySet:
        temps[instr.dst] = expr_eval::array_set_value(
          temps[instr.src_a],
          temps[instr.src_b],
          temps[instr.src_c]);
        break;
      case ExprKind::Abs:
        temps[instr.dst] = expr_eval::abs_value(temps[instr.src_a]);
        break;
      case ExprKind::Not:
        temps[instr.dst] = expr_eval::not_value(temps[instr.src_a]);
        break;
      case ExprKind::Less:
        temps[instr.dst] = expr_eval::less_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::LessEqual:
        temps[instr.dst] = expr_eval::less_equal_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Greater:
        temps[instr.dst] = expr_eval::greater_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::GreaterEqual:
        temps[instr.dst] = expr_eval::greater_equal_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Equal:
        temps[instr.dst] = expr_eval::equal_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::NotEqual:
        temps[instr.dst] = expr_eval::not_equal_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Add:
        temps[instr.dst] = expr_eval::add_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Sub:
        temps[instr.dst] = expr_eval::sub_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Mul:
        temps[instr.dst] = expr_eval::mul_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::MatMul:
        temps[instr.dst] = expr_eval::matmul_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Div:
        temps[instr.dst] = expr_eval::div_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Pow:
        temps[instr.dst] = expr_eval::pow_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Mod:
        temps[instr.dst] = expr_eval::mod_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::FloorDiv:
        temps[instr.dst] = expr_eval::floor_div_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::BitAnd:
        temps[instr.dst] = expr_eval::bit_and_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::BitOr:
        temps[instr.dst] = expr_eval::bit_or_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::BitXor:
        temps[instr.dst] = expr_eval::bit_xor_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::LShift:
        temps[instr.dst] = expr_eval::lshift_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::RShift:
        temps[instr.dst] = expr_eval::rshift_values(temps[instr.src_a], temps[instr.src_b]);
        break;
      case ExprKind::Clamp:
        temps[instr.dst] = expr_eval::clamp_values(temps[instr.src_a], temps[instr.src_b], temps[instr.src_c]);
        break;
      case ExprKind::Log:
        temps[instr.dst] = expr_eval::log_value(temps[instr.src_a]);
        break;
      case ExprKind::Sin:
        temps[instr.dst] = expr_eval::sin_value(temps[instr.src_a]);
        break;
      case ExprKind::Neg:
        temps[instr.dst] = expr_eval::neg_value(temps[instr.src_a]);
        break;
      case ExprKind::BitNot:
        temps[instr.dst] = expr_eval::bit_not_value(temps[instr.src_a]);
        break;
      case ExprKind::Ref:
        temps[instr.dst] = expr::float_value(0.0);
        break;
    }
  }
}
