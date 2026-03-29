#pragma once

void Module::process(const std::vector<bool> * output_materialize_mask)
{
  const bool use_composite_programs = has_nested_modules_ || has_delay_states_;
#ifdef EGRESS_LLVM_ORC_JIT
  if (numeric_output_scalar_mask_.size() != outputs.size())
  {
    numeric_output_scalar_mask_.assign(outputs.size(), false);
    numeric_output_scalars_.assign(outputs.size(), 0.0);
  }
  else
  {
    std::fill(numeric_output_scalar_mask_.begin(), numeric_output_scalar_mask_.end(), false);
  }
#endif
#ifdef EGRESS_LLVM_ORC_JIT
  if (!numeric_input_override_active_)
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
        sample_index_,
        numeric_param_ptrs_.data(),
        numeric_int_inputs_.data(),
        numeric_int_registers_.data(),
        numeric_int_array_ptrs_.empty() ? nullptr : numeric_int_array_ptrs_.data(),
        numeric_int_temps_.data());
      capture_numeric_scalar_outputs(program_, numeric_output_info_, numeric_temps_);

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
            !(*output_materialize_mask)[output_id])
        {
          continue;
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
          numeric_array_storage_,
          &numeric_int_temps_,
          &numeric_int_array_storage_);
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
          const bool is_int = register_id < registers_.size() &&
            (registers_[register_id].type == ValueType::Int ||
             registers_[register_id].type == ValueType::Bool);
          if (is_int)
          {
            if (target >= 0 && static_cast<std::size_t>(target) < numeric_int_temps_.size())
            {
              numeric_next_int_registers_[register_id] = numeric_int_temps_[static_cast<std::size_t>(target)];
            }
            else
            {
              numeric_next_int_registers_[register_id] = numeric_int_registers_[register_id];
            }
            numeric_next_registers_[register_id] = numeric_registers_[register_id];
          }
          else
          {
            if (target >= 0)
            {
              numeric_next_registers_[register_id] = numeric_temps_[static_cast<std::size_t>(target)];
            }
            else
            {
              numeric_next_registers_[register_id] = numeric_registers_[register_id];
            }
            numeric_next_int_registers_[register_id] = numeric_int_registers_[register_id];
          }
        }
        else
        {
          numeric_next_registers_[register_id] = numeric_registers_[register_id];
          numeric_next_int_registers_[register_id] = numeric_int_registers_[register_id];
        }
      }

      // Preserve anonymous registers (e.g., SmoothedParam state) written in-kernel
      for (unsigned int register_id = static_cast<unsigned int>(program_.register_targets.size());
           register_id < numeric_registers_.size();
           ++register_id)
      {
        numeric_next_registers_[register_id] = numeric_registers_[register_id];
      }
      for (unsigned int register_id = static_cast<unsigned int>(program_.register_targets.size());
           register_id < numeric_int_registers_.size();
           ++register_id)
      {
        numeric_next_int_registers_[register_id] = numeric_int_registers_[register_id];
      }

      numeric_registers_.swap(numeric_next_registers_);
      numeric_int_registers_.swap(numeric_next_int_registers_);

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
          if (register_id < numeric_register_arrays_.size())
          {
            numeric_register_arrays_[register_id] = dst;
          }
          continue;
        }
        std::copy(src.begin(), src.end(), dst.begin());
        if (register_id < numeric_register_arrays_.size())
        {
          numeric_register_arrays_[register_id] = dst;
        }
      }

      value_registers_dirty_ = true;
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
      {
        ensure_numeric_jit_state_current(
          composite_output_jit_,
          composite_output_program_,
          inputs,
          static_cast<unsigned int>(inputs.size()));
        if (composite_output_jit_.kernel != nullptr &&
            run_numeric_jit_state(composite_output_jit_, inputs))
        {
          capture_numeric_scalar_outputs(
            composite_output_program_,
            composite_output_jit_.output_info,
            composite_output_jit_.temps);
          materialize_numeric_outputs(composite_output_jit_, composite_output_program_, outputs, output_materialize_mask);
          used_jit_outputs = true;
        }
      }
      if (!used_jit_outputs)
#endif
      {
        throw std::runtime_error("JIT unavailable for composite output program: " + jit_status_);
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
    if (nested_it->second < nested_input_jit_states_.size() &&
        nested_input_jit_states_[nested_it->second])
    {
      NumericJitState & input_jit = *nested_input_jit_states_[nested_it->second];
      ensure_numeric_jit_state_current(
        input_jit,
        nested.input_program,
        inputs,
        static_cast<unsigned int>(inputs.size()));
      if (run_numeric_jit_state(input_jit, inputs))
      {
        if (nested.module->try_set_direct_numeric_inputs(input_jit, nested.input_program))
        {
          used_jit_inputs = true;
        }
        else
        {
          materialize_numeric_outputs(input_jit, nested.input_program, nested.module->inputs);
          used_jit_inputs = true;
        }
      }
    }
    if (!used_jit_inputs)
#endif
    {
      throw std::runtime_error("JIT unavailable for nested module input program: " + jit_status_);
    }
    nested.module->process();
  }

  const CompiledProgram & output_program =
    use_composite_programs ? composite_output_program_ : program_;
  const CompiledProgram & register_program =
    use_composite_programs ? composite_register_program_ : program_;

  if (!use_composite_programs)
  {
    throw std::runtime_error("JIT kernel unavailable at process time: " + jit_status_);
  }
  else if (!composite_outputs_materialized)
  {
    throw std::invalid_argument("Composite module schedule did not materialize the output boundary.");
  }

#ifdef EGRESS_LLVM_ORC_JIT
  if (!used_composite_body_jit)
  {
    if (use_composite_programs &&
        (ensure_numeric_jit_state_current(
           composite_register_jit_,
           composite_register_program_,
           inputs,
           static_cast<unsigned int>(inputs.size())),
         composite_register_jit_.kernel != nullptr) &&
        run_numeric_jit_state(composite_register_jit_, inputs))
    {
      apply_numeric_register_targets(composite_register_jit_, register_program);
    }
    else
    {
      throw std::runtime_error("JIT unavailable for register program: " + jit_status_);
    }
  }
#endif

  if (!delay_update_program_.output_targets.empty())
  {
    if (used_composite_body_jit)
    {
      // Delay states were already advanced by the fused composite body kernel.
    }
    else
    {
#ifdef EGRESS_LLVM_ORC_JIT
      if ((ensure_numeric_jit_state_current(
           delay_update_jit_,
           delay_update_program_,
           inputs,
           static_cast<unsigned int>(inputs.size())),
          delay_update_jit_.kernel != nullptr) &&
        run_numeric_jit_state(delay_update_jit_, inputs))
    {
      if (!update_numeric_delay_states_from_outputs(delay_update_jit_, delay_update_program_))
      {
        next_delay_states_ = delay_states_;
        materialize_numeric_outputs(delay_update_jit_, delay_update_program_, next_delay_states_);
        delay_states_.swap(next_delay_states_);
        value_delay_states_dirty_ = false;
      }
    }
      else
#endif
      {
        throw std::runtime_error("JIT unavailable for delay update program: " + jit_status_);
      }
    }
  }

  if (!use_composite_programs)
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
#ifdef EGRESS_LLVM_ORC_JIT
  numeric_input_override_active_ = false;
#endif
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
    case ExprKind::SmoothedParam:
    case ExprKind::TriggerParam:
    {
      // Assign the anonymous register slot (user_register_count_ + anon_index)
      const auto it = param_anon_reg_map_.find(expr->control_param);
      instr.slot_id = (it != param_anon_reg_map_.end())
        ? (user_register_count_ + it->second)
        : 0;
      instr.control_param = expr->control_param;
      break;
    }
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

void Module::eval_program(const CompiledProgram & expr, std::vector<Value> & temps)
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
        if (delay_it == delay_state_lookup_.end())
        {
          temps[instr.dst] = expr::float_value(0.0);
          break;
        }
#ifdef EGRESS_LLVM_ORC_JIT
        double scalar = 0.0;
        if (try_get_numeric_delay_scalar(static_cast<unsigned int>(delay_it->second), scalar))
        {
          temps[instr.dst] = expr::float_value(scalar);
          break;
        }
        if (const auto * values = try_get_numeric_delay_array(static_cast<unsigned int>(delay_it->second)))
        {
          std::vector<Value> items(values->size(), expr::float_value(0.0));
          for (std::size_t item_id = 0; item_id < values->size(); ++item_id)
          {
            assign_scalar_numeric_value(items[item_id], (*values)[item_id]);
          }
          temps[instr.dst] = expr::array_value(std::move(items));
          break;
        }
#endif
        temps[instr.dst] = delay_states_[delay_it->second];
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
      case ExprKind::Select:
        temps[instr.dst] = expr_eval::select_values(temps[instr.src_a], temps[instr.src_b], temps[instr.src_c]);
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
      case ExprKind::SmoothedParam:
      {
        // One-pole lowpass smoother: y[n] = y[n-1] + coeff * (target - y[n-1])
        // coeff = 1 - exp(-1 / (time_const * sample_rate))
        // The anonymous register at slot_id stores the current smoothed value.
        // Writing to next_registers_ here is safe: the slot is beyond user register range
        // and the register swap at end of process() will persist the updated state.
        if (instr.control_param && instr.slot_id < registers_.size())
        {
          const double target = instr.control_param->value.load(std::memory_order_relaxed);
          const double current = egress_expr::to_float64(registers_[instr.slot_id]);
          const double tc = instr.control_param->time_const;
          const double coeff = (tc > 0.0)
            ? 1.0 - std::exp(-1.0 / (tc * sample_rate_))
            : 1.0;
          const double new_val = current + coeff * (target - current);
          temps[instr.dst] = egress_expr::float_value(new_val);
          // Side-effect: advance smoother state (written to next_registers_, swapped end of process())
          if (instr.slot_id < next_registers_.size())
          {
            next_registers_[instr.slot_id] = egress_expr::float_value(new_val);
          }
        }
        else
        {
          temps[instr.dst] = egress_expr::float_value(0.0);
        }
        break;
      }
      case ExprKind::TriggerParam:
      {
        // Read the per-frame snapshot written by Graph before the processing loop.
        // The Graph does a single exchange(0.0) per frame so all modules see the same value.
        if (instr.control_param)
        {
          temps[instr.dst] = egress_expr::float_value(instr.control_param->frame_value.load(std::memory_order_relaxed));
        }
        else
        {
          temps[instr.dst] = egress_expr::float_value(0.0);
        }
        break;
      }
    }
  }
}
