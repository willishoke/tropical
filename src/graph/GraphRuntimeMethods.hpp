#pragma once

void Graph::wait_for_runtime_available(uint32_t runtime_index) const
{
  while (audio_processing_.load(std::memory_order_acquire) &&
         audio_runtime_index_.load(std::memory_order_acquire) == runtime_index)
  {
    std::this_thread::yield();
  }
}

Graph::RuntimeState Graph::build_runtime_locked() const
{
  RuntimeState runtime;
  runtime.modules.reserve(control_modules_.size());
  runtime.name_to_id.reserve(control_modules_.size());

  for (const auto & [name, module] : control_modules_)
  {
    const uint32_t module_id = static_cast<uint32_t>(runtime.modules.size());
    runtime.name_to_id.emplace(name, module_id);

    ModuleSlot slot;
    slot.name = name;
    slot.module = module.module;
    slot.input_program.result_registers.resize(module.in_count, 0);
    runtime.modules.push_back(std::move(slot));
  }

  for (const auto & [name, module] : control_modules_)
  {
    auto id_it = runtime.name_to_id.find(name);
    if (id_it == runtime.name_to_id.end())
    {
      continue;
    }

    ModuleSlot & slot = runtime.modules[id_it->second];
    const unsigned int input_count = static_cast<unsigned int>(slot.input_program.result_registers.size());
    slot.input_program = compile_input_program(module.input_exprs, input_count, runtime);
    slot.input_registers.assign(slot.input_program.register_count, float_value(0.0));
    slot.output_materialize_mask.assign(module.out_count, false);
    slot.indexed_output_indices.assign(module.out_count, {});
    slot.indexed_prev_output_values.assign(module.out_count, {});
  }

  for (auto & consumer_slot : runtime.modules)
  {
    for (auto & instr : consumer_slot.input_program.instructions)
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        continue;
      }

      ModuleSlot & source_slot = runtime.modules[instr.ref_module_id];
      if (instr.ref_output_id >= source_slot.output_materialize_mask.size())
      {
        continue;
      }

      if (instr.opcode == OpCode::Ref)
      {
        source_slot.output_materialize_mask[instr.ref_output_id] = true;
        continue;
      }

      if (instr.opcode != OpCode::RefIndex || instr.ref_index < 0)
      {
        continue;
      }
      auto & indices = source_slot.indexed_output_indices[instr.ref_output_id];
      auto it = std::find(indices.begin(), indices.end(), instr.ref_index);
      uint32_t cache_slot = 0;
      if (it == indices.end())
      {
        cache_slot = static_cast<uint32_t>(indices.size());
        indices.push_back(instr.ref_index);
        source_slot.indexed_prev_output_values[instr.ref_output_id].push_back(float_value(0.0));
      }
      else
      {
        cache_slot = static_cast<uint32_t>(std::distance(indices.begin(), it));
      }
      instr.src_a = cache_slot;
    }
  }

  runtime.mix.reserve(control_mix_.size());
  for (const auto & tap : control_mix_)
  {
    auto it = runtime.name_to_id.find(tap.first);
    if (it == runtime.name_to_id.end())
    {
      continue;
    }

    const uint32_t module_id = it->second;
    if (module_id >= runtime.modules.size() ||
        !runtime.modules[module_id].module ||
        tap.second >= runtime.modules[module_id].module->outputs.size())
    {
      continue;
    }

    runtime.modules[module_id].output_materialize_mask[tap.second] = true;
    runtime.mix.push_back(MixTap{module_id, tap.second});
  }

  runtime.mix_exprs.reserve(control_mix_exprs_.size());
  for (const auto & mix_expr_spec : control_mix_exprs_)
  {
    MixExpr mix_expr;
    ExprSpecPtr inlined = egress_expr_inline::inline_functions(mix_expr_spec);
    mix_expr.result_register = compile_expr_node(inlined, mix_expr.program, runtime);
    mix_expr.registers.assign(mix_expr.program.register_count, float_value(0.0));
    runtime.mix_exprs.push_back(std::move(mix_expr));
  }

  for (auto & mix_expr : runtime.mix_exprs)
  {
    for (auto & instr : mix_expr.program.instructions)
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        continue;
      }

      ModuleSlot & source_slot = runtime.modules[instr.ref_module_id];
      if (instr.ref_output_id >= source_slot.output_materialize_mask.size())
      {
        continue;
      }

      if (instr.opcode == OpCode::Ref)
      {
        source_slot.output_materialize_mask[instr.ref_output_id] = true;
        continue;
      }

      if (instr.opcode != OpCode::RefIndex || instr.ref_index < 0)
      {
        continue;
      }

      auto & indices = source_slot.indexed_output_indices[instr.ref_output_id];
      auto it = std::find(indices.begin(), indices.end(), instr.ref_index);
      uint32_t cache_slot = 0;
      if (it == indices.end())
      {
        cache_slot = static_cast<uint32_t>(indices.size());
        indices.push_back(instr.ref_index);
        source_slot.indexed_prev_output_values[instr.ref_output_id].push_back(float_value(0.0));
      }
      else
      {
        cache_slot = static_cast<uint32_t>(std::distance(indices.begin(), it));
      }
      instr.src_a = cache_slot;
    }
  }

  runtime.taps.reserve(control_taps_.size());
  for (const auto & tap_spec : control_taps_)
  {
    egress_graph_detail::OutputTap tap;
    tap.buffer.buffers[0].assign(bufferLength_, 0.0);
    tap.buffer.buffers[1].assign(bufferLength_, 0.0);
    if (!tap_spec.active)
    {
      runtime.taps.push_back(std::move(tap));
      continue;
    }

    const auto it = runtime.name_to_id.find(tap_spec.output.first);
    if (it == runtime.name_to_id.end())
    {
      runtime.taps.push_back(std::move(tap));
      continue;
    }

    const uint32_t module_id = it->second;
    if (module_id >= runtime.modules.size() ||
        !runtime.modules[module_id].module ||
        tap_spec.output.second >= runtime.modules[module_id].module->outputs.size())
    {
      runtime.taps.push_back(std::move(tap));
      continue;
    }

    runtime.modules[module_id].output_materialize_mask[tap_spec.output.second] = true;
    tap.module_id = module_id;
    tap.output_id = tap_spec.output.second;
    tap.valid = true;
    runtime.taps.push_back(std::move(tap));
  }

  return runtime;
}

void Graph::rebuild_and_publish_runtime_locked()
{
  const uint32_t active = active_runtime_index_.load(std::memory_order_acquire);
  const uint32_t inactive = 1U - active;
  wait_for_runtime_available(inactive);
  runtimes_[inactive] = build_runtime_locked();
  active_runtime_index_.store(inactive, std::memory_order_release);
}

void Graph::eval_instruction(const RuntimeState & runtime, const ExprInstr & instr, Value * registers)
{
  switch (instr.opcode)
  {
    case OpCode::Literal:
      registers[instr.dst] = instr.literal;
      break;
    case OpCode::Ref:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
      if (!slot.module || instr.ref_output_id >= slot.module->prev_outputs.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      registers[instr.dst] = slot.module->prev_outputs[instr.ref_output_id];
      break;
    }
    case OpCode::RefIndex:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
      if (!slot.module || instr.ref_output_id >= slot.module->prev_outputs.size() || instr.ref_index < 0)
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      if (instr.ref_output_id < slot.indexed_prev_output_values.size() &&
          instr.src_a < slot.indexed_prev_output_values[instr.ref_output_id].size())
      {
        registers[instr.dst] = slot.indexed_prev_output_values[instr.ref_output_id][instr.src_a];
        break;
      }
      const Value & output = slot.module->prev_outputs[instr.ref_output_id];
      const std::size_t item_index = static_cast<std::size_t>(instr.ref_index);
      if (expr::is_array(output))
      {
        if (item_index >= output.array_items.size())
        {
          throw std::out_of_range("Array index out of range.");
        }
        registers[instr.dst] = output.array_items[item_index];
        break;
      }
      if (expr::is_matrix(output))
      {
        registers[instr.dst] = expr::array_from_matrix_row(output, item_index);
        break;
      }
      registers[instr.dst] = float_value(0.0);
      break;
    }
    case OpCode::ArrayPack:
    {
      std::vector<Value> items;
      items.reserve(instr.args.size());
      for (uint32_t src : instr.args)
      {
        if (expr::is_array(registers[src]) || expr::is_matrix(registers[src]))
        {
          throw std::invalid_argument("Nested arrays are not supported.");
        }
        items.push_back(registers[src]);
      }
      registers[instr.dst] = expr::array_value(std::move(items));
      break;
    }
    case OpCode::Index:
    {
      const Value & array_value = registers[instr.src_a];
      const int64_t index = expr::to_int64(registers[instr.src_b]);
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
        registers[instr.dst] = array_value.array_items[static_cast<std::size_t>(index)];
        break;
      }
      if (expr::is_matrix(array_value))
      {
        registers[instr.dst] = expr::array_from_matrix_row(array_value, static_cast<std::size_t>(index));
        break;
      }
      registers[instr.dst] = float_value(0.0);
      break;
    }
    case OpCode::ArraySet:
      registers[instr.dst] = expr_eval::array_set_value(
        registers[instr.src_a],
        registers[instr.src_b],
        registers[instr.src_c]);
      break;
    case OpCode::Add:
      registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::AddConst:
      registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], instr.literal);
      break;
    case OpCode::Sub:
      registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::SubConstRhs:
      registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], instr.literal);
      break;
    case OpCode::SubConstLhs:
      registers[instr.dst] = expr_eval::sub_values(instr.literal, registers[instr.src_a]);
      break;
    case OpCode::Mul:
      registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::MulConst:
      registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], instr.literal);
      break;
    case OpCode::Div:
      registers[instr.dst] = expr_eval::div_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::MatMul:
      registers[instr.dst] = expr_eval::matmul_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::DivConstLhs:
      registers[instr.dst] = expr_eval::div_values(instr.literal, registers[instr.src_a]);
      break;
    case OpCode::Mod:
      registers[instr.dst] = expr_eval::mod_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::FloorDiv:
      registers[instr.dst] = expr_eval::floor_div_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::BitAnd:
      registers[instr.dst] = expr_eval::bit_and_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::BitOr:
      registers[instr.dst] = expr_eval::bit_or_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::BitXor:
      registers[instr.dst] = expr_eval::bit_xor_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::LShift:
      registers[instr.dst] = expr_eval::lshift_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::RShift:
      registers[instr.dst] = expr_eval::rshift_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Abs:
      registers[instr.dst] = expr_eval::abs_value(registers[instr.src_a]);
      break;
    case OpCode::Clamp:
      registers[instr.dst] = expr_eval::clamp_values(registers[instr.src_a], registers[instr.src_b], registers[instr.src_c]);
      break;
    case OpCode::Log:
      registers[instr.dst] = expr_eval::log_value(registers[instr.src_a]);
      break;
    case OpCode::Neg:
      registers[instr.dst] = expr_eval::neg_value(registers[instr.src_a]);
      break;
    case OpCode::Not:
      registers[instr.dst] = expr_eval::not_value(registers[instr.src_a]);
      break;
    case OpCode::BitNot:
      registers[instr.dst] = expr_eval::bit_not_value(registers[instr.src_a]);
      break;
    case OpCode::Sin:
      registers[instr.dst] = expr_eval::sin_value(registers[instr.src_a]);
      break;
    case OpCode::Pow:
      registers[instr.dst] = expr_eval::pow_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Less:
      registers[instr.dst] = expr_eval::less_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::LessEqual:
      registers[instr.dst] = expr_eval::less_equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Greater:
      registers[instr.dst] = expr_eval::greater_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::GreaterEqual:
      registers[instr.dst] = expr_eval::greater_equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Equal:
      registers[instr.dst] = expr_eval::equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::NotEqual:
      registers[instr.dst] = expr_eval::not_equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
  }
}

void Graph::eval_mix_instruction(const RuntimeState & runtime, const ExprInstr & instr, Value * registers)
{
  switch (instr.opcode)
  {
    case OpCode::Ref:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
      if (!slot.module || instr.ref_output_id >= slot.module->outputs.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      registers[instr.dst] = slot.module->outputs[instr.ref_output_id];
      break;
    }
    case OpCode::RefIndex:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
      if (!slot.module || instr.ref_output_id >= slot.module->outputs.size() || instr.ref_index < 0)
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
#ifdef EGRESS_LLVM_ORC_JIT
      const bool can_read_numeric_output =
        slot.module->jit_kernel_ != nullptr &&
        instr.ref_output_id < slot.module->numeric_output_info_.size() &&
        static_cast<Module::NumericValueKind>(slot.module->numeric_output_info_[instr.ref_output_id].kind) == Module::NumericValueKind::Array &&
        slot.module->numeric_output_info_[instr.ref_output_id].array_slot < slot.module->numeric_array_storage_.size();
      if (can_read_numeric_output)
      {
        const auto & values = slot.module->numeric_array_storage_[slot.module->numeric_output_info_[instr.ref_output_id].array_slot];
        if (static_cast<std::size_t>(instr.ref_index) >= values.size())
        {
          throw std::out_of_range("Array index out of range.");
        }
        registers[instr.dst] = expr::float_value(Module::clamp_output_scalar(values[static_cast<std::size_t>(instr.ref_index)]));
        break;
      }
#endif
      const Value & output = slot.module->outputs[instr.ref_output_id];
      const std::size_t item_index = static_cast<std::size_t>(instr.ref_index);
      if (expr::is_array(output))
      {
        if (item_index >= output.array_items.size())
        {
          throw std::out_of_range("Array index out of range.");
        }
        registers[instr.dst] = output.array_items[item_index];
        break;
      }
      if (expr::is_matrix(output))
      {
        registers[instr.dst] = expr::array_from_matrix_row(output, item_index);
        break;
      }
      registers[instr.dst] = float_value(0.0);
      break;
    }
    default:
      eval_instruction(runtime, instr, registers);
      break;
  }
}

uint32_t Graph::compile_expr_node(
  const ExprSpecPtr & expr,
  CompiledInputProgram & compiled,
  const RuntimeState & runtime) const
{
  if (!expr)
  {
    ExprInstr instr;
    instr.opcode = OpCode::Literal;
    instr.dst = compiled.register_count++;
    instr.literal = float_value(0.0);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Literal)
  {
    ExprInstr instr;
    instr.opcode = OpCode::Literal;
    instr.dst = compiled.register_count++;
    instr.literal = expr->literal;
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Ref)
  {
    auto it = runtime.name_to_id.find(expr->module_name);
    if (it == runtime.name_to_id.end())
    {
      ExprInstr instr;
      instr.opcode = OpCode::Literal;
      instr.dst = compiled.register_count++;
      instr.literal = float_value(0.0);
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    ExprInstr instr;
    instr.opcode = OpCode::Ref;
    instr.dst = compiled.register_count++;
    instr.ref_module_id = it->second;
    instr.ref_output_id = expr->output_id;
    compiled.instructions.push_back(instr);

    return instr.dst;
  }

  if (expr->kind == ExprKind::ArrayPack)
  {
    ExprInstr instr;
    instr.opcode = OpCode::ArrayPack;
    instr.dst = compiled.register_count++;
    instr.args.reserve(expr->args.size());
    for (const auto & arg : expr->args)
    {
      instr.args.push_back(compile_expr_node(arg, compiled, runtime));
    }
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Index)
  {
    if (expr->lhs && expr->lhs->kind == ExprKind::Ref &&
        expr->rhs && expr->rhs->kind == ExprKind::Literal &&
        expr->rhs->literal.type != ValueType::Array &&
        expr->rhs->literal.type != ValueType::Matrix)
    {
      const int64_t raw_index = expr::to_int64(expr->rhs->literal);
      if (raw_index >= 0)
      {
        auto it = runtime.name_to_id.find(expr->lhs->module_name);
        if (it != runtime.name_to_id.end())
        {
          ExprInstr instr;
          instr.opcode = OpCode::RefIndex;
          instr.dst = compiled.register_count++;
          instr.ref_module_id = it->second;
          instr.ref_output_id = expr->lhs->output_id;
          instr.ref_index = raw_index;
          instr.src_a = std::numeric_limits<uint32_t>::max();
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }
    }

    ExprInstr instr;
    instr.opcode = OpCode::Index;
    instr.dst = compiled.register_count++;
    instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
    instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::ArraySet)
  {
    ExprInstr instr;
    instr.opcode = OpCode::ArraySet;
    instr.dst = compiled.register_count++;
    instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
    instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
    instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, runtime);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Abs ||
      expr->kind == ExprKind::Neg ||
      expr->kind == ExprKind::Not ||
      expr->kind == ExprKind::BitNot ||
      expr->kind == ExprKind::Log ||
      expr->kind == ExprKind::Sin)
  {
    const uint32_t operand = compile_expr_node(expr->lhs, compiled, runtime);
    ExprInstr instr;
    instr.opcode = expr->kind == ExprKind::Abs
                     ? OpCode::Abs
                     : (expr->kind == ExprKind::Neg
                          ? OpCode::Neg
                          : (expr->kind == ExprKind::Not
                               ? OpCode::Not
                               : (expr->kind == ExprKind::BitNot
                                    ? OpCode::BitNot
                                    : (expr->kind == ExprKind::Log ? OpCode::Log : OpCode::Sin))));
    instr.dst = compiled.register_count++;
    instr.src_a = operand;
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Clamp)
  {
    ExprInstr instr;
    instr.opcode = OpCode::Clamp;
    instr.dst = compiled.register_count++;
    instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
    instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
    instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, runtime);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Add)
  {
    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::AddConst;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::AddConst;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = expr->rhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  if (expr->kind == ExprKind::Mul)
  {
    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::MulConst;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::MulConst;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = expr->rhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  if (expr->kind == ExprKind::Sub)
  {
    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::SubConstRhs;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = expr->rhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::SubConstLhs;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  if (expr->kind == ExprKind::Div)
  {
    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::MulConst;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = to_float64(expr->rhs->literal) == 0.0
                        ? float_value(0.0)
                        : float_value(1.0 / to_float64(expr->rhs->literal));
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::DivConstLhs;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
  const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);

  ExprInstr instr;
  switch (expr->kind)
  {
    case ExprKind::Add:
      instr.opcode = OpCode::Add;
      break;
    case ExprKind::Sub:
      instr.opcode = OpCode::Sub;
      break;
    case ExprKind::Mul:
      instr.opcode = OpCode::Mul;
      break;
    case ExprKind::Div:
      instr.opcode = OpCode::Div;
      break;
    case ExprKind::MatMul:
      instr.opcode = OpCode::MatMul;
      break;
    case ExprKind::Pow:
      instr.opcode = OpCode::Pow;
      break;
    case ExprKind::Mod:
      instr.opcode = OpCode::Mod;
      break;
    case ExprKind::FloorDiv:
      instr.opcode = OpCode::FloorDiv;
      break;
    case ExprKind::BitAnd:
      instr.opcode = OpCode::BitAnd;
      break;
    case ExprKind::BitOr:
      instr.opcode = OpCode::BitOr;
      break;
    case ExprKind::BitXor:
      instr.opcode = OpCode::BitXor;
      break;
    case ExprKind::LShift:
      instr.opcode = OpCode::LShift;
      break;
    case ExprKind::RShift:
      instr.opcode = OpCode::RShift;
      break;
    case ExprKind::Less:
      instr.opcode = OpCode::Less;
      break;
    case ExprKind::LessEqual:
      instr.opcode = OpCode::LessEqual;
      break;
    case ExprKind::Greater:
      instr.opcode = OpCode::Greater;
      break;
    case ExprKind::GreaterEqual:
      instr.opcode = OpCode::GreaterEqual;
      break;
    case ExprKind::Equal:
      instr.opcode = OpCode::Equal;
      break;
    case ExprKind::NotEqual:
      instr.opcode = OpCode::NotEqual;
      break;
    default:
      instr.opcode = OpCode::Literal;
      instr.literal = float_value(0.0);
      break;
  }
  instr.dst = compiled.register_count++;
  instr.src_a = lhs;
  instr.src_b = rhs;
  compiled.instructions.push_back(instr);
  return instr.dst;
}

Graph::CompiledInputProgram Graph::compile_input_program(
  const std::vector<ExprSpecPtr> & exprs,
  unsigned int input_count,
  const RuntimeState & runtime) const
{
  CompiledInputProgram compiled;
  compiled.result_registers.assign(input_count, 0);

  for (unsigned int input_id = 0; input_id < input_count; ++input_id)
  {
    const ExprSpecPtr expr = input_id < exprs.size() ? exprs[input_id] : nullptr;
    ExprSpecPtr inlined = egress_expr_inline::inline_functions(expr);
    compiled.result_registers[input_id] = compile_expr_node(inlined, compiled, runtime);
  }

  return compiled;
}

void Graph::eval_input_program(
  const RuntimeState & runtime,
  const CompiledInputProgram & program,
  std::vector<Value> & registers,
  std::vector<Value> & inputs) const
{
  if (program.instructions.empty())
  {
    for (auto & input : inputs)
    {
      input = float_value(0.0);
    }
    return;
  }

  if (registers.size() < program.register_count)
  {
    registers.resize(program.register_count, float_value(0.0));
  }

  for (const auto & instr : program.instructions)
  {
    eval_instruction(runtime, instr, registers.data());
  }

  const unsigned int input_limit = std::min(
    static_cast<unsigned int>(inputs.size()),
    static_cast<unsigned int>(program.result_registers.size()));

  for (unsigned int input_id = 0; input_id < input_limit; ++input_id)
  {
    inputs[input_id] = registers[program.result_registers[input_id]];
  }
}
