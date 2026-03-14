#pragma once

#ifdef EGRESS_LLVM_ORC_JIT

bool Module::supports_numeric_jit_expr_kind(ExprKind kind) const
{
  switch (kind)
  {
    case ExprKind::Literal:
    case ExprKind::InputValue:
    case ExprKind::RegisterValue:
    case ExprKind::NestedValue:
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
    case ExprKind::Not:
    case ExprKind::Less:
    case ExprKind::LessEqual:
    case ExprKind::Greater:
    case ExprKind::GreaterEqual:
    case ExprKind::Equal:
    case ExprKind::NotEqual:
    case ExprKind::Add:
    case ExprKind::Sub:
    case ExprKind::Mul:
    case ExprKind::Div:
    case ExprKind::MatMul:
    case ExprKind::Pow:
    case ExprKind::Mod:
    case ExprKind::FloorDiv:
    case ExprKind::BitAnd:
    case ExprKind::BitOr:
    case ExprKind::BitXor:
    case ExprKind::LShift:
    case ExprKind::RShift:
    case ExprKind::Abs:
    case ExprKind::Clamp:
    case ExprKind::Index:
    case ExprKind::Log:
    case ExprKind::Sin:
    case ExprKind::Neg:
    case ExprKind::BitNot:
    case ExprKind::ArrayPack:
      return true;
    default:
      return false;
  }
}

void Module::assign_scalar_numeric_value(Value & dst, double value)
{
  const double clamped = clamp_output_scalar(value);
  dst.type = ValueType::Float;
  dst.int_value = static_cast<int64_t>(clamped);
  dst.float_value = clamped;
  dst.bool_value = clamped != 0.0;
  dst.array_items.clear();
  dst.matrix_items.clear();
  dst.matrix_rows = 0;
  dst.matrix_cols = 0;
}

void Module::assign_numeric_value_to(
  Value & dst,
  const NumericOutputInfo & info,
  uint32_t scalar_register,
  const std::vector<double> & numeric_temps,
  const std::vector<std::vector<double>> & numeric_array_storage)
{
  switch (static_cast<NumericValueKind>(info.kind))
  {
    case NumericValueKind::Scalar:
      assign_scalar_numeric_value(dst, numeric_temps[scalar_register]);
      return;
    case NumericValueKind::Array:
    {
      const auto & values = numeric_array_storage[info.array_slot];
      if (dst.type != ValueType::Array || dst.array_items.size() != values.size())
      {
        std::vector<Value> items(values.size(), expr::float_value(0.0));
        dst = expr::array_value(std::move(items));
      }
      for (std::size_t i = 0; i < values.size(); ++i)
      {
        assign_scalar_numeric_value(dst.array_items[i], values[i]);
      }
      return;
    }
    case NumericValueKind::Matrix:
    {
      const auto & values = numeric_array_storage[info.array_slot];
      if (dst.type != ValueType::Matrix ||
          dst.matrix_rows != info.matrix_rows ||
          dst.matrix_cols != info.matrix_cols ||
          dst.matrix_items.size() != values.size())
      {
        std::vector<Value> items(values.size(), expr::float_value(0.0));
        dst = expr::matrix_value(info.matrix_rows, info.matrix_cols, std::move(items));
      }
      for (std::size_t i = 0; i < values.size(); ++i)
      {
        assign_scalar_numeric_value(dst.matrix_items[i], values[i]);
      }
      return;
    }
  }
}

bool Module::value_to_scalar_double(const Value & value, double & out)
{
  if (value.type == ValueType::Array || value.type == ValueType::Matrix)
  {
    return false;
  }
  out = to_float64(value);
  return true;
}

bool Module::add_array_values_to_jit_table(const std::vector<Value> & values, uint32_t & out_slot)
{
  std::vector<double> numeric_values;
  numeric_values.reserve(values.size());
  for (const Value & item : values)
  {
    double scalar = 0.0;
    if (!value_to_scalar_double(item, scalar))
    {
      return false;
    }
    numeric_values.push_back(scalar);
  }

  out_slot = static_cast<uint32_t>(numeric_array_storage_.size());
  numeric_array_storage_.push_back(std::move(numeric_values));
  return true;
}

bool Module::add_matrix_values_to_jit_table(const Value & value, uint32_t & out_slot)
{
  if (value.type != ValueType::Matrix)
  {
    return false;
  }
  return add_array_values_to_jit_table(value.matrix_items, out_slot);
}

uint32_t Module::allocate_array_slot_with_size(std::size_t size)
{
  uint32_t slot = static_cast<uint32_t>(numeric_array_storage_.size());
  numeric_array_storage_.push_back(std::vector<double>(size, 0.0));
  return slot;
}

bool Module::configure_numeric_inputs_for_jit(const std::vector<Value> & current_inputs)
{
  numeric_input_info_.assign(current_inputs.size(), NumericInputInfo{});

  for (unsigned int input_id = 0; input_id < current_inputs.size(); ++input_id)
  {
    const Value & input = current_inputs[input_id];
    if (input.type == ValueType::Matrix)
    {
      return false;
    }
    if (input.type != ValueType::Array)
    {
      continue;
    }

    uint32_t array_slot = 0;
    if (!add_array_values_to_jit_table(input.array_items, array_slot))
    {
      return false;
    }

    NumericInputInfo & info = numeric_input_info_[input_id];
    info.is_scalar = false;
    info.array_slot = array_slot;
    info.array_size = static_cast<uint32_t>(input.array_items.size());
  }

  return true;
}

bool Module::numeric_input_layout_matches(const std::vector<Value> & current_inputs) const
{
  if (numeric_input_info_.size() != current_inputs.size())
  {
    return false;
  }

  for (unsigned int input_id = 0; input_id < current_inputs.size(); ++input_id)
  {
    const Value & input = current_inputs[input_id];
    const NumericInputInfo & info = numeric_input_info_[input_id];
    if (input.type == ValueType::Matrix)
    {
      return false;
    }
    if (input.type == ValueType::Array)
    {
      if (info.is_scalar || info.array_size != input.array_items.size())
      {
        return false;
      }
      continue;
    }
    if (!info.is_scalar)
    {
      return false;
    }
  }

  return true;
}

bool Module::sync_numeric_inputs_from_values()
{
  if (numeric_inputs_.size() < inputs.size())
  {
    numeric_inputs_.assign(inputs.size(), 0.0);
  }

  if (numeric_input_info_.size() != inputs.size())
  {
    return false;
  }

  for (unsigned int input_id = 0; input_id < inputs.size(); ++input_id)
  {
    const Value & input = inputs[input_id];
    const NumericInputInfo & info = numeric_input_info_[input_id];
    if (info.is_scalar)
    {
      if (input.type == ValueType::Array || input.type == ValueType::Matrix)
      {
        return false;
      }
      numeric_inputs_[input_id] = expr::to_float64(input);
      continue;
    }

    if (input.type != ValueType::Array || input.array_items.size() != info.array_size)
    {
      return false;
    }
    if (info.array_slot >= numeric_array_storage_.size())
    {
      return false;
    }

    auto & dst = numeric_array_storage_[info.array_slot];
    if (dst.size() != info.array_size)
    {
      return false;
    }

    for (unsigned int item_id = 0; item_id < input.array_items.size(); ++item_id)
    {
      double scalar = 0.0;
      if (!value_to_scalar_double(input.array_items[item_id], scalar))
      {
        return false;
      }
      dst[item_id] = scalar;
    }
  }

  return true;
}

bool Module::build_numeric_program(const std::vector<Value> & current_inputs, egress_jit::NumericProgram & numeric_program)
{
  if (program_.register_count == 0)
  {
    return false;
  }

  numeric_program.instructions.clear();
  numeric_program.register_count = program_.register_count;
  numeric_array_storage_.clear();
  register_scalar_mask_.assign(registers_.size(), true);
  register_array_slot_.assign(registers_.size(), 0);
  array_register_targets_.assign(registers_.size(), -1);
  array_register_can_swap_.assign(registers_.size(), false);
  numeric_output_info_.clear();

  if (!configure_numeric_inputs_for_jit(current_inputs))
  {
    return false;
  }

  std::vector<NumericRegInfo> reg_info(program_.register_count);

  for (unsigned int reg_slot = 0; reg_slot < registers_.size(); ++reg_slot)
  {
    const Value & reg = registers_[reg_slot];
    if (reg.type == ValueType::Array)
    {
      uint32_t array_slot = 0;
      if (!add_array_values_to_jit_table(reg.array_items, array_slot))
      {
        return false;
      }
      register_scalar_mask_[reg_slot] = false;
      register_array_slot_[reg_slot] = array_slot;
    }
    else if (reg.type == ValueType::Matrix)
    {
      return false;
    }
  }

  for (const Instr & instr : program_.instructions)
  {
    if (!supports_numeric_jit_expr_kind(instr.kind))
    {
      return false;
    }

    egress_jit::NumericInstr jit_instr;
    jit_instr.dst = instr.dst;
    jit_instr.src_a = instr.src_a;
    jit_instr.src_b = instr.src_b;
    jit_instr.src_c = instr.src_c;
    jit_instr.slot_id = instr.slot_id;

    bool emit_instruction = true;
    bool require_scalar_inputs = true;

    switch (instr.kind)
    {
      case ExprKind::Literal:
      {
        if (instr.literal.type == ValueType::Array)
        {
          uint32_t array_slot = 0;
          if (!add_array_values_to_jit_table(instr.literal.array_items, array_slot))
          {
            return false;
          }
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = array_slot;
          reg_info[instr.dst].array_size = static_cast<uint32_t>(instr.literal.array_items.size());
          emit_instruction = false;
        }
        else if (instr.literal.type == ValueType::Matrix)
        {
          uint32_t array_slot = 0;
          if (!add_matrix_values_to_jit_table(instr.literal, array_slot))
          {
            return false;
          }
          reg_info[instr.dst].kind = NumericValueKind::Matrix;
          reg_info[instr.dst].array_slot = array_slot;
          reg_info[instr.dst].array_size = static_cast<uint32_t>(instr.literal.matrix_items.size());
          reg_info[instr.dst].matrix_rows = static_cast<uint32_t>(instr.literal.matrix_rows);
          reg_info[instr.dst].matrix_cols = static_cast<uint32_t>(instr.literal.matrix_cols);
          emit_instruction = false;
        }
        else
        {
          jit_instr.op = egress_jit::NumericOp::Literal;
          jit_instr.literal = to_float64(instr.literal);
          reg_info[instr.dst].kind = NumericValueKind::Scalar;
          reg_info[instr.dst].scalar_is_constant = true;
          reg_info[instr.dst].scalar_constant = jit_instr.literal;
        }
        break;
      }
      case ExprKind::InputValue:
        if (instr.slot_id >= numeric_input_info_.size())
        {
          return false;
        }
        if (numeric_input_info_[instr.slot_id].is_scalar)
        {
          jit_instr.op = egress_jit::NumericOp::InputValue;
          reg_info[instr.dst].kind = NumericValueKind::Scalar;
        }
        else
        {
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = numeric_input_info_[instr.slot_id].array_slot;
          reg_info[instr.dst].array_size = numeric_input_info_[instr.slot_id].array_size;
          emit_instruction = false;
        }
        break;
      case ExprKind::RegisterValue:
        if (instr.slot_id >= register_scalar_mask_.size())
        {
          return false;
        }
        if (!register_scalar_mask_[instr.slot_id])
        {
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = register_array_slot_[instr.slot_id];
          reg_info[instr.dst].array_size = static_cast<uint32_t>(numeric_array_storage_[register_array_slot_[instr.slot_id]].size());
          emit_instruction = false;
        }
        else
        {
          jit_instr.op = egress_jit::NumericOp::RegisterValue;
          reg_info[instr.dst].kind = NumericValueKind::Scalar;
        }
        break;
      case ExprKind::SampleRate:
        jit_instr.op = egress_jit::NumericOp::SampleRate;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::SampleIndex:
        jit_instr.op = egress_jit::NumericOp::SampleIndex;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::ArrayPack:
      {
        bool all_constant = true;
        std::vector<Value> packed_values;
        packed_values.reserve(instr.args.size());
        for (uint32_t src : instr.args)
        {
          if (src >= reg_info.size() ||
              reg_info[src].kind != NumericValueKind::Scalar)
          {
            return false;
          }
          if (!reg_info[src].scalar_is_constant)
          {
            all_constant = false;
          }
          packed_values.push_back(float_value(reg_info[src].scalar_constant));
        }

        const uint32_t array_size = static_cast<uint32_t>(instr.args.size());
        uint32_t array_slot = 0;
        if (all_constant)
        {
          if (!add_array_values_to_jit_table(packed_values, array_slot))
          {
            return false;
          }
          emit_instruction = false;
        }
        else
        {
          array_slot = allocate_array_slot_with_size(array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayPack;
          jit_instr.dst = array_slot;
          jit_instr.args = instr.args;
          require_scalar_inputs = false;
        }

        reg_info[instr.dst].kind = NumericValueKind::Array;
        reg_info[instr.dst].array_slot = array_slot;
        reg_info[instr.dst].array_size = array_size;
        break;
      }
      case ExprKind::Index:
        if (instr.src_a >= reg_info.size() || instr.src_b >= reg_info.size())
        {
          return false;
        }
        if (reg_info[instr.src_a].kind == NumericValueKind::Scalar ||
            reg_info[instr.src_b].kind != NumericValueKind::Scalar)
        {
          return false;
        }
        if (reg_info[instr.src_a].kind == NumericValueKind::Matrix)
        {
          return false;
        }
        jit_instr.op = egress_jit::NumericOp::IndexArray;
        jit_instr.src_a = instr.src_b;
        jit_instr.slot_id = reg_info[instr.src_a].array_slot;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Not:
        jit_instr.op = egress_jit::NumericOp::Not;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Less:
      case ExprKind::LessEqual:
      case ExprKind::Greater:
      case ExprKind::GreaterEqual:
      case ExprKind::Equal:
      case ExprKind::NotEqual:
      {
        const NumericRegInfo & lhs = reg_info[instr.src_a];
        const NumericRegInfo & rhs = reg_info[instr.src_b];
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Scalar)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          switch (instr.kind)
          {
            case ExprKind::Less:
              jit_instr.op = egress_jit::NumericOp::ArrayLessScalar;
              break;
            case ExprKind::LessEqual:
              jit_instr.op = egress_jit::NumericOp::ArrayLessEqualScalar;
              break;
            case ExprKind::Greater:
              jit_instr.op = egress_jit::NumericOp::ArrayGreaterScalar;
              break;
            case ExprKind::GreaterEqual:
              jit_instr.op = egress_jit::NumericOp::ArrayGreaterEqualScalar;
              break;
            case ExprKind::Equal:
              jit_instr.op = egress_jit::NumericOp::ArrayEqualScalar;
              break;
            case ExprKind::NotEqual:
              jit_instr.op = egress_jit::NumericOp::ArrayNotEqualScalar;
              break;
            default:
              break;
          }
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = instr.src_b;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind != NumericValueKind::Scalar || rhs.kind != NumericValueKind::Scalar)
        {
          return false;
        }
        switch (instr.kind)
        {
          case ExprKind::Less:
            jit_instr.op = egress_jit::NumericOp::Less;
            break;
          case ExprKind::LessEqual:
            jit_instr.op = egress_jit::NumericOp::LessEqual;
            break;
          case ExprKind::Greater:
            jit_instr.op = egress_jit::NumericOp::Greater;
            break;
          case ExprKind::GreaterEqual:
            jit_instr.op = egress_jit::NumericOp::GreaterEqual;
            break;
          case ExprKind::Equal:
            jit_instr.op = egress_jit::NumericOp::Equal;
            break;
          case ExprKind::NotEqual:
            jit_instr.op = egress_jit::NumericOp::NotEqual;
            break;
          default:
            break;
        }
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
      case ExprKind::Add:
      {
        const NumericRegInfo & lhs = reg_info[instr.src_a];
        const NumericRegInfo & rhs = reg_info[instr.src_b];
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Array)
        {
          if (lhs.array_size != rhs.array_size)
          {
            return false;
          }
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayAdd;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = rhs.array_slot;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Scalar)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayAddScalar;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = instr.src_b;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind == NumericValueKind::Scalar && rhs.kind == NumericValueKind::Array)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(rhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayAddScalar;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = rhs.array_slot;
          jit_instr.src_b = instr.src_a;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = rhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Scalar)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayDivScalar;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = instr.src_b;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind != NumericValueKind::Scalar || rhs.kind != NumericValueKind::Scalar)
        {
          return false;
        }
        jit_instr.op = egress_jit::NumericOp::Add;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
      case ExprKind::Sub:
      {
        const NumericRegInfo & lhs = reg_info[instr.src_a];
        const NumericRegInfo & rhs = reg_info[instr.src_b];
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Array)
        {
          if (lhs.array_size != rhs.array_size)
          {
            return false;
          }
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArraySub;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = rhs.array_slot;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind != NumericValueKind::Scalar || rhs.kind != NumericValueKind::Scalar)
        {
          return false;
        }
        jit_instr.op = egress_jit::NumericOp::Sub;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
      case ExprKind::Mul:
      {
        const NumericRegInfo & lhs = reg_info[instr.src_a];
        const NumericRegInfo & rhs = reg_info[instr.src_b];
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Array)
        {
          if (lhs.array_size != rhs.array_size)
          {
            return false;
          }
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayMul;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = rhs.array_slot;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Scalar)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayMulScalar;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = instr.src_b;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind == NumericValueKind::Scalar && rhs.kind == NumericValueKind::Array)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(rhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayMulScalar;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = rhs.array_slot;
          jit_instr.src_b = instr.src_a;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = rhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind != NumericValueKind::Scalar || rhs.kind != NumericValueKind::Scalar)
        {
          return false;
        }
        jit_instr.op = egress_jit::NumericOp::Mul;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
      case ExprKind::Div:
      {
        const NumericRegInfo & lhs = reg_info[instr.src_a];
        const NumericRegInfo & rhs = reg_info[instr.src_b];
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Array)
        {
          if (lhs.array_size != rhs.array_size)
          {
            return false;
          }
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayDiv;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = rhs.array_slot;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Scalar)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayDivScalar;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = instr.src_b;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind != NumericValueKind::Scalar || rhs.kind != NumericValueKind::Scalar)
        {
          return false;
        }
        jit_instr.op = egress_jit::NumericOp::Div;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
      case ExprKind::MatMul:
      {
        const NumericRegInfo & lhs = reg_info[instr.src_a];
        const NumericRegInfo & rhs = reg_info[instr.src_b];
        if (lhs.kind != NumericValueKind::Matrix || rhs.kind != NumericValueKind::Array)
        {
          return false;
        }
        if (lhs.matrix_cols != rhs.array_size)
        {
          return false;
        }
        const uint32_t dst_slot = allocate_array_slot_with_size(lhs.matrix_rows);
        jit_instr.op = egress_jit::NumericOp::MatMul;
        jit_instr.dst = dst_slot;
        jit_instr.src_a = lhs.array_slot;
        jit_instr.src_b = rhs.array_slot;
        jit_instr.src_c = lhs.matrix_rows;
        jit_instr.slot_id = lhs.matrix_cols;
        reg_info[instr.dst].kind = NumericValueKind::Array;
        reg_info[instr.dst].array_slot = dst_slot;
        reg_info[instr.dst].array_size = lhs.matrix_rows;
        require_scalar_inputs = false;
        break;
      }
      case ExprKind::Pow:
        jit_instr.op = egress_jit::NumericOp::Pow;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Mod:
      {
        const NumericRegInfo & lhs = reg_info[instr.src_a];
        const NumericRegInfo & rhs = reg_info[instr.src_b];
        if (lhs.kind == NumericValueKind::Array && rhs.kind == NumericValueKind::Scalar)
        {
          const uint32_t dst_slot = allocate_array_slot_with_size(lhs.array_size);
          jit_instr.op = egress_jit::NumericOp::ArrayModScalar;
          jit_instr.dst = dst_slot;
          jit_instr.src_a = lhs.array_slot;
          jit_instr.src_b = instr.src_b;
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = dst_slot;
          reg_info[instr.dst].array_size = lhs.array_size;
          require_scalar_inputs = false;
          break;
        }
        if (lhs.kind != NumericValueKind::Scalar || rhs.kind != NumericValueKind::Scalar)
        {
          return false;
        }
        jit_instr.op = egress_jit::NumericOp::Mod;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
      case ExprKind::FloorDiv:
        jit_instr.op = egress_jit::NumericOp::FloorDiv;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::BitAnd:
        jit_instr.op = egress_jit::NumericOp::BitAnd;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::BitOr:
        jit_instr.op = egress_jit::NumericOp::BitOr;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::BitXor:
        jit_instr.op = egress_jit::NumericOp::BitXor;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::LShift:
        jit_instr.op = egress_jit::NumericOp::LShift;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::RShift:
        jit_instr.op = egress_jit::NumericOp::RShift;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Abs:
        jit_instr.op = egress_jit::NumericOp::Abs;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Clamp:
        jit_instr.op = egress_jit::NumericOp::Clamp;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Log:
        jit_instr.op = egress_jit::NumericOp::Log;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Sin:
        jit_instr.op = egress_jit::NumericOp::Sin;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::Neg:
        jit_instr.op = egress_jit::NumericOp::Neg;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::BitNot:
        jit_instr.op = egress_jit::NumericOp::BitNot;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      default:
        return false;
    }

    if (emit_instruction)
    {
      const bool preserve_constant =
        instr.kind == ExprKind::Literal &&
        reg_info[instr.dst].kind == NumericValueKind::Scalar &&
        reg_info[instr.dst].scalar_is_constant;

      if (require_scalar_inputs &&
          instr.kind != ExprKind::Literal &&
          instr.kind != ExprKind::InputValue &&
          instr.kind != ExprKind::RegisterValue &&
          instr.kind != ExprKind::SampleRate &&
          instr.kind != ExprKind::SampleIndex &&
          instr.kind != ExprKind::Index)
      {
        if (instr.src_a >= reg_info.size() || reg_info[instr.src_a].kind != NumericValueKind::Scalar)
        {
          return false;
        }
        if (egress_module_detail::is_local_binary(instr.kind))
        {
          if (instr.src_b >= reg_info.size() || reg_info[instr.src_b].kind != NumericValueKind::Scalar)
          {
            return false;
          }
        }
        if (egress_module_detail::is_local_ternary(instr.kind))
        {
          if (instr.src_b >= reg_info.size() || reg_info[instr.src_b].kind != NumericValueKind::Scalar ||
              instr.src_c >= reg_info.size() || reg_info[instr.src_c].kind != NumericValueKind::Scalar)
          {
            return false;
          }
        }
      }

      if (!preserve_constant)
      {
        reg_info[instr.dst].scalar_is_constant = false;
        reg_info[instr.dst].scalar_constant = 0.0;
      }
      numeric_program.instructions.push_back(jit_instr);
    }
  }

  numeric_output_info_.reserve(program_.output_targets.size());
  for (uint32_t output_reg : program_.output_targets)
  {
    if (output_reg >= reg_info.size())
    {
      return false;
    }
    NumericOutputInfo output_info;
    output_info.kind = static_cast<uint8_t>(reg_info[output_reg].kind);
    output_info.array_slot = reg_info[output_reg].array_slot;
    output_info.matrix_rows = reg_info[output_reg].matrix_rows;
    output_info.matrix_cols = reg_info[output_reg].matrix_cols;
    numeric_output_info_.push_back(output_info);
  }

  for (unsigned int reg_slot = 0; reg_slot < program_.register_targets.size(); ++reg_slot)
  {
    const int32_t target = program_.register_targets[reg_slot];
    if (target < 0)
    {
      continue;
    }

    if (static_cast<std::size_t>(target) >= reg_info.size())
    {
      return false;
    }

    if (register_scalar_mask_[reg_slot])
    {
      if (reg_info[target].kind != NumericValueKind::Scalar)
      {
        return false;
      }
      continue;
    }

    if (reg_info[target].kind != NumericValueKind::Array)
    {
      return false;
    }

    const uint32_t dst_slot = register_array_slot_[reg_slot];
    if (dst_slot >= numeric_array_storage_.size())
    {
      return false;
    }
    if (numeric_array_storage_[dst_slot].size() != reg_info[target].array_size)
    {
      return false;
    }

    array_register_targets_[reg_slot] = static_cast<int32_t>(reg_info[target].array_slot);
  }

  std::vector<uint32_t> array_target_use_counts(numeric_array_storage_.size(), 0);
  for (unsigned int reg_slot = 0; reg_slot < array_register_targets_.size(); ++reg_slot)
  {
    const int32_t src_slot = array_register_targets_[reg_slot];
    if (src_slot < 0)
    {
      continue;
    }
    if (static_cast<std::size_t>(src_slot) >= array_target_use_counts.size())
    {
      return false;
    }
    ++array_target_use_counts[static_cast<std::size_t>(src_slot)];
  }

  for (unsigned int reg_slot = 0; reg_slot < array_register_targets_.size(); ++reg_slot)
  {
    const int32_t src_slot = array_register_targets_[reg_slot];
    if (src_slot < 0 || register_scalar_mask_[reg_slot])
    {
      continue;
    }
    const uint32_t dst_slot = register_array_slot_[reg_slot];
    if (dst_slot >= numeric_array_storage_.size() ||
        static_cast<std::size_t>(src_slot) >= numeric_array_storage_.size())
    {
      return false;
    }
    array_register_can_swap_[reg_slot] =
      dst_slot != static_cast<uint32_t>(src_slot) &&
      array_target_use_counts[static_cast<std::size_t>(src_slot)] == 1;
  }

  return true;
}

void Module::initialize_numeric_jit(const std::vector<Value> & current_inputs)
{
  jit_kernel_ = nullptr;
#ifdef EGRESS_PROFILE
  numeric_jit_instruction_count_ = 0;

#endif

  auto & jit = egress_jit::OrcJitEngine::instance();
  if (!jit.available())
  {
    jit_status_ = jit.init_error();
    return;
  }

  egress_jit::NumericProgram numeric_program;
  if (!build_numeric_program(current_inputs, numeric_program))
  {
    jit_status_ = "numeric compatibility check failed";
    return;
  }

#ifdef EGRESS_PROFILE
  numeric_jit_instruction_count_ = static_cast<uint64_t>(numeric_program.instructions.size());
#endif

  auto kernel_or_err = jit.compile_numeric_program(numeric_program, "egress_udm_kernel");
  if (!kernel_or_err)
  {
    jit_status_ = llvm::toString(kernel_or_err.takeError());
    return;
  }

  jit_kernel_ = *kernel_or_err;
  numeric_temps_.assign(program_.register_count, 0.0);
  numeric_registers_.resize(registers_.size(), 0.0);
  numeric_next_registers_.resize(registers_.size(), 0.0);
  numeric_array_ptrs_.resize(numeric_array_storage_.size(), nullptr);
  numeric_array_sizes_.resize(numeric_array_storage_.size(), 0);
  for (std::size_t i = 0; i < numeric_array_storage_.size(); ++i)
  {
    numeric_array_ptrs_[i] = numeric_array_storage_[i].empty() ? nullptr : numeric_array_storage_[i].data();
    numeric_array_sizes_[i] = static_cast<uint64_t>(numeric_array_storage_[i].size());
  }
  for (unsigned int i = 0; i < registers_.size(); ++i)
  {
    numeric_registers_[i] = register_scalar_mask_[i] ? to_float64(registers_[i]) : 0.0;
  }
  jit_status_ = "numeric JIT active";
}

void Module::ensure_numeric_jit_current()
{
  if (numeric_input_layout_matches(inputs))
  {
    return;
  }

  initialize_numeric_jit(inputs);
}

#endif
