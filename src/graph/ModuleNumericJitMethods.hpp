#pragma once

#ifdef EGRESS_LLVM_ORC_JIT

namespace
{
inline bool copy_numeric_scalar_values(
  const std::vector<Value> & values,
  AggregateScalarType aggregate_scalar_type,
  std::vector<double> & dst)
{
  if (aggregate_scalar_type == AggregateScalarType::NonScalar)
  {
    return false;
  }
  if (dst.size() != values.size())
  {
    return false;
  }

  for (std::size_t item_id = 0; item_id < values.size(); ++item_id)
  {
    const Value & item = values[item_id];
    if (aggregate_scalar_type == AggregateScalarType::Bool)
    {
      dst[item_id] = item.bool_value ? 1.0 : 0.0;
      continue;
    }
    if (aggregate_scalar_type == AggregateScalarType::Int)
    {
      dst[item_id] = static_cast<double>(item.int_value);
      continue;
    }
    if (aggregate_scalar_type == AggregateScalarType::Float)
    {
      dst[item_id] = item.float_value;
      continue;
    }

    if (item.type == ValueType::Array || item.type == ValueType::Matrix)
    {
      return false;
    }
    dst[item_id] = expr::to_float64(item);
  }

  return true;
}

inline bool copy_numeric_aggregate_value(const Value & value, std::vector<double> & dst)
{
  if (!expr::aggregate_has_numeric_scalars(value))
  {
    return false;
  }

  if (value.type == ValueType::Array)
  {
    return copy_numeric_scalar_values(value.array_items, value.aggregate_scalar_type, dst);
  }
  if (value.type == ValueType::Matrix)
  {
    return copy_numeric_scalar_values(value.matrix_items, value.aggregate_scalar_type, dst);
  }
  return false;
}
}

bool Module::supports_numeric_jit_expr_kind(ExprKind kind) const
{
  switch (kind)
  {
    case ExprKind::Literal:
    case ExprKind::InputValue:
    case ExprKind::RegisterValue:
    case ExprKind::NestedValue:
    case ExprKind::DelayValue:
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
    case ExprKind::Select:
    case ExprKind::Index:
    case ExprKind::ArraySet:
    case ExprKind::Log:
    case ExprKind::Sin:
    case ExprKind::Neg:
    case ExprKind::BitNot:
    case ExprKind::ArrayPack:
    case ExprKind::SmoothedParam:
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
  dst.aggregate_scalar_type = AggregateScalarType::None;
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
      dst.aggregate_scalar_type = AggregateScalarType::Float;
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
      dst.aggregate_scalar_type = AggregateScalarType::Float;
      return;
    }
  }
}

Module::NumericValueRef Module::make_numeric_value_ref(
  const NumericOutputInfo & info,
  uint32_t scalar_register)
{
  NumericValueRef ref;
  ref.kind = static_cast<NumericValueKind>(info.kind);
  ref.scalar_register = scalar_register;
  ref.array_slot = info.array_slot;
  switch (ref.kind)
  {
    case NumericValueKind::Scalar:
      return ref;
    case NumericValueKind::Array:
      if (info.array_slot < std::numeric_limits<uint32_t>::max())
      {
        ref.array_size = 0;
      }
      return ref;
    case NumericValueKind::Matrix:
      ref.matrix_rows = info.matrix_rows;
      ref.matrix_cols = info.matrix_cols;
      ref.array_size = info.matrix_rows * info.matrix_cols;
      return ref;
  }
  return ref;
}

bool Module::try_get_numeric_output_ref(unsigned int output_id, NumericValueRef & out) const
{
  if (jit_kernel_ == nullptr ||
      output_id >= numeric_output_info_.size() ||
      output_id >= program_.output_targets.size())
  {
    return false;
  }

  out = make_numeric_value_ref(numeric_output_info_[output_id], program_.output_targets[output_id]);
  if (out.kind == NumericValueKind::Array || out.kind == NumericValueKind::Matrix)
  {
    if (out.array_slot >= numeric_array_storage_.size())
    {
      return false;
    }
    out.array_size = static_cast<uint32_t>(numeric_array_storage_[out.array_slot].size());
  }
  return true;
}

const std::vector<double> * Module::try_get_numeric_output_array_values(unsigned int output_id, bool previous) const
{
  if (previous)
  {
    if (output_id >= numeric_prev_output_array_mask_.size() ||
        output_id >= numeric_prev_output_arrays_.size() ||
        !numeric_prev_output_array_mask_[output_id])
    {
      return nullptr;
    }
    return &numeric_prev_output_arrays_[output_id];
  }

  NumericValueRef ref;
  if (!try_get_numeric_output_ref(output_id, ref) ||
      ref.kind != NumericValueKind::Array ||
      ref.array_slot >= numeric_array_storage_.size())
  {
    return nullptr;
  }
  return &numeric_array_storage_[ref.array_slot];
}

bool Module::try_get_numeric_scalar_output(unsigned int output_id, bool previous, double & out) const
{
  const auto & scalar_mask = previous ? numeric_prev_output_scalar_mask_ : numeric_output_scalar_mask_;
  const auto & scalar_values = previous ? numeric_prev_output_scalars_ : numeric_output_scalars_;
  if (output_id >= scalar_mask.size() ||
      output_id >= scalar_values.size() ||
      !scalar_mask[output_id])
  {
    return false;
  }
  out = scalar_values[output_id];
  return true;
}

const Value & Module::materialize_output_value(unsigned int output_id, bool previous)
{
  auto & destinations = previous ? prev_outputs : outputs;
  if (output_id >= destinations.size())
  {
    static Value zero = expr::float_value(0.0);
    return zero;
  }

  if (!previous)
  {
    NumericValueRef ref;
    if (try_get_numeric_output_ref(output_id, ref) &&
        output_id < numeric_output_info_.size() &&
        output_id < program_.output_targets.size())
    {
      assign_numeric_value_to(
        destinations[output_id],
        numeric_output_info_[output_id],
        program_.output_targets[output_id],
        numeric_temps_,
        numeric_array_storage_);
      return destinations[output_id];
    }
  }
  else
  {
    double scalar = 0.0;
    if (try_get_numeric_scalar_output(output_id, true, scalar))
    {
      assign_scalar_numeric_value(destinations[output_id], scalar);
      return destinations[output_id];
    }
    const auto * values = try_get_numeric_output_array_values(output_id, true);
    if (values != nullptr)
    {
      std::vector<Value> items(values->size(), expr::float_value(0.0));
      for (std::size_t item_id = 0; item_id < values->size(); ++item_id)
      {
        assign_scalar_numeric_value(items[item_id], (*values)[item_id]);
      }
      destinations[output_id] = expr::array_value(std::move(items));
      return destinations[output_id];
    }
  }

  return destinations[output_id];
}

void Module::capture_numeric_prev_array_outputs()
{
  if (numeric_prev_output_array_mask_.size() != outputs.size())
  {
    numeric_prev_output_array_mask_.assign(outputs.size(), false);
    numeric_prev_output_arrays_.assign(outputs.size(), {});
  }

  for (unsigned int output_id = 0; output_id < outputs.size(); ++output_id)
  {
    NumericValueRef ref;
    if (!try_get_numeric_output_ref(output_id, ref) ||
        ref.kind != NumericValueKind::Array ||
        ref.array_slot >= numeric_array_storage_.size())
    {
      numeric_prev_output_array_mask_[output_id] = false;
      if (output_id < numeric_prev_output_arrays_.size())
      {
        numeric_prev_output_arrays_[output_id].clear();
      }
      continue;
    }

    numeric_prev_output_array_mask_[output_id] = true;
    numeric_prev_output_arrays_[output_id] = numeric_array_storage_[ref.array_slot];
  }
}

void Module::capture_numeric_scalar_outputs(
  const CompiledProgram & compiled_program,
  const std::vector<NumericOutputInfo> & output_info,
  const std::vector<double> & temps,
  std::size_t start_output_id,
  std::size_t output_count)
{
  const std::size_t available = std::min(compiled_program.output_targets.size(), output_info.size());
  if (start_output_id >= available)
  {
    return;
  }
  const std::size_t end_output_id =
    output_count == std::numeric_limits<std::size_t>::max()
      ? available
      : std::min(available, start_output_id + output_count);
  if (numeric_output_scalar_mask_.size() < available)
  {
    numeric_output_scalar_mask_.assign(available, false);
    numeric_output_scalars_.assign(available, 0.0);
  }
  for (std::size_t output_id = start_output_id; output_id < end_output_id; ++output_id)
  {
    if (static_cast<NumericValueKind>(output_info[output_id].kind) != NumericValueKind::Scalar)
    {
      numeric_output_scalar_mask_[output_id] = false;
      continue;
    }
    const uint32_t scalar_register = compiled_program.output_targets[output_id];
    if (scalar_register >= temps.size())
    {
      numeric_output_scalar_mask_[output_id] = false;
      continue;
    }
    numeric_output_scalar_mask_[output_id] = true;
    numeric_output_scalars_[output_id] = Module::clamp_output_scalar(temps[scalar_register]);
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
  return add_array_values_to_jit_table(numeric_array_storage_, values, out_slot);
}

bool Module::add_array_value_to_jit_table(const Value & value, uint32_t & out_slot)
{
  return add_array_value_to_jit_table(numeric_array_storage_, value, out_slot);
}

bool Module::add_array_values_to_jit_table(
  std::vector<std::vector<double>> & array_storage,
  const std::vector<Value> & values,
  uint32_t & out_slot)
{
  std::vector<double> numeric_values;
  numeric_values.reserve(values.size());
  const AggregateScalarType aggregate_scalar_type = expr::infer_aggregate_scalar_type(values);
  if (aggregate_scalar_type == AggregateScalarType::NonScalar)
  {
    return false;
  }
  numeric_values.resize(values.size(), 0.0);
  if (!copy_numeric_scalar_values(values, aggregate_scalar_type, numeric_values))
  {
    return false;
  }

  out_slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::move(numeric_values));
  return true;
}

bool Module::add_array_value_to_jit_table(
  std::vector<std::vector<double>> & array_storage,
  const Value & value,
  uint32_t & out_slot)
{
  if (value.type != ValueType::Array || !expr::aggregate_has_numeric_scalars(value))
  {
    return false;
  }

  std::vector<double> numeric_values(value.array_items.size(), 0.0);
  if (!copy_numeric_aggregate_value(value, numeric_values))
  {
    return false;
  }

  out_slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::move(numeric_values));
  return true;
}

bool Module::add_matrix_values_to_jit_table(const Value & value, uint32_t & out_slot)
{
  return add_matrix_values_to_jit_table(numeric_array_storage_, value, out_slot);
}

bool Module::add_matrix_values_to_jit_table(
  std::vector<std::vector<double>> & array_storage,
  const Value & value,
  uint32_t & out_slot)
{
  if (value.type != ValueType::Matrix)
  {
    return false;
  }
  if (!expr::aggregate_has_numeric_scalars(value))
  {
    return false;
  }

  std::vector<double> numeric_values(value.matrix_items.size(), 0.0);
  if (!copy_numeric_aggregate_value(value, numeric_values))
  {
    return false;
  }

  out_slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::move(numeric_values));
  return true;
}

uint32_t Module::allocate_array_slot_with_size(std::size_t size)
{
  return allocate_array_slot_with_size(numeric_array_storage_, size);
}

uint32_t Module::allocate_array_slot_with_size(
  std::vector<std::vector<double>> & array_storage,
  std::size_t size)
{
  uint32_t slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::vector<double>(size, 0.0));
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
    if (!add_array_value_to_jit_table(input, array_slot))
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
#ifdef EGRESS_PROFILE
  const auto sync_start = std::chrono::steady_clock::now();
#endif
  if (numeric_inputs_.size() < inputs.size())
  {
    numeric_inputs_.assign(inputs.size(), 0.0);
  }

  if (numeric_input_info_.size() != inputs.size())
  {
    return false;
  }

  if (numeric_input_override_active_)
  {
    if (numeric_input_scalar_override_.size() < inputs.size())
    {
      numeric_input_override_active_ = false;
      return false;
    }
    for (unsigned int input_id = 0; input_id < inputs.size(); ++input_id)
    {
      if (!numeric_input_info_[input_id].is_scalar)
      {
        const NumericInputInfo & info = numeric_input_info_[input_id];
        if (info.array_slot >= numeric_array_storage_.size() ||
            info.array_slot >= numeric_array_ptrs_.size() ||
            info.array_slot >= numeric_array_sizes_.size() ||
            numeric_array_storage_[info.array_slot].size() != info.array_size)
        {
          numeric_input_override_active_ = false;
          return false;
        }
        continue;
      }
      numeric_inputs_[input_id] = numeric_input_scalar_override_[input_id];
    }
    numeric_input_override_active_ = false;
#ifdef EGRESS_PROFILE
    record_numeric_input_sync_profile(
      static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sync_start).count()));
#endif
    return true;
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

    if (!copy_numeric_aggregate_value(input, dst))
    {
      return false;
    }
  }

#ifdef EGRESS_PROFILE
  record_numeric_input_sync_profile(
    static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sync_start).count()));
#endif
  return true;
}

bool Module::try_set_direct_numeric_inputs(
  const NumericJitState & state,
  const CompiledProgram & compiled_program)
{
  if (jit_kernel_ == nullptr ||
      numeric_input_info_.size() != inputs.size() ||
      compiled_program.output_targets.size() != inputs.size() ||
      state.output_info.size() < inputs.size())
  {
    return false;
  }

  if (numeric_input_scalar_override_.size() < inputs.size())
  {
    numeric_input_scalar_override_.assign(inputs.size(), 0.0);
  }

  for (unsigned int input_id = 0; input_id < inputs.size(); ++input_id)
  {
    const NumericInputInfo & dst_info = numeric_input_info_[input_id];
    const NumericOutputInfo & src_info = state.output_info[input_id];
    const NumericValueKind src_kind = static_cast<NumericValueKind>(src_info.kind);
    if (dst_info.is_scalar)
    {
      if (src_kind != NumericValueKind::Scalar)
      {
        numeric_input_override_active_ = false;
        return false;
      }
      const uint32_t src_reg = compiled_program.output_targets[input_id];
      if (src_reg >= state.temps.size())
      {
        numeric_input_override_active_ = false;
        return false;
      }
      numeric_input_scalar_override_[input_id] = state.temps[src_reg];
      continue;
    }

    if (src_kind != NumericValueKind::Array ||
        src_info.array_slot >= state.array_storage.size() ||
        dst_info.array_slot >= numeric_array_storage_.size() ||
        dst_info.array_slot >= numeric_array_ptrs_.size() ||
        dst_info.array_slot >= numeric_array_sizes_.size())
    {
      numeric_input_override_active_ = false;
      return false;
    }

    const auto & src_values = state.array_storage[src_info.array_slot];
    if (src_values.size() != dst_info.array_size)
    {
      numeric_input_override_active_ = false;
      return false;
    }

    auto & dst_values = numeric_array_storage_[dst_info.array_slot];
    dst_values = src_values;
    numeric_array_ptrs_[dst_info.array_slot] = dst_values.empty() ? nullptr : dst_values.data();
    numeric_array_sizes_[dst_info.array_slot] = static_cast<uint64_t>(dst_values.size());
  }

  numeric_input_override_active_ = true;
  return true;
}

bool Module::configure_numeric_inputs_for_jit(
  NumericJitState & state,
  const std::vector<Value> & current_inputs)
{
  state.input_info.assign(current_inputs.size(), NumericInputInfo{});

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
    if (!add_array_value_to_jit_table(state.array_storage, input, array_slot))
    {
      return false;
    }

    NumericInputInfo & info = state.input_info[input_id];
    info.is_scalar = false;
    info.array_slot = array_slot;
    info.array_size = static_cast<uint32_t>(input.array_items.size());
  }

  return true;
}

bool Module::numeric_input_layout_matches(
  const NumericJitState & state,
  const std::vector<Value> & current_inputs) const
{
  if (state.input_info.size() != current_inputs.size())
  {
    return false;
  }

  for (unsigned int input_id = 0; input_id < current_inputs.size(); ++input_id)
  {
    const Value & input = current_inputs[input_id];
    const NumericInputInfo & info = state.input_info[input_id];
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

bool Module::sync_numeric_inputs_from_values(
  NumericJitState & state,
  const std::vector<Value> & current_inputs)
{
#ifdef EGRESS_PROFILE
  const auto sync_start = std::chrono::steady_clock::now();
#endif
  if (state.inputs.size() < current_inputs.size())
  {
    state.inputs.assign(current_inputs.size(), 0.0);
  }

  if (state.input_info.size() != current_inputs.size())
  {
    return false;
  }

  for (unsigned int input_id = 0; input_id < current_inputs.size(); ++input_id)
  {
    const Value & input = current_inputs[input_id];
    const NumericInputInfo & info = state.input_info[input_id];
    if (info.is_scalar)
    {
      if (input.type == ValueType::Array || input.type == ValueType::Matrix)
      {
        return false;
      }
      state.inputs[input_id] = expr::to_float64(input);
      continue;
    }

    if (input.type != ValueType::Array || input.array_items.size() != info.array_size)
    {
      return false;
    }
    if (info.array_slot >= state.array_storage.size())
    {
      return false;
    }

    auto & dst = state.array_storage[info.array_slot];
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

#ifdef EGRESS_PROFILE
  record_numeric_input_sync_profile(
    static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sync_start).count()));
#endif
  return true;
}

bool Module::sync_numeric_register_arrays_from_values(NumericJitState & state)
{
  for (unsigned int reg_id = 0; reg_id < registers_.size(); ++reg_id)
  {
    if (reg_id >= state.register_scalar_mask.size() || state.register_scalar_mask[reg_id])
    {
      continue;
    }
    const uint32_t array_slot = state.register_array_slot[reg_id];
    if (array_slot >= state.array_storage.size())
    {
      return false;
    }
    if (reg_id >= numeric_register_arrays_.size())
    {
      return false;
    }
    const auto & reg_values = numeric_register_arrays_[reg_id];
    auto & dst = state.array_storage[array_slot];
    if (reg_values.size() != dst.size())
    {
      return false;
    }
    std::copy(reg_values.begin(), reg_values.end(), dst.begin());
  }
  return true;
}

void Module::ensure_value_registers_current()
{
  if (registers_.size() < numeric_registers_.size())
  {
    registers_.resize(numeric_registers_.size(), expr::float_value(0.0));
  }
  if (next_registers_.size() < registers_.size())
  {
    next_registers_.resize(registers_.size(), expr::float_value(0.0));
  }
  for (unsigned int register_id = 0; register_id < registers_.size(); ++register_id)
  {
    if (register_id < register_scalar_mask_.size() && !register_scalar_mask_[register_id])
    {
      const std::size_t item_count =
        register_id < numeric_register_arrays_.size() ? numeric_register_arrays_[register_id].size() : 0;
      std::vector<Value> items(item_count, expr::float_value(0.0));
      if (register_id < numeric_register_arrays_.size())
      {
        for (std::size_t item_id = 0; item_id < numeric_register_arrays_[register_id].size(); ++item_id)
        {
          assign_scalar_numeric_value(items[item_id], numeric_register_arrays_[register_id][item_id]);
        }
      }
      registers_[register_id] = expr::array_value(std::move(items));
      next_registers_[register_id] = registers_[register_id];
      continue;
    }

    if (register_id < numeric_registers_.size())
    {
      assign_scalar_numeric_value(registers_[register_id], numeric_registers_[register_id]);
      next_registers_[register_id] = registers_[register_id];
    }
  }
  value_registers_dirty_ = false;
}

void Module::ensure_value_delay_states_current()
{
  if (delay_states_.size() < numeric_delay_scalars_.size())
  {
    delay_states_.resize(numeric_delay_scalars_.size(), expr::float_value(0.0));
  }
  if (next_delay_states_.size() < delay_states_.size())
  {
    next_delay_states_.resize(delay_states_.size(), expr::float_value(0.0));
  }
  for (unsigned int delay_id = 0; delay_id < delay_states_.size(); ++delay_id)
  {
    if (delay_id < numeric_delay_scalar_mask_.size() && numeric_delay_scalar_mask_[delay_id])
    {
      assign_scalar_numeric_value(delay_states_[delay_id], numeric_delay_scalars_[delay_id]);
      next_delay_states_[delay_id] = delay_states_[delay_id];
      continue;
    }
    if (delay_id < numeric_delay_array_mask_.size() && numeric_delay_array_mask_[delay_id])
    {
      const auto & src = numeric_delay_arrays_[delay_id];
      std::vector<Value> items(src.size(), expr::float_value(0.0));
      for (std::size_t item_id = 0; item_id < src.size(); ++item_id)
      {
        assign_scalar_numeric_value(items[item_id], src[item_id]);
      }
      delay_states_[delay_id] = expr::array_value(std::move(items));
      next_delay_states_[delay_id] = delay_states_[delay_id];
    }
  }
  value_delay_states_dirty_ = false;
}

bool Module::try_get_numeric_delay_scalar(unsigned int delay_id, double & out) const
{
  if (delay_id >= numeric_delay_scalar_mask_.size() ||
      delay_id >= numeric_delay_scalars_.size() ||
      !numeric_delay_scalar_mask_[delay_id])
  {
    return false;
  }
  out = numeric_delay_scalars_[delay_id];
  return true;
}

const std::vector<double> * Module::try_get_numeric_delay_array(unsigned int delay_id) const
{
  if (delay_id >= numeric_delay_array_mask_.size() ||
      delay_id >= numeric_delay_arrays_.size() ||
      !numeric_delay_array_mask_[delay_id])
  {
    return nullptr;
  }
  return &numeric_delay_arrays_[delay_id];
}

bool Module::update_numeric_delay_states_from_outputs(
  const NumericJitState & state,
  const CompiledProgram & compiled_program,
  std::size_t start_output_id)
{
  const std::size_t available =
    std::min(state.output_info.size(), compiled_program.output_targets.size());
  if (start_output_id > available)
  {
    return false;
  }
  const std::size_t delay_count = available - start_output_id;
  if (numeric_delay_scalar_mask_.size() != delay_count)
  {
    numeric_delay_scalar_mask_.assign(delay_count, false);
    numeric_delay_scalars_.assign(delay_count, 0.0);
    numeric_delay_array_mask_.assign(delay_count, false);
    numeric_delay_arrays_.assign(delay_count, {});
  }

  for (std::size_t delay_id = 0; delay_id < delay_count; ++delay_id)
  {
    const std::size_t output_id = start_output_id + delay_id;
    const auto kind = static_cast<NumericValueKind>(state.output_info[output_id].kind);
    numeric_delay_scalar_mask_[delay_id] = false;
    numeric_delay_array_mask_[delay_id] = false;
    if (kind == NumericValueKind::Scalar)
    {
      const uint32_t reg = compiled_program.output_targets[output_id];
      if (reg >= state.temps.size())
      {
        return false;
      }
      numeric_delay_scalar_mask_[delay_id] = true;
      numeric_delay_scalars_[delay_id] = clamp_output_scalar(state.temps[reg]);
      numeric_delay_arrays_[delay_id].clear();
      continue;
    }
    if (kind == NumericValueKind::Array)
    {
      const uint32_t array_slot = state.output_info[output_id].array_slot;
      if (array_slot >= state.array_storage.size())
      {
        return false;
      }
      numeric_delay_array_mask_[delay_id] = true;
      auto & dst = numeric_delay_arrays_[delay_id];
      dst = state.array_storage[array_slot];
      for (double & value : dst)
      {
        value = clamp_output_scalar(value);
      }
      continue;
    }
    return false;
  }

  value_delay_states_dirty_ = true;
  return true;
}

bool Module::prepare_numeric_jit_program(
  const CompiledProgram & source_program,
  unsigned int base_input_count,
  PreparedNumericJitProgram & prepared) const
{
  prepared.program = source_program;
  prepared.synthetic_inputs.clear();

  auto synthetic_index_for = [&](NumericSyntheticInputKind kind, uint32_t slot_id, uint32_t output_id) {
    for (const auto & input : prepared.synthetic_inputs)
    {
      if (input.kind == kind && input.slot_id == slot_id && input.output_id == output_id)
      {
        return input.input_slot;
      }
    }
    const uint32_t input_slot = static_cast<uint32_t>(base_input_count + prepared.synthetic_inputs.size());
    prepared.synthetic_inputs.push_back(NumericSyntheticInput{kind, slot_id, output_id, input_slot});
    return input_slot;
  };

  for (auto & instr : prepared.program.instructions)
  {
    switch (instr.kind)
    {
      case ExprKind::NestedValue:
        instr.kind = ExprKind::InputValue;
        instr.slot_id = synthetic_index_for(NumericSyntheticInputKind::NestedOutput, instr.slot_id, instr.output_id);
        instr.output_id = 0;
        break;
      case ExprKind::DelayValue:
        instr.kind = ExprKind::InputValue;
        instr.slot_id = synthetic_index_for(NumericSyntheticInputKind::DelayState, instr.slot_id, 0);
        break;
      default:
        break;
    }
  }

  return true;
}

std::vector<Value> Module::build_numeric_jit_inputs(
  const std::vector<Value> & current_inputs,
  const PreparedNumericJitProgram & prepared) const
{
  std::vector<Value> values = current_inputs;
  values.reserve(current_inputs.size() + prepared.synthetic_inputs.size());
  for (const auto & input : prepared.synthetic_inputs)
  {
    switch (input.kind)
    {
      case NumericSyntheticInputKind::NestedOutput:
      {
        const auto nested_it = nested_module_lookup_.find(input.slot_id);
        if (nested_it == nested_module_lookup_.end())
        {
          values.push_back(expr::float_value(0.0));
          break;
        }
        const NestedModuleRuntime & nested = nested_modules_[nested_it->second];
        values.push_back(
          input.output_id < nested.module->outputs.size()
            ? nested.module->outputs[input.output_id]
            : expr::float_value(0.0));
        break;
      }
      case NumericSyntheticInputKind::DelayState:
      {
        const auto delay_it = delay_state_lookup_.find(input.slot_id);
        if (delay_it == delay_state_lookup_.end())
        {
          values.push_back(expr::float_value(0.0));
          break;
        }
        const std::size_t delay_id = delay_it->second;
        double scalar = 0.0;
        if (try_get_numeric_delay_scalar(static_cast<unsigned int>(delay_id), scalar))
        {
          values.push_back(expr::float_value(scalar));
          break;
        }
        if (const auto * array_values = try_get_numeric_delay_array(static_cast<unsigned int>(delay_id)))
        {
          std::vector<Value> items(array_values->size(), expr::float_value(0.0));
          for (std::size_t item_id = 0; item_id < array_values->size(); ++item_id)
          {
            assign_scalar_numeric_value(items[item_id], (*array_values)[item_id]);
          }
          values.push_back(expr::array_value(std::move(items)));
          break;
        }
        values.push_back(delay_states_[delay_id]);
        break;
      }
    }
  }
  return values;
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
      if (!add_array_value_to_jit_table(reg, array_slot))
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
          if (!add_array_value_to_jit_table(instr.literal, array_slot))
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
      case ExprKind::ArraySet:
        // src_a = array, src_b = index (scalar), src_c = value (scalar)
        if (instr.src_a >= reg_info.size() || instr.src_b >= reg_info.size() || instr.src_c >= reg_info.size())
        {
          return false;
        }
        if (reg_info[instr.src_a].kind != NumericValueKind::Array ||
            reg_info[instr.src_b].kind != NumericValueKind::Scalar ||
            reg_info[instr.src_c].kind != NumericValueKind::Scalar)
        {
          return false;
        }
        jit_instr.op = egress_jit::NumericOp::SetArrayElement;
        jit_instr.slot_id = reg_info[instr.src_a].array_slot;
        jit_instr.src_a = instr.src_b;  // index
        jit_instr.src_b = instr.src_c;  // value
        // dst aliases the same array slot (in-place write; reads precede writes in the program)
        reg_info[instr.dst].kind = NumericValueKind::Array;
        reg_info[instr.dst].array_slot = reg_info[instr.src_a].array_slot;
        reg_info[instr.dst].array_size = reg_info[instr.src_a].array_size;
        require_scalar_inputs = false;
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
      case ExprKind::Select:
        jit_instr.op = egress_jit::NumericOp::Select;
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
      case ExprKind::SmoothedParam:
      {
        if (!instr.control_param)
        {
          return false;
        }
        const double tc = instr.control_param->time_const;
        const double coeff = (tc > 0.0)
          ? 1.0 - std::exp(-1.0 / (tc * sample_rate_))
          : 1.0;
        jit_instr.op = egress_jit::NumericOp::SmoothedParam;
        jit_instr.literal = coeff;
        jit_instr.param_ptr = reinterpret_cast<uint64_t>(&instr.control_param->value);
        jit_instr.slot_id = instr.slot_id;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
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
          instr.kind != ExprKind::SmoothedParam &&
          instr.kind != ExprKind::Index &&
          instr.kind != ExprKind::ArraySet)
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

bool Module::build_numeric_program(
  const CompiledProgram & compiled_program,
  NumericJitState & state,
  const std::vector<Value> & current_inputs,
  egress_jit::NumericProgram & numeric_program)
{
  CompiledProgram saved_program = std::move(program_);
  std::vector<std::vector<double>> saved_array_storage = std::move(numeric_array_storage_);
  std::vector<bool> saved_register_scalar_mask = std::move(register_scalar_mask_);
  std::vector<uint32_t> saved_register_array_slot = std::move(register_array_slot_);
  std::vector<int32_t> saved_array_register_targets = std::move(array_register_targets_);
  std::vector<bool> saved_array_register_can_swap = std::move(array_register_can_swap_);
  std::vector<NumericInputInfo> saved_input_info = std::move(numeric_input_info_);
  std::vector<NumericOutputInfo> saved_output_info = std::move(numeric_output_info_);

  program_ = compiled_program;
  numeric_array_storage_ = state.array_storage;
  register_scalar_mask_ = state.register_scalar_mask;
  register_array_slot_ = state.register_array_slot;
  array_register_targets_ = state.array_register_targets;
  array_register_can_swap_ = state.array_register_can_swap;
  numeric_input_info_ = state.input_info;
  numeric_output_info_ = state.output_info;

  const bool ok = build_numeric_program(current_inputs, numeric_program);

  if (ok)
  {
    state.array_storage = std::move(numeric_array_storage_);
    state.register_scalar_mask = std::move(register_scalar_mask_);
    state.register_array_slot = std::move(register_array_slot_);
    state.array_register_targets = std::move(array_register_targets_);
    state.array_register_can_swap = std::move(array_register_can_swap_);
    state.input_info = std::move(numeric_input_info_);
    state.output_info = std::move(numeric_output_info_);
  }

  program_ = std::move(saved_program);
  numeric_array_storage_ = std::move(saved_array_storage);
  register_scalar_mask_ = std::move(saved_register_scalar_mask);
  register_array_slot_ = std::move(saved_register_array_slot);
  array_register_targets_ = std::move(saved_array_register_targets);
  array_register_can_swap_ = std::move(saved_array_register_can_swap);
  numeric_input_info_ = std::move(saved_input_info);
  numeric_output_info_ = std::move(saved_output_info);

  return ok;
}

void Module::initialize_numeric_jit_state(
  NumericJitState & state,
  const CompiledProgram & source_program,
  const std::vector<Value> & current_inputs,
  unsigned int base_input_count,
  const std::string & symbol_prefix)
{
  if (value_registers_dirty_)
  {
    ensure_value_registers_current();
  }
  state = NumericJitState{};
  if (!prepare_numeric_jit_program(source_program, base_input_count, state.prepared))
  {
    return;
  }

  std::vector<Value> jit_inputs = build_numeric_jit_inputs(current_inputs, state.prepared);
  egress_jit::NumericProgram numeric_program;
  if (!build_numeric_program(state.prepared.program, state, jit_inputs, numeric_program))
  {
    return;
  }

  auto & jit = egress_jit::OrcJitEngine::instance();
  auto kernel_or_err = jit.compile_numeric_program(numeric_program, symbol_prefix);
  if (!kernel_or_err)
  {
    return;
  }

  state.kernel = *kernel_or_err;

  // Build per-instance param_ptrs in canonical (first-appearance) order.
  state.param_ptrs.clear();
  {
    std::unordered_map<uint64_t, uint32_t> seen;
    for (const auto & instr : numeric_program.instructions)
    {
      if (instr.op == egress_jit::NumericOp::SmoothedParam && instr.param_ptr != 0 &&
          seen.find(instr.param_ptr) == seen.end())
      {
        seen.emplace(instr.param_ptr, static_cast<uint32_t>(state.param_ptrs.size()));
        state.param_ptrs.push_back(instr.param_ptr);
      }
    }
  }

  state.temps.assign(state.prepared.program.register_count, 0.0);
  state.inputs.assign(jit_inputs.size(), 0.0);
  state.array_ptrs.resize(state.array_storage.size(), nullptr);
  state.array_sizes.resize(state.array_storage.size(), 0);
  for (std::size_t i = 0; i < state.array_storage.size(); ++i)
  {
    state.array_ptrs[i] = state.array_storage[i].empty() ? nullptr : state.array_storage[i].data();
    state.array_sizes[i] = static_cast<uint64_t>(state.array_storage[i].size());
  }
#ifdef EGRESS_PROFILE
  state.instruction_count = static_cast<uint64_t>(numeric_program.instructions.size());
#endif
}

void Module::ensure_numeric_jit_state_current(
  NumericJitState & state,
  const CompiledProgram & source_program,
  const std::vector<Value> & current_inputs,
  unsigned int base_input_count,
  const std::string & symbol_prefix)
{
  if (state.kernel != nullptr)
  {
    const std::vector<Value> jit_inputs = build_numeric_jit_inputs(current_inputs, state.prepared);
    if (numeric_input_layout_matches(state, jit_inputs))
    {
      return;
    }
  }

  initialize_numeric_jit_state(state, source_program, current_inputs, base_input_count, symbol_prefix);
}

bool Module::prepare_composite_body_jit_program(CompositeBodyJitState & state) const
{
  if ((!has_nested_modules_ && !has_delay_states_) ||
      composite_schedule_.empty() ||
      composite_schedule_.back() != composite_output_boundary_id_)
  {
    return false;
  }

  state = CompositeBodyJitState{};
  state.output_count = static_cast<uint32_t>(composite_output_program_.output_targets.size());
  state.delay_output_count = static_cast<uint32_t>(delay_update_program_.output_targets.size());

  auto synthetic_index_for = [&](NumericSyntheticInputKind kind, uint32_t slot_id, uint32_t output_id) {
    for (const auto & input : state.state.prepared.synthetic_inputs)
    {
      if (input.kind == kind && input.slot_id == slot_id && input.output_id == output_id)
      {
        return input.input_slot;
      }
    }
    const uint32_t input_slot = static_cast<uint32_t>(input_count_ + state.state.prepared.synthetic_inputs.size());
    state.state.prepared.synthetic_inputs.push_back(NumericSyntheticInput{kind, slot_id, output_id, input_slot});
    return input_slot;
  };

  auto append_program = [&](const CompiledProgram & source_program, bool append_outputs, bool append_registers)
  {
    const uint32_t reg_base = state.program.register_count;
    state.program.register_count += source_program.register_count;
    for (const auto & instr : source_program.instructions)
    {
      Instr translated = instr;
      translated.dst += reg_base;
      translated.src_a += reg_base;
      translated.src_b += reg_base;
      translated.src_c += reg_base;
      for (auto & arg : translated.args)
      {
        arg += reg_base;
      }
      switch (translated.kind)
      {
        case ExprKind::NestedValue:
          translated.kind = ExprKind::InputValue;
          translated.slot_id = synthetic_index_for(
            NumericSyntheticInputKind::NestedOutput,
            instr.slot_id,
            instr.output_id);
          translated.output_id = 0;
          break;
        case ExprKind::DelayValue:
          translated.kind = ExprKind::InputValue;
          translated.slot_id = synthetic_index_for(
            NumericSyntheticInputKind::DelayState,
            instr.slot_id,
            0);
          break;
        default:
          break;
      }
      state.program.instructions.push_back(std::move(translated));
    }

    if (append_outputs)
    {
      for (uint32_t target : source_program.output_targets)
      {
        state.program.output_targets.push_back(reg_base + target);
      }
    }
    if (append_registers)
    {
      for (int32_t target : source_program.register_targets)
      {
        state.program.register_targets.push_back(target >= 0 ? static_cast<int32_t>(reg_base + static_cast<uint32_t>(target)) : -1);
      }
    }
  };

  append_program(composite_output_program_, true, false);
  append_program(delay_update_program_, true, false);
  append_program(composite_register_program_, false, true);
  state.state.prepared.program = state.program;
  return true;
}

void Module::initialize_composite_body_jit(const std::vector<Value> & current_inputs)
{
  composite_body_jit_ = CompositeBodyJitState{};
  if (!prepare_composite_body_jit_program(composite_body_jit_))
  {
    return;
  }

  std::vector<Value> jit_inputs = build_numeric_jit_inputs(current_inputs, composite_body_jit_.state.prepared);
  egress_jit::NumericProgram numeric_program;
  if (!build_numeric_program(composite_body_jit_.state.prepared.program, composite_body_jit_.state, jit_inputs, numeric_program))
  {
    composite_body_jit_ = CompositeBodyJitState{};
    return;
  }

  const std::size_t fused_output_count =
    static_cast<std::size_t>(composite_body_jit_.output_count + composite_body_jit_.delay_output_count);
  if (composite_body_jit_.state.output_info.size() < fused_output_count)
  {
    composite_body_jit_ = CompositeBodyJitState{};
    return;
  }
  for (std::size_t output_id = 0; output_id < fused_output_count; ++output_id)
  {
    if (static_cast<NumericValueKind>(composite_body_jit_.state.output_info[output_id].kind) != NumericValueKind::Scalar)
    {
      composite_body_jit_ = CompositeBodyJitState{};
      return;
    }
  }
  for (bool scalar_register : composite_body_jit_.state.register_scalar_mask)
  {
    if (!scalar_register)
    {
      composite_body_jit_ = CompositeBodyJitState{};
      return;
    }
  }

  auto & jit = egress_jit::OrcJitEngine::instance();
  auto kernel_or_err = jit.compile_numeric_program(numeric_program, "egress_udm_composite_body");
  if (!kernel_or_err)
  {
    composite_body_jit_ = CompositeBodyJitState{};
    return;
  }

  composite_body_jit_.state.kernel = *kernel_or_err;
  composite_body_jit_.state.temps.assign(composite_body_jit_.state.prepared.program.register_count, 0.0);
  composite_body_jit_.state.inputs.assign(jit_inputs.size(), 0.0);
  composite_body_jit_.state.array_ptrs.resize(composite_body_jit_.state.array_storage.size(), nullptr);
  composite_body_jit_.state.array_sizes.resize(composite_body_jit_.state.array_storage.size(), 0);
  for (std::size_t i = 0; i < composite_body_jit_.state.array_storage.size(); ++i)
  {
    composite_body_jit_.state.array_ptrs[i] =
      composite_body_jit_.state.array_storage[i].empty() ? nullptr : composite_body_jit_.state.array_storage[i].data();
    composite_body_jit_.state.array_sizes[i] =
      static_cast<uint64_t>(composite_body_jit_.state.array_storage[i].size());
  }
#ifdef EGRESS_PROFILE
  composite_body_jit_.instruction_count = static_cast<uint64_t>(numeric_program.instructions.size());
#endif
}

void Module::ensure_composite_body_jit_current()
{
  if (composite_body_jit_.state.kernel != nullptr)
  {
    const std::vector<Value> jit_inputs = build_numeric_jit_inputs(inputs, composite_body_jit_.state.prepared);
    if (numeric_input_layout_matches(composite_body_jit_.state, jit_inputs))
    {
      return;
    }
  }
  initialize_composite_body_jit(inputs);
}

bool Module::run_numeric_jit_state(
  NumericJitState & state,
  const std::vector<Value> & current_inputs)
{
  if (state.kernel == nullptr)
  {
    return false;
  }

  const std::vector<Value> jit_inputs = build_numeric_jit_inputs(current_inputs, state.prepared);
  if (!sync_numeric_register_arrays_from_values(state) ||
      !sync_numeric_inputs_from_values(state, jit_inputs))
  {
    state.kernel = nullptr;
    return false;
  }

  state.kernel(
    state.inputs.data(),
    numeric_registers_.data(),
    state.array_ptrs.data(),
    state.array_sizes.data(),
    state.temps.data(),
    sample_rate_,
    sample_index_,
    state.param_ptrs.data());

  return true;
}

bool Module::run_composite_body_jit(const std::vector<bool> * output_materialize_mask)
{
  if (composite_body_jit_.state.kernel == nullptr ||
      !run_numeric_jit_state(composite_body_jit_.state, inputs))
  {
    return false;
  }

  capture_numeric_scalar_outputs(
    composite_body_jit_.program,
    composite_body_jit_.state.output_info,
    composite_body_jit_.state.temps,
    0,
    composite_body_jit_.output_count);
  materialize_numeric_outputs_range(
    composite_body_jit_.state,
    composite_body_jit_.program,
    0,
    composite_body_jit_.output_count,
    outputs,
    output_materialize_mask);

  apply_numeric_register_targets(composite_body_jit_.state, composite_body_jit_.program);

  if (composite_body_jit_.delay_output_count > 0)
  {
    if (!update_numeric_delay_states_from_outputs(
          composite_body_jit_.state,
          composite_body_jit_.program,
          composite_body_jit_.output_count))
    {
      next_delay_states_ = delay_states_;
      materialize_numeric_outputs_range(
        composite_body_jit_.state,
        composite_body_jit_.program,
        composite_body_jit_.output_count,
        composite_body_jit_.delay_output_count,
        next_delay_states_);
      delay_states_.swap(next_delay_states_);
      value_delay_states_dirty_ = false;
    }
  }

  return true;
}

void Module::materialize_numeric_outputs(
  const NumericJitState & state,
  const CompiledProgram & compiled_program,
  std::vector<Value> & destinations,
  const std::vector<bool> * materialize_mask)
{
#ifdef EGRESS_PROFILE
  const auto materialize_start = std::chrono::steady_clock::now();
  uint64_t materialized_scalar_outputs = 0;
  uint64_t materialized_array_outputs = 0;
  uint64_t materialized_matrix_outputs = 0;
#endif
  for (unsigned int output_id = 0; output_id < compiled_program.output_targets.size(); ++output_id)
  {
    if (materialize_mask != nullptr &&
        output_id < materialize_mask->size() &&
        !(*materialize_mask)[output_id] &&
        output_id < state.output_info.size())
    {
      const NumericValueKind output_kind = static_cast<NumericValueKind>(state.output_info[output_id].kind);
      if (output_kind == NumericValueKind::Array || output_kind == NumericValueKind::Matrix)
      {
        continue;
      }
    }

#ifdef EGRESS_PROFILE
    switch (static_cast<NumericValueKind>(state.output_info[output_id].kind))
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
      destinations[output_id],
      state.output_info[output_id],
      compiled_program.output_targets[output_id],
      state.temps,
      state.array_storage);
  }
#ifdef EGRESS_PROFILE
  record_numeric_output_materialize_profile(
    static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - materialize_start).count()),
    materialized_scalar_outputs,
    materialized_array_outputs,
    materialized_matrix_outputs);
#endif
}

void Module::materialize_numeric_outputs_range(
  const NumericJitState & state,
  const CompiledProgram & compiled_program,
  std::size_t start_output_id,
  std::size_t output_count,
  std::vector<Value> & destinations,
  const std::vector<bool> * materialize_mask)
{
#ifdef EGRESS_PROFILE
  const auto materialize_start = std::chrono::steady_clock::now();
  uint64_t materialized_scalar_outputs = 0;
  uint64_t materialized_array_outputs = 0;
  uint64_t materialized_matrix_outputs = 0;
#endif
  for (std::size_t output_index = 0; output_index < output_count; ++output_index)
  {
    const std::size_t compiled_output_id = start_output_id + output_index;
    if (compiled_output_id >= compiled_program.output_targets.size() ||
        compiled_output_id >= state.output_info.size() ||
        output_index >= destinations.size())
    {
      continue;
    }
    if (materialize_mask != nullptr &&
        output_index < materialize_mask->size() &&
        !(*materialize_mask)[output_index] &&
        static_cast<NumericValueKind>(state.output_info[compiled_output_id].kind) != NumericValueKind::Scalar)
    {
      continue;
    }

#ifdef EGRESS_PROFILE
    switch (static_cast<NumericValueKind>(state.output_info[compiled_output_id].kind))
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
      destinations[output_index],
      state.output_info[compiled_output_id],
      compiled_program.output_targets[compiled_output_id],
      state.temps,
      state.array_storage);
  }
#ifdef EGRESS_PROFILE
  record_numeric_output_materialize_profile(
    static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - materialize_start).count()),
    materialized_scalar_outputs,
    materialized_array_outputs,
    materialized_matrix_outputs);
#endif
}

void Module::apply_numeric_register_targets(
  NumericJitState & state,
  const CompiledProgram & compiled_program)
{
  for (unsigned int register_id = 0; register_id < compiled_program.register_targets.size(); ++register_id)
  {
    const int32_t target = compiled_program.register_targets[register_id];
    if (state.register_scalar_mask[register_id])
    {
      if (target >= 0)
      {
        numeric_next_registers_[register_id] = state.temps[static_cast<std::size_t>(target)];
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

  // Preserve anonymous registers (e.g., SmoothedParam state) written in-kernel
  for (unsigned int register_id = static_cast<unsigned int>(compiled_program.register_targets.size());
       register_id < numeric_registers_.size();
       ++register_id)
  {
    numeric_next_registers_[register_id] = numeric_registers_[register_id];
  }

  numeric_registers_.swap(numeric_next_registers_);

  for (unsigned int register_id = 0; register_id < state.array_register_targets.size(); ++register_id)
  {
    if (state.register_scalar_mask[register_id])
    {
      continue;
    }
    const int32_t src_slot = state.array_register_targets[register_id];
    if (src_slot < 0)
    {
      continue;
    }
    const uint32_t dst_slot = state.register_array_slot[register_id];
    if (dst_slot >= state.array_storage.size() ||
        static_cast<std::size_t>(src_slot) >= state.array_storage.size())
    {
      continue;
    }
    auto & dst = state.array_storage[dst_slot];
    auto & src = state.array_storage[static_cast<std::size_t>(src_slot)];
    if (dst.size() != src.size())
    {
      continue;
    }
    if (register_id < state.array_register_can_swap.size() && state.array_register_can_swap[register_id])
    {
      dst.swap(src);
      if (dst_slot < state.array_ptrs.size())
      {
        state.array_ptrs[dst_slot] = dst.empty() ? nullptr : dst.data();
        state.array_sizes[dst_slot] = static_cast<uint64_t>(dst.size());
      }
      if (static_cast<std::size_t>(src_slot) < state.array_ptrs.size())
      {
        state.array_ptrs[static_cast<std::size_t>(src_slot)] = src.empty() ? nullptr : src.data();
        state.array_sizes[static_cast<std::size_t>(src_slot)] = static_cast<uint64_t>(src.size());
      }
      continue;
    }
    std::copy(src.begin(), src.end(), dst.begin());
  }

  if (numeric_register_arrays_.size() < compiled_program.register_targets.size())
  {
    numeric_register_arrays_.resize(compiled_program.register_targets.size());
  }
  for (unsigned int register_id = 0; register_id < compiled_program.register_targets.size(); ++register_id)
  {
    if (register_id < state.register_scalar_mask.size() && !state.register_scalar_mask[register_id])
    {
      if (register_id >= state.register_array_slot.size())
      {
        continue;
      }
      const uint32_t array_slot = state.register_array_slot[register_id];
      if (array_slot >= state.array_storage.size())
      {
        continue;
      }
      numeric_register_arrays_[register_id] = state.array_storage[array_slot];
    }
  }
  value_registers_dirty_ = true;
}

void Module::sync_value_registers_from_numeric_state(const NumericJitState & state)
{
#ifdef EGRESS_PROFILE
  const auto sync_start = std::chrono::steady_clock::now();
  uint64_t materialized_scalar_registers = 0;
  uint64_t materialized_array_registers = 0;
#endif
  for (unsigned int register_id = 0; register_id < registers_.size(); ++register_id)
  {
    if (register_id < state.register_scalar_mask.size() && state.register_scalar_mask[register_id])
    {
      assign_scalar_numeric_value(registers_[register_id], numeric_registers_[register_id]);
      next_registers_[register_id] = registers_[register_id];
#ifdef EGRESS_PROFILE
      ++materialized_scalar_registers;
#endif
      continue;
    }
    if (register_id >= state.register_array_slot.size())
    {
      continue;
    }
    const uint32_t array_slot = state.register_array_slot[register_id];
    if (array_slot >= state.array_storage.size())
    {
      continue;
    }
    std::vector<Value> items(state.array_storage[array_slot].size(), expr::float_value(0.0));
    for (std::size_t item_id = 0; item_id < items.size(); ++item_id)
    {
      assign_scalar_numeric_value(items[item_id], state.array_storage[array_slot][item_id]);
    }
    registers_[register_id] = expr::array_value(std::move(items));
    next_registers_[register_id] = registers_[register_id];
#ifdef EGRESS_PROFILE
    ++materialized_array_registers;
#endif
  }
#ifdef EGRESS_PROFILE
  record_numeric_register_sync_profile(
    static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sync_start).count()),
    materialized_scalar_registers,
    materialized_array_registers);
#endif
  value_registers_dirty_ = false;
}

void Module::initialize_numeric_jit(const std::vector<Value> & current_inputs)
{
  if (value_registers_dirty_)
  {
    ensure_value_registers_current();
  }
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

  if (has_dynamic_registers_)
  {
    jit_status_ = "numeric JIT disabled for dynamic array_state registers";
    return;
  }

  if (!has_nested_modules_ && !has_delay_states_)
  {
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

    // Build per-instance param_ptrs in canonical (first-appearance) order.
    numeric_param_ptrs_.clear();
    {
      std::unordered_map<uint64_t, uint32_t> seen;
      for (const auto & instr : numeric_program.instructions)
      {
        if (instr.op == egress_jit::NumericOp::SmoothedParam && instr.param_ptr != 0 &&
            seen.find(instr.param_ptr) == seen.end())
        {
          seen.emplace(instr.param_ptr, static_cast<uint32_t>(numeric_param_ptrs_.size()));
          numeric_param_ptrs_.push_back(instr.param_ptr);
        }
      }
    }

    numeric_temps_.assign(program_.register_count, 0.0);
    numeric_array_ptrs_.resize(numeric_array_storage_.size(), nullptr);
    numeric_array_sizes_.resize(numeric_array_storage_.size(), 0);
    for (std::size_t i = 0; i < numeric_array_storage_.size(); ++i)
    {
      numeric_array_ptrs_[i] = numeric_array_storage_[i].empty() ? nullptr : numeric_array_storage_[i].data();
      numeric_array_sizes_[i] = static_cast<uint64_t>(numeric_array_storage_[i].size());
    }
  }

  numeric_registers_.resize(registers_.size(), 0.0);
  numeric_next_registers_.resize(registers_.size(), 0.0);
  numeric_register_arrays_.resize(registers_.size());
  for (unsigned int i = 0; i < registers_.size(); ++i)
  {
    const bool scalar_register =
      i >= register_scalar_mask_.size() || register_scalar_mask_[i];
    numeric_registers_[i] = scalar_register ? to_float64(registers_[i]) : 0.0;
    if (!scalar_register &&
        i < register_array_slot_.size() &&
        register_array_slot_[i] < numeric_array_storage_.size())
    {
      numeric_register_arrays_[i] = numeric_array_storage_[register_array_slot_[i]];
    }
    else if (i < numeric_register_arrays_.size())
    {
      numeric_register_arrays_[i].clear();
    }
  }
  value_registers_dirty_ = false;

  numeric_delay_scalar_mask_.assign(delay_states_.size(), false);
  numeric_delay_scalars_.assign(delay_states_.size(), 0.0);
  numeric_delay_array_mask_.assign(delay_states_.size(), false);
  numeric_delay_arrays_.assign(delay_states_.size(), {});
  for (unsigned int i = 0; i < delay_states_.size(); ++i)
  {
    double scalar = 0.0;
    if (value_to_scalar_double(delay_states_[i], scalar))
    {
      numeric_delay_scalar_mask_[i] = true;
      numeric_delay_scalars_[i] = clamp_output_scalar(scalar);
      continue;
    }
    if (delay_states_[i].type == ValueType::Array)
    {
      std::vector<double> values(delay_states_[i].array_items.size(), 0.0);
      if (copy_numeric_aggregate_value(delay_states_[i], values))
      {
        for (double & value : values)
        {
          value = clamp_output_scalar(value);
        }
        numeric_delay_array_mask_[i] = true;
        numeric_delay_arrays_[i] = std::move(values);
      }
    }
  }
  value_delay_states_dirty_ = false;

  if (!has_nested_modules_ && !has_delay_states_)
  {
    jit_status_ = "numeric JIT active";
    return;
  }

  composite_register_jit_ = NumericJitState{};
  delay_update_jit_ = NumericJitState{};
  composite_output_jit_ = NumericJitState{};
  composite_body_jit_ = CompositeBodyJitState{};
  nested_input_jit_states_.clear();
  nested_input_jit_states_.reserve(nested_modules_.size());
  for (std::size_t nested_id = 0; nested_id < nested_modules_.size(); ++nested_id)
  {
    nested_input_jit_states_.push_back(std::make_unique<NumericJitState>());
  }

  if (!has_nested_modules_)
  {
    initialize_composite_body_jit(current_inputs);
    ensure_numeric_jit_state_current(
      composite_output_jit_,
      composite_output_program_,
      current_inputs,
      static_cast<unsigned int>(current_inputs.size()),
      "egress_udm_composite_output");
    ensure_numeric_jit_state_current(
      composite_register_jit_,
      composite_register_program_,
      current_inputs,
      static_cast<unsigned int>(current_inputs.size()),
      "egress_udm_composite_register");
    ensure_numeric_jit_state_current(
      delay_update_jit_,
      delay_update_program_,
      current_inputs,
      static_cast<unsigned int>(current_inputs.size()),
      "egress_udm_delay_update");
  }

  jit_status_ = "numeric JIT active";
}

void Module::ensure_numeric_jit_current()
{
  if (has_dynamic_registers_)
  {
    return;
  }

  if (!has_nested_modules_ && !has_delay_states_)
  {
    if (numeric_input_layout_matches(inputs))
    {
      return;
    }
    initialize_numeric_jit(inputs);
    return;
  }

  if (nested_input_jit_states_.size() != nested_modules_.size())
  {
    nested_input_jit_states_.clear();
    nested_input_jit_states_.reserve(nested_modules_.size());
    for (std::size_t nested_id = 0; nested_id < nested_modules_.size(); ++nested_id)
    {
      nested_input_jit_states_.push_back(std::make_unique<NumericJitState>());
    }
  }
}

#endif
