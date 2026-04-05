#pragma once

/**
 * NumericProgramBuilder.hpp — Convert CompiledProgram to NumericProgram for JIT.
 *
 * Extracted from Module class. These are pure functions with no instance state.
 */

#include "graph/GraphTypes.hpp"
#include "graph/ModuleNumericJit.hpp"
#include "graph/ModuleProgram.hpp"
#include "graph/TypeRegistry.hpp"
#include "jit/OrcJitEngine.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace egress_runtime
{

using Instr = egress_module_detail::Instr;
using CompiledProgram = egress_module_detail::CompiledProgram;
using NumericInputInfo = egress_module_detail::NumericInputInfo;
using NumericOutputInfo = egress_module_detail::NumericOutputInfo;
using NumericValueKind = egress_module_detail::NumericValueKind;
using NumericValueRef = egress_module_detail::NumericValueRef;
using NumericRegInfo = egress_module_detail::NumericRegInfo;

struct NumericJitState
{
  egress_jit::NumericKernelFn kernel = nullptr;
  std::vector<uint64_t> param_ptrs;
  std::vector<NumericInputInfo> input_info;
  std::vector<NumericOutputInfo> output_info;
  std::vector<int64_t> inputs;
  std::vector<int64_t> temps;
  std::vector<std::vector<int64_t>> array_storage;
  std::vector<int64_t *> array_ptrs;
  std::vector<uint64_t> array_sizes;
  std::vector<bool> register_scalar_mask;
  std::vector<uint32_t> register_array_slot;
  std::vector<int32_t> array_register_targets;
  std::vector<bool> array_register_can_swap;
  std::vector<egress_jit::JitScalarType> register_target_types;
};

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

egress_jit::JitScalarType infer_jit_scalar_type(
  ExprKind kind,
  egress_jit::JitScalarType a,
  egress_jit::JitScalarType b = egress_jit::JitScalarType::Float)
{
  using JST = egress_jit::JitScalarType;
  switch (kind)
  {
    case ExprKind::Less:
    case ExprKind::LessEqual:
    case ExprKind::Greater:
    case ExprKind::GreaterEqual:
    case ExprKind::Equal:
    case ExprKind::NotEqual:
    case ExprKind::Not:
      return JST::Bool;

    case ExprKind::BitAnd:
    case ExprKind::BitOr:
    case ExprKind::BitXor:
    case ExprKind::LShift:
    case ExprKind::RShift:
    case ExprKind::BitNot:
    case ExprKind::FloorDiv:
    case ExprKind::SampleIndex:
      return JST::Int;

    case ExprKind::Div:
    case ExprKind::Sin:
    case ExprKind::Log:
    case ExprKind::Pow:
    case ExprKind::SampleRate:
    case ExprKind::SmoothedParam:
      return JST::Float;

    case ExprKind::Add:
    case ExprKind::Sub:
    case ExprKind::Mul:
    case ExprKind::Mod:
      if (a == JST::Float || b == JST::Float) return JST::Float;
      return JST::Int;

    case ExprKind::Abs:
    case ExprKind::Neg:
      return a;

    case ExprKind::Clamp:
    case ExprKind::Select:
      if (a == JST::Float || b == JST::Float) return JST::Float;
      return JST::Int;

    // FieldAccess returns the field's scalar type; the registry lookup happens
    // in build_numeric_program_impl where reg_info is available.
    // For single-scalar inference fallback, return Float.
    case ExprKind::FieldAccess:
    case ExprKind::ConstructStruct:
    case ExprKind::ConstructVariant:
    case ExprKind::MatchVariant:
      return JST::Float;

    default:
      return JST::Float;
  }
}
} // anonymous namespace

inline bool value_to_scalar_double(const Value & value, double & out)
{
  if (value.type == ValueType::Array || value.type == ValueType::Matrix)
  {
    return false;
  }
  out = to_float64(value);
  return true;
}

inline bool add_array_values_to_jit_table(
  std::vector<std::vector<int64_t>> & array_storage,
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

  std::vector<int64_t> int_values(numeric_values.size());
  for (std::size_t i = 0; i < numeric_values.size(); ++i)
  {
    int_values[i] = std::bit_cast<int64_t>(numeric_values[i]);
  }
  out_slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::move(int_values));
  return true;
}

inline bool add_array_value_to_jit_table(
  std::vector<std::vector<int64_t>> & array_storage,
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

  std::vector<int64_t> int_values(numeric_values.size());
  for (std::size_t i = 0; i < numeric_values.size(); ++i)
  {
    int_values[i] = std::bit_cast<int64_t>(numeric_values[i]);
  }
  out_slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::move(int_values));
  return true;
}

inline bool add_matrix_values_to_jit_table(
  std::vector<std::vector<int64_t>> & array_storage,
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

  std::vector<int64_t> int_values(numeric_values.size());
  for (std::size_t i = 0; i < numeric_values.size(); ++i)
  {
    int_values[i] = std::bit_cast<int64_t>(numeric_values[i]);
  }
  out_slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::move(int_values));
  return true;
}

inline uint32_t allocate_array_slot_with_size(
  std::vector<std::vector<int64_t>> & array_storage,
  std::size_t size)
{
  uint32_t slot = static_cast<uint32_t>(array_storage.size());
  array_storage.push_back(std::vector<int64_t>(size, 0));
  return slot;
}

inline bool configure_numeric_inputs_for_jit(
  NumericJitState & state,
  const std::vector<Value> & current_inputs)
{
  state.input_info.assign(current_inputs.size(), NumericInputInfo{});

  for (unsigned int input_id = 0; input_id < current_inputs.size(); ++input_id)
  {
    const Value & input = current_inputs[input_id];
    NumericInputInfo & info = state.input_info[input_id];

    if (input.type == ValueType::Matrix)
    {
      return false;
    }

    // Determine scalar type
    if (input.type == ValueType::Int)
      info.scalar_type = egress_jit::JitScalarType::Int;
    else if (input.type == ValueType::Bool)
      info.scalar_type = egress_jit::JitScalarType::Bool;

    if (input.type != ValueType::Array)
    {
      continue;
    }

    // Array element type
    if (input.aggregate_scalar_type == AggregateScalarType::Int)
      info.array_element_type = egress_jit::JitScalarType::Int;
    else if (input.aggregate_scalar_type == AggregateScalarType::Bool)
      info.array_element_type = egress_jit::JitScalarType::Bool;

    uint32_t array_slot = 0;
    if (!add_array_value_to_jit_table(state.array_storage, input, array_slot))
    {
      return false;
    }

    info.is_scalar = false;
    info.array_slot = array_slot;
    info.array_size = static_cast<uint32_t>(input.array_items.size());
  }

  return true;
}

inline bool supports_numeric_jit_expr_kind(ExprKind kind)
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
    case ExprKind::ConstructStruct:
    case ExprKind::FieldAccess:
    case ExprKind::ConstructVariant:
    case ExprKind::MatchVariant:
      return true;
    default:
      return false;
  }
}

inline bool build_numeric_program(
  const CompiledProgram & program,
  const std::vector<Value> & registers,
  double sample_rate,
  const std::vector<Value> & current_inputs,
  egress_jit::NumericProgram & numeric_program,
  NumericJitState & state,
  const egress::TypeRegistry* registry)
{
  if (program.register_count == 0)
  {
    return false;
  }

  numeric_program.instructions.clear();
  numeric_program.register_count = program.register_count;
  state.array_storage.clear();
  state.register_scalar_mask.assign(registers.size(), true);
  state.register_array_slot.assign(registers.size(), 0);
  state.array_register_targets.assign(registers.size(), -1);
  state.array_register_can_swap.assign(registers.size(), false);
  state.register_target_types.assign(registers.size(), egress_jit::JitScalarType::Float);
  state.output_info.clear();

  if (!configure_numeric_inputs_for_jit(state, current_inputs))
  {
    return false;
  }

  std::vector<NumericRegInfo> reg_info(program.register_count);

  for (unsigned int reg_slot = 0; reg_slot < registers.size(); ++reg_slot)
  {
    const Value & reg = registers[reg_slot];
    if (reg.type == ValueType::Array)
    {
      uint32_t array_slot = 0;
      if (!add_array_value_to_jit_table(state.array_storage, reg, array_slot))
      {
        return false;
      }
      state.register_scalar_mask[reg_slot] = false;
      state.register_array_slot[reg_slot] = array_slot;
    }
    else if (reg.type == ValueType::Matrix)
    {
      return false;
    }
    else if (reg.type == ValueType::Int || reg.type == ValueType::Bool)
    {
      state.register_target_types[reg_slot] =
        reg.type == ValueType::Bool ? egress_jit::JitScalarType::Bool : egress_jit::JitScalarType::Int;
    }
  }

  for (const Instr & instr : program.instructions)
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
          if (!add_array_value_to_jit_table(state.array_storage, instr.literal, array_slot))
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
          if (!add_matrix_values_to_jit_table(state.array_storage, instr.literal, array_slot))
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
          if (instr.literal.type == ValueType::Int)
          {
            jit_instr.int_literal = instr.literal.int_value;
            reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
            reg_info[instr.dst].int_constant = instr.literal.int_value;
          }
          else if (instr.literal.type == ValueType::Bool)
          {
            jit_instr.int_literal = instr.literal.bool_value ? 1LL : 0LL;
            reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Bool;
            reg_info[instr.dst].int_constant = jit_instr.int_literal;
          }
          // else: Float (default)
        }
        break;
      }
      case ExprKind::InputValue:
        if (instr.slot_id >= state.input_info.size())
        {
          return false;
        }
        if (state.input_info[instr.slot_id].is_scalar)
        {
          const egress_jit::JitScalarType input_type = state.input_info[instr.slot_id].scalar_type;
          jit_instr.op = egress_jit::NumericOp::InputValue;
          jit_instr.src_a_type = input_type;
          jit_instr.dst_type = input_type;
          reg_info[instr.dst].kind = NumericValueKind::Scalar;
          reg_info[instr.dst].scalar_type = input_type;
        }
        else
        {
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = state.input_info[instr.slot_id].array_slot;
          reg_info[instr.dst].array_size = state.input_info[instr.slot_id].array_size;
          reg_info[instr.dst].array_element_type = state.input_info[instr.slot_id].array_element_type;
          emit_instruction = false;
        }
        break;
      case ExprKind::RegisterValue:
        if (instr.slot_id >= state.register_scalar_mask.size())
        {
          return false;
        }
        if (!state.register_scalar_mask[instr.slot_id])
        {
          reg_info[instr.dst].kind = NumericValueKind::Array;
          reg_info[instr.dst].array_slot = state.register_array_slot[instr.slot_id];
          reg_info[instr.dst].array_size = static_cast<uint32_t>(state.array_storage[state.register_array_slot[instr.slot_id]].size());
          emit_instruction = false;
        }
        else
        {
          const egress_jit::JitScalarType reg_type =
            (instr.slot_id < state.register_target_types.size() &&
             state.register_target_types[instr.slot_id] != egress_jit::JitScalarType::Float)
              ? state.register_target_types[instr.slot_id]
              : egress_jit::JitScalarType::Float;
          jit_instr.op = egress_jit::NumericOp::RegisterValue;
          jit_instr.src_a_type = reg_type;
          jit_instr.dst_type = reg_type;
          reg_info[instr.dst].kind = NumericValueKind::Scalar;
          reg_info[instr.dst].scalar_type = reg_type;
        }
        break;
      case ExprKind::SampleRate:
        jit_instr.op = egress_jit::NumericOp::SampleRate;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      case ExprKind::SampleIndex:
        jit_instr.op = egress_jit::NumericOp::SampleIndex;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
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
          if (!add_array_values_to_jit_table(state.array_storage, packed_values, array_slot))
          {
            return false;
          }
          emit_instruction = false;
        }
        else
        {
          array_slot = allocate_array_slot_with_size(state.array_storage, array_size);
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
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Bool;
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Bool;
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, rhs.array_size);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
        reg_info[instr.dst].scalar_type = infer_jit_scalar_type(ExprKind::Add, lhs.scalar_type, rhs.scalar_type);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
        reg_info[instr.dst].scalar_type = infer_jit_scalar_type(ExprKind::Sub, lhs.scalar_type, rhs.scalar_type);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, rhs.array_size);
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
        reg_info[instr.dst].scalar_type = infer_jit_scalar_type(ExprKind::Mul, lhs.scalar_type, rhs.scalar_type);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Float;
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
        const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.matrix_rows);
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
          const uint32_t dst_slot = allocate_array_slot_with_size(state.array_storage, lhs.array_size);
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
        reg_info[instr.dst].scalar_type = infer_jit_scalar_type(ExprKind::Mod, lhs.scalar_type, rhs.scalar_type);
        break;
      }
      case ExprKind::FloorDiv:
        jit_instr.op = egress_jit::NumericOp::FloorDiv;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
        break;
      case ExprKind::BitAnd:
        jit_instr.op = egress_jit::NumericOp::BitAnd;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
        break;
      case ExprKind::BitOr:
        jit_instr.op = egress_jit::NumericOp::BitOr;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
        break;
      case ExprKind::BitXor:
        jit_instr.op = egress_jit::NumericOp::BitXor;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
        break;
      case ExprKind::LShift:
        jit_instr.op = egress_jit::NumericOp::LShift;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
        break;
      case ExprKind::RShift:
        jit_instr.op = egress_jit::NumericOp::RShift;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
        break;
      case ExprKind::Abs:
        jit_instr.op = egress_jit::NumericOp::Abs;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = reg_info[instr.src_a].scalar_type;
        break;
      case ExprKind::Clamp:
        jit_instr.op = egress_jit::NumericOp::Clamp;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = infer_jit_scalar_type(
          ExprKind::Clamp, reg_info[instr.src_a].scalar_type, reg_info[instr.src_b].scalar_type);
        break;
      case ExprKind::Select:
        jit_instr.op = egress_jit::NumericOp::Select;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = infer_jit_scalar_type(
          ExprKind::Select, reg_info[instr.src_b].scalar_type, reg_info[instr.src_c].scalar_type);
        break;
      case ExprKind::Log:
        jit_instr.op = egress_jit::NumericOp::Log;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Float;
        break;
      case ExprKind::Sin:
        jit_instr.op = egress_jit::NumericOp::Sin;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Float;
        break;
      case ExprKind::Neg:
        jit_instr.op = egress_jit::NumericOp::Neg;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = reg_info[instr.src_a].scalar_type;
        break;
      case ExprKind::BitNot:
        jit_instr.op = egress_jit::NumericOp::BitNot;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Int;
        break;
      case ExprKind::SmoothedParam:
      {
        if (!instr.control_param)
        {
          return false;
        }
        const double tc = instr.control_param->time_const;
        const double coeff = (tc > 0.0)
          ? 1.0 - std::exp(-1.0 / (tc * sample_rate))
          : 1.0;
        jit_instr.op = egress_jit::NumericOp::SmoothedParam;
        jit_instr.literal = coeff;
        jit_instr.param_ptr = reinterpret_cast<uint64_t>(&instr.control_param->value);
        jit_instr.slot_id = instr.slot_id;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        break;
      }
      case ExprKind::ConstructStruct:
      {
        // Allocate consecutive temp slots for each field.
        // Each field is a scalar; emit CopySlot for each.
        if (!registry) return false;
        const auto * type_def = registry->find(instr.type_name);
        if (!type_def || type_def->kind != egress::TypeDef::Kind::Struct)
          return false;
        const uint32_t n_fields = static_cast<uint32_t>(type_def->fields.size());
        if (instr.args.size() < n_fields) return false;

        // Validate all field sources are scalar
        for (uint32_t fi = 0; fi < n_fields; ++fi)
        {
          const uint32_t src = instr.args[fi];
          if (src >= reg_info.size() || reg_info[src].kind != NumericValueKind::Scalar)
            return false;
        }

        // Allocate n_fields new temp slots
        const uint32_t base = numeric_program.register_count;
        numeric_program.register_count += n_fields;
        reg_info.resize(numeric_program.register_count);

        // Emit CopySlot for each field
        for (uint32_t fi = 0; fi < n_fields; ++fi)
        {
          const uint32_t src = instr.args[fi];
          const egress_jit::JitScalarType ftype = type_def->fields[fi].scalar_type;
          egress_jit::NumericInstr copy_instr;
          copy_instr.op = egress_jit::NumericOp::CopySlot;
          copy_instr.dst = base + fi;
          copy_instr.src_a = src;
          copy_instr.src_a_type = ftype;
          copy_instr.dst_type = ftype;
          numeric_program.instructions.push_back(copy_instr);

          reg_info[base + fi].kind = NumericValueKind::Scalar;
          reg_info[base + fi].scalar_type = ftype;
        }

        reg_info[instr.dst].kind = NumericValueKind::CompoundSlot;
        reg_info[instr.dst].compound_base_slot = base;
        emit_instruction = false;
        break;
      }
      case ExprKind::FieldAccess:
      {
        if (!registry) return false;
        const auto * type_def = registry->find(instr.type_name);
        if (!type_def || type_def->kind != egress::TypeDef::Kind::Struct)
          return false;
        const uint32_t field_index = instr.slot_id;
        if (field_index >= type_def->fields.size()) return false;

        if (instr.src_a >= reg_info.size() ||
            reg_info[instr.src_a].kind != NumericValueKind::CompoundSlot)
          return false;

        const uint32_t field_slot = reg_info[instr.src_a].compound_base_slot + field_index;
        if (field_slot >= reg_info.size()) return false;

        const egress_jit::JitScalarType ftype = type_def->fields[field_index].scalar_type;
        jit_instr.op = egress_jit::NumericOp::CopySlot;
        jit_instr.src_a = field_slot;
        jit_instr.src_a_type = ftype;
        jit_instr.dst_type = ftype;
        reg_info[instr.dst].kind = NumericValueKind::Scalar;
        reg_info[instr.dst].scalar_type = ftype;
        require_scalar_inputs = false;  // src_a is compound; we resolved it above
        break;
      }
      case ExprKind::ConstructVariant:
      {
        if (!registry) return false;
        const auto * type_def = registry->find(instr.type_name);
        if (!type_def || type_def->kind != egress::TypeDef::Kind::Sum)
          return false;
        const uint32_t variant_idx = instr.slot_id;
        if (variant_idx >= type_def->variants.size()) return false;
        const auto & variant = type_def->variants[variant_idx];
        const uint32_t n_payload = static_cast<uint32_t>(variant.payload.size());
        if (instr.args.size() < n_payload) return false;

        // Validate all payload sources are scalar
        for (uint32_t pi = 0; pi < n_payload; ++pi)
        {
          const uint32_t src = instr.args[pi];
          if (src >= reg_info.size() || reg_info[src].kind != NumericValueKind::Scalar)
            return false;
        }

        // slot layout: [discriminant, payload_0, payload_1, ...]
        const uint32_t slot_count = type_def->slot_count();
        const uint32_t base = numeric_program.register_count;
        numeric_program.register_count += slot_count;
        reg_info.resize(numeric_program.register_count);

        // Emit Literal(int, variant_idx) → discriminant slot
        {
          egress_jit::NumericInstr disc_instr;
          disc_instr.op = egress_jit::NumericOp::Literal;
          disc_instr.dst = base;
          disc_instr.int_literal = static_cast<int64_t>(variant_idx);
          disc_instr.literal = static_cast<double>(variant_idx);
          disc_instr.dst_type = egress_jit::JitScalarType::Int;
          numeric_program.instructions.push_back(disc_instr);
          reg_info[base].kind = NumericValueKind::Scalar;
          reg_info[base].scalar_type = egress_jit::JitScalarType::Int;
          reg_info[base].scalar_is_constant = true;
          reg_info[base].int_constant = static_cast<int64_t>(variant_idx);
        }

        // Emit CopySlot for each payload field
        for (uint32_t pi = 0; pi < n_payload; ++pi)
        {
          const uint32_t src = instr.args[pi];
          const egress_jit::JitScalarType ftype = variant.payload[pi].scalar_type;
          egress_jit::NumericInstr copy_instr;
          copy_instr.op = egress_jit::NumericOp::CopySlot;
          copy_instr.dst = base + 1 + pi;
          copy_instr.src_a = src;
          copy_instr.src_a_type = ftype;
          copy_instr.dst_type = ftype;
          numeric_program.instructions.push_back(copy_instr);
          reg_info[base + 1 + pi].kind = NumericValueKind::Scalar;
          reg_info[base + 1 + pi].scalar_type = ftype;
        }

        reg_info[instr.dst].kind = NumericValueKind::CompoundSlot;
        reg_info[instr.dst].compound_base_slot = base;
        emit_instruction = false;
        break;
      }
      case ExprKind::MatchVariant:
      {
        if (!registry) return false;
        const auto * type_def = registry->find(instr.type_name);
        if (!type_def || type_def->kind != egress::TypeDef::Kind::Sum)
          return false;
        const uint32_t n_variants = static_cast<uint32_t>(type_def->variants.size());
        if (instr.args.size() < n_variants) return false;

        // Scrutinee must be a CompoundSlot from ConstructVariant
        if (instr.src_a >= reg_info.size() ||
            reg_info[instr.src_a].kind != NumericValueKind::CompoundSlot)
          return false;
        const uint32_t disc_slot = reg_info[instr.src_a].compound_base_slot;

        // Determine max payload size (result slot count)
        uint32_t max_payload = 0;
        for (const auto & v : type_def->variants)
          max_payload = std::max(max_payload, static_cast<uint32_t>(v.payload.size()));

        if (max_payload == 0)
        {
          // No payload: just alias any branch result or return 0
          reg_info[instr.dst].kind = NumericValueKind::Scalar;
          reg_info[instr.dst].scalar_type = egress_jit::JitScalarType::Float;
          jit_instr.op = egress_jit::NumericOp::Literal;
          jit_instr.literal = 0.0;
          jit_instr.dst_type = egress_jit::JitScalarType::Float;
          break;
        }

        // Validate all branch args are CompoundSlot
        for (uint32_t vi = 0; vi < n_variants; ++vi)
        {
          const uint32_t bsrc = instr.args[vi];
          if (bsrc >= reg_info.size() || reg_info[bsrc].kind != NumericValueKind::CompoundSlot)
            return false;
        }

        // Allocate result slots
        const uint32_t result_base = numeric_program.register_count;
        numeric_program.register_count += max_payload;
        reg_info.resize(numeric_program.register_count);

        // For each payload slot j: emit Select chain across all variants
        for (uint32_t j = 0; j < max_payload; ++j)
        {
          uint32_t current_result = 0;
          bool have_result = false;

          for (uint32_t vi = 0; vi < n_variants; ++vi)
          {
            const uint32_t branch_base = reg_info[instr.args[vi]].compound_base_slot;
            const uint32_t n_payload_vi = static_cast<uint32_t>(type_def->variants[vi].payload.size());
            const egress_jit::JitScalarType ftype = (j < n_payload_vi)
              ? type_def->variants[vi].payload[j].scalar_type
              : egress_jit::JitScalarType::Float;
            const uint32_t branch_slot = branch_base + 1 + j; // +1 for discriminant

            if (!have_result)
            {
              // First variant: result_j starts as branch_0_j (CopySlot)
              const uint32_t slot = result_base + j;
              egress_jit::NumericInstr copy_instr;
              copy_instr.op = egress_jit::NumericOp::CopySlot;
              copy_instr.dst = slot;
              copy_instr.src_a = (j < n_payload_vi) ? branch_slot : 0;
              copy_instr.src_a_type = ftype;
              copy_instr.dst_type = ftype;
              numeric_program.instructions.push_back(copy_instr);
              reg_info[slot].kind = NumericValueKind::Scalar;
              reg_info[slot].scalar_type = ftype;
              current_result = slot;
              have_result = true;
            }
            else
            {
              // Subsequent variants: emit Equal(disc_slot, vi) then Select
              // Emit Equal: cond_slot = (disc == vi)
              const uint32_t cond_slot = numeric_program.register_count++;
              reg_info.resize(numeric_program.register_count);
              {
                egress_jit::NumericInstr eq_instr;
                eq_instr.op = egress_jit::NumericOp::Equal;
                eq_instr.dst = cond_slot;
                eq_instr.src_a = disc_slot;
                eq_instr.src_b = numeric_program.register_count; // points to literal below
                eq_instr.src_a_type = egress_jit::JitScalarType::Int;
                eq_instr.src_b_type = egress_jit::JitScalarType::Int;
                eq_instr.dst_type = egress_jit::JitScalarType::Bool;

                // Emit literal vi
                const uint32_t lit_slot = numeric_program.register_count++;
                reg_info.resize(numeric_program.register_count);
                eq_instr.src_b = lit_slot;

                egress_jit::NumericInstr lit_instr;
                lit_instr.op = egress_jit::NumericOp::Literal;
                lit_instr.dst = lit_slot;
                lit_instr.int_literal = static_cast<int64_t>(vi);
                lit_instr.literal = static_cast<double>(vi);
                lit_instr.dst_type = egress_jit::JitScalarType::Int;
                numeric_program.instructions.push_back(lit_instr);
                reg_info[lit_slot].kind = NumericValueKind::Scalar;
                reg_info[lit_slot].scalar_type = egress_jit::JitScalarType::Int;

                numeric_program.instructions.push_back(eq_instr);
                reg_info[cond_slot].kind = NumericValueKind::Scalar;
                reg_info[cond_slot].scalar_type = egress_jit::JitScalarType::Bool;
              }

              // Emit Select: new_result_j = select(cond, branch_i_j, prev_result_j)
              const uint32_t new_result = numeric_program.register_count++;
              reg_info.resize(numeric_program.register_count);
              {
                egress_jit::NumericInstr sel_instr;
                sel_instr.op = egress_jit::NumericOp::Select;
                sel_instr.dst = new_result;
                sel_instr.src_a = cond_slot;
                sel_instr.src_b = (j < n_payload_vi) ? branch_slot : current_result;
                sel_instr.src_c = current_result;
                sel_instr.src_a_type = egress_jit::JitScalarType::Bool;
                sel_instr.src_b_type = ftype;
                sel_instr.src_c_type = ftype;
                sel_instr.dst_type = ftype;
                numeric_program.instructions.push_back(sel_instr);
                reg_info[new_result].kind = NumericValueKind::Scalar;
                reg_info[new_result].scalar_type = ftype;
              }
              current_result = new_result;
            }
          }
        }

        reg_info[instr.dst].kind = NumericValueKind::CompoundSlot;
        reg_info[instr.dst].compound_base_slot = result_base;
        emit_instruction = false;
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
      if (reg_info[instr.dst].kind == NumericValueKind::Scalar)
      {
        jit_instr.dst_type = reg_info[instr.dst].scalar_type;
      }
      if (instr.src_a < reg_info.size() && reg_info[instr.src_a].kind == NumericValueKind::Scalar)
      {
        jit_instr.src_a_type = reg_info[instr.src_a].scalar_type;
      }
      if (instr.src_b < reg_info.size() && reg_info[instr.src_b].kind == NumericValueKind::Scalar)
      {
        jit_instr.src_b_type = reg_info[instr.src_b].scalar_type;
      }
      if (instr.src_c < reg_info.size() && reg_info[instr.src_c].kind == NumericValueKind::Scalar)
      {
        jit_instr.src_c_type = reg_info[instr.src_c].scalar_type;
      }
      numeric_program.instructions.push_back(jit_instr);
    }
  }

  state.output_info.reserve(program.output_targets.size());
  for (uint32_t output_reg : program.output_targets)
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
    output_info.scalar_type = reg_info[output_reg].scalar_type;
    output_info.array_element_type = reg_info[output_reg].array_element_type;
    state.output_info.push_back(output_info);
  }

  for (unsigned int reg_slot = 0; reg_slot < program.register_targets.size(); ++reg_slot)
  {
    const int32_t target = program.register_targets[reg_slot];
    if (target < 0)
    {
      continue;
    }

    if (static_cast<std::size_t>(target) >= reg_info.size())
    {
      return false;
    }

    if (state.register_scalar_mask[reg_slot])
    {
      if (reg_info[target].kind != NumericValueKind::Scalar)
      {
        return false;
      }
      state.register_target_types[reg_slot] = reg_info[target].scalar_type;
      continue;
    }

    if (reg_info[target].kind != NumericValueKind::Array)
    {
      return false;
    }

    const uint32_t dst_slot = state.register_array_slot[reg_slot];
    if (dst_slot >= state.array_storage.size())
    {
      return false;
    }
    if (state.array_storage[dst_slot].size() != reg_info[target].array_size)
    {
      return false;
    }

    state.array_register_targets[reg_slot] = static_cast<int32_t>(reg_info[target].array_slot);
  }

  std::vector<uint32_t> array_target_use_counts(state.array_storage.size(), 0);
  for (unsigned int reg_slot = 0; reg_slot < state.array_register_targets.size(); ++reg_slot)
  {
    const int32_t src_slot = state.array_register_targets[reg_slot];
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

  for (unsigned int reg_slot = 0; reg_slot < state.array_register_targets.size(); ++reg_slot)
  {
    const int32_t src_slot = state.array_register_targets[reg_slot];
    if (src_slot < 0 || state.register_scalar_mask[reg_slot])
    {
      continue;
    }
    const uint32_t dst_slot = state.register_array_slot[reg_slot];
    if (dst_slot >= state.array_storage.size() ||
        static_cast<std::size_t>(src_slot) >= state.array_storage.size())
    {
      return false;
    }
    state.array_register_can_swap[reg_slot] =
      dst_slot != static_cast<uint32_t>(src_slot) &&
      array_target_use_counts[static_cast<std::size_t>(src_slot)] == 1;
  }

  return true;
}

} // namespace egress_runtime
