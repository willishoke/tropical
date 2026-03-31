#pragma once

#include "graph/GraphTypes.hpp"
#include "jit/OrcJitEngine.hpp"

#include <cstdint>

namespace egress_module_detail
{
struct NumericInputInfo
{
  bool is_scalar = true;
  uint32_t array_slot = 0;
  uint32_t array_size = 0;
  egress_jit::JitScalarType scalar_type = egress_jit::JitScalarType::Float;
  egress_jit::JitScalarType array_element_type = egress_jit::JitScalarType::Float;
};

struct NumericOutputInfo
{
  uint8_t kind = 0;
  uint32_t array_slot = 0;
  uint32_t matrix_rows = 0;
  uint32_t matrix_cols = 0;
  egress_jit::JitScalarType scalar_type = egress_jit::JitScalarType::Float;
  egress_jit::JitScalarType array_element_type = egress_jit::JitScalarType::Float;
};

enum class NumericValueKind
{
  Scalar,
  Array,
  Matrix,
  CompoundSlot
};

struct NumericValueRef
{
  NumericValueKind kind = NumericValueKind::Scalar;
  uint32_t scalar_register = 0;
  uint32_t array_slot = 0;
  uint32_t array_size = 0;
  uint32_t matrix_rows = 0;
  uint32_t matrix_cols = 0;
};

struct NumericRegInfo
{
  NumericValueKind kind = NumericValueKind::Scalar;
  uint32_t array_slot = 0;
  uint32_t array_size = 0;
  uint32_t matrix_rows = 0;
  uint32_t matrix_cols = 0;
  bool scalar_is_constant = false;
  double scalar_constant = 0.0;
  int64_t int_constant = 0;
  egress_jit::JitScalarType scalar_type = egress_jit::JitScalarType::Float;
  egress_jit::JitScalarType array_element_type = egress_jit::JitScalarType::Float;
  uint16_t compound_type_id = 0xFFFF;
  uint32_t compound_base_slot = 0;
};
}  // namespace egress_module_detail
