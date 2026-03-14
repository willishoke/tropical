#pragma once

#include "graph/GraphTypes.hpp"

#include <cstdint>

namespace egress_module_detail
{
struct NumericInputInfo
{
  bool is_scalar = true;
  uint32_t array_slot = 0;
  uint32_t array_size = 0;
};

struct NumericOutputInfo
{
  uint8_t kind = 0;
  uint32_t array_slot = 0;
  uint32_t matrix_rows = 0;
  uint32_t matrix_cols = 0;
};

enum class NumericValueKind
{
  Scalar,
  Array,
  Matrix
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
};
}  // namespace egress_module_detail
