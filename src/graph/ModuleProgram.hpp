#pragma once

#include "graph/GraphTypes.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace egress_module_detail
{
struct Instr
{
  ExprKind kind = ExprKind::Literal;
  uint32_t dst = 0;
  uint32_t src_a = 0;
  uint32_t src_b = 0;
  uint32_t src_c = 0;
  unsigned int slot_id = 0;
  unsigned int output_id = 0;
  Value literal;
  std::vector<uint32_t> args;
  // For SmoothedParam: raw non-owning pointer to the control parameter
  egress_expr::ControlParam * control_param = nullptr;
  // For ADT nodes: repurposed module_name from ExprSpec as type_name
  std::string type_name;
};

struct CompiledProgram
{
  std::vector<Instr> instructions;
  std::vector<uint32_t> output_targets;
  std::vector<int32_t> register_targets;
  uint32_t register_count = 0;
};

inline bool is_local_unary(ExprKind kind)
{
  return kind == ExprKind::Abs ||
         kind == ExprKind::Neg ||
         kind == ExprKind::Not ||
         kind == ExprKind::BitNot ||
         kind == ExprKind::Log ||
         kind == ExprKind::Sin;
}

inline bool is_local_ternary(ExprKind kind)
{
  return kind == ExprKind::Clamp || kind == ExprKind::ArraySet || kind == ExprKind::Select;
}

inline bool is_local_binary(ExprKind kind)
{
  switch (kind)
  {
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
    case ExprKind::Index:
      return true;
    default:
      return false;
  }
}
}  // namespace egress_module_detail
