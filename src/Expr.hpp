#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace egress_expr
{
enum class ValueType : uint8_t
{
  Int,
  Float,
  Bool,
  Array
};

struct Value
{
  ValueType type = ValueType::Float;
  int64_t int_value = 0;
  double float_value = 0.0;
  bool bool_value = false;
  std::vector<Value> array_items;
};

enum class ExprKind
{
  Literal,
  Ref,
  InputValue,
  RegisterValue,
  SampleRate,
  SampleIndex,
  Function,
  Call,
  ArrayPack,
  Index,
  Not,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Equal,
  NotEqual,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Mod,
  FloorDiv,
  BitAnd,
  BitOr,
  BitXor,
  LShift,
  RShift,
  Sin,
  Neg,
  BitNot
};

struct ExprSpec
{
  ExprKind kind = ExprKind::Literal;
  Value literal;
  std::string module_name;
  unsigned int output_id = 0;
  unsigned int slot_id = 0;
  unsigned int param_count = 0;
  std::shared_ptr<ExprSpec> lhs;
  std::shared_ptr<ExprSpec> rhs;
  std::vector<std::shared_ptr<ExprSpec>> args;
};

using ExprSpecPtr = std::shared_ptr<ExprSpec>;

inline Value int_value(int64_t value)
{
  Value result;
  result.type = ValueType::Int;
  result.int_value = value;
  result.float_value = static_cast<double>(value);
  result.bool_value = value != 0;
  return result;
}

inline Value float_value(double value)
{
  Value result;
  result.type = ValueType::Float;
  result.int_value = static_cast<int64_t>(value);
  result.float_value = value;
  result.bool_value = value != 0.0;
  return result;
}

inline Value bool_value(bool value)
{
  Value result;
  result.type = ValueType::Bool;
  result.int_value = value ? 1 : 0;
  result.float_value = value ? 1.0 : 0.0;
  result.bool_value = value;
  return result;
}

inline Value array_value(std::vector<Value> items)
{
  Value result;
  result.type = ValueType::Array;
  result.array_items = std::move(items);
  result.int_value = static_cast<int64_t>(result.array_items.size());
  result.float_value = static_cast<double>(result.array_items.size());
  result.bool_value = !result.array_items.empty();
  return result;
}

inline bool is_truthy(const Value & value)
{
  switch (value.type)
  {
    case ValueType::Bool:
      return value.bool_value;
    case ValueType::Int:
      return value.int_value != 0;
    case ValueType::Float:
      return value.float_value != 0.0;
    case ValueType::Array:
      throw std::invalid_argument("Array truthiness is ambiguous.");
  }
  return false;
}

inline double to_float64(const Value & value)
{
  switch (value.type)
  {
    case ValueType::Bool:
      return value.bool_value ? 1.0 : 0.0;
    case ValueType::Int:
      return static_cast<double>(value.int_value);
    case ValueType::Float:
      return value.float_value;
    case ValueType::Array:
      throw std::invalid_argument("Expected scalar float-compatible value, got array.");
  }
  return 0.0;
}

inline int64_t to_int64(const Value & value)
{
  switch (value.type)
  {
    case ValueType::Bool:
      return value.bool_value ? 1 : 0;
    case ValueType::Int:
      return value.int_value;
    case ValueType::Float:
      return static_cast<int64_t>(value.float_value);
    case ValueType::Array:
      throw std::invalid_argument("Expected scalar int-compatible value, got array.");
  }
  return 0;
}

inline ValueType arithmetic_type(const Value & lhs, const Value & rhs)
{
  return lhs.type == ValueType::Float || rhs.type == ValueType::Float ? ValueType::Float : ValueType::Int;
}

inline bool is_array(const Value & value)
{
  return value.type == ValueType::Array;
}

template <typename Func>
inline Value map_unary(const Value & value, Func func)
{
  if (is_array(value))
  {
    std::vector<Value> items;
    items.reserve(value.array_items.size());
    for (const Value & item : value.array_items)
    {
      if (is_array(item))
      {
        throw std::invalid_argument("Nested arrays are not supported.");
      }
      items.push_back(func(item));
    }
    return array_value(std::move(items));
  }
  return func(value);
}

template <typename Func>
inline Value map_binary(const Value & lhs, const Value & rhs, Func func)
{
  if (is_array(lhs) && is_array(rhs))
  {
    if (lhs.array_items.size() != rhs.array_items.size())
    {
      throw std::invalid_argument("Array shapes do not match.");
    }

    std::vector<Value> items;
    items.reserve(lhs.array_items.size());
    for (std::size_t i = 0; i < lhs.array_items.size(); ++i)
    {
      if (is_array(lhs.array_items[i]) || is_array(rhs.array_items[i]))
      {
        throw std::invalid_argument("Nested arrays are not supported.");
      }
      items.push_back(func(lhs.array_items[i], rhs.array_items[i]));
    }
    return array_value(std::move(items));
  }

  if (is_array(lhs))
  {
    std::vector<Value> items;
    items.reserve(lhs.array_items.size());
    for (const Value & item : lhs.array_items)
    {
      if (is_array(item))
      {
        throw std::invalid_argument("Nested arrays are not supported.");
      }
      items.push_back(func(item, rhs));
    }
    return array_value(std::move(items));
  }

  if (is_array(rhs))
  {
    std::vector<Value> items;
    items.reserve(rhs.array_items.size());
    for (const Value & item : rhs.array_items)
    {
      if (is_array(item))
      {
        throw std::invalid_argument("Nested arrays are not supported.");
      }
      items.push_back(func(lhs, item));
    }
    return array_value(std::move(items));
  }

  return func(lhs, rhs);
}

inline ExprSpecPtr literal_expr(Value value)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::Literal;
  expr->literal = std::move(value);
  return expr;
}

inline ExprSpecPtr literal_expr(double value)
{
  return literal_expr(float_value(value));
}

inline ExprSpecPtr literal_expr(int64_t value)
{
  return literal_expr(int_value(value));
}

inline ExprSpecPtr literal_expr(bool value)
{
  return literal_expr(bool_value(value));
}

inline ExprSpecPtr ref_expr(std::string module_name, unsigned int output_id)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::Ref;
  expr->module_name = std::move(module_name);
  expr->output_id = output_id;
  return expr;
}

inline ExprSpecPtr input_value_expr(unsigned int input_id)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::InputValue;
  expr->slot_id = input_id;
  return expr;
}

inline ExprSpecPtr register_value_expr(unsigned int register_id)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::RegisterValue;
  expr->slot_id = register_id;
  return expr;
}

inline ExprSpecPtr sample_rate_expr()
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::SampleRate;
  return expr;
}

inline ExprSpecPtr sample_index_expr()
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::SampleIndex;
  return expr;
}

inline ExprSpecPtr function_expr(unsigned int param_count, ExprSpecPtr body)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::Function;
  expr->param_count = param_count;
  expr->lhs = std::move(body);
  return expr;
}

inline ExprSpecPtr call_expr(ExprSpecPtr callee, std::vector<ExprSpecPtr> args)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::Call;
  expr->lhs = std::move(callee);
  expr->args = std::move(args);
  return expr;
}

inline ExprSpecPtr unary_expr(ExprKind kind, ExprSpecPtr operand)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = kind;
  expr->lhs = std::move(operand);
  return expr;
}

inline ExprSpecPtr binary_expr(ExprKind kind, ExprSpecPtr lhs, ExprSpecPtr rhs)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = kind;
  expr->lhs = std::move(lhs);
  expr->rhs = std::move(rhs);
  return expr;
}

inline ExprSpecPtr array_pack_expr(std::vector<ExprSpecPtr> items)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::ArrayPack;
  expr->args = std::move(items);
  return expr;
}

inline ExprSpecPtr index_expr(ExprSpecPtr array_expr, ExprSpecPtr index_expr)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::Index;
  expr->lhs = std::move(array_expr);
  expr->rhs = std::move(index_expr);
  return expr;
}
}  // namespace egress_expr
