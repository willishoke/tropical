#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace egress_expr
{

// ControlParam: lock-free parameter for control-rate values.
// Written from UI/control thread, read per-sample by the DSP evaluator.
// One-pole lowpass smoothing (time_const in seconds) is applied automatically.
// frame_value: written once per frame by the Graph before module processing (used by TriggerParam).
// It is std::atomic<double> because the write happens on the audio thread and the reads happen on
// parallel worker threads. Relaxed ordering is sufficient: the snapshot loop runs before workers
// are dispatched, so the sequencing is established by the worker dispatch barrier, not by frame_value itself.
struct ControlParam
{
  std::atomic<double> value;
  double time_const;
  std::atomic<double> frame_value{0.0};

  ControlParam(double init, double tc) : value(init), time_const(tc) {}

  // Non-copyable (std::atomic is non-copyable)
  ControlParam(const ControlParam &) = delete;
  ControlParam & operator=(const ControlParam &) = delete;
};
enum class ValueType : uint8_t
{
  Int,
  Float,
  Bool,
  Array,
  Matrix
};

enum class AggregateScalarType : uint8_t
{
  None,
  Bool,
  Int,
  Float,
  MixedNumeric,
  NonScalar
};

struct Value
{
  ValueType type = ValueType::Float;
  int64_t int_value = 0;
  double float_value = 0.0;
  bool bool_value = false;
  std::vector<Value> array_items;
  std::vector<Value> matrix_items;
  std::size_t matrix_rows = 0;
  std::size_t matrix_cols = 0;
  AggregateScalarType aggregate_scalar_type = AggregateScalarType::None;
};

enum class ExprKind
{
  Literal,
  Ref,
  InputValue,
  RegisterValue,
  NestedValue,
  DelayValue,
  SampleRate,
  SampleIndex,
  Function,
  Call,
  ArrayPack,
  Index,
  ArraySet,
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
  MatMul,
  Pow,
  Mod,
  FloorDiv,
  BitAnd,
  BitOr,
  BitXor,
  LShift,
  RShift,
  Abs,
  Clamp,
  Log,
  Sin,
  Neg,
  BitNot,
  SmoothedParam,
  Select,
  TriggerParam
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
  // For SmoothedParam: raw non-owning pointer; EgressParam/Param object must outlive module
  ControlParam * control_param = nullptr;
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

inline AggregateScalarType scalar_type_from_value_type(ValueType type)
{
  switch (type)
  {
    case ValueType::Bool:
      return AggregateScalarType::Bool;
    case ValueType::Int:
      return AggregateScalarType::Int;
    case ValueType::Float:
      return AggregateScalarType::Float;
    case ValueType::Array:
    case ValueType::Matrix:
      return AggregateScalarType::NonScalar;
  }
  return AggregateScalarType::NonScalar;
}

inline AggregateScalarType infer_aggregate_scalar_type(const std::vector<Value> & items)
{
  AggregateScalarType result = AggregateScalarType::None;
  for (const Value & item : items)
  {
    const AggregateScalarType item_type = scalar_type_from_value_type(item.type);
    if (item_type == AggregateScalarType::NonScalar)
    {
      return AggregateScalarType::NonScalar;
    }
    if (result == AggregateScalarType::None)
    {
      result = item_type;
      continue;
    }
    if (result != item_type)
    {
      result = AggregateScalarType::MixedNumeric;
    }
  }
  return result;
}

inline bool aggregate_has_numeric_scalars(const Value & value)
{
  if (value.type != ValueType::Array && value.type != ValueType::Matrix)
  {
    return false;
  }
  return value.aggregate_scalar_type != AggregateScalarType::NonScalar;
}

inline Value array_value(std::vector<Value> items)
{
  Value result;
  result.type = ValueType::Array;
  result.array_items = std::move(items);
  result.int_value = static_cast<int64_t>(result.array_items.size());
  result.float_value = static_cast<double>(result.array_items.size());
  result.bool_value = !result.array_items.empty();
  result.aggregate_scalar_type = infer_aggregate_scalar_type(result.array_items);
  return result;
}

inline Value matrix_value(std::size_t rows, std::size_t cols, std::vector<Value> items)
{
  if (rows * cols != items.size())
  {
    throw std::invalid_argument("Matrix size does not match item count.");
  }
  Value result;
  result.type = ValueType::Matrix;
  result.matrix_rows = rows;
  result.matrix_cols = cols;
  result.matrix_items = std::move(items);
  result.int_value = static_cast<int64_t>(rows);
  result.float_value = static_cast<double>(cols);
  result.bool_value = rows != 0 && cols != 0;
  result.aggregate_scalar_type = infer_aggregate_scalar_type(result.matrix_items);
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
    case ValueType::Matrix:
      throw std::invalid_argument("Matrix truthiness is ambiguous.");
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
    case ValueType::Matrix:
      throw std::invalid_argument("Expected scalar float-compatible value, got matrix.");
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
    case ValueType::Matrix:
      throw std::invalid_argument("Expected scalar int-compatible value, got matrix.");
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

inline bool is_matrix(const Value & value)
{
  return value.type == ValueType::Matrix;
}

inline Value array_from_matrix_row(const Value & value, std::size_t row)
{
  if (!is_matrix(value))
  {
    throw std::invalid_argument("Expected matrix value.");
  }
  if (row >= value.matrix_rows)
  {
    throw std::out_of_range("Matrix row out of range.");
  }
  std::vector<Value> items;
  items.reserve(value.matrix_cols);
  const std::size_t offset = row * value.matrix_cols;
  for (std::size_t c = 0; c < value.matrix_cols; ++c)
  {
    items.push_back(value.matrix_items[offset + c]);
  }
  return array_value(std::move(items));
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
      if (is_array(item) || is_matrix(item))
      {
        throw std::invalid_argument("Nested arrays are not supported.");
      }
      items.push_back(func(item));
    }
    return array_value(std::move(items));
  }
  if (is_matrix(value))
  {
    std::vector<Value> items;
    items.reserve(value.matrix_items.size());
    for (const Value & item : value.matrix_items)
    {
      if (is_array(item) || is_matrix(item))
      {
        throw std::invalid_argument("Nested matrices are not supported.");
      }
      items.push_back(func(item));
    }
    return matrix_value(value.matrix_rows, value.matrix_cols, std::move(items));
  }
  return func(value);
}

template <typename Func>
inline Value map_binary(const Value & lhs, const Value & rhs, Func func)
{
  if ((is_matrix(lhs) && is_array(rhs)) || (is_array(lhs) && is_matrix(rhs)))
  {
    throw std::invalid_argument("Matrix and array operations are not supported.");
  }

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
      if (is_array(lhs.array_items[i]) || is_matrix(lhs.array_items[i]) ||
          is_array(rhs.array_items[i]) || is_matrix(rhs.array_items[i]))
      {
        throw std::invalid_argument("Nested arrays are not supported.");
      }
      items.push_back(func(lhs.array_items[i], rhs.array_items[i]));
    }
    return array_value(std::move(items));
  }

  if (is_matrix(lhs) && is_matrix(rhs))
  {
    if (lhs.matrix_rows != rhs.matrix_rows || lhs.matrix_cols != rhs.matrix_cols)
    {
      throw std::invalid_argument("Matrix shapes do not match.");
    }

    std::vector<Value> items;
    items.reserve(lhs.matrix_items.size());
    for (std::size_t i = 0; i < lhs.matrix_items.size(); ++i)
    {
      if (is_array(lhs.matrix_items[i]) || is_matrix(lhs.matrix_items[i]) ||
          is_array(rhs.matrix_items[i]) || is_matrix(rhs.matrix_items[i]))
      {
        throw std::invalid_argument("Nested matrices are not supported.");
      }
      items.push_back(func(lhs.matrix_items[i], rhs.matrix_items[i]));
    }
    return matrix_value(lhs.matrix_rows, lhs.matrix_cols, std::move(items));
  }

  if (is_matrix(lhs))
  {
    std::vector<Value> items;
    items.reserve(lhs.matrix_items.size());
    for (const Value & item : lhs.matrix_items)
    {
      if (is_array(item) || is_matrix(item))
      {
        throw std::invalid_argument("Nested matrices are not supported.");
      }
      items.push_back(func(item, rhs));
    }
    return matrix_value(lhs.matrix_rows, lhs.matrix_cols, std::move(items));
  }

  if (is_matrix(rhs))
  {
    std::vector<Value> items;
    items.reserve(rhs.matrix_items.size());
    for (const Value & item : rhs.matrix_items)
    {
      if (is_array(item) || is_matrix(item))
      {
        throw std::invalid_argument("Nested matrices are not supported.");
      }
      items.push_back(func(lhs, item));
    }
    return matrix_value(rhs.matrix_rows, rhs.matrix_cols, std::move(items));
  }

  if (is_array(lhs))
  {
    std::vector<Value> items;
    items.reserve(lhs.array_items.size());
    for (const Value & item : lhs.array_items)
    {
      if (is_array(item) || is_matrix(item))
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
      if (is_array(item) || is_matrix(item))
      {
        throw std::invalid_argument("Nested arrays are not supported.");
      }
      items.push_back(func(lhs, item));
    }
    return array_value(std::move(items));
  }

  return func(lhs, rhs);
}

template <typename Func>
inline Value map_ternary(const Value & first, const Value & second, const Value & third, Func func)
{
  std::size_t matrix_rows = 0;
  std::size_t matrix_cols = 0;
  bool have_matrix = false;

  for (const Value * value : {&first, &second, &third})
  {
    if (!is_matrix(*value))
    {
      continue;
    }

    if (!have_matrix)
    {
      matrix_rows = value->matrix_rows;
      matrix_cols = value->matrix_cols;
      have_matrix = true;
      continue;
    }

    if (value->matrix_rows != matrix_rows || value->matrix_cols != matrix_cols)
    {
      throw std::invalid_argument("Matrix shapes do not match.");
    }
  }

  if (have_matrix && (is_array(first) || is_array(second) || is_array(third)))
  {
    throw std::invalid_argument("Matrix and array operations are not supported.");
  }

  if (have_matrix)
  {
    const std::size_t item_count = matrix_rows * matrix_cols;
    std::vector<Value> items;
    items.reserve(item_count);
    for (std::size_t i = 0; i < item_count; ++i)
    {
      const Value * first_item = is_matrix(first) ? &first.matrix_items[i] : &first;
      const Value * second_item = is_matrix(second) ? &second.matrix_items[i] : &second;
      const Value * third_item = is_matrix(third) ? &third.matrix_items[i] : &third;

      if (is_array(*first_item) || is_matrix(*first_item) ||
          is_array(*second_item) || is_matrix(*second_item) ||
          is_array(*third_item) || is_matrix(*third_item))
      {
        throw std::invalid_argument("Nested matrices are not supported.");
      }

      items.push_back(func(*first_item, *second_item, *third_item));
    }

    return matrix_value(matrix_rows, matrix_cols, std::move(items));
  }

  std::size_t array_size = 0;
  bool have_array = false;

  for (const Value * value : {&first, &second, &third})
  {
    if (!is_array(*value))
    {
      continue;
    }

    if (!have_array)
    {
      array_size = value->array_items.size();
      have_array = true;
      continue;
    }

    if (value->array_items.size() != array_size)
    {
      throw std::invalid_argument("Array shapes do not match.");
    }
  }

  if (!have_array)
  {
    return func(first, second, third);
  }

  std::vector<Value> items;
  items.reserve(array_size);
  for (std::size_t i = 0; i < array_size; ++i)
  {
    const Value * first_item = is_array(first) ? &first.array_items[i] : &first;
    const Value * second_item = is_array(second) ? &second.array_items[i] : &second;
    const Value * third_item = is_array(third) ? &third.array_items[i] : &third;

    if (is_array(*first_item) || is_matrix(*first_item) ||
        is_array(*second_item) || is_matrix(*second_item) ||
        is_array(*third_item) || is_matrix(*third_item))
    {
      throw std::invalid_argument("Nested arrays are not supported.");
    }

    items.push_back(func(*first_item, *second_item, *third_item));
  }

  return array_value(std::move(items));
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

inline ExprSpecPtr nested_value_expr(unsigned int node_id, unsigned int output_id)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::NestedValue;
  expr->slot_id = node_id;
  expr->output_id = output_id;
  return expr;
}

inline ExprSpecPtr delay_value_expr(unsigned int node_id)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::DelayValue;
  expr->slot_id = node_id;
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

inline ExprSpecPtr array_set_expr(ExprSpecPtr array_expr, ExprSpecPtr index_expr, ExprSpecPtr value_expr)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::ArraySet;
  expr->lhs = std::move(array_expr);
  expr->rhs = std::move(index_expr);
  expr->args.push_back(std::move(value_expr));
  return expr;
}

inline ExprSpecPtr clamp_expr(ExprSpecPtr value_expr, ExprSpecPtr min_expr, ExprSpecPtr max_expr)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::Clamp;
  expr->lhs = std::move(value_expr);
  expr->rhs = std::move(min_expr);
  expr->args.push_back(std::move(max_expr));
  return expr;
}

inline ExprSpecPtr select_expr(ExprSpecPtr cond_expr, ExprSpecPtr then_expr, ExprSpecPtr else_expr)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::Select;
  expr->lhs = std::move(cond_expr);
  expr->rhs = std::move(then_expr);
  expr->args.push_back(std::move(else_expr));
  return expr;
}

// Creates a SmoothedParam expression. The ControlParam pointer is non-owning;
// the caller (EgressParam) must ensure lifetime extends past any module using this expr.
inline ExprSpecPtr smoothed_param_expr(ControlParam * param)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::SmoothedParam;
  expr->control_param = param;
  return expr;
}

// Creates a TriggerParam expression. The Graph snapshots frame_value once per frame
// (via exchange) before module processing, so all modules see the same trigger state.
inline ExprSpecPtr trigger_param_expr(ControlParam * param)
{
  auto expr = std::make_shared<ExprSpec>();
  expr->kind = ExprKind::TriggerParam;
  expr->control_param = param;
  return expr;
}
}  // namespace egress_expr
