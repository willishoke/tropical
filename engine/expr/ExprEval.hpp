#pragma once

#include "expr/Expr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace egress_expr_eval
{
namespace expr = egress_expr;

inline int normalize_shift(const expr::Value & value)
{
  int64_t shift = expr::to_int64(value);
  if (shift < 0)
  {
    return 0;
  }
  if (shift > 63)
  {
    return 63;
  }
  return static_cast<int>(shift);
}

inline expr::Value add_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    if (expr::arithmetic_type(left, right) == expr::ValueType::Float)
    {
      return expr::float_value(expr::to_float64(left) + expr::to_float64(right));
    }
    return expr::int_value(expr::to_int64(left) + expr::to_int64(right));
  });
}

inline expr::Value sub_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    if (expr::arithmetic_type(left, right) == expr::ValueType::Float)
    {
      return expr::float_value(expr::to_float64(left) - expr::to_float64(right));
    }
    return expr::int_value(expr::to_int64(left) - expr::to_int64(right));
  });
}

inline expr::Value mul_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    if (expr::arithmetic_type(left, right) == expr::ValueType::Float)
    {
      return expr::float_value(expr::to_float64(left) * expr::to_float64(right));
    }
    return expr::int_value(expr::to_int64(left) * expr::to_int64(right));
  });
}

inline expr::Value div_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    const double denominator = expr::to_float64(right);
    return denominator == 0.0 ? expr::float_value(0.0) : expr::float_value(expr::to_float64(left) / denominator);
  });
}

inline expr::Value pow_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::float_value(std::pow(expr::to_float64(left), expr::to_float64(right)));
  });
}

inline expr::Value mod_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    if (expr::arithmetic_type(left, right) == expr::ValueType::Float)
    {
      const double denominator = expr::to_float64(right);
      return denominator == 0.0 ? expr::float_value(0.0)
                                : expr::float_value(std::fmod(expr::to_float64(left), denominator));
    }
    const int64_t denominator = expr::to_int64(right);
    return denominator == 0 ? expr::int_value(0) : expr::int_value(expr::to_int64(left) % denominator);
  });
}

inline expr::Value floor_div_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    const double denominator = expr::to_float64(right);
    return denominator == 0.0 ? expr::float_value(0.0)
                              : expr::float_value(std::floor(expr::to_float64(left) / denominator));
  });
}

inline expr::Value bit_and_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::int_value(expr::to_int64(left) & expr::to_int64(right));
  });
}

inline expr::Value bit_or_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::int_value(expr::to_int64(left) | expr::to_int64(right));
  });
}

inline expr::Value bit_xor_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::int_value(expr::to_int64(left) ^ expr::to_int64(right));
  });
}

inline expr::Value lshift_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::int_value(expr::to_int64(left) << normalize_shift(right));
  });
}

inline expr::Value rshift_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::int_value(expr::to_int64(left) >> normalize_shift(right));
  });
}

inline expr::Value not_value(const expr::Value & value)
{
  return expr::map_unary(value, [](const expr::Value & item) {
    return expr::bool_value(!expr::is_truthy(item));
  });
}

inline expr::Value abs_value(const expr::Value & value)
{
  return expr::map_unary(value, [](const expr::Value & item) {
    if (item.type == expr::ValueType::Float)
    {
      return expr::float_value(std::fabs(expr::to_float64(item)));
    }
    return expr::int_value(std::llabs(expr::to_int64(item)));
  });
}

inline expr::Value neg_value(const expr::Value & value)
{
  return expr::map_unary(value, [](const expr::Value & item) {
    if (item.type == expr::ValueType::Float)
    {
      return expr::float_value(-expr::to_float64(item));
    }
    return expr::int_value(-expr::to_int64(item));
  });
}

inline expr::Value bit_not_value(const expr::Value & value)
{
  return expr::map_unary(value, [](const expr::Value & item) {
    return expr::int_value(~expr::to_int64(item));
  });
}

inline expr::Value clamp_values(const expr::Value & value, const expr::Value & min_value, const expr::Value & max_value)
{
  return expr::map_ternary(value, min_value, max_value, [](const expr::Value & item, const expr::Value & lo, const expr::Value & hi) {
    if (expr::arithmetic_type(item, lo) == expr::ValueType::Float || expr::arithmetic_type(item, hi) == expr::ValueType::Float)
    {
      const double item_value = expr::to_float64(item);
      const double min_scalar = expr::to_float64(lo);
      const double max_scalar = expr::to_float64(hi);
      return expr::float_value(std::fmin(std::fmax(item_value, min_scalar), max_scalar));
    }

    const int64_t item_value = expr::to_int64(item);
    const int64_t min_scalar = expr::to_int64(lo);
    const int64_t max_scalar = expr::to_int64(hi);
    return expr::int_value(std::min(std::max(item_value, min_scalar), max_scalar));
  });
}

inline expr::Value select_values(const expr::Value & cond, const expr::Value & then_val, const expr::Value & else_val)
{
  return expr::map_ternary(cond, then_val, else_val,
    [](const expr::Value & c, const expr::Value & t, const expr::Value & e) {
      return expr::is_truthy(c) ? t : e;
    });
}

inline expr::Value less_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::bool_value(expr::to_float64(left) < expr::to_float64(right));
  });
}

inline expr::Value less_equal_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::bool_value(expr::to_float64(left) <= expr::to_float64(right));
  });
}

inline expr::Value greater_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::bool_value(expr::to_float64(left) > expr::to_float64(right));
  });
}

inline expr::Value greater_equal_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::bool_value(expr::to_float64(left) >= expr::to_float64(right));
  });
}

inline expr::Value equal_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::bool_value(expr::to_float64(left) == expr::to_float64(right));
  });
}

inline expr::Value not_equal_values(const expr::Value & lhs, const expr::Value & rhs)
{
  return expr::map_binary(lhs, rhs, [](const expr::Value & left, const expr::Value & right) {
    return expr::bool_value(expr::to_float64(left) != expr::to_float64(right));
  });
}

inline expr::Value matmul_values(const expr::Value & lhs, const expr::Value & rhs)
{
  if (expr::is_matrix(lhs) && expr::is_matrix(rhs))
  {
    if (lhs.matrix_cols != rhs.matrix_rows)
    {
      throw std::invalid_argument("Matrix shapes do not align for multiplication.");
    }

    const std::size_t rows = lhs.matrix_rows;
    const std::size_t cols = rhs.matrix_cols;
    const std::size_t inner = lhs.matrix_cols;
    std::vector<expr::Value> items;
    items.reserve(rows * cols);

    for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
      {
        double acc = 0.0;
        for (std::size_t k = 0; k < inner; ++k)
        {
          const expr::Value & left = lhs.matrix_items[r * inner + k];
          const expr::Value & right = rhs.matrix_items[k * cols + c];
          acc += expr::to_float64(left) * expr::to_float64(right);
        }
        items.push_back(expr::float_value(acc));
      }
    }

    return expr::matrix_value(rows, cols, std::move(items));
  }

  if (expr::is_matrix(lhs) && expr::is_array(rhs))
  {
    if (lhs.matrix_cols != rhs.array_items.size())
    {
      throw std::invalid_argument("Matrix and vector shapes do not align for multiplication.");
    }

    const std::size_t rows = lhs.matrix_rows;
    const std::size_t cols = lhs.matrix_cols;
    std::vector<expr::Value> items;
    items.reserve(rows);

    for (std::size_t r = 0; r < rows; ++r)
    {
      double acc = 0.0;
      for (std::size_t c = 0; c < cols; ++c)
      {
        const expr::Value & left = lhs.matrix_items[r * cols + c];
        const expr::Value & right = rhs.array_items[c];
        acc += expr::to_float64(left) * expr::to_float64(right);
      }
      items.push_back(expr::float_value(acc));
    }

    return expr::array_value(std::move(items));
  }

  throw std::invalid_argument("MatMul requires matrix * vector or matrix * matrix.");
}

inline expr::Value array_set_value(const expr::Value & array, const expr::Value & index, const expr::Value & value)
{
  if (!expr::is_array(array))
  {
    throw std::invalid_argument("ArraySet requires an array value.");
  }
  if (expr::is_array(value) || expr::is_matrix(value))
  {
    throw std::invalid_argument("ArraySet requires a scalar replacement value.");
  }

  const int64_t raw_index = expr::to_int64(index);
  if (raw_index < 0)
  {
    throw std::out_of_range("Array index out of range.");
  }

  const std::size_t item_index = static_cast<std::size_t>(raw_index);
  if (item_index >= array.array_items.size())
  {
    throw std::out_of_range("Array index out of range.");
  }

  std::vector<expr::Value> items = array.array_items;
  items[item_index] = value;
  return expr::array_value(std::move(items));
}

inline expr::Value sin_value(const expr::Value & value)
{
  return expr::map_unary(value, [](const expr::Value & item) {
    return expr::float_value(std::sin(expr::to_float64(item)));
  });
}

inline expr::Value log_value(const expr::Value & value)
{
  return expr::map_unary(value, [](const expr::Value & item) {
    return expr::float_value(std::log(expr::to_float64(item)));
  });
}
}
