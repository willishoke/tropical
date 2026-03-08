#pragma once

#include "Expr.hpp"

#include <cmath>

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

inline double fast_sin(double x)
{
  constexpr double pi = 3.14159265358979323846;
  constexpr double two_pi = 2.0 * pi;
  constexpr double B = 4.0 / pi;
  constexpr double C = -4.0 / (pi * pi);
  constexpr double P = 0.225;

  x = std::fmod(x + pi, two_pi);
  if (x < 0.0)
  {
    x += two_pi;
  }
  x -= pi;

  const double y = B * x + C * x * std::fabs(x);
  return P * (y * std::fabs(y) - y) + y;
}

inline expr::Value sin_value(const expr::Value & value)
{
  return expr::map_unary(value, [](const expr::Value & item) {
    return expr::float_value(fast_sin(expr::to_float64(item)));
  });
}
}
