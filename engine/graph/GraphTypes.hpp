#pragma once

#include "expr/Expr.hpp"
#include "expr/ExprEval.hpp"
#include "expr/ExprRewrite.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace expr = tropical_expr;
namespace expr_eval = tropical_expr_eval;

class Module;

using Signal = double;
using inputID = std::pair<std::string, unsigned int>;
using outputID = tropical_expr_rewrite::OutputRef;
using mPtr = std::unique_ptr<Module>;
using ExprValueType = expr::ValueType;
using ExprValue = expr::Value;
using ExprAggregateScalarType = expr::AggregateScalarType;
using ExprKind = expr::ExprKind;
using ExprSpec = expr::ExprSpec;
using ExprSpecPtr = expr::ExprSpecPtr;
using ValueType = ExprValueType;
using Value = ExprValue;
using AggregateScalarType = ExprAggregateScalarType;
using expr::arithmetic_type;
using expr::array_value;
using expr::bool_value;
using expr::float_value;
using expr::int_value;
using expr::is_array;
using expr::is_matrix;
using expr::is_truthy;
using expr::map_binary;
using expr::map_unary;
using expr::to_float64;
using expr::to_int64;
using tropical_expr_rewrite::append_expr;
using tropical_expr_rewrite::collect_refs;
using tropical_expr_rewrite::replace_refs_with_zero;
using tropical_expr_rewrite::simplify_expr;
