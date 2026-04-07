#pragma once

#include "expr/Expr.hpp"

#include <cstddef>
#include <unordered_map>

namespace tropical_expr_inline
{
std::size_t structural_hash(
  const tropical_expr::ExprSpecPtr & expr,
  std::unordered_map<const tropical_expr::ExprSpec *, std::size_t> & cache);

bool structural_equal(
  const tropical_expr::ExprSpecPtr & lhs,
  const tropical_expr::ExprSpecPtr & rhs);

tropical_expr::ExprSpecPtr inline_functions(
  const tropical_expr::ExprSpecPtr & expr,
  unsigned int inline_depth = 0);
}  // namespace tropical_expr_inline
