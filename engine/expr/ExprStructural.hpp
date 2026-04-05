#pragma once

#include "expr/Expr.hpp"

#include <cstddef>
#include <unordered_map>

namespace egress_expr_inline
{
std::size_t structural_hash(
  const egress_expr::ExprSpecPtr & expr,
  std::unordered_map<const egress_expr::ExprSpec *, std::size_t> & cache);

bool structural_equal(
  const egress_expr::ExprSpecPtr & lhs,
  const egress_expr::ExprSpecPtr & rhs);

egress_expr::ExprSpecPtr inline_functions(
  const egress_expr::ExprSpecPtr & expr,
  unsigned int inline_depth = 0);
}  // namespace egress_expr_inline
