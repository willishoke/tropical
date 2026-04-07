#pragma once

#include "expr/Expr.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tropical_expr_rewrite
{
using OutputRef = std::pair<std::string, unsigned int>;

tropical_expr::ExprSpecPtr append_expr(
  const tropical_expr::ExprSpecPtr & lhs,
  const tropical_expr::ExprSpecPtr & rhs);

tropical_expr::ExprSpecPtr simplify_expr(const tropical_expr::ExprSpecPtr & expr);

tropical_expr::ExprSpecPtr replace_refs_with_zero(
  const tropical_expr::ExprSpecPtr & expr,
  const std::string & module_name,
  unsigned int output_id,
  bool remove_all_outputs,
  bool & removed_any);

void collect_refs(
  const tropical_expr::ExprSpecPtr & expr,
  std::vector<OutputRef> & refs);
}  // namespace tropical_expr_rewrite
