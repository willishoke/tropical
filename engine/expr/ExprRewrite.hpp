#pragma once

#include "expr/Expr.hpp"

#include <string>
#include <utility>
#include <vector>

namespace egress_expr_rewrite
{
using OutputRef = std::pair<std::string, unsigned int>;

egress_expr::ExprSpecPtr append_expr(
  const egress_expr::ExprSpecPtr & lhs,
  const egress_expr::ExprSpecPtr & rhs);

egress_expr::ExprSpecPtr simplify_expr(const egress_expr::ExprSpecPtr & expr);

egress_expr::ExprSpecPtr replace_refs_with_zero(
  const egress_expr::ExprSpecPtr & expr,
  const std::string & module_name,
  unsigned int output_id,
  bool remove_all_outputs,
  bool & removed_any);

void collect_refs(
  const egress_expr::ExprSpecPtr & expr,
  std::vector<OutputRef> & refs);
}  // namespace egress_expr_rewrite
