#include "expr/ExprRewrite.hpp"

#include "expr/ExprEval.hpp"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace expr = egress_expr;
namespace expr_eval = egress_expr_eval;

namespace
{
using ExprKind = expr::ExprKind;
using ExprSpecPtr = expr::ExprSpecPtr;
using ValueType = expr::ValueType;

bool is_zero_expr(const ExprSpecPtr & expr_spec)
{
  return expr_spec != nullptr &&
         expr_spec->kind == ExprKind::Literal &&
         expr_spec->literal.type != ValueType::Array &&
         expr_spec->literal.type != ValueType::Matrix &&
         !expr::is_truthy(expr_spec->literal);
}

bool is_one_expr(const ExprSpecPtr & expr_spec)
{
  return expr_spec != nullptr && expr_spec->kind == ExprKind::Literal &&
         expr_spec->literal.type != ValueType::Array &&
         expr_spec->literal.type != ValueType::Matrix &&
         ((expr_spec->literal.type == ValueType::Bool && expr_spec->literal.bool_value) ||
          (expr_spec->literal.type == ValueType::Int && expr_spec->literal.int_value == 1) ||
          (expr_spec->literal.type == ValueType::Float && expr_spec->literal.float_value == 1.0));
}
}  // namespace

namespace egress_expr_rewrite
{
ExprSpecPtr append_expr(const ExprSpecPtr & lhs, const ExprSpecPtr & rhs)
{
  if (!lhs)
  {
    return rhs;
  }
  if (!rhs)
  {
    return lhs;
  }
  return expr::binary_expr(ExprKind::Add, lhs, rhs);
}

ExprSpecPtr simplify_expr(const ExprSpecPtr & expr_spec)
{
  if (!expr_spec)
  {
    return nullptr;
  }

  switch (expr_spec->kind)
  {
    case ExprKind::Literal:
    case ExprKind::Ref:
    case ExprKind::InputValue:
    case ExprKind::RegisterValue:
    case ExprKind::NestedValue:
    case ExprKind::DelayValue:
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
    case ExprKind::SmoothedParam:
      return expr_spec;
    case ExprKind::ArrayPack:
    {
      std::vector<ExprSpecPtr> items;
      items.reserve(expr_spec->args.size());
      bool all_literal = true;
      std::vector<expr::Value> literal_items;
      literal_items.reserve(expr_spec->args.size());
      for (const auto & arg : expr_spec->args)
      {
        ExprSpecPtr item = simplify_expr(arg);
        if (!item)
        {
          item = expr::literal_expr(0.0);
        }
        if (item->kind != ExprKind::Literal ||
            item->literal.type == ValueType::Array ||
            item->literal.type == ValueType::Matrix)
        {
          all_literal = false;
        }
        else
        {
          literal_items.push_back(item->literal);
        }
        items.push_back(std::move(item));
      }
      if (all_literal)
      {
        return expr::literal_expr(expr::array_value(std::move(literal_items)));
      }
      return expr::array_pack_expr(std::move(items));
    }
    case ExprKind::ArraySet:
    {
      ExprSpecPtr array_expr = simplify_expr(expr_spec->lhs);
      ExprSpecPtr index_expr = simplify_expr(expr_spec->rhs);
      ExprSpecPtr value_expr = simplify_expr(expr_spec->args.empty() ? nullptr : expr_spec->args.front());
      if (!array_expr)
      {
        array_expr = expr::literal_expr(0.0);
      }
      if (!index_expr)
      {
        index_expr = expr::literal_expr(static_cast<int64_t>(0));
      }
      if (!value_expr)
      {
        value_expr = expr::literal_expr(0.0);
      }
      if (array_expr->kind == ExprKind::Literal &&
          index_expr->kind == ExprKind::Literal &&
          value_expr->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::array_set_value(
          array_expr->literal,
          index_expr->literal,
          value_expr->literal));
      }
      if (index_expr->kind == ExprKind::Literal)
      {
        const int64_t raw_index = expr::to_int64(index_expr->literal);
        if (raw_index >= 0)
        {
          const std::size_t item_index = static_cast<std::size_t>(raw_index);
          if (array_expr->kind == ExprKind::Literal && expr::is_array(array_expr->literal))
          {
            if (item_index < array_expr->literal.array_items.size())
            {
              std::vector<ExprSpecPtr> items;
              items.reserve(array_expr->literal.array_items.size());
              for (const auto & item : array_expr->literal.array_items)
              {
                items.push_back(expr::literal_expr(item));
              }
              items[item_index] = value_expr;
              return simplify_expr(expr::array_pack_expr(std::move(items)));
            }
          }
        }
      }
      if (array_expr->kind == ExprKind::ArrayPack &&
          index_expr->kind == ExprKind::Literal)
      {
        const int64_t raw_index = expr::to_int64(index_expr->literal);
        if (raw_index >= 0)
        {
          const std::size_t item_index = static_cast<std::size_t>(raw_index);
          if (item_index < array_expr->args.size())
          {
            std::vector<ExprSpecPtr> items = array_expr->args;
            items[item_index] = value_expr;
            return simplify_expr(expr::array_pack_expr(std::move(items)));
          }
        }
      }
      return expr::array_set_expr(array_expr, index_expr, value_expr);
    }
    case ExprKind::Clamp:
    {
      ExprSpecPtr value = simplify_expr(expr_spec->lhs);
      ExprSpecPtr min_value = simplify_expr(expr_spec->rhs);
      ExprSpecPtr max_value = simplify_expr(expr_spec->args.empty() ? nullptr : expr_spec->args.front());
      if (!value)
      {
        value = expr::literal_expr(0.0);
      }
      if (!min_value)
      {
        min_value = expr::literal_expr(0.0);
      }
      if (!max_value)
      {
        max_value = expr::literal_expr(0.0);
      }
      if (value->kind == ExprKind::Literal && min_value->kind == ExprKind::Literal && max_value->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::clamp_values(value->literal, min_value->literal, max_value->literal));
      }
      return expr::clamp_expr(value, min_value, max_value);
    }
    case ExprKind::Select:
    {
      ExprSpecPtr cond = simplify_expr(expr_spec->lhs);
      ExprSpecPtr then_val = simplify_expr(expr_spec->rhs);
      ExprSpecPtr else_val = simplify_expr(expr_spec->args.empty() ? nullptr : expr_spec->args.front());
      if (!cond)  { cond = expr::literal_expr(0.0); }
      if (!then_val) { then_val = expr::literal_expr(0.0); }
      if (!else_val) { else_val = expr::literal_expr(0.0); }
      if (cond->kind == ExprKind::Literal && then_val->kind == ExprKind::Literal && else_val->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::select_values(cond->literal, then_val->literal, else_val->literal));
      }
      return expr::select_expr(cond, then_val, else_val);
    }
    case ExprKind::Function:
      return expr::function_expr(expr_spec->param_count, simplify_expr(expr_spec->lhs));
    case ExprKind::Call:
    {
      std::vector<ExprSpecPtr> args;
      args.reserve(expr_spec->args.size());
      for (const auto & arg : expr_spec->args)
      {
        args.push_back(simplify_expr(arg));
      }
      return expr::call_expr(simplify_expr(expr_spec->lhs), std::move(args));
    }
    case ExprKind::Index:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs)
      {
        lhs = expr::literal_expr(0.0);
      }
      if (!rhs)
      {
        rhs = expr::literal_expr(static_cast<int64_t>(0));
      }
      if (lhs->kind == ExprKind::Literal &&
          rhs->kind == ExprKind::Literal &&
          expr::is_array(lhs->literal))
      {
        const int64_t raw_index = expr::to_int64(rhs->literal);
        if (raw_index >= 0)
        {
          const std::size_t item_index = static_cast<std::size_t>(raw_index);
          if (item_index < lhs->literal.array_items.size())
          {
            return expr::literal_expr(lhs->literal.array_items[item_index]);
          }
        }
      }
      return expr::binary_expr(ExprKind::Index, lhs, rhs);
    }
    case ExprKind::Neg:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      if (!lhs || is_zero_expr(lhs))
      {
        return nullptr;
      }
      if (lhs->kind == ExprKind::Literal)
      {
        if (lhs->literal.type == ValueType::Float)
        {
          return expr::literal_expr(-lhs->literal.float_value);
        }
        return expr::literal_expr(-expr::to_int64(lhs->literal));
      }
      return expr::unary_expr(ExprKind::Neg, lhs);
    }
    case ExprKind::Abs:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      if (!lhs || is_zero_expr(lhs))
      {
        return nullptr;
      }
      if (lhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::abs_value(lhs->literal));
      }
      return expr::unary_expr(ExprKind::Abs, lhs);
    }
    case ExprKind::Not:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      if (!lhs)
      {
        return expr::literal_expr(true);
      }
      if (lhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::not_value(lhs->literal));
      }
      return expr::unary_expr(ExprKind::Not, lhs);
    }
    case ExprKind::BitNot:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      if (!lhs)
      {
        return nullptr;
      }
      if (lhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(~expr::to_int64(lhs->literal));
      }
      return expr::unary_expr(ExprKind::BitNot, lhs);
    }
    case ExprKind::Log:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      if (!lhs)
      {
        return expr::literal_expr(expr_eval::log_value(expr::float_value(0.0)));
      }
      if (lhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::log_value(lhs->literal));
      }
      return expr::unary_expr(ExprKind::Log, lhs);
    }
    case ExprKind::Sin:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      if (!lhs)
      {
        return nullptr;
      }
      if (lhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(std::sin(expr::to_float64(lhs->literal)));
      }
      return expr::unary_expr(ExprKind::Sin, lhs);
    }
    case ExprKind::Less:
    case ExprKind::LessEqual:
    case ExprKind::Greater:
    case ExprKind::GreaterEqual:
    case ExprKind::Equal:
    case ExprKind::NotEqual:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs)
      {
        lhs = expr::literal_expr(0.0);
      }
      if (!rhs)
      {
        rhs = expr::literal_expr(0.0);
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        switch (expr_spec->kind)
        {
          case ExprKind::Less:
            return expr::literal_expr(expr_eval::less_values(lhs->literal, rhs->literal));
          case ExprKind::LessEqual:
            return expr::literal_expr(expr_eval::less_equal_values(lhs->literal, rhs->literal));
          case ExprKind::Greater:
            return expr::literal_expr(expr_eval::greater_values(lhs->literal, rhs->literal));
          case ExprKind::GreaterEqual:
            return expr::literal_expr(expr_eval::greater_equal_values(lhs->literal, rhs->literal));
          case ExprKind::Equal:
            return expr::literal_expr(expr_eval::equal_values(lhs->literal, rhs->literal));
          case ExprKind::NotEqual:
            return expr::literal_expr(expr_eval::not_equal_values(lhs->literal, rhs->literal));
          default:
            break;
        }
      }
      return expr::binary_expr(expr_spec->kind, lhs, rhs);
    }
    case ExprKind::Add:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs || is_zero_expr(lhs))
      {
        return rhs;
      }
      if (!rhs || is_zero_expr(rhs))
      {
        return lhs;
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::add_values(lhs->literal, rhs->literal));
      }
      return expr::binary_expr(ExprKind::Add, lhs, rhs);
    }
    case ExprKind::Sub:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!rhs || is_zero_expr(rhs))
      {
        return lhs;
      }
      if (!lhs || is_zero_expr(lhs))
      {
        return simplify_expr(expr::unary_expr(ExprKind::Neg, rhs));
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::sub_values(lhs->literal, rhs->literal));
      }
      return expr::binary_expr(ExprKind::Sub, lhs, rhs);
    }
    case ExprKind::Mul:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs || !rhs || is_zero_expr(lhs) || is_zero_expr(rhs))
      {
        return nullptr;
      }
      if (is_one_expr(lhs))
      {
        return rhs;
      }
      if (is_one_expr(rhs))
      {
        return lhs;
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::mul_values(lhs->literal, rhs->literal));
      }
      return expr::binary_expr(ExprKind::Mul, lhs, rhs);
    }
    case ExprKind::MatMul:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs || !rhs)
      {
        return nullptr;
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::matmul_values(lhs->literal, rhs->literal));
      }
      return expr::binary_expr(ExprKind::MatMul, lhs, rhs);
    }
    case ExprKind::Div:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs || is_zero_expr(lhs))
      {
        return nullptr;
      }
      if (!rhs)
      {
        return nullptr;
      }
      if (is_one_expr(rhs))
      {
        return lhs;
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::div_values(lhs->literal, rhs->literal));
      }
      return expr::binary_expr(ExprKind::Div, lhs, rhs);
    }
    case ExprKind::Pow:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs || !rhs)
      {
        return nullptr;
      }
      if (is_zero_expr(rhs))
      {
        return expr::literal_expr(1.0);
      }
      if (is_one_expr(rhs))
      {
        return lhs;
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        return expr::literal_expr(expr_eval::pow_values(lhs->literal, rhs->literal));
      }
      return expr::binary_expr(ExprKind::Pow, lhs, rhs);
    }
    case ExprKind::Mod:
    case ExprKind::FloorDiv:
    case ExprKind::BitAnd:
    case ExprKind::BitOr:
    case ExprKind::BitXor:
    case ExprKind::LShift:
    case ExprKind::RShift:
    {
      ExprSpecPtr lhs = simplify_expr(expr_spec->lhs);
      ExprSpecPtr rhs = simplify_expr(expr_spec->rhs);
      if (!lhs || !rhs)
      {
        return nullptr;
      }
      if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
      {
        switch (expr_spec->kind)
        {
          case ExprKind::Mod:
            return expr::literal_expr(expr_eval::mod_values(lhs->literal, rhs->literal));
          case ExprKind::FloorDiv:
            return expr::literal_expr(expr_eval::floor_div_values(lhs->literal, rhs->literal));
          case ExprKind::BitAnd:
            return expr::literal_expr(expr_eval::bit_and_values(lhs->literal, rhs->literal));
          case ExprKind::BitOr:
            return expr::literal_expr(expr_eval::bit_or_values(lhs->literal, rhs->literal));
          case ExprKind::BitXor:
            return expr::literal_expr(expr_eval::bit_xor_values(lhs->literal, rhs->literal));
          case ExprKind::LShift:
            return expr::literal_expr(expr_eval::lshift_values(lhs->literal, rhs->literal));
          case ExprKind::RShift:
            return expr::literal_expr(expr_eval::rshift_values(lhs->literal, rhs->literal));
          default:
            break;
        }
      }
      return expr::binary_expr(expr_spec->kind, lhs, rhs);
    }
  }

  return expr_spec;
}

ExprSpecPtr replace_refs_with_zero(
  const ExprSpecPtr & expr_spec,
  const std::string & module_name,
  unsigned int output_id,
  bool remove_all_outputs,
  bool & removed_any)
{
  if (!expr_spec)
  {
    return nullptr;
  }

  if (expr_spec->kind == ExprKind::Ref)
  {
    const bool matches = expr_spec->module_name == module_name &&
                         (remove_all_outputs || expr_spec->output_id == output_id);
    if (matches)
    {
      removed_any = true;
      return nullptr;
    }
    return expr_spec;
  }

  if (expr_spec->kind == ExprKind::Literal)
  {
    return expr_spec;
  }

  if (expr_spec->kind == ExprKind::ArrayPack)
  {
    std::vector<ExprSpecPtr> items;
    items.reserve(expr_spec->args.size());
    for (const auto & arg : expr_spec->args)
    {
      items.push_back(replace_refs_with_zero(arg, module_name, output_id, remove_all_outputs, removed_any));
    }
    return expr::array_pack_expr(std::move(items));
  }

  if (expr_spec->kind == ExprKind::ArraySet)
  {
    return simplify_expr(expr::array_set_expr(
      replace_refs_with_zero(expr_spec->lhs, module_name, output_id, remove_all_outputs, removed_any),
      replace_refs_with_zero(expr_spec->rhs, module_name, output_id, remove_all_outputs, removed_any),
      replace_refs_with_zero(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), module_name, output_id, remove_all_outputs, removed_any)));
  }

  if (expr_spec->kind == ExprKind::Clamp)
  {
    return simplify_expr(expr::clamp_expr(
      replace_refs_with_zero(expr_spec->lhs, module_name, output_id, remove_all_outputs, removed_any),
      replace_refs_with_zero(expr_spec->rhs, module_name, output_id, remove_all_outputs, removed_any),
      replace_refs_with_zero(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), module_name, output_id, remove_all_outputs, removed_any)));
  }

  if (expr_spec->kind == ExprKind::Function)
  {
    return expr::function_expr(
      expr_spec->param_count,
      replace_refs_with_zero(expr_spec->lhs, module_name, output_id, remove_all_outputs, removed_any));
  }

  if (expr_spec->kind == ExprKind::Call)
  {
    std::vector<ExprSpecPtr> args;
    args.reserve(expr_spec->args.size());
    for (const auto & arg : expr_spec->args)
    {
      args.push_back(replace_refs_with_zero(arg, module_name, output_id, remove_all_outputs, removed_any));
    }
    return expr::call_expr(
      replace_refs_with_zero(expr_spec->lhs, module_name, output_id, remove_all_outputs, removed_any),
      std::move(args));
  }

  if (expr_spec->kind == ExprKind::InputValue ||
      expr_spec->kind == ExprKind::RegisterValue ||
      expr_spec->kind == ExprKind::NestedValue ||
      expr_spec->kind == ExprKind::DelayValue ||
      expr_spec->kind == ExprKind::SampleRate ||
      expr_spec->kind == ExprKind::SampleIndex)
  {
    return expr_spec;
  }

  ExprSpecPtr lhs = replace_refs_with_zero(expr_spec->lhs, module_name, output_id, remove_all_outputs, removed_any);
  ExprSpecPtr rhs = replace_refs_with_zero(expr_spec->rhs, module_name, output_id, remove_all_outputs, removed_any);

  if (expr_spec->kind == ExprKind::Abs)
  {
    return simplify_expr(expr::unary_expr(ExprKind::Abs, lhs));
  }

  if (expr_spec->kind == ExprKind::Neg)
  {
    return simplify_expr(expr::unary_expr(ExprKind::Neg, lhs));
  }

  if (expr_spec->kind == ExprKind::Not)
  {
    return simplify_expr(expr::unary_expr(ExprKind::Not, lhs));
  }

  if (expr_spec->kind == ExprKind::BitNot)
  {
    return simplify_expr(expr::unary_expr(ExprKind::BitNot, lhs));
  }

  if (expr_spec->kind == ExprKind::Log)
  {
    return simplify_expr(expr::unary_expr(ExprKind::Log, lhs));
  }

  if (expr_spec->kind == ExprKind::Sin)
  {
    return simplify_expr(expr::unary_expr(ExprKind::Sin, lhs));
  }

  return simplify_expr(expr::binary_expr(expr_spec->kind, lhs, rhs));
}

void collect_refs(const ExprSpecPtr & expr_spec, std::vector<OutputRef> & refs)
{
  if (!expr_spec)
  {
    return;
  }

  if (expr_spec->kind == ExprKind::Ref)
  {
    refs.emplace_back(expr_spec->module_name, expr_spec->output_id);
    return;
  }

  if (expr_spec->kind == ExprKind::Function)
  {
    collect_refs(expr_spec->lhs, refs);
    return;
  }

  if (expr_spec->kind == ExprKind::Call)
  {
    collect_refs(expr_spec->lhs, refs);
    for (const auto & arg : expr_spec->args)
    {
      collect_refs(arg, refs);
    }
    return;
  }

  if (expr_spec->kind == ExprKind::ArrayPack)
  {
    for (const auto & arg : expr_spec->args)
    {
      collect_refs(arg, refs);
    }
    return;
  }

  if (expr_spec->kind == ExprKind::ArraySet)
  {
    collect_refs(expr_spec->lhs, refs);
    collect_refs(expr_spec->rhs, refs);
    if (!expr_spec->args.empty())
    {
      collect_refs(expr_spec->args.front(), refs);
    }
    return;
  }

  if (expr_spec->kind == ExprKind::Clamp)
  {
    collect_refs(expr_spec->lhs, refs);
    collect_refs(expr_spec->rhs, refs);
    if (!expr_spec->args.empty())
    {
      collect_refs(expr_spec->args.front(), refs);
    }
    return;
  }

  collect_refs(expr_spec->lhs, refs);
  collect_refs(expr_spec->rhs, refs);
}
}  // namespace egress_expr_rewrite
