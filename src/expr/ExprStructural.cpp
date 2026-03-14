#include "expr/ExprStructural.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace expr = egress_expr;

namespace
{
using ExprKind = expr::ExprKind;
using ExprSpec = expr::ExprSpec;
using ExprSpecPtr = expr::ExprSpecPtr;
using Value = expr::Value;
using ValueType = expr::ValueType;

std::size_t hash_mix(std::size_t seed, std::size_t value)
{
  constexpr std::size_t kMul = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
  seed ^= value + kMul + (seed << 6) + (seed >> 2);
  return seed;
}

std::size_t hash_value(const Value & value)
{
  std::size_t seed = static_cast<std::size_t>(value.type);
  switch (value.type)
  {
    case ValueType::Int:
      return hash_mix(seed, std::hash<int64_t>{}(value.int_value));
    case ValueType::Float:
      return hash_mix(seed, std::hash<double>{}(value.float_value));
    case ValueType::Bool:
      return hash_mix(seed, std::hash<bool>{}(value.bool_value));
    case ValueType::Array:
      seed = hash_mix(seed, std::hash<std::size_t>{}(value.array_items.size()));
      for (const Value & item : value.array_items)
      {
        seed = hash_mix(seed, hash_value(item));
      }
      return seed;
    case ValueType::Matrix:
      seed = hash_mix(seed, std::hash<std::size_t>{}(value.matrix_rows));
      seed = hash_mix(seed, std::hash<std::size_t>{}(value.matrix_cols));
      for (const Value & item : value.matrix_items)
      {
        seed = hash_mix(seed, hash_value(item));
      }
      return seed;
  }
  return seed;
}

bool value_equal(const Value & lhs, const Value & rhs)
{
  if (lhs.type != rhs.type)
  {
    return false;
  }

  switch (lhs.type)
  {
    case ValueType::Int:
      return lhs.int_value == rhs.int_value;
    case ValueType::Float:
      return lhs.float_value == rhs.float_value;
    case ValueType::Bool:
      return lhs.bool_value == rhs.bool_value;
    case ValueType::Array:
      if (lhs.array_items.size() != rhs.array_items.size())
      {
        return false;
      }
      for (std::size_t i = 0; i < lhs.array_items.size(); ++i)
      {
        if (!value_equal(lhs.array_items[i], rhs.array_items[i]))
        {
          return false;
        }
      }
      return true;
    case ValueType::Matrix:
      if (lhs.matrix_rows != rhs.matrix_rows || lhs.matrix_cols != rhs.matrix_cols)
      {
        return false;
      }
      if (lhs.matrix_items.size() != rhs.matrix_items.size())
      {
        return false;
      }
      for (std::size_t i = 0; i < lhs.matrix_items.size(); ++i)
      {
        if (!value_equal(lhs.matrix_items[i], rhs.matrix_items[i]))
        {
          return false;
        }
      }
      return true;
  }

  return false;
}

bool is_pure_function_body(const ExprSpecPtr & expr_spec, unsigned int param_count)
{
  if (!expr_spec)
  {
    return true;
  }

  switch (expr_spec->kind)
  {
    case ExprKind::Ref:
    case ExprKind::RegisterValue:
    case ExprKind::NestedValue:
    case ExprKind::SampleIndex:
      return false;
    case ExprKind::InputValue:
      return expr_spec->slot_id < param_count;
    case ExprKind::Function:
      return false;
    case ExprKind::Call:
      if (!is_pure_function_body(expr_spec->lhs, param_count))
      {
        return false;
      }
      for (const auto & arg : expr_spec->args)
      {
        if (!is_pure_function_body(arg, param_count))
        {
          return false;
        }
      }
      return true;
    case ExprKind::Clamp:
      return is_pure_function_body(expr_spec->lhs, param_count) &&
             is_pure_function_body(expr_spec->rhs, param_count) &&
             is_pure_function_body(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), param_count);
    case ExprKind::ArraySet:
      return is_pure_function_body(expr_spec->lhs, param_count) &&
             is_pure_function_body(expr_spec->rhs, param_count) &&
             is_pure_function_body(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), param_count);
    case ExprKind::ArrayPack:
      for (const auto & arg : expr_spec->args)
      {
        if (!is_pure_function_body(arg, param_count))
        {
          return false;
        }
      }
      return true;
    default:
      break;
  }

  return is_pure_function_body(expr_spec->lhs, param_count) &&
         is_pure_function_body(expr_spec->rhs, param_count);
}

ExprSpecPtr clone_with_subst(
  const ExprSpecPtr & expr_spec,
  const std::vector<ExprSpecPtr> & args)
{
  if (!expr_spec)
  {
    return nullptr;
  }

  switch (expr_spec->kind)
  {
    case ExprKind::InputValue:
      if (expr_spec->slot_id >= args.size())
      {
        throw std::invalid_argument("Function argument index out of range.");
      }
      return args[expr_spec->slot_id];
    case ExprKind::Literal:
    case ExprKind::Ref:
    case ExprKind::RegisterValue:
    case ExprKind::NestedValue:
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
      return expr_spec;
    case ExprKind::Function:
      return expr::function_expr(expr_spec->param_count, clone_with_subst(expr_spec->lhs, args));
    case ExprKind::Call:
    {
      std::vector<ExprSpecPtr> call_args;
      call_args.reserve(expr_spec->args.size());
      for (const auto & arg : expr_spec->args)
      {
        call_args.push_back(clone_with_subst(arg, args));
      }
      return expr::call_expr(clone_with_subst(expr_spec->lhs, args), std::move(call_args));
    }
    case ExprKind::ArrayPack:
    {
      std::vector<ExprSpecPtr> items;
      items.reserve(expr_spec->args.size());
      for (const auto & arg : expr_spec->args)
      {
        items.push_back(clone_with_subst(arg, args));
      }
      return expr::array_pack_expr(std::move(items));
    }
    case ExprKind::Clamp:
      return expr::clamp_expr(
        clone_with_subst(expr_spec->lhs, args),
        clone_with_subst(expr_spec->rhs, args),
        clone_with_subst(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), args));
    case ExprKind::ArraySet:
      return expr::array_set_expr(
        clone_with_subst(expr_spec->lhs, args),
        clone_with_subst(expr_spec->rhs, args),
        clone_with_subst(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), args));
    case ExprKind::Index:
      return expr::index_expr(clone_with_subst(expr_spec->lhs, args), clone_with_subst(expr_spec->rhs, args));
    default:
      break;
  }

  if (expr_spec->kind == ExprKind::Abs ||
      expr_spec->kind == ExprKind::Neg ||
      expr_spec->kind == ExprKind::Not ||
      expr_spec->kind == ExprKind::BitNot ||
      expr_spec->kind == ExprKind::Log ||
      expr_spec->kind == ExprKind::Sin)
  {
    return expr::unary_expr(expr_spec->kind, clone_with_subst(expr_spec->lhs, args));
  }

  return expr::binary_expr(
    expr_spec->kind,
    clone_with_subst(expr_spec->lhs, args),
    clone_with_subst(expr_spec->rhs, args));
}
}  // namespace

namespace egress_expr_inline
{
std::size_t structural_hash(
  const ExprSpecPtr & expr_spec,
  std::unordered_map<const ExprSpec *, std::size_t> & cache)
{
  if (!expr_spec)
  {
    return static_cast<std::size_t>(0x51ed270bu);
  }

  auto it = cache.find(expr_spec.get());
  if (it != cache.end())
  {
    return it->second;
  }

  std::size_t seed = static_cast<std::size_t>(expr_spec->kind);
  switch (expr_spec->kind)
  {
    case ExprKind::Literal:
      seed = hash_mix(seed, hash_value(expr_spec->literal));
      break;
    case ExprKind::Ref:
      seed = hash_mix(seed, std::hash<std::string>{}(expr_spec->module_name));
      seed = hash_mix(seed, std::hash<unsigned int>{}(expr_spec->output_id));
      break;
    case ExprKind::InputValue:
    case ExprKind::RegisterValue:
    case ExprKind::NestedValue:
      seed = hash_mix(seed, std::hash<unsigned int>{}(expr_spec->slot_id));
      if (expr_spec->kind == ExprKind::NestedValue)
      {
        seed = hash_mix(seed, std::hash<unsigned int>{}(expr_spec->output_id));
      }
      break;
    case ExprKind::Function:
      seed = hash_mix(seed, std::hash<unsigned int>{}(expr_spec->param_count));
      seed = hash_mix(seed, structural_hash(expr_spec->lhs, cache));
      break;
    case ExprKind::Call:
      seed = hash_mix(seed, structural_hash(expr_spec->lhs, cache));
      seed = hash_mix(seed, std::hash<std::size_t>{}(expr_spec->args.size()));
      for (const auto & arg : expr_spec->args)
      {
        seed = hash_mix(seed, structural_hash(arg, cache));
      }
      break;
    case ExprKind::Clamp:
      seed = hash_mix(seed, structural_hash(expr_spec->lhs, cache));
      seed = hash_mix(seed, structural_hash(expr_spec->rhs, cache));
      seed = hash_mix(seed, structural_hash(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), cache));
      break;
    case ExprKind::ArraySet:
      seed = hash_mix(seed, structural_hash(expr_spec->lhs, cache));
      seed = hash_mix(seed, structural_hash(expr_spec->rhs, cache));
      seed = hash_mix(seed, structural_hash(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), cache));
      break;
    case ExprKind::ArrayPack:
    {
      seed = hash_mix(seed, std::hash<std::size_t>{}(expr_spec->args.size()));
      for (const auto & arg : expr_spec->args)
      {
        seed = hash_mix(seed, structural_hash(arg, cache));
      }
      break;
    }
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
      break;
    default:
      seed = hash_mix(seed, structural_hash(expr_spec->lhs, cache));
      seed = hash_mix(seed, structural_hash(expr_spec->rhs, cache));
      break;
  }

  cache.emplace(expr_spec.get(), seed);
  return seed;
}

bool structural_equal(const ExprSpecPtr & lhs, const ExprSpecPtr & rhs)
{
  if (lhs == rhs)
  {
    return true;
  }
  if (!lhs || !rhs || lhs->kind != rhs->kind)
  {
    return false;
  }

  switch (lhs->kind)
  {
    case ExprKind::Literal:
      if (lhs->literal.type != rhs->literal.type)
      {
        return false;
      }
      switch (lhs->literal.type)
      {
        case ValueType::Int:
          return lhs->literal.int_value == rhs->literal.int_value;
        case ValueType::Float:
          return lhs->literal.float_value == rhs->literal.float_value;
        case ValueType::Bool:
          return lhs->literal.bool_value == rhs->literal.bool_value;
        case ValueType::Array:
          if (lhs->literal.array_items.size() != rhs->literal.array_items.size())
          {
            return false;
          }
          for (std::size_t i = 0; i < lhs->literal.array_items.size(); ++i)
          {
            if (!value_equal(lhs->literal.array_items[i], rhs->literal.array_items[i]))
            {
              return false;
            }
          }
          return true;
        case ValueType::Matrix:
          if (lhs->literal.matrix_rows != rhs->literal.matrix_rows ||
              lhs->literal.matrix_cols != rhs->literal.matrix_cols ||
              lhs->literal.matrix_items.size() != rhs->literal.matrix_items.size())
          {
            return false;
          }
          for (std::size_t i = 0; i < lhs->literal.matrix_items.size(); ++i)
          {
            if (!value_equal(lhs->literal.matrix_items[i], rhs->literal.matrix_items[i]))
            {
              return false;
            }
          }
          return true;
      }
      return true;
    case ExprKind::Ref:
      return lhs->module_name == rhs->module_name && lhs->output_id == rhs->output_id;
    case ExprKind::InputValue:
    case ExprKind::RegisterValue:
      return lhs->slot_id == rhs->slot_id;
    case ExprKind::NestedValue:
      return lhs->slot_id == rhs->slot_id && lhs->output_id == rhs->output_id;
    case ExprKind::Function:
      return lhs->param_count == rhs->param_count && structural_equal(lhs->lhs, rhs->lhs);
    case ExprKind::Call:
      if (!structural_equal(lhs->lhs, rhs->lhs) || lhs->args.size() != rhs->args.size())
      {
        return false;
      }
      for (std::size_t i = 0; i < lhs->args.size(); ++i)
      {
        if (!structural_equal(lhs->args[i], rhs->args[i]))
        {
          return false;
        }
      }
      return true;
    case ExprKind::Clamp:
      return structural_equal(lhs->lhs, rhs->lhs) &&
             structural_equal(lhs->rhs, rhs->rhs) &&
             structural_equal(lhs->args.empty() ? nullptr : lhs->args.front(), rhs->args.empty() ? nullptr : rhs->args.front());
    case ExprKind::ArraySet:
      return structural_equal(lhs->lhs, rhs->lhs) &&
             structural_equal(lhs->rhs, rhs->rhs) &&
             structural_equal(lhs->args.empty() ? nullptr : lhs->args.front(), rhs->args.empty() ? nullptr : rhs->args.front());
    case ExprKind::ArrayPack:
    {
      if (lhs->args.size() != rhs->args.size())
      {
        return false;
      }
      for (std::size_t i = 0; i < lhs->args.size(); ++i)
      {
        if (!structural_equal(lhs->args[i], rhs->args[i]))
        {
          return false;
        }
      }
      return true;
    }
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
      return true;
    default:
      return structural_equal(lhs->lhs, rhs->lhs) && structural_equal(lhs->rhs, rhs->rhs);
  }
}

ExprSpecPtr inline_functions(const ExprSpecPtr & expr_spec, unsigned int inline_depth)
{
  if (!expr_spec)
  {
    return nullptr;
  }

  if (inline_depth > 32)
  {
    throw std::invalid_argument("Function inlining exceeded maximum depth.");
  }

  switch (expr_spec->kind)
  {
    case ExprKind::Function:
    {
      ExprSpecPtr body = inline_functions(expr_spec->lhs, inline_depth);
      return expr::function_expr(expr_spec->param_count, body);
    }
    case ExprKind::Call:
    {
      ExprSpecPtr callee = inline_functions(expr_spec->lhs, inline_depth);
      if (!callee || callee->kind != ExprKind::Function)
      {
        throw std::invalid_argument("Call expression requires a function value.");
      }

      std::vector<ExprSpecPtr> call_args;
      call_args.reserve(expr_spec->args.size());
      for (const auto & arg : expr_spec->args)
      {
        call_args.push_back(inline_functions(arg, inline_depth));
      }

      if (callee->param_count != call_args.size())
      {
        throw std::invalid_argument("Function call argument count mismatch.");
      }

      if (!is_pure_function_body(callee->lhs, callee->param_count))
      {
        throw std::invalid_argument("Function body must be pure (no refs, registers, or sample index).");
      }

      ExprSpecPtr substituted = clone_with_subst(callee->lhs, call_args);
      return inline_functions(substituted, inline_depth + 1);
    }
    case ExprKind::ArrayPack:
    {
      std::vector<ExprSpecPtr> items;
      items.reserve(expr_spec->args.size());
      for (const auto & arg : expr_spec->args)
      {
        items.push_back(inline_functions(arg, inline_depth));
      }
      return expr::array_pack_expr(std::move(items));
    }
    case ExprKind::Clamp:
      return expr::clamp_expr(
        inline_functions(expr_spec->lhs, inline_depth),
        inline_functions(expr_spec->rhs, inline_depth),
        inline_functions(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), inline_depth));
    case ExprKind::ArraySet:
      return expr::array_set_expr(
        inline_functions(expr_spec->lhs, inline_depth),
        inline_functions(expr_spec->rhs, inline_depth),
        inline_functions(expr_spec->args.empty() ? nullptr : expr_spec->args.front(), inline_depth));
    case ExprKind::Index:
      return expr::index_expr(inline_functions(expr_spec->lhs, inline_depth), inline_functions(expr_spec->rhs, inline_depth));
    default:
      break;
  }

  if (expr_spec->kind == ExprKind::Abs ||
      expr_spec->kind == ExprKind::Neg ||
      expr_spec->kind == ExprKind::Not ||
      expr_spec->kind == ExprKind::BitNot ||
      expr_spec->kind == ExprKind::Log ||
      expr_spec->kind == ExprKind::Sin)
  {
    return expr::unary_expr(expr_spec->kind, inline_functions(expr_spec->lhs, inline_depth));
  }

  if (expr_spec->kind == ExprKind::Literal ||
      expr_spec->kind == ExprKind::Ref ||
      expr_spec->kind == ExprKind::InputValue ||
      expr_spec->kind == ExprKind::RegisterValue ||
      expr_spec->kind == ExprKind::NestedValue ||
      expr_spec->kind == ExprKind::SampleRate ||
      expr_spec->kind == ExprKind::SampleIndex)
  {
    return expr_spec;
  }

  return expr::binary_expr(
    expr_spec->kind,
    inline_functions(expr_spec->lhs, inline_depth),
    inline_functions(expr_spec->rhs, inline_depth));
}
}  // namespace egress_expr_inline
