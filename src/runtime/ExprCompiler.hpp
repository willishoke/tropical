#pragma once

/**
 * ExprCompiler.hpp — Compile ExprSpec trees into a register-allocated CompiledProgram.
 *
 * Free-function equivalent of Module::compile_program() + compile_expr_node().
 * No Module instance required — param_anon_reg_map and user_register_count
 * are passed explicitly.
 */

#include "graph/GraphTypes.hpp"
#include "graph/ModuleProgram.hpp"
#include "expr/ExprStructural.hpp"

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace egress_runtime
{

using Instr = egress_module_detail::Instr;
using CompiledProgram = egress_module_detail::CompiledProgram;

// Walk an expression tree and collect unique SmoothedParam/TriggerParam pointers.
// Assigns each a sequential anonymous register index (0-based) in map.
inline void walk_expr_for_params(
  const ExprSpecPtr & expr,
  std::unordered_map<egress_expr::ControlParam *, uint32_t> & map,
  uint32_t & next_idx)
{
  if (!expr) return;
  if (expr->kind == ExprKind::SmoothedParam || expr->kind == ExprKind::TriggerParam)
  {
    if (expr->control_param && map.find(expr->control_param) == map.end())
    {
      map[expr->control_param] = next_idx++;
    }
    return;
  }
  walk_expr_for_params(expr->lhs, map, next_idx);
  walk_expr_for_params(expr->rhs, map, next_idx);
  for (const auto & arg : expr->args)
  {
    walk_expr_for_params(arg, map, next_idx);
  }
}

// Collect TriggerParam ControlParam pointers from an expression tree.
inline void collect_trigger_params(
  const ExprSpecPtr & expr,
  std::unordered_set<egress_expr::ControlParam *> & out)
{
  if (!expr) return;
  if (expr->kind == ExprKind::TriggerParam)
  {
    if (expr->control_param) out.insert(expr->control_param);
    return;
  }
  collect_trigger_params(expr->lhs, out);
  collect_trigger_params(expr->rhs, out);
  for (const auto & arg : expr->args) collect_trigger_params(arg, out);
}

namespace detail
{

inline uint32_t compile_expr_node(
  const ExprSpecPtr & expr,
  CompiledProgram & compiled,
  const std::unordered_map<egress_expr::ControlParam *, uint32_t> & param_anon_reg_map,
  uint32_t user_register_count,
  std::unordered_map<std::size_t, std::vector<std::pair<ExprSpecPtr, uint32_t>>> & memo,
  std::unordered_map<const ExprSpec *, std::size_t> & hash_cache)
{
  const std::size_t hash = egress_expr_inline::structural_hash(expr, hash_cache);
  auto memo_it = memo.find(hash);
  if (memo_it != memo.end())
  {
    for (const auto & candidate : memo_it->second)
    {
      if (egress_expr_inline::structural_equal(expr, candidate.first))
      {
        return candidate.second;
      }
    }
  }

  if (!expr)
  {
    Instr instr;
    instr.kind = ExprKind::Literal;
    instr.dst = compiled.register_count++;
    instr.literal = expr::float_value(0.0);
    compiled.instructions.push_back(std::move(instr));
    memo[hash].push_back(std::make_pair(expr, compiled.instructions.back().dst));
    return compiled.instructions.back().dst;
  }

  Instr instr;
  instr.kind = expr->kind;
  instr.dst = compiled.register_count++;

  switch (expr->kind)
  {
    case ExprKind::Literal:
      instr.literal = expr->literal;
      break;
    case ExprKind::InputValue:
    case ExprKind::RegisterValue:
      instr.slot_id = expr->slot_id;
      break;
    case ExprKind::NestedValue:
      instr.slot_id = expr->slot_id;
      instr.output_id = expr->output_id;
      break;
    case ExprKind::DelayValue:
      instr.slot_id = expr->slot_id;
      break;
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
      break;
    case ExprKind::ArrayPack:
      instr.args.reserve(expr->args.size());
      for (const auto & arg : expr->args)
      {
        instr.args.push_back(compile_expr_node(arg, compiled, param_anon_reg_map, user_register_count, memo, hash_cache));
      }
      break;
    case ExprKind::SmoothedParam:
    case ExprKind::TriggerParam:
    {
      const auto it = param_anon_reg_map.find(expr->control_param);
      instr.slot_id = (it != param_anon_reg_map.end())
        ? (user_register_count + it->second)
        : 0;
      instr.control_param = expr->control_param;
      break;
    }
    case ExprKind::ConstructStruct:
    {
      instr.type_name = expr->module_name;
      instr.args.reserve(expr->args.size());
      for (const auto & arg : expr->args)
      {
        instr.args.push_back(compile_expr_node(arg, compiled, param_anon_reg_map, user_register_count, memo, hash_cache));
      }
      break;
    }
    case ExprKind::FieldAccess:
    {
      instr.type_name = expr->module_name;
      instr.slot_id = expr->slot_id;
      instr.src_a = compile_expr_node(expr->lhs, compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
      break;
    }
    case ExprKind::ConstructVariant:
    {
      instr.type_name = expr->module_name;
      instr.slot_id = expr->slot_id;
      instr.args.reserve(expr->args.size());
      for (const auto & arg : expr->args)
      {
        instr.args.push_back(compile_expr_node(arg, compiled, param_anon_reg_map, user_register_count, memo, hash_cache));
      }
      break;
    }
    case ExprKind::MatchVariant:
    {
      instr.type_name = expr->module_name;
      instr.src_a = compile_expr_node(expr->lhs, compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
      instr.args.reserve(expr->args.size());
      for (const auto & arg : expr->args)
      {
        instr.args.push_back(compile_expr_node(arg, compiled, param_anon_reg_map, user_register_count, memo, hash_cache));
      }
      break;
    }
    default:
      if (egress_module_detail::is_local_unary(expr->kind))
      {
        instr.src_a = compile_expr_node(expr->lhs, compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
      }
      else if (egress_module_detail::is_local_ternary(expr->kind))
      {
        instr.src_a = compile_expr_node(expr->lhs, compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
        instr.src_b = compile_expr_node(expr->rhs, compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
        instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
      }
      else if (egress_module_detail::is_local_binary(expr->kind))
      {
        instr.src_a = compile_expr_node(expr->lhs, compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
        instr.src_b = compile_expr_node(expr->rhs, compiled, param_anon_reg_map, user_register_count, memo, hash_cache);
      }
      else
      {
        throw std::invalid_argument("Unsupported module expression kind.");
      }
      break;
  }

  compiled.instructions.push_back(std::move(instr));
  memo[hash].push_back(std::make_pair(expr, compiled.instructions.back().dst));
  return compiled.instructions.back().dst;
}

} // namespace detail

/**
 * Compile expression trees into a register-allocated CompiledProgram.
 *
 * @param output_exprs     Expression trees for each output
 * @param register_exprs   Expression trees for each register update
 * @param param_anon_reg_map  Map from ControlParam* to anonymous register index
 * @param user_register_count  Number of user-defined registers (before anonymous ones)
 */
inline CompiledProgram compile_expr_program(
  const std::vector<ExprSpecPtr> & output_exprs,
  const std::vector<ExprSpecPtr> & register_exprs,
  const std::unordered_map<egress_expr::ControlParam *, uint32_t> & param_anon_reg_map,
  uint32_t user_register_count)
{
  CompiledProgram compiled;
  compiled.output_targets.reserve(output_exprs.size());
  compiled.register_targets.assign(register_exprs.size(), -1);

  std::unordered_map<std::size_t, std::vector<std::pair<ExprSpecPtr, uint32_t>>> memo;
  std::unordered_map<const ExprSpec *, std::size_t> hash_cache;
  for (const auto & expr : output_exprs)
  {
    ExprSpecPtr inlined = egress_expr_inline::inline_functions(expr);
    compiled.output_targets.push_back(
      detail::compile_expr_node(inlined, compiled, param_anon_reg_map, user_register_count, memo, hash_cache));
  }
  for (unsigned int i = 0; i < register_exprs.size(); ++i)
  {
    if (register_exprs[i])
    {
      ExprSpecPtr inlined = egress_expr_inline::inline_functions(register_exprs[i]);
      compiled.register_targets[i] = static_cast<int32_t>(
        detail::compile_expr_node(inlined, compiled, param_anon_reg_map, user_register_count, memo, hash_cache));
    }
  }
  return compiled;
}

} // namespace egress_runtime
