#include "Expr.hpp"
#include "ExprEval.hpp"
#include "OrcJitEngine.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#ifdef EGRESS_PROFILE
#include <chrono>
#endif
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace expr = egress_expr;
namespace expr_eval = egress_expr_eval;

class Module;
class Graph;

using Signal = double;

using inputID = std::pair<std::string, unsigned int>;
using outputID = std::pair<std::string, unsigned int>;
using mPtr = std::unique_ptr<Module>;
using ExprValueType = expr::ValueType;
using ExprValue = expr::Value;
using ExprKind = expr::ExprKind;
using ExprSpec = expr::ExprSpec;
using ExprSpecPtr = expr::ExprSpecPtr;
using ValueType = ExprValueType;
using Value = ExprValue;
using expr::arithmetic_type;
using expr::array_value;
using expr::bool_value;
using expr::float_value;
using expr::int_value;
using expr::is_array;
using expr::is_truthy;
using expr::map_binary;
using expr::map_unary;
using expr::to_float64;
using expr::to_int64;

namespace egress_expr_inline
{
  static inline std::size_t hash_mix(std::size_t seed, std::size_t value)
  {
    constexpr std::size_t kMul = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    seed ^= value + kMul + (seed << 6) + (seed >> 2);
    return seed;
  }

  static inline std::size_t hash_value(const Value & value)
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
    }
    return seed;
  }

  static inline std::size_t structural_hash(
    const ExprSpecPtr & expr,
    std::unordered_map<const ExprSpec *, std::size_t> & cache)
  {
    if (!expr)
    {
      return static_cast<std::size_t>(0x51ed270bu);
    }

    auto it = cache.find(expr.get());
    if (it != cache.end())
    {
      return it->second;
    }

    std::size_t seed = static_cast<std::size_t>(expr->kind);
    switch (expr->kind)
    {
      case ExprKind::Literal:
        seed = hash_mix(seed, hash_value(expr->literal));
        break;
      case ExprKind::Ref:
        seed = hash_mix(seed, std::hash<std::string>{}(expr->module_name));
        seed = hash_mix(seed, std::hash<unsigned int>{}(expr->output_id));
        break;
      case ExprKind::InputValue:
      case ExprKind::RegisterValue:
        seed = hash_mix(seed, std::hash<unsigned int>{}(expr->slot_id));
        break;
      case ExprKind::Function:
        seed = hash_mix(seed, std::hash<unsigned int>{}(expr->param_count));
        seed = hash_mix(seed, structural_hash(expr->lhs, cache));
        break;
      case ExprKind::Call:
        seed = hash_mix(seed, structural_hash(expr->lhs, cache));
        seed = hash_mix(seed, std::hash<std::size_t>{}(expr->args.size()));
        for (const auto & arg : expr->args)
        {
          seed = hash_mix(seed, structural_hash(arg, cache));
        }
        break;
      case ExprKind::Clamp:
        seed = hash_mix(seed, structural_hash(expr->lhs, cache));
        seed = hash_mix(seed, structural_hash(expr->rhs, cache));
        seed = hash_mix(seed, structural_hash(expr->args.empty() ? nullptr : expr->args.front(), cache));
        break;
      case ExprKind::ArrayPack:
      {
        seed = hash_mix(seed, std::hash<std::size_t>{}(expr->args.size()));
        for (const auto & arg : expr->args)
        {
          seed = hash_mix(seed, structural_hash(arg, cache));
        }
        break;
      }
      case ExprKind::SampleRate:
      case ExprKind::SampleIndex:
        break;
      default:
        seed = hash_mix(seed, structural_hash(expr->lhs, cache));
        seed = hash_mix(seed, structural_hash(expr->rhs, cache));
        break;
    }

    cache.emplace(expr.get(), seed);
    return seed;
  }

  static inline bool value_equal(const Value & lhs, const Value & rhs)
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
    }

    return false;
  }

  static inline bool structural_equal(const ExprSpecPtr & lhs, const ExprSpecPtr & rhs)
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
        }
        return true;
      case ExprKind::Ref:
        return lhs->module_name == rhs->module_name && lhs->output_id == rhs->output_id;
      case ExprKind::InputValue:
      case ExprKind::RegisterValue:
        return lhs->slot_id == rhs->slot_id;
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

  static inline bool is_pure_function_body(const ExprSpecPtr & expr, unsigned int param_count)
  {
    if (!expr)
    {
      return true;
    }

    switch (expr->kind)
    {
      case ExprKind::Ref:
      case ExprKind::RegisterValue:
      case ExprKind::SampleIndex:
        return false;
      case ExprKind::InputValue:
        return expr->slot_id < param_count;
      case ExprKind::Function:
        return false;
      case ExprKind::Call:
        if (!is_pure_function_body(expr->lhs, param_count))
        {
          return false;
        }
        for (const auto & arg : expr->args)
        {
          if (!is_pure_function_body(arg, param_count))
          {
            return false;
          }
        }
        return true;
      case ExprKind::Clamp:
        return is_pure_function_body(expr->lhs, param_count) &&
               is_pure_function_body(expr->rhs, param_count) &&
               is_pure_function_body(expr->args.empty() ? nullptr : expr->args.front(), param_count);
      case ExprKind::ArrayPack:
        for (const auto & arg : expr->args)
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

    return is_pure_function_body(expr->lhs, param_count) &&
           is_pure_function_body(expr->rhs, param_count);
  }

  static inline ExprSpecPtr clone_with_subst(
    const ExprSpecPtr & expr,
    const std::vector<ExprSpecPtr> & args)
  {
    if (!expr)
    {
      return nullptr;
    }

    switch (expr->kind)
    {
      case ExprKind::InputValue:
        if (expr->slot_id >= args.size())
        {
          throw std::invalid_argument("Function argument index out of range.");
        }
        return args[expr->slot_id];
      case ExprKind::Literal:
      case ExprKind::Ref:
      case ExprKind::RegisterValue:
      case ExprKind::SampleRate:
      case ExprKind::SampleIndex:
        return expr;
      case ExprKind::Function:
        return expr::function_expr(expr->param_count, clone_with_subst(expr->lhs, args));
      case ExprKind::Call:
      {
        std::vector<ExprSpecPtr> call_args;
        call_args.reserve(expr->args.size());
        for (const auto & arg : expr->args)
        {
          call_args.push_back(clone_with_subst(arg, args));
        }
        return expr::call_expr(clone_with_subst(expr->lhs, args), std::move(call_args));
      }
      case ExprKind::ArrayPack:
      {
        std::vector<ExprSpecPtr> items;
        items.reserve(expr->args.size());
        for (const auto & arg : expr->args)
        {
          items.push_back(clone_with_subst(arg, args));
        }
        return expr::array_pack_expr(std::move(items));
      }
      case ExprKind::Clamp:
        return expr::clamp_expr(
          clone_with_subst(expr->lhs, args),
          clone_with_subst(expr->rhs, args),
          clone_with_subst(expr->args.empty() ? nullptr : expr->args.front(), args));
      case ExprKind::Index:
        return expr::index_expr(clone_with_subst(expr->lhs, args), clone_with_subst(expr->rhs, args));
      default:
        break;
    }

    if (expr->kind == ExprKind::Abs ||
        expr->kind == ExprKind::Neg ||
        expr->kind == ExprKind::Not ||
        expr->kind == ExprKind::BitNot ||
        expr->kind == ExprKind::Log ||
        expr->kind == ExprKind::Sin)
    {
      return expr::unary_expr(expr->kind, clone_with_subst(expr->lhs, args));
    }

    return expr::binary_expr(expr->kind, clone_with_subst(expr->lhs, args), clone_with_subst(expr->rhs, args));
  }

  static inline ExprSpecPtr inline_functions(const ExprSpecPtr & expr, unsigned int inline_depth = 0)
  {
    if (!expr)
    {
      return nullptr;
    }

    if (inline_depth > 32)
    {
      throw std::invalid_argument("Function inlining exceeded maximum depth.");
    }

    switch (expr->kind)
    {
      case ExprKind::Function:
      {
        ExprSpecPtr body = inline_functions(expr->lhs, inline_depth);
        return expr::function_expr(expr->param_count, body);
      }
      case ExprKind::Call:
      {
        ExprSpecPtr callee = inline_functions(expr->lhs, inline_depth);
        if (!callee || callee->kind != ExprKind::Function)
        {
          throw std::invalid_argument("Call expression requires a function value.");
        }

        std::vector<ExprSpecPtr> call_args;
        call_args.reserve(expr->args.size());
        for (const auto & arg : expr->args)
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
        items.reserve(expr->args.size());
        for (const auto & arg : expr->args)
        {
          items.push_back(inline_functions(arg, inline_depth));
        }
        return expr::array_pack_expr(std::move(items));
      }
      case ExprKind::Clamp:
        return expr::clamp_expr(
          inline_functions(expr->lhs, inline_depth),
          inline_functions(expr->rhs, inline_depth),
          inline_functions(expr->args.empty() ? nullptr : expr->args.front(), inline_depth));
      case ExprKind::Index:
        return expr::index_expr(inline_functions(expr->lhs, inline_depth), inline_functions(expr->rhs, inline_depth));
      default:
        break;
    }

    if (expr->kind == ExprKind::Abs ||
        expr->kind == ExprKind::Neg ||
        expr->kind == ExprKind::Not ||
        expr->kind == ExprKind::BitNot ||
        expr->kind == ExprKind::Log ||
        expr->kind == ExprKind::Sin)
    {
      return expr::unary_expr(expr->kind, inline_functions(expr->lhs, inline_depth));
    }

    if (expr->kind == ExprKind::Literal ||
        expr->kind == ExprKind::Ref ||
        expr->kind == ExprKind::InputValue ||
        expr->kind == ExprKind::RegisterValue ||
        expr->kind == ExprKind::SampleRate ||
        expr->kind == ExprKind::SampleIndex)
    {
      return expr;
    }

    return expr::binary_expr(expr->kind, inline_functions(expr->lhs, inline_depth), inline_functions(expr->rhs, inline_depth));
  }
}  // namespace egress_expr_inline

class Module
{
  public:
#ifdef EGRESS_PROFILE
    struct CompileStats
    {
      uint64_t instruction_count = 0;
      uint64_t register_count = 0;
      uint64_t numeric_jit_instruction_count = 0;
      std::string jit_status;
    };
#endif

    struct RegisterArraySpec
    {
      bool enabled = false;
      unsigned int source_input_id = 0;
      Value init_value;
    };

    virtual ~Module() = default;

    Module(
      unsigned int input_count,
      std::vector<ExprSpecPtr> output_exprs,
      std::vector<ExprSpecPtr> register_exprs,
      std::vector<Value> initial_registers,
      std::vector<RegisterArraySpec> register_array_specs,
      double sample_rate)
      : inputs(input_count, expr::float_value(0.0)),
        outputs(output_exprs.size(), expr::float_value(0.0)),
        prev_outputs(output_exprs.size(), expr::float_value(0.0)),
        input_count_(input_count),
        registers_(std::move(initial_registers)),
        next_registers_(registers_),
        register_array_specs_(std::move(register_array_specs)),
        sample_rate_(sample_rate)
    {
      has_dynamic_registers_ = false;
      for (const auto & spec : register_array_specs_)
      {
        if (spec.enabled)
        {
          has_dynamic_registers_ = true;
          break;
        }
      }

      program_ = compile_program(output_exprs, register_exprs);
      temps_.assign(program_.register_count, expr::float_value(0.0));

#ifdef EGRESS_LLVM_ORC_JIT
      if (!has_dynamic_registers_)
      {
        initialize_numeric_jit();
      }
#endif
    }

    void process()
    {
      resize_array_registers_to_inputs();
#ifdef EGRESS_LLVM_ORC_JIT
      if (jit_kernel_ && inputs_are_scalar())
      {
        if (numeric_inputs_.size() < inputs.size())
        {
          numeric_inputs_.assign(inputs.size(), 0.0);
        }
        for (unsigned int i = 0; i < inputs.size(); ++i)
        {
          numeric_inputs_[i] = expr::to_float64(inputs[i]);
        }

        jit_kernel_(
          numeric_inputs_.data(),
          numeric_registers_.data(),
          numeric_array_ptrs_.data(),
          numeric_array_sizes_.data(),
          numeric_temps_.data(),
          sample_rate_,
          sample_index_);

        for (unsigned int output_id = 0; output_id < program_.output_targets.size(); ++output_id)
        {
          outputs[output_id] = expr::float_value(numeric_temps_[program_.output_targets[output_id]]);
        }

        for (unsigned int register_id = 0; register_id < program_.register_targets.size(); ++register_id)
        {
          const int32_t target = program_.register_targets[register_id];
          if (target >= 0)
          {
            numeric_next_registers_[register_id] = numeric_temps_[static_cast<std::size_t>(target)];
          }
          else
          {
            numeric_next_registers_[register_id] = numeric_registers_[register_id];
          }
        }

        numeric_registers_.swap(numeric_next_registers_);
        ++sample_index_;
        postprocess();
        return;
      }
#endif

      eval_program(program_, temps_);

      for (unsigned int output_id = 0; output_id < program_.output_targets.size(); ++output_id)
      {
        outputs[output_id] = temps_[program_.output_targets[output_id]];
      }

      for (unsigned int register_id = 0; register_id < program_.register_targets.size(); ++register_id)
      {
        const int32_t target = program_.register_targets[register_id];
        if (target >= 0)
        {
          next_registers_[register_id] = temps_[static_cast<std::size_t>(target)];
        }
        else
        {
          next_registers_[register_id] = registers_[register_id];
        }
      }

      registers_.swap(next_registers_);
      ++sample_index_;
      postprocess();
    }

    unsigned int input_count() const
    {
      return input_count_;
    }

    unsigned int output_count() const
    {
      return static_cast<unsigned int>(program_.output_targets.size());
    }

    unsigned int register_count() const
    {
      return static_cast<unsigned int>(registers_.size());
    }

    #ifdef EGRESS_PROFILE
        CompileStats compile_stats() const
        {
      CompileStats stats;
      stats.instruction_count = static_cast<uint64_t>(program_.instructions.size());
      stats.register_count = static_cast<uint64_t>(program_.register_count);
    #ifdef EGRESS_LLVM_ORC_JIT
      stats.numeric_jit_instruction_count = numeric_jit_instruction_count_;
      stats.jit_status = jit_status_;
    #else
      stats.jit_status = "LLVM ORC JIT disabled";
    #endif
      return stats;
        }
    #endif

  protected:
    void postprocess()
    {
      for (auto & in : inputs)
      {
        in = expr::float_value(0.0);
      }

      for (auto & out : outputs)
      {
        clamp_output_value(out);
      }
    }

    std::vector<Value> inputs;
    std::vector<Value> outputs;
    std::vector<Value> prev_outputs;

  private:
    friend class Graph;

    static void clamp_output_value(Value & value)
    {
      if (value.type == ValueType::Array)
      {
        for (auto & item : value.array_items)
        {
          clamp_output_value(item);
        }
        return;
      }
      const double clamped = std::fmax(-10.0, std::fmin(10.0, expr::to_float64(value)));
      value = expr::float_value(clamped);
    }

    bool inputs_are_scalar() const
    {
      for (const auto & input : inputs)
      {
        if (input.type == ValueType::Array)
        {
          return false;
        }
      }
      return true;
    }

    void resize_array_registers_to_inputs()
    {
      if (!has_dynamic_registers_)
      {
        return;
      }

      for (unsigned int reg_id = 0; reg_id < register_array_specs_.size(); ++reg_id)
      {
        const RegisterArraySpec & spec = register_array_specs_[reg_id];
        if (!spec.enabled)
        {
          continue;
        }

        if (spec.source_input_id >= inputs.size())
        {
          throw std::invalid_argument("Array register spec references unknown input id.");
        }

        const Value & input = inputs[spec.source_input_id];
        if (input.type != ValueType::Array)
        {
          throw std::invalid_argument("Array register requires array-valued input.");
        }

        const std::size_t desired = input.array_items.size();
        Value & reg = registers_[reg_id];
        bool resized = false;

        if (reg.type != ValueType::Array)
        {
          std::vector<Value> items(desired, spec.init_value);
          reg = expr::array_value(std::move(items));
          resized = true;
        }
        else if (reg.array_items.size() != desired)
        {
          std::vector<Value> items = reg.array_items;
          items.resize(desired, spec.init_value);
          reg = expr::array_value(std::move(items));
          resized = true;
        }

        if (resized)
        {
          next_registers_[reg_id] = reg;
        }
      }
    }

    struct Instr
    {
      ExprKind kind = ExprKind::Literal;
      uint32_t dst = 0;
      uint32_t src_a = 0;
      uint32_t src_b = 0;
      uint32_t src_c = 0;
      unsigned int slot_id = 0;
      Value literal;
      std::vector<uint32_t> args;
    };

    struct CompiledProgram
    {
      std::vector<Instr> instructions;
      std::vector<uint32_t> output_targets;
      std::vector<int32_t> register_targets;
      uint32_t register_count = 0;
    };

    static bool is_local_unary(ExprKind kind)
    {
      return kind == ExprKind::Abs ||
             kind == ExprKind::Neg ||
             kind == ExprKind::Not ||
             kind == ExprKind::BitNot ||
             kind == ExprKind::Log ||
             kind == ExprKind::Sin;
    }

    static bool is_local_ternary(ExprKind kind)
    {
      return kind == ExprKind::Clamp;
    }

    static bool is_local_binary(ExprKind kind)
    {
      switch (kind)
      {
        case ExprKind::Less:
        case ExprKind::LessEqual:
        case ExprKind::Greater:
        case ExprKind::GreaterEqual:
        case ExprKind::Equal:
        case ExprKind::NotEqual:
        case ExprKind::Add:
        case ExprKind::Sub:
        case ExprKind::Mul:
        case ExprKind::Div:
        case ExprKind::Pow:
        case ExprKind::Mod:
        case ExprKind::FloorDiv:
        case ExprKind::BitAnd:
        case ExprKind::BitOr:
        case ExprKind::BitXor:
        case ExprKind::LShift:
        case ExprKind::RShift:
        case ExprKind::Index:
          return true;
        default:
          return false;
      }
    }

    CompiledProgram compile_program(
      const std::vector<ExprSpecPtr> & output_exprs,
      const std::vector<ExprSpecPtr> & register_exprs)
    {
      CompiledProgram compiled;
      compiled.output_targets.reserve(output_exprs.size());
      compiled.register_targets.assign(register_exprs.size(), -1);

      std::unordered_map<std::size_t, std::vector<std::pair<ExprSpecPtr, uint32_t>>> memo;
      std::unordered_map<const ExprSpec *, std::size_t> hash_cache;
      for (const auto & expr : output_exprs)
      {
        ExprSpecPtr inlined = egress_expr_inline::inline_functions(expr);
        compiled.output_targets.push_back(compile_expr_node(inlined, compiled, memo, hash_cache));
      }
      for (unsigned int i = 0; i < register_exprs.size(); ++i)
      {
        if (register_exprs[i])
        {
          ExprSpecPtr inlined = egress_expr_inline::inline_functions(register_exprs[i]);
          compiled.register_targets[i] = static_cast<int32_t>(compile_expr_node(inlined, compiled, memo, hash_cache));
        }
      }
      return compiled;
    }

    uint32_t compile_expr_node(
      const ExprSpecPtr & expr,
      CompiledProgram & compiled,
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
        case ExprKind::SampleRate:
        case ExprKind::SampleIndex:
          break;
        case ExprKind::ArrayPack:
          instr.args.reserve(expr->args.size());
          for (const auto & arg : expr->args)
          {
            instr.args.push_back(compile_expr_node(arg, compiled, memo, hash_cache));
          }
          break;
        default:
          if (is_local_unary(expr->kind))
          {
            instr.src_a = compile_expr_node(expr->lhs, compiled, memo, hash_cache);
          }
          else if (is_local_ternary(expr->kind))
          {
            instr.src_a = compile_expr_node(expr->lhs, compiled, memo, hash_cache);
            instr.src_b = compile_expr_node(expr->rhs, compiled, memo, hash_cache);
            instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, memo, hash_cache);
          }
          else if (is_local_binary(expr->kind))
          {
            instr.src_a = compile_expr_node(expr->lhs, compiled, memo, hash_cache);
            instr.src_b = compile_expr_node(expr->rhs, compiled, memo, hash_cache);
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

    void eval_program(const CompiledProgram & expr, std::vector<Value> & temps) const
    {
      if (expr.instructions.empty())
      {
        return;
      }

      if (temps.size() < expr.register_count)
      {
        temps.resize(expr.register_count, expr::float_value(0.0));
      }

      for (const Instr & instr : expr.instructions)
      {
        switch (instr.kind)
        {
          case ExprKind::Function:
          case ExprKind::Call:
            throw std::invalid_argument("Function values must be inlined before evaluation.");
          case ExprKind::Literal:
            temps[instr.dst] = instr.literal;
            break;
          case ExprKind::InputValue:
            temps[instr.dst] = instr.slot_id < inputs.size()
                                 ? inputs[instr.slot_id]
                                 : expr::float_value(0.0);
            break;
          case ExprKind::RegisterValue:
            temps[instr.dst] = instr.slot_id < registers_.size()
                                 ? registers_[instr.slot_id]
                                 : expr::float_value(0.0);
            break;
          case ExprKind::SampleRate:
            temps[instr.dst] = expr::float_value(sample_rate_);
            break;
          case ExprKind::SampleIndex:
            temps[instr.dst] = expr::int_value(static_cast<int64_t>(sample_index_));
            break;
          case ExprKind::ArrayPack:
          {
            std::vector<Value> items;
            items.reserve(instr.args.size());
            for (uint32_t src : instr.args)
            {
              if (expr::is_array(temps[src]))
              {
                throw std::invalid_argument("Nested arrays are not supported.");
              }
              items.push_back(temps[src]);
            }
            temps[instr.dst] = expr::array_value(std::move(items));
            break;
          }
          case ExprKind::Index:
          {
            const Value & array_value = temps[instr.src_a];
            if (!expr::is_array(array_value))
            {
              temps[instr.dst] = expr::float_value(0.0);
              break;
            }
            const int64_t index = expr::to_int64(temps[instr.src_b]);
            if (index < 0 || static_cast<std::size_t>(index) >= array_value.array_items.size())
            {
              throw std::out_of_range("Array index out of range.");
            }
            temps[instr.dst] = array_value.array_items[static_cast<std::size_t>(index)];
            break;
          }
          case ExprKind::Abs:
            temps[instr.dst] = expr_eval::abs_value(temps[instr.src_a]);
            break;
          case ExprKind::Not:
            temps[instr.dst] = expr_eval::not_value(temps[instr.src_a]);
            break;
          case ExprKind::Less:
            temps[instr.dst] = expr_eval::less_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::LessEqual:
            temps[instr.dst] = expr_eval::less_equal_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Greater:
            temps[instr.dst] = expr_eval::greater_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::GreaterEqual:
            temps[instr.dst] = expr_eval::greater_equal_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Equal:
            temps[instr.dst] = expr_eval::equal_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::NotEqual:
            temps[instr.dst] = expr_eval::not_equal_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Add:
            temps[instr.dst] = expr_eval::add_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Sub:
            temps[instr.dst] = expr_eval::sub_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Mul:
            temps[instr.dst] = expr_eval::mul_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Div:
            temps[instr.dst] = expr_eval::div_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Pow:
            temps[instr.dst] = expr_eval::pow_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Mod:
            temps[instr.dst] = expr_eval::mod_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::FloorDiv:
            temps[instr.dst] = expr_eval::floor_div_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::BitAnd:
            temps[instr.dst] = expr_eval::bit_and_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::BitOr:
            temps[instr.dst] = expr_eval::bit_or_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::BitXor:
            temps[instr.dst] = expr_eval::bit_xor_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::LShift:
            temps[instr.dst] = expr_eval::lshift_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::RShift:
            temps[instr.dst] = expr_eval::rshift_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case ExprKind::Clamp:
            temps[instr.dst] = expr_eval::clamp_values(temps[instr.src_a], temps[instr.src_b], temps[instr.src_c]);
            break;
          case ExprKind::Log:
            temps[instr.dst] = expr_eval::log_value(temps[instr.src_a]);
            break;
          case ExprKind::Sin:
            temps[instr.dst] = expr_eval::sin_value(temps[instr.src_a]);
            break;
          case ExprKind::Neg:
            temps[instr.dst] = expr_eval::neg_value(temps[instr.src_a]);
            break;
          case ExprKind::BitNot:
            temps[instr.dst] = expr_eval::bit_not_value(temps[instr.src_a]);
            break;
          case ExprKind::Ref:
            temps[instr.dst] = expr::float_value(0.0);
            break;
        }
      }
    }

    unsigned int input_count_ = 0;
    CompiledProgram program_;
    std::vector<Value> temps_;
    std::vector<Value> registers_;
    std::vector<Value> next_registers_;
    std::vector<RegisterArraySpec> register_array_specs_;
    bool has_dynamic_registers_ = false;
    double sample_rate_ = 44100.0;
    uint64_t sample_index_ = 0;
#ifdef EGRESS_LLVM_ORC_JIT
    egress_jit::NumericKernelFn jit_kernel_ = nullptr;
    std::vector<double> numeric_inputs_;
    std::vector<double> numeric_temps_;
    std::vector<double> numeric_registers_;
    std::vector<double> numeric_next_registers_;
    std::vector<std::vector<double>> numeric_array_storage_;
    std::vector<const double *> numeric_array_ptrs_;
    std::vector<uint64_t> numeric_array_sizes_;
    std::vector<bool> register_scalar_mask_;
    std::vector<uint32_t> register_array_slot_;
    std::string jit_status_;
  #ifdef EGRESS_PROFILE
    uint64_t numeric_jit_instruction_count_ = 0;
  #endif

    bool supports_numeric_jit_expr_kind(ExprKind kind) const
    {
      switch (kind)
      {
        case ExprKind::Literal:
        case ExprKind::InputValue:
        case ExprKind::RegisterValue:
        case ExprKind::SampleRate:
        case ExprKind::SampleIndex:
        case ExprKind::Not:
        case ExprKind::Less:
        case ExprKind::LessEqual:
        case ExprKind::Greater:
        case ExprKind::GreaterEqual:
        case ExprKind::Equal:
        case ExprKind::NotEqual:
        case ExprKind::Add:
        case ExprKind::Sub:
        case ExprKind::Mul:
        case ExprKind::Div:
        case ExprKind::Pow:
        case ExprKind::Mod:
        case ExprKind::FloorDiv:
        case ExprKind::BitAnd:
        case ExprKind::BitOr:
        case ExprKind::BitXor:
        case ExprKind::LShift:
        case ExprKind::RShift:
        case ExprKind::Abs:
        case ExprKind::Clamp:
        case ExprKind::Index:
        case ExprKind::Log:
        case ExprKind::Sin:
        case ExprKind::Neg:
        case ExprKind::BitNot:
        case ExprKind::ArrayPack:
          return true;
        default:
          return false;
      }
    }

    struct NumericRegInfo
    {
      bool is_scalar = true;
      uint32_t array_slot = 0;
      bool scalar_is_constant = false;
      double scalar_constant = 0.0;
    };

    static bool value_to_scalar_double(const Value & value, double & out)
    {
      if (value.type == ValueType::Array)
      {
        return false;
      }
      out = to_float64(value);
      return true;
    }

    bool add_array_values_to_jit_table(const std::vector<Value> & values, uint32_t & out_slot)
    {
      std::vector<double> numeric_values;
      numeric_values.reserve(values.size());
      for (const Value & item : values)
      {
        double scalar = 0.0;
        if (!value_to_scalar_double(item, scalar))
        {
          return false;
        }
        numeric_values.push_back(scalar);
      }

      out_slot = static_cast<uint32_t>(numeric_array_storage_.size());
      numeric_array_storage_.push_back(std::move(numeric_values));
      return true;
    }

    bool build_numeric_program(egress_jit::NumericProgram & numeric_program)
    {
      if (program_.register_count == 0)
      {
        return false;
      }

      numeric_program.instructions.clear();
      numeric_program.register_count = program_.register_count;
      numeric_array_storage_.clear();
      register_scalar_mask_.assign(registers_.size(), true);
      register_array_slot_.assign(registers_.size(), 0);

      std::vector<NumericRegInfo> reg_info(program_.register_count);

      for (unsigned int reg_slot = 0; reg_slot < registers_.size(); ++reg_slot)
      {
        const Value & reg = registers_[reg_slot];
        if (reg.type == ValueType::Array)
        {
          uint32_t array_slot = 0;
          if (!add_array_values_to_jit_table(reg.array_items, array_slot))
          {
            return false;
          }
          register_scalar_mask_[reg_slot] = false;
          register_array_slot_[reg_slot] = array_slot;
        }
      }

      for (const Instr & instr : program_.instructions)
      {
        if (!supports_numeric_jit_expr_kind(instr.kind))
        {
          return false;
        }

        egress_jit::NumericInstr jit_instr;
        jit_instr.dst = instr.dst;
        jit_instr.src_a = instr.src_a;
        jit_instr.src_b = instr.src_b;
        jit_instr.src_c = instr.src_c;
        jit_instr.slot_id = instr.slot_id;

        bool emit_instruction = true;

        switch (instr.kind)
        {
          case ExprKind::Literal:
          {
            if (instr.literal.type == ValueType::Array)
            {
              uint32_t array_slot = 0;
              if (!add_array_values_to_jit_table(instr.literal.array_items, array_slot))
              {
                return false;
              }
              reg_info[instr.dst].is_scalar = false;
              reg_info[instr.dst].array_slot = array_slot;
              emit_instruction = false;
            }
            else
            {
              jit_instr.op = egress_jit::NumericOp::Literal;
              jit_instr.literal = to_float64(instr.literal);
              reg_info[instr.dst].is_scalar = true;
              reg_info[instr.dst].scalar_is_constant = true;
              reg_info[instr.dst].scalar_constant = jit_instr.literal;
            }
            break;
          }
          case ExprKind::InputValue:
            jit_instr.op = egress_jit::NumericOp::InputValue;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::RegisterValue:
            if (instr.slot_id >= register_scalar_mask_.size())
            {
              return false;
            }
            if (!register_scalar_mask_[instr.slot_id])
            {
              reg_info[instr.dst].is_scalar = false;
              reg_info[instr.dst].array_slot = register_array_slot_[instr.slot_id];
              emit_instruction = false;
            }
            else
            {
              jit_instr.op = egress_jit::NumericOp::RegisterValue;
              reg_info[instr.dst].is_scalar = true;
            }
            break;
          case ExprKind::SampleRate:
            jit_instr.op = egress_jit::NumericOp::SampleRate;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::SampleIndex:
            jit_instr.op = egress_jit::NumericOp::SampleIndex;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::ArrayPack:
          {
            std::vector<Value> packed_values;
            packed_values.reserve(instr.args.size());
            for (uint32_t src : instr.args)
            {
              if (src >= reg_info.size() || !reg_info[src].is_scalar || !reg_info[src].scalar_is_constant)
              {
                return false;
              }
              packed_values.push_back(float_value(reg_info[src].scalar_constant));
            }
            uint32_t array_slot = 0;
            if (!add_array_values_to_jit_table(packed_values, array_slot))
            {
              return false;
            }
            reg_info[instr.dst].is_scalar = false;
            reg_info[instr.dst].array_slot = array_slot;
            emit_instruction = false;
            break;
          }
          case ExprKind::Index:
            if (instr.src_a >= reg_info.size() || instr.src_b >= reg_info.size())
            {
              return false;
            }
            if (reg_info[instr.src_a].is_scalar || !reg_info[instr.src_b].is_scalar)
            {
              return false;
            }
            jit_instr.op = egress_jit::NumericOp::IndexArray;
            jit_instr.src_a = instr.src_b;
            jit_instr.slot_id = reg_info[instr.src_a].array_slot;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Not:
            jit_instr.op = egress_jit::NumericOp::Not;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Less:
            jit_instr.op = egress_jit::NumericOp::Less;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::LessEqual:
            jit_instr.op = egress_jit::NumericOp::LessEqual;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Greater:
            jit_instr.op = egress_jit::NumericOp::Greater;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::GreaterEqual:
            jit_instr.op = egress_jit::NumericOp::GreaterEqual;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Equal:
            jit_instr.op = egress_jit::NumericOp::Equal;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::NotEqual:
            jit_instr.op = egress_jit::NumericOp::NotEqual;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Add:
            jit_instr.op = egress_jit::NumericOp::Add;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Sub:
            jit_instr.op = egress_jit::NumericOp::Sub;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Mul:
            jit_instr.op = egress_jit::NumericOp::Mul;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Div:
            jit_instr.op = egress_jit::NumericOp::Div;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Pow:
            jit_instr.op = egress_jit::NumericOp::Pow;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Mod:
            jit_instr.op = egress_jit::NumericOp::Mod;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::FloorDiv:
            jit_instr.op = egress_jit::NumericOp::FloorDiv;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::BitAnd:
            jit_instr.op = egress_jit::NumericOp::BitAnd;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::BitOr:
            jit_instr.op = egress_jit::NumericOp::BitOr;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::BitXor:
            jit_instr.op = egress_jit::NumericOp::BitXor;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::LShift:
            jit_instr.op = egress_jit::NumericOp::LShift;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::RShift:
            jit_instr.op = egress_jit::NumericOp::RShift;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Abs:
            jit_instr.op = egress_jit::NumericOp::Abs;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Clamp:
            jit_instr.op = egress_jit::NumericOp::Clamp;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Log:
            jit_instr.op = egress_jit::NumericOp::Log;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Sin:
            jit_instr.op = egress_jit::NumericOp::Sin;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::Neg:
            jit_instr.op = egress_jit::NumericOp::Neg;
            reg_info[instr.dst].is_scalar = true;
            break;
          case ExprKind::BitNot:
            jit_instr.op = egress_jit::NumericOp::BitNot;
            reg_info[instr.dst].is_scalar = true;
            break;
          default:
            return false;
        }

        if (emit_instruction)
        {
          if (instr.kind != ExprKind::Literal &&
              instr.kind != ExprKind::InputValue &&
              instr.kind != ExprKind::RegisterValue &&
              instr.kind != ExprKind::SampleRate &&
              instr.kind != ExprKind::SampleIndex &&
              instr.kind != ExprKind::Index)
          {
            if (instr.src_a >= reg_info.size() || !reg_info[instr.src_a].is_scalar)
            {
              return false;
            }
            if (is_local_binary(instr.kind))
            {
              if (instr.src_b >= reg_info.size() || !reg_info[instr.src_b].is_scalar)
              {
                return false;
              }
            }
            if (is_local_ternary(instr.kind))
            {
              if (instr.src_b >= reg_info.size() || !reg_info[instr.src_b].is_scalar ||
                  instr.src_c >= reg_info.size() || !reg_info[instr.src_c].is_scalar)
              {
                return false;
              }
            }
          }

          reg_info[instr.dst].scalar_is_constant = false;
          reg_info[instr.dst].scalar_constant = 0.0;
          numeric_program.instructions.push_back(jit_instr);
        }
      }

      for (uint32_t output_reg : program_.output_targets)
      {
        if (output_reg >= reg_info.size() || !reg_info[output_reg].is_scalar)
        {
          return false;
        }
      }

      for (unsigned int reg_slot = 0; reg_slot < program_.register_targets.size(); ++reg_slot)
      {
        if (!register_scalar_mask_[reg_slot] && program_.register_targets[reg_slot] >= 0)
        {
          return false;
        }
      }

      return true;
    }

    void initialize_numeric_jit()
    {
      auto & jit = egress_jit::OrcJitEngine::instance();
      if (!jit.available())
      {
        jit_status_ = jit.init_error();
        return;
      }

      egress_jit::NumericProgram numeric_program;
      if (!build_numeric_program(numeric_program))
      {
        jit_status_ = "numeric compatibility check failed";
        return;
      }

    #ifdef EGRESS_PROFILE
      numeric_jit_instruction_count_ = static_cast<uint64_t>(numeric_program.instructions.size());
    #endif

      auto kernel_or_err = jit.compile_numeric_program(numeric_program, "egress_udm_kernel");
      if (!kernel_or_err)
      {
        jit_status_ = llvm::toString(kernel_or_err.takeError());
        return;
      }

      jit_kernel_ = *kernel_or_err;
      numeric_temps_.assign(program_.register_count, 0.0);
      numeric_registers_.resize(registers_.size(), 0.0);
      numeric_next_registers_.resize(registers_.size(), 0.0);
      numeric_array_ptrs_.resize(numeric_array_storage_.size(), nullptr);
      numeric_array_sizes_.resize(numeric_array_storage_.size(), 0);
      for (std::size_t i = 0; i < numeric_array_storage_.size(); ++i)
      {
        numeric_array_ptrs_[i] = numeric_array_storage_[i].empty() ? nullptr : numeric_array_storage_[i].data();
        numeric_array_sizes_[i] = static_cast<uint64_t>(numeric_array_storage_[i].size());
      }
      for (unsigned int i = 0; i < registers_.size(); ++i)
      {
        numeric_registers_[i] = register_scalar_mask_[i] ? to_float64(registers_[i]) : 0.0;
      }
      jit_status_ = "numeric JIT active";
    }
#endif
};

class Graph
{
  public:
#ifdef EGRESS_PROFILE
    struct ModuleCompileStats
    {
      bool found = false;
      uint64_t instruction_count = 0;
      uint64_t register_count = 0;
      uint64_t numeric_jit_instruction_count = 0;
      std::string jit_status;
    };
#endif

    struct ModuleProfileStats
    {
      std::string module_name;
      uint64_t call_count = 0;
      double avg_call_ms = 0.0;
      double max_call_ms = 0.0;
    };

    struct ProfileStats
    {
      bool enabled = false;
      uint64_t callback_count = 0;
      double avg_callback_ms = 0.0;
      double max_callback_ms = 0.0;
      std::vector<ModuleProfileStats> modules;
    };

    explicit Graph(unsigned int bufferLength)
      : bufferLength_(bufferLength), outputBuffer(bufferLength, 0.0)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      runtimes_[0] = build_runtime_locked();
      runtimes_[1] = build_runtime_locked();
    }

    void process()
    {
#ifdef EGRESS_PROFILE
      const auto callback_start = std::chrono::steady_clock::now();
      struct LocalModuleStats
      {
        uint64_t call_count = 0;
        uint64_t total_ns = 0;
        uint64_t max_ns = 0;
      };
      std::unordered_map<std::string, LocalModuleStats> local_module_stats;
      local_module_stats.reserve(16);
#endif

      const uint32_t runtime_index = active_runtime_index_.load(std::memory_order_acquire);
      audio_runtime_index_.store(runtime_index, std::memory_order_release);
      audio_processing_.store(true, std::memory_order_release);

      RuntimeState & runtime = runtimes_[runtime_index];

      for (unsigned int sample = 0; sample < bufferLength_; ++sample)
      {
        for (auto & slot : runtime.modules)
        {
          if (!slot.module)
          {
            continue;
          }

#ifdef EGRESS_PROFILE
          const auto module_start = std::chrono::steady_clock::now();
#endif

          eval_input_program(runtime, slot.input_program, slot.input_registers, slot.module->inputs);

          slot.module->process();

#ifdef EGRESS_PROFILE
          const auto module_end = std::chrono::steady_clock::now();
          const uint64_t module_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(module_end - module_start).count());
          auto & stats = local_module_stats[slot.name];
          ++stats.call_count;
          stats.total_ns += module_ns;
          if (module_ns > stats.max_ns)
          {
            stats.max_ns = module_ns;
          }
#endif
        }

        double mixed = 0.0;
        for (const auto & tap : runtime.mix)
        {
          if (tap.module_id >= runtime.modules.size())
          {
            continue;
          }

          const auto & slot = runtime.modules[tap.module_id];
          if (tap.output_id >= slot.module->outputs.size())
          {
            continue;
          }

          mixed += expr::to_float64(slot.module->outputs[tap.output_id]) / 20.0;
        }
        outputBuffer[sample] = mixed;

        for (auto & slot : runtime.modules)
        {
          if (!slot.module)
          {
            continue;
          }
          slot.module->prev_outputs.swap(slot.module->outputs);
        }
      }

      audio_processing_.store(false, std::memory_order_release);

#ifdef EGRESS_PROFILE
      const auto callback_end = std::chrono::steady_clock::now();
      const uint64_t callback_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(callback_end - callback_start).count());
      profile_callback_count_.fetch_add(1, std::memory_order_relaxed);
      profile_total_callback_ns_.fetch_add(callback_ns, std::memory_order_relaxed);
      update_atomic_max(profile_max_callback_ns_, callback_ns);

      std::lock_guard<std::mutex> lock(profile_mutex_);
      for (const auto & [name, stats] : local_module_stats)
      {
        auto & dst = module_profile_stats_[name];
        dst.call_count += stats.call_count;
        dst.total_ns += stats.total_ns;
        if (stats.max_ns > dst.max_ns)
        {
          dst.max_ns = stats.max_ns;
        }
      }
#endif
    }

    ProfileStats profile_stats() const
    {
      ProfileStats stats;
#ifdef EGRESS_PROFILE
      stats.enabled = true;
      const uint64_t callback_count = profile_callback_count_.load(std::memory_order_relaxed);
      const uint64_t total_ns = profile_total_callback_ns_.load(std::memory_order_relaxed);
      const uint64_t max_ns = profile_max_callback_ns_.load(std::memory_order_relaxed);

      stats.callback_count = callback_count;
      stats.avg_callback_ms = callback_count == 0 ? 0.0 : (static_cast<double>(total_ns) / static_cast<double>(callback_count)) / 1e6;
      stats.max_callback_ms = static_cast<double>(max_ns) / 1e6;

      std::lock_guard<std::mutex> lock(profile_mutex_);
      stats.modules.reserve(module_profile_stats_.size());
      for (const auto & [name, module_stats] : module_profile_stats_)
      {
        ModuleProfileStats module;
        module.module_name = name;
        module.call_count = module_stats.call_count;
        module.avg_call_ms = module.call_count == 0
                               ? 0.0
                               : (static_cast<double>(module_stats.total_ns) / static_cast<double>(module.call_count)) / 1e6;
        module.max_call_ms = static_cast<double>(module_stats.max_ns) / 1e6;
        stats.modules.push_back(std::move(module));
      }
#endif
      return stats;
    }

    void reset_profile_stats()
    {
#ifdef EGRESS_PROFILE
      profile_callback_count_.store(0, std::memory_order_relaxed);
      profile_total_callback_ns_.store(0, std::memory_order_relaxed);
      profile_max_callback_ns_.store(0, std::memory_order_relaxed);

      std::lock_guard<std::mutex> lock(profile_mutex_);
      module_profile_stats_.clear();
#endif
    }

#ifdef EGRESS_PROFILE
    ModuleCompileStats module_compile_stats(const std::string & module_name) const
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      ModuleCompileStats result;
      auto it = control_modules_.find(module_name);
      if (it == control_modules_.end() || !it->second.module)
      {
        return result;
      }

      const Module::CompileStats stats = it->second.module->compile_stats();
      result.found = true;
      result.instruction_count = stats.instruction_count;
      result.register_count = stats.register_count;
      result.numeric_jit_instruction_count = stats.numeric_jit_instruction_count;
      result.jit_status = stats.jit_status;
      return result;
    }
#endif

    bool addModule(std::string name, mPtr new_module)
    {
      if (!new_module)
      {
        return false;
      }

      const unsigned int in_count = static_cast<unsigned int>(new_module->inputs.size());
      const unsigned int out_count = static_cast<unsigned int>(new_module->outputs.size());

      std::lock_guard<std::mutex> lock(pending_mutex_);
      if (control_modules_.find(name) != control_modules_.end())
      {
        return false;
      }

      ControlModule module;
      module.module = std::shared_ptr<Module>(std::move(new_module));
      module.in_count = in_count;
      module.out_count = out_count;
      module.input_exprs.assign(in_count, nullptr);

      control_modules_.emplace(std::move(name), std::move(module));
      rebuild_and_publish_runtime_locked();
      return true;
    }

    bool remove_module(const std::string & module_name)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      if (control_modules_.find(module_name) == control_modules_.end())
      {
        return false;
      }

      control_modules_.erase(module_name);

      control_mix_.erase(
        std::remove_if(
          control_mix_.begin(),
          control_mix_.end(),
          [&module_name](const outputID & out)
          {
            return out.first == module_name;
          }),
        control_mix_.end());

      std::vector<std::pair<std::string, unsigned int>> updated_inputs;
      for (auto & [name, module] : control_modules_)
      {
        for (unsigned int input_id = 0; input_id < module.input_exprs.size(); ++input_id)
        {
          bool removed_any = false;
          const ExprSpecPtr updated = replace_refs_with_zero(module.input_exprs[input_id], module_name, 0, true, removed_any);
          if (removed_any)
          {
            module.input_exprs[input_id] = simplify_expr(updated);
            updated_inputs.emplace_back(name, input_id);
          }
        }
      }

      (void)updated_inputs;
      rebuild_and_publish_runtime_locked();

      return true;
    }

    bool addOutput(outputID output)
    {
      const std::string & module_name = output.first;
      const unsigned int output_id = output.second;

      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto module_it = control_modules_.find(module_name);
      if (module_it == control_modules_.end())
      {
        return false;
      }

      if (output_id >= module_it->second.out_count)
      {
        return false;
      }

      control_mix_.push_back(output);
      rebuild_and_publish_runtime_locked();
      return true;
    }

    bool connect(
      std::string src_module,
      unsigned int src_output_id,
      std::string dst_module,
      unsigned int dst_input_id)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto src_it = control_modules_.find(src_module);
      auto dst_it = control_modules_.find(dst_module);
      if (src_it == control_modules_.end() || dst_it == control_modules_.end())
      {
        return false;
      }

      if (src_output_id >= src_it->second.out_count || dst_input_id >= dst_it->second.in_count)
      {
        return false;
      }

      ExprSpecPtr ref = expr::ref_expr(src_module, src_output_id);
      ExprSpecPtr & current = control_modules_[dst_module].input_exprs[dst_input_id];
      current = simplify_expr(append_expr(current, ref));
      rebuild_and_publish_runtime_locked();
      return true;
    }

    bool remove_connection(
      std::string src_module,
      unsigned int src_output_id,
      std::string dst_module,
      unsigned int dst_input_id)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto dst_module_it = control_modules_.find(dst_module);
      if (dst_module_it == control_modules_.end() || dst_input_id >= dst_module_it->second.input_exprs.size())
      {
        return false;
      }

      bool removed_any = false;
      ExprSpecPtr updated = replace_refs_with_zero(
        dst_module_it->second.input_exprs[dst_input_id],
        src_module,
        src_output_id,
        false,
        removed_any);

      if (!removed_any)
      {
        return false;
      }

      dst_module_it->second.input_exprs[dst_input_id] = simplify_expr(updated);
      rebuild_and_publish_runtime_locked();
      return true;
    }

    bool set_input_expr(const std::string & module_name, unsigned int input_id, ExprSpecPtr expr)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto module_it = control_modules_.find(module_name);
      if (module_it == control_modules_.end() || input_id >= module_it->second.in_count)
      {
        return false;
      }

      if (!validate_expr_refs(expr, false))
      {
        return false;
      }

      control_modules_[module_name].input_exprs[input_id] = simplify_expr(expr);
      rebuild_and_publish_runtime_locked();
      return true;
    }

    ExprSpecPtr get_input_expr(const std::string & module_name, unsigned int input_id) const
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto module_it = control_modules_.find(module_name);
      if (module_it == control_modules_.end() || input_id >= module_it->second.input_exprs.size())
      {
        return nullptr;
      }
      return module_it->second.input_exprs[input_id];
    }

    std::vector<outputID> incoming_connections(const std::string & dst_module, unsigned int dst_input_id) const
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      std::vector<outputID> sources;
      auto module_it = control_modules_.find(dst_module);
      if (module_it == control_modules_.end() || dst_input_id >= module_it->second.input_exprs.size())
      {
        return sources;
      }

      collect_refs(module_it->second.input_exprs[dst_input_id], sources);
      return sources;
    }

    unsigned int getBufferLength() const
    {
      return bufferLength_;
    }

    std::vector<double> outputBuffer;

  private:
    friend class Module;

    struct ModuleShape
    {
      unsigned int in_count;
      unsigned int out_count;
    };

    struct ControlModule
    {
      std::shared_ptr<Module> module;
      unsigned int in_count = 0;
      unsigned int out_count = 0;
      std::vector<ExprSpecPtr> input_exprs;
    };

    enum class OpCode
    {
      Literal,
      Ref,
      ArrayPack,
      Index,
      Add,
      AddConst,
      Sub,
      SubConstRhs,
      SubConstLhs,
      Mul,
      MulConst,
      Div,
      Pow,
      DivConstLhs,
      Mod,
      FloorDiv,
      BitAnd,
      BitOr,
      BitXor,
      LShift,
      RShift,
      Abs,
      Clamp,
      Log,
      Neg,
      Not,
      BitNot,
      Sin,
      Less,
      LessEqual,
      Greater,
      GreaterEqual,
      Equal,
      NotEqual
    };

    struct ExprInstr;
    struct RuntimeState;

    struct ExprInstr
    {
      OpCode opcode = OpCode::Literal;
      uint32_t dst = 0;
      uint32_t src_a = 0;
      uint32_t src_b = 0;
      uint32_t src_c = 0;
      std::vector<uint32_t> args;
      Value literal;
      uint32_t ref_module_id = 0;
      unsigned int ref_output_id = 0;
    };

    struct CompiledInputProgram
    {
      std::vector<ExprInstr> instructions;
      uint32_t register_count = 0;
      std::vector<uint32_t> result_registers;
    };

    struct ModuleSlot
    {
      std::string name;
      std::shared_ptr<Module> module;
      CompiledInputProgram input_program;
      std::vector<Value> input_registers;
    };

    struct MixTap
    {
      uint32_t module_id;
      unsigned int output_id;
    };

    struct RuntimeState
    {
      std::vector<ModuleSlot> modules;
      std::unordered_map<std::string, uint32_t> name_to_id;
      std::vector<MixTap> mix;
    };

#ifdef EGRESS_PROFILE
    struct ModuleTimingCounters
    {
      uint64_t call_count = 0;
      uint64_t total_ns = 0;
      uint64_t max_ns = 0;
    };

    static void update_atomic_max(std::atomic<uint64_t> & dst, uint64_t candidate)
    {
      uint64_t current = dst.load(std::memory_order_relaxed);
      while (current < candidate && !dst.compare_exchange_weak(current, candidate, std::memory_order_relaxed))
      {
      }
    }
#endif

    static bool is_zero_expr(const ExprSpecPtr & expr)
    {
      return expr != nullptr &&
             expr->kind == ExprKind::Literal &&
             expr->literal.type != ValueType::Array &&
             !is_truthy(expr->literal);
    }

    static bool is_one_expr(const ExprSpecPtr & expr)
    {
      return expr != nullptr && expr->kind == ExprKind::Literal &&
             expr->literal.type != ValueType::Array &&
             ((expr->literal.type == ValueType::Bool && expr->literal.bool_value) ||
              (expr->literal.type == ValueType::Int && expr->literal.int_value == 1) ||
              (expr->literal.type == ValueType::Float && expr->literal.float_value == 1.0));
    }

    static ExprSpecPtr append_expr(const ExprSpecPtr & lhs, const ExprSpecPtr & rhs)
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

    static ExprSpecPtr simplify_expr(const ExprSpecPtr & expr)
    {
      if (!expr)
      {
        return nullptr;
      }

      switch (expr->kind)
      {
        case ExprKind::Literal:
        case ExprKind::Ref:
        case ExprKind::InputValue:
        case ExprKind::RegisterValue:
        case ExprKind::SampleRate:
        case ExprKind::SampleIndex:
        case ExprKind::ArrayPack:
          return expr;
        case ExprKind::Clamp:
        {
          ExprSpecPtr value = simplify_expr(expr->lhs);
          ExprSpecPtr min_value = simplify_expr(expr->rhs);
          ExprSpecPtr max_value = simplify_expr(expr->args.empty() ? nullptr : expr->args.front());
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
        case ExprKind::Function:
          return expr::function_expr(expr->param_count, simplify_expr(expr->lhs));
        case ExprKind::Call:
        {
          std::vector<ExprSpecPtr> args;
          args.reserve(expr->args.size());
          for (const auto & arg : expr->args)
          {
            args.push_back(simplify_expr(arg));
          }
          return expr::call_expr(simplify_expr(expr->lhs), std::move(args));
        }
        case ExprKind::Index:
          return expr::binary_expr(ExprKind::Index, simplify_expr(expr->lhs), simplify_expr(expr->rhs));
        case ExprKind::Neg:
        {
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
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
            return expr::literal_expr(-to_int64(lhs->literal));
          }
          return expr::unary_expr(ExprKind::Neg, lhs);
        }
        case ExprKind::Abs:
        {
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          if (!lhs)
          {
            return nullptr;
          }
          if (lhs->kind == ExprKind::Literal)
          {
            return expr::literal_expr(~to_int64(lhs->literal));
          }
          return expr::unary_expr(ExprKind::BitNot, lhs);
        }
        case ExprKind::Log:
        {
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          if (!lhs)
          {
            return nullptr;
          }
          if (lhs->kind == ExprKind::Literal)
          {
            return expr::literal_expr(std::sin(to_float64(lhs->literal)));
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          ExprSpecPtr rhs = simplify_expr(expr->rhs);
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
            switch (expr->kind)
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
          return expr::binary_expr(expr->kind, lhs, rhs);
        }
        case ExprKind::Add:
        {
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          ExprSpecPtr rhs = simplify_expr(expr->rhs);
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          ExprSpecPtr rhs = simplify_expr(expr->rhs);
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          ExprSpecPtr rhs = simplify_expr(expr->rhs);
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
        case ExprKind::Div:
        {
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          ExprSpecPtr rhs = simplify_expr(expr->rhs);
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          ExprSpecPtr rhs = simplify_expr(expr->rhs);
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
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          ExprSpecPtr rhs = simplify_expr(expr->rhs);
          if (!lhs || !rhs)
          {
            return nullptr;
          }
          if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
          {
            switch (expr->kind)
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
          return expr::binary_expr(expr->kind, lhs, rhs);
        }
      }

      return expr;
    }

    static ExprSpecPtr replace_refs_with_zero(
      const ExprSpecPtr & expr,
      const std::string & module_name,
      unsigned int output_id,
      bool remove_all_outputs,
      bool & removed_any)
    {
      if (!expr)
      {
        return nullptr;
      }

      if (expr->kind == ExprKind::Ref)
      {
        const bool matches = expr->module_name == module_name &&
                             (remove_all_outputs || expr->output_id == output_id);
        if (matches)
        {
          removed_any = true;
          return nullptr;
        }
        return expr;
      }

      if (expr->kind == ExprKind::Literal)
      {
        return expr;
      }

      if (expr->kind == ExprKind::ArrayPack)
      {
        std::vector<ExprSpecPtr> items;
        items.reserve(expr->args.size());
        for (const auto & arg : expr->args)
        {
          items.push_back(replace_refs_with_zero(arg, module_name, output_id, remove_all_outputs, removed_any));
        }
        return expr::array_pack_expr(std::move(items));
      }

      if (expr->kind == ExprKind::Clamp)
      {
        return simplify_expr(expr::clamp_expr(
          replace_refs_with_zero(expr->lhs, module_name, output_id, remove_all_outputs, removed_any),
          replace_refs_with_zero(expr->rhs, module_name, output_id, remove_all_outputs, removed_any),
          replace_refs_with_zero(expr->args.empty() ? nullptr : expr->args.front(), module_name, output_id, remove_all_outputs, removed_any)));
      }

      if (expr->kind == ExprKind::Function)
      {
        return expr::function_expr(
          expr->param_count,
          replace_refs_with_zero(expr->lhs, module_name, output_id, remove_all_outputs, removed_any));
      }

      if (expr->kind == ExprKind::Call)
      {
        std::vector<ExprSpecPtr> args;
        args.reserve(expr->args.size());
        for (const auto & arg : expr->args)
        {
          args.push_back(replace_refs_with_zero(arg, module_name, output_id, remove_all_outputs, removed_any));
        }
        return expr::call_expr(
          replace_refs_with_zero(expr->lhs, module_name, output_id, remove_all_outputs, removed_any),
          std::move(args));
      }

      if (expr->kind == ExprKind::InputValue ||
          expr->kind == ExprKind::RegisterValue ||
          expr->kind == ExprKind::SampleRate ||
          expr->kind == ExprKind::SampleIndex)
      {
        return expr;
      }

      ExprSpecPtr lhs = replace_refs_with_zero(expr->lhs, module_name, output_id, remove_all_outputs, removed_any);
      ExprSpecPtr rhs = replace_refs_with_zero(expr->rhs, module_name, output_id, remove_all_outputs, removed_any);

      if (expr->kind == ExprKind::Abs)
      {
        return simplify_expr(expr::unary_expr(ExprKind::Abs, lhs));
      }

      if (expr->kind == ExprKind::Neg)
      {
        return simplify_expr(expr::unary_expr(ExprKind::Neg, lhs));
      }

      if (expr->kind == ExprKind::Not)
      {
        return simplify_expr(expr::unary_expr(ExprKind::Not, lhs));
      }

      if (expr->kind == ExprKind::BitNot)
      {
        return simplify_expr(expr::unary_expr(ExprKind::BitNot, lhs));
      }

      if (expr->kind == ExprKind::Log)
      {
        return simplify_expr(expr::unary_expr(ExprKind::Log, lhs));
      }

      if (expr->kind == ExprKind::Sin)
      {
        return simplify_expr(expr::unary_expr(ExprKind::Sin, lhs));
      }

      return simplify_expr(expr::binary_expr(expr->kind, lhs, rhs));
    }

    static void collect_refs(const ExprSpecPtr & expr, std::vector<outputID> & refs)
    {
      if (!expr)
      {
        return;
      }

      if (expr->kind == ExprKind::Ref)
      {
        refs.emplace_back(expr->module_name, expr->output_id);
        return;
      }

      if (expr->kind == ExprKind::Function)
      {
        collect_refs(expr->lhs, refs);
        return;
      }

      if (expr->kind == ExprKind::Call)
      {
        collect_refs(expr->lhs, refs);
        for (const auto & arg : expr->args)
        {
          collect_refs(arg, refs);
        }
        return;
      }

      if (expr->kind == ExprKind::ArrayPack)
      {
        for (const auto & arg : expr->args)
        {
          collect_refs(arg, refs);
        }
        return;
      }

      if (expr->kind == ExprKind::Clamp)
      {
        collect_refs(expr->lhs, refs);
        collect_refs(expr->rhs, refs);
        if (!expr->args.empty())
        {
          collect_refs(expr->args.front(), refs);
        }
        return;
      }

      collect_refs(expr->lhs, refs);
      collect_refs(expr->rhs, refs);
    }

    bool validate_expr_refs(const ExprSpecPtr & expr, bool allow_input_values) const
    {
      if (!expr)
      {
        return true;
      }

      if (expr->kind == ExprKind::Ref)
      {
        auto it = control_modules_.find(expr->module_name);
        return it != control_modules_.end() && expr->output_id < it->second.out_count;
      }

      if (expr->kind == ExprKind::Literal)
      {
        return true;
      }

      if (expr->kind == ExprKind::Function)
      {
        return validate_expr_refs(expr->lhs, true);
      }

      if (expr->kind == ExprKind::Call)
      {
        if (!validate_expr_refs(expr->lhs, true))
        {
          return false;
        }
        for (const auto & arg : expr->args)
        {
          if (!validate_expr_refs(arg, allow_input_values))
          {
            return false;
          }
        }
        return true;
      }

      if (expr->kind == ExprKind::Clamp)
      {
        return validate_expr_refs(expr->lhs, allow_input_values) &&
               validate_expr_refs(expr->rhs, allow_input_values) &&
               validate_expr_refs(expr->args.empty() ? nullptr : expr->args.front(), allow_input_values);
      }

      if (expr->kind == ExprKind::ArrayPack)
      {
        for (const auto & item : expr->args)
        {
          if (!validate_expr_refs(item, allow_input_values))
          {
            return false;
          }
        }
        return true;
      }

      if (expr->kind == ExprKind::Index)
      {
        return validate_expr_refs(expr->lhs, allow_input_values) &&
               validate_expr_refs(expr->rhs, allow_input_values);
      }

      if (expr->kind == ExprKind::InputValue ||
          expr->kind == ExprKind::RegisterValue ||
          expr->kind == ExprKind::SampleRate ||
          expr->kind == ExprKind::SampleIndex)
      {
        return allow_input_values && expr->kind == ExprKind::InputValue;
      }

      return validate_expr_refs(expr->lhs, allow_input_values) && validate_expr_refs(expr->rhs, allow_input_values);
    }

    void wait_for_runtime_available(uint32_t runtime_index) const
    {
      while (audio_processing_.load(std::memory_order_acquire) &&
             audio_runtime_index_.load(std::memory_order_acquire) == runtime_index)
      {
        std::this_thread::yield();
      }
    }

    RuntimeState build_runtime_locked() const
    {
      RuntimeState runtime;
      runtime.modules.reserve(control_modules_.size());
      runtime.name_to_id.reserve(control_modules_.size());

      for (const auto & [name, module] : control_modules_)
      {
        const uint32_t module_id = static_cast<uint32_t>(runtime.modules.size());
        runtime.name_to_id.emplace(name, module_id);

        ModuleSlot slot;
        slot.name = name;
        slot.module = module.module;
        slot.input_program.result_registers.resize(module.in_count, 0);
        runtime.modules.push_back(std::move(slot));
      }

      for (const auto & [name, module] : control_modules_)
      {
        auto id_it = runtime.name_to_id.find(name);
        if (id_it == runtime.name_to_id.end())
        {
          continue;
        }

        ModuleSlot & slot = runtime.modules[id_it->second];
        const unsigned int input_count = static_cast<unsigned int>(slot.input_program.result_registers.size());
        slot.input_program = compile_input_program(module.input_exprs, input_count, runtime);
        slot.input_registers.assign(slot.input_program.register_count, float_value(0.0));
      }

      runtime.mix.reserve(control_mix_.size());
      for (const auto & tap : control_mix_)
      {
        auto it = runtime.name_to_id.find(tap.first);
        if (it == runtime.name_to_id.end())
        {
          continue;
        }

        const uint32_t module_id = it->second;
        if (module_id >= runtime.modules.size() ||
            !runtime.modules[module_id].module ||
            tap.second >= runtime.modules[module_id].module->outputs.size())
        {
          continue;
        }

        runtime.mix.push_back(MixTap{module_id, tap.second});
      }

      return runtime;
    }

    void rebuild_and_publish_runtime_locked()
    {
      const uint32_t active = active_runtime_index_.load(std::memory_order_acquire);
      const uint32_t inactive = 1U - active;
      wait_for_runtime_available(inactive);
      runtimes_[inactive] = build_runtime_locked();
      active_runtime_index_.store(inactive, std::memory_order_release);
    }

    static void eval_instruction(const RuntimeState & runtime, const ExprInstr & instr, Value * registers)
    {
      switch (instr.opcode)
      {
        case OpCode::Literal:
          registers[instr.dst] = instr.literal;
          break;
        case OpCode::Ref:
        {
          if (instr.ref_module_id >= runtime.modules.size())
          {
            registers[instr.dst] = float_value(0.0);
            break;
          }
          const auto & slot = runtime.modules[instr.ref_module_id];
          if (!slot.module || instr.ref_output_id >= slot.module->prev_outputs.size())
          {
            registers[instr.dst] = float_value(0.0);
            break;
          }
          registers[instr.dst] = slot.module->prev_outputs[instr.ref_output_id];
          break;
        }
        case OpCode::ArrayPack:
        {
          std::vector<Value> items;
          items.reserve(instr.args.size());
          for (uint32_t src : instr.args)
          {
            if (expr::is_array(registers[src]))
            {
              throw std::invalid_argument("Nested arrays are not supported.");
            }
            items.push_back(registers[src]);
          }
          registers[instr.dst] = expr::array_value(std::move(items));
          break;
        }
        case OpCode::Index:
        {
          const Value & array_value = registers[instr.src_a];
          if (!expr::is_array(array_value))
          {
            registers[instr.dst] = float_value(0.0);
            break;
          }
          const int64_t index = expr::to_int64(registers[instr.src_b]);
          if (index < 0 || static_cast<std::size_t>(index) >= array_value.array_items.size())
          {
            throw std::out_of_range("Array index out of range.");
          }
          registers[instr.dst] = array_value.array_items[static_cast<std::size_t>(index)];
          break;
        }
        case OpCode::Add:
          registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::AddConst:
          registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], instr.literal);
          break;
        case OpCode::Sub:
          registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::SubConstRhs:
          registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], instr.literal);
          break;
        case OpCode::SubConstLhs:
          registers[instr.dst] = expr_eval::sub_values(instr.literal, registers[instr.src_a]);
          break;
        case OpCode::Mul:
          registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::MulConst:
          registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], instr.literal);
          break;
        case OpCode::Div:
          registers[instr.dst] = expr_eval::div_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::DivConstLhs:
          registers[instr.dst] = expr_eval::div_values(instr.literal, registers[instr.src_a]);
          break;
        case OpCode::Mod:
          registers[instr.dst] = expr_eval::mod_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::FloorDiv:
          registers[instr.dst] = expr_eval::floor_div_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::BitAnd:
          registers[instr.dst] = expr_eval::bit_and_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::BitOr:
          registers[instr.dst] = expr_eval::bit_or_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::BitXor:
          registers[instr.dst] = expr_eval::bit_xor_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::LShift:
          registers[instr.dst] = expr_eval::lshift_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::RShift:
          registers[instr.dst] = expr_eval::rshift_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::Abs:
          registers[instr.dst] = expr_eval::abs_value(registers[instr.src_a]);
          break;
        case OpCode::Clamp:
          registers[instr.dst] = expr_eval::clamp_values(registers[instr.src_a], registers[instr.src_b], registers[instr.src_c]);
          break;
        case OpCode::Log:
          registers[instr.dst] = expr_eval::log_value(registers[instr.src_a]);
          break;
        case OpCode::Neg:
          registers[instr.dst] = expr_eval::neg_value(registers[instr.src_a]);
          break;
        case OpCode::Not:
          registers[instr.dst] = expr_eval::not_value(registers[instr.src_a]);
          break;
        case OpCode::BitNot:
          registers[instr.dst] = expr_eval::bit_not_value(registers[instr.src_a]);
          break;
        case OpCode::Sin:
          registers[instr.dst] = expr_eval::sin_value(registers[instr.src_a]);
          break;
        case OpCode::Pow:
          registers[instr.dst] = expr_eval::pow_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::Less:
          registers[instr.dst] = expr_eval::less_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::LessEqual:
          registers[instr.dst] = expr_eval::less_equal_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::Greater:
          registers[instr.dst] = expr_eval::greater_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::GreaterEqual:
          registers[instr.dst] = expr_eval::greater_equal_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::Equal:
          registers[instr.dst] = expr_eval::equal_values(registers[instr.src_a], registers[instr.src_b]);
          break;
        case OpCode::NotEqual:
          registers[instr.dst] = expr_eval::not_equal_values(registers[instr.src_a], registers[instr.src_b]);
          break;
      }
    }

    uint32_t compile_expr_node(
      const ExprSpecPtr & expr,
      CompiledInputProgram & compiled,
      const RuntimeState & runtime) const
    {
      if (!expr)
      {
        ExprInstr instr;
        instr.opcode = OpCode::Literal;
        instr.dst = compiled.register_count++;
        instr.literal = float_value(0.0);
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Literal)
      {
        ExprInstr instr;
        instr.opcode = OpCode::Literal;
        instr.dst = compiled.register_count++;
        instr.literal = expr->literal;
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Ref)
      {
        auto it = runtime.name_to_id.find(expr->module_name);
        if (it == runtime.name_to_id.end())
        {
          ExprInstr instr;
          instr.opcode = OpCode::Literal;
          instr.dst = compiled.register_count++;
          instr.literal = float_value(0.0);
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        ExprInstr instr;
        instr.opcode = OpCode::Ref;
        instr.dst = compiled.register_count++;
        instr.ref_module_id = it->second;
        instr.ref_output_id = expr->output_id;
        compiled.instructions.push_back(instr);

        return instr.dst;
      }

      if (expr->kind == ExprKind::ArrayPack)
      {
        ExprInstr instr;
        instr.opcode = OpCode::ArrayPack;
        instr.dst = compiled.register_count++;
        instr.args.reserve(expr->args.size());
        for (const auto & arg : expr->args)
        {
          instr.args.push_back(compile_expr_node(arg, compiled, runtime));
        }
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Index)
      {
        ExprInstr instr;
        instr.opcode = OpCode::Index;
        instr.dst = compiled.register_count++;
        instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
        instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Abs ||
          expr->kind == ExprKind::Neg ||
          expr->kind == ExprKind::Not ||
          expr->kind == ExprKind::BitNot ||
          expr->kind == ExprKind::Log ||
          expr->kind == ExprKind::Sin)
      {
        const uint32_t operand = compile_expr_node(expr->lhs, compiled, runtime);
        ExprInstr instr;
        instr.opcode = expr->kind == ExprKind::Abs
                         ? OpCode::Abs
                         : (expr->kind == ExprKind::Neg
                              ? OpCode::Neg
                              : (expr->kind == ExprKind::Not
                                   ? OpCode::Not
                                   : (expr->kind == ExprKind::BitNot
                                        ? OpCode::BitNot
                                        : (expr->kind == ExprKind::Log ? OpCode::Log : OpCode::Sin))));
        instr.dst = compiled.register_count++;
        instr.src_a = operand;
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Clamp)
      {
        ExprInstr instr;
        instr.opcode = OpCode::Clamp;
        instr.dst = compiled.register_count++;
        instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
        instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
        instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, runtime);
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Add)
      {
        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::AddConst;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::AddConst;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = expr->rhs->literal;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      if (expr->kind == ExprKind::Mul)
      {
        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::MulConst;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::MulConst;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = expr->rhs->literal;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      if (expr->kind == ExprKind::Sub)
      {
        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::SubConstRhs;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = expr->rhs->literal;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::SubConstLhs;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      if (expr->kind == ExprKind::Div)
      {
        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::MulConst;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = to_float64(expr->rhs->literal) == 0.0
                            ? float_value(0.0)
                            : float_value(1.0 / to_float64(expr->rhs->literal));
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
          ExprInstr instr;
          instr.opcode = OpCode::DivConstLhs;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);

      ExprInstr instr;
      switch (expr->kind)
      {
        case ExprKind::Add:
          instr.opcode = OpCode::Add;
          break;
        case ExprKind::Sub:
          instr.opcode = OpCode::Sub;
          break;
        case ExprKind::Mul:
          instr.opcode = OpCode::Mul;
          break;
        case ExprKind::Div:
          instr.opcode = OpCode::Div;
          break;
        case ExprKind::Pow:
          instr.opcode = OpCode::Pow;
          break;
        case ExprKind::Mod:
          instr.opcode = OpCode::Mod;
          break;
        case ExprKind::FloorDiv:
          instr.opcode = OpCode::FloorDiv;
          break;
        case ExprKind::BitAnd:
          instr.opcode = OpCode::BitAnd;
          break;
        case ExprKind::BitOr:
          instr.opcode = OpCode::BitOr;
          break;
        case ExprKind::BitXor:
          instr.opcode = OpCode::BitXor;
          break;
        case ExprKind::LShift:
          instr.opcode = OpCode::LShift;
          break;
        case ExprKind::RShift:
          instr.opcode = OpCode::RShift;
          break;
        case ExprKind::Less:
          instr.opcode = OpCode::Less;
          break;
        case ExprKind::LessEqual:
          instr.opcode = OpCode::LessEqual;
          break;
        case ExprKind::Greater:
          instr.opcode = OpCode::Greater;
          break;
        case ExprKind::GreaterEqual:
          instr.opcode = OpCode::GreaterEqual;
          break;
        case ExprKind::Equal:
          instr.opcode = OpCode::Equal;
          break;
        case ExprKind::NotEqual:
          instr.opcode = OpCode::NotEqual;
          break;
        default:
          instr.opcode = OpCode::Literal;
          instr.literal = float_value(0.0);
          break;
      }
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.src_b = rhs;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    CompiledInputProgram compile_input_program(
      const std::vector<ExprSpecPtr> & exprs,
      unsigned int input_count,
      const RuntimeState & runtime) const
    {
      CompiledInputProgram compiled;
      compiled.result_registers.assign(input_count, 0);

      for (unsigned int input_id = 0; input_id < input_count; ++input_id)
      {
        const ExprSpecPtr expr = input_id < exprs.size() ? exprs[input_id] : nullptr;
        ExprSpecPtr inlined = egress_expr_inline::inline_functions(expr);
        compiled.result_registers[input_id] = compile_expr_node(inlined, compiled, runtime);
      }

      return compiled;
    }

    void eval_input_program(
      const RuntimeState & runtime,
      const CompiledInputProgram & program,
      std::vector<Value> & registers,
      std::vector<Value> & inputs) const
    {
      if (program.instructions.empty())
      {
        for (auto & input : inputs)
        {
          input = float_value(0.0);
        }
        return;
      }

      if (registers.size() < program.register_count)
      {
        registers.resize(program.register_count, float_value(0.0));
      }

      for (const auto & instr : program.instructions)
      {
        eval_instruction(runtime, instr, registers.data());
      }

      const unsigned int input_limit = std::min(
        static_cast<unsigned int>(inputs.size()),
        static_cast<unsigned int>(program.result_registers.size()));

      for (unsigned int input_id = 0; input_id < input_limit; ++input_id)
      {
        inputs[input_id] = registers[program.result_registers[input_id]];
      }
    }

    unsigned int bufferLength_ = 0;

    std::array<RuntimeState, 2> runtimes_;
    std::atomic<uint32_t> active_runtime_index_{0};
    std::atomic<uint32_t> audio_runtime_index_{0};
    std::atomic<bool> audio_processing_{false};

    std::unordered_map<std::string, ControlModule> control_modules_;
    std::vector<outputID> control_mix_;

    mutable std::mutex pending_mutex_;

  #ifdef EGRESS_PROFILE
    std::atomic<uint64_t> profile_callback_count_{0};
    std::atomic<uint64_t> profile_total_callback_ns_{0};
    std::atomic<uint64_t> profile_max_callback_ns_{0};
    mutable std::mutex profile_mutex_;
    std::unordered_map<std::string, ModuleTimingCounters> module_profile_stats_;
  #endif
};
