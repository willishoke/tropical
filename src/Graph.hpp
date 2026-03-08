#include "Expr.hpp"
#include "Module.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using inputID = std::pair<std::string, unsigned int>;
using outputID = std::pair<std::string, unsigned int>;
using mPtr = std::unique_ptr<Module>;

class Graph
{
  public:
    using ValueType = egress_expr::ValueType;
    using Value = egress_expr::Value;
    using ExprKind = egress_expr::ExprKind;
    using ExprSpec = egress_expr::ExprSpec;
    using ExprSpecPtr = egress_expr::ExprSpecPtr;

    explicit Graph(unsigned int bufferLength)
      : bufferLength_(bufferLength), outputBuffer(bufferLength, 0.0)
    {
    }

    void process()
    {
      apply_pending_commands();

      for (unsigned int sample = 0; sample < bufferLength_; ++sample)
      {
        for (uint32_t module_id : execution_order_)
        {
          auto & slot = modules_[module_id];
          if (!slot.active)
          {
            continue;
          }

          for (unsigned int input_id = 0; input_id < slot.input_exprs.size(); ++input_id)
          {
            slot.module->inputs[input_id] = eval_expr(slot.input_exprs[input_id], slot.input_registers[input_id]);
          }

          slot.module->process();
        }

        double mixed = 0.0;
        for (const auto & tap : mix_)
        {
          mixed += modules_[tap.module_id].module->outputs[tap.output_id] / 20.0;
        }
        outputBuffer[sample] = mixed;
      }
    }

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

      control_modules_.emplace(name, ModuleShape{in_count, out_count});
      control_input_exprs_.emplace(name, std::vector<ExprSpecPtr>(in_count));

      Command command;
      command.type = CommandType::AddModule;
      command.module_name = std::move(name);
      command.module = std::move(new_module);
      command_queue_.push_back(std::move(command));
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
      control_input_exprs_.erase(module_name);

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
      for (auto & [name, inputs] : control_input_exprs_)
      {
        for (unsigned int input_id = 0; input_id < inputs.size(); ++input_id)
        {
          bool removed_any = false;
          const ExprSpecPtr updated = replace_refs_with_zero(inputs[input_id], module_name, 0, true, removed_any);
          if (removed_any)
          {
            inputs[input_id] = simplify_expr(updated);
            updated_inputs.emplace_back(name, input_id);
          }
        }
      }

      Command remove_command;
      remove_command.type = CommandType::RemoveModule;
      remove_command.module_name = module_name;
      command_queue_.push_back(std::move(remove_command));

      for (const auto & [name, input_id] : updated_inputs)
      {
        queue_set_input_expr(name, input_id, control_input_exprs_[name][input_id]);
      }

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

      Command command;
      command.type = CommandType::AddOutput;
      command.module_name = module_name;
      command.src_output_id = output_id;
      command_queue_.push_back(std::move(command));
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

      ExprSpecPtr ref = ref_expr(src_module, src_output_id);
      ExprSpecPtr & current = control_input_exprs_[dst_module][dst_input_id];
      current = simplify_expr(append_expr(current, ref));
      queue_set_input_expr(dst_module, dst_input_id, current);
      return true;
    }

    bool remove_connection(
      std::string src_module,
      unsigned int src_output_id,
      std::string dst_module,
      unsigned int dst_input_id)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto dst_inputs_it = control_input_exprs_.find(dst_module);
      if (dst_inputs_it == control_input_exprs_.end() || dst_input_id >= dst_inputs_it->second.size())
      {
        return false;
      }

      bool removed_any = false;
      ExprSpecPtr updated = replace_refs_with_zero(
        dst_inputs_it->second[dst_input_id],
        src_module,
        src_output_id,
        false,
        removed_any);

      if (!removed_any)
      {
        return false;
      }

      dst_inputs_it->second[dst_input_id] = simplify_expr(updated);
      queue_set_input_expr(dst_module, dst_input_id, dst_inputs_it->second[dst_input_id]);
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

      if (!validate_expr_refs(expr))
      {
        return false;
      }

      control_input_exprs_[module_name][input_id] = simplify_expr(expr);
      queue_set_input_expr(module_name, input_id, control_input_exprs_[module_name][input_id]);
      return true;
    }

    ExprSpecPtr get_input_expr(const std::string & module_name, unsigned int input_id) const
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto module_it = control_input_exprs_.find(module_name);
      if (module_it == control_input_exprs_.end() || input_id >= module_it->second.size())
      {
        return nullptr;
      }
      return module_it->second[input_id];
    }

    std::vector<outputID> incoming_connections(const std::string & dst_module, unsigned int dst_input_id) const
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      std::vector<outputID> sources;
      auto module_it = control_input_exprs_.find(dst_module);
      if (module_it == control_input_exprs_.end() || dst_input_id >= module_it->second.size())
      {
        return sources;
      }

      collect_refs(module_it->second[dst_input_id], sources);
      return sources;
    }

    unsigned int getBufferLength() const
    {
      return bufferLength_;
    }

    static ExprSpecPtr literal_expr(Value value)
    {
      return egress_expr::literal_expr(std::move(value));
    }

    static ExprSpecPtr literal_expr(double value)
    {
      return egress_expr::literal_expr(value);
    }

    static ExprSpecPtr literal_expr(int64_t value)
    {
      return egress_expr::literal_expr(value);
    }

    static ExprSpecPtr literal_expr(bool value)
    {
      return egress_expr::literal_expr(value);
    }

    static Value int_literal_value(int64_t value)
    {
      return egress_expr::int_value(value);
    }

    static Value float_literal_value(double value)
    {
      return egress_expr::float_value(value);
    }

    static Value bool_literal_value(bool value)
    {
      return egress_expr::bool_value(value);
    }

    static Value array_literal_value(std::vector<Value> items)
    {
      return egress_expr::array_value(std::move(items));
    }

    static ExprSpecPtr ref_expr(std::string module_name, unsigned int output_id)
    {
      return egress_expr::ref_expr(std::move(module_name), output_id);
    }

    static ExprSpecPtr input_value_expr(unsigned int input_id)
    {
      return egress_expr::input_value_expr(input_id);
    }

    static ExprSpecPtr register_value_expr(unsigned int register_id)
    {
      return egress_expr::register_value_expr(register_id);
    }

    static ExprSpecPtr sample_rate_expr()
    {
      return egress_expr::sample_rate_expr();
    }

    static ExprSpecPtr sample_index_expr()
    {
      return egress_expr::sample_index_expr();
    }

    static ExprSpecPtr unary_expr(ExprKind kind, ExprSpecPtr operand)
    {
      return egress_expr::unary_expr(kind, std::move(operand));
    }

    static ExprSpecPtr binary_expr(ExprKind kind, ExprSpecPtr lhs, ExprSpecPtr rhs)
    {
      return egress_expr::binary_expr(kind, std::move(lhs), std::move(rhs));
    }

    static ExprSpecPtr array_pack_expr(std::vector<ExprSpecPtr> items)
    {
      return egress_expr::array_pack_expr(std::move(items));
    }

    static ExprSpecPtr index_expr(ExprSpecPtr array_expr, ExprSpecPtr index_expr)
    {
      return egress_expr::index_expr(std::move(array_expr), std::move(index_expr));
    }

    std::vector<double> outputBuffer;

  private:
    friend class UserDefinedModule;

    struct ModuleShape
    {
      unsigned int in_count;
      unsigned int out_count;
    };

    struct ExprInstr;
    using ExprKernel = void (*)(const Graph &, const ExprInstr &, Value *);

    struct ExprInstr
    {
      ExprKind kind = ExprKind::Literal;
      uint32_t dst = 0;
      uint32_t src_a = 0;
      uint32_t src_b = 0;
      Value literal;
      uint32_t ref_module_id = 0;
      unsigned int ref_output_id = 0;
      ExprKernel kernel = nullptr;
    };

    struct CompiledExpr
    {
      std::vector<ExprInstr> instructions;
      std::vector<uint32_t> dependencies;
      uint32_t register_count = 0;
      uint32_t result_register = 0;
    };

    struct ModuleSlot
    {
      std::string name;
      mPtr module;
      std::vector<CompiledExpr> input_exprs;
      std::vector<std::vector<Value>> input_registers;
      bool active = false;
    };

    struct MixTap
    {
      uint32_t module_id;
      unsigned int output_id;
    };

    enum class CommandType
    {
      AddModule,
      RemoveModule,
      SetInputExpr,
      AddOutput
    };

    struct Command
    {
      CommandType type{};
      std::string module_name;
      unsigned int src_output_id = 0;
      unsigned int dst_input_id = 0;
      ExprSpecPtr expr;
      mPtr module;
    };

    static Value int_value(int64_t value)
    {
      return egress_expr::int_value(value);
    }

    static Value float_value(double value)
    {
      return egress_expr::float_value(value);
    }

    static Value bool_value(bool value)
    {
      return egress_expr::bool_value(value);
    }

    static Value array_value(std::vector<Value> items)
    {
      return egress_expr::array_value(std::move(items));
    }

    static bool is_truthy(const Value & value)
    {
      return egress_expr::is_truthy(value);
    }

    static double to_float64(const Value & value)
    {
      return egress_expr::to_float64(value);
    }

    static int64_t to_int64(const Value & value)
    {
      return egress_expr::to_int64(value);
    }

    static ValueType arithmetic_type(const Value & lhs, const Value & rhs)
    {
      return egress_expr::arithmetic_type(lhs, rhs);
    }

    static bool is_array(const Value & value)
    {
      return egress_expr::is_array(value);
    }

    template <typename Func>
    static Value map_unary(const Value & value, Func func)
    {
      return egress_expr::map_unary(value, func);
    }

    template <typename Func>
    static Value map_binary(const Value & lhs, const Value & rhs, Func func)
    {
      return egress_expr::map_binary(lhs, rhs, func);
    }

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
      return binary_expr(ExprKind::Add, lhs, rhs);
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
        case ExprKind::Index:
          return binary_expr(ExprKind::Index, simplify_expr(expr->lhs), simplify_expr(expr->rhs));
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
              return literal_expr(-lhs->literal.float_value);
            }
            return literal_expr(-to_int64(lhs->literal));
          }
          return unary_expr(ExprKind::Neg, lhs);
        }
        case ExprKind::Not:
        {
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          if (!lhs)
          {
            return literal_expr(true);
          }
          if (lhs->kind == ExprKind::Literal)
          {
            return literal_expr(map_unary(lhs->literal, [](const Value & value) { return bool_value(!is_truthy(value)); }));
          }
          return unary_expr(ExprKind::Not, lhs);
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
            return literal_expr(~to_int64(lhs->literal));
          }
          return unary_expr(ExprKind::BitNot, lhs);
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
            return literal_expr(std::sin(to_float64(lhs->literal)));
          }
          return unary_expr(ExprKind::Sin, lhs);
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
            lhs = literal_expr(0.0);
          }
          if (!rhs)
          {
            rhs = literal_expr(0.0);
          }
          if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
          {
            return literal_expr(map_binary(lhs->literal, rhs->literal, [expr](const Value & left, const Value & right) {
              bool result = false;
              switch (expr->kind)
              {
                case ExprKind::Less:
                  result = to_float64(left) < to_float64(right);
                  break;
                case ExprKind::LessEqual:
                  result = to_float64(left) <= to_float64(right);
                  break;
                case ExprKind::Greater:
                  result = to_float64(left) > to_float64(right);
                  break;
                case ExprKind::GreaterEqual:
                  result = to_float64(left) >= to_float64(right);
                  break;
                case ExprKind::Equal:
                  result = to_float64(left) == to_float64(right);
                  break;
                case ExprKind::NotEqual:
                  result = to_float64(left) != to_float64(right);
                  break;
                default:
                  break;
              }
              return bool_value(result);
            }));
          }
          return binary_expr(expr->kind, lhs, rhs);
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
            return literal_expr(map_binary(lhs->literal, rhs->literal, [](const Value & left, const Value & right) {
              return add_values(left, right);
            }));
          }
          return binary_expr(ExprKind::Add, lhs, rhs);
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
            return simplify_expr(unary_expr(ExprKind::Neg, rhs));
          }
          if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal)
          {
            return literal_expr(map_binary(lhs->literal, rhs->literal, [](const Value & left, const Value & right) {
              return sub_values(left, right);
            }));
          }
          return binary_expr(ExprKind::Sub, lhs, rhs);
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
            return literal_expr(map_binary(lhs->literal, rhs->literal, [](const Value & left, const Value & right) {
              return mul_values(left, right);
            }));
          }
          return binary_expr(ExprKind::Mul, lhs, rhs);
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
            return literal_expr(map_binary(lhs->literal, rhs->literal, [](const Value & left, const Value & right) {
              return div_values(left, right);
            }));
          }
          return binary_expr(ExprKind::Div, lhs, rhs);
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
            return literal_expr(map_binary(lhs->literal, rhs->literal, [expr](const Value & left, const Value & right) {
              switch (expr->kind)
              {
                case ExprKind::Mod:
                  if (arithmetic_type(left, right) == ValueType::Float)
                  {
                    return to_float64(right) == 0.0 ? float_value(0.0) : float_value(std::fmod(to_float64(left), to_float64(right)));
                  }
                  return to_int64(right) == 0 ? int_value(0) : int_value(to_int64(left) % to_int64(right));
                case ExprKind::FloorDiv:
                  return to_float64(right) == 0.0 ? float_value(0.0) : float_value(std::floor(to_float64(left) / to_float64(right)));
                case ExprKind::BitAnd:
                  return int_value(to_int64(left) & to_int64(right));
                case ExprKind::BitOr:
                  return int_value(to_int64(left) | to_int64(right));
                case ExprKind::BitXor:
                  return int_value(to_int64(left) ^ to_int64(right));
                case ExprKind::LShift:
                  return int_value(to_int64(left) << normalize_shift(right));
                case ExprKind::RShift:
                  return int_value(to_int64(left) >> normalize_shift(right));
                default:
                  return float_value(0.0);
              }
            }));
          }
          return binary_expr(expr->kind, lhs, rhs);
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
        return array_pack_expr(std::move(items));
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

      if (expr->kind == ExprKind::Neg)
      {
        return simplify_expr(unary_expr(ExprKind::Neg, lhs));
      }

      if (expr->kind == ExprKind::Not)
      {
        return simplify_expr(unary_expr(ExprKind::Not, lhs));
      }

      if (expr->kind == ExprKind::BitNot)
      {
        return simplify_expr(unary_expr(ExprKind::BitNot, lhs));
      }

      if (expr->kind == ExprKind::Sin)
      {
        return simplify_expr(unary_expr(ExprKind::Sin, lhs));
      }

      return simplify_expr(binary_expr(expr->kind, lhs, rhs));
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

      if (expr->kind == ExprKind::ArrayPack)
      {
        for (const auto & arg : expr->args)
        {
          collect_refs(arg, refs);
        }
        return;
      }

      collect_refs(expr->lhs, refs);
      collect_refs(expr->rhs, refs);
    }

    bool validate_expr_refs(const ExprSpecPtr & expr) const
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
        return expr->literal.type != ValueType::Array;
      }

      if (expr->kind == ExprKind::ArrayPack || expr->kind == ExprKind::Index)
      {
        return false;
      }

      if (expr->kind == ExprKind::InputValue ||
          expr->kind == ExprKind::RegisterValue ||
          expr->kind == ExprKind::SampleRate ||
          expr->kind == ExprKind::SampleIndex)
      {
        return false;
      }

      return validate_expr_refs(expr->lhs) && validate_expr_refs(expr->rhs);
    }

    void queue_set_input_expr(const std::string & module_name, unsigned int input_id, const ExprSpecPtr & expr)
    {
      Command command;
      command.type = CommandType::SetInputExpr;
      command.module_name = module_name;
      command.dst_input_id = input_id;
      command.expr = expr;
      command_queue_.push_back(std::move(command));
    }

    void apply_pending_commands()
    {
      std::vector<Command> local_commands;
      {
        std::unique_lock<std::mutex> lock(pending_mutex_, std::try_to_lock);
        if (!lock.owns_lock())
        {
          return;
        }

        if (command_queue_.empty())
        {
          return;
        }
        local_commands.swap(command_queue_);
      }

      bool needs_order_rebuild = false;
      for (auto & command : local_commands)
      {
        switch (command.type)
        {
          case CommandType::AddModule:
            apply_add_module(command.module_name, std::move(command.module));
            needs_order_rebuild = true;
            break;
          case CommandType::RemoveModule:
            apply_remove_module(command.module_name);
            needs_order_rebuild = true;
            break;
          case CommandType::SetInputExpr:
            apply_set_input_expr(command.module_name, command.dst_input_id, command.expr);
            needs_order_rebuild = true;
            break;
          case CommandType::AddOutput:
            apply_add_output(command.module_name, command.src_output_id);
            break;
        }
      }

      if (needs_order_rebuild)
      {
        rebuild_execution_order();
      }
    }

    void apply_add_module(std::string module_name, mPtr module)
    {
      if (!module || name_to_id_.find(module_name) != name_to_id_.end())
      {
        return;
      }

      const std::size_t input_count = module->inputs.size();

      uint32_t module_id = 0;
      if (!free_ids_.empty())
      {
        module_id = free_ids_.back();
        free_ids_.pop_back();
        modules_[module_id].name = std::move(module_name);
        modules_[module_id].module = std::move(module);
        modules_[module_id].input_exprs.assign(modules_[module_id].module->inputs.size(), CompiledExpr{});
        modules_[module_id].input_registers.assign(modules_[module_id].module->inputs.size(), std::vector<Value>{});
        modules_[module_id].active = true;
      }
      else
      {
        module_id = static_cast<uint32_t>(modules_.size());
        modules_.push_back(ModuleSlot{
          std::move(module_name),
          std::move(module),
          std::vector<CompiledExpr>(input_count),
          std::vector<std::vector<Value>>(input_count),
          true});
      }

      name_to_id_[modules_[module_id].name] = module_id;
    }

    void apply_remove_module(const std::string & module_name)
    {
      auto it = name_to_id_.find(module_name);
      if (it == name_to_id_.end())
      {
        return;
      }

      const uint32_t module_id = it->second;
      name_to_id_.erase(it);

      mix_.erase(
        std::remove_if(
          mix_.begin(),
          mix_.end(),
          [module_id](const MixTap & tap)
          {
            return tap.module_id == module_id;
          }),
        mix_.end());

      modules_[module_id].module.reset();
      modules_[module_id].input_exprs.clear();
      modules_[module_id].input_registers.clear();
      modules_[module_id].name.clear();
      modules_[module_id].active = false;
      free_ids_.push_back(module_id);
    }

    static void exec_literal(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = instr.literal;
    }

    static void exec_ref(const Graph & graph, const ExprInstr & instr, Value * registers)
    {
      if (instr.ref_module_id >= graph.modules_.size() || !graph.modules_[instr.ref_module_id].active)
      {
        registers[instr.dst] = float_value(0.0);
        return;
      }

      registers[instr.dst] = float_value(graph.modules_[instr.ref_module_id].module->outputs[instr.ref_output_id]);
    }

    static Value add_values(const Value & lhs, const Value & rhs)
    {
      return map_binary(lhs, rhs, [](const Value & left, const Value & right) {
        if (arithmetic_type(left, right) == ValueType::Float)
        {
          return float_value(to_float64(left) + to_float64(right));
        }
        return int_value(to_int64(left) + to_int64(right));
      });
    }

    static Value sub_values(const Value & lhs, const Value & rhs)
    {
      return map_binary(lhs, rhs, [](const Value & left, const Value & right) {
        if (arithmetic_type(left, right) == ValueType::Float)
        {
          return float_value(to_float64(left) - to_float64(right));
        }
        return int_value(to_int64(left) - to_int64(right));
      });
    }

    static Value mul_values(const Value & lhs, const Value & rhs)
    {
      return map_binary(lhs, rhs, [](const Value & left, const Value & right) {
        if (arithmetic_type(left, right) == ValueType::Float)
        {
          return float_value(to_float64(left) * to_float64(right));
        }
        return int_value(to_int64(left) * to_int64(right));
      });
    }

    static Value div_values(const Value & lhs, const Value & rhs)
    {
      return map_binary(lhs, rhs, [](const Value & left, const Value & right) {
        const double denominator = to_float64(right);
        return denominator == 0.0 ? float_value(0.0) : float_value(to_float64(left) / denominator);
      });
    }

    static void exec_add(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = add_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_add_const(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = add_values(registers[instr.src_a], instr.literal);
    }

    static void exec_sub(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = sub_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_sub_const_rhs(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = sub_values(registers[instr.src_a], instr.literal);
    }

    static void exec_sub_const_lhs(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = sub_values(instr.literal, registers[instr.src_a]);
    }

    static void exec_mul(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = mul_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_mul_const(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = mul_values(registers[instr.src_a], instr.literal);
    }

    static void exec_div(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = div_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_div_const_lhs(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = div_values(instr.literal, registers[instr.src_a]);
    }

    static void exec_neg(const Graph &, const ExprInstr & instr, Value * registers)
    {
      if (registers[instr.src_a].type == ValueType::Float)
      {
        registers[instr.dst] = float_value(-to_float64(registers[instr.src_a]));
      }
      else
      {
        registers[instr.dst] = int_value(-to_int64(registers[instr.src_a]));
      }
    }

    static double fast_sin(double x)
    {
      constexpr double pi = 3.14159265358979323846;
      constexpr double two_pi = 2.0 * pi;
      constexpr double half_pi = 0.5 * pi;
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

    static void exec_sin(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = float_value(fast_sin(to_float64(registers[instr.src_a])));
    }

    static void exec_mod(const Graph &, const ExprInstr & instr, Value * registers)
    {
      if (arithmetic_type(registers[instr.src_a], registers[instr.src_b]) == ValueType::Float)
      {
        const double denominator = to_float64(registers[instr.src_b]);
        registers[instr.dst] = denominator == 0.0 ? float_value(0.0)
                                                  : float_value(std::fmod(to_float64(registers[instr.src_a]), denominator));
        return;
      }

      const int64_t denominator = to_int64(registers[instr.src_b]);
      registers[instr.dst] = denominator == 0 ? int_value(0) : int_value(to_int64(registers[instr.src_a]) % denominator);
    }

    static void exec_floor_div(const Graph &, const ExprInstr & instr, Value * registers)
    {
      const double denominator = to_float64(registers[instr.src_b]);
      registers[instr.dst] = denominator == 0.0 ? float_value(0.0)
                                                : float_value(std::floor(to_float64(registers[instr.src_a]) / denominator));
    }

    static void exec_bit_and(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = int_value(to_int64(registers[instr.src_a]) & to_int64(registers[instr.src_b]));
    }

    static void exec_bit_or(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = int_value(to_int64(registers[instr.src_a]) | to_int64(registers[instr.src_b]));
    }

    static void exec_bit_xor(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = int_value(to_int64(registers[instr.src_a]) ^ to_int64(registers[instr.src_b]));
    }

    static void exec_lshift(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = int_value(to_int64(registers[instr.src_a]) << normalize_shift(registers[instr.src_b]));
    }

    static void exec_rshift(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = int_value(to_int64(registers[instr.src_a]) >> normalize_shift(registers[instr.src_b]));
    }

    static void exec_not(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = bool_value(!is_truthy(registers[instr.src_a]));
    }

    static void exec_bit_not(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = int_value(~to_int64(registers[instr.src_a]));
    }

    static void exec_less(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = bool_value(to_float64(registers[instr.src_a]) < to_float64(registers[instr.src_b]));
    }

    static void exec_less_equal(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = bool_value(to_float64(registers[instr.src_a]) <= to_float64(registers[instr.src_b]));
    }

    static void exec_greater(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = bool_value(to_float64(registers[instr.src_a]) > to_float64(registers[instr.src_b]));
    }

    static void exec_greater_equal(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = bool_value(to_float64(registers[instr.src_a]) >= to_float64(registers[instr.src_b]));
    }

    static void exec_equal(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = bool_value(to_float64(registers[instr.src_a]) == to_float64(registers[instr.src_b]));
    }

    static void exec_not_equal(const Graph &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = bool_value(to_float64(registers[instr.src_a]) != to_float64(registers[instr.src_b]));
    }

    static int normalize_shift(const Value & value)
    {
      int64_t shift = to_int64(value);
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

    static ExprKernel kernel_for_kind(ExprKind kind)
    {
      switch (kind)
      {
        case ExprKind::Literal:
          return &Graph::exec_literal;
        case ExprKind::Ref:
          return &Graph::exec_ref;
        case ExprKind::InputValue:
        case ExprKind::RegisterValue:
        case ExprKind::SampleRate:
        case ExprKind::SampleIndex:
        case ExprKind::ArrayPack:
        case ExprKind::Index:
          return &Graph::exec_literal;
        case ExprKind::Not:
          return &Graph::exec_not;
        case ExprKind::Less:
          return &Graph::exec_less;
        case ExprKind::LessEqual:
          return &Graph::exec_less_equal;
        case ExprKind::Greater:
          return &Graph::exec_greater;
        case ExprKind::GreaterEqual:
          return &Graph::exec_greater_equal;
        case ExprKind::Equal:
          return &Graph::exec_equal;
        case ExprKind::NotEqual:
          return &Graph::exec_not_equal;
        case ExprKind::Add:
          return &Graph::exec_add;
        case ExprKind::Sub:
          return &Graph::exec_sub;
        case ExprKind::Mul:
          return &Graph::exec_mul;
        case ExprKind::Div:
          return &Graph::exec_div;
        case ExprKind::Mod:
          return &Graph::exec_mod;
        case ExprKind::FloorDiv:
          return &Graph::exec_floor_div;
        case ExprKind::BitAnd:
          return &Graph::exec_bit_and;
        case ExprKind::BitOr:
          return &Graph::exec_bit_or;
        case ExprKind::BitXor:
          return &Graph::exec_bit_xor;
        case ExprKind::LShift:
          return &Graph::exec_lshift;
        case ExprKind::RShift:
          return &Graph::exec_rshift;
        case ExprKind::Sin:
          return &Graph::exec_sin;
        case ExprKind::Neg:
          return &Graph::exec_neg;
        case ExprKind::BitNot:
          return &Graph::exec_bit_not;
      }

      return &Graph::exec_literal;
    }

    uint32_t compile_expr_node(
      const ExprSpecPtr & expr,
      CompiledExpr & compiled,
      std::vector<uint8_t> & dependency_marks) const
    {
      if (!expr)
      {
        ExprInstr instr;
        instr.kind = ExprKind::Literal;
        instr.dst = compiled.register_count++;
        instr.literal = float_value(0.0);
        instr.kernel = kernel_for_kind(instr.kind);
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Literal)
      {
        ExprInstr instr;
        instr.kind = ExprKind::Literal;
        instr.dst = compiled.register_count++;
        instr.literal = expr->literal;
        instr.kernel = kernel_for_kind(instr.kind);
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Ref)
      {
        auto it = name_to_id_.find(expr->module_name);
        if (it == name_to_id_.end())
        {
          ExprInstr instr;
          instr.kind = ExprKind::Literal;
          instr.dst = compiled.register_count++;
          instr.literal = float_value(0.0);
          instr.kernel = kernel_for_kind(instr.kind);
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        ExprInstr instr;
        instr.kind = ExprKind::Ref;
        instr.dst = compiled.register_count++;
        instr.ref_module_id = it->second;
        instr.ref_output_id = expr->output_id;
        instr.kernel = kernel_for_kind(instr.kind);
        compiled.instructions.push_back(instr);

        if (it->second < dependency_marks.size() && !dependency_marks[it->second])
        {
          dependency_marks[it->second] = 1;
          compiled.dependencies.push_back(it->second);
        }

        return instr.dst;
      }

      if (expr->kind == ExprKind::Neg || expr->kind == ExprKind::Not || expr->kind == ExprKind::BitNot || expr->kind == ExprKind::Sin)
      {
        const uint32_t operand = compile_expr_node(expr->lhs, compiled, dependency_marks);
        ExprInstr instr;
        instr.kind = expr->kind;
        instr.dst = compiled.register_count++;
        instr.src_a = operand;
        instr.kernel = kernel_for_kind(instr.kind);
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Add)
      {
        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Add;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          instr.kernel = &Graph::exec_add_const;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Add;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = expr->rhs->literal;
          instr.kernel = &Graph::exec_add_const;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      if (expr->kind == ExprKind::Mul)
      {
        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Mul;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          instr.kernel = &Graph::exec_mul_const;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Mul;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = expr->rhs->literal;
          instr.kernel = &Graph::exec_mul_const;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      if (expr->kind == ExprKind::Sub)
      {
        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Sub;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = expr->rhs->literal;
          instr.kernel = &Graph::exec_sub_const_rhs;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Sub;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          instr.kernel = &Graph::exec_sub_const_lhs;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      if (expr->kind == ExprKind::Div)
      {
        if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
        {
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Mul;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = to_float64(expr->rhs->literal) == 0.0
                            ? float_value(0.0)
                            : float_value(1.0 / to_float64(expr->rhs->literal));
          instr.kernel = &Graph::exec_mul_const;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Div;
          instr.dst = compiled.register_count++;
          instr.src_a = rhs;
          instr.literal = expr->lhs->literal;
          instr.kernel = &Graph::exec_div_const_lhs;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }

      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, dependency_marks);
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, dependency_marks);

      ExprInstr instr;
      instr.kind = expr->kind;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.src_b = rhs;
      instr.kernel = kernel_for_kind(instr.kind);
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    CompiledExpr compile_expr(const ExprSpecPtr & expr) const
    {
      CompiledExpr compiled;
      std::vector<uint8_t> dependency_marks(modules_.size(), 0);
      compiled.result_register = compile_expr_node(expr, compiled, dependency_marks);
      return compiled;
    }

    void apply_set_input_expr(const std::string & module_name, unsigned int input_id, const ExprSpecPtr & expr)
    {
      auto it = name_to_id_.find(module_name);
      if (it == name_to_id_.end())
      {
        return;
      }

      auto & slot = modules_[it->second];
      if (!slot.active || input_id >= slot.input_exprs.size())
      {
        return;
      }

      slot.input_exprs[input_id] = compile_expr(expr);
      slot.input_registers[input_id].assign(slot.input_exprs[input_id].register_count, float_value(0.0));
    }

    void apply_add_output(const std::string & module_name, unsigned int output_id)
    {
      auto it = name_to_id_.find(module_name);
      if (it == name_to_id_.end())
      {
        return;
      }
      mix_.push_back(MixTap{it->second, output_id});
    }

    double eval_expr(const CompiledExpr & expr, std::vector<Value> & registers) const
    {
      if (expr.instructions.empty())
      {
        return 0.0;
      }

      if (registers.size() < expr.register_count)
      {
        registers.resize(expr.register_count, float_value(0.0));
      }

      for (const auto & instr : expr.instructions)
      {
        instr.kernel(*this, instr, registers.data());
      }

      return to_float64(registers[expr.result_register]);
    }

    void rebuild_execution_order()
    {
      const std::size_t count = modules_.size();
      std::vector<unsigned int> indegree(count, 0);
      std::vector<std::vector<uint32_t>> adjacency(count);
      std::vector<uint32_t> active_ids;
      active_ids.reserve(count);

      for (uint32_t id = 0; id < modules_.size(); ++id)
      {
        if (!modules_[id].active)
        {
          continue;
        }

        active_ids.push_back(id);
        for (const auto & expr : modules_[id].input_exprs)
        {
          for (uint32_t src_id : expr.dependencies)
          {
            if (src_id >= modules_.size() || !modules_[src_id].active || src_id == id)
            {
              continue;
            }
            adjacency[src_id].push_back(id);
            ++indegree[id];
          }
        }
      }

      std::deque<uint32_t> ready;
      for (uint32_t id : active_ids)
      {
        if (indegree[id] == 0)
        {
          ready.push_back(id);
        }
      }

      std::vector<uint32_t> ordered;
      ordered.reserve(active_ids.size());
      while (!ready.empty())
      {
        const uint32_t id = ready.front();
        ready.pop_front();
        ordered.push_back(id);

        for (uint32_t dst_id : adjacency[id])
        {
          if (--indegree[dst_id] == 0)
          {
            ready.push_back(dst_id);
          }
        }
      }

      if (ordered.size() == active_ids.size())
      {
        execution_order_ = std::move(ordered);
        return;
      }

      std::vector<uint8_t> seen(count, 0);
      std::vector<uint32_t> fallback;
      fallback.reserve(active_ids.size());
      for (uint32_t id : execution_order_)
      {
        if (id < modules_.size() && modules_[id].active && !seen[id])
        {
          fallback.push_back(id);
          seen[id] = 1;
        }
      }
      for (uint32_t id : active_ids)
      {
        if (!seen[id])
        {
          fallback.push_back(id);
        }
      }
      execution_order_ = std::move(fallback);
    }

    unsigned int bufferLength_ = 0;

    std::vector<ModuleSlot> modules_;
    std::unordered_map<std::string, uint32_t> name_to_id_;
    std::vector<uint32_t> execution_order_;
    std::vector<MixTap> mix_;
    std::vector<uint32_t> free_ids_;

    std::unordered_map<std::string, ModuleShape> control_modules_;
    std::unordered_map<std::string, std::vector<ExprSpecPtr>> control_input_exprs_;
    std::vector<outputID> control_mix_;

    mutable std::mutex pending_mutex_;
    std::vector<Command> command_queue_;
};

class UserDefinedModule : public Module
{
  public:
    UserDefinedModule(
      unsigned int input_count,
      std::vector<Graph::ExprSpecPtr> output_exprs,
      std::vector<Graph::ExprSpecPtr> register_exprs,
      std::vector<Graph::Value> initial_registers,
      double sample_rate)
      : Module(input_count, static_cast<unsigned int>(output_exprs.size())),
        input_count_(input_count),
        registers_(std::move(initial_registers)),
        next_registers_(registers_),
        sample_rate_(sample_rate)
    {
      program_ = compile_program(output_exprs, register_exprs);
      temps_.assign(program_.register_count, Graph::float_value(0.0));
    }

    void process() override
    {
      eval_program(program_, temps_);

      for (unsigned int output_id = 0; output_id < program_.output_targets.size(); ++output_id)
      {
        outputs[output_id] = Graph::to_float64(temps_[program_.output_targets[output_id]]);
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
      Module::postprocess();
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

  private:
    struct Instr
    {
      Graph::ExprKind kind = Graph::ExprKind::Literal;
      uint32_t dst = 0;
      uint32_t src_a = 0;
      uint32_t src_b = 0;
      unsigned int slot_id = 0;
      Graph::Value literal;
      std::vector<uint32_t> args;
    };

    struct CompiledProgram
    {
      std::vector<Instr> instructions;
      std::vector<uint32_t> output_targets;
      std::vector<int32_t> register_targets;
      uint32_t register_count = 0;
    };

    static bool is_local_unary(Graph::ExprKind kind)
    {
      return kind == Graph::ExprKind::Neg ||
             kind == Graph::ExprKind::Not ||
             kind == Graph::ExprKind::BitNot ||
             kind == Graph::ExprKind::Sin;
    }

    static bool is_local_binary(Graph::ExprKind kind)
    {
      switch (kind)
      {
        case Graph::ExprKind::Less:
        case Graph::ExprKind::LessEqual:
        case Graph::ExprKind::Greater:
        case Graph::ExprKind::GreaterEqual:
        case Graph::ExprKind::Equal:
        case Graph::ExprKind::NotEqual:
        case Graph::ExprKind::Add:
        case Graph::ExprKind::Sub:
        case Graph::ExprKind::Mul:
        case Graph::ExprKind::Div:
        case Graph::ExprKind::Mod:
        case Graph::ExprKind::FloorDiv:
        case Graph::ExprKind::BitAnd:
        case Graph::ExprKind::BitOr:
        case Graph::ExprKind::BitXor:
        case Graph::ExprKind::LShift:
        case Graph::ExprKind::RShift:
        case Graph::ExprKind::Index:
          return true;
        default:
          return false;
      }
    }

    CompiledProgram compile_program(
      const std::vector<Graph::ExprSpecPtr> & output_exprs,
      const std::vector<Graph::ExprSpecPtr> & register_exprs)
    {
      CompiledProgram compiled;
      compiled.output_targets.reserve(output_exprs.size());
      compiled.register_targets.assign(register_exprs.size(), -1);

      std::unordered_map<const Graph::ExprSpec *, uint32_t> memo;
      for (const auto & expr : output_exprs)
      {
        compiled.output_targets.push_back(compile_expr_node(expr, compiled, memo));
      }
      for (unsigned int i = 0; i < register_exprs.size(); ++i)
      {
        if (register_exprs[i])
        {
          compiled.register_targets[i] = static_cast<int32_t>(compile_expr_node(register_exprs[i], compiled, memo));
        }
      }
      return compiled;
    }

    uint32_t compile_expr_node(
      const Graph::ExprSpecPtr & expr,
      CompiledProgram & compiled,
      std::unordered_map<const Graph::ExprSpec *, uint32_t> & memo)
    {
      if (!expr)
      {
        static const Graph::ExprSpec zero_expr = [] {
          Graph::ExprSpec expr;
          expr.kind = Graph::ExprKind::Literal;
          expr.literal = Graph::float_value(0.0);
          return expr;
        }();
        auto it = memo.find(&zero_expr);
        if (it != memo.end())
        {
          return it->second;
        }

        Instr instr;
        instr.kind = Graph::ExprKind::Literal;
        instr.dst = compiled.register_count++;
        instr.literal = Graph::float_value(0.0);
        compiled.instructions.push_back(std::move(instr));
        memo.emplace(&zero_expr, compiled.instructions.back().dst);
        return compiled.instructions.back().dst;
      }

      auto memo_it = memo.find(expr.get());
      if (memo_it != memo.end())
      {
        return memo_it->second;
      }

      Instr instr;
      instr.kind = expr->kind;
      instr.dst = compiled.register_count++;

      switch (expr->kind)
      {
        case Graph::ExprKind::Literal:
          instr.literal = expr->literal;
          break;
        case Graph::ExprKind::InputValue:
        case Graph::ExprKind::RegisterValue:
          instr.slot_id = expr->slot_id;
          break;
        case Graph::ExprKind::SampleRate:
        case Graph::ExprKind::SampleIndex:
          break;
        case Graph::ExprKind::ArrayPack:
          instr.args.reserve(expr->args.size());
          for (const auto & arg : expr->args)
          {
            instr.args.push_back(compile_expr_node(arg, compiled, memo));
          }
          break;
        default:
          if (is_local_unary(expr->kind))
          {
            instr.src_a = compile_expr_node(expr->lhs, compiled, memo);
          }
          else if (is_local_binary(expr->kind))
          {
            instr.src_a = compile_expr_node(expr->lhs, compiled, memo);
            instr.src_b = compile_expr_node(expr->rhs, compiled, memo);
          }
          else
          {
            throw std::invalid_argument("Unsupported user-defined module expression node.");
          }
          break;
      }

      compiled.instructions.push_back(std::move(instr));
      memo.emplace(expr.get(), compiled.instructions.back().dst);
      return compiled.instructions.back().dst;
    }

    void eval_program(const CompiledProgram & expr, std::vector<Graph::Value> & temps) const
    {
      if (expr.instructions.empty())
      {
        return;
      }

      if (temps.size() < expr.register_count)
      {
        temps.resize(expr.register_count, Graph::float_value(0.0));
      }

      for (const Instr & instr : expr.instructions)
      {
        switch (instr.kind)
        {
          case Graph::ExprKind::Literal:
            temps[instr.dst] = instr.literal;
            break;
          case Graph::ExprKind::InputValue:
            temps[instr.dst] = instr.slot_id < inputs.size()
                                 ? Graph::float_value(inputs[instr.slot_id])
                                 : Graph::float_value(0.0);
            break;
          case Graph::ExprKind::RegisterValue:
            temps[instr.dst] = instr.slot_id < registers_.size()
                                 ? registers_[instr.slot_id]
                                 : Graph::float_value(0.0);
            break;
          case Graph::ExprKind::SampleRate:
            temps[instr.dst] = Graph::float_value(sample_rate_);
            break;
          case Graph::ExprKind::SampleIndex:
            temps[instr.dst] = Graph::int_value(static_cast<int64_t>(sample_index_));
            break;
          case Graph::ExprKind::ArrayPack:
          {
            std::vector<Graph::Value> items;
            items.reserve(instr.args.size());
            for (uint32_t src : instr.args)
            {
              if (Graph::is_array(temps[src]))
              {
                throw std::invalid_argument("Nested arrays are not supported.");
              }
              items.push_back(temps[src]);
            }
            temps[instr.dst] = Graph::array_value(std::move(items));
            break;
          }
          case Graph::ExprKind::Index:
          {
            const Graph::Value & array_value = temps[instr.src_a];
            if (!Graph::is_array(array_value))
            {
              throw std::invalid_argument("Indexing requires an array value.");
            }
            const int64_t index = Graph::to_int64(temps[instr.src_b]);
            if (index < 0 || static_cast<std::size_t>(index) >= array_value.array_items.size())
            {
              throw std::out_of_range("Array index out of range.");
            }
            temps[instr.dst] = array_value.array_items[static_cast<std::size_t>(index)];
            break;
          }
          case Graph::ExprKind::Not:
            temps[instr.dst] = Graph::map_unary(temps[instr.src_a], [](const Graph::Value & value) {
              return Graph::bool_value(!Graph::is_truthy(value));
            });
            break;
          case Graph::ExprKind::Less:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::bool_value(Graph::to_float64(lhs) < Graph::to_float64(rhs));
            });
            break;
          case Graph::ExprKind::LessEqual:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::bool_value(Graph::to_float64(lhs) <= Graph::to_float64(rhs));
            });
            break;
          case Graph::ExprKind::Greater:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::bool_value(Graph::to_float64(lhs) > Graph::to_float64(rhs));
            });
            break;
          case Graph::ExprKind::GreaterEqual:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::bool_value(Graph::to_float64(lhs) >= Graph::to_float64(rhs));
            });
            break;
          case Graph::ExprKind::Equal:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::bool_value(Graph::to_float64(lhs) == Graph::to_float64(rhs));
            });
            break;
          case Graph::ExprKind::NotEqual:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::bool_value(Graph::to_float64(lhs) != Graph::to_float64(rhs));
            });
            break;
          case Graph::ExprKind::Add:
            temps[instr.dst] = Graph::add_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case Graph::ExprKind::Sub:
            temps[instr.dst] = Graph::sub_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case Graph::ExprKind::Mul:
            temps[instr.dst] = Graph::mul_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case Graph::ExprKind::Div:
            temps[instr.dst] = Graph::div_values(temps[instr.src_a], temps[instr.src_b]);
            break;
          case Graph::ExprKind::Mod:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              if (Graph::arithmetic_type(lhs, rhs) == Graph::ValueType::Float)
              {
                const double denom = Graph::to_float64(rhs);
                return denom == 0.0 ? Graph::float_value(0.0) : Graph::float_value(std::fmod(Graph::to_float64(lhs), denom));
              }
              const int64_t denom = Graph::to_int64(rhs);
              return denom == 0 ? Graph::int_value(0) : Graph::int_value(Graph::to_int64(lhs) % denom);
            });
            break;
          case Graph::ExprKind::FloorDiv:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              const double denom = Graph::to_float64(rhs);
              return denom == 0.0 ? Graph::float_value(0.0) : Graph::float_value(std::floor(Graph::to_float64(lhs) / denom));
            });
            break;
          case Graph::ExprKind::BitAnd:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::int_value(Graph::to_int64(lhs) & Graph::to_int64(rhs));
            });
            break;
          case Graph::ExprKind::BitOr:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::int_value(Graph::to_int64(lhs) | Graph::to_int64(rhs));
            });
            break;
          case Graph::ExprKind::BitXor:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::int_value(Graph::to_int64(lhs) ^ Graph::to_int64(rhs));
            });
            break;
          case Graph::ExprKind::LShift:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::int_value(Graph::to_int64(lhs) << Graph::normalize_shift(rhs));
            });
            break;
          case Graph::ExprKind::RShift:
            temps[instr.dst] = Graph::map_binary(temps[instr.src_a], temps[instr.src_b], [](const Graph::Value & lhs, const Graph::Value & rhs) {
              return Graph::int_value(Graph::to_int64(lhs) >> Graph::normalize_shift(rhs));
            });
            break;
          case Graph::ExprKind::Sin:
            temps[instr.dst] = Graph::map_unary(temps[instr.src_a], [](const Graph::Value & value) {
              return Graph::float_value(std::sin(Graph::to_float64(value)));
            });
            break;
          case Graph::ExprKind::Neg:
            temps[instr.dst] = Graph::map_unary(temps[instr.src_a], [](const Graph::Value & value) {
              if (value.type == Graph::ValueType::Float)
              {
                return Graph::float_value(-Graph::to_float64(value));
              }
              return Graph::int_value(-Graph::to_int64(value));
            });
            break;
          case Graph::ExprKind::BitNot:
            temps[instr.dst] = Graph::map_unary(temps[instr.src_a], [](const Graph::Value & value) {
              return Graph::int_value(~Graph::to_int64(value));
            });
            break;
          case Graph::ExprKind::Ref:
            temps[instr.dst] = Graph::float_value(0.0);
            break;
        }
      }
    }

    unsigned int input_count_ = 0;
    CompiledProgram program_;
    std::vector<Graph::Value> temps_;
    std::vector<Graph::Value> registers_;
    std::vector<Graph::Value> next_registers_;
    double sample_rate_ = 44100.0;
    uint64_t sample_index_ = 0;
};
