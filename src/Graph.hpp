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
    enum class ExprKind
    {
      Literal,
      Ref,
      InputValue,
      RegisterValue,
      SampleRate,
      SampleIndex,
      Less,
      LessEqual,
      Greater,
      GreaterEqual,
      Equal,
      NotEqual,
      Add,
      Sub,
      Mul,
      Div,
      Mod,
      FloorDiv,
      BitAnd,
      BitOr,
      BitXor,
      LShift,
      RShift,
      Sin,
      Neg,
      BitNot
    };

    struct ExprSpec
    {
      ExprKind kind = ExprKind::Literal;
      double literal = 0.0;
      std::string module_name;
      unsigned int output_id = 0;
      unsigned int slot_id = 0;
      std::shared_ptr<ExprSpec> lhs;
      std::shared_ptr<ExprSpec> rhs;
    };

    using ExprSpecPtr = std::shared_ptr<ExprSpec>;

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

      ExprSpecPtr ref = make_ref_expr(src_module, src_output_id);
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

    static ExprSpecPtr literal_expr(double value)
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = ExprKind::Literal;
      expr->literal = value;
      return expr;
    }

    static ExprSpecPtr ref_expr(std::string module_name, unsigned int output_id)
    {
      return make_ref_expr(std::move(module_name), output_id);
    }

    static ExprSpecPtr input_value_expr(unsigned int input_id)
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = ExprKind::InputValue;
      expr->slot_id = input_id;
      return expr;
    }

    static ExprSpecPtr register_value_expr(unsigned int register_id)
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = ExprKind::RegisterValue;
      expr->slot_id = register_id;
      return expr;
    }

    static ExprSpecPtr sample_rate_expr()
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = ExprKind::SampleRate;
      return expr;
    }

    static ExprSpecPtr sample_index_expr()
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = ExprKind::SampleIndex;
      return expr;
    }

    static ExprSpecPtr unary_expr(ExprKind kind, ExprSpecPtr operand)
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = kind;
      expr->lhs = std::move(operand);
      return expr;
    }

    static ExprSpecPtr binary_expr(ExprKind kind, ExprSpecPtr lhs, ExprSpecPtr rhs)
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = kind;
      expr->lhs = std::move(lhs);
      expr->rhs = std::move(rhs);
      return expr;
    }

    std::vector<double> outputBuffer;

  private:
    struct ModuleShape
    {
      unsigned int in_count;
      unsigned int out_count;
    };

    enum class ValueType : uint8_t
    {
      Scalar
    };

    struct ExprInstr;
    using ExprKernel = void (*)(const Graph &, const ExprInstr &, double *);

    struct ExprInstr
    {
      ExprKind kind = ExprKind::Literal;
      ValueType value_type = ValueType::Scalar;
      uint32_t dst = 0;
      uint32_t src_a = 0;
      uint32_t src_b = 0;
      double literal = 0.0;
      uint32_t ref_module_id = 0;
      unsigned int ref_output_id = 0;
      ExprKernel kernel = nullptr;
    };

    struct CompiledExpr
    {
      ValueType value_type = ValueType::Scalar;
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
      std::vector<std::vector<double>> input_registers;
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

    static ExprSpecPtr make_ref_expr(std::string module_name, unsigned int output_id)
    {
      auto expr = std::make_shared<ExprSpec>();
      expr->kind = ExprKind::Ref;
      expr->module_name = std::move(module_name);
      expr->output_id = output_id;
      return expr;
    }

    static bool is_zero_expr(const ExprSpecPtr & expr)
    {
      return expr != nullptr && expr->kind == ExprKind::Literal && expr->literal == 0.0;
    }

    static bool is_one_expr(const ExprSpecPtr & expr)
    {
      return expr != nullptr && expr->kind == ExprKind::Literal && expr->literal == 1.0;
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
          return expr;
        case ExprKind::Neg:
        {
          ExprSpecPtr lhs = simplify_expr(expr->lhs);
          if (!lhs || is_zero_expr(lhs))
          {
            return nullptr;
          }
          if (lhs->kind == ExprKind::Literal)
          {
            return literal_expr(-lhs->literal);
          }
          return unary_expr(ExprKind::Neg, lhs);
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
            return literal_expr(static_cast<double>(~to_int64(lhs->literal)));
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
            return literal_expr(std::sin(lhs->literal));
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
            bool result = false;
            switch (expr->kind)
            {
              case ExprKind::Less:
                result = lhs->literal < rhs->literal;
                break;
              case ExprKind::LessEqual:
                result = lhs->literal <= rhs->literal;
                break;
              case ExprKind::Greater:
                result = lhs->literal > rhs->literal;
                break;
              case ExprKind::GreaterEqual:
                result = lhs->literal >= rhs->literal;
                break;
              case ExprKind::Equal:
                result = lhs->literal == rhs->literal;
                break;
              case ExprKind::NotEqual:
                result = lhs->literal != rhs->literal;
                break;
              default:
                break;
            }
            return literal_expr(result ? 1.0 : 0.0);
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
            return literal_expr(lhs->literal + rhs->literal);
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
            return literal_expr(lhs->literal - rhs->literal);
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
            return literal_expr(lhs->literal * rhs->literal);
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
          if (lhs->kind == ExprKind::Literal && rhs->kind == ExprKind::Literal && rhs->literal != 0.0)
          {
            return literal_expr(lhs->literal / rhs->literal);
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
            switch (expr->kind)
            {
              case ExprKind::Mod:
                return literal_expr(rhs->literal == 0.0 ? 0.0 : std::fmod(lhs->literal, rhs->literal));
              case ExprKind::FloorDiv:
                return literal_expr(rhs->literal == 0.0 ? 0.0 : std::floor(lhs->literal / rhs->literal));
              case ExprKind::BitAnd:
                return literal_expr(static_cast<double>(to_int64(lhs->literal) & to_int64(rhs->literal)));
              case ExprKind::BitOr:
                return literal_expr(static_cast<double>(to_int64(lhs->literal) | to_int64(rhs->literal)));
              case ExprKind::BitXor:
                return literal_expr(static_cast<double>(to_int64(lhs->literal) ^ to_int64(rhs->literal)));
              case ExprKind::LShift:
                return literal_expr(static_cast<double>(to_int64(lhs->literal) << normalize_shift(rhs->literal)));
              case ExprKind::RShift:
                return literal_expr(static_cast<double>(to_int64(lhs->literal) >> normalize_shift(rhs->literal)));
              default:
                break;
            }
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
        modules_[module_id].input_registers.assign(modules_[module_id].module->inputs.size(), std::vector<double>{});
        modules_[module_id].active = true;
      }
      else
      {
        module_id = static_cast<uint32_t>(modules_.size());
        modules_.push_back(ModuleSlot{
          std::move(module_name),
          std::move(module),
          std::vector<CompiledExpr>(input_count),
          std::vector<std::vector<double>>(input_count),
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

    static void exec_literal(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = instr.literal;
    }

    static void exec_ref(const Graph & graph, const ExprInstr & instr, double * registers)
    {
      if (instr.ref_module_id >= graph.modules_.size() || !graph.modules_[instr.ref_module_id].active)
      {
        registers[instr.dst] = 0.0;
        return;
      }

      registers[instr.dst] = graph.modules_[instr.ref_module_id].module->outputs[instr.ref_output_id];
    }

    static void exec_add(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] + registers[instr.src_b];
    }

    static void exec_add_const(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] + instr.literal;
    }

    static void exec_sub(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] - registers[instr.src_b];
    }

    static void exec_sub_const_rhs(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] - instr.literal;
    }

    static void exec_sub_const_lhs(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = instr.literal - registers[instr.src_a];
    }

    static void exec_mul(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] * registers[instr.src_b];
    }

    static void exec_mul_const(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] * instr.literal;
    }

    static void exec_div(const Graph &, const ExprInstr & instr, double * registers)
    {
      const double denominator = registers[instr.src_b];
      registers[instr.dst] = denominator == 0.0 ? 0.0 : registers[instr.src_a] / denominator;
    }

    static void exec_div_const_lhs(const Graph &, const ExprInstr & instr, double * registers)
    {
      const double denominator = registers[instr.src_a];
      if (denominator == 0.0)
      {
        registers[instr.dst] = instr.literal < 0.0 ? -std::numeric_limits<double>::infinity()
                                                   : std::numeric_limits<double>::infinity();
        return;
      }

      registers[instr.dst] = instr.literal / denominator;
    }

    static void exec_neg(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = -registers[instr.src_a];
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

    static void exec_sin(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = fast_sin(registers[instr.src_a]);
    }

    static void exec_mod(const Graph &, const ExprInstr & instr, double * registers)
    {
      const double denominator = registers[instr.src_b];
      registers[instr.dst] = denominator == 0.0 ? 0.0 : std::fmod(registers[instr.src_a], denominator);
    }

    static void exec_floor_div(const Graph &, const ExprInstr & instr, double * registers)
    {
      const double denominator = registers[instr.src_b];
      registers[instr.dst] = denominator == 0.0 ? 0.0 : std::floor(registers[instr.src_a] / denominator);
    }

    static void exec_bit_and(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = static_cast<double>(to_int64(registers[instr.src_a]) & to_int64(registers[instr.src_b]));
    }

    static void exec_bit_or(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = static_cast<double>(to_int64(registers[instr.src_a]) | to_int64(registers[instr.src_b]));
    }

    static void exec_bit_xor(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = static_cast<double>(to_int64(registers[instr.src_a]) ^ to_int64(registers[instr.src_b]));
    }

    static void exec_lshift(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = static_cast<double>(to_int64(registers[instr.src_a]) << normalize_shift(registers[instr.src_b]));
    }

    static void exec_rshift(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = static_cast<double>(to_int64(registers[instr.src_a]) >> normalize_shift(registers[instr.src_b]));
    }

    static void exec_bit_not(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = static_cast<double>(~to_int64(registers[instr.src_a]));
    }

    static void exec_less(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] < registers[instr.src_b] ? 1.0 : 0.0;
    }

    static void exec_less_equal(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] <= registers[instr.src_b] ? 1.0 : 0.0;
    }

    static void exec_greater(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] > registers[instr.src_b] ? 1.0 : 0.0;
    }

    static void exec_greater_equal(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] >= registers[instr.src_b] ? 1.0 : 0.0;
    }

    static void exec_equal(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] == registers[instr.src_b] ? 1.0 : 0.0;
    }

    static void exec_not_equal(const Graph &, const ExprInstr & instr, double * registers)
    {
      registers[instr.dst] = registers[instr.src_a] != registers[instr.src_b] ? 1.0 : 0.0;
    }

    static int64_t to_int64(double value)
    {
      return static_cast<int64_t>(value);
    }

    static int normalize_shift(double value)
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
          return &Graph::exec_literal;
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
        instr.value_type = ValueType::Scalar;
        instr.dst = compiled.register_count++;
        instr.literal = 0.0;
        instr.kernel = kernel_for_kind(instr.kind);
        compiled.instructions.push_back(instr);
        return instr.dst;
      }

      if (expr->kind == ExprKind::Literal)
      {
        ExprInstr instr;
        instr.kind = ExprKind::Literal;
        instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
          instr.dst = compiled.register_count++;
          instr.literal = 0.0;
          instr.kernel = kernel_for_kind(instr.kind);
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        ExprInstr instr;
        instr.kind = ExprKind::Ref;
        instr.value_type = ValueType::Scalar;
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

      if (expr->kind == ExprKind::Neg || expr->kind == ExprKind::BitNot || expr->kind == ExprKind::Sin)
      {
        const uint32_t operand = compile_expr_node(expr->lhs, compiled, dependency_marks);
        ExprInstr instr;
        instr.kind = expr->kind;
        instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
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
          instr.value_type = ValueType::Scalar;
          instr.dst = compiled.register_count++;
          instr.src_a = lhs;
          instr.literal = expr->rhs->literal == 0.0 ? 0.0 : 1.0 / expr->rhs->literal;
          instr.kernel = &Graph::exec_mul_const;
          compiled.instructions.push_back(instr);
          return instr.dst;
        }

        if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
        {
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, dependency_marks);
          ExprInstr instr;
          instr.kind = ExprKind::Div;
          instr.value_type = ValueType::Scalar;
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
      instr.value_type = ValueType::Scalar;
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
      slot.input_registers[input_id].assign(slot.input_exprs[input_id].register_count, 0.0);
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

    double eval_expr(const CompiledExpr & expr, std::vector<double> & registers) const
    {
      if (expr.instructions.empty())
      {
        return 0.0;
      }

      if (registers.size() < expr.register_count)
      {
        registers.resize(expr.register_count, 0.0);
      }

      for (const auto & instr : expr.instructions)
      {
        instr.kernel(*this, instr, registers.data());
      }

      return registers[expr.result_register];
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
      std::vector<double> initial_registers,
      double sample_rate)
      : Module(input_count, static_cast<unsigned int>(output_exprs.size())),
        input_count_(input_count),
        output_exprs_(std::move(output_exprs)),
        register_exprs_(std::move(register_exprs)),
        registers_(std::move(initial_registers)),
        next_registers_(registers_),
        sample_rate_(sample_rate)
    {
    }

    void process() override
    {
      for (unsigned int output_id = 0; output_id < output_exprs_.size(); ++output_id)
      {
        outputs[output_id] = eval_expr(output_exprs_[output_id]);
      }

      next_registers_ = registers_;
      for (unsigned int register_id = 0; register_id < register_exprs_.size(); ++register_id)
      {
        if (register_exprs_[register_id])
        {
          next_registers_[register_id] = eval_expr(register_exprs_[register_id]);
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
      return static_cast<unsigned int>(output_exprs_.size());
    }

    unsigned int register_count() const
    {
      return static_cast<unsigned int>(registers_.size());
    }

  private:
    double eval_expr(const Graph::ExprSpecPtr & expr) const
    {
      if (!expr)
      {
        return 0.0;
      }

      switch (expr->kind)
      {
        case Graph::ExprKind::Literal:
          return expr->literal;
        case Graph::ExprKind::InputValue:
          return expr->slot_id < inputs.size() ? inputs[expr->slot_id] : 0.0;
        case Graph::ExprKind::RegisterValue:
          return expr->slot_id < registers_.size() ? registers_[expr->slot_id] : 0.0;
        case Graph::ExprKind::SampleRate:
          return sample_rate_;
        case Graph::ExprKind::SampleIndex:
          return static_cast<double>(sample_index_);
        case Graph::ExprKind::Less:
          return eval_expr(expr->lhs) < eval_expr(expr->rhs) ? 1.0 : 0.0;
        case Graph::ExprKind::LessEqual:
          return eval_expr(expr->lhs) <= eval_expr(expr->rhs) ? 1.0 : 0.0;
        case Graph::ExprKind::Greater:
          return eval_expr(expr->lhs) > eval_expr(expr->rhs) ? 1.0 : 0.0;
        case Graph::ExprKind::GreaterEqual:
          return eval_expr(expr->lhs) >= eval_expr(expr->rhs) ? 1.0 : 0.0;
        case Graph::ExprKind::Equal:
          return eval_expr(expr->lhs) == eval_expr(expr->rhs) ? 1.0 : 0.0;
        case Graph::ExprKind::NotEqual:
          return eval_expr(expr->lhs) != eval_expr(expr->rhs) ? 1.0 : 0.0;
        case Graph::ExprKind::Add:
          return eval_expr(expr->lhs) + eval_expr(expr->rhs);
        case Graph::ExprKind::Sub:
          return eval_expr(expr->lhs) - eval_expr(expr->rhs);
        case Graph::ExprKind::Mul:
          return eval_expr(expr->lhs) * eval_expr(expr->rhs);
        case Graph::ExprKind::Div:
        {
          const double rhs = eval_expr(expr->rhs);
          return rhs == 0.0 ? 0.0 : eval_expr(expr->lhs) / rhs;
        }
        case Graph::ExprKind::Mod:
        {
          const double rhs = eval_expr(expr->rhs);
          return rhs == 0.0 ? 0.0 : std::fmod(eval_expr(expr->lhs), rhs);
        }
        case Graph::ExprKind::FloorDiv:
        {
          const double rhs = eval_expr(expr->rhs);
          return rhs == 0.0 ? 0.0 : std::floor(eval_expr(expr->lhs) / rhs);
        }
        case Graph::ExprKind::BitAnd:
          return static_cast<double>(to_int64(eval_expr(expr->lhs)) & to_int64(eval_expr(expr->rhs)));
        case Graph::ExprKind::BitOr:
          return static_cast<double>(to_int64(eval_expr(expr->lhs)) | to_int64(eval_expr(expr->rhs)));
        case Graph::ExprKind::BitXor:
          return static_cast<double>(to_int64(eval_expr(expr->lhs)) ^ to_int64(eval_expr(expr->rhs)));
        case Graph::ExprKind::LShift:
          return static_cast<double>(to_int64(eval_expr(expr->lhs)) << normalize_shift(eval_expr(expr->rhs)));
        case Graph::ExprKind::RShift:
          return static_cast<double>(to_int64(eval_expr(expr->lhs)) >> normalize_shift(eval_expr(expr->rhs)));
        case Graph::ExprKind::Sin:
          return std::sin(eval_expr(expr->lhs));
        case Graph::ExprKind::Neg:
          return -eval_expr(expr->lhs);
        case Graph::ExprKind::BitNot:
          return static_cast<double>(~to_int64(eval_expr(expr->lhs)));
        case Graph::ExprKind::Ref:
          return 0.0;
      }

      return 0.0;
    }

    static int64_t to_int64(double value)
    {
      return static_cast<int64_t>(value);
    }

    static int normalize_shift(double value)
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

    unsigned int input_count_ = 0;
    std::vector<Graph::ExprSpecPtr> output_exprs_;
    std::vector<Graph::ExprSpecPtr> register_exprs_;
    std::vector<double> registers_;
    std::vector<double> next_registers_;
    double sample_rate_ = 44100.0;
    uint64_t sample_index_ = 0;
};
