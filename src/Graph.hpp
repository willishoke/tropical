#include "Expr.hpp"
#include "ExprEval.hpp"
#include "Module.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
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

class Graph
{
  public:
    explicit Graph(unsigned int bufferLength)
      : bufferLength_(bufferLength), outputBuffer(bufferLength, 0.0)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      runtimes_[0] = build_runtime_locked();
      runtimes_[1] = build_runtime_locked();
    }

    void process()
    {
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

          for (unsigned int input_id = 0; input_id < slot.input_exprs.size(); ++input_id)
          {
            slot.module->inputs[input_id] = eval_expr(runtime, slot.input_exprs[input_id], slot.input_registers[input_id]);
          }

          slot.module->process();
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

          mixed += slot.module->outputs[tap.output_id] / 20.0;
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

      if (!validate_expr_refs(expr))
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
    friend class UserDefinedModule;

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

    struct ExprInstr;
    struct RuntimeState;
    using ExprKernel = void (*)(const RuntimeState &, const ExprInstr &, Value *);

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
      uint32_t register_count = 0;
      uint32_t result_register = 0;
    };

    struct ModuleSlot
    {
      std::string name;
      std::shared_ptr<Module> module;
      std::vector<CompiledExpr> input_exprs;
      std::vector<std::vector<Value>> input_registers;
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
        slot.input_exprs.resize(module.in_count);
        slot.input_registers.resize(module.in_count);
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
        const unsigned int input_limit = std::min(
          static_cast<unsigned int>(slot.input_exprs.size()),
          static_cast<unsigned int>(module.input_exprs.size()));

        for (unsigned int input_id = 0; input_id < input_limit; ++input_id)
        {
          slot.input_exprs[input_id] = compile_expr(module.input_exprs[input_id], runtime);
          slot.input_registers[input_id].assign(slot.input_exprs[input_id].register_count, float_value(0.0));
        }
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

    static void exec_literal(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = instr.literal;
    }

    static void exec_ref(const RuntimeState & runtime, const ExprInstr & instr, Value * registers)
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        return;
      }

      const auto & slot = runtime.modules[instr.ref_module_id];
      if (!slot.module || instr.ref_output_id >= slot.module->prev_outputs.size())
      {
        registers[instr.dst] = float_value(0.0);
        return;
      }

      registers[instr.dst] = float_value(slot.module->prev_outputs[instr.ref_output_id]);
    }

    static void exec_add(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_add_const(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], instr.literal);
    }

    static void exec_sub(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_sub_const_rhs(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], instr.literal);
    }

    static void exec_sub_const_lhs(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::sub_values(instr.literal, registers[instr.src_a]);
    }

    static void exec_mul(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_mul_const(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], instr.literal);
    }

    static void exec_div(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::div_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_div_const_lhs(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::div_values(instr.literal, registers[instr.src_a]);
    }

    static void exec_neg(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::neg_value(registers[instr.src_a]);
    }

    static void exec_sin(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::sin_value(registers[instr.src_a]);
    }

    static void exec_mod(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::mod_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_floor_div(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::floor_div_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_bit_and(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::bit_and_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_bit_or(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::bit_or_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_bit_xor(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::bit_xor_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_lshift(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::lshift_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_rshift(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::rshift_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_not(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::not_value(registers[instr.src_a]);
    }

    static void exec_bit_not(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::bit_not_value(registers[instr.src_a]);
    }

    static void exec_less(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::less_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_less_equal(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::less_equal_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_greater(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::greater_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_greater_equal(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::greater_equal_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_equal(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::equal_values(registers[instr.src_a], registers[instr.src_b]);
    }

    static void exec_not_equal(const RuntimeState &, const ExprInstr & instr, Value * registers)
    {
      registers[instr.dst] = expr_eval::not_equal_values(registers[instr.src_a], registers[instr.src_b]);
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
      const RuntimeState & runtime) const
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
        auto it = runtime.name_to_id.find(expr->module_name);
        if (it == runtime.name_to_id.end())
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

        return instr.dst;
      }

      if (expr->kind == ExprKind::Neg || expr->kind == ExprKind::Not || expr->kind == ExprKind::BitNot || expr->kind == ExprKind::Sin)
      {
        const uint32_t operand = compile_expr_node(expr->lhs, compiled, runtime);
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
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
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
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
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
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
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
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
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
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
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
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
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
          const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
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
          const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
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

      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);

      ExprInstr instr;
      instr.kind = expr->kind;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.src_b = rhs;
      instr.kernel = kernel_for_kind(instr.kind);
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    CompiledExpr compile_expr(const ExprSpecPtr & expr, const RuntimeState & runtime) const
    {
      CompiledExpr compiled;
      compiled.result_register = compile_expr_node(expr, compiled, runtime);
      return compiled;
    }

    double eval_expr(const RuntimeState & runtime, const CompiledExpr & expr, std::vector<Value> & registers) const
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
        instr.kernel(runtime, instr, registers.data());
      }

      return to_float64(registers[expr.result_register]);
    }

    unsigned int bufferLength_ = 0;

    std::array<RuntimeState, 2> runtimes_;
    std::atomic<uint32_t> active_runtime_index_{0};
    std::atomic<uint32_t> audio_runtime_index_{0};
    std::atomic<bool> audio_processing_{false};

    std::unordered_map<std::string, ControlModule> control_modules_;
    std::vector<outputID> control_mix_;

    mutable std::mutex pending_mutex_;
};

class UserDefinedModule : public Module
{
  public:
    UserDefinedModule(
      unsigned int input_count,
      std::vector<ExprSpecPtr> output_exprs,
      std::vector<ExprSpecPtr> register_exprs,
      std::vector<Value> initial_registers,
      double sample_rate)
      : Module(input_count, static_cast<unsigned int>(output_exprs.size())),
        input_count_(input_count),
        registers_(std::move(initial_registers)),
        next_registers_(registers_),
        sample_rate_(sample_rate)
    {
      program_ = compile_program(output_exprs, register_exprs);
      temps_.assign(program_.register_count, expr::float_value(0.0));
    }

    void process() override
    {
      eval_program(program_, temps_);

      for (unsigned int output_id = 0; output_id < program_.output_targets.size(); ++output_id)
      {
        outputs[output_id] = expr::to_float64(temps_[program_.output_targets[output_id]]);
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
      ExprKind kind = ExprKind::Literal;
      uint32_t dst = 0;
      uint32_t src_a = 0;
      uint32_t src_b = 0;
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
      return kind == ExprKind::Neg ||
             kind == ExprKind::Not ||
             kind == ExprKind::BitNot ||
             kind == ExprKind::Sin;
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

      std::unordered_map<const ExprSpec *, uint32_t> memo;
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
      const ExprSpecPtr & expr,
      CompiledProgram & compiled,
      std::unordered_map<const ExprSpec *, uint32_t> & memo)
    {
      if (!expr)
      {
        static const ExprSpec zero_expr = [] {
          ExprSpec expr;
          expr.kind = ExprKind::Literal;
          expr.literal = expr::float_value(0.0);
          return expr;
        }();
        auto it = memo.find(&zero_expr);
        if (it != memo.end())
        {
          return it->second;
        }

        Instr instr;
        instr.kind = ExprKind::Literal;
        instr.dst = compiled.register_count++;
        instr.literal = expr::float_value(0.0);
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
          case ExprKind::Literal:
            temps[instr.dst] = instr.literal;
            break;
          case ExprKind::InputValue:
            temps[instr.dst] = instr.slot_id < inputs.size()
                                 ? expr::float_value(inputs[instr.slot_id])
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
              throw std::invalid_argument("Indexing requires an array value.");
            }
            const int64_t index = expr::to_int64(temps[instr.src_b]);
            if (index < 0 || static_cast<std::size_t>(index) >= array_value.array_items.size())
            {
              throw std::out_of_range("Array index out of range.");
            }
            temps[instr.dst] = array_value.array_items[static_cast<std::size_t>(index)];
            break;
          }
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
    double sample_rate_ = 44100.0;
    uint64_t sample_index_ = 0;
};
