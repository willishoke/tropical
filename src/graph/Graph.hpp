#include "graph/GraphTypes.hpp"
#include "expr/ExprStructural.hpp"
#include "graph/GraphRuntime.hpp"
#include "graph/Module.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

class Graph;

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
      uint64_t nested_module_count = 0;
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

    ~Graph()
    {
      std::vector<std::thread> threads_to_join;
      {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        stop_worker_pool_locked(threads_to_join);
      }
      for (auto & worker : threads_to_join)
      {
        if (worker.joinable())
        {
          worker.join();
        }
      }
    }

    void set_worker_count(unsigned int worker_count)
    {
      const unsigned int normalized = std::max(1U, worker_count);
      std::lock_guard<std::mutex> pending_lock(pending_mutex_);
      wait_for_runtime_available(0);
      wait_for_runtime_available(1);
      std::vector<std::thread> threads_to_join;
      {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        if (worker_count_ == normalized)
        {
          return;
        }

        stop_worker_pool_locked(threads_to_join);
        worker_count_ = normalized;
        if (worker_count_ > 1)
        {
          start_worker_pool_locked(worker_count_ - 1);
        }
      }
      for (auto & worker : threads_to_join)
      {
        if (worker.joinable())
        {
          worker.join();
        }
      }
    }

    unsigned int worker_count() const
    {
      return worker_count_;
    }

    void process()
    {
#ifdef EGRESS_PROFILE
      const auto callback_start = std::chrono::steady_clock::now();
#endif

      const uint32_t runtime_index = active_runtime_index_.load(std::memory_order_acquire);
      audio_runtime_index_.store(runtime_index, std::memory_order_release);
      audio_processing_.store(true, std::memory_order_release);

      RuntimeState & runtime = runtimes_[runtime_index];
#ifdef EGRESS_PROFILE
      std::vector<ProcessModuleTiming> local_module_stats(runtime.modules.size());
#endif
      std::vector<uint32_t> tap_write_indices;
      tap_write_indices.reserve(runtime.taps.size());
      for (const auto & tap : runtime.taps)
      {
        if (!tap.valid)
        {
          tap_write_indices.push_back(0);
          continue;
        }
        const uint32_t readable = tap.buffer.readable.load(std::memory_order_acquire);
        tap_write_indices.push_back(1U - readable);
      }

      for (unsigned int sample = 0; sample < bufferLength_; ++sample)
      {
        parallel_next_module_index_.store(0, std::memory_order_relaxed);
        if (worker_count_ > 1 && runtime.modules.size() > 1)
        {
#ifdef EGRESS_PROFILE
          start_parallel_module_batch(runtime, &local_module_stats);
#else
          start_parallel_module_batch(runtime);
#endif
        }
        else
        {
#ifdef EGRESS_PROFILE
          execute_parallel_module_work(runtime, &local_module_stats);
#else
          execute_parallel_module_work(runtime);
#endif
        }

        if (!runtime.taps.empty())
        {
          for (std::size_t tap_id = 0; tap_id < runtime.taps.size(); ++tap_id)
          {
            auto & tap = runtime.taps[tap_id];
            if (!tap.valid || tap.module_id >= runtime.modules.size())
            {
              continue;
            }
            const auto & slot = runtime.modules[tap.module_id];
            if (!slot.module || tap.output_id >= slot.module->outputs.size())
            {
              continue;
            }
            const Value & output = slot.module->outputs[tap.output_id];
            if (expr::is_array(output) || expr::is_matrix(output))
            {
              tap.buffer.buffers[tap_write_indices[tap_id]][sample] = 0.0;
            }
            else
            {
              tap.buffer.buffers[tap_write_indices[tap_id]][sample] = expr::to_float64(output);
            }
          }
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
        for (auto & mix_expr : runtime.mix_exprs)
        {
          if (mix_expr.registers.size() < mix_expr.program.register_count)
          {
            mix_expr.registers.resize(mix_expr.program.register_count, float_value(0.0));
          }
          for (const auto & instr : mix_expr.program.instructions)
          {
            eval_mix_instruction(runtime, instr, mix_expr.registers.data());
          }
          if (mix_expr.result_register < mix_expr.registers.size())
          {
            mixed += expr::to_float64(mix_expr.registers[mix_expr.result_register]) / 20.0;
          }
        }
        outputBuffer[sample] = mixed;

        for (auto & slot : runtime.modules)
        {
          if (!slot.module)
          {
            continue;
          }

          for (unsigned int output_id = 0; output_id < slot.indexed_output_indices.size(); ++output_id)
          {
            const auto & indices = slot.indexed_output_indices[output_id];
            auto & cached_values = slot.indexed_prev_output_values[output_id];
            if (indices.empty() || cached_values.size() != indices.size())
            {
              continue;
            }

#ifdef EGRESS_LLVM_ORC_JIT
            const bool can_read_numeric_output =
              slot.module->jit_kernel_ != nullptr &&
              output_id < slot.module->numeric_output_info_.size() &&
              static_cast<Module::NumericValueKind>(slot.module->numeric_output_info_[output_id].kind) == Module::NumericValueKind::Array &&
              slot.module->numeric_output_info_[output_id].array_slot < slot.module->numeric_array_storage_.size();
            if (can_read_numeric_output)
            {
              const auto & values = slot.module->numeric_array_storage_[slot.module->numeric_output_info_[output_id].array_slot];
              for (std::size_t index_id = 0; index_id < indices.size(); ++index_id)
              {
                const int64_t raw_index = indices[index_id];
                if (raw_index < 0 || static_cast<std::size_t>(raw_index) >= values.size())
                {
                  continue;
                }
                cached_values[index_id] = expr::float_value(Module::clamp_output_scalar(values[static_cast<std::size_t>(raw_index)]));
              }
              continue;
            }
#endif

            if (output_id < slot.module->outputs.size())
            {
              const Value & output = slot.module->outputs[output_id];
              if (expr::is_array(output))
              {
                for (std::size_t index_id = 0; index_id < indices.size(); ++index_id)
                {
                  const int64_t raw_index = indices[index_id];
                  if (raw_index < 0 || static_cast<std::size_t>(raw_index) >= output.array_items.size())
                  {
                    continue;
                  }
                  cached_values[index_id] = output.array_items[static_cast<std::size_t>(raw_index)];
                }
              }
            }
          }

          slot.module->prev_outputs.swap(slot.module->outputs);
        }

        advance_module_sample_indices(runtime);
      }

      if (!runtime.taps.empty())
      {
        for (std::size_t tap_id = 0; tap_id < runtime.taps.size(); ++tap_id)
        {
          auto & tap = runtime.taps[tap_id];
          if (!tap.valid)
          {
            continue;
          }
          tap.buffer.readable.store(tap_write_indices[tap_id], std::memory_order_release);
        }
      }

      audio_processing_.store(false, std::memory_order_release);

#ifdef EGRESS_PROFILE
      const auto callback_end = std::chrono::steady_clock::now();
      const uint64_t callback_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(callback_end - callback_start).count());
      profile_callback_count_.fetch_add(1, std::memory_order_relaxed);
      profile_total_callback_ns_.fetch_add(callback_ns, std::memory_order_relaxed);
      egress_graph_detail::update_atomic_max(profile_max_callback_ns_, callback_ns);

      std::lock_guard<std::mutex> lock(profile_mutex_);
      for (std::size_t module_id = 0; module_id < runtime.modules.size(); ++module_id)
      {
        const auto & slot = runtime.modules[module_id];
        const auto & stats = local_module_stats[module_id];
        if (!slot.module || stats.call_count == 0)
        {
          continue;
        }
        auto & dst = module_profile_stats_[slot.name];
        dst.call_count += stats.call_count;
        dst.total_ns += stats.total_ns;
        if (stats.max_ns > dst.max_ns)
        {
          dst.max_ns = stats.max_ns;
        }
      }
#endif
    }

    void prime_numeric_jit()
    {
      std::lock_guard<std::mutex> pending_lock(pending_mutex_);
      wait_for_runtime_available(0);
      wait_for_runtime_available(1);

      const uint32_t runtime_index = active_runtime_index_.load(std::memory_order_acquire);
      RuntimeState & runtime = runtimes_[runtime_index];
      for (auto & slot : runtime.modules)
      {
        if (!slot.module)
        {
          continue;
        }

        eval_input_program(runtime, slot.input_program, slot.input_registers, slot.module->inputs);
#ifdef EGRESS_LLVM_ORC_JIT
        if (!slot.module->has_dynamic_registers_)
        {
          slot.module->ensure_numeric_jit_current();
        }
#endif
      }
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
      result.nested_module_count = stats.nested_module_count;
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
      for (auto & mix_expr : control_mix_exprs_)
      {
        bool removed_any = false;
        mix_expr = simplify_expr(replace_refs_with_zero(mix_expr, module_name, 0, true, removed_any));
      }
      for (auto & [name, module] : control_modules_)
      {
        for (unsigned int input_id = 0; input_id < module.input_exprs.size(); ++input_id)
        {
          bool removed_any = false;
          const ExprSpecPtr updated = replace_refs_with_zero(module.input_exprs[input_id], module_name, 0, true, removed_any);
          if (removed_any)
          {
            module.input_exprs[input_id] = simplify_expr(updated);
          }
        }
      }
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

    bool addOutputExpr(const ExprSpecPtr & expr)
    {
      if (!expr)
      {
        return false;
      }

      std::lock_guard<std::mutex> lock(pending_mutex_);
      if (!validate_expr_refs(expr, false))
      {
        return false;
      }

      control_mix_exprs_.push_back(simplify_expr(expr));
      rebuild_and_publish_runtime_locked();
      return true;
    }

    std::size_t addOutputTap(const std::string & module_name, unsigned int output_id)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      ControlTap tap;
      tap.active = true;
      tap.output = std::make_pair(module_name, output_id);
      control_taps_.push_back(std::move(tap));
      const std::size_t tap_id = control_taps_.size() - 1;
      rebuild_and_publish_runtime_locked();
      return tap_id;
    }

    bool removeOutputTap(std::size_t tap_id)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      if (tap_id >= control_taps_.size())
      {
        return false;
      }
      control_taps_[tap_id].active = false;
      rebuild_and_publish_runtime_locked();
      return true;
    }

    std::vector<double> outputTapBuffer(std::size_t tap_id) const
    {
      const uint32_t runtime_index = active_runtime_index_.load(std::memory_order_acquire);
      const RuntimeState & runtime = runtimes_[runtime_index];
      if (tap_id >= runtime.taps.size())
      {
        return {};
      }
      const auto & tap = runtime.taps[tap_id];
      if (!tap.valid)
      {
        return {};
      }
      const uint32_t readable = tap.buffer.readable.load(std::memory_order_acquire);
      return tap.buffer.buffers[readable];
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

      module_it->second.input_exprs[input_id] = simplify_expr(expr);

      wait_for_runtime_available(0);
      wait_for_runtime_available(1);
      const bool updated0 =
        recompile_module_inputs_in_runtime(runtimes_[0], module_name, module_it->second.input_exprs);
      const bool updated1 =
        recompile_module_inputs_in_runtime(runtimes_[1], module_name, module_it->second.input_exprs);
      if (updated0 && updated1)
      {
        return true;
      }

      runtimes_[0] = build_runtime_locked();
      runtimes_[1] = build_runtime_locked();
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

    bool prime_module_inputs_if_local(const std::string & module_name)
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto module_it = control_modules_.find(module_name);
      if (module_it == control_modules_.end() || !module_it->second.module)
      {
        return false;
      }

      for (const auto & input_expr : module_it->second.input_exprs)
      {
        std::vector<outputID> refs;
        collect_refs(input_expr, refs);
        if (!refs.empty())
        {
          return false;
        }
      }

      RuntimeState runtime;
      const unsigned int input_count = static_cast<unsigned int>(module_it->second.input_exprs.size());
      const CompiledInputProgram input_program = compile_input_program(module_it->second.input_exprs, input_count, runtime);
      std::vector<Value> input_registers(input_program.register_count, float_value(0.0));
      std::vector<Value> input_values(input_count, float_value(0.0));
      eval_input_program(runtime, input_program, input_registers, input_values);
      module_it->second.module->inputs = std::move(input_values);
#ifdef EGRESS_LLVM_ORC_JIT
      if (!module_it->second.module->has_dynamic_registers_)
      {
        module_it->second.module->ensure_numeric_jit_current();
      }
#endif
      return true;
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
    using ModuleShape = egress_graph_detail::ModuleShape;
    using ControlModule = egress_graph_detail::ControlModule;
    using OpCode = egress_graph_detail::OpCode;
    using ExprInstr = egress_graph_detail::ExprInstr;
    using CompiledInputProgram = egress_graph_detail::CompiledInputProgram;
    using ModuleSlot = egress_graph_detail::ModuleSlot;
    using MixTap = egress_graph_detail::MixTap;
    using MixExpr = egress_graph_detail::MixExpr;
    using RuntimeState = egress_graph_detail::RuntimeState;
#ifdef EGRESS_PROFILE
    using ModuleTimingCounters = egress_graph_detail::ModuleTimingCounters;
    using ProcessModuleTiming = egress_graph_detail::ProcessModuleTiming;
#endif

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

      if (expr->kind == ExprKind::ArraySet)
      {
        return validate_expr_refs(expr->lhs, allow_input_values) &&
               validate_expr_refs(expr->rhs, allow_input_values) &&
               validate_expr_refs(expr->args.empty() ? nullptr : expr->args.front(), allow_input_values);
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

    void wait_for_runtime_available(uint32_t runtime_index) const;

    RuntimeState build_runtime_locked() const;

    void rebuild_and_publish_runtime_locked();

    void refresh_runtime_ref_metadata(RuntimeState & runtime) const;

    bool recompile_module_inputs_in_runtime(
      RuntimeState & runtime,
      const std::string & module_name,
      const std::vector<ExprSpecPtr> & exprs) const;

    static void eval_instruction(const RuntimeState & runtime, const ExprInstr & instr, Value * registers);

    static void eval_mix_instruction(const RuntimeState & runtime, const ExprInstr & instr, Value * registers);

    uint32_t compile_expr_node(
      const ExprSpecPtr & expr,
      CompiledInputProgram & compiled,
      const RuntimeState & runtime) const;

    CompiledInputProgram compile_input_program(
      const std::vector<ExprSpecPtr> & exprs,
      unsigned int input_count,
      const RuntimeState & runtime) const;

    void eval_input_program(
      const RuntimeState & runtime,
      const CompiledInputProgram & program,
      std::vector<Value> & registers,
      std::vector<Value> & inputs) const;

    static uint64_t estimate_module_execution_cost(const Module & module);

    void advance_module_sample_indices(RuntimeState & runtime) const
    {
      for (auto & slot : runtime.modules)
      {
        if (slot.module)
        {
          slot.module->advance_sample_index_tree();
        }
      }
    }

    bool wait_for_next_parallel_epoch(uint64_t seen_generation)
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      worker_cv_.wait(lock, [this, seen_generation] {
        return worker_shutdown_.load(std::memory_order_acquire) ||
               parallel_generation_ != seen_generation;
      });
      return !worker_shutdown_.load(std::memory_order_acquire);
    }

    bool finish_parallel_batch_participant(uint64_t generation, bool wait_for_batch_completion)
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      const uint32_t remaining = parallel_active_participants_ > 0
        ? --parallel_active_participants_
        : 0;
      if (remaining == 0)
      {
        parallel_completed_generation_ = generation;
        worker_done_cv_.notify_one();
        worker_cv_.notify_all();
        return true;
      }

      if (!wait_for_batch_completion)
      {
        return false;
      }

      worker_cv_.wait(lock, [this, generation] {
        return worker_shutdown_.load(std::memory_order_acquire) ||
               parallel_completed_generation_ >= generation ||
               parallel_generation_ != generation;
      });
      return !worker_shutdown_.load(std::memory_order_acquire);
    }

#ifdef EGRESS_PROFILE
    void execute_parallel_module_work(RuntimeState & runtime, std::vector<ProcessModuleTiming> * local_stats)
#else
    void execute_parallel_module_work(RuntimeState & runtime)
#endif
    {
      while (true)
      {
        const uint32_t slot_id = parallel_next_module_index_.fetch_add(1, std::memory_order_relaxed);
        if (slot_id >= runtime.modules.size())
        {
          break;
        }

        auto & slot = runtime.modules[slot_id];
        if (!slot.module)
        {
          continue;
        }

#ifdef EGRESS_PROFILE
        const auto module_start = std::chrono::steady_clock::now();
#endif
        eval_input_program(runtime, slot.input_program, slot.input_registers, slot.module->inputs);
        slot.module->process(&slot.output_materialize_mask);
#ifdef EGRESS_PROFILE
        if (local_stats != nullptr)
        {
          const auto module_end = std::chrono::steady_clock::now();
          const uint64_t module_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(module_end - module_start).count());
          auto & stats = (*local_stats)[slot_id];
          ++stats.call_count;
          stats.total_ns += module_ns;
          if (module_ns > stats.max_ns)
          {
            stats.max_ns = module_ns;
          }
        }
#endif
      }
    }

#ifdef EGRESS_PROFILE
    void start_parallel_module_batch(RuntimeState & runtime, std::vector<ProcessModuleTiming> * local_stats)
#else
    void start_parallel_module_batch(RuntimeState & runtime)
#endif
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      if (worker_threads_.empty())
      {
        lock.unlock();
#ifdef EGRESS_PROFILE
        execute_parallel_module_work(runtime, local_stats);
#else
        execute_parallel_module_work(runtime);
#endif
        return;
      }

      uint32_t active_modules = 0;
      for (const auto & slot : runtime.modules)
      {
        if (slot.module)
        {
          ++active_modules;
        }
      }
      const uint32_t desired_helpers =
        active_modules > 0
          ? std::min<uint32_t>(static_cast<uint32_t>(worker_threads_.size()), active_modules - 1)
          : 0;
      if (desired_helpers == 0)
      {
        lock.unlock();
#ifdef EGRESS_PROFILE
        execute_parallel_module_work(runtime, local_stats);
#else
        execute_parallel_module_work(runtime);
#endif
        return;
      }

      parallel_runtime_ = &runtime;
      parallel_helper_slots_ = desired_helpers;
      parallel_active_participants_ = 1;
#ifdef EGRESS_PROFILE
      parallel_local_stats_ = local_stats;
#endif
      const uint64_t generation = ++parallel_generation_;
      for (uint32_t helper_id = 0; helper_id < desired_helpers; ++helper_id)
      {
        worker_cv_.notify_one();
      }
      lock.unlock();

#ifdef EGRESS_PROFILE
      execute_parallel_module_work(runtime, local_stats);
#else
      execute_parallel_module_work(runtime);
#endif

      if (!finish_parallel_batch_participant(generation, false))
      {
        lock.lock();
        worker_done_cv_.wait(lock, [this, generation] { return parallel_completed_generation_ >= generation; });
      }
      else
      {
        lock.lock();
      }
      parallel_runtime_ = nullptr;
#ifdef EGRESS_PROFILE
      parallel_local_stats_ = nullptr;
#endif
    }

    void worker_main()
    {
      uint64_t seen_generation = 0;
      while (true)
      {
        if (!wait_for_next_parallel_epoch(seen_generation))
        {
          return;
        }
        std::unique_lock<std::mutex> lock(worker_mutex_);
        if (worker_shutdown_.load(std::memory_order_acquire))
        {
          return;
        }

        seen_generation = parallel_generation_;
        if (parallel_helper_slots_ == 0 ||
            parallel_completed_generation_ >= seen_generation ||
            parallel_runtime_ == nullptr)
        {
          continue;
        }
        --parallel_helper_slots_;
        ++parallel_active_participants_;
        RuntimeState * runtime = parallel_runtime_;
#ifdef EGRESS_PROFILE
        auto * local_stats = parallel_local_stats_;
#endif
        lock.unlock();

        if (runtime != nullptr)
        {
#ifdef EGRESS_PROFILE
          execute_parallel_module_work(*runtime, local_stats);
#else
          execute_parallel_module_work(*runtime);
#endif
        }

        if (!finish_parallel_batch_participant(seen_generation, true))
        {
          continue;
        }
      }
    }

    void start_worker_pool_locked(unsigned int helper_count)
    {
      worker_shutdown_.store(false, std::memory_order_release);
      worker_threads_.reserve(helper_count);
      for (unsigned int i = 0; i < helper_count; ++i)
      {
        worker_threads_.emplace_back([this] { worker_main(); });
      }
    }

    void stop_worker_pool_locked(std::vector<std::thread> & threads_to_join)
    {
      worker_shutdown_.store(true, std::memory_order_release);
      worker_cv_.notify_all();
      threads_to_join.swap(worker_threads_);
      parallel_runtime_ = nullptr;
      parallel_helper_slots_ = 0;
      parallel_active_participants_ = 0;
      parallel_completed_generation_ = parallel_generation_;
#ifdef EGRESS_PROFILE
      parallel_local_stats_ = nullptr;
#endif
    }

    unsigned int bufferLength_ = 0;

    std::array<RuntimeState, 2> runtimes_;
    std::atomic<uint32_t> active_runtime_index_{0};
    std::atomic<uint32_t> audio_runtime_index_{0};
    std::atomic<bool> audio_processing_{false};

    std::unordered_map<std::string, ControlModule> control_modules_;
    std::vector<outputID> control_mix_;
    std::vector<ExprSpecPtr> control_mix_exprs_;
    struct ControlTap
    {
      bool active = false;
      outputID output;
    };
    std::vector<ControlTap> control_taps_;

    mutable std::mutex pending_mutex_;
    std::atomic<uint32_t> parallel_next_module_index_{0};
    mutable std::mutex worker_mutex_;
    std::condition_variable worker_cv_;
    std::condition_variable worker_done_cv_;
    std::vector<std::thread> worker_threads_;
    RuntimeState * parallel_runtime_ = nullptr;
    uint64_t parallel_generation_ = 0;
    uint32_t parallel_active_participants_ = 0;
    uint64_t parallel_completed_generation_ = 0;
    uint32_t parallel_helper_slots_ = 0;
    unsigned int worker_count_ = 1;
    std::atomic<bool> worker_shutdown_{false};

  #ifdef EGRESS_PROFILE
    std::vector<ProcessModuleTiming> * parallel_local_stats_ = nullptr;
    std::atomic<uint64_t> profile_callback_count_{0};
    std::atomic<uint64_t> profile_total_callback_ns_{0};
    std::atomic<uint64_t> profile_max_callback_ns_{0};
    mutable std::mutex profile_mutex_;
    std::unordered_map<std::string, ModuleTimingCounters> module_profile_stats_;
  #endif
};
