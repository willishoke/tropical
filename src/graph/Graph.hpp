#pragma once
#include "graph/GraphTypes.hpp"
#include "expr/ExprStructural.hpp"
#include "graph/GraphRuntime.hpp"
#include "graph/Module.hpp"
#include "graph/TypeRegistry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

    struct ModuleRuntimeStats
    {
      bool found = false;
      Module::RuntimeStats stats;
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
    #ifdef EGRESS_PROFILE
      struct FusedSyncStats
      {
        uint64_t call_count = 0;
        double total_ms = 0.0;
        double max_ms = 0.0;
        uint64_t output_copy_count = 0;
      };
    #endif

      bool enabled = false;
      uint64_t callback_count = 0;
      double avg_callback_ms = 0.0;
      double max_callback_ms = 0.0;
      bool primitive_body_available = false;
      bool primitive_body_covers_all_modules = false;
      bool input_kernel_available = false;
      uint64_t fused_input_use_count = 0;
      uint64_t fused_body_use_count = 0;
      std::string fusion_candidate_reason;
      std::string primitive_body_status;
      std::string input_kernel_status;
      std::vector<ModuleProfileStats> modules;
    #ifdef EGRESS_PROFILE
      FusedSyncStats fused_current_output_sync;
      FusedSyncStats fused_prev_output_sync;
    #endif
    };

    explicit Graph(unsigned int bufferLength)
      : bufferLength_(bufferLength), outputBuffer(bufferLength, 0.0),
        type_registry_(std::make_shared<egress::TypeRegistry>())
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      runtimes_[0] = build_runtime_locked();
      runtimes_[1] = build_runtime_locked();
    }

    egress::TypeRegistry& type_registry() { return *type_registry_; }
    const egress::TypeRegistry& type_registry() const { return *type_registry_; }

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

    void set_fusion_enabled(bool enabled)
    {
      fusion_enabled_ = enabled;
    }

    bool fusion_enabled() const
    {
      return fusion_enabled_;
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
        const bool can_use_fused_body =
          runtime.fused_graph != nullptr && runtime.fused_graph->primitive_body_available;
        const bool use_fused_inputs = run_fused_input_kernel(runtime, can_use_fused_body);
        const bool used_fused_body = can_use_fused_body && use_fused_inputs && run_fused_primitive_body_kernel(runtime);
#ifdef EGRESS_PROFILE
        if (use_fused_inputs)
        {
          profile_fused_input_use_count_.fetch_add(1, std::memory_order_relaxed);
        }
        if (used_fused_body)
        {
          profile_fused_body_use_count_.fetch_add(1, std::memory_order_relaxed);
        }
#endif
        const bool body_covers_all_modules =
          used_fused_body &&
          runtime.fused_graph != nullptr &&
          runtime.fused_graph->primitive_body_covers_all_modules;
        const bool module_use_fused_inputs = use_fused_inputs && !used_fused_body;
        if (use_fused_inputs && can_use_fused_body && !used_fused_body)
        {
          run_fused_input_kernel(runtime, false);
        }
        // Snapshot all TriggerParam values once per sample before any module processes.
        // This must run unconditionally — even when the fused body covers all modules — so that
        // triggers fire correctly regardless of which execution path is taken.
        for (auto * p : runtime.trigger_params)
        {
          p->frame_value.store(p->value.exchange(0.0, std::memory_order_acq_rel), std::memory_order_relaxed);
        }
        if (!body_covers_all_modules)
        {
          parallel_next_module_index_.store(0, std::memory_order_relaxed);
          if (worker_count_ > 1 && runtime.modules.size() > 1)
          {
#ifdef EGRESS_PROFILE
            start_parallel_module_batch(runtime, module_use_fused_inputs, &local_module_stats);
#else
            start_parallel_module_batch(runtime, module_use_fused_inputs);
#endif
          }
          else
          {
#ifdef EGRESS_PROFILE
            execute_parallel_module_work(runtime, module_use_fused_inputs, &local_module_stats);
#else
            execute_parallel_module_work(runtime, module_use_fused_inputs);
#endif
          }
        }

        sync_fused_current_outputs(runtime);

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
#ifdef EGRESS_LLVM_ORC_JIT
            double numeric_scalar = 0.0;
            bool mask_ok = tap.output_id < slot.output_materialize_mask.size() &&
                !slot.output_materialize_mask[tap.output_id];
            double pre_scalar = 0.0;
            bool scalar_ok = mask_ok && slot.module->try_get_numeric_scalar_output(tap.output_id, false, pre_scalar);
            numeric_scalar = pre_scalar;
            if (scalar_ok)
            {
              tap.buffer.buffers[tap_write_indices[tap_id]][sample] = numeric_scalar;
              continue;
            }
#endif
            const Value & output = slot.module->materialize_output_value(tap.output_id, false);
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

#ifdef EGRESS_LLVM_ORC_JIT
          double numeric_scalar = 0.0;
          if (tap.output_id < slot.output_materialize_mask.size() &&
              !slot.output_materialize_mask[tap.output_id] &&
              slot.module->try_get_numeric_scalar_output(tap.output_id, false, numeric_scalar))
          {
            mixed += numeric_scalar / 20.0;
            continue;
          }
#endif
          mixed += expr::to_float64(slot.module->materialize_output_value(tap.output_id, false)) / 20.0;
        }
        if (!run_fused_mix_kernel(runtime, mixed))
        {
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
        }
        outputBuffer[sample] = mixed;

        for (std::size_t module_id = 0; module_id < runtime.modules.size(); ++module_id)
        {
          auto & slot = runtime.modules[module_id];
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
            const auto * numeric_values = slot.module->try_get_numeric_output_array_values(output_id);
            if (numeric_values != nullptr)
            {
              for (std::size_t index_id = 0; index_id < indices.size(); ++index_id)
              {
                const int64_t raw_index = indices[index_id];
                if (raw_index < 0 || static_cast<std::size_t>(raw_index) >= numeric_values->size())
                {
                  continue;
                }
                cached_values[index_id] =
                  expr::float_value((*numeric_values)[static_cast<std::size_t>(raw_index)]);
              }
              if (runtime.fused_graph != nullptr &&
                  module_id < runtime.fused_graph->module_output_spans.size())
              {
                const auto & span = runtime.fused_graph->module_output_spans[module_id];
                const uint32_t source_slot = span.first_output_slot + output_id;
                if (source_slot < runtime.fused_graph->indexed_prev_values.size())
                {
                  runtime.fused_graph->indexed_prev_values[source_slot] = cached_values;
                }
              }
              continue;
            }
#endif

            if (output_id < slot.module->outputs.size())
            {
              const Value & output = slot.module->materialize_output_value(output_id, false);
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

            if (runtime.fused_graph != nullptr &&
                module_id < runtime.fused_graph->module_output_spans.size())
            {
              const auto & span = runtime.fused_graph->module_output_spans[module_id];
              const uint32_t source_slot = span.first_output_slot + output_id;
              if (source_slot < runtime.fused_graph->indexed_prev_values.size())
              {
                runtime.fused_graph->indexed_prev_values[source_slot] = cached_values;
              }
            }
          }

          slot.module->prev_outputs.swap(slot.module->outputs);
  #ifdef EGRESS_LLVM_ORC_JIT
          slot.module->capture_numeric_prev_array_outputs();
          slot.module->numeric_prev_output_scalar_mask_.swap(slot.module->numeric_output_scalar_mask_);
          slot.module->numeric_prev_output_scalars_.swap(slot.module->numeric_output_scalars_);
  #endif
        }

        sync_fused_prev_outputs(runtime);
        advance_module_sample_indices(runtime);
      }

      // Apply output gain envelope (fade-in / fade-out) to the completed buffer.
      {
        int fi = fade_in_remaining_.load(std::memory_order_relaxed);
        int fo = fade_out_remaining_.load(std::memory_order_relaxed);
        if (fi > 0 || fo != -1)
        {
          for (unsigned int s = 0; s < bufferLength_; ++s)
          {
            if (fi > 0)
            {
              const double t = 1.0 - static_cast<double>(fi) / kFadeSamples_;
              outputBuffer[s] *= t * t * (3.0 - 2.0 * t);
              --fi;
            }
            if (fo != -1)
            {
              if (fo > 0)
              {
                const double t = static_cast<double>(fo) / kFadeSamples_;
                outputBuffer[s] *= t * t * (3.0 - 2.0 * t);
                --fo;
              }
              else
              {
                outputBuffer[s] = 0.0;  // fade complete — hold at silence
              }
            }
          }
          fade_in_remaining_.store(fi, std::memory_order_relaxed);
          fade_out_remaining_.store(fo, std::memory_order_relaxed);
        }
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

    // Signal a fade-in: outputBuffer will be smoothstep-scaled from 0→1
    // over the next `samples` samples. Called by the DAC before startStream.
    void begin_fade_in(int samples = 2048)
    {
      fade_out_remaining_.store(-1, std::memory_order_relaxed);
      fade_in_remaining_.store(samples, std::memory_order_relaxed);
    }

    // Signal a fade-out: outputBuffer will be smoothstep-scaled from 1→0
    // over the next `samples` samples, then held at silence. Called by the
    // DAC before stopStream.
    void begin_fade_out(int samples = 2048)
    {
      fade_out_remaining_.store(samples, std::memory_order_release);
    }

    // Returns true once the fade-out has completed and the engine is holding
    // at silence (fade_out_remaining_ == 0).  Poll this after begin_fade_out()
    // to know when it is safe to call stopStream() without an audible click.
    bool is_fade_out_complete() const
    {
      return fade_out_remaining_.load(std::memory_order_acquire) == 0;
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
        slot.module->ensure_numeric_jit_current();
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
      const uint32_t runtime_index = active_runtime_index_.load(std::memory_order_acquire);
      if (runtime_index < 2)
      {
        const auto & runtime = runtimes_[runtime_index];
        if (runtime.fused_graph)
        {
          stats.primitive_body_available = runtime.fused_graph->primitive_body_available;
          stats.primitive_body_covers_all_modules = runtime.fused_graph->primitive_body_covers_all_modules;
          stats.input_kernel_available = runtime.fused_graph->input_kernel.available;
          stats.fused_input_use_count = profile_fused_input_use_count_.load(std::memory_order_relaxed);
          stats.fused_body_use_count = profile_fused_body_use_count_.load(std::memory_order_relaxed);
          stats.fusion_candidate_reason = runtime.fused_graph->candidate_reason;
          stats.primitive_body_status = runtime.fused_graph->primitive_body_status;
          stats.input_kernel_status = runtime.fused_graph->input_kernel.status;
        }
      }
      stats.fused_current_output_sync.call_count =
        profile_fused_current_sync_call_count_.load(std::memory_order_relaxed);
      stats.fused_current_output_sync.total_ms =
        static_cast<double>(profile_fused_current_sync_total_ns_.load(std::memory_order_relaxed)) / 1e6;
      stats.fused_current_output_sync.max_ms =
        static_cast<double>(profile_fused_current_sync_max_ns_.load(std::memory_order_relaxed)) / 1e6;
      stats.fused_current_output_sync.output_copy_count =
        profile_fused_current_sync_output_copy_count_.load(std::memory_order_relaxed);
      stats.fused_prev_output_sync.call_count =
        profile_fused_prev_sync_call_count_.load(std::memory_order_relaxed);
      stats.fused_prev_output_sync.total_ms =
        static_cast<double>(profile_fused_prev_sync_total_ns_.load(std::memory_order_relaxed)) / 1e6;
      stats.fused_prev_output_sync.max_ms =
        static_cast<double>(profile_fused_prev_sync_max_ns_.load(std::memory_order_relaxed)) / 1e6;
      stats.fused_prev_output_sync.output_copy_count =
        profile_fused_prev_sync_output_copy_count_.load(std::memory_order_relaxed);

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
      profile_fused_current_sync_call_count_.store(0, std::memory_order_relaxed);
      profile_fused_current_sync_total_ns_.store(0, std::memory_order_relaxed);
      profile_fused_current_sync_max_ns_.store(0, std::memory_order_relaxed);
      profile_fused_current_sync_output_copy_count_.store(0, std::memory_order_relaxed);
      profile_fused_prev_sync_call_count_.store(0, std::memory_order_relaxed);
      profile_fused_prev_sync_total_ns_.store(0, std::memory_order_relaxed);
      profile_fused_prev_sync_max_ns_.store(0, std::memory_order_relaxed);
      profile_fused_prev_sync_output_copy_count_.store(0, std::memory_order_relaxed);
      profile_fused_input_use_count_.store(0, std::memory_order_relaxed);
      profile_fused_body_use_count_.store(0, std::memory_order_relaxed);

      {
        std::lock_guard<std::mutex> lock(profile_mutex_);
        module_profile_stats_.clear();
      }

      std::lock_guard<std::mutex> pending_lock(pending_mutex_);
      for (auto & [name, module] : control_modules_)
      {
        (void)name;
        if (module.module)
        {
          module.module->reset_runtime_stats();
        }
      }
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

    ModuleRuntimeStats module_runtime_stats(const std::string & module_name) const
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      ModuleRuntimeStats result;
      auto it = control_modules_.find(module_name);
      if (it == control_modules_.end() || !it->second.module)
      {
        return result;
      }

      result.found = true;
      result.stats = it->second.module->runtime_stats();
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
      module.module->type_registry_ = type_registry_.get();
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

    void begin_update()
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      batch_update_active_ = true;
      batch_dirty_modules_.clear();
    }

    bool end_update()
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      batch_update_active_ = false;
      if (batch_dirty_modules_.empty())
      {
        return true;
      }

      // Double-buffer protocol (mirrors rebuild_and_publish_runtime_locked):
      // 1. Update the currently-inactive runtime — audio is on the active one so this is safe.
      // 2. Atomic-swap so audio switches to the freshly updated runtime.
      // 3. Update the formerly-active (now-inactive) runtime — audio has moved away from it.
      //
      // The previous approach waited for both runtimes to be free simultaneously, which only
      // happens during the brief !audio_processing_ window between callbacks. That window is
      // not a lock: the audio callback can restart immediately and read a runtime we are still
      // writing, causing a data race and crash.

      const uint32_t active   = active_runtime_index_.load(std::memory_order_acquire);
      const uint32_t inactive = 1U - active;

      // Step 1: update inactive (audio is on `active`, so `inactive` is free).
      wait_for_runtime_available(inactive);
      bool all_ok = true;
      for (const auto & name : batch_dirty_modules_)
      {
        auto it = control_modules_.find(name);
        if (it == control_modules_.end())
        {
          continue;
        }
        if (!recompile_module_inputs_in_runtime(runtimes_[inactive], name, it->second.input_exprs))
        {
          all_ok = false;
        }
      }
      if (!all_ok)
      {
        runtimes_[inactive] = build_runtime_locked();
      }

      // Step 2: swap — audio now uses the updated `inactive` runtime.
      active_runtime_index_.store(inactive, std::memory_order_release);

      // Step 3: update formerly-active (now-inactive) runtime.
      // Audio has read the new active_runtime_index_, so it will no longer use `active`.
      wait_for_runtime_available(active);
      all_ok = true;
      for (const auto & name : batch_dirty_modules_)
      {
        auto it = control_modules_.find(name);
        if (it == control_modules_.end())
        {
          continue;
        }
        if (!recompile_module_inputs_in_runtime(runtimes_[active], name, it->second.input_exprs))
        {
          all_ok = false;
        }
      }
      if (!all_ok)
      {
        runtimes_[active] = build_runtime_locked();
      }

      batch_dirty_modules_.clear();
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

      if (batch_update_active_)
      {
        batch_dirty_modules_.insert(module_name);
        return true;
      }

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
      module_it->second.module->ensure_numeric_jit_current();
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
    using FusedGraphState = egress_graph_detail::FusedGraphState;
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

    void rebuild_fused_graph_state(RuntimeState & runtime) const;

    void sync_fused_current_outputs(RuntimeState & runtime) const;

    void sync_fused_prev_outputs(RuntimeState & runtime) const;

    bool run_fused_input_kernel(RuntimeState & runtime, bool allow_primitive_body_inputs) const;

    bool run_fused_primitive_body_kernel(RuntimeState & runtime) const;

    bool run_fused_mix_kernel(RuntimeState & runtime, double & mixed) const;

    bool recompile_module_inputs_in_runtime(
      RuntimeState & runtime,
      const std::string & module_name,
      const std::vector<ExprSpecPtr> & exprs) const;

    std::unique_ptr<FusedGraphState> build_fused_graph_state(const RuntimeState & runtime) const;

    static uint64_t fused_source_output_key(uint32_t module_id, unsigned int output_id)
    {
      return (static_cast<uint64_t>(module_id) << 32) | static_cast<uint64_t>(output_id);
    }

    static bool supports_fused_numeric_opcode(OpCode opcode);

    static bool is_fused_numeric_candidate(const CompiledInputProgram & program);

    static bool program_uses_current_outputs(const CompiledInputProgram & program);

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

    struct WorkerThreadPolicy
    {
      bool has_sched_params = false;
      int sched_policy = 0;
      sched_param sched_params{};
#ifdef __APPLE__
      bool has_qos_class = false;
      qos_class_t qos_class = QOS_CLASS_UNSPECIFIED;
      int qos_relative_priority = 0;
#endif
    };

    static bool worker_thread_policy_equals(
      const WorkerThreadPolicy & lhs,
      const WorkerThreadPolicy & rhs)
    {
      if (lhs.has_sched_params != rhs.has_sched_params)
      {
        return false;
      }
      if (lhs.has_sched_params &&
          (lhs.sched_policy != rhs.sched_policy ||
           lhs.sched_params.sched_priority != rhs.sched_params.sched_priority))
      {
        return false;
      }
#ifdef __APPLE__
      if (lhs.has_qos_class != rhs.has_qos_class)
      {
        return false;
      }
      if (lhs.has_qos_class &&
          (lhs.qos_class != rhs.qos_class ||
           lhs.qos_relative_priority != rhs.qos_relative_priority))
      {
        return false;
      }
#endif
      return true;
    }

    static WorkerThreadPolicy capture_current_worker_thread_policy()
    {
      WorkerThreadPolicy policy;
      pthread_t current = pthread_self();
      if (pthread_getschedparam(current, &policy.sched_policy, &policy.sched_params) == 0)
      {
        policy.has_sched_params = true;
      }
#ifdef __APPLE__
      qos_class_t qos_class = QOS_CLASS_UNSPECIFIED;
      int relative_priority = 0;
      if (pthread_get_qos_class_np(current, &qos_class, &relative_priority) == 0 &&
          qos_class != QOS_CLASS_UNSPECIFIED)
      {
        policy.has_qos_class = true;
        policy.qos_class = qos_class;
        policy.qos_relative_priority = relative_priority;
      }
#endif
      return policy;
    }

    void refresh_worker_thread_policy_locked()
    {
      const WorkerThreadPolicy desired = capture_current_worker_thread_policy();
      if (worker_thread_policy_generation_ > 0 &&
          worker_thread_policy_equals(worker_thread_policy_, desired))
      {
        return;
      }
      worker_thread_policy_ = desired;
      ++worker_thread_policy_generation_;
    }

    static void apply_worker_thread_policy(const WorkerThreadPolicy & policy)
    {
#ifdef __APPLE__
      if (policy.has_qos_class)
      {
        pthread_set_qos_class_self_np(policy.qos_class, policy.qos_relative_priority);
      }
#endif
      if (policy.has_sched_params)
      {
        pthread_setschedparam(pthread_self(), policy.sched_policy, &policy.sched_params);
      }
    }

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
    void execute_parallel_module_work(
      RuntimeState & runtime,
      bool use_fused_inputs,
      std::vector<ProcessModuleTiming> * local_stats)
#else
    void execute_parallel_module_work(RuntimeState & runtime, bool use_fused_inputs)
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
        if (use_fused_inputs &&
            runtime.fused_graph != nullptr &&
            slot_id < runtime.fused_graph->primitive_body_module_mask.size() &&
            runtime.fused_graph->primitive_body_module_mask[slot_id])
        {
          continue;
        }
        if (!use_fused_inputs)
        {
          eval_input_program(runtime, slot.input_program, slot.input_registers, slot.module->inputs);
        }
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
    void start_parallel_module_batch(
      RuntimeState & runtime,
      bool use_fused_inputs,
      std::vector<ProcessModuleTiming> * local_stats)
#else
    void start_parallel_module_batch(RuntimeState & runtime, bool use_fused_inputs)
#endif
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      refresh_worker_thread_policy_locked();
      if (worker_threads_.empty())
      {
        lock.unlock();
#ifdef EGRESS_PROFILE
        execute_parallel_module_work(runtime, use_fused_inputs, local_stats);
#else
        execute_parallel_module_work(runtime, use_fused_inputs);
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
        execute_parallel_module_work(runtime, use_fused_inputs, local_stats);
#else
        execute_parallel_module_work(runtime, use_fused_inputs);
#endif
        return;
      }

      parallel_runtime_ = &runtime;
      parallel_helper_slots_ = desired_helpers;
      parallel_active_participants_ = 1;
      parallel_use_fused_inputs_ = use_fused_inputs;
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
      execute_parallel_module_work(runtime, use_fused_inputs, local_stats);
#else
      execute_parallel_module_work(runtime, use_fused_inputs);
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
      uint64_t seen_policy_generation = 0;
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
        const uint64_t policy_generation = worker_thread_policy_generation_;
        const WorkerThreadPolicy policy = worker_thread_policy_;
#ifdef EGRESS_PROFILE
        auto * local_stats = parallel_local_stats_;
#endif
        lock.unlock();

        if (policy_generation != seen_policy_generation)
        {
          apply_worker_thread_policy(policy);
          seen_policy_generation = policy_generation;
        }

        if (runtime != nullptr)
        {
#ifdef EGRESS_PROFILE
          execute_parallel_module_work(*runtime, parallel_use_fused_inputs_, local_stats);
#else
          execute_parallel_module_work(*runtime, parallel_use_fused_inputs_);
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
      parallel_use_fused_inputs_ = false;
#ifdef EGRESS_PROFILE
      parallel_local_stats_ = nullptr;
#endif
    }

    unsigned int bufferLength_ = 0;
    std::shared_ptr<egress::TypeRegistry> type_registry_;

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
    bool batch_update_active_ = false;
    std::unordered_set<std::string> batch_dirty_modules_;
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
    bool parallel_use_fused_inputs_ = false;
    unsigned int worker_count_ = 1;
    bool fusion_enabled_ = false;
    std::atomic<bool> worker_shutdown_{false};
    WorkerThreadPolicy worker_thread_policy_{};
    uint64_t worker_thread_policy_generation_ = 0;

  #ifdef EGRESS_PROFILE
    static void update_profile_max(std::atomic<uint64_t> & dst, uint64_t candidate)
    {
      uint64_t current = dst.load(std::memory_order_relaxed);
      while (current < candidate &&
             !dst.compare_exchange_weak(current, candidate, std::memory_order_relaxed))
      {
      }
    }

    void record_fused_sync_profile(
      bool previous_outputs,
      uint64_t elapsed_ns,
      uint64_t output_copy_count) const
    {
      if (previous_outputs)
      {
        profile_fused_prev_sync_call_count_.fetch_add(1, std::memory_order_relaxed);
        profile_fused_prev_sync_total_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
        update_profile_max(profile_fused_prev_sync_max_ns_, elapsed_ns);
        profile_fused_prev_sync_output_copy_count_.fetch_add(output_copy_count, std::memory_order_relaxed);
        return;
      }

      profile_fused_current_sync_call_count_.fetch_add(1, std::memory_order_relaxed);
      profile_fused_current_sync_total_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
      update_profile_max(profile_fused_current_sync_max_ns_, elapsed_ns);
      profile_fused_current_sync_output_copy_count_.fetch_add(output_copy_count, std::memory_order_relaxed);
    }

    std::vector<ProcessModuleTiming> * parallel_local_stats_ = nullptr;
    std::atomic<uint64_t> profile_callback_count_{0};
    std::atomic<uint64_t> profile_total_callback_ns_{0};
    std::atomic<uint64_t> profile_max_callback_ns_{0};
    mutable std::atomic<uint64_t> profile_fused_current_sync_call_count_{0};
    mutable std::atomic<uint64_t> profile_fused_current_sync_total_ns_{0};
    mutable std::atomic<uint64_t> profile_fused_current_sync_max_ns_{0};
    mutable std::atomic<uint64_t> profile_fused_current_sync_output_copy_count_{0};
    mutable std::atomic<uint64_t> profile_fused_prev_sync_call_count_{0};
    mutable std::atomic<uint64_t> profile_fused_prev_sync_total_ns_{0};
    mutable std::atomic<uint64_t> profile_fused_prev_sync_max_ns_{0};
    mutable std::atomic<uint64_t> profile_fused_prev_sync_output_copy_count_{0};
    mutable std::atomic<uint64_t> profile_fused_input_use_count_{0};
    mutable std::atomic<uint64_t> profile_fused_body_use_count_{0};
    mutable std::mutex profile_mutex_;
    std::unordered_map<std::string, ModuleTimingCounters> module_profile_stats_;
  #endif

    static constexpr int kFadeSamples_ = 2048;
    std::atomic<int> fade_in_remaining_{0};
    std::atomic<int> fade_out_remaining_{-1};  // -1 = inactive; 0 = hold at silence
};
