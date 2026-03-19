#pragma once

#include "graph/GraphTypes.hpp"
#include "graph/ModuleNumericJit.hpp"
#include "graph/ModuleProgram.hpp"
#include "expr/ExprStructural.hpp"
#include "jit/OrcJitEngine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class Graph;

class Module
{
  public:
#ifdef EGRESS_PROFILE
    struct CompileStats
    {
      uint64_t instruction_count = 0;
      uint64_t register_count = 0;
      uint64_t numeric_jit_instruction_count = 0;
      uint64_t nested_module_count = 0;
      std::string jit_status;
    };

    struct RuntimeStats
    {
      uint64_t numeric_input_sync_call_count = 0;
      uint64_t numeric_input_sync_total_ns = 0;
      uint64_t numeric_input_sync_max_ns = 0;
      uint64_t numeric_output_materialize_call_count = 0;
      uint64_t numeric_output_materialize_total_ns = 0;
      uint64_t numeric_output_materialize_max_ns = 0;
      uint64_t materialized_scalar_outputs = 0;
      uint64_t materialized_array_outputs = 0;
      uint64_t materialized_matrix_outputs = 0;
      uint64_t numeric_register_sync_call_count = 0;
      uint64_t numeric_register_sync_total_ns = 0;
      uint64_t numeric_register_sync_max_ns = 0;
      uint64_t materialized_scalar_registers = 0;
      uint64_t materialized_array_registers = 0;
    };
#endif

    struct RegisterArraySpec
    {
      bool enabled = false;
      unsigned int source_input_id = 0;
      Value init_value;
    };

    struct DelayStateSpec
    {
      uint32_t node_id = 0;
      Value initial_value;
      ExprSpecPtr update_expr;
    };

    struct NestedModuleSpec
    {
      uint32_t node_id = 0;
      std::string label;
      unsigned int input_count = 0;
      std::vector<ExprSpecPtr> input_exprs;
      std::vector<ExprSpecPtr> output_exprs;
      std::vector<ExprSpecPtr> register_exprs;
      std::vector<Value> initial_registers;
      std::vector<RegisterArraySpec> register_array_specs;
      std::vector<DelayStateSpec> delay_state_specs;
      std::vector<NestedModuleSpec> nested_module_specs;
      std::vector<uint32_t> composite_schedule;
      uint32_t output_boundary_id = std::numeric_limits<uint32_t>::max();
      double sample_rate = 44100.0;
    };

    virtual ~Module() = default;

    Module(
      unsigned int input_count,
      std::vector<ExprSpecPtr> output_exprs,
      std::vector<ExprSpecPtr> register_exprs,
      std::vector<Value> initial_registers,
      std::vector<RegisterArraySpec> register_array_specs,
      std::vector<DelayStateSpec> delay_state_specs,
      std::vector<NestedModuleSpec> nested_module_specs,
      std::vector<uint32_t> composite_schedule,
      uint32_t output_boundary_id,
      double sample_rate)
      : inputs(input_count, expr::float_value(0.0)),
        outputs(output_exprs.size(), expr::float_value(0.0)),
        prev_outputs(output_exprs.size(), expr::float_value(0.0)),
        input_count_(input_count),
        registers_(std::move(initial_registers)),
        next_registers_(registers_),
        register_array_specs_(std::move(register_array_specs)),
        sample_rate_(sample_rate),
        composite_output_boundary_id_(output_boundary_id)
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

      // Pre-walk expression trees to find SmoothedParam nodes and allocate anonymous registers.
      // Anonymous registers are appended after user-defined registers in registers_/next_registers_.
      user_register_count_ = static_cast<uint32_t>(registers_.size());
      {
        uint32_t next_anon_idx = 0;
        for (const auto & e : output_exprs) walk_expr_for_params(e, param_anon_reg_map_, next_anon_idx);
        for (const auto & e : register_exprs) walk_expr_for_params(e, param_anon_reg_map_, next_anon_idx);
        for (const auto & ds : delay_state_specs) walk_expr_for_params(ds.update_expr, param_anon_reg_map_, next_anon_idx);
        has_smoothed_params_ = !param_anon_reg_map_.empty();
        // Collect TriggerParam pointers separately so Graph can snapshot them per frame.
        for (const auto & e : output_exprs) collect_trigger_params(e, trigger_params_);
        for (const auto & e : register_exprs) collect_trigger_params(e, trigger_params_);
        for (const auto & ds : delay_state_specs) collect_trigger_params(ds.update_expr, trigger_params_);
        if (has_smoothed_params_)
        {
          // Build sorted list so register slots are assigned deterministically
          std::vector<std::pair<uint32_t, egress_expr::ControlParam *>> sorted_anon;
          sorted_anon.reserve(param_anon_reg_map_.size());
          for (const auto & kv : param_anon_reg_map_) sorted_anon.push_back({kv.second, kv.first});
          std::sort(sorted_anon.begin(), sorted_anon.end());
          for (const auto & [idx, p] : sorted_anon)
          {
            const double init = p->value.load(std::memory_order_relaxed);
            registers_.push_back(egress_expr::float_value(init));
            next_registers_.push_back(egress_expr::float_value(init));
          }
        }
      }

      program_ = compile_program(output_exprs, register_exprs);
      std::vector<ExprSpecPtr> delay_update_exprs;
      delay_update_exprs.reserve(delay_state_specs.size());
      delay_states_.reserve(delay_state_specs.size());
      next_delay_states_.reserve(delay_state_specs.size());
      for (const auto & delay_state_spec : delay_state_specs)
      {
        delay_state_lookup_[delay_state_spec.node_id] = delay_states_.size();
        delay_states_.push_back(delay_state_spec.initial_value);
        next_delay_states_.push_back(delay_state_spec.initial_value);
        delay_update_exprs.push_back(delay_state_spec.update_expr);
      }
      if (!delay_update_exprs.empty())
      {
        delay_update_program_ = compile_program(delay_update_exprs, {});
      }
      composite_schedule_ = std::move(composite_schedule);
      for (auto & nested_module_spec : nested_module_specs)
      {
        NestedModuleRuntime runtime;
        runtime.node_id = nested_module_spec.node_id;
        runtime.input_program = compile_program(nested_module_spec.input_exprs, {});
        runtime.input_temps.assign(runtime.input_program.register_count, expr::float_value(0.0));
        runtime.module = std::make_unique<Module>(
          nested_module_spec.input_count,
          std::move(nested_module_spec.output_exprs),
          std::move(nested_module_spec.register_exprs),
          std::move(nested_module_spec.initial_registers),
          std::move(nested_module_spec.register_array_specs),
          std::move(nested_module_spec.delay_state_specs),
          std::move(nested_module_spec.nested_module_specs),
          std::move(nested_module_spec.composite_schedule),
          nested_module_spec.output_boundary_id,
          nested_module_spec.sample_rate);
        nested_module_lookup_[runtime.node_id] = nested_modules_.size();
        nested_modules_.push_back(std::move(runtime));
      }
      has_nested_modules_ = !nested_modules_.empty();
      has_delay_states_ = !delay_states_.empty();
      if (has_nested_modules_ || has_delay_states_)
      {
        if (composite_output_boundary_id_ == std::numeric_limits<uint32_t>::max())
        {
          throw std::invalid_argument("Composite runtime requires a valid output boundary node.");
        }
        composite_output_program_ = compile_program(output_exprs, {});
        composite_register_program_ = compile_program({}, register_exprs);
      }
      const uint32_t temp_register_count = std::max(
        program_.register_count,
        std::max(
          composite_output_program_.register_count,
          std::max(composite_register_program_.register_count, delay_update_program_.register_count)));
      temps_.assign(temp_register_count, expr::float_value(0.0));

#ifdef EGRESS_LLVM_ORC_JIT
      if (!has_dynamic_registers_ && !has_smoothed_params_)
      {
        initialize_numeric_jit(inputs);
      }
      else if (has_smoothed_params_)
      {
        jit_status_ = "numeric JIT disabled for SmoothedParam (one-pole smoother)";
      }
      else
      {
        jit_status_ = "numeric JIT disabled for dynamic array_state registers";
      }
#endif
    }

    void process(const std::vector<bool> * output_materialize_mask = nullptr);

    unsigned int input_count() const;

    unsigned int output_count() const;

    unsigned int register_count() const;

    // Returns the set of TriggerParam ControlParam pointers used by this module.
    // The Graph uses this to snapshot trigger values once per frame before processing.
    const std::unordered_set<egress_expr::ControlParam *> & trigger_params() const
    {
      return trigger_params_;
    }

  #ifdef EGRESS_PROFILE
    CompileStats compile_stats() const;

    RuntimeStats runtime_stats() const;

    void reset_runtime_stats();
  #endif

  protected:
    void reset_inputs_after_process();

    void postprocess();

    std::vector<Value> inputs;
    std::vector<Value> outputs;
    std::vector<Value> prev_outputs;

  private:
    friend class Graph;
    void advance_sample_index_tree();

    using Instr = egress_module_detail::Instr;
    using CompiledProgram = egress_module_detail::CompiledProgram;
    using NumericInputInfo = egress_module_detail::NumericInputInfo;
    using NumericOutputInfo = egress_module_detail::NumericOutputInfo;
    using NumericValueKind = egress_module_detail::NumericValueKind;
    using NumericValueRef = egress_module_detail::NumericValueRef;
    using NumericRegInfo = egress_module_detail::NumericRegInfo;

    struct NestedModuleRuntime
    {
      uint32_t node_id = 0;
      CompiledProgram input_program;
      std::vector<Value> input_temps;
      std::unique_ptr<Module> module;
    };

    static double clamp_output_scalar(double value);

    static void clamp_output_value(Value & value);

    void resize_array_registers_to_inputs();

    CompiledProgram compile_program(
      const std::vector<ExprSpecPtr> & output_exprs,
      const std::vector<ExprSpecPtr> & register_exprs);

    uint32_t compile_expr_node(
      const ExprSpecPtr & expr,
      CompiledProgram & compiled,
      std::unordered_map<std::size_t, std::vector<std::pair<ExprSpecPtr, uint32_t>>> & memo,
      std::unordered_map<const ExprSpec *, std::size_t> & hash_cache);

    void eval_program(const CompiledProgram & expr, std::vector<Value> & temps);

    const Value & materialize_output_value(unsigned int output_id, bool previous = false);

    // Walk an expression tree and collect unique ControlParam pointers.
    // Assigns each a sequential anonymous register index (0-based) in map.
    static void collect_trigger_params(
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

    static void walk_expr_for_params(
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

    unsigned int input_count_ = 0;
    uint32_t user_register_count_ = 0;
    std::unordered_map<egress_expr::ControlParam *, uint32_t> param_anon_reg_map_;
    std::unordered_set<egress_expr::ControlParam *> trigger_params_;
    bool has_smoothed_params_ = false;
    CompiledProgram program_;
    CompiledProgram composite_output_program_;
    CompiledProgram composite_register_program_;
    CompiledProgram delay_update_program_;
    std::vector<Value> temps_;
    std::vector<Value> registers_;
    std::vector<Value> next_registers_;
    std::vector<Value> delay_states_;
    std::vector<Value> next_delay_states_;
    std::vector<NestedModuleRuntime> nested_modules_;
    std::unordered_map<uint32_t, std::size_t> nested_module_lookup_;
    std::unordered_map<uint32_t, std::size_t> delay_state_lookup_;
    std::vector<uint32_t> composite_schedule_;
    uint32_t composite_output_boundary_id_ = std::numeric_limits<uint32_t>::max();
    std::vector<RegisterArraySpec> register_array_specs_;
    bool has_dynamic_registers_ = false;
    bool has_nested_modules_ = false;
    bool has_delay_states_ = false;
    double sample_rate_ = 44100.0;
    uint64_t sample_index_ = 0;
#ifdef EGRESS_LLVM_ORC_JIT
    egress_jit::NumericKernelFn jit_kernel_ = nullptr;
    std::vector<double> numeric_inputs_;
    bool numeric_input_override_active_ = false;
    std::vector<double> numeric_input_scalar_override_;
    std::vector<double> numeric_temps_;
    std::vector<std::vector<double>> numeric_array_storage_;
    std::vector<double *> numeric_array_ptrs_;
    std::vector<uint64_t> numeric_array_sizes_;
    std::vector<bool> register_scalar_mask_;
    std::vector<uint32_t> register_array_slot_;
    std::vector<int32_t> array_register_targets_;
    std::vector<bool> array_register_can_swap_;
    std::vector<NumericInputInfo> numeric_input_info_;
    std::vector<NumericOutputInfo> numeric_output_info_;
    enum class NumericSyntheticInputKind : uint8_t
    {
      NestedOutput,
      DelayState
    };

    struct NumericSyntheticInput
    {
      NumericSyntheticInputKind kind = NumericSyntheticInputKind::NestedOutput;
      uint32_t slot_id = 0;
      uint32_t output_id = 0;
      uint32_t input_slot = 0;
    };

    struct PreparedNumericJitProgram
    {
      CompiledProgram program;
      std::vector<NumericSyntheticInput> synthetic_inputs;
    };

    struct NumericJitState
    {
      egress_jit::NumericKernelFn kernel = nullptr;
      PreparedNumericJitProgram prepared;
      std::vector<NumericInputInfo> input_info;
      std::vector<NumericOutputInfo> output_info;
      std::vector<double> inputs;
      std::vector<double> temps;
      std::vector<std::vector<double>> array_storage;
      std::vector<double *> array_ptrs;
      std::vector<uint64_t> array_sizes;
      std::vector<bool> register_scalar_mask;
      std::vector<uint32_t> register_array_slot;
      std::vector<int32_t> array_register_targets;
      std::vector<bool> array_register_can_swap;
  #ifdef EGRESS_PROFILE
      uint64_t instruction_count = 0;
   #endif
    };

    struct CompositeBodyJitState
    {
      NumericJitState state;
      CompiledProgram program;
      uint32_t output_count = 0;
      uint32_t delay_output_count = 0;
    #ifdef EGRESS_PROFILE
      uint64_t instruction_count = 0;
    #endif
    };

    std::vector<double> numeric_registers_;
    std::vector<double> numeric_next_registers_;
    std::vector<std::vector<double>> numeric_register_arrays_;
    bool value_registers_dirty_ = false;
    std::vector<bool> numeric_output_scalar_mask_;
    std::vector<double> numeric_output_scalars_;
    std::vector<bool> numeric_prev_output_scalar_mask_;
    std::vector<double> numeric_prev_output_scalars_;
    std::vector<bool> numeric_prev_output_array_mask_;
    std::vector<std::vector<double>> numeric_prev_output_arrays_;
    std::vector<bool> numeric_delay_scalar_mask_;
    std::vector<double> numeric_delay_scalars_;
    std::vector<bool> numeric_delay_array_mask_;
    std::vector<std::vector<double>> numeric_delay_arrays_;
    bool value_delay_states_dirty_ = false;
    CompositeBodyJitState composite_body_jit_;
    NumericJitState composite_output_jit_;
    NumericJitState composite_register_jit_;
    NumericJitState delay_update_jit_;
    std::vector<std::unique_ptr<NumericJitState>> nested_input_jit_states_;
    std::string jit_status_;
  #ifdef EGRESS_PROFILE
    uint64_t numeric_jit_instruction_count_ = 0;

    static void update_profile_max(std::atomic<uint64_t> & dst, uint64_t candidate);

    void record_numeric_input_sync_profile(uint64_t elapsed_ns);

    void record_numeric_output_materialize_profile(
      uint64_t elapsed_ns,
      uint64_t scalar_count,
      uint64_t array_count,
      uint64_t matrix_count);

    void record_numeric_register_sync_profile(
      uint64_t elapsed_ns,
      uint64_t scalar_count,
      uint64_t array_count);

    std::atomic<uint64_t> profile_numeric_input_sync_call_count_{0};
    std::atomic<uint64_t> profile_numeric_input_sync_total_ns_{0};
    std::atomic<uint64_t> profile_numeric_input_sync_max_ns_{0};
    std::atomic<uint64_t> profile_numeric_output_materialize_call_count_{0};
    std::atomic<uint64_t> profile_numeric_output_materialize_total_ns_{0};
    std::atomic<uint64_t> profile_numeric_output_materialize_max_ns_{0};
    std::atomic<uint64_t> profile_materialized_scalar_outputs_{0};
    std::atomic<uint64_t> profile_materialized_array_outputs_{0};
    std::atomic<uint64_t> profile_materialized_matrix_outputs_{0};
    std::atomic<uint64_t> profile_numeric_register_sync_call_count_{0};
    std::atomic<uint64_t> profile_numeric_register_sync_total_ns_{0};
    std::atomic<uint64_t> profile_numeric_register_sync_max_ns_{0};
    std::atomic<uint64_t> profile_materialized_scalar_registers_{0};
    std::atomic<uint64_t> profile_materialized_array_registers_{0};
  #endif

    bool supports_numeric_jit_expr_kind(ExprKind kind) const;

    static void assign_scalar_numeric_value(Value & dst, double value);

    static void assign_numeric_value_to(
      Value & dst,
      const NumericOutputInfo & info,
      uint32_t scalar_register,
      const std::vector<double> & numeric_temps,
      const std::vector<std::vector<double>> & numeric_array_storage);

    static NumericValueRef make_numeric_value_ref(
      const NumericOutputInfo & info,
      uint32_t scalar_register);

    bool try_get_numeric_output_ref(unsigned int output_id, NumericValueRef & out) const;

    const std::vector<double> * try_get_numeric_output_array_values(unsigned int output_id, bool previous = false) const;

    bool try_get_numeric_scalar_output(unsigned int output_id, bool previous, double & out) const;

    void capture_numeric_prev_array_outputs();

    void capture_numeric_scalar_outputs(
      const CompiledProgram & compiled_program,
      const std::vector<NumericOutputInfo> & output_info,
      const std::vector<double> & temps,
      std::size_t start_output_id = 0,
      std::size_t output_count = std::numeric_limits<std::size_t>::max());

    static bool value_to_scalar_double(const Value & value, double & out);

    bool add_array_value_to_jit_table(const Value & value, uint32_t & out_slot);

    bool add_array_values_to_jit_table(const std::vector<Value> & values, uint32_t & out_slot);

    bool add_matrix_values_to_jit_table(const Value & value, uint32_t & out_slot);

    uint32_t allocate_array_slot_with_size(std::size_t size);

    static bool add_array_values_to_jit_table(
      std::vector<std::vector<double>> & array_storage,
      const std::vector<Value> & values,
      uint32_t & out_slot);

    static bool add_array_value_to_jit_table(
      std::vector<std::vector<double>> & array_storage,
      const Value & value,
      uint32_t & out_slot);

    static bool add_matrix_values_to_jit_table(
      std::vector<std::vector<double>> & array_storage,
      const Value & value,
      uint32_t & out_slot);

    static uint32_t allocate_array_slot_with_size(
      std::vector<std::vector<double>> & array_storage,
      std::size_t size);

    bool configure_numeric_inputs_for_jit(const std::vector<Value> & current_inputs);

    bool numeric_input_layout_matches(const std::vector<Value> & current_inputs) const;

    bool sync_numeric_inputs_from_values();

    bool try_set_direct_numeric_inputs(
      const NumericJitState & state,
      const CompiledProgram & compiled_program);

    void ensure_value_registers_current();

    void ensure_value_delay_states_current();

    bool try_get_numeric_delay_scalar(unsigned int delay_id, double & out) const;

    const std::vector<double> * try_get_numeric_delay_array(unsigned int delay_id) const;

    bool update_numeric_delay_states_from_outputs(
      const NumericJitState & state,
      const CompiledProgram & compiled_program,
      std::size_t start_output_id = 0);

    bool prepare_numeric_jit_program(
      const CompiledProgram & source_program,
      unsigned int base_input_count,
      PreparedNumericJitProgram & prepared) const;

    std::vector<Value> build_numeric_jit_inputs(
      const std::vector<Value> & current_inputs,
      const PreparedNumericJitProgram & prepared) const;

    bool configure_numeric_inputs_for_jit(
      NumericJitState & state,
      const std::vector<Value> & current_inputs);

    bool numeric_input_layout_matches(
      const NumericJitState & state,
      const std::vector<Value> & current_inputs) const;

    bool sync_numeric_inputs_from_values(
      NumericJitState & state,
      const std::vector<Value> & current_inputs);

    bool sync_numeric_register_arrays_from_values(NumericJitState & state);

    bool build_numeric_program(
      const std::vector<Value> & current_inputs,
      egress_jit::NumericProgram & numeric_program);

    bool build_numeric_program(
      const CompiledProgram & compiled_program,
      NumericJitState & state,
      const std::vector<Value> & current_inputs,
      egress_jit::NumericProgram & numeric_program);

    void initialize_numeric_jit_state(
      NumericJitState & state,
      const CompiledProgram & source_program,
      const std::vector<Value> & current_inputs,
      unsigned int base_input_count,
      const std::string & symbol_prefix);

    void ensure_numeric_jit_state_current(
      NumericJitState & state,
      const CompiledProgram & source_program,
      const std::vector<Value> & current_inputs,
      unsigned int base_input_count,
      const std::string & symbol_prefix);

    bool run_numeric_jit_state(
      NumericJitState & state,
      const std::vector<Value> & current_inputs);

    bool prepare_composite_body_jit_program(CompositeBodyJitState & state) const;

    void initialize_composite_body_jit(const std::vector<Value> & current_inputs);

    void ensure_composite_body_jit_current();

    bool run_composite_body_jit(const std::vector<bool> * output_materialize_mask);

    void materialize_numeric_outputs(
      const NumericJitState & state,
      const CompiledProgram & compiled_program,
      std::vector<Value> & destinations,
      const std::vector<bool> * materialize_mask = nullptr);

    void materialize_numeric_outputs_range(
      const NumericJitState & state,
      const CompiledProgram & compiled_program,
      std::size_t start_output_id,
      std::size_t output_count,
      std::vector<Value> & destinations,
      const std::vector<bool> * materialize_mask = nullptr);

    void apply_numeric_register_targets(
      NumericJitState & state,
      const CompiledProgram & compiled_program);

    void sync_value_registers_from_numeric_state(const NumericJitState & state);

    void initialize_numeric_jit(const std::vector<Value> & current_inputs);

    void ensure_numeric_jit_current();

  #endif
};
