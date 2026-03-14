#pragma once

#include "graph/GraphTypes.hpp"
#include "graph/ModuleNumericJit.hpp"
#include "graph/ModuleProgram.hpp"
#include "expr/ExprStructural.hpp"
#include "jit/OrcJitEngine.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
      uint64_t composite_update_count = 0;
      uint64_t nested_module_count = 0;
      std::string jit_status;
    };
#endif

    struct RegisterArraySpec
    {
      bool enabled = false;
      unsigned int source_input_id = 0;
      Value init_value;
    };

    struct CompositeUpdateSpec
    {
      std::string label;
      std::vector<ExprSpecPtr> register_exprs;
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
      std::vector<CompositeUpdateSpec> composite_update_specs;
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
      std::vector<CompositeUpdateSpec> composite_update_specs,
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

      program_ = compile_program(output_exprs, register_exprs);
      for (const auto & composite_update_spec : composite_update_specs)
      {
        composite_update_programs_.push_back(compile_program({}, composite_update_spec.register_exprs));
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
          std::move(nested_module_spec.composite_update_specs),
          std::move(nested_module_spec.nested_module_specs),
          std::move(nested_module_spec.composite_schedule),
          nested_module_spec.output_boundary_id,
          nested_module_spec.sample_rate);
        nested_module_lookup_[runtime.node_id] = nested_modules_.size();
        nested_modules_.push_back(std::move(runtime));
      }
      has_composite_updates_ = !composite_update_programs_.empty();
      has_nested_modules_ = !nested_modules_.empty();
      if (has_composite_updates_ || has_nested_modules_)
      {
        composite_output_program_ = compile_program(output_exprs, {});
        composite_register_program_ = compile_program({}, register_exprs);
      }
      const uint32_t temp_register_count = std::max(
        program_.register_count,
        std::max(composite_output_program_.register_count, composite_register_program_.register_count));
      temps_.assign(temp_register_count, expr::float_value(0.0));

#ifdef EGRESS_LLVM_ORC_JIT
      if (!has_dynamic_registers_ && !has_composite_updates_ && !has_nested_modules_)
      {
        initialize_numeric_jit(inputs);
      }
      else if (has_composite_updates_ || has_nested_modules_)
      {
        jit_status_ = "numeric JIT disabled for composite runtime";
      }
#endif
    }

    void process(const std::vector<bool> * output_materialize_mask = nullptr);

    unsigned int input_count() const;

    unsigned int output_count() const;

    unsigned int register_count() const;

  #ifdef EGRESS_PROFILE
    CompileStats compile_stats() const;
  #endif

  protected:
    void reset_inputs_after_process();

    void postprocess();

    std::vector<Value> inputs;
    std::vector<Value> outputs;
    std::vector<Value> prev_outputs;

  private:
    friend class Graph;
    using Instr = egress_module_detail::Instr;
    using CompiledProgram = egress_module_detail::CompiledProgram;
    using NumericInputInfo = egress_module_detail::NumericInputInfo;
    using NumericOutputInfo = egress_module_detail::NumericOutputInfo;
    using NumericValueKind = egress_module_detail::NumericValueKind;
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

    void eval_program(const CompiledProgram & expr, std::vector<Value> & temps) const;

    unsigned int input_count_ = 0;
    CompiledProgram program_;
    CompiledProgram composite_output_program_;
    CompiledProgram composite_register_program_;
    std::vector<Value> temps_;
    std::vector<Value> registers_;
    std::vector<Value> next_registers_;
    std::vector<CompiledProgram> composite_update_programs_;
    std::vector<NestedModuleRuntime> nested_modules_;
    std::unordered_map<uint32_t, std::size_t> nested_module_lookup_;
    std::vector<uint32_t> composite_schedule_;
    uint32_t composite_output_boundary_id_ = std::numeric_limits<uint32_t>::max();
    std::vector<RegisterArraySpec> register_array_specs_;
    bool has_dynamic_registers_ = false;
    bool has_composite_updates_ = false;
    bool has_nested_modules_ = false;
    double sample_rate_ = 44100.0;
    uint64_t sample_index_ = 0;
#ifdef EGRESS_LLVM_ORC_JIT
    egress_jit::NumericKernelFn jit_kernel_ = nullptr;
    std::vector<double> numeric_inputs_;
    std::vector<double> numeric_temps_;
    std::vector<double> numeric_registers_;
    std::vector<double> numeric_next_registers_;
    std::vector<std::vector<double>> numeric_array_storage_;
    std::vector<double *> numeric_array_ptrs_;
    std::vector<uint64_t> numeric_array_sizes_;
    std::vector<bool> register_scalar_mask_;
    std::vector<uint32_t> register_array_slot_;
    std::vector<int32_t> array_register_targets_;
    std::vector<bool> array_register_can_swap_;
    std::vector<NumericInputInfo> numeric_input_info_;
    std::vector<NumericOutputInfo> numeric_output_info_;
    std::string jit_status_;
  #ifdef EGRESS_PROFILE
    uint64_t numeric_jit_instruction_count_ = 0;
  #endif

    bool supports_numeric_jit_expr_kind(ExprKind kind) const;

    static void assign_scalar_numeric_value(Value & dst, double value);

    static void assign_numeric_value_to(
      Value & dst,
      const NumericOutputInfo & info,
      uint32_t scalar_register,
      const std::vector<double> & numeric_temps,
      const std::vector<std::vector<double>> & numeric_array_storage);

    static bool value_to_scalar_double(const Value & value, double & out);

    bool add_array_values_to_jit_table(const std::vector<Value> & values, uint32_t & out_slot);

    bool add_matrix_values_to_jit_table(const Value & value, uint32_t & out_slot);

    uint32_t allocate_array_slot_with_size(std::size_t size);

    bool configure_numeric_inputs_for_jit(const std::vector<Value> & current_inputs);

    bool numeric_input_layout_matches(const std::vector<Value> & current_inputs) const;

    bool sync_numeric_inputs_from_values();

    bool build_numeric_program(const std::vector<Value> & current_inputs, egress_jit::NumericProgram & numeric_program);

    void initialize_numeric_jit(const std::vector<Value> & current_inputs);

    void ensure_numeric_jit_current();

  #endif
};
