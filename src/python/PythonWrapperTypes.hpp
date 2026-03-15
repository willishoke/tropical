#pragma once

class PythonGraph
{
  public:
    explicit PythonGraph(unsigned int buffer_length) : graph_(buffer_length) {}

    bool destroy_module(const std::string & module_name)
    {
      return graph_.remove_module(module_name);
    }

    bool connect(
      const std::string & from_module,
      unsigned int from_output_id,
      const std::string & to_module,
      unsigned int to_input_id)
    {
      return graph_.connect(from_module, from_output_id, to_module, to_input_id);
    }

    bool disconnect(
      const std::string & from_module,
      unsigned int from_output_id,
      const std::string & to_module,
      unsigned int to_input_id)
    {
      return graph_.remove_connection(from_module, from_output_id, to_module, to_input_id);
    }

    bool add_output(const std::string & module_name, unsigned int output_id)
    {
      return graph_.addOutput(std::make_pair(module_name, output_id));
    }

    bool add_output_expr(const expr::ExprSpecPtr & spec)
    {
      return graph_.addOutputExpr(spec);
    }

    std::size_t add_output_tap(const std::string & module_name, unsigned int output_id)
    {
      return graph_.addOutputTap(module_name, output_id);
    }

    bool remove_output_tap(std::size_t tap_id)
    {
      return graph_.removeOutputTap(tap_id);
    }

    std::vector<double> output_tap_buffer(std::size_t tap_id) const
    {
      return graph_.outputTapBuffer(tap_id);
    }

    void process()
    {
      graph_.process();
    }

    void prime_numeric_jit()
    {
      graph_.prime_numeric_jit();
    }

    void set_worker_count(unsigned int worker_count)
    {
      graph_.set_worker_count(worker_count);
    }

    void set_fusion_enabled(bool enabled)
    {
      graph_.set_fusion_enabled(enabled);
    }

    bool fusion_enabled() const
    {
      return graph_.fusion_enabled();
    }

    unsigned int worker_count() const
    {
      return graph_.worker_count();
    }

    std::vector<double> output_buffer() const
    {
      return graph_.outputBuffer;
    }

    Graph & graph()
    {
      return graph_;
    }

    py::dict profile_stats() const
    {
      const auto stats = graph_.profile_stats();
      py::dict result;
      result["enabled"] = stats.enabled;
      result["callback_count"] = stats.callback_count;
      result["avg_callback_ms"] = stats.avg_callback_ms;
      result["max_callback_ms"] = stats.max_callback_ms;
      result["primitive_body_available"] = stats.primitive_body_available;
      result["primitive_body_covers_all_modules"] = stats.primitive_body_covers_all_modules;
      result["input_kernel_available"] = stats.input_kernel_available;
      result["fused_input_use_count"] = stats.fused_input_use_count;
      result["fused_body_use_count"] = stats.fused_body_use_count;
      result["fusion_candidate_reason"] = stats.fusion_candidate_reason;
      result["primitive_body_status"] = stats.primitive_body_status;
      result["input_kernel_status"] = stats.input_kernel_status;

      py::list modules;
      for (const auto & module : stats.modules)
      {
        py::dict row;
        row["module_name"] = module.module_name;
        row["call_count"] = module.call_count;
        row["avg_call_ms"] = module.avg_call_ms;
        row["max_call_ms"] = module.max_call_ms;
        modules.append(std::move(row));
      }

      py::list sorted_modules;
      std::vector<py::dict> module_rows;
      module_rows.reserve(py::len(modules));
      for (const auto & item : modules)
      {
        module_rows.push_back(py::reinterpret_borrow<py::dict>(item));
      }
      std::sort(
        module_rows.begin(),
        module_rows.end(),
        [](const py::dict & a, const py::dict & b)
        {
          return py::float_(a["max_call_ms"]).cast<double>() > py::float_(b["max_call_ms"]).cast<double>();
        });
      for (auto & row : module_rows)
      {
        sorted_modules.append(std::move(row));
      }

      result["modules"] = std::move(sorted_modules);
#ifdef EGRESS_PROFILE
      py::dict boxing;
      py::dict fused_current_output_sync;
      fused_current_output_sync["call_count"] = stats.fused_current_output_sync.call_count;
      fused_current_output_sync["total_ms"] = stats.fused_current_output_sync.total_ms;
      fused_current_output_sync["max_ms"] = stats.fused_current_output_sync.max_ms;
      fused_current_output_sync["output_copy_count"] = stats.fused_current_output_sync.output_copy_count;
      boxing["fused_current_output_sync"] = std::move(fused_current_output_sync);

      py::dict fused_prev_output_sync;
      fused_prev_output_sync["call_count"] = stats.fused_prev_output_sync.call_count;
      fused_prev_output_sync["total_ms"] = stats.fused_prev_output_sync.total_ms;
      fused_prev_output_sync["max_ms"] = stats.fused_prev_output_sync.max_ms;
      fused_prev_output_sync["output_copy_count"] = stats.fused_prev_output_sync.output_copy_count;
      boxing["fused_prev_output_sync"] = std::move(fused_prev_output_sync);
      result["boxing"] = std::move(boxing);
#endif
      return result;
    }

    void reset_profile_stats()
    {
      graph_.reset_profile_stats();
    }

    std::string next_name(const std::string & prefix)
    {
      const auto id = ++name_counters_[prefix];
      return prefix + std::to_string(id);
    }

    bool add_module(
      const std::string & module_name,
      unsigned int input_count,
      std::vector<expr::ExprSpecPtr> output_exprs,
      std::vector<expr::ExprSpecPtr> register_exprs,
      std::vector<expr::Value> initial_registers,
      std::vector<Module::RegisterArraySpec> register_array_specs,
      std::vector<Module::DelayStateSpec> delay_state_specs,
      std::vector<Module::NestedModuleSpec> nested_module_specs,
      std::vector<uint32_t> composite_schedule,
      uint32_t output_boundary_id,
      double sample_rate)
    {
      return graph_.addModule(
        module_name,
        std::make_unique<Module>(
          input_count,
          std::move(output_exprs),
          std::move(register_exprs),
          std::move(initial_registers),
          std::move(register_array_specs),
          std::move(delay_state_specs),
          std::move(nested_module_specs),
          std::move(composite_schedule),
          output_boundary_id,
          sample_rate));
    }

  private:
    Graph graph_;
    std::unordered_map<std::string, uint64_t> name_counters_;
};

static PythonGraph & default_graph()
{
  static PythonGraph graph(1024);
  return graph;
}

struct OutputPort
{
  PythonGraph * graph = nullptr;
  std::string module_name;
  unsigned int output_id = 0;
};

struct InputPort
{
  PythonGraph * graph = nullptr;
  std::string module_name;
  unsigned int input_id = 0;
};

struct SignalExpr
{
  PythonGraph * graph = nullptr;
  expr::ExprSpecPtr spec;
  std::vector<egress_composition::PortRef> sources;
};

static void assign_input(const InputPort & input, const py::handle & value);
