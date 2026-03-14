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
