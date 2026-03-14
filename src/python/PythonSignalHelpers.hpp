#pragma once

static PythonGraph * merge_graphs(PythonGraph * lhs, PythonGraph * rhs);
static SignalExpr make_signal_expr(
  PythonGraph * graph,
  expr::ExprSpecPtr spec,
  std::vector<egress_composition::PortRef> sources = {});
static SignalExpr coerce_expr(const py::handle & value);
static bool is_matrix_literal(const py::handle & value);
static expr::Value matrix_value_from_py(const py::handle & value);
static expr::Value value_from_py(const py::handle & value);

struct SymbolMap
{
  enum class Kind
  {
    Input,
    Register
  };

  Kind kind = Kind::Input;
  std::unordered_map<std::string, unsigned int> slots;
};

struct ModuleDefinition
{
  std::string type_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> register_names;
  std::vector<expr::Value> initial_registers;
  std::vector<expr::ExprSpecPtr> input_defaults;
  std::vector<expr::ExprSpecPtr> output_exprs;
  std::vector<expr::ExprSpecPtr> register_exprs;
  std::vector<Module::CompositeUpdateSpec> composite_update_specs;
  std::vector<Module::NestedModuleSpec> nested_module_specs;
  std::vector<Module::RegisterArraySpec> register_array_specs;
  std::shared_ptr<egress_composition::CompositeModuleSpec> composite_spec;
  std::shared_ptr<egress_composition::LoweredCompositeModule> lowered_composite;
  std::vector<uint32_t> composite_schedule;
  uint32_t output_boundary_id = std::numeric_limits<uint32_t>::max();
  double sample_rate = 44100.0;
};

struct PureFunctionDefinition
{
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<expr::ExprSpecPtr> output_exprs;
};

struct DefinitionBuildContext
{
  struct PendingCompositeUpdateSpec
  {
    std::string label;
    std::vector<std::pair<uint32_t, expr::ExprSpecPtr>> register_updates;
  };

  std::vector<std::string> register_names;
  std::vector<expr::Value> initial_registers;
  std::vector<expr::ExprSpecPtr> register_exprs;
  std::vector<PendingCompositeUpdateSpec> composite_update_specs;
  std::vector<Module::NestedModuleSpec> nested_module_specs;
  std::vector<Module::RegisterArraySpec> register_array_specs;
  std::shared_ptr<egress_composition::CompositeModuleSpec> composite_spec;
  uint32_t input_boundary_id = 0;
  uint32_t output_boundary_id = 0;
  double sample_rate = 44100.0;
  std::size_t hidden_counter = 0;

  unsigned int append_register(
    std::string name,
    expr::Value initial_value,
    Module::RegisterArraySpec array_spec)
  {
    const unsigned int slot = static_cast<unsigned int>(register_names.size());
    register_names.push_back(std::move(name));
    initial_registers.push_back(std::move(initial_value));
    register_exprs.push_back(nullptr);
    register_array_specs.push_back(std::move(array_spec));
    return slot;
  }

  unsigned int append_hidden_register(
    const std::string & base_name,
    const expr::Value & initial_value,
    const Module::RegisterArraySpec & array_spec)
  {
    const std::string hidden_name = "__stateful_" + std::to_string(hidden_counter++) + "_" + base_name;
    return append_register(hidden_name, initial_value, array_spec);
  }

  unsigned int append_hidden_delay_register(const expr::Value & initial_value)
  {
    return append_register(
      "__delay_" + std::to_string(hidden_counter++),
      initial_value,
      Module::RegisterArraySpec{});
  }

  void add_composite_update(
    std::string label,
    std::vector<std::pair<uint32_t, expr::ExprSpecPtr>> register_updates)
  {
    composite_update_specs.push_back(PendingCompositeUpdateSpec{
      std::move(label),
      std::move(register_updates)});
  }

  void initialize_composite_spec(const std::string & label, uint32_t input_count, uint32_t output_count)
  {
    composite_spec = std::make_shared<egress_composition::CompositeModuleSpec>();
    input_boundary_id = composite_spec->add_node(
      egress_composition::NodeKind::InputBoundary,
      label + ":inputs",
      0,
      input_count);
    output_boundary_id = composite_spec->add_node(
      egress_composition::NodeKind::OutputBoundary,
      label + ":outputs",
      output_count,
      0);
    composite_spec->input_boundary_id = input_boundary_id;
    composite_spec->output_boundary_id = output_boundary_id;
  }
};

static thread_local DefinitionBuildContext * current_definition_context_ = nullptr;

class ScopedDefinitionBuildContext
{
  public:
    explicit ScopedDefinitionBuildContext(DefinitionBuildContext * context)
      : previous_(current_definition_context_)
    {
      current_definition_context_ = context;
    }

    ~ScopedDefinitionBuildContext()
    {
      current_definition_context_ = previous_;
    }

  private:
    DefinitionBuildContext * previous_ = nullptr;
};

static expr::ExprSpecPtr clone_with_input_and_register_subst(
  const expr::ExprSpecPtr & expr,
  const std::vector<expr::ExprSpecPtr> & input_args,
  const std::vector<unsigned int> & register_slots)
{
  if (!expr)
  {
    return nullptr;
  }

  switch (expr->kind)
  {
    case ExprKind::Literal:
    case ExprKind::Ref:
    case ExprKind::NestedValue:
    case ExprKind::SampleRate:
    case ExprKind::SampleIndex:
      return expr;
    case ExprKind::InputValue:
      if (expr->slot_id >= input_args.size())
      {
        throw std::invalid_argument("Stateful function argument index out of range.");
      }
      return input_args[expr->slot_id];
    case ExprKind::RegisterValue:
      if (expr->slot_id >= register_slots.size())
      {
        throw std::invalid_argument("Stateful function register index out of range.");
      }
      return expr::register_value_expr(register_slots[expr->slot_id]);
    case ExprKind::Function:
      return expr::function_expr(
        expr->param_count,
        clone_with_input_and_register_subst(expr->lhs, input_args, register_slots));
    case ExprKind::Call:
    {
      std::vector<expr::ExprSpecPtr> args;
      args.reserve(expr->args.size());
      for (const auto & arg : expr->args)
      {
        args.push_back(clone_with_input_and_register_subst(arg, input_args, register_slots));
      }
      return expr::call_expr(
        clone_with_input_and_register_subst(expr->lhs, input_args, register_slots),
        std::move(args));
    }
    case ExprKind::ArrayPack:
    {
      std::vector<expr::ExprSpecPtr> items;
      items.reserve(expr->args.size());
      for (const auto & arg : expr->args)
      {
        items.push_back(clone_with_input_and_register_subst(arg, input_args, register_slots));
      }
      return expr::array_pack_expr(std::move(items));
    }
    case ExprKind::Clamp:
      return expr::clamp_expr(
        clone_with_input_and_register_subst(expr->lhs, input_args, register_slots),
        clone_with_input_and_register_subst(expr->rhs, input_args, register_slots),
        clone_with_input_and_register_subst(expr->args.empty() ? nullptr : expr->args.front(), input_args, register_slots));
    case ExprKind::ArraySet:
      return expr::array_set_expr(
        clone_with_input_and_register_subst(expr->lhs, input_args, register_slots),
        clone_with_input_and_register_subst(expr->rhs, input_args, register_slots),
        clone_with_input_and_register_subst(expr->args.empty() ? nullptr : expr->args.front(), input_args, register_slots));
    case ExprKind::Index:
      return expr::index_expr(
        clone_with_input_and_register_subst(expr->lhs, input_args, register_slots),
        clone_with_input_and_register_subst(expr->rhs, input_args, register_slots));
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
    return expr::unary_expr(
      expr->kind,
      clone_with_input_and_register_subst(expr->lhs, input_args, register_slots));
  }

  return expr::binary_expr(
    expr->kind,
    clone_with_input_and_register_subst(expr->lhs, input_args, register_slots),
    clone_with_input_and_register_subst(expr->rhs, input_args, register_slots));
}

static std::vector<egress_composition::PortRef> dedupe_sources(
  std::vector<egress_composition::PortRef> sources)
{
  std::vector<egress_composition::PortRef> unique;
  unique.reserve(sources.size());
  for (const auto & source : sources)
  {
    const auto it = std::find(unique.begin(), unique.end(), source);
    if (it == unique.end())
    {
      unique.push_back(source);
    }
  }
  return unique;
}

static std::vector<egress_composition::PortRef> merge_sources(
  const std::vector<egress_composition::PortRef> & lhs,
  const std::vector<egress_composition::PortRef> & rhs)
{
  std::vector<egress_composition::PortRef> merged = lhs;
  merged.insert(merged.end(), rhs.begin(), rhs.end());
  return dedupe_sources(std::move(merged));
}

static std::vector<Module::CompositeUpdateSpec> finalize_composite_updates(
  const DefinitionBuildContext & build_context)
{
  std::vector<Module::CompositeUpdateSpec> finalized;
  finalized.reserve(build_context.composite_update_specs.size());
  for (const auto & pending : build_context.composite_update_specs)
  {
    Module::CompositeUpdateSpec spec;
    spec.label = pending.label;
    spec.register_exprs.assign(build_context.register_names.size(), nullptr);
    for (const auto & update : pending.register_updates)
    {
      if (update.first >= spec.register_exprs.size())
      {
        throw std::invalid_argument("Composite update register slot out of range.");
      }
      spec.register_exprs[update.first] = update.second;
    }
    finalized.push_back(std::move(spec));
  }
  return finalized;
}

static std::vector<Module::NestedModuleSpec> finalize_nested_modules(
  const DefinitionBuildContext & build_context)
{
  return build_context.nested_module_specs;
}

static uint32_t add_composite_call_node(
  const std::string & label,
  egress_composition::NodeKind kind,
  uint32_t input_count,
  uint32_t output_count,
  const std::vector<expr::ExprSpecPtr> & call_args,
  const std::vector<std::vector<egress_composition::PortRef>> * arg_sources = nullptr)
{
  if (current_definition_context_ == nullptr || !current_definition_context_->composite_spec)
  {
    return std::numeric_limits<uint32_t>::max();
  }

  const uint32_t node_id = current_definition_context_->composite_spec->add_node(
    kind,
    label,
    input_count,
    output_count);

  if (!arg_sources)
  {
    return node_id;
  }

  for (std::size_t input_id = 0; input_id < arg_sources->size() && input_id < call_args.size(); ++input_id)
  {
    for (const auto & source : (*arg_sources)[input_id])
    {
      current_definition_context_->composite_spec->add_edge(
        source,
        egress_composition::PortRef{node_id, static_cast<uint32_t>(input_id)},
        egress_composition::ConnectionTiming::SameTick);
    }
  }

  return node_id;
}

static std::vector<SignalExpr> require_local_call_args(
  const py::args & args,
  const char * label)
{
  std::vector<SignalExpr> call_args;
  call_args.reserve(args.size());
  for (const py::handle & arg : args)
  {
    const SignalExpr signal = coerce_expr(arg);
    if (signal.graph != nullptr)
    {
      throw std::invalid_argument(std::string(label) + " arguments cannot capture graph ports.");
    }
    call_args.push_back(signal);
  }
  return call_args;
}

class PyModuleInstance
{
  public:
    PyModuleInstance(std::shared_ptr<ModuleDefinition> definition, PythonGraph & graph)
      : definition_(std::move(definition)), graph_(&graph), name_(graph_->next_name(definition_->type_name))
    {
      if (!graph_->add_module(
            name_,
            static_cast<unsigned int>(definition_->input_names.size()),
            definition_->output_exprs,
            definition_->register_exprs,
            definition_->initial_registers,
            definition_->register_array_specs,
            definition_->composite_update_specs,
            definition_->nested_module_specs,
            definition_->composite_schedule,
            definition_->output_boundary_id,
            definition_->sample_rate))
      {
        throw std::invalid_argument("Failed to create module '" + name_ + "'.");
      }

      for (std::size_t i = 0; i < definition_->input_defaults.size(); ++i)
      {
        if (!definition_->input_defaults[i])
        {
          continue;
        }
        if (!graph_->graph().set_input_expr(
              name_,
              static_cast<unsigned int>(i),
              definition_->input_defaults[i]))
        {
          throw std::invalid_argument("Failed to set default input '" + definition_->input_names[i] + "'.");
        }
      }

      graph_->graph().prime_module_inputs_if_local(name_);
    }

    const std::string & name() const { return name_; }

    InputPort get_input(const std::string & attr) const
    {
      return InputPort{graph_, name_, lookup(definition_->input_names, attr, "input")};
    }

    OutputPort get_output(const std::string & attr) const
    {
      return OutputPort{graph_, name_, lookup(definition_->output_names, attr, "output")};
    }

    py::object getattr(const std::string & attr) const
    {
      for (unsigned int i = 0; i < definition_->input_names.size(); ++i)
      {
        if (definition_->input_names[i] == attr)
        {
          return py::cast(InputPort{graph_, name_, i});
        }
      }

      for (unsigned int i = 0; i < definition_->output_names.size(); ++i)
      {
        if (definition_->output_names[i] == attr)
        {
          return py::cast(OutputPort{graph_, name_, i});
        }
      }

      throw py::attribute_error("Unknown attribute '" + attr + "'.");
    }

    void setattr(const std::string & attr, const py::object & value) const
    {
      for (unsigned int i = 0; i < definition_->input_names.size(); ++i)
      {
        if (definition_->input_names[i] == attr)
        {
          assign_input(InputPort{graph_, name_, i}, value);
          return;
        }
      }

      throw py::attribute_error("Cannot assign attribute '" + attr + "'.");
    }

    py::list dir() const
    {
      std::unordered_set<std::string> names;
      names.insert("name");
#ifdef EGRESS_PROFILE
      names.insert("compile_stats");
#endif

      for (const auto & input_name : definition_->input_names)
      {
        names.insert(input_name);
      }
      for (const auto & output_name : definition_->output_names)
      {
        names.insert(output_name);
      }

      std::vector<std::string> sorted(names.begin(), names.end());
      std::sort(sorted.begin(), sorted.end());

      py::list result;
      for (const auto & name : sorted)
      {
        result.append(name);
      }
      return result;
    }

#ifdef EGRESS_PROFILE
    py::dict compile_stats() const
    {
      const auto stats = graph_->graph().module_compile_stats(name_);
      if (!stats.found)
      {
        throw py::key_error("Unknown module '" + name_ + "'.");
      }

      py::dict result;
      result["instruction_count"] = stats.instruction_count;
      result["register_count"] = stats.register_count;
      result["numeric_jit_instruction_count"] = stats.numeric_jit_instruction_count;
      result["composite_update_count"] = stats.composite_update_count;
      result["nested_module_count"] = stats.nested_module_count;
      result["jit_status"] = stats.jit_status;
      return result;
    }
#endif

  private:
    static unsigned int lookup(
      const std::vector<std::string> & names,
      const std::string & attr,
      const char * kind)
    {
      for (unsigned int i = 0; i < names.size(); ++i)
      {
        if (names[i] == attr)
        {
          return i;
        }
      }

      throw std::invalid_argument("Unknown " + std::string(kind) + " '" + attr + "'.");
    }

    std::shared_ptr<ModuleDefinition> definition_;
    PythonGraph * graph_ = nullptr;
    std::string name_;
};

class PyModuleType
{
  public:
    explicit PyModuleType(std::shared_ptr<ModuleDefinition> definition)
      : definition_(std::move(definition))
    {
    }

    const std::string & name() const { return definition_->type_name; }

    py::dict composition_stats() const
    {
      py::dict result;
      if (!definition_->composite_spec)
      {
        result["node_count"] = 0;
        result["edge_count"] = 0;
        result["same_tick_edge_count"] = 0;
        result["delayed_edge_count"] = 0;
        return result;
      }

      uint64_t same_tick_edges = 0;
      uint64_t delayed_edges = 0;
      for (const auto & edge : definition_->composite_spec->edges)
      {
        if (edge.timing == egress_composition::ConnectionTiming::SameTick)
        {
          ++same_tick_edges;
        }
        else
        {
          ++delayed_edges;
        }
      }

      result["node_count"] = static_cast<uint64_t>(definition_->composite_spec->nodes.size());
      result["edge_count"] = static_cast<uint64_t>(definition_->composite_spec->edges.size());
      result["same_tick_edge_count"] = same_tick_edges;
      result["delayed_edge_count"] = delayed_edges;
      result["nested_module_count"] = static_cast<uint64_t>(definition_->nested_module_specs.size());
      result["same_tick_schedule_size"] = definition_->lowered_composite != nullptr
                                            ? static_cast<uint64_t>(definition_->lowered_composite->same_tick_schedule.size())
                                            : uint64_t(0);
      result["delayed_node_count"] = definition_->lowered_composite != nullptr
                                       ? static_cast<uint64_t>(definition_->lowered_composite->delayed_node_count)
                                       : uint64_t(0);
      result["scheduled_node_count"] = definition_->lowered_composite != nullptr
                                         ? static_cast<uint64_t>(definition_->lowered_composite->scheduled_nodes.size())
                                         : uint64_t(0);
      result["delayed_state_count"] = definition_->lowered_composite != nullptr
                                        ? static_cast<uint64_t>(definition_->lowered_composite->delayed_edges.size())
                                        : uint64_t(0);
      return result;
    }

    PyModuleInstance instantiate() const
    {
      return PyModuleInstance(definition_, default_graph());
    }

    py::object call(const py::args & args) const
    {
      if (current_definition_context_ == nullptr)
      {
        if (!args.empty())
        {
          throw std::invalid_argument(
            "Module types only accept signal arguments inside define_module process bodies.");
        }
        return py::cast(instantiate());
      }

      if (args.size() > definition_->input_names.size())
      {
        throw std::invalid_argument(
          "Module call expects at most " + std::to_string(definition_->input_names.size()) + " arguments.");
      }

      std::vector<SignalExpr> call_args = require_local_call_args(args, "Module call");
      for (std::size_t i = call_args.size(); i < definition_->input_names.size(); ++i)
      {
        const expr::ExprSpecPtr & default_expr = definition_->input_defaults[i];
        if (!default_expr)
        {
          throw std::invalid_argument(
            "Missing argument for module input '" + definition_->input_names[i] + "'.");
        }
        call_args.push_back(make_signal_expr(nullptr, default_expr));
      }

      std::vector<std::vector<egress_composition::PortRef>> arg_sources;
      std::vector<expr::ExprSpecPtr> arg_specs;
      arg_specs.reserve(call_args.size());
      arg_sources.reserve(call_args.size());
      for (const auto & arg : call_args)
      {
        arg_specs.push_back(arg.spec);
        arg_sources.push_back(arg.sources);
      }

      const uint32_t call_node_id = add_composite_call_node(
        definition_->type_name,
        egress_composition::NodeKind::ModuleCall,
        static_cast<uint32_t>(definition_->input_names.size()),
        static_cast<uint32_t>(definition_->output_names.size()),
        arg_specs,
        &arg_sources);

      Module::NestedModuleSpec nested_spec;
      nested_spec.node_id = call_node_id;
      nested_spec.label = definition_->type_name;
      nested_spec.input_count = static_cast<unsigned int>(definition_->input_names.size());
      nested_spec.input_exprs = arg_specs;
      nested_spec.output_exprs = definition_->output_exprs;
      nested_spec.register_exprs = definition_->register_exprs;
      nested_spec.initial_registers = definition_->initial_registers;
      nested_spec.register_array_specs = definition_->register_array_specs;
      nested_spec.composite_update_specs = definition_->composite_update_specs;
      nested_spec.nested_module_specs = definition_->nested_module_specs;
      nested_spec.composite_schedule = definition_->composite_schedule;
      nested_spec.output_boundary_id = definition_->output_boundary_id;
      nested_spec.sample_rate = definition_->sample_rate;
      current_definition_context_->nested_module_specs.push_back(std::move(nested_spec));

      if (definition_->output_names.size() == 1)
      {
        return py::cast(make_signal_expr(
          nullptr,
          expr::nested_value_expr(call_node_id, 0),
          call_node_id == std::numeric_limits<uint32_t>::max()
            ? std::vector<egress_composition::PortRef>{}
            : std::vector<egress_composition::PortRef>{egress_composition::PortRef{call_node_id, 0}}));
      }

      py::tuple outputs(definition_->output_names.size());
      for (std::size_t output_id = 0; output_id < definition_->output_names.size(); ++output_id)
      {
        outputs[output_id] = py::cast(make_signal_expr(
          nullptr,
          expr::nested_value_expr(call_node_id, static_cast<unsigned int>(output_id)),
          call_node_id == std::numeric_limits<uint32_t>::max()
            ? std::vector<egress_composition::PortRef>{}
            : std::vector<egress_composition::PortRef>{
                egress_composition::PortRef{call_node_id, static_cast<uint32_t>(output_id)}}));
      }
      return outputs;
    }

  private:
    std::shared_ptr<ModuleDefinition> definition_;
};

class PyPureFunctionType
{
  public:
    explicit PyPureFunctionType(std::shared_ptr<PureFunctionDefinition> definition)
      : definition_(std::move(definition))
    {
    }

    py::object call(const py::args & args) const
    {
      if (args.size() != definition_->input_names.size())
      {
        throw std::invalid_argument("Pure function expects " + std::to_string(definition_->input_names.size()) +
                                    " arguments.");
      }

      PythonGraph * graph = nullptr;
      std::vector<expr::ExprSpecPtr> call_args;
      call_args.reserve(args.size());

      for (const py::handle & arg : args)
      {
        const SignalExpr expr = coerce_expr(arg);
        graph = merge_graphs(graph, expr.graph);
        call_args.push_back(expr.spec);
      }

      if (definition_->output_exprs.size() == 1)
      {
        auto fn = expr::function_expr(static_cast<unsigned int>(definition_->input_names.size()),
                                      definition_->output_exprs[0]);
        auto call_expr = expr::call_expr(std::move(fn), call_args);
        return py::cast(make_signal_expr(graph, std::move(call_expr)));
      }

      py::tuple outputs(definition_->output_exprs.size());
      for (std::size_t i = 0; i < definition_->output_exprs.size(); ++i)
      {
        auto fn = expr::function_expr(static_cast<unsigned int>(definition_->input_names.size()),
                                      definition_->output_exprs[i]);
        auto call_expr = expr::call_expr(std::move(fn), call_args);
        outputs[i] = py::cast(make_signal_expr(graph, std::move(call_expr)));
      }
      return outputs;
    }

  private:
    std::shared_ptr<PureFunctionDefinition> definition_;
};

static PythonGraph * merge_graphs(PythonGraph * lhs, PythonGraph * rhs)
{
  if (lhs != nullptr && rhs != nullptr && lhs != rhs)
  {
    throw std::invalid_argument("Expression operands belong to different graphs.");
  }
  return lhs != nullptr ? lhs : rhs;
}

static SignalExpr make_signal_expr(
  PythonGraph * graph,
  expr::ExprSpecPtr spec,
  std::vector<egress_composition::PortRef> sources)
{
  SignalExpr expr;
  expr.graph = graph;
  expr.spec = std::move(spec);
  expr.sources = dedupe_sources(std::move(sources));
  return expr;
}

static SignalExpr make_literal_expr(double value)
{
  return make_signal_expr(nullptr, expr::literal_expr(value));
}

static SignalExpr make_literal_expr(int64_t value)
{
  return make_signal_expr(nullptr, expr::literal_expr(value));
}

static SignalExpr make_literal_expr(bool value)
{
  return make_signal_expr(nullptr, expr::literal_expr(value));
}

static SignalExpr make_array_expr(const py::iterable & values);
static SignalExpr make_matrix_expr(const py::handle & value);

static SignalExpr make_output_expr(const OutputPort & out)
{
  return make_signal_expr(out.graph, expr::ref_expr(out.module_name, out.output_id));
}

static SignalExpr current_input_expr(const InputPort & input)
{
  auto spec = input.graph->graph().get_input_expr(input.module_name, input.input_id);
  if (!spec)
  {
    spec = expr::literal_expr(0.0);
  }
  return make_signal_expr(input.graph, std::move(spec));
}

static SignalExpr coerce_expr(const py::handle & value)
{
  if (py::isinstance<SignalExpr>(value))
  {
    return value.cast<SignalExpr>();
  }

  if (py::isinstance<OutputPort>(value))
  {
    return make_output_expr(value.cast<OutputPort>());
  }

  if (py::isinstance<InputPort>(value))
  {
    return current_input_expr(value.cast<InputPort>());
  }

  if (py::isinstance<py::bool_>(value))
  {
    return make_literal_expr(value.cast<bool>());
  }

  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value))
  {
    if (is_matrix_literal(value))
    {
      return make_matrix_expr(value);
    }
    return make_array_expr(value.cast<py::iterable>());
  }

  if (py::isinstance<py::int_>(value))
  {
    return make_literal_expr(value.cast<int64_t>());
  }

  if (py::isinstance<py::float_>(value))
  {
    return make_literal_expr(value.cast<double>());
  }

  throw std::invalid_argument("Expected an output port, input expression, arithmetic expression, or numeric literal.");
}

static SignalExpr make_unary_expr(ExprKind kind, const SignalExpr & operand)
{
  return make_signal_expr(operand.graph, expr::unary_expr(kind, operand.spec), operand.sources);
}

static SignalExpr make_binary_expr(ExprKind kind, const SignalExpr & lhs, const SignalExpr & rhs)
{
  return make_signal_expr(
    merge_graphs(lhs.graph, rhs.graph),
    expr::binary_expr(kind, lhs.spec, rhs.spec),
    merge_sources(lhs.sources, rhs.sources));
}

static SignalExpr add_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Add, lhs, coerce_expr(rhs));
}

static SignalExpr sub_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Sub, lhs, coerce_expr(rhs));
}

static SignalExpr mul_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Mul, lhs, coerce_expr(rhs));
}

static SignalExpr div_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Div, lhs, coerce_expr(rhs));
}

static SignalExpr pow_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Pow, lhs, coerce_expr(rhs));
}

static SignalExpr mod_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Mod, lhs, coerce_expr(rhs));
}

static SignalExpr matmul_expr(const py::handle & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::MatMul, coerce_expr(lhs), coerce_expr(rhs));
}

static SignalExpr floor_div_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::FloorDiv, lhs, coerce_expr(rhs));
}

static SignalExpr bit_and_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::BitAnd, lhs, coerce_expr(rhs));
}

static SignalExpr bit_or_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::BitOr, lhs, coerce_expr(rhs));
}

static SignalExpr bit_xor_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::BitXor, lhs, coerce_expr(rhs));
}

static SignalExpr lshift_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::LShift, lhs, coerce_expr(rhs));
}

static SignalExpr rshift_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::RShift, lhs, coerce_expr(rhs));
}

static SignalExpr sin_expr(const py::handle & value)
{
  return make_unary_expr(ExprKind::Sin, coerce_expr(value));
}

static SignalExpr abs_expr(const py::handle & value)
{
  return make_unary_expr(ExprKind::Abs, coerce_expr(value));
}

static SignalExpr log_expr(const py::handle & value)
{
  return make_unary_expr(ExprKind::Log, coerce_expr(value));
}

static SignalExpr clamp_expr(const py::handle & value, const py::handle & min_value, const py::handle & max_value)
{
  const SignalExpr signal = coerce_expr(value);
  const SignalExpr min_signal = coerce_expr(min_value);
  const SignalExpr max_signal = coerce_expr(max_value);
  return make_signal_expr(
    merge_graphs(merge_graphs(signal.graph, min_signal.graph), max_signal.graph),
    expr::clamp_expr(signal.spec, min_signal.spec, max_signal.spec),
    merge_sources(merge_sources(signal.sources, min_signal.sources), max_signal.sources));
}

static SignalExpr logical_not_expr(const py::handle & value)
{
  return make_unary_expr(ExprKind::Not, coerce_expr(value));
}

static SignalExpr index_expr(const SignalExpr & value, const py::handle & index)
{
  const SignalExpr idx = coerce_expr(index);
  return make_signal_expr(
    merge_graphs(value.graph, idx.graph),
    expr::index_expr(value.spec, idx.spec),
    merge_sources(value.sources, idx.sources));
}

static SignalExpr array_set_expr(const SignalExpr & array, const py::handle & index, const py::handle & value)
{
  const SignalExpr idx = coerce_expr(index);
  const SignalExpr replacement = coerce_expr(value);
  return make_signal_expr(
    merge_graphs(merge_graphs(array.graph, idx.graph), replacement.graph),
    expr::array_set_expr(array.spec, idx.spec, replacement.spec),
    merge_sources(merge_sources(array.sources, idx.sources), replacement.sources));
}

static SignalExpr less_expr(const py::handle & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Less, coerce_expr(lhs), coerce_expr(rhs));
}

static SignalExpr less_equal_expr(const py::handle & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::LessEqual, coerce_expr(lhs), coerce_expr(rhs));
}

static SignalExpr greater_expr(const py::handle & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Greater, coerce_expr(lhs), coerce_expr(rhs));
}

static SignalExpr greater_equal_expr(const py::handle & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::GreaterEqual, coerce_expr(lhs), coerce_expr(rhs));
}

static SignalExpr equal_expr(const py::handle & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Equal, coerce_expr(lhs), coerce_expr(rhs));
}

static SignalExpr not_equal_expr(const py::handle & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::NotEqual, coerce_expr(lhs), coerce_expr(rhs));
}

static SignalExpr symbol_expr(SymbolMap::Kind kind, unsigned int slot_id)
{
  if (kind == SymbolMap::Kind::Input)
  {
    std::vector<egress_composition::PortRef> sources;
    if (current_definition_context_ != nullptr && current_definition_context_->composite_spec)
    {
      sources.push_back(egress_composition::PortRef{
        current_definition_context_->input_boundary_id,
        slot_id});
    }
    return make_signal_expr(nullptr, expr::input_value_expr(slot_id), std::move(sources));
  }
  return make_signal_expr(nullptr, expr::register_value_expr(slot_id));
}

static SignalExpr sample_rate_expr()
{
  return make_signal_expr(nullptr, expr::sample_rate_expr());
}

static SignalExpr sample_index_expr()
{
  return make_signal_expr(nullptr, expr::sample_index_expr());
}

static SignalExpr delay_expr(const py::handle & value, const py::handle & init)
{
  if (current_definition_context_ == nullptr)
  {
    throw std::invalid_argument(
      "eg.delay(...) may only be called inside define_module process bodies.");
  }

  const SignalExpr signal = coerce_expr(value);
  if (signal.graph != nullptr)
  {
    throw std::invalid_argument("eg.delay(...) cannot capture graph ports.");
  }

  const expr::Value initial_value = init.is_none() ? expr::float_value(0.0) : value_from_py(init);
  const unsigned int delay_slot = current_definition_context_->append_hidden_delay_register(initial_value);
  current_definition_context_->add_composite_update(
    "delay",
    {std::make_pair(delay_slot, signal.spec)});
  uint32_t delay_node_id = std::numeric_limits<uint32_t>::max();
  if (current_definition_context_->composite_spec)
  {
    delay_node_id = current_definition_context_->composite_spec->add_node(
      egress_composition::NodeKind::Delay,
      "delay",
      1,
      1);
    for (const auto & source : signal.sources)
    {
      current_definition_context_->composite_spec->add_edge(
        source,
        egress_composition::PortRef{delay_node_id, 0},
        egress_composition::ConnectionTiming::Delayed);
    }
  }
  return make_signal_expr(
    nullptr,
    expr::register_value_expr(delay_slot),
    delay_node_id == std::numeric_limits<uint32_t>::max()
      ? std::vector<egress_composition::PortRef>{}
      : std::vector<egress_composition::PortRef>{egress_composition::PortRef{delay_node_id, 0}});
}

static SignalExpr make_array_expr(const py::iterable & values)
{
  std::vector<expr::ExprSpecPtr> items;
  PythonGraph * graph = nullptr;
  std::vector<egress_composition::PortRef> sources;
  for (const py::handle & value : values)
  {
    const SignalExpr expr = coerce_expr(value);
    graph = merge_graphs(graph, expr.graph);
    if (expr.spec && expr.spec->kind == ExprKind::ArrayPack)
    {
      throw std::invalid_argument("Nested arrays are not supported.");
    }
    if (expr.spec && expr.spec->kind == ExprKind::Literal && expr.spec->literal.type == expr::ValueType::Matrix)
    {
      throw std::invalid_argument("Matrix literals cannot be nested inside arrays.");
    }
    items.push_back(expr.spec);
    sources = merge_sources(sources, expr.sources);
  }
  return make_signal_expr(graph, expr::array_pack_expr(std::move(items)), std::move(sources));
}

static SignalExpr make_matrix_expr(const py::handle & value)
{
  const expr::Value matrix = matrix_value_from_py(value);
  return make_signal_expr(nullptr, expr::literal_expr(matrix));
}

static void assign_input_expr(const InputPort & input, expr::ExprSpecPtr expr)
{
  if (!input.graph->graph().set_input_expr(input.module_name, input.input_id, std::move(expr)))
  {
    throw std::invalid_argument("Failed to assign expression to input.");
  }
}

static void assign_input(const InputPort & input, const py::handle & value)
{
  if (value.is_none())
  {
    assign_input_expr(input, nullptr);
    return;
  }

  const SignalExpr expr = coerce_expr(value);
  if (expr.graph != nullptr && expr.graph != input.graph)
  {
    throw std::invalid_argument("Assigned expression belongs to a different graph.");
  }

  assign_input_expr(input, expr.spec);
}

static void assign_input_index(const InputPort & input, const py::handle & index, const py::handle & value)
{
  auto base = input.graph->graph().get_input_expr(input.module_name, input.input_id);
  if (!base)
  {
    throw std::invalid_argument("Indexed assignment requires an existing input expression; assign the array first.");
  }

  const SignalExpr updated = array_set_expr(make_signal_expr(input.graph, std::move(base)), index, value);
  assign_input_expr(input, updated.spec);
}

static bool connect_ports(const OutputPort & out, const InputPort & in)
{
  if (out.graph != in.graph)
  {
    throw std::invalid_argument("Ports belong to different graphs.");
  }
  return out.graph->connect(out.module_name, out.output_id, in.module_name, in.input_id);
}

static bool disconnect_ports(const OutputPort & out, const InputPort & in)
{
  if (out.graph != in.graph)
  {
    throw std::invalid_argument("Ports belong to different graphs.");
  }
  return out.graph->disconnect(out.module_name, out.output_id, in.module_name, in.input_id);
}

static bool add_output_port(const OutputPort & out)
{
  return out.graph->add_output(out.module_name, out.output_id);
}

static bool add_output_expr(const SignalExpr & expr)
{
  if (expr.graph == nullptr)
  {
    throw std::invalid_argument("Graph outputs must reference a graph expression.");
  }
  return expr.graph->add_output_expr(expr.spec);
}

static std::vector<OutputPort> incoming_ports(const InputPort & in)
{
  std::vector<OutputPort> results;
  const auto sources = in.graph->graph().incoming_connections(in.module_name, in.input_id);
  results.reserve(sources.size());
  for (const auto & source : sources)
  {
    results.push_back(OutputPort{in.graph, source.first, source.second});
  }
  return results;
}

static py::object input_iadd(const InputPort & input, const py::handle & rhs)
{
  return py::cast(add_expr(current_input_expr(input), rhs));
}

static py::object input_isub(const InputPort & input, const py::handle & rhs)
{
  return py::cast(sub_expr(current_input_expr(input), rhs));
}

static py::object input_imul(const InputPort & input, const py::handle & rhs)
{
  return py::cast(mul_expr(current_input_expr(input), rhs));
}

static py::object input_idiv(const InputPort & input, const py::handle & rhs)
{
  return py::cast(div_expr(current_input_expr(input), rhs));
}

static py::object input_imod(const InputPort & input, const py::handle & rhs)
{
  return py::cast(mod_expr(current_input_expr(input), rhs));
}

static py::object input_ifloor_div(const InputPort & input, const py::handle & rhs)
{
  return py::cast(floor_div_expr(current_input_expr(input), rhs));
}

static py::object input_iand(const InputPort & input, const py::handle & rhs)
{
  return py::cast(bit_and_expr(current_input_expr(input), rhs));
}

static py::object input_ior(const InputPort & input, const py::handle & rhs)
{
  return py::cast(bit_or_expr(current_input_expr(input), rhs));
}

static py::object input_ixor(const InputPort & input, const py::handle & rhs)
{
  return py::cast(bit_xor_expr(current_input_expr(input), rhs));
}

static py::object input_ilshift(const InputPort & input, const py::handle & rhs)
{
  return py::cast(lshift_expr(current_input_expr(input), rhs));
}

static py::object input_irshift(const InputPort & input, const py::handle & rhs)
{
  return py::cast(rshift_expr(current_input_expr(input), rhs));
}
