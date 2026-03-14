#pragma once

static PythonGraph * merge_graphs(PythonGraph * lhs, PythonGraph * rhs);
static SignalExpr make_signal_expr(PythonGraph * graph, expr::ExprSpecPtr spec);
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
  std::vector<Module::RegisterArraySpec> register_array_specs;
  double sample_rate = 44100.0;
};

struct PureFunctionDefinition
{
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<expr::ExprSpecPtr> output_exprs;
};

struct StatefulFunctionDefinition
{
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> register_names;
  std::vector<expr::Value> initial_registers;
  std::vector<expr::ExprSpecPtr> output_exprs;
  std::vector<expr::ExprSpecPtr> register_exprs;
  std::vector<Module::RegisterArraySpec> register_array_specs;
};

struct DefinitionBuildContext
{
  std::vector<std::string> register_names;
  std::vector<expr::Value> initial_registers;
  std::vector<expr::ExprSpecPtr> register_exprs;
  std::vector<Module::RegisterArraySpec> register_array_specs;
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

static py::object inline_stateful_body(
  const std::vector<std::string> & input_names,
  const std::vector<std::string> & output_names,
  const std::vector<std::string> & register_names,
  const std::vector<expr::Value> & initial_registers,
  const std::vector<expr::ExprSpecPtr> & output_exprs,
  const std::vector<expr::ExprSpecPtr> & register_exprs,
  const std::vector<Module::RegisterArraySpec> & register_array_specs,
  const std::vector<expr::ExprSpecPtr> & call_args,
  const char * label)
{
  if (current_definition_context_ == nullptr)
  {
    throw std::invalid_argument(
      std::string(label) + " may only be called inside define_module or define_stateful_function process bodies.");
  }

  if (call_args.size() != input_names.size())
  {
    throw std::invalid_argument(
      std::string(label) + " expects " + std::to_string(input_names.size()) + " arguments.");
  }

  std::vector<unsigned int> register_slots;
  register_slots.reserve(register_names.size());
  for (std::size_t i = 0; i < register_names.size(); ++i)
  {
    const Module::RegisterArraySpec & array_spec = register_array_specs[i];
    if (array_spec.enabled)
    {
      throw std::invalid_argument(
        std::string("Dynamic array_state is not supported in ") + label + " yet.");
    }
    register_slots.push_back(current_definition_context_->append_hidden_register(
      register_names[i],
      initial_registers[i],
      array_spec));
  }

  for (std::size_t i = 0; i < register_exprs.size(); ++i)
  {
    if (!register_exprs[i])
    {
      continue;
    }
    current_definition_context_->register_exprs[register_slots[i]] =
      clone_with_input_and_register_subst(register_exprs[i], call_args, register_slots);
  }

  if (output_exprs.size() == 1)
  {
    return py::cast(make_signal_expr(
      nullptr,
      clone_with_input_and_register_subst(output_exprs[0], call_args, register_slots)));
  }

  py::tuple outputs(output_exprs.size());
  for (std::size_t i = 0; i < output_exprs.size(); ++i)
  {
    outputs[i] = py::cast(make_signal_expr(
      nullptr,
      clone_with_input_and_register_subst(output_exprs[i], call_args, register_slots)));
  }
  return outputs;
}

static std::vector<expr::ExprSpecPtr> require_local_call_args(
  const py::args & args,
  const char * label)
{
  std::vector<expr::ExprSpecPtr> call_args;
  call_args.reserve(args.size());
  for (const py::handle & arg : args)
  {
    const SignalExpr signal = coerce_expr(arg);
    if (signal.graph != nullptr)
    {
      throw std::invalid_argument(std::string(label) + " arguments cannot capture graph ports.");
    }
    call_args.push_back(signal.spec);
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
            "Module types only accept signal arguments inside define_module or define_stateful_function process bodies.");
        }
        return py::cast(instantiate());
      }

      if (args.size() > definition_->input_names.size())
      {
        throw std::invalid_argument(
          "Module call expects at most " + std::to_string(definition_->input_names.size()) + " arguments.");
      }

      std::vector<expr::ExprSpecPtr> call_args = require_local_call_args(args, "Module call");
      for (std::size_t i = call_args.size(); i < definition_->input_names.size(); ++i)
      {
        const expr::ExprSpecPtr & default_expr = definition_->input_defaults[i];
        if (!default_expr)
        {
          throw std::invalid_argument(
            "Missing argument for module input '" + definition_->input_names[i] + "'.");
        }
        call_args.push_back(default_expr);
      }

      return inline_stateful_body(
        definition_->input_names,
        definition_->output_names,
        definition_->register_names,
        definition_->initial_registers,
        definition_->output_exprs,
        definition_->register_exprs,
        definition_->register_array_specs,
        call_args,
        "Module call");
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

class PyStatefulFunctionType
{
  public:
    explicit PyStatefulFunctionType(std::shared_ptr<StatefulFunctionDefinition> definition)
      : definition_(std::move(definition))
    {
    }

    py::object call(const py::args & args) const
    {
      return inline_stateful_body(
        definition_->input_names,
        definition_->output_names,
        definition_->register_names,
        definition_->initial_registers,
        definition_->output_exprs,
        definition_->register_exprs,
        definition_->register_array_specs,
        require_local_call_args(args, "Stateful function"),
        "Stateful function");
    }

  private:
    std::shared_ptr<StatefulFunctionDefinition> definition_;
};

static PythonGraph * merge_graphs(PythonGraph * lhs, PythonGraph * rhs)
{
  if (lhs != nullptr && rhs != nullptr && lhs != rhs)
  {
    throw std::invalid_argument("Expression operands belong to different graphs.");
  }
  return lhs != nullptr ? lhs : rhs;
}

static SignalExpr make_signal_expr(PythonGraph * graph, expr::ExprSpecPtr spec)
{
  SignalExpr expr;
  expr.graph = graph;
  expr.spec = std::move(spec);
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
  return make_signal_expr(operand.graph, expr::unary_expr(kind, operand.spec));
}

static SignalExpr make_binary_expr(ExprKind kind, const SignalExpr & lhs, const SignalExpr & rhs)
{
  return make_signal_expr(merge_graphs(lhs.graph, rhs.graph), expr::binary_expr(kind, lhs.spec, rhs.spec));
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
    expr::clamp_expr(signal.spec, min_signal.spec, max_signal.spec));
}

static SignalExpr logical_not_expr(const py::handle & value)
{
  return make_unary_expr(ExprKind::Not, coerce_expr(value));
}

static SignalExpr index_expr(const SignalExpr & value, const py::handle & index)
{
  const SignalExpr idx = coerce_expr(index);
  return make_signal_expr(merge_graphs(value.graph, idx.graph), expr::index_expr(value.spec, idx.spec));
}

static SignalExpr array_set_expr(const SignalExpr & array, const py::handle & index, const py::handle & value)
{
  const SignalExpr idx = coerce_expr(index);
  const SignalExpr replacement = coerce_expr(value);
  return make_signal_expr(
    merge_graphs(merge_graphs(array.graph, idx.graph), replacement.graph),
    expr::array_set_expr(array.spec, idx.spec, replacement.spec));
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
    return make_signal_expr(nullptr, expr::input_value_expr(slot_id));
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
      "eg.delay(...) may only be called inside define_module or define_stateful_function process bodies.");
  }

  const SignalExpr signal = coerce_expr(value);
  if (signal.graph != nullptr)
  {
    throw std::invalid_argument("eg.delay(...) cannot capture graph ports.");
  }

  const expr::Value initial_value = init.is_none() ? expr::float_value(0.0) : value_from_py(init);
  const unsigned int delay_slot = current_definition_context_->append_hidden_delay_register(initial_value);
  current_definition_context_->register_exprs[delay_slot] = signal.spec;
  return make_signal_expr(nullptr, expr::register_value_expr(delay_slot));
}

static SignalExpr make_array_expr(const py::iterable & values)
{
  std::vector<expr::ExprSpecPtr> items;
  PythonGraph * graph = nullptr;
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
  }
  return make_signal_expr(graph, expr::array_pack_expr(std::move(items)));
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
