#pragma once

static expr::Value scalar_value_from_py(const py::handle & value)
{
  if (py::isinstance<py::bool_>(value))
  {
    return expr::bool_value(value.cast<bool>());
  }
  if (py::isinstance<py::int_>(value))
  {
    return expr::int_value(value.cast<int64_t>());
  }
  if (py::isinstance<py::float_>(value))
  {
    return expr::float_value(value.cast<double>());
  }
  throw std::invalid_argument("Expected bool, int, or float.");
}

static bool is_matrix_literal(const py::handle & value)
{
  if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value))
  {
    return false;
  }

  for (const py::handle & item : value.cast<py::iterable>())
  {
    return py::isinstance<py::list>(item) || py::isinstance<py::tuple>(item);
  }

  return false;
}

static expr::Value matrix_value_from_py(const py::handle & value)
{
  if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value))
  {
    throw std::invalid_argument("Expected a list of rows for matrix literal.");
  }

  std::vector<expr::Value> items;
  std::size_t rows = 0;
  std::size_t cols = 0;

  for (const py::handle & row : value.cast<py::iterable>())
  {
    if (!py::isinstance<py::list>(row) && !py::isinstance<py::tuple>(row))
    {
      throw std::invalid_argument("Matrix rows must be lists or tuples.");
    }

    std::vector<expr::Value> row_items;
    for (const py::handle & item : row.cast<py::iterable>())
    {
      row_items.push_back(scalar_value_from_py(item));
    }

    if (rows == 0)
    {
      cols = row_items.size();
    }
    else if (row_items.size() != cols)
    {
      throw std::invalid_argument("Matrix rows must all be the same length.");
    }

    items.insert(items.end(), row_items.begin(), row_items.end());
    ++rows;
  }

  return expr::matrix_value(rows, cols, std::move(items));
}

static expr::Value value_from_py(const py::handle & value)
{
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value))
  {
    if (is_matrix_literal(value))
    {
      return matrix_value_from_py(value);
    }
    std::vector<expr::Value> items;
    for (const py::handle & item : value.cast<py::iterable>())
    {
      const expr::Value scalar = scalar_value_from_py(item);
      items.push_back(scalar);
    }
    return expr::array_value(std::move(items));
  }
  return scalar_value_from_py(value);
}

static bool is_array_state_spec(const py::handle & value)
{
  if (!py::isinstance<py::dict>(value))
  {
    return false;
  }

  const py::dict dict = value.cast<py::dict>();
  if (!dict.contains("__egress_array_state__"))
  {
    return false;
  }

  return py::cast<bool>(dict["__egress_array_state__"]);
}

static unsigned int require_input_id(
  const std::vector<std::string> & input_names,
  const py::dict & spec)
{
  if (!spec.contains("input"))
  {
    throw std::invalid_argument("array_state spec must include 'input'.");
  }

  if (!py::isinstance<py::str>(spec["input"]))
  {
    throw std::invalid_argument("array_state 'input' must be a string.");
  }

  const std::string input_name = py::cast<std::string>(spec["input"]);
  auto it = std::find(input_names.begin(), input_names.end(), input_name);
  if (it == input_names.end())
  {
    throw std::invalid_argument("Unknown input '" + input_name + "'.");
  }

  return static_cast<unsigned int>(std::distance(input_names.begin(), it));
}

static py::dict require_dict(const py::handle & value, const char * label)
{
  if (!py::isinstance<py::dict>(value))
  {
    throw std::invalid_argument(std::string(label) + " must be a dict.");
  }
  return value.cast<py::dict>();
}

static std::vector<std::string> require_names(const py::iterable & values, const char * label)
{
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;

  for (const py::handle & value : values)
  {
    if (!py::isinstance<py::str>(value))
    {
      throw std::invalid_argument(std::string(label) + " must contain only strings.");
    }

    const std::string name = value.cast<std::string>();
    if (!seen.insert(name).second)
    {
      throw std::invalid_argument("Duplicate name '" + name + "' in " + label + ".");
    }
    names.push_back(name);
  }

  return names;
}

static std::shared_ptr<ModuleDefinition> define_module_impl(
  const std::string & name,
  const py::iterable & inputs,
  const py::iterable & outputs,
  const py::dict & regs,
  const py::function & process,
  double sample_rate,
  const py::object & input_defaults)
{
  auto definition = std::make_shared<ModuleDefinition>();
  definition->type_name = name;
  definition->input_names = require_names(inputs, "inputs");
  definition->output_names = require_names(outputs, "outputs");
  definition->sample_rate = sample_rate;
  definition->input_defaults.assign(definition->input_names.size(), nullptr);

  if (!input_defaults.is_none())
  {
    const py::dict defaults_dict = require_dict(input_defaults, "input_defaults");
    for (const auto & item : defaults_dict)
    {
      const py::handle key = item.first;
      const py::handle value = item.second;
      if (!py::isinstance<py::str>(key))
      {
        throw std::invalid_argument("input_defaults keys must be strings.");
      }

      const std::string input_name = key.cast<std::string>();
      auto it = std::find(definition->input_names.begin(), definition->input_names.end(), input_name);
      if (it == definition->input_names.end())
      {
        throw std::invalid_argument("Unknown input '" + input_name + "' in input_defaults.");
      }

      if (value.is_none())
      {
        definition->input_defaults[static_cast<std::size_t>(std::distance(definition->input_names.begin(), it))] = nullptr;
        continue;
      }

      const SignalExpr expr = coerce_expr(value);
      if (expr.graph != nullptr)
      {
        throw std::invalid_argument("input_defaults cannot capture graph ports.");
      }

      definition->input_defaults[static_cast<std::size_t>(std::distance(definition->input_names.begin(), it))] = expr.spec;
    }
  }

  DefinitionBuildContext build_context;
  build_context.sample_rate = sample_rate;
  build_context.initialize_composite_spec(
    name,
    static_cast<uint32_t>(definition->input_names.size()),
    static_cast<uint32_t>(definition->output_names.size()));

  SymbolMap input_symbols;
  input_symbols.kind = SymbolMap::Kind::Input;
  for (unsigned int i = 0; i < definition->input_names.size(); ++i)
  {
    input_symbols.slots.emplace(definition->input_names[i], i);
  }

  SymbolMap register_symbols;
  register_symbols.kind = SymbolMap::Kind::Register;

  unsigned int register_id = 0;
  for (const auto & item : regs)
  {
    const py::handle key = item.first;
    const py::handle value = item.second;
    if (!py::isinstance<py::str>(key))
    {
      throw std::invalid_argument("regs keys must be strings.");
    }
    if (!py::isinstance<py::bool_>(value) &&
        !py::isinstance<py::float_>(value) &&
        !py::isinstance<py::int_>(value) &&
        !py::isinstance<py::list>(value) &&
        !py::isinstance<py::tuple>(value) &&
        !is_array_state_spec(value))
    {
      throw std::invalid_argument(
        "regs values must be bool, int, float, 1-D arrays, matrices, or array_state specs.");
    }

    const std::string reg_name = key.cast<std::string>();
    if (!register_symbols.slots.emplace(reg_name, register_id).second)
    {
      throw std::invalid_argument("Duplicate register '" + reg_name + "'.");
    }

    Module::RegisterArraySpec array_spec;
    expr::Value initial_value;
    if (is_array_state_spec(value))
    {
      const py::dict spec = value.cast<py::dict>();
      array_spec.enabled = true;
      array_spec.source_input_id = require_input_id(definition->input_names, spec);
      if (spec.contains("init"))
      {
        array_spec.init_value = scalar_value_from_py(spec["init"]);
      }
      else
      {
        array_spec.init_value = expr::float_value(0.0);
      }
      initial_value = expr::array_value({});
    }
    else
    {
      initial_value = value_from_py(value);
    }

    build_context.append_register(reg_name, std::move(initial_value), std::move(array_spec));
    ++register_id;
  }

  py::object result;
  {
    ScopedDefinitionBuildContext scoped_context(&build_context);
    result = process(py::cast(input_symbols), py::cast(register_symbols));
  }
  if (!py::isinstance<py::tuple>(result))
  {
    throw std::invalid_argument("process must return a tuple: (outputs, next_regs).");
  }

  py::tuple returned = result.cast<py::tuple>();
  if (returned.size() != 2)
  {
    throw std::invalid_argument("process must return exactly two dicts: (outputs, next_regs).");
  }

  const py::dict output_values = require_dict(returned[0], "process outputs");
  const py::dict register_values = require_dict(returned[1], "process next_regs");

  definition->output_exprs.assign(definition->output_names.size(), nullptr);
  definition->register_exprs = build_context.register_exprs;
  std::unordered_set<std::string> assigned_outputs;
  for (const auto & item : output_values)
  {
    const std::string output_name = py::cast<std::string>(item.first);
    auto it = std::find(definition->output_names.begin(), definition->output_names.end(), output_name);
    if (it == definition->output_names.end())
    {
      throw std::invalid_argument("Unknown output '" + output_name + "'.");
    }
    if (!assigned_outputs.insert(output_name).second)
    {
      throw std::invalid_argument("Output '" + output_name + "' assigned more than once.");
    }

    const SignalExpr expr = coerce_expr(item.second);
    if (expr.graph != nullptr)
    {
      throw std::invalid_argument("Module expressions cannot capture graph ports.");
    }
    const uint32_t output_id = static_cast<uint32_t>(std::distance(definition->output_names.begin(), it));
    definition->output_exprs[static_cast<std::size_t>(output_id)] = expr.spec;
    if (build_context.composite_spec)
    {
      for (const auto & source : expr.sources)
      {
        build_context.composite_spec->add_edge(
          source,
          egress_composition::PortRef{build_context.output_boundary_id, output_id},
          egress_composition::ConnectionTiming::SameTick);
      }
    }
  }

  for (const auto & item : register_values)
  {
    const std::string reg_name = py::cast<std::string>(item.first);
    auto it = register_symbols.slots.find(reg_name);
    if (it == register_symbols.slots.end())
    {
      throw std::invalid_argument("Unknown register '" + reg_name + "'.");
    }

    const SignalExpr expr = coerce_expr(item.second);
    if (expr.graph != nullptr)
    {
      throw std::invalid_argument("Module expressions cannot capture graph ports.");
    }
    definition->register_exprs[it->second] = expr.spec;
  }

  definition->delay_state_specs = finalize_delay_states(build_context);
  definition->register_names = std::move(build_context.register_names);
  definition->initial_registers = std::move(build_context.initial_registers);
  definition->register_array_specs = std::move(build_context.register_array_specs);
  definition->composite_spec = build_context.composite_spec;

  if (definition->composite_spec)
  {
    const auto validation = egress_composition::validate_composite_module(*definition->composite_spec);
    if (!validation.ok)
    {
      throw std::invalid_argument(validation.message);
    }
    definition->lowered_composite = std::make_shared<egress_composition::LoweredCompositeModule>(
      egress_composition::lower_composite_module(*definition->composite_spec));
    definition->composite_schedule = definition->lowered_composite->same_tick_schedule;
    definition->output_boundary_id = definition->composite_spec->output_boundary_id;
  }

  definition->nested_module_specs = finalize_nested_modules(build_context);
  return definition;
}

static std::shared_ptr<PureFunctionDefinition> define_pure_function_impl(
  const py::iterable & inputs,
  const py::iterable & outputs,
  const py::function & process)
{
  auto definition = std::make_shared<PureFunctionDefinition>();
  definition->input_names = require_names(inputs, "inputs");
  definition->output_names = require_names(outputs, "outputs");

  SymbolMap input_symbols;
  input_symbols.kind = SymbolMap::Kind::Input;
  for (unsigned int i = 0; i < definition->input_names.size(); ++i)
  {
    input_symbols.slots.emplace(definition->input_names[i], i);
  }

  py::object result = process(py::cast(input_symbols));
  if (!py::isinstance<py::dict>(result))
  {
    throw std::invalid_argument("process must return a dict of outputs.");
  }

  const py::dict output_values = require_dict(result, "process outputs");
  definition->output_exprs.assign(definition->output_names.size(), nullptr);
  std::unordered_set<std::string> assigned_outputs;
  for (const auto & item : output_values)
  {
    const std::string output_name = py::cast<std::string>(item.first);
    auto it = std::find(definition->output_names.begin(), definition->output_names.end(), output_name);
    if (it == definition->output_names.end())
    {
      throw std::invalid_argument("Unknown output '" + output_name + "'.");
    }
    if (!assigned_outputs.insert(output_name).second)
    {
      throw std::invalid_argument("Output '" + output_name + "' assigned more than once.");
    }

    const SignalExpr expr = coerce_expr(item.second);
    if (expr.graph != nullptr)
    {
      throw std::invalid_argument("Pure function expressions cannot capture graph ports.");
    }
    definition->output_exprs[static_cast<std::size_t>(std::distance(definition->output_names.begin(), it))] = expr.spec;
  }

  return definition;
}
