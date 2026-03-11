#include "Graph.hpp"

#include "../lib/rtaudio/RtAudio.h"

#include <algorithm>
#include <cstdint>
#ifdef EGRESS_PROFILE
#include <atomic>
#include <chrono>
#endif
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace expr = egress_expr;

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

    std::string graph_jit_status() const
    {
      return graph_.graph_jit_status();
    }

    std::string next_name(const std::string & prefix)
    {
      const auto id = ++name_counters_[prefix];
      return prefix + std::to_string(id);
    }

    bool add_vco(const std::string & module_name, double frequency_hz)
    {
      return graph_.addModule(module_name, std::make_unique<VCO>(frequency_hz));
    }

    bool add_mux(const std::string & module_name)
    {
      return graph_.addModule(module_name, std::make_unique<MUX>());
    }

    bool add_vca(const std::string & module_name)
    {
      return graph_.addModule(module_name, std::make_unique<VCA>());
    }

    bool add_env(const std::string & module_name, double rise_ms, double fall_ms)
    {
      return graph_.addModule(module_name, std::make_unique<ENV>(rise_ms, fall_ms));
    }

    bool add_delay(const std::string & module_name, double buffer_size_samples)
    {
      return graph_.addModule(module_name, std::make_unique<DELAY>(buffer_size_samples));
    }

    bool add_const(const std::string & module_name, double value)
    {
      return graph_.addModule(module_name, std::make_unique<CONST>(value));
    }

    bool add_lowpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<LOWPASS>(freq_hz, res));
    }

    bool add_highpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<HIGHPASS>(freq_hz, res));
    }

    bool add_bandpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<BANDPASS>(freq_hz, res));
    }

    bool add_notch(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<NOTCH>(freq_hz, res));
    }

    bool add_allpass(const std::string & module_name, double freq_hz, double res)
    {
      return graph_.addModule(module_name, std::make_unique<ALLPASS>(freq_hz, res));
    }

    bool add_user_defined_module(
      const std::string & module_name,
      unsigned int input_count,
      std::vector<expr::ExprSpecPtr> output_exprs,
      std::vector<expr::ExprSpecPtr> register_exprs,
      std::vector<expr::Value> initial_registers,
      double sample_rate)
    {
      return graph_.addModule(
        module_name,
        std::make_unique<UserDefinedModule>(
          input_count,
          std::move(output_exprs),
          std::move(register_exprs),
          std::move(initial_registers),
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
};

static void assign_input(const InputPort & input, const py::handle & value);

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

struct StatefulModuleDefinition
{
  std::string type_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> register_names;
  std::vector<expr::Value> initial_registers;
  std::vector<expr::ExprSpecPtr> output_exprs;
  std::vector<expr::ExprSpecPtr> register_exprs;
  double sample_rate = 44100.0;
};

class PyStatefulModuleInstance
{
  public:
    PyStatefulModuleInstance(std::shared_ptr<StatefulModuleDefinition> definition, PythonGraph & graph)
      : definition_(std::move(definition)), graph_(&graph), name_(graph_->next_name(definition_->type_name))
    {
      if (!graph_->add_user_defined_module(
            name_,
            static_cast<unsigned int>(definition_->input_names.size()),
            definition_->output_exprs,
            definition_->register_exprs,
            definition_->initial_registers,
            definition_->sample_rate))
      {
        throw std::invalid_argument("Failed to create module '" + name_ + "'.");
      }
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

    std::shared_ptr<StatefulModuleDefinition> definition_;
    PythonGraph * graph_ = nullptr;
    std::string name_;
};

class PyStatefulModuleType
{
  public:
    explicit PyStatefulModuleType(std::shared_ptr<StatefulModuleDefinition> definition)
      : definition_(std::move(definition))
    {
    }

    const std::string & name() const { return definition_->type_name; }

    PyStatefulModuleInstance instantiate() const
    {
      return PyStatefulModuleInstance(definition_, default_graph());
    }

  private:
    std::shared_ptr<StatefulModuleDefinition> definition_;
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

static SignalExpr mod_expr(const SignalExpr & lhs, const py::handle & rhs)
{
  return make_binary_expr(ExprKind::Mod, lhs, coerce_expr(rhs));
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

static SignalExpr logical_not_expr(const py::handle & value)
{
  return make_unary_expr(ExprKind::Not, coerce_expr(value));
}

static SignalExpr index_expr(const SignalExpr & value, const py::handle & index)
{
  const SignalExpr idx = coerce_expr(index);
  return make_signal_expr(merge_graphs(value.graph, idx.graph), expr::index_expr(value.spec, idx.spec));
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

static SignalExpr make_array_expr(const py::iterable & values)
{
  std::vector<expr::ExprSpecPtr> items;
  for (const py::handle & value : values)
  {
    const SignalExpr expr = coerce_expr(value);
    if (expr.graph != nullptr)
    {
      throw std::invalid_argument("Array literals cannot capture graph ports.");
    }
    items.push_back(expr.spec);
  }
  return make_signal_expr(nullptr, expr::array_pack_expr(std::move(items)));
}

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

static expr::Value value_from_py(const py::handle & value)
{
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value))
  {
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

static std::shared_ptr<StatefulModuleDefinition> define_stateful_module_impl(
  const std::string & name,
  const py::iterable & inputs,
  const py::iterable & outputs,
  const py::dict & regs,
  const py::function & process,
  double sample_rate)
{
  auto definition = std::make_shared<StatefulModuleDefinition>();
  definition->type_name = name;
  definition->input_names = require_names(inputs, "inputs");
  definition->output_names = require_names(outputs, "outputs");
  definition->sample_rate = sample_rate;

  SymbolMap input_symbols;
  input_symbols.kind = SymbolMap::Kind::Input;
  for (unsigned int i = 0; i < definition->input_names.size(); ++i)
  {
    input_symbols.slots.emplace(definition->input_names[i], i);
  }

  SymbolMap register_symbols;
  register_symbols.kind = SymbolMap::Kind::Register;
  definition->register_exprs.assign(regs.size(), nullptr);
  definition->register_names.reserve(regs.size());
  definition->initial_registers.reserve(regs.size());

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
        !py::isinstance<py::tuple>(value))
    {
      throw std::invalid_argument("regs values must be bool, int, float, or 1-D arrays of those scalars.");
    }

    const std::string reg_name = key.cast<std::string>();
    if (!register_symbols.slots.emplace(reg_name, register_id).second)
    {
      throw std::invalid_argument("Duplicate register '" + reg_name + "'.");
    }

    definition->register_names.push_back(reg_name);
    definition->initial_registers.push_back(value_from_py(value));
    ++register_id;
  }

  py::object result = process(py::cast(input_symbols), py::cast(register_symbols));
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
      throw std::invalid_argument("Stateful module expressions cannot capture graph ports.");
    }
    definition->output_exprs[static_cast<std::size_t>(std::distance(definition->output_names.begin(), it))] = expr.spec;
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
      throw std::invalid_argument("Stateful module expressions cannot capture graph ports.");
    }
    definition->register_exprs[it->second] = expr.spec;
  }

  return definition;
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

class PyVCO
{
  public:
    explicit PyVCO(double frequency_hz)
      : graph_(&default_graph()), name_(graph_->next_name("vco"))
    {
      if (!graph_->add_vco(name_, frequency_hz))
      {
        throw std::invalid_argument("Failed to create VCO '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    OutputPort saw() const { return OutputPort{graph_, name_, VCO::SAW}; }
    OutputPort tri() const { return OutputPort{graph_, name_, VCO::TRI}; }
    OutputPort sin() const { return OutputPort{graph_, name_, VCO::SIN}; }
    OutputPort sqr() const { return OutputPort{graph_, name_, VCO::SQR}; }

    InputPort fm() const { return InputPort{graph_, name_, VCO::FM}; }
    InputPort fm_index() const { return InputPort{graph_, name_, VCO::FM_INDEX}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyMUX
{
  public:
    PyMUX() : graph_(&default_graph()), name_(graph_->next_name("mux"))
    {
      if (!graph_->add_mux(name_))
      {
        throw std::invalid_argument("Failed to create MUX '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort in1() const { return InputPort{graph_, name_, MUX::IN1}; }
    InputPort in2() const { return InputPort{graph_, name_, MUX::IN2}; }
    InputPort ctrl() const { return InputPort{graph_, name_, MUX::CTRL}; }
    OutputPort out() const { return OutputPort{graph_, name_, MUX::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyVCA
{
  public:
    PyVCA() : graph_(&default_graph()), name_(graph_->next_name("vca"))
    {
      if (!graph_->add_vca(name_))
      {
        throw std::invalid_argument("Failed to create VCA '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort in1() const { return InputPort{graph_, name_, VCA::IN1}; }
    InputPort in2() const { return InputPort{graph_, name_, VCA::IN2}; }
    OutputPort out() const { return OutputPort{graph_, name_, VCA::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyENV
{
  public:
    PyENV(double rise_ms, double fall_ms) : graph_(&default_graph()), name_(graph_->next_name("env"))
    {
      if (!graph_->add_env(name_, rise_ms, fall_ms))
      {
        throw std::invalid_argument("Failed to create ENV '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort trig() const { return InputPort{graph_, name_, ENV::TRIG}; }
    InputPort rise() const { return InputPort{graph_, name_, ENV::RISE}; }
    InputPort fall() const { return InputPort{graph_, name_, ENV::FALL}; }
    OutputPort out() const { return OutputPort{graph_, name_, ENV::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyDELAY
{
  public:
    explicit PyDELAY(double buffer_size_samples)
      : graph_(&default_graph()), name_(graph_->next_name("delay"))
    {
      if (!graph_->add_delay(name_, buffer_size_samples))
      {
        throw std::invalid_argument("Failed to create DELAY '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }

    InputPort in() const { return InputPort{graph_, name_, DELAY::IN}; }
    OutputPort out() const { return OutputPort{graph_, name_, DELAY::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyCONST
{
  public:
    explicit PyCONST(double value) : graph_(&default_graph()), name_(graph_->next_name("const"))
    {
      if (!graph_->add_const(name_, value))
      {
        throw std::invalid_argument("Failed to create CONST '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    OutputPort out() const { return OutputPort{graph_, name_, CONST::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyLOWPASS
{
  public:
    PyLOWPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("lowpass"))
    {
      if (!graph_->add_lowpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create LOWPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, LOWPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, LOWPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, LOWPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, LOWPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyHIGHPASS
{
  public:
    PyHIGHPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("highpass"))
    {
      if (!graph_->add_highpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create HIGHPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, HIGHPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, HIGHPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, HIGHPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, HIGHPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyBANDPASS
{
  public:
    PyBANDPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("bandpass"))
    {
      if (!graph_->add_bandpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create BANDPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, BANDPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, BANDPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, BANDPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, BANDPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyNOTCH
{
  public:
    PyNOTCH(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("notch"))
    {
      if (!graph_->add_notch(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create NOTCH '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, NOTCH::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, NOTCH::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, NOTCH::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, NOTCH::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PyALLPASS
{
  public:
    PyALLPASS(double freq_hz, double res = 0.707) : graph_(&default_graph()), name_(graph_->next_name("allpass"))
    {
      if (!graph_->add_allpass(name_, freq_hz, res))
      {
        throw std::invalid_argument("Failed to create ALLPASS '" + name_ + "'.");
      }
    }

    const std::string & name() const { return name_; }
    InputPort in() const { return InputPort{graph_, name_, ALLPASS::IN}; }
    InputPort freq() const { return InputPort{graph_, name_, ALLPASS::FREQ}; }
    InputPort res() const { return InputPort{graph_, name_, ALLPASS::RES}; }
    OutputPort out() const { return OutputPort{graph_, name_, ALLPASS::OUT}; }

  private:
    PythonGraph * graph_;
    std::string name_;
};

class PythonDAC
{
  public:
    explicit PythonDAC(unsigned int sample_rate = 44100, unsigned int channels = 2)
      : PythonDAC(default_graph(), sample_rate, channels)
    {
    }

  private:
    explicit PythonDAC(PythonGraph & graph, unsigned int sample_rate, unsigned int channels)
      : graph_(graph), sample_rate_(sample_rate), channels_(channels), running_(false)
    {
    }

  public:
    ~PythonDAC()
    {
      stop();
    }

    void start()
    {
      if (running_)
      {
        return;
      }

      if (audio_.getDeviceCount() < 1)
      {
        throw std::runtime_error("No audio output devices found.");
      }

      RtAudio::StreamParameters out_params;
      out_params.deviceId = audio_.getDefaultOutputDevice();
      out_params.nChannels = channels_;
      out_params.firstChannel = 0;

      unsigned int buffer_frames = graph_.graph().getBufferLength();

      audio_.openStream(
        &out_params,
        nullptr,
        RTAUDIO_FLOAT64,
        sample_rate_,
        &buffer_frames,
        &PythonDAC::fill_buffer,
        this);

      audio_.startStream();
      running_ = true;
    }

    void stop()
    {
      if (!audio_.isStreamOpen())
      {
        running_ = false;
        return;
      }

      if (audio_.isStreamRunning())
      {
        audio_.stopStream();
      }
      audio_.closeStream();
      running_ = false;
    }

    bool is_running() const
    {
      return running_;
    }

    py::dict callback_timing_stats() const
    {
#ifdef EGRESS_PROFILE
      const uint64_t callbacks = callback_count_.load(std::memory_order_relaxed);
      const uint64_t total_ns = total_callback_ns_.load(std::memory_order_relaxed);
      const uint64_t max_ns = max_callback_ns_.load(std::memory_order_relaxed);
      const uint64_t overruns = overrun_count_.load(std::memory_order_relaxed);

      py::dict stats;
      stats["enabled"] = true;
      stats["callback_count"] = callbacks;
      stats["avg_callback_ms"] = callbacks == 0 ? 0.0 : (static_cast<double>(total_ns) / static_cast<double>(callbacks)) / 1e6;
      stats["max_callback_ms"] = static_cast<double>(max_ns) / 1e6;
      stats["overrun_count"] = overruns;
      return stats;
#else
      py::dict stats;
      stats["enabled"] = false;
      stats["callback_count"] = 0;
      stats["avg_callback_ms"] = 0.0;
      stats["max_callback_ms"] = 0.0;
      stats["overrun_count"] = 0;
      return stats;
#endif
    }

    void reset_callback_timing_stats()
    {
#ifdef EGRESS_PROFILE
      callback_count_.store(0, std::memory_order_relaxed);
      total_callback_ns_.store(0, std::memory_order_relaxed);
      max_callback_ns_.store(0, std::memory_order_relaxed);
      overrun_count_.store(0, std::memory_order_relaxed);
#endif
    }

  private:
#ifdef EGRESS_PROFILE
    static void update_max_callback_ns(std::atomic<uint64_t> & dst, uint64_t candidate)
    {
      uint64_t current = dst.load(std::memory_order_relaxed);
      while (current < candidate && !dst.compare_exchange_weak(current, candidate, std::memory_order_relaxed))
      {
      }
    }
#endif

    static int fill_buffer(
      void * output_buffer,
      void *,
      unsigned int n_buffer_frames,
      double,
      RtAudioStreamStatus status,
      void * user_data)
    {
      auto * self = static_cast<PythonDAC *>(user_data);
      auto * out = static_cast<double *>(output_buffer);

    #ifdef EGRESS_PROFILE
      const auto callback_start = std::chrono::steady_clock::now();
    #endif

      self->graph_.graph().process();
      const auto & source = self->graph_.graph().outputBuffer;

      for (unsigned int i = 0; i < n_buffer_frames; ++i)
      {
        const double sample = i < source.size() ? source[i] : 0.0;
        for (unsigned int c = 0; c < self->channels_; ++c)
        {
          *out++ = sample;
        }
      }

#ifdef EGRESS_PROFILE
      const auto callback_end = std::chrono::steady_clock::now();
      const uint64_t elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(callback_end - callback_start).count());

      self->callback_count_.fetch_add(1, std::memory_order_relaxed);
      self->total_callback_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
      update_max_callback_ns(self->max_callback_ns_, elapsed_ns);

      const uint64_t budget_ns = static_cast<uint64_t>(
        (static_cast<double>(n_buffer_frames) * 1e9) / static_cast<double>(self->sample_rate_));
      if (elapsed_ns > budget_ns || status != 0)
      {
        self->overrun_count_.fetch_add(1, std::memory_order_relaxed);
      }
#else
      (void)status;
#endif

      return 0;
    }

    PythonGraph & graph_;
    RtAudio audio_;
    unsigned int sample_rate_;
    unsigned int channels_;
    bool running_;
  #ifdef EGRESS_PROFILE
    std::atomic<uint64_t> callback_count_{0};
    std::atomic<uint64_t> total_callback_ns_{0};
    std::atomic<uint64_t> max_callback_ns_{0};
    std::atomic<uint64_t> overrun_count_{0};
  #endif
};

PYBIND11_MODULE(egress, m)
{
  m.doc() = "Python frontend for egress graph/modules and DAC control.";

  py::class_<PythonGraph>(m, "Graph")
    .def("destroy_module", &PythonGraph::destroy_module, py::arg("module_name"))
    .def(
      "connect",
      &PythonGraph::connect,
      py::arg("from_module"),
      py::arg("from_output_id"),
      py::arg("to_module"),
      py::arg("to_input_id"))
    .def(
      "disconnect",
      &PythonGraph::disconnect,
      py::arg("from_module"),
      py::arg("from_output_id"),
      py::arg("to_module"),
      py::arg("to_input_id"))
    .def("add_output", &PythonGraph::add_output, py::arg("module_name"), py::arg("output_id"))
    .def("process", &PythonGraph::process)
    .def("output_buffer", &PythonGraph::output_buffer)
    .def("profile_stats", &PythonGraph::profile_stats)
    .def("reset_profile_stats", &PythonGraph::reset_profile_stats)
    .def("graph_jit_status", &PythonGraph::graph_jit_status);

  m.def("graph", []() -> PythonGraph & { return default_graph(); }, py::return_value_policy::reference);

  py::class_<SignalExpr>(m, "SignalExpr")
    .def("__add__", [](const SignalExpr & lhs, const py::object & rhs) { return add_expr(lhs, rhs); })
    .def("__radd__", [](const SignalExpr & rhs, const py::object & lhs) { return add_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__sub__", [](const SignalExpr & lhs, const py::object & rhs) { return sub_expr(lhs, rhs); })
    .def("__rsub__", [](const SignalExpr & rhs, const py::object & lhs) { return sub_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__mul__", [](const SignalExpr & lhs, const py::object & rhs) { return mul_expr(lhs, rhs); })
    .def("__rmul__", [](const SignalExpr & rhs, const py::object & lhs) { return mul_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__truediv__", [](const SignalExpr & lhs, const py::object & rhs) { return div_expr(lhs, rhs); })
    .def("__rtruediv__", [](const SignalExpr & rhs, const py::object & lhs) { return div_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__floordiv__", [](const SignalExpr & lhs, const py::object & rhs) { return floor_div_expr(lhs, rhs); })
    .def("__rfloordiv__", [](const SignalExpr & rhs, const py::object & lhs) { return floor_div_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__mod__", [](const SignalExpr & lhs, const py::object & rhs) { return mod_expr(lhs, rhs); })
    .def("__rmod__", [](const SignalExpr & rhs, const py::object & lhs) { return mod_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__and__", [](const SignalExpr & lhs, const py::object & rhs) { return bit_and_expr(lhs, rhs); })
    .def("__rand__", [](const SignalExpr & rhs, const py::object & lhs) { return bit_and_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__or__", [](const SignalExpr & lhs, const py::object & rhs) { return bit_or_expr(lhs, rhs); })
    .def("__ror__", [](const SignalExpr & rhs, const py::object & lhs) { return bit_or_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__xor__", [](const SignalExpr & lhs, const py::object & rhs) { return bit_xor_expr(lhs, rhs); })
    .def("__rxor__", [](const SignalExpr & rhs, const py::object & lhs) { return bit_xor_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__lshift__", [](const SignalExpr & lhs, const py::object & rhs) { return lshift_expr(lhs, rhs); })
    .def("__rlshift__", [](const SignalExpr & rhs, const py::object & lhs) { return lshift_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__rshift__", [](const SignalExpr & lhs, const py::object & rhs) { return rshift_expr(lhs, rhs); })
    .def("__rrshift__", [](const SignalExpr & rhs, const py::object & lhs) { return rshift_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__lt__", [](const SignalExpr & lhs, const py::object & rhs) { return less_expr(py::cast(lhs), rhs); })
    .def("__le__", [](const SignalExpr & lhs, const py::object & rhs) { return less_equal_expr(py::cast(lhs), rhs); })
    .def("__gt__", [](const SignalExpr & lhs, const py::object & rhs) { return greater_expr(py::cast(lhs), rhs); })
    .def("__ge__", [](const SignalExpr & lhs, const py::object & rhs) { return greater_equal_expr(py::cast(lhs), rhs); })
    .def("__eq__", [](const SignalExpr & lhs, const py::object & rhs) { return equal_expr(py::cast(lhs), rhs); })
    .def("__ne__", [](const SignalExpr & lhs, const py::object & rhs) { return not_equal_expr(py::cast(lhs), rhs); })
    .def("__getitem__", [](const SignalExpr & value, const py::object & index) { return index_expr(value, index); })
    .def("__bool__", [](const SignalExpr &) -> bool {
      throw py::type_error("Symbolic expressions do not have Python truthiness; use eg.logical_not(...) or comparisons.");
    })
    .def("__neg__", [](const SignalExpr & expr) { return make_unary_expr(ExprKind::Neg, expr); })
    .def("__invert__", [](const SignalExpr & expr) { return make_unary_expr(ExprKind::BitNot, expr); });

  py::class_<OutputPort>(m, "OutputPort")
    .def_property_readonly("module_name", [](const OutputPort & p) { return p.module_name; })
    .def_property_readonly("output_id", [](const OutputPort & p) { return p.output_id; })
    .def("__add__", [](const OutputPort & lhs, const py::object & rhs) { return add_expr(make_output_expr(lhs), rhs); })
    .def("__radd__", [](const OutputPort & rhs, const py::object & lhs) { return add_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__sub__", [](const OutputPort & lhs, const py::object & rhs) { return sub_expr(make_output_expr(lhs), rhs); })
    .def("__rsub__", [](const OutputPort & rhs, const py::object & lhs) { return sub_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__mul__", [](const OutputPort & lhs, const py::object & rhs) { return mul_expr(make_output_expr(lhs), rhs); })
    .def("__rmul__", [](const OutputPort & rhs, const py::object & lhs) { return mul_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__truediv__", [](const OutputPort & lhs, const py::object & rhs) { return div_expr(make_output_expr(lhs), rhs); })
    .def("__rtruediv__", [](const OutputPort & rhs, const py::object & lhs) { return div_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__floordiv__", [](const OutputPort & lhs, const py::object & rhs) { return floor_div_expr(make_output_expr(lhs), rhs); })
    .def("__rfloordiv__", [](const OutputPort & rhs, const py::object & lhs) { return floor_div_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__mod__", [](const OutputPort & lhs, const py::object & rhs) { return mod_expr(make_output_expr(lhs), rhs); })
    .def("__rmod__", [](const OutputPort & rhs, const py::object & lhs) { return mod_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__and__", [](const OutputPort & lhs, const py::object & rhs) { return bit_and_expr(make_output_expr(lhs), rhs); })
    .def("__rand__", [](const OutputPort & rhs, const py::object & lhs) { return bit_and_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__or__", [](const OutputPort & lhs, const py::object & rhs) { return bit_or_expr(make_output_expr(lhs), rhs); })
    .def("__ror__", [](const OutputPort & rhs, const py::object & lhs) { return bit_or_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__xor__", [](const OutputPort & lhs, const py::object & rhs) { return bit_xor_expr(make_output_expr(lhs), rhs); })
    .def("__rxor__", [](const OutputPort & rhs, const py::object & lhs) { return bit_xor_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__lshift__", [](const OutputPort & lhs, const py::object & rhs) { return lshift_expr(make_output_expr(lhs), rhs); })
    .def("__rlshift__", [](const OutputPort & rhs, const py::object & lhs) { return lshift_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__rshift__", [](const OutputPort & lhs, const py::object & rhs) { return rshift_expr(make_output_expr(lhs), rhs); })
    .def("__rrshift__", [](const OutputPort & rhs, const py::object & lhs) { return rshift_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__lt__", [](const OutputPort & lhs, const py::object & rhs) { return less_expr(py::cast(make_output_expr(lhs)), rhs); })
    .def("__le__", [](const OutputPort & lhs, const py::object & rhs) { return less_equal_expr(py::cast(make_output_expr(lhs)), rhs); })
    .def("__gt__", [](const OutputPort & lhs, const py::object & rhs) { return greater_expr(py::cast(make_output_expr(lhs)), rhs); })
    .def("__ge__", [](const OutputPort & lhs, const py::object & rhs) { return greater_equal_expr(py::cast(make_output_expr(lhs)), rhs); })
    .def("__eq__", [](const OutputPort & lhs, const py::object & rhs) { return equal_expr(py::cast(make_output_expr(lhs)), rhs); })
    .def("__ne__", [](const OutputPort & lhs, const py::object & rhs) { return not_equal_expr(py::cast(make_output_expr(lhs)), rhs); })
    .def("__bool__", [](const OutputPort &) -> bool {
      throw py::type_error("Ports do not have Python truthiness; use eg.logical_not(...) or comparisons.");
    })
    .def("__neg__", [](const OutputPort & out) { return make_unary_expr(ExprKind::Neg, make_output_expr(out)); })
    .def("__invert__", [](const OutputPort & out) { return make_unary_expr(ExprKind::BitNot, make_output_expr(out)); });

  py::class_<InputPort>(m, "InputPort")
    .def_property_readonly("module_name", [](const InputPort & p) { return p.module_name; })
    .def_property_readonly("input_id", [](const InputPort & p) { return p.input_id; })
    .def_property_readonly("expr", [](const InputPort & p) { return current_input_expr(p); })
    .def("assign", [](const InputPort & in, const py::object & value) { assign_input(in, value); }, py::arg("value"))
    .def("__add__", [](const InputPort & lhs, const py::object & rhs) { return add_expr(current_input_expr(lhs), rhs); })
    .def("__radd__", [](const InputPort & rhs, const py::object & lhs) { return add_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__sub__", [](const InputPort & lhs, const py::object & rhs) { return sub_expr(current_input_expr(lhs), rhs); })
    .def("__rsub__", [](const InputPort & rhs, const py::object & lhs) { return sub_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__mul__", [](const InputPort & lhs, const py::object & rhs) { return mul_expr(current_input_expr(lhs), rhs); })
    .def("__rmul__", [](const InputPort & rhs, const py::object & lhs) { return mul_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__truediv__", [](const InputPort & lhs, const py::object & rhs) { return div_expr(current_input_expr(lhs), rhs); })
    .def("__rtruediv__", [](const InputPort & rhs, const py::object & lhs) { return div_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__floordiv__", [](const InputPort & lhs, const py::object & rhs) { return floor_div_expr(current_input_expr(lhs), rhs); })
    .def("__rfloordiv__", [](const InputPort & rhs, const py::object & lhs) { return floor_div_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__mod__", [](const InputPort & lhs, const py::object & rhs) { return mod_expr(current_input_expr(lhs), rhs); })
    .def("__rmod__", [](const InputPort & rhs, const py::object & lhs) { return mod_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__and__", [](const InputPort & lhs, const py::object & rhs) { return bit_and_expr(current_input_expr(lhs), rhs); })
    .def("__rand__", [](const InputPort & rhs, const py::object & lhs) { return bit_and_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__or__", [](const InputPort & lhs, const py::object & rhs) { return bit_or_expr(current_input_expr(lhs), rhs); })
    .def("__ror__", [](const InputPort & rhs, const py::object & lhs) { return bit_or_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__xor__", [](const InputPort & lhs, const py::object & rhs) { return bit_xor_expr(current_input_expr(lhs), rhs); })
    .def("__rxor__", [](const InputPort & rhs, const py::object & lhs) { return bit_xor_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__lshift__", [](const InputPort & lhs, const py::object & rhs) { return lshift_expr(current_input_expr(lhs), rhs); })
    .def("__rlshift__", [](const InputPort & rhs, const py::object & lhs) { return lshift_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__rshift__", [](const InputPort & lhs, const py::object & rhs) { return rshift_expr(current_input_expr(lhs), rhs); })
    .def("__rrshift__", [](const InputPort & rhs, const py::object & lhs) { return rshift_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__lt__", [](const InputPort & lhs, const py::object & rhs) { return less_expr(py::cast(current_input_expr(lhs)), rhs); })
    .def("__le__", [](const InputPort & lhs, const py::object & rhs) { return less_equal_expr(py::cast(current_input_expr(lhs)), rhs); })
    .def("__gt__", [](const InputPort & lhs, const py::object & rhs) { return greater_expr(py::cast(current_input_expr(lhs)), rhs); })
    .def("__ge__", [](const InputPort & lhs, const py::object & rhs) { return greater_equal_expr(py::cast(current_input_expr(lhs)), rhs); })
    .def("__eq__", [](const InputPort & lhs, const py::object & rhs) { return equal_expr(py::cast(current_input_expr(lhs)), rhs); })
    .def("__ne__", [](const InputPort & lhs, const py::object & rhs) { return not_equal_expr(py::cast(current_input_expr(lhs)), rhs); })
    .def("__bool__", [](const InputPort &) -> bool {
      throw py::type_error("Ports do not have Python truthiness; use eg.logical_not(...) or comparisons.");
    })
    .def("__iadd__", [](const InputPort & lhs, const py::object & rhs) { return input_iadd(lhs, rhs); })
    .def("__isub__", [](const InputPort & lhs, const py::object & rhs) { return input_isub(lhs, rhs); })
    .def("__imul__", [](const InputPort & lhs, const py::object & rhs) { return input_imul(lhs, rhs); })
    .def("__itruediv__", [](const InputPort & lhs, const py::object & rhs) { return input_idiv(lhs, rhs); })
    .def("__ifloordiv__", [](const InputPort & lhs, const py::object & rhs) { return input_ifloor_div(lhs, rhs); })
    .def("__imod__", [](const InputPort & lhs, const py::object & rhs) { return input_imod(lhs, rhs); })
    .def("__iand__", [](const InputPort & lhs, const py::object & rhs) { return input_iand(lhs, rhs); })
    .def("__ior__", [](const InputPort & lhs, const py::object & rhs) { return input_ior(lhs, rhs); })
    .def("__ixor__", [](const InputPort & lhs, const py::object & rhs) { return input_ixor(lhs, rhs); })
    .def("__ilshift__", [](const InputPort & lhs, const py::object & rhs) { return input_ilshift(lhs, rhs); })
    .def("__irshift__", [](const InputPort & lhs, const py::object & rhs) { return input_irshift(lhs, rhs); })
    .def("__neg__", [](const InputPort & p) { return make_unary_expr(ExprKind::Neg, current_input_expr(p)); })
    .def("__invert__", [](const InputPort & p) { return make_unary_expr(ExprKind::BitNot, current_input_expr(p)); });

  py::class_<SymbolMap>(m, "_SymbolMap")
    .def("__getitem__", [](const SymbolMap & symbols, const std::string & name) {
      auto it = symbols.slots.find(name);
      if (it == symbols.slots.end())
      {
        throw py::key_error("Unknown symbol '" + name + "'.");
      }
      return symbol_expr(symbols.kind, it->second);
    });

  py::class_<PyStatefulModuleInstance>(m, "StatefulModule")
    .def_property_readonly("name", &PyStatefulModuleInstance::name)
    .def("get_input", &PyStatefulModuleInstance::get_input, py::arg("name"))
    .def("get_output", &PyStatefulModuleInstance::get_output, py::arg("name"))
    .def("__getattr__", &PyStatefulModuleInstance::getattr, py::arg("name"))
    .def("__setattr__", &PyStatefulModuleInstance::setattr, py::arg("name"), py::arg("value"));

  py::class_<PyStatefulModuleType>(m, "StatefulModuleType")
    .def_property_readonly("name", &PyStatefulModuleType::name)
    .def("__call__", &PyStatefulModuleType::instantiate);

  m.def("connect", &connect_ports, py::arg("out"), py::arg("in"));
  m.def("disconnect", &disconnect_ports, py::arg("out"), py::arg("in"));
  m.def("add_output", &add_output_port, py::arg("out"));
  m.def("incoming", &incoming_ports, py::arg("in"));
  m.def("sin", [](const py::object & value) { return sin_expr(value); }, py::arg("value"));
  m.def("array", [](const py::iterable & values) { return make_array_expr(values); }, py::arg("values"));
  m.def("logical_not", [](const py::object & value) { return logical_not_expr(value); }, py::arg("value"));
  m.def("sample_rate", []() { return sample_rate_expr(); });
  m.def("sample_index", []() { return sample_index_expr(); });
  m.def(
    "define_stateful_module",
    [](const std::string & name,
       const py::iterable & inputs,
       const py::iterable & outputs,
       const py::dict & regs,
       const py::function & process,
       double sample_rate)
    {
      return PyStatefulModuleType(define_stateful_module_impl(name, inputs, outputs, regs, process, sample_rate));
    },
    py::arg("name"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("regs"),
    py::arg("process"),
    py::arg("sample_rate") = 44100.0);

  py::class_<PyVCO>(m, "VCO")
    .def(py::init<double>(), py::arg("frequency_hz"))
    .def_property_readonly("name", &PyVCO::name)
    .def_property_readonly("saw", &PyVCO::saw)
    .def_property_readonly("tri", &PyVCO::tri)
    .def_property_readonly("sin", &PyVCO::sin)
    .def_property_readonly("sqr", &PyVCO::sqr)
    .def_property("fm", &PyVCO::fm, [](PyVCO & self, const py::object & value) { assign_input(self.fm(), value); })
    .def_property("fm_index", &PyVCO::fm_index, [](PyVCO & self, const py::object & value) { assign_input(self.fm_index(), value); });

  py::class_<PyMUX>(m, "MUX")
    .def(py::init<>())
    .def_property_readonly("name", &PyMUX::name)
    .def_property("input1", &PyMUX::in1, [](PyMUX & self, const py::object & value) { assign_input(self.in1(), value); })
    .def_property("input2", &PyMUX::in2, [](PyMUX & self, const py::object & value) { assign_input(self.in2(), value); })
    .def_property("control", &PyMUX::ctrl, [](PyMUX & self, const py::object & value) { assign_input(self.ctrl(), value); })
    .def_property_readonly("output", &PyMUX::out);

  py::class_<PyVCA>(m, "VCA")
    .def(py::init<>())
    .def_property_readonly("name", &PyVCA::name)
    .def_property("input1", &PyVCA::in1, [](PyVCA & self, const py::object & value) { assign_input(self.in1(), value); })
    .def_property("input2", &PyVCA::in2, [](PyVCA & self, const py::object & value) { assign_input(self.in2(), value); })
    .def_property_readonly("output", &PyVCA::out);

  py::class_<PyENV>(m, "ENV")
    .def(py::init<double, double>(), py::arg("rise_ms"), py::arg("fall_ms"))
    .def_property_readonly("name", &PyENV::name)
    .def_property("trig", &PyENV::trig, [](PyENV & self, const py::object & value) { assign_input(self.trig(), value); })
    .def_property("rise", &PyENV::rise, [](PyENV & self, const py::object & value) { assign_input(self.rise(), value); })
    .def_property("fall", &PyENV::fall, [](PyENV & self, const py::object & value) { assign_input(self.fall(), value); })
    .def_property_readonly("output", &PyENV::out);

  py::class_<PyDELAY>(m, "DELAY")
    .def(py::init<double>(), py::arg("buffer_size_samples"))
    .def_property_readonly("name", &PyDELAY::name)
    .def_property("input", &PyDELAY::in, [](PyDELAY & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property_readonly("output", &PyDELAY::out);

  py::class_<PyCONST>(m, "CONST")
    .def(py::init<double>(), py::arg("value"))
    .def_property_readonly("name", &PyCONST::name)
    .def_property_readonly("output", &PyCONST::out);

  py::class_<PyLOWPASS>(m, "LOWPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyLOWPASS::name)
    .def_property("input", &PyLOWPASS::in, [](PyLOWPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyLOWPASS::freq, [](PyLOWPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyLOWPASS::res, [](PyLOWPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyLOWPASS::out);

  py::class_<PyHIGHPASS>(m, "HIGHPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyHIGHPASS::name)
    .def_property("input", &PyHIGHPASS::in, [](PyHIGHPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyHIGHPASS::freq, [](PyHIGHPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyHIGHPASS::res, [](PyHIGHPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyHIGHPASS::out);

  py::class_<PyBANDPASS>(m, "BANDPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyBANDPASS::name)
    .def_property("input", &PyBANDPASS::in, [](PyBANDPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyBANDPASS::freq, [](PyBANDPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyBANDPASS::res, [](PyBANDPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyBANDPASS::out);

  py::class_<PyNOTCH>(m, "NOTCH")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyNOTCH::name)
    .def_property("input", &PyNOTCH::in, [](PyNOTCH & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyNOTCH::freq, [](PyNOTCH & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyNOTCH::res, [](PyNOTCH & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyNOTCH::out);

  py::class_<PyALLPASS>(m, "ALLPASS")
    .def(py::init<double, double>(), py::arg("freq"), py::arg("res") = 0.707)
    .def_property_readonly("name", &PyALLPASS::name)
    .def_property("input", &PyALLPASS::in, [](PyALLPASS & self, const py::object & value) { assign_input(self.in(), value); })
    .def_property("freq", &PyALLPASS::freq, [](PyALLPASS & self, const py::object & value) { assign_input(self.freq(), value); })
    .def_property("res", &PyALLPASS::res, [](PyALLPASS & self, const py::object & value) { assign_input(self.res(), value); })
    .def_property_readonly("output", &PyALLPASS::out);

  py::class_<PythonDAC>(m, "DAC")
    .def(py::init<unsigned int, unsigned int>(), py::arg("sample_rate") = 44100, py::arg("channels") = 2)
    .def("start", &PythonDAC::start)
    .def("stop", &PythonDAC::stop)
    .def("is_running", &PythonDAC::is_running)
    .def("callback_timing_stats", &PythonDAC::callback_timing_stats)
    .def("reset_callback_timing_stats", &PythonDAC::reset_callback_timing_stats);

  py::enum_<VCO::Ins>(m, "VCOIn")
    .value("FM", VCO::Ins::FM)
    .value("FM_INDEX", VCO::Ins::FM_INDEX)
    .value("IN_COUNT", VCO::Ins::IN_COUNT);

  py::enum_<VCO::Outs>(m, "VCOOut")
    .value("SAW", VCO::Outs::SAW)
    .value("TRI", VCO::Outs::TRI)
    .value("SIN", VCO::Outs::SIN)
    .value("SQR", VCO::Outs::SQR)
    .value("OUT_COUNT", VCO::Outs::OUT_COUNT);
}
