#pragma once

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
    .def("add_output_tap", &PythonGraph::add_output_tap, py::arg("module_name"), py::arg("output_id"))
    .def("remove_output_tap", &PythonGraph::remove_output_tap, py::arg("tap_id"))
    .def("process", &PythonGraph::process)
    .def("output_buffer", &PythonGraph::output_buffer)
    .def("output_tap_buffer", &PythonGraph::output_tap_buffer, py::arg("tap_id"))
    .def("profile_stats", &PythonGraph::profile_stats)
    .def("reset_profile_stats", &PythonGraph::reset_profile_stats);

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
    .def("__pow__", [](const SignalExpr & lhs, const py::object & rhs) { return pow_expr(lhs, rhs); })
    .def("__rpow__", [](const SignalExpr & rhs, const py::object & lhs) { return pow_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__matmul__", [](const SignalExpr & lhs, const py::object & rhs) { return matmul_expr(py::cast(lhs), rhs); })
    .def("__rmatmul__", [](const SignalExpr & rhs, const py::object & lhs) { return matmul_expr(lhs, py::cast(rhs)); })
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
    .def("__abs__", [](const SignalExpr & expr) { return make_unary_expr(ExprKind::Abs, expr); })
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
    .def("__pow__", [](const OutputPort & lhs, const py::object & rhs) { return pow_expr(make_output_expr(lhs), rhs); })
    .def("__rpow__", [](const OutputPort & rhs, const py::object & lhs) { return pow_expr(coerce_expr(lhs), py::cast(rhs)); })
    .def("__matmul__", [](const OutputPort & lhs, const py::object & rhs) { return matmul_expr(py::cast(make_output_expr(lhs)), rhs); })
    .def("__rmatmul__", [](const OutputPort & rhs, const py::object & lhs) { return matmul_expr(lhs, py::cast(make_output_expr(rhs))); })
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
    .def("__getitem__", [](const OutputPort & value, const py::object & index) {
      return index_expr(make_output_expr(value), index);
    })
    .def("__bool__", [](const OutputPort &) -> bool {
      throw py::type_error("Ports do not have Python truthiness; use eg.logical_not(...) or comparisons.");
    })
    .def("__abs__", [](const OutputPort & out) { return make_unary_expr(ExprKind::Abs, make_output_expr(out)); })
    .def("__neg__", [](const OutputPort & out) { return make_unary_expr(ExprKind::Neg, make_output_expr(out)); })
    .def("__invert__", [](const OutputPort & out) { return make_unary_expr(ExprKind::BitNot, make_output_expr(out)); });

  py::class_<InputPort>(m, "InputPort")
    .def_property_readonly("module_name", [](const InputPort & p) { return p.module_name; })
    .def_property_readonly("input_id", [](const InputPort & p) { return p.input_id; })
    .def_property_readonly("expr", [](const InputPort & p) { return current_input_expr(p); })
    .def("assign", [](const InputPort & in, const py::object & value) { assign_input(in, value); }, py::arg("value"))
    .def("assign_index", [](const InputPort & in, const py::object & index, const py::object & value) { assign_input_index(in, index, value); }, py::arg("index"), py::arg("value"))
    .def("__add__", [](const InputPort & lhs, const py::object & rhs) { return add_expr(current_input_expr(lhs), rhs); })
    .def("__radd__", [](const InputPort & rhs, const py::object & lhs) { return add_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__sub__", [](const InputPort & lhs, const py::object & rhs) { return sub_expr(current_input_expr(lhs), rhs); })
    .def("__rsub__", [](const InputPort & rhs, const py::object & lhs) { return sub_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__mul__", [](const InputPort & lhs, const py::object & rhs) { return mul_expr(current_input_expr(lhs), rhs); })
    .def("__rmul__", [](const InputPort & rhs, const py::object & lhs) { return mul_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__truediv__", [](const InputPort & lhs, const py::object & rhs) { return div_expr(current_input_expr(lhs), rhs); })
    .def("__rtruediv__", [](const InputPort & rhs, const py::object & lhs) { return div_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__pow__", [](const InputPort & lhs, const py::object & rhs) { return pow_expr(current_input_expr(lhs), rhs); })
    .def("__rpow__", [](const InputPort & rhs, const py::object & lhs) { return pow_expr(coerce_expr(lhs), py::cast(current_input_expr(rhs))); })
    .def("__matmul__", [](const InputPort & lhs, const py::object & rhs) { return matmul_expr(py::cast(current_input_expr(lhs)), rhs); })
    .def("__rmatmul__", [](const InputPort & rhs, const py::object & lhs) { return matmul_expr(lhs, py::cast(current_input_expr(rhs))); })
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
    .def("__getitem__", [](const InputPort & value, const py::object & index) {
      return index_expr(current_input_expr(value), index);
    })
    .def("__setitem__", [](const InputPort & value, const py::object & index, const py::object & replacement) {
      assign_input_index(value, index, replacement);
    })
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
    .def("__abs__", [](const InputPort & p) { return make_unary_expr(ExprKind::Abs, current_input_expr(p)); })
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

  py::class_<PyModuleInstance>(m, "Module")
    .def_property_readonly("name", &PyModuleInstance::name)
    .def("__getattr__", &PyModuleInstance::getattr, py::arg("name"))
    .def("__setattr__", &PyModuleInstance::setattr, py::arg("name"), py::arg("value"))
    .def("__dir__", &PyModuleInstance::dir)
#ifdef EGRESS_PROFILE
    .def_property_readonly("compile_stats", &PyModuleInstance::compile_stats)
#endif
    ;

  py::class_<PyModuleType>(m, "ModuleType")
    .def_property_readonly("name", &PyModuleType::name)
    .def("__call__", &PyModuleType::instantiate);

  py::class_<PyPureFunctionType>(m, "PureFunction")
    .def("__call__", &PyPureFunctionType::call);

  py::class_<PyStatefulFunctionType>(m, "StatefulFunction")
    .def("__call__", &PyStatefulFunctionType::call);

  m.def(
    "array_state",
    [](const std::string & input, const py::object & init)
    {
      if (!init.is_none())
      {
        (void)scalar_value_from_py(init);
      }
      py::dict spec;
      spec["__egress_array_state__"] = true;
      spec["input"] = input;
      if (!init.is_none())
      {
        spec["init"] = init;
      }
      return spec;
    },
    py::arg("input"),
    py::arg("init") = py::float_(0.0));

  m.def("connect", &connect_ports, py::arg("out"), py::arg("in"));
  m.def("disconnect", &disconnect_ports, py::arg("out"), py::arg("in"));
  m.def("add_output", [](const py::object & out) {
    if (py::isinstance<OutputPort>(out))
    {
      return add_output_port(out.cast<OutputPort>());
    }
    return add_output_expr(out.cast<SignalExpr>());
  }, py::arg("out"));
  m.def("incoming", &incoming_ports, py::arg("in"));
  m.def("abs", [](const py::object & value) { return abs_expr(value); }, py::arg("value"));
  m.def(
    "clamp",
    [](const py::object & value, const py::object & min_value, const py::object & max_value)
    {
      return clamp_expr(value, min_value, max_value);
    },
    py::arg("value"),
    py::arg("min_value"),
    py::arg("max_value"));
  m.def("log", [](const py::object & value) { return log_expr(value); }, py::arg("value"));
  m.def("sin", [](const py::object & value) { return sin_expr(value); }, py::arg("value"));
  m.def("pow", [](const py::object & lhs, const py::object & rhs) { return pow_expr(coerce_expr(lhs), rhs); }, py::arg("lhs"), py::arg("rhs"));
  m.def("array", [](const py::iterable & values) { return make_array_expr(values); }, py::arg("values"));
  m.def("array_set", [](const py::object & array, const py::object & index, const py::object & value) {
    return array_set_expr(coerce_expr(array), index, value);
  }, py::arg("array"), py::arg("index"), py::arg("value"));
  m.def("matrix", [](const py::object & rows) { return make_matrix_expr(rows); }, py::arg("rows"));
  m.def("matmul", [](const py::object & lhs, const py::object & rhs) { return matmul_expr(lhs, rhs); },
        py::arg("lhs"), py::arg("rhs"));
  m.def("logical_not", [](const py::object & value) { return logical_not_expr(value); }, py::arg("value"));
  m.def("sample_rate", []() { return sample_rate_expr(); });
  m.def("sample_index", []() { return sample_index_expr(); });
  m.def(
    "define_module",
    [](const std::string & name,
       const py::iterable & inputs,
       const py::iterable & outputs,
       const py::dict & regs,
       const py::function & process,
       double sample_rate,
       const py::object & input_defaults)
    {
      return PyModuleType(define_module_impl(name, inputs, outputs, regs, process, sample_rate, input_defaults));
    },
    py::arg("name"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("regs"),
    py::arg("process"),
    py::arg("sample_rate") = 44100.0,
    py::arg("input_defaults") = py::none());

  m.def(
    "define_pure_function",
    [](const py::iterable & inputs,
       const py::iterable & outputs,
       const py::function & process)
    {
      return PyPureFunctionType(define_pure_function_impl(inputs, outputs, process));
    },
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("process"));

  m.def(
    "define_stateful_function",
    [](const py::iterable & inputs,
       const py::iterable & outputs,
       const py::dict & regs,
       const py::function & process)
    {
      return PyStatefulFunctionType(define_stateful_function_impl(inputs, outputs, regs, process));
    },
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("regs"),
    py::arg("process"));

  py::class_<PythonDAC>(m, "DAC")
    .def(py::init<unsigned int, unsigned int>(), py::arg("sample_rate") = 44100, py::arg("channels") = 2)
    .def("start", &PythonDAC::start)
    .def("stop", &PythonDAC::stop)
    .def("is_running", &PythonDAC::is_running)
    .def("callback_timing_stats", &PythonDAC::callback_timing_stats)
    .def("reset_callback_timing_stats", &PythonDAC::reset_callback_timing_stats);

}
