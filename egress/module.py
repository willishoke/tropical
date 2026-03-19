"""
define_module / define_pure_function DSL — pure Python.

Builds SymbolMaps, calls the user's process function, captures SignalExpr
outputs, then calls the egress_module_spec_* C API to construct the module
spec.  The result is a ModuleType whose instances can be added to a Graph.
"""

import ctypes
from typing import Callable, Dict, List, Optional, Sequence
from . import _bindings as _b
from .expr import (
    SignalExpr, _coerce,
    input_expr, register_expr, nested_output_expr, delay_value_expr,
)

__all__ = [
    "define_module",
    "define_pure_function",
    "delay",
    "array_state",
    "ModuleType",
    "PureFunction",
]

# ---------- Thread-local build context ----------
# Stored as a simple Python stack (list) to avoid threading.local complexity.
# The DSL is single-threaded during module definition.

_context_stack: List["_BuildContext"] = []


def _current_context() -> Optional["_BuildContext"]:
    return _context_stack[-1] if _context_stack else None


class _BuildContext:
    """Accumulates state during a define_module process() call."""

    def __init__(self, input_count: int, sample_rate: float):
        self.input_count = input_count
        self.sample_rate = sample_rate

        # Parallel lists for registers
        self.register_names: List[str] = []
        self.initial_values: List  = []    # raw Python scalars / lists
        self.register_exprs: List[Optional[SignalExpr]] = []
        self.register_array_specs: List[Optional[dict]] = []  # None or array_state spec dict

        # Delay states: list of (node_id, init_value, update_expr: SignalExpr)
        self.delay_states: List[tuple] = []
        # Next delay node id (per-context counter)
        self._delay_node_counter: int = 0

        # Nested module specs: list of (node_id, nested_spec_handle, input_exprs: List[SignalExpr])
        self.nested_modules: List[tuple] = []

    def allocate_delay_node_id(self) -> int:
        nid = self._delay_node_counter
        self._delay_node_counter += 1
        return nid

    def add_delay(self, node_id: int, init_value, update_expr: SignalExpr):
        self.delay_states.append((node_id, init_value, update_expr))

    def __enter__(self):
        _context_stack.append(self)
        return self

    def __exit__(self, *_):
        _context_stack.pop()


# ---------- SymbolMap ----------

class SymbolMap:
    """Maps signal names → SignalExpr slot references."""

    def __init__(self, kind: str, names: List[str]):
        self._kind = kind
        self._slots: Dict[str, int] = {n: i for i, n in enumerate(names)}

    def __getitem__(self, name: str) -> SignalExpr:
        if name not in self._slots:
            raise KeyError(f"Unknown {self._kind} symbol '{name}'.")
        slot_id = self._slots[name]
        if self._kind == "input":
            return input_expr(slot_id)
        else:
            return register_expr(slot_id)


# ---------- array_state helper ----------

class _ArrayStateSpec:
    """Sentinel returned by array_state(); used to mark dynamic array registers."""
    def __init__(self, input_name: str, init):
        self.input_name = input_name
        self.init = init


def array_state(input: str, init=0.0) -> _ArrayStateSpec:
    """
    Declare a dynamic array register whose size mirrors a named input.
    Used inside define_module regs dict values.
    """
    return _ArrayStateSpec(input, init)


# ---------- delay() DSL function ----------

def delay(value, init=None) -> SignalExpr:
    """
    Create a one-sample delay within a define_module process body.
    Returns a SignalExpr referencing the previous-tick value.
    """
    ctx = _current_context()
    if ctx is None:
        raise RuntimeError(
            "delay() may only be called inside define_module process bodies."
        )
    sig = _coerce(value)
    node_id = ctx.allocate_delay_node_id()
    init_val = 0.0 if init is None else init
    ctx.add_delay(node_id, init_val, sig)
    return delay_value_expr(node_id)


# ---------- Value helpers ----------

def _scalar_value_handle(v):
    """Return a new egress_value_t handle for a Python scalar."""
    if isinstance(v, bool):
        return _b.check(_b.egress_value_bool(v), "value_bool")
    if isinstance(v, int):
        return _b.check(_b.egress_value_int(v), "value_int")
    if isinstance(v, float):
        return _b.check(_b.egress_value_float(v), "value_float")
    raise TypeError(f"Expected bool/int/float, got {type(v).__name__}")


def _value_handle(v):
    """Return an egress_value_t handle for a Python scalar, list, or nested list."""
    if isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
        return _scalar_value_handle(v)
    if isinstance(v, (list, tuple)):
        # Detect matrix (list of lists)
        if v and isinstance(v[0], (list, tuple)):
            rows = len(v)
            cols = len(v[0])
            items = []
            for row in v:
                for item in row:
                    items.append(_scalar_value_handle(item))
            arr = (ctypes.c_void_p * len(items))(*items)
            h = _b.check(_b.egress_value_matrix(arr, rows, cols), "value_matrix")
            for item_h in items:
                _b.egress_value_free(item_h)
            return h
        else:
            items = [_scalar_value_handle(item) for item in v]
            arr = (ctypes.c_void_p * len(items))(*items)
            h = _b.check(_b.egress_value_array(arr, len(items)), "value_array")
            for item_h in items:
                _b.egress_value_free(item_h)
            return h
    raise TypeError(f"Cannot convert {type(v).__name__} to egress value")


# ---------- ModuleType ----------

class ModuleType:
    """
    A compiled module type.  Call instances to create Module instances on the
    default graph, or call inside a define_module process to create nested refs.
    """

    def __init__(self, definition: dict, graph=None):
        """
        definition keys:
          type_name, input_names, output_names, register_names,
          initial_values, output_exprs, register_exprs,
          delay_states, nested_modules, register_array_specs, sample_rate
        """
        self._def = definition
        self._graph = graph  # default graph for top-level instantiation
        self._yaml_source = None  # set by load_module_from_yaml for round-trip

    @property
    def name(self) -> str:
        return self._def["type_name"]

    def __call__(self, *args, graph=None):
        """
        If called inside define_module: create a nested module call expression.
        Otherwise: instantiate the module on the graph.
        """
        ctx = _current_context()
        if ctx is not None:
            return self._nested_call(ctx, args)
        return self._instantiate(graph or self._graph, args)

    def _instantiate(self, graph, args):
        if args:
            raise TypeError(
                "Module types only accept signal arguments inside define_module "
                "process bodies."
            )
        if graph is None:
            raise RuntimeError(
                "No graph provided for module instantiation. "
                "Pass graph= or use inside a Graph context."
            )
        name = graph.next_name(self._def["type_name"])
        spec_h = self._build_spec()
        try:
            ok = graph.add_module(name, spec_h)
        finally:
            _b.egress_module_spec_free(spec_h)
        if not ok:
            raise RuntimeError(f"Failed to add module '{name}' to graph.")

        # Set input defaults
        for i, default_expr in enumerate(self._def.get("input_defaults", [])):
            if default_expr is not None:
                graph.set_input_expr(name, i, default_expr)

        return ModuleInstance(self._def, graph, name)

    def _nested_call(self, ctx: _BuildContext, args):
        """Inside a build context: create nested module spec and return output exprs."""
        d = self._def
        input_names = d["input_names"]

        if len(args) > len(input_names):
            raise TypeError(
                f"Module call expects at most {len(input_names)} arguments."
            )

        # Coerce call-site arguments; fill with defaults for missing ones
        call_args: List[SignalExpr] = []
        for i, arg in enumerate(args):
            call_args.append(_coerce(arg))
        input_defaults = d.get("input_defaults", [])
        for i in range(len(args), len(input_names)):
            if i < len(input_defaults) and input_defaults[i] is not None:
                call_args.append(input_defaults[i])
            else:
                raise TypeError(
                    f"Missing argument for module input '{input_names[i]}'."
                )

        # Build the nested spec
        nested_h = _b.check(
            _b.egress_nested_spec_new(len(input_names), d["sample_rate"]),
            "nested_spec_new",
        )
        node_id = int(_b.egress_nested_spec_node_id(nested_h))

        # Add input expressions (call-site arguments)
        for arg in call_args:
            _b.egress_nested_spec_add_input_expr(nested_h, arg._h)

        # Add output expressions (the module's own output computations)
        for expr_h in d["output_expr_handles"]:
            _b.egress_nested_spec_add_output(nested_h, expr_h)

        # Add registers
        for body_h, init_h, array_spec in d["register_specs"]:
            if array_spec is not None:
                _b.egress_nested_spec_add_register_array(
                    nested_h, array_spec["source_input_id"], init_h)
            else:
                _b.egress_nested_spec_add_register(nested_h, body_h, init_h)

        # Add delay states
        for delay_node_id, init_h, update_h in d["delay_spec_handles"]:
            _b.egress_nested_spec_add_delay_state(nested_h, init_h, update_h)

        # Add inner nested modules (e.g. delay lines inside comb filters)
        for _, inner_nested_h in d.get("nested_spec_handles", []):
            _b.egress_nested_spec_add_nested(nested_h, inner_nested_h)

        # Register with context
        ctx.nested_modules.append((node_id, nested_h))

        # Return output expressions
        output_names = d["output_names"]
        outputs = [
            nested_output_expr(node_id, out_id)
            for out_id in range(len(output_names))
        ]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _build_spec(self):
        """Build and return an egress_module_spec_t handle (caller must free)."""
        d = self._def
        spec_h = _b.check(
            _b.egress_module_spec_new(
                len(d["input_names"]), d["sample_rate"]
            ),
            "module_spec_new",
        )
        for expr_h in d["output_expr_handles"]:
            _b.egress_module_spec_add_output(spec_h, expr_h)
        for body_h, init_h, array_spec in d["register_specs"]:
            if array_spec is not None:
                _b.egress_module_spec_add_register_array(
                    spec_h, array_spec["source_input_id"], init_h)
            else:
                _b.egress_module_spec_add_register(spec_h, body_h, init_h)
        for delay_node_id, init_h, update_h in d["delay_spec_handles"]:
            _b.egress_module_spec_add_delay_state(
                spec_h, init_h, update_h)
        for _, nested_h in d.get("nested_spec_handles", []):
            _b.egress_module_spec_add_nested(spec_h, nested_h)
        return spec_h


# ---------- ModuleInstance ----------

class ModuleInstance:
    """A live module instance attached to a graph."""

    def __init__(self, definition: dict, graph, name: str):
        self._def = definition
        self._graph = graph
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __getattr__(self, attr: str):
        d = self._def
        for i, n in enumerate(d.get("input_names", [])):
            if n == attr:
                return _InputPort(self._graph, self._name, i)
        for i, n in enumerate(d.get("output_names", [])):
            if n == attr:
                return _OutputPort(self._graph, self._name, i)
        raise AttributeError(f"Unknown attribute '{attr}'.")

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            object.__setattr__(self, attr, value)
            return
        d = self._def
        for i, n in enumerate(d.get("input_names", [])):
            if n == attr:
                self._graph.set_input_expr(self._name, i, _coerce(value))
                return
        raise AttributeError(f"Cannot assign attribute '{attr}'.")


class _OutputPort:
    def __init__(self, graph, module_name: str, output_id: int):
        self._graph = graph
        self.module_name = module_name
        self.output_id = output_id

    def _as_expr(self) -> SignalExpr:
        from .expr import ref_expr
        return ref_expr(self.module_name, self.output_id)

    # Forward all arithmetic/comparison to a SignalExpr
    def __add__(self, rhs):  return self._as_expr() + rhs
    def __radd__(self, lhs): return _coerce(lhs) + self._as_expr()
    def __sub__(self, rhs):  return self._as_expr() - rhs
    def __rsub__(self, lhs): return _coerce(lhs) - self._as_expr()
    def __mul__(self, rhs):  return self._as_expr() * rhs
    def __rmul__(self, lhs): return _coerce(lhs) * self._as_expr()
    def __truediv__(self, rhs):  return self._as_expr() / rhs
    def __rtruediv__(self, lhs): return _coerce(lhs) / self._as_expr()
    def __pow__(self, rhs):  return self._as_expr() ** rhs
    def __rpow__(self, lhs): return _coerce(lhs) ** self._as_expr()
    def __neg__(self):  return -self._as_expr()
    def __abs__(self):  return abs(self._as_expr())
    def __getitem__(self, idx): return self._as_expr()[idx]
    def __lt__(self, rhs): return self._as_expr() < rhs
    def __le__(self, rhs): return self._as_expr() <= rhs
    def __gt__(self, rhs): return self._as_expr() > rhs
    def __ge__(self, rhs): return self._as_expr() >= rhs
    def __eq__(self, rhs): return self._as_expr().__eq__(rhs)
    def __ne__(self, rhs): return self._as_expr().__ne__(rhs)
    def __bool__(self):
        raise TypeError("Ports have no Python truthiness.")


class _InputPort:
    def __init__(self, graph, module_name: str, input_id: int):
        self._graph = graph
        self.module_name = module_name
        self.input_id = input_id

    @property
    def expr(self) -> Optional[SignalExpr]:
        return self._graph.get_input_expr(self.module_name, self.input_id)

    def assign(self, value):
        self._graph.set_input_expr(self.module_name, self.input_id, _coerce(value))

    def _current(self) -> SignalExpr:
        e = self.expr
        if e is None:
            return _coerce(0.0)
        return e

    def __iadd__(self, rhs): self.assign(self._current() + rhs); return self
    def __isub__(self, rhs): self.assign(self._current() - rhs); return self
    def __imul__(self, rhs): self.assign(self._current() * rhs); return self
    def __itruediv__(self, rhs): self.assign(self._current() / rhs); return self
    def __bool__(self):
        raise TypeError("Ports have no Python truthiness.")


# ---------- define_module ----------

def define_module(
    name: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    regs: dict,
    process: Callable,
    sample_rate: float = 44100.0,
    input_defaults: Optional[dict] = None,
) -> ModuleType:
    """
    Define a reusable module type.

    process(inputs_map, regs_map) -> (outputs_dict, next_regs_dict)
    """
    input_names = list(inputs)
    output_names = list(outputs)

    # Build register info from the regs dict (order-preserving)
    reg_names = []
    reg_initial_values = []
    reg_array_specs = []  # None or {"source_input_id": int, "init": scalar}

    for reg_name, reg_init in regs.items():
        reg_names.append(reg_name)
        if isinstance(reg_init, _ArrayStateSpec):
            spec_obj = reg_init
            src_id = input_names.index(spec_obj.input_name)
            reg_initial_values.append([])  # empty array placeholder
            reg_array_specs.append({"source_input_id": src_id, "init": spec_obj.init})
        else:
            reg_initial_values.append(reg_init)
            reg_array_specs.append(None)

    # Parse input_defaults
    parsed_defaults: List[Optional[SignalExpr]] = [None] * len(input_names)
    if input_defaults:
        for k, v in input_defaults.items():
            idx = input_names.index(k)
            parsed_defaults[idx] = _coerce(v) if v is not None else None

    # Build symbol maps
    inputs_map = SymbolMap("input", input_names)
    regs_map = SymbolMap("register", reg_names)

    # Run the process function inside a build context
    ctx = _BuildContext(len(input_names), sample_rate)
    with ctx:
        result = process(inputs_map, regs_map)

    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("process must return a tuple: (outputs_dict, next_regs_dict).")

    out_dict, next_regs_dict = result
    if not isinstance(out_dict, dict):
        raise TypeError("process outputs must be a dict.")
    if not isinstance(next_regs_dict, dict):
        raise TypeError("process next_regs must be a dict.")

    # Collect output expressions (as SignalExpr handles)
    output_exprs: List[SignalExpr] = [None] * len(output_names)
    for out_name, out_val in out_dict.items():
        idx = output_names.index(out_name)
        output_exprs[idx] = _coerce(out_val)

    for i, e in enumerate(output_exprs):
        if e is None:
            raise ValueError(f"Output '{output_names[i]}' was not assigned.")

    # Collect register update expressions
    reg_update_exprs: List[Optional[SignalExpr]] = [None] * len(reg_names)
    for reg_name, reg_val in next_regs_dict.items():
        idx = reg_names.index(reg_name)
        if reg_array_specs[idx] is not None:
            # Array state registers don't have an update expression
            reg_update_exprs[idx] = None
        else:
            reg_update_exprs[idx] = _coerce(reg_val)

    # Build egress_value_t handles for register initial values
    # These are owned by the definition dict and freed when no longer needed.
    # For simplicity we keep them alive in the definition.
    reg_spec_tuples = []  # (body_handle, init_handle, array_spec_or_None)
    _live_value_handles = []
    _live_expr_handles = []

    for i, (init_val, array_spec, update_expr) in enumerate(
        zip(reg_initial_values, reg_array_specs, reg_update_exprs)
    ):
        init_h = _value_handle(init_val if array_spec is None else
                                (array_spec["init"] if array_spec["init"] is not None else 0.0))
        _live_value_handles.append(init_h)

        if array_spec is not None:
            reg_spec_tuples.append((None, init_h, array_spec))
        else:
            body_h = update_expr._h if update_expr is not None else None
            reg_spec_tuples.append((body_h, init_h, None))

    # Build delay spec tuples
    delay_spec_tuples = []
    for node_id, init_py_val, update_expr in ctx.delay_states:
        init_h = _value_handle(init_py_val if init_py_val is not None else 0.0)
        _live_value_handles.append(init_h)
        update_h = update_expr._h
        delay_spec_tuples.append((node_id, init_h, update_h))

    # Keep handles for output expressions
    output_expr_handles = [e._h for e in output_exprs]

    definition = {
        "type_name": name,
        "input_names": input_names,
        "output_names": output_names,
        "register_names": reg_names,
        "sample_rate": sample_rate,
        "input_defaults": parsed_defaults,
        "output_expr_handles": output_expr_handles,
        "register_specs": reg_spec_tuples,
        "delay_spec_handles": delay_spec_tuples,
        "nested_spec_handles": [(nid, nh) for nid, nh in ctx.nested_modules],
        # Keep Python objects alive to prevent GC of their C handles
        "_live_output_exprs": output_exprs,
        "_live_reg_update_exprs": reg_update_exprs,
        "_live_value_handles": _live_value_handles,
        "_live_delay_update_exprs": [ds[2] for ds in ctx.delay_states],
    }

    return ModuleType(definition)


# ---------- define_pure_function ----------

class PureFunction:
    """A pure stateless function callable from inside define_module."""

    def __init__(self, definition: dict):
        self._def = definition

    def __call__(self, *args) -> SignalExpr:
        d = self._def
        if len(args) != len(d["input_names"]):
            raise TypeError(
                f"PureFunction expects {len(d['input_names'])} arguments."
            )
        call_args = [_coerce(a) for a in args]
        arg_handles = [a._h for a in call_args]
        arr = (ctypes.c_void_p * len(arg_handles))(*arg_handles)

        outputs = []
        for body_h in d["output_expr_handles"]:
            fn_h = _b.check(
                _b.egress_expr_function(len(d["input_names"]), body_h),
                "expr_function",
            )
            call_h = _b.check(
                _b.egress_expr_call(fn_h, arr, len(arg_handles)),
                "expr_call",
            )
            _b.egress_expr_free(fn_h)
            outputs.append(SignalExpr._from_handle(call_h))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def define_pure_function(
    inputs: Sequence[str],
    outputs: Sequence[str],
    process: Callable,
) -> PureFunction:
    """
    Define a stateless pure function.

    process(inputs_map) -> outputs_dict
    """
    input_names = list(inputs)
    output_names = list(outputs)

    inputs_map = SymbolMap("input", input_names)
    result = process(inputs_map)

    if not isinstance(result, dict):
        raise TypeError("process must return a dict of outputs.")

    output_exprs: List[Optional[SignalExpr]] = [None] * len(output_names)
    for out_name, out_val in result.items():
        idx = output_names.index(out_name)
        output_exprs[idx] = _coerce(out_val)

    for i, e in enumerate(output_exprs):
        if e is None:
            raise ValueError(f"Output '{output_names[i]}' was not assigned.")

    definition = {
        "input_names": input_names,
        "output_names": output_names,
        "output_expr_handles": [e._h for e in output_exprs],
        "_live_output_exprs": output_exprs,
    }
    return PureFunction(definition)
