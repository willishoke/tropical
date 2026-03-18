"""
egress — DSP graph DSL with a stable C shared library backend.

This package is a pure-Python ctypes wrapper around libegress.
No Python-version-specific binary is required; a single libegress.dylib/.so
works across all CPython versions.

Quick start
-----------
    import egress as eg

    g = eg.Graph(512)

    Osc = eg.define_module(
        "Osc",
        inputs=["freq"],
        outputs=["out"],
        regs={"phase": 0.0},
        process=lambda i, r: (
            {"out": eg.sin(r["phase"] * 2.0 * 3.14159)},
            {"phase": r["phase"] + i["freq"] / eg.sample_rate()},
        ),
    )
    osc = Osc(graph=g)
    osc.freq.assign(440.0)
    eg.add_output(osc.out, graph=g)
    g.process()
    print(g.output_buffer[:4])
"""

from .graph import Graph
from .expr import (
    SignalExpr,
    sin,
    log,
    abs_ as abs,
    logical_not,
    clamp,
    select,
    pow_ as pow,
    matmul,
    array,
    matrix,
    array_set,
    sample_rate,
    sample_index,
)
from .module import (
    define_module,
    define_pure_function,
    delay,
    array_state,
    ModuleType,
    PureFunction,
    _OutputPort,
    _InputPort,
)
from .audio import DAC
from .param import Param
from .udp import UDPParamListener, parse_text, parse_osc
from .yaml_schema import load_module_from_yaml, load_patch_from_yaml, save_patch_to_yaml


def connect(out, inp):
    """Connect an output port to an input port."""
    if not isinstance(out, _OutputPort) or not isinstance(inp, _InputPort):
        raise TypeError("connect() requires OutputPort and InputPort arguments.")
    if out._graph is not inp._graph:
        raise ValueError("Ports belong to different graphs.")
    return out._graph.connect(
        out.module_name, out.output_id, inp.module_name, inp.input_id
    )


def disconnect(out, inp):
    """Disconnect an output port from an input port."""
    if not isinstance(out, _OutputPort) or not isinstance(inp, _InputPort):
        raise TypeError("disconnect() requires OutputPort and InputPort arguments.")
    if out._graph is not inp._graph:
        raise ValueError("Ports belong to different graphs.")
    return out._graph.disconnect(
        out.module_name, out.output_id, inp.module_name, inp.input_id
    )


def add_output(port_or_expr, graph=None):
    """Add a graph mix output from an output port or a SignalExpr."""
    if isinstance(port_or_expr, _OutputPort):
        g = port_or_expr._graph
        return g.add_output(port_or_expr.module_name, port_or_expr.output_id)
    if isinstance(port_or_expr, SignalExpr):
        if graph is None:
            raise ValueError(
                "graph= must be provided when adding a SignalExpr output."
            )
        return graph.add_output_expr(port_or_expr)
    raise TypeError(
        "add_output() requires an OutputPort or SignalExpr."
    )


__all__ = [
    "Graph",
    "SignalExpr",
    "sin", "log", "abs", "logical_not", "clamp", "select", "pow", "matmul",
    "array", "matrix", "array_set",
    "sample_rate", "sample_index",
    "define_module", "define_pure_function",
    "delay", "array_state",
    "ModuleType", "PureFunction",
    "DAC",
    "connect", "disconnect", "add_output",
    "Param",
    "UDPParamListener", "parse_text", "parse_osc",
    "load_module_from_yaml", "load_patch_from_yaml", "save_patch_to_yaml",
]
