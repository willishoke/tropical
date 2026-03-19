"""
Graph — wraps egress_graph_t via ctypes.
"""

import ctypes
from . import _bindings as _b
from .expr import SignalExpr

__all__ = ["Graph"]


class Graph:
    """
    An egress processing graph.  Manages a set of modules, connections,
    and mix outputs.
    """

    def __init__(self, buffer_length: int = 512):
        self._h = _b.check(_b.egress_graph_new(buffer_length), "graph_new")
        self._buffer_length = buffer_length
        self._name_counters = {}

    def __del__(self):
        if self._h:
            _b.egress_graph_free(self._h)
            self._h = None

    # ---- Module management ----

    def add_module(self, name: str, spec) -> bool:
        """Add a module to the graph from a module spec handle (c_void_p int)."""
        return bool(_b.egress_graph_add_module(
            self._h, name.encode(), spec))

    def remove_module(self, name: str) -> bool:
        return bool(_b.egress_graph_remove_module(self._h, name.encode()))

    # Alias used by the DSL
    destroy_module = remove_module

    # ---- Connections ----

    def connect(
        self,
        src_module: str,
        src_output_id: int,
        dst_module: str,
        dst_input_id: int,
    ) -> bool:
        return bool(_b.egress_graph_connect(
            self._h,
            src_module.encode(),
            src_output_id,
            dst_module.encode(),
            dst_input_id,
        ))

    def disconnect(
        self,
        src_module: str,
        src_output_id: int,
        dst_module: str,
        dst_input_id: int,
    ) -> bool:
        return bool(_b.egress_graph_disconnect(
            self._h,
            src_module.encode(),
            src_output_id,
            dst_module.encode(),
            dst_input_id,
        ))

    # ---- Input expressions ----

    def set_input_expr(
        self,
        module_name: str,
        input_id: int,
        expr,  # SignalExpr or None
    ) -> bool:
        handle = expr._h if expr is not None else None
        return bool(_b.egress_graph_set_input_expr(
            self._h, module_name.encode(), input_id, handle))

    def get_input_expr(self, module_name: str, input_id: int):
        h = _b.egress_graph_get_input_expr(
            self._h, module_name.encode(), input_id)
        if not h:
            return None
        return SignalExpr._from_handle(h)

    # ---- Outputs ----

    def add_output(self, module_name: str, output_id: int) -> bool:
        return bool(_b.egress_graph_add_output(
            self._h, module_name.encode(), output_id))

    def add_output_expr(self, expr: SignalExpr) -> bool:
        return bool(_b.egress_graph_add_output_expr(self._h, expr._h))

    def add_output_tap(self, module_name: str, output_id: int) -> int:
        tap_id = _b.egress_graph_add_output_tap(
            self._h, module_name.encode(), output_id)
        return int(tap_id)

    def remove_output_tap(self, tap_id: int) -> bool:
        return bool(_b.egress_graph_remove_output_tap(self._h, tap_id))

    # ---- Processing ----

    def process(self):
        _b.egress_graph_process(self._h)

    def prime_numeric_jit(self):
        _b.egress_graph_prime_jit(self._h)

    # ---- Buffers ----

    @property
    def output_buffer(self):
        """
        Return a memoryview (read-only, float64) over the output buffer.
        Valid until the next process() call.
        """
        length = _b.egress_graph_get_buffer_length(self._h)
        ptr = _b.egress_graph_output_buffer(self._h)
        if not ptr:
            return []
        return list(ptr[:length])

    def output_tap_buffer(self, tap_id: int):
        """Return a list of doubles for the given tap."""
        out_len = ctypes.c_size_t(0)
        ptr = _b.egress_graph_tap_buffer(self._h, tap_id, ctypes.byref(out_len))
        if not ptr:
            return []
        return list(ptr[: out_len.value])

    # ---- Configuration ----

    def set_worker_count(self, n: int):
        _b.egress_graph_set_worker_count(self._h, n)

    @property
    def worker_count(self) -> int:
        return int(_b.egress_graph_get_worker_count(self._h))

    @worker_count.setter
    def worker_count(self, n: int):
        self.set_worker_count(n)

    def set_fusion_enabled(self, enabled: bool):
        _b.egress_graph_set_fusion_enabled(self._h, enabled)

    @property
    def fusion_enabled(self) -> bool:
        return bool(_b.egress_graph_get_fusion_enabled(self._h))

    @fusion_enabled.setter
    def fusion_enabled(self, v: bool):
        self.set_fusion_enabled(v)

    @property
    def buffer_length(self) -> int:
        return int(_b.egress_graph_get_buffer_length(self._h))

    # ---- Name generation (used by DSL) ----

    def next_name(self, prefix: str) -> str:
        count = self._name_counters.get(prefix, 0) + 1
        self._name_counters[prefix] = count
        return f"{prefix}{count}"
