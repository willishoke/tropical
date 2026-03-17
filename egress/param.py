"""
Param — a control-rate parameter with built-in one-pole smoothing.

A Param wraps an atomic double that can be updated from any thread. When used
in a module expression, each sample is smoothed with a one-pole lowpass whose
time constant is baked in at construction time. The smoothing state is stored
as an anonymous register inside each consuming module, so multiple modules
referencing the same Param each get independent smoothing state.

Usage::

    freq = eg.Param(440.0, time_const=0.01)   # 10 ms smoothing
    vco.freq.assign(freq)                       # assign once at setup

    # from UI / control thread (any frequency):
    freq.value = 550.0   # atomic write; DSP interpolates smoothly
"""

from . import _bindings as _b

__all__ = ["Param"]


class Param:
    """
    Lock-free, auto-smoothed control-rate parameter.

    Parameters
    ----------
    init_value : float
        Initial parameter value (also the starting smoothed value — no startup artifact).
    time_const : float
        One-pole lowpass time constant in seconds (default 5 ms).
        A value of 0.0 means no smoothing (instant response).
    """

    __slots__ = ("_h",)

    def __init__(self, init_value: float, time_const: float = 0.005):
        self._h = _b.check(
            _b.egress_param_new(float(init_value), float(time_const)), "param_new"
        )

    def __del__(self):
        if self._h:
            _b.egress_param_free(self._h)
            self._h = None

    @property
    def value(self) -> float:
        """Current parameter value (atomic load — safe from any thread)."""
        return _b.egress_param_get(self._h)

    @value.setter
    def value(self, v: float):
        """Set parameter value (atomic store — safe from any thread)."""
        _b.egress_param_set(self._h, float(v))

    def _as_expr(self):
        """Return a SmoothedParam SignalExpr for use in module expressions."""
        from .expr import SignalExpr
        return SignalExpr._from_handle(
            _b.check(_b.egress_expr_param(self._h), "expr_param")
        )

    # Arithmetic forwarding — Param can appear directly in expressions
    def __add__(self, rhs):
        return self._as_expr() + rhs

    def __radd__(self, lhs):
        from .expr import _coerce
        return _coerce(lhs) + self._as_expr()

    def __sub__(self, rhs):
        return self._as_expr() - rhs

    def __rsub__(self, lhs):
        from .expr import _coerce
        return _coerce(lhs) - self._as_expr()

    def __mul__(self, rhs):
        return self._as_expr() * rhs

    def __rmul__(self, lhs):
        from .expr import _coerce
        return _coerce(lhs) * self._as_expr()

    def __truediv__(self, rhs):
        return self._as_expr() / rhs

    def __rtruediv__(self, lhs):
        from .expr import _coerce
        return _coerce(lhs) / self._as_expr()

    def __neg__(self):
        return -self._as_expr()

    def __repr__(self):
        return f"Param(value={self.value!r})"
