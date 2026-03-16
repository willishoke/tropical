"""
SignalExpr — symbolic expression wrapper for the egress DSL.

Holds a c_void_p handle to an egress_expr_t. Ownership is tracked by the
Python object; __del__ calls egress_expr_free.

All arithmetic / comparison / bitwise operator overloads produce new
SignalExpr objects by calling the C API binary/unary expression factories.
"""

import ctypes
from . import _bindings as _b

__all__ = [
    "SignalExpr",
    "sin", "log", "abs_", "logical_not",
    "clamp", "pow_", "matmul",
    "array", "array_set",
    "sample_rate", "sample_index",
    "input_expr", "register_expr", "ref_expr",
    "nested_output_expr", "delay_value_expr",
]


def _coerce(value):
    """Convert a Python scalar or SignalExpr to a SignalExpr."""
    if isinstance(value, SignalExpr):
        return value
    if isinstance(value, bool):
        return SignalExpr._from_handle(_b.check(
            _b.egress_expr_literal_bool(value), "literal_bool"))
    if isinstance(value, int):
        return SignalExpr._from_handle(_b.check(
            _b.egress_expr_literal_int(value), "literal_int"))
    if isinstance(value, float):
        return SignalExpr._from_handle(_b.check(
            _b.egress_expr_literal_float(value), "literal_float"))
    if isinstance(value, (list, tuple)):
        return array(value)
    raise TypeError(
        f"Cannot coerce {type(value).__name__} to SignalExpr; "
        "expected SignalExpr, bool, int, float, or list/tuple."
    )


def _binary(kind, lhs, rhs):
    l = _coerce(lhs) if not isinstance(lhs, SignalExpr) else lhs
    r = _coerce(rhs) if not isinstance(rhs, SignalExpr) else rhs
    h = _b.check(_b.egress_expr_binary(kind, l._h, r._h), "binary_expr")
    return SignalExpr._from_handle(h)


def _unary(kind, operand):
    op = _coerce(operand) if not isinstance(operand, SignalExpr) else operand
    h = _b.check(_b.egress_expr_unary(kind, op._h), "unary_expr")
    return SignalExpr._from_handle(h)


class SignalExpr:
    """
    Symbolic signal expression.  Wraps an opaque egress_expr_t handle.
    """

    __slots__ = ("_h",)

    def __init__(self):
        # Use _from_handle to construct; direct construction is disabled.
        self._h = None

    @classmethod
    def _from_handle(cls, handle):
        obj = object.__new__(cls)
        obj._h = handle
        return obj

    def __del__(self):
        if self._h:
            _b.egress_expr_free(self._h)
            self._h = None

    # Prevent accidental truthiness testing
    def __bool__(self):
        raise TypeError(
            "Symbolic expressions have no Python truthiness; "
            "use logical_not() or comparison operators."
        )

    # ---- Arithmetic ----
    def __add__(self, rhs):  return _binary(_b.EXPR_ADD, self, rhs)
    def __radd__(self, lhs): return _binary(_b.EXPR_ADD, lhs, self)
    def __sub__(self, rhs):  return _binary(_b.EXPR_SUB, self, rhs)
    def __rsub__(self, lhs): return _binary(_b.EXPR_SUB, lhs, self)
    def __mul__(self, rhs):  return _binary(_b.EXPR_MUL, self, rhs)
    def __rmul__(self, lhs): return _binary(_b.EXPR_MUL, lhs, self)
    def __truediv__(self, rhs):  return _binary(_b.EXPR_DIV, self, rhs)
    def __rtruediv__(self, lhs): return _binary(_b.EXPR_DIV, lhs, self)
    def __floordiv__(self, rhs):  return _binary(_b.EXPR_FLOOR_DIV, self, rhs)
    def __rfloordiv__(self, lhs): return _binary(_b.EXPR_FLOOR_DIV, lhs, self)
    def __mod__(self, rhs):  return _binary(_b.EXPR_MOD, self, rhs)
    def __rmod__(self, lhs): return _binary(_b.EXPR_MOD, lhs, self)
    def __pow__(self, rhs):  return _binary(_b.EXPR_POW, self, rhs)
    def __rpow__(self, lhs): return _binary(_b.EXPR_POW, lhs, self)
    def __matmul__(self, rhs):  return _binary(_b.EXPR_MATMUL, self, rhs)
    def __rmatmul__(self, lhs): return _binary(_b.EXPR_MATMUL, lhs, self)

    # ---- Bitwise ----
    def __and__(self, rhs):   return _binary(_b.EXPR_BIT_AND, self, rhs)
    def __rand__(self, lhs):  return _binary(_b.EXPR_BIT_AND, lhs, self)
    def __or__(self, rhs):    return _binary(_b.EXPR_BIT_OR, self, rhs)
    def __ror__(self, lhs):   return _binary(_b.EXPR_BIT_OR, lhs, self)
    def __xor__(self, rhs):   return _binary(_b.EXPR_BIT_XOR, self, rhs)
    def __rxor__(self, lhs):  return _binary(_b.EXPR_BIT_XOR, lhs, self)
    def __lshift__(self, rhs):  return _binary(_b.EXPR_LSHIFT, self, rhs)
    def __rlshift__(self, lhs): return _binary(_b.EXPR_LSHIFT, lhs, self)
    def __rshift__(self, rhs):  return _binary(_b.EXPR_RSHIFT, self, rhs)
    def __rrshift__(self, lhs): return _binary(_b.EXPR_RSHIFT, lhs, self)

    # ---- Comparison ----
    def __lt__(self, rhs): return _binary(_b.EXPR_LESS, self, rhs)
    def __le__(self, rhs): return _binary(_b.EXPR_LESS_EQUAL, self, rhs)
    def __gt__(self, rhs): return _binary(_b.EXPR_GREATER, self, rhs)
    def __ge__(self, rhs): return _binary(_b.EXPR_GREATER_EQUAL, self, rhs)
    def __eq__(self, rhs): return _binary(_b.EXPR_EQUAL, self, rhs)
    def __ne__(self, rhs): return _binary(_b.EXPR_NOT_EQUAL, self, rhs)

    # ---- Unary ----
    def __abs__(self):    return _unary(_b.EXPR_ABS, self)
    def __neg__(self):    return _unary(_b.EXPR_NEG, self)
    def __invert__(self): return _unary(_b.EXPR_BIT_NOT, self)

    # ---- Array indexing ----
    def __getitem__(self, idx):
        i = _coerce(idx)
        h = _b.check(_b.egress_expr_index(self._h, i._h), "expr_index")
        return SignalExpr._from_handle(h)


# ---------- Module-level DSL functions ----------

def sin(value):
    return _unary(_b.EXPR_SIN, value)

def log(value):
    return _unary(_b.EXPR_LOG, value)

def abs_(value):
    return _unary(_b.EXPR_ABS, value)

def logical_not(value):
    return _unary(_b.EXPR_NOT, value)

def clamp(value, lo, hi):
    v = _coerce(value)
    l = _coerce(lo)
    h_hi = _coerce(hi)
    h = _b.check(_b.egress_expr_clamp(v._h, l._h, h_hi._h), "clamp")
    return SignalExpr._from_handle(h)

def pow_(lhs, rhs):
    return _binary(_b.EXPR_POW, lhs, rhs)

def matmul(lhs, rhs):
    return _binary(_b.EXPR_MATMUL, lhs, rhs)

def array(values):
    """Create an array-pack expression from a list of scalars/SignalExprs."""
    items = [_coerce(v) for v in values]
    handles = [e._h for e in items]
    arr = (ctypes.c_void_p * len(handles))(*handles)
    h = _b.check(_b.egress_expr_array_pack(arr, len(handles)), "array_pack")
    return SignalExpr._from_handle(h)


def matrix(rows):
    """Create a matrix literal expression from a list of rows (list of lists of floats)."""
    from .module import _scalar_value_handle
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    item_handles = []
    for row in rows:
        for item in row:
            item_handles.append(_scalar_value_handle(float(item)))
    arr = (ctypes.c_void_p * len(item_handles))(*item_handles)
    val_h = _b.check(_b.egress_value_matrix(arr, n_rows, n_cols), "value_matrix")
    for ih in item_handles:
        _b.egress_value_free(ih)
    expr_h = _b.check(_b.egress_expr_literal_value(val_h), "literal_value")
    _b.egress_value_free(val_h)
    return SignalExpr._from_handle(expr_h)

def array_set(arr_expr, idx, val):
    a = _coerce(arr_expr)
    i = _coerce(idx)
    v = _coerce(val)
    h = _b.check(_b.egress_expr_array_set(a._h, i._h, v._h), "array_set")
    return SignalExpr._from_handle(h)

def sample_rate():
    h = _b.check(_b.egress_expr_sample_rate(), "sample_rate")
    return SignalExpr._from_handle(h)

def sample_index():
    h = _b.check(_b.egress_expr_sample_index(), "sample_index")
    return SignalExpr._from_handle(h)

def input_expr(input_id: int) -> "SignalExpr":
    h = _b.check(_b.egress_expr_input(input_id), "expr_input")
    return SignalExpr._from_handle(h)

def register_expr(reg_id: int) -> "SignalExpr":
    h = _b.check(_b.egress_expr_register(reg_id), "expr_register")
    return SignalExpr._from_handle(h)

def ref_expr(module_name: str, output_id: int) -> "SignalExpr":
    h = _b.check(
        _b.egress_expr_ref(module_name.encode(), output_id), "expr_ref")
    return SignalExpr._from_handle(h)

def nested_output_expr(node_id: int, output_id: int) -> "SignalExpr":
    h = _b.check(
        _b.egress_expr_nested_output(node_id, output_id), "nested_output")
    return SignalExpr._from_handle(h)

def delay_value_expr(node_id: int) -> "SignalExpr":
    h = _b.check(_b.egress_expr_delay_value(node_id), "delay_value")
    return SignalExpr._from_handle(h)
