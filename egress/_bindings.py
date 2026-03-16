"""
ctypes loader for libegress — declares all argtypes/restypes.
All opaque handles are c_void_p.
"""

import ctypes
import ctypes.util
import os
import sys

# ---------- Load the shared library ----------

def _find_lib():
    # 1. Next to this package (e.g. egress/libegress.dylib)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("libegress.dylib", "libegress.so", "egress.dll"):
        candidate = os.path.join(here, name)
        if os.path.exists(candidate):
            return candidate

    # 2. Parent directory (common cmake build-tree layout)
    parent = os.path.dirname(here)
    for name in ("libegress.dylib", "libegress.so", "egress.dll"):
        candidate = os.path.join(parent, name)
        if os.path.exists(candidate):
            return candidate

    # 3. Common build subdirectories
    for build_dir in ("build", "build-jit-profile", "build-jit-ctypes", "build-jit", "build-profile", "build-ctypes"):
        candidate_dir = os.path.join(parent, build_dir)
        for name in ("libegress.dylib", "libegress.so"):
            candidate = os.path.join(candidate_dir, name)
            if os.path.exists(candidate):
                return candidate

    # 4. System search path
    found = ctypes.util.find_library("egress")
    if found:
        return found

    raise OSError(
        "libegress not found. Build with cmake (target egress_core) and ensure "
        "libegress.dylib/.so is on the library path or adjacent to the egress "
        "Python package directory."
    )


_lib = ctypes.CDLL(_find_lib())

# Convenience type aliases
_c   = ctypes.c_void_p
_b   = ctypes.c_bool
_d   = ctypes.c_double
_i64 = ctypes.c_int64
_u   = ctypes.c_uint
_sz  = ctypes.c_size_t
_str = ctypes.c_char_p
_int = ctypes.c_int


def _fn(name, restype, *argtypes):
    func = getattr(_lib, name)
    func.restype = restype
    func.argtypes = list(argtypes)
    return func


# ---------- Error API ----------
egress_last_error = _fn("egress_last_error", _str)

# ---------- Value API ----------
egress_value_float   = _fn("egress_value_float",   _c, _d)
egress_value_int     = _fn("egress_value_int",     _c, _i64)
egress_value_bool    = _fn("egress_value_bool",    _c, _b)
egress_value_array   = _fn("egress_value_array",   _c, ctypes.POINTER(_c), _sz)
egress_value_matrix  = _fn("egress_value_matrix",  _c, ctypes.POINTER(_c), _sz, _sz)
egress_value_to_float = _fn("egress_value_to_float", _d, _c)
egress_value_to_int   = _fn("egress_value_to_int",   _i64, _c)
egress_value_free    = _fn("egress_value_free",    None, _c)

# ---------- Expression factory API ----------
egress_expr_literal_float  = _fn("egress_expr_literal_float",  _c, _d)
egress_expr_literal_int    = _fn("egress_expr_literal_int",    _c, _i64)
egress_expr_literal_bool   = _fn("egress_expr_literal_bool",   _c, _b)
egress_expr_literal_value  = _fn("egress_expr_literal_value",  _c, _c)
egress_expr_input          = _fn("egress_expr_input",          _c, _u)
egress_expr_register       = _fn("egress_expr_register",       _c, _u)
egress_expr_nested_output  = _fn("egress_expr_nested_output",  _c, _u, _u)
egress_expr_delay_value    = _fn("egress_expr_delay_value",    _c, _u)
egress_expr_ref            = _fn("egress_expr_ref",            _c, _str, _u)
egress_expr_sample_rate    = _fn("egress_expr_sample_rate",    _c)
egress_expr_sample_index   = _fn("egress_expr_sample_index",   _c)
egress_expr_unary          = _fn("egress_expr_unary",          _c, _int, _c)
egress_expr_binary         = _fn("egress_expr_binary",         _c, _int, _c, _c)
egress_expr_clamp          = _fn("egress_expr_clamp",          _c, _c, _c, _c)
egress_expr_array_pack     = _fn("egress_expr_array_pack",     _c, ctypes.POINTER(_c), _sz)
egress_expr_index          = _fn("egress_expr_index",          _c, _c, _c)
egress_expr_array_set      = _fn("egress_expr_array_set",      _c, _c, _c, _c)
egress_expr_function       = _fn("egress_expr_function",       _c, _u, _c)
egress_expr_call           = _fn("egress_expr_call",           _c, _c, ctypes.POINTER(_c), _sz)
egress_expr_free           = _fn("egress_expr_free",           None, _c)

# ---------- Module spec builder API ----------
egress_module_spec_new                = _fn("egress_module_spec_new",                _c, _u, _d)
egress_module_spec_add_output         = _fn("egress_module_spec_add_output",         None, _c, _c)
egress_module_spec_add_register       = _fn("egress_module_spec_add_register",       None, _c, _c, _c)
egress_module_spec_add_register_array = _fn("egress_module_spec_add_register_array", None, _c, _u, _c)
egress_module_spec_add_delay_state    = _fn("egress_module_spec_add_delay_state",    _u,   _c, _c, _c)
egress_module_spec_add_nested         = _fn("egress_module_spec_add_nested",         None, _c, _c)
egress_module_spec_set_composite_schedule = _fn(
    "egress_module_spec_set_composite_schedule", None, _c, ctypes.POINTER(_u), _sz)
egress_module_spec_set_output_boundary = _fn(
    "egress_module_spec_set_output_boundary", None, _c, _u)
egress_module_spec_free               = _fn("egress_module_spec_free",               None, _c)

# ---------- Nested spec builder API ----------
egress_nested_spec_new                = _fn("egress_nested_spec_new",                _c, _u, _d)
egress_nested_spec_node_id            = _fn("egress_nested_spec_node_id",            _u, _c)
egress_nested_spec_add_input_expr     = _fn("egress_nested_spec_add_input_expr",     None, _c, _c)
egress_nested_spec_add_output         = _fn("egress_nested_spec_add_output",         None, _c, _c)
egress_nested_spec_add_register       = _fn("egress_nested_spec_add_register",       None, _c, _c, _c)
egress_nested_spec_add_register_array = _fn("egress_nested_spec_add_register_array", None, _c, _u, _c)
egress_nested_spec_add_delay_state    = _fn("egress_nested_spec_add_delay_state",    _u,   _c, _c, _c)
egress_nested_spec_add_nested         = _fn("egress_nested_spec_add_nested",         None, _c, _c)
egress_nested_spec_set_composite_schedule = _fn(
    "egress_nested_spec_set_composite_schedule", None, _c, ctypes.POINTER(_u), _sz)
egress_nested_spec_set_output_boundary = _fn(
    "egress_nested_spec_set_output_boundary", None, _c, _u)
egress_nested_spec_free               = _fn("egress_nested_spec_free",               None, _c)

# ---------- Graph API ----------
egress_graph_new               = _fn("egress_graph_new",               _c, _u)
egress_graph_free              = _fn("egress_graph_free",              None, _c)
egress_graph_add_module        = _fn("egress_graph_add_module",        _b,  _c, _str, _c)
egress_graph_remove_module     = _fn("egress_graph_remove_module",     _b,  _c, _str)
egress_graph_connect           = _fn("egress_graph_connect",           _b,  _c, _str, _u, _str, _u)
egress_graph_disconnect        = _fn("egress_graph_disconnect",        _b,  _c, _str, _u, _str, _u)
egress_graph_set_input_expr    = _fn("egress_graph_set_input_expr",    _b,  _c, _str, _u, _c)
egress_graph_get_input_expr    = _fn("egress_graph_get_input_expr",    _c,  _c, _str, _u)
egress_graph_add_output        = _fn("egress_graph_add_output",        _b,  _c, _str, _u)
egress_graph_add_output_expr   = _fn("egress_graph_add_output_expr",   _b,  _c, _c)
egress_graph_add_output_tap    = _fn("egress_graph_add_output_tap",    _sz, _c, _str, _u)
egress_graph_remove_output_tap = _fn("egress_graph_remove_output_tap", _b,  _c, _sz)
egress_graph_process           = _fn("egress_graph_process",           None, _c)
egress_graph_prime_jit         = _fn("egress_graph_prime_jit",         None, _c)
egress_graph_output_buffer     = _fn("egress_graph_output_buffer",     ctypes.POINTER(_d), _c)
egress_graph_tap_buffer        = _fn(
    "egress_graph_tap_buffer", ctypes.POINTER(_d), _c, _sz, ctypes.POINTER(_sz))
egress_graph_set_worker_count  = _fn("egress_graph_set_worker_count",  None, _c, _u)
egress_graph_get_worker_count  = _fn("egress_graph_get_worker_count",  _u,   _c)
egress_graph_set_fusion_enabled = _fn("egress_graph_set_fusion_enabled", None, _c, _b)
egress_graph_get_fusion_enabled = _fn("egress_graph_get_fusion_enabled", _b,   _c)
egress_graph_get_buffer_length  = _fn("egress_graph_get_buffer_length",  _u,   _c)

# ---------- DAC API ----------
egress_dac_new        = _fn("egress_dac_new",        _c,   _c, _u, _u)
egress_dac_free       = _fn("egress_dac_free",       None, _c)
egress_dac_start      = _fn("egress_dac_start",      None, _c)
egress_dac_stop       = _fn("egress_dac_stop",       None, _c)
egress_dac_is_running = _fn("egress_dac_is_running", _b,   _c)

class EgressDacStats(ctypes.Structure):
    _fields_ = [
        ("callback_count",  ctypes.c_uint64),
        ("avg_callback_ms", ctypes.c_double),
        ("max_callback_ms", ctypes.c_double),
        ("underrun_count",  ctypes.c_uint64),
        ("overrun_count",   ctypes.c_uint64),
    ]

egress_dac_get_stats   = _fn("egress_dac_get_stats",   None, _c, ctypes.POINTER(EgressDacStats))
egress_dac_reset_stats = _fn("egress_dac_reset_stats", None, _c)

# ---------- ExprKind constants ----------
EXPR_LITERAL        = 0
EXPR_REF            = 1
EXPR_INPUT          = 2
EXPR_REGISTER       = 3
EXPR_NESTED         = 4
EXPR_DELAY          = 5
EXPR_SAMPLE_RATE    = 6
EXPR_SAMPLE_INDEX   = 7
EXPR_FUNCTION       = 8
EXPR_CALL           = 9
EXPR_ARRAY_PACK     = 10
EXPR_INDEX          = 11
EXPR_ARRAY_SET      = 12
EXPR_NOT            = 13
EXPR_LESS           = 14
EXPR_LESS_EQUAL     = 15
EXPR_GREATER        = 16
EXPR_GREATER_EQUAL  = 17
EXPR_EQUAL          = 18
EXPR_NOT_EQUAL      = 19
EXPR_ADD            = 20
EXPR_SUB            = 21
EXPR_MUL            = 22
EXPR_DIV            = 23
EXPR_MATMUL         = 24
EXPR_POW            = 25
EXPR_MOD            = 26
EXPR_FLOOR_DIV      = 27
EXPR_BIT_AND        = 28
EXPR_BIT_OR         = 29
EXPR_BIT_XOR        = 30
EXPR_LSHIFT         = 31
EXPR_RSHIFT         = 32
EXPR_ABS            = 33
EXPR_CLAMP          = 34
EXPR_LOG            = 35
EXPR_SIN            = 36
EXPR_NEG            = 37
EXPR_BIT_NOT        = 38


# ---------- Helpers ----------

def check(handle, name="operation"):
    """Raise RuntimeError if handle is None/0, including last C error message."""
    if not handle:
        msg = egress_last_error()
        raise RuntimeError(
            f"{name} failed: {msg.decode() if msg else '(no error)'}"
        )
    return handle


def make_handle_array(handles):
    """Build a ctypes c_void_p array from a list of integer handles."""
    arr = (ctypes.c_void_p * len(handles))(*handles)
    return arr, len(handles)


def make_uint_array(ints):
    """Build a ctypes c_uint array from a list of ints."""
    arr = (ctypes.c_uint * len(ints))(*ints)
    return arr, len(ints)
