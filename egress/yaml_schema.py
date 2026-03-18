"""
Declarative YAML schema for Egress modules and patches.

Public API:
    load_module_from_yaml(yaml_str, module_registry={}, param_registry={}) -> ModuleType
    load_patch_from_yaml(yaml_str, base_dir, param_registry={}) -> (Graph, instances_dict)
    save_patch_to_yaml(graph, instances, type_file_map, connections=(), patch_outputs=(), uses_map={}) -> str
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ruamel.yaml import YAML

from . import _bindings as _b
from .expr import (
    SignalExpr, _coerce,
    input_expr, register_expr, nested_output_expr, delay_value_expr,
    clamp, select, array, array_set,
    sample_rate as sample_rate_expr,
    sample_index as sample_index_expr,
)
from .module import (
    ModuleType, _BuildContext,
    _value_handle,
)

__all__ = [
    "load_module_from_yaml",
    "load_patch_from_yaml",
    "save_patch_to_yaml",
]

# ---- YAML setup ----

def _make_yaml() -> YAML:
    y = YAML()
    y.version = (1, 2)
    return y


# ---- Op dispatch tables ----

_UNARY_OPS: Dict[str, int] = {
    "sin":     _b.EXPR_SIN,
    "log":     _b.EXPR_LOG,
    "abs":     _b.EXPR_ABS,
    "neg":     _b.EXPR_NEG,
    "not":     _b.EXPR_NOT,
    "bit_not": _b.EXPR_BIT_NOT,
}

_BINARY_OPS: Dict[str, int] = {
    "add":           _b.EXPR_ADD,
    "sub":           _b.EXPR_SUB,
    "mul":           _b.EXPR_MUL,
    "div":           _b.EXPR_DIV,
    "floor_div":     _b.EXPR_FLOOR_DIV,
    "mod":           _b.EXPR_MOD,
    "pow":           _b.EXPR_POW,
    "matmul":        _b.EXPR_MATMUL,
    "bit_and":       _b.EXPR_BIT_AND,
    "bit_or":        _b.EXPR_BIT_OR,
    "bit_xor":       _b.EXPR_BIT_XOR,
    "lshift":        _b.EXPR_LSHIFT,
    "rshift":        _b.EXPR_RSHIFT,
    "less":          _b.EXPR_LESS,
    "less_equal":    _b.EXPR_LESS_EQUAL,
    "greater":       _b.EXPR_GREATER,
    "greater_equal": _b.EXPR_GREATER_EQUAL,
    "equal":         _b.EXPR_EQUAL,
    "not_equal":     _b.EXPR_NOT_EQUAL,
}


# ---- LoadContext ----

@dataclass
class _LoadContext:
    """Tracks name→index mappings and resolved node references during YAML loading."""
    input_names: List[str]
    register_names: List[str]
    delay_id_map: Dict[str, int]                  # yaml id str → C node_id int
    nested_output_map: Dict[str, List[SignalExpr]] # yaml id str → output expr list
    param_registry: Dict[str, Any]                 # str → Param
    module_registry: Dict[str, Any]                # str → ModuleType


# ---- Expression builder ----

def _build_expr(node: dict, lctx: _LoadContext) -> SignalExpr:
    """Recursively build a SignalExpr from a YAML ExprNode dict."""
    op = node["op"]
    args = node.get("args", [])

    if op == "literal":
        if "items" in node:
            return array([_coerce(float(v)) for v in node["items"]])
        return _coerce(node["value"])

    if op == "input":
        idx = lctx.input_names.index(node["name"])
        return input_expr(idx)

    if op == "register":
        idx = lctx.register_names.index(node["name"])
        return register_expr(idx)

    if op == "param":
        p = lctx.param_registry[node["name"]]
        return p._as_expr()

    if op == "trigger_param":
        p = lctx.param_registry[node["name"]]
        return p._as_expr()

    if op == "sample_rate":
        return sample_rate_expr()

    if op == "sample_index":
        return sample_index_expr()

    if op == "delay":
        node_id = lctx.delay_id_map[node["id"]]
        return delay_value_expr(node_id)

    if op == "nested_output":
        outputs = lctx.nested_output_map[node["nested_id"]]
        return outputs[node["output_index"]]

    if op == "nested_input":
        # Used inside inline specs; maps to the inline module's own input by index
        return input_expr(node["index"])

    if op in _UNARY_OPS:
        a = _build_expr(args[0], lctx)
        h = _b.check(_b.egress_expr_unary(_UNARY_OPS[op], a._h), f"unary:{op}")
        return SignalExpr._from_handle(h)

    if op in _BINARY_OPS:
        l = _build_expr(args[0], lctx)
        r = _build_expr(args[1], lctx)
        h = _b.check(_b.egress_expr_binary(_BINARY_OPS[op], l._h, r._h), f"binary:{op}")
        return SignalExpr._from_handle(h)

    if op == "clamp":
        v = _build_expr(args[0], lctx)
        lo = _build_expr(args[1], lctx)
        hi = _build_expr(args[2], lctx)
        return clamp(v, lo, hi)

    if op == "select":
        cond = _build_expr(args[0], lctx)
        then_val = _build_expr(args[1], lctx)
        else_val = _build_expr(args[2], lctx)
        return select(cond, then_val, else_val)

    if op == "index":
        arr_e = _build_expr(args[0], lctx)
        idx_e = _build_expr(args[1], lctx)
        return arr_e[idx_e]

    if op == "array_set":
        arr_e = _build_expr(args[0], lctx)
        idx_e = _build_expr(args[1], lctx)
        val_e = _build_expr(args[2], lctx)
        return array_set(arr_e, idx_e, val_e)

    if op == "array_pack":
        return array([_build_expr(a, lctx) for a in args])

    raise ValueError(f"Unknown expr op: {op!r}")


# ---- Parse register initial value ----

def _parse_init(init_val):
    """Parse register init: scalar float or {items: [...]} for array."""
    if isinstance(init_val, dict) and "items" in init_val:
        return [float(v) for v in init_val["items"]]
    return float(init_val)


# ---- Nested module builders ----

def _build_nested_modules(
    nested_specs: list,
    lctx: _LoadContext,
    build_ctx: _BuildContext,
    live_objects: list,
):
    """Build nested module entries, registering outputs in lctx.nested_output_map."""
    for spec in nested_specs:
        nid = spec["id"]
        if "type" in spec:
            mtype = lctx.module_registry[spec["type"]]
            input_exprs = [_build_expr(e, lctx) for e in spec.get("input_exprs", [])]
            result = mtype._nested_call(build_ctx, input_exprs)
            if isinstance(result, tuple):
                lctx.nested_output_map[nid] = list(result)
            else:
                lctx.nested_output_map[nid] = [result]
        elif "inline" in spec:
            inline = spec["inline"]
            call_input_exprs = [_build_expr(e, lctx) for e in spec.get("input_exprs", [])]
            outputs = _build_inline_nested(
                inline, lctx, build_ctx, call_input_exprs, live_objects
            )
            lctx.nested_output_map[nid] = outputs
        else:
            raise ValueError(
                f"Nested module {nid!r} must have either 'type' or 'inline' key"
            )


def _build_inline_nested(
    inline_spec: dict,
    parent_lctx: _LoadContext,
    outer_build_ctx: _BuildContext,
    call_input_exprs: List[SignalExpr],
    live_objects: list,
) -> List[SignalExpr]:
    """
    Build an inline nested module spec and register it with outer_build_ctx.
    Returns a list of SignalExprs referencing the nested module's outputs.
    """
    input_count = inline_spec.get("input_count", len(call_input_exprs))
    sample_rate = float(inline_spec.get("sample_rate", 44100.0))

    # Register names for the inline sub-context
    inline_reg_names = [r["name"] for r in inline_spec.get("registers", [])]
    for r in inline_spec.get("array_registers", []):
        inline_reg_names.append(r["name"])

    # Pre-assign sequential delay IDs for the inline module's own delays
    inline_delay_id_map: Dict[str, int] = {}
    for i, ds in enumerate(inline_spec.get("delay_states", [])):
        inline_delay_id_map[ds["id"]] = i

    sub_lctx = _LoadContext(
        input_names=[],  # inline inputs accessed via nested_input by index
        register_names=inline_reg_names,
        delay_id_map=inline_delay_id_map,
        nested_output_map={},
        param_registry=parent_lctx.param_registry,
        module_registry=parent_lctx.module_registry,
    )

    # Build any sub-nested modules for this inline spec using a temp context,
    # so their handles are added directly to this inline nested_h (not the outer one)
    temp_ctx = _BuildContext(input_count, sample_rate)
    _build_nested_modules(
        inline_spec.get("nested_modules", []),
        sub_lctx, temp_ctx, live_objects
    )

    # Create the nested spec handle
    nested_h = _b.check(
        _b.egress_nested_spec_new(input_count, sample_rate),
        "nested_spec_new",
    )
    node_id = int(_b.egress_nested_spec_node_id(nested_h))

    # Add call-site input exprs
    for e in call_input_exprs:
        _b.egress_nested_spec_add_input_expr(nested_h, e._h)

    # Add sub-nested modules (built via temp_ctx) to this inline nested_h
    for _, inner_h in temp_ctx.nested_modules:
        _b.egress_nested_spec_add_nested(nested_h, inner_h)

    # Build and add output exprs
    output_exprs_raw = inline_spec.get("output_exprs", {})
    output_exprs_dict: Dict[int, SignalExpr] = {}
    for idx_str, expr_node in output_exprs_raw.items():
        e = _build_expr(expr_node, sub_lctx)
        output_exprs_dict[int(idx_str)] = e
        live_objects.append(e)
    for i in range(len(output_exprs_dict)):
        _b.egress_nested_spec_add_output(nested_h, output_exprs_dict[i]._h)

    # Build registers
    for reg_spec in inline_spec.get("registers", []):
        init_val = _parse_init(reg_spec["init"])
        init_h = _value_handle(init_val)
        live_objects.append(init_h)
        update_expr = _build_expr(reg_spec["update"], sub_lctx)
        live_objects.append(update_expr)
        _b.egress_nested_spec_add_register(nested_h, update_expr._h, init_h)

    for arr_r in inline_spec.get("array_registers", []):
        src_idx = int(arr_r.get("source_input_index", 0))
        init_h = _value_handle(float(arr_r.get("init_scalar", 0.0)))
        live_objects.append(init_h)
        _b.egress_nested_spec_add_register_array(nested_h, src_idx, init_h)

    # Build delay states
    for ds in inline_spec.get("delay_states", []):
        init_h = _value_handle(float(ds.get("init", 0.0)))
        live_objects.append(init_h)
        update_expr = _build_expr(ds["update"], sub_lctx)
        live_objects.append(update_expr)
        _b.egress_nested_spec_add_delay_state(nested_h, init_h, update_expr._h)

    # Register with outer build context so the top-level module spec includes it
    outer_build_ctx.nested_modules.append((node_id, nested_h))

    return [nested_output_expr(node_id, i) for i in range(len(output_exprs_dict))]


# ---- Main YAML loader ----

def load_module_from_yaml(
    yaml_str: str,
    module_registry: dict = {},
    param_registry: dict = {},
) -> ModuleType:
    """
    Load a ModuleType from a YAML string.

    module_registry: str → ModuleType, for resolving 'type:' nested module references.
    param_registry:  str → Param,       for resolving 'param' expr nodes.
    """
    _yaml = _make_yaml()
    data = _yaml.load(io.StringIO(yaml_str))

    if data.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported schema_version: {data.get('schema_version')!r}. Expected 1."
        )

    name = data["name"]
    sample_rate = float(data.get("sample_rate", 44100.0))
    input_names = list(data.get("inputs", []))
    output_names = list(data.get("outputs", []))

    reg_specs_raw = list(data.get("registers", []))
    arr_reg_specs_raw = list(data.get("array_registers", []))
    delay_specs_raw = list(data.get("delay_states", []))
    nested_specs_raw = list(data.get("nested_modules", []))
    output_exprs_raw = dict(data.get("output_exprs", {}))

    # All register names in declaration order (regular first, then array)
    all_reg_names = (
        [r["name"] for r in reg_specs_raw]
        + [r["name"] for r in arr_reg_specs_raw]
    )

    lctx = _LoadContext(
        input_names=input_names,
        register_names=all_reg_names,
        delay_id_map={},
        nested_output_map={},
        param_registry=param_registry,
        module_registry=module_registry,
    )

    # Extra live objects: value handles and exprs for inline nested specs
    _extra_live: list = []

    build_ctx = _BuildContext(len(input_names), sample_rate)
    with build_ctx:
        # Pre-assign C delay node IDs (sequential, matching C API assignment order)
        for ds in delay_specs_raw:
            nid = build_ctx.allocate_delay_node_id()
            lctx.delay_id_map[ds["id"]] = nid

        # Build nested modules (registers them in build_ctx.nested_modules)
        _build_nested_modules(nested_specs_raw, lctx, build_ctx, _extra_live)

        # Build delay update exprs and register in build_ctx.delay_states
        for ds in delay_specs_raw:
            node_id = lctx.delay_id_map[ds["id"]]
            init_val = float(ds.get("init", 0.0))
            update_expr = _build_expr(ds["update"], lctx)
            build_ctx.add_delay(node_id, init_val, update_expr)

        # Build register update exprs (regular only; array registers have none)
        reg_update_exprs: List[Optional[SignalExpr]] = []
        for r in reg_specs_raw:
            reg_update_exprs.append(_build_expr(r["update"], lctx))
        for _ in arr_reg_specs_raw:
            reg_update_exprs.append(None)

        # Build output exprs
        output_exprs: List[SignalExpr] = []
        for out_name in output_names:
            if out_name not in output_exprs_raw:
                raise ValueError(f"Output '{out_name}' not found in output_exprs")
            output_exprs.append(_build_expr(output_exprs_raw[out_name], lctx))

    # ---- Assemble definition dict (mirrors define_module's format) ----

    _live_value_handles: list = []
    reg_spec_tuples: list = []

    for i, r in enumerate(reg_specs_raw):
        init_val = _parse_init(r["init"])
        init_h = _value_handle(init_val)
        _live_value_handles.append(init_h)
        update_expr = reg_update_exprs[i]
        body_h = update_expr._h if update_expr is not None else None
        reg_spec_tuples.append((body_h, init_h, None))

    for arr_r in arr_reg_specs_raw:
        src_id = input_names.index(arr_r["source_input"])
        init_scalar = float(arr_r.get("init_scalar", 0.0))
        init_h = _value_handle(init_scalar)
        _live_value_handles.append(init_h)
        array_spec = {"source_input_id": src_id, "init": init_scalar}
        reg_spec_tuples.append((None, init_h, array_spec))

    delay_spec_tuples: list = []
    for node_id, init_py_val, update_expr in build_ctx.delay_states:
        init_h = _value_handle(init_py_val if init_py_val is not None else 0.0)
        _live_value_handles.append(init_h)
        delay_spec_tuples.append((node_id, init_h, update_expr._h))

    definition = {
        "type_name": name,
        "input_names": input_names,
        "output_names": output_names,
        "register_names": all_reg_names,
        "sample_rate": sample_rate,
        "input_defaults": [None] * len(input_names),
        "output_expr_handles": [e._h for e in output_exprs],
        "register_specs": reg_spec_tuples,
        "delay_spec_handles": delay_spec_tuples,
        "nested_spec_handles": list(build_ctx.nested_modules),
        # Keep Python objects alive to prevent GC of their C handles
        "_live_output_exprs": output_exprs,
        "_live_reg_update_exprs": reg_update_exprs,
        "_live_value_handles": _live_value_handles,
        "_live_delay_update_exprs": [ds[2] for ds in build_ctx.delay_states],
        "_live_extra": _extra_live,
    }

    return ModuleType(definition)


# ---- Patch loader ----

def load_patch_from_yaml(
    yaml_str: str,
    base_dir: Path,
    param_registry: dict = {},
) -> tuple:
    """
    Load a patch from YAML: instantiate modules, apply connections, add outputs.

    Returns (graph, instances_dict) where instances_dict maps name → ModuleInstance.
    """
    from .graph import Graph

    _yaml = _make_yaml()
    data = _yaml.load(io.StringIO(yaml_str))

    if data.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported patch schema_version: {data.get('schema_version')!r}"
        )

    base_dir = Path(base_dir)

    # ---- Load module types ----
    module_registry: Dict[str, ModuleType] = {}
    module_type_map: Dict[str, ModuleType] = {}  # instance name → type

    for mod_entry in data.get("modules", []):
        inst_name = mod_entry["name"]
        type_file = mod_entry.get("type_file")
        if type_file is None:
            raise ValueError(f"Module '{inst_name}' missing 'type_file'")

        # Load 'uses' sub-registries first
        sub_registry: Dict[str, ModuleType] = {}
        for dep_name, dep_file in mod_entry.get("uses", {}).items():
            dep_path = base_dir / dep_file
            dep_yaml = dep_path.read_text()
            dep_type = load_module_from_yaml(dep_yaml, {}, param_registry)
            sub_registry[dep_name] = dep_type

        type_path = base_dir / type_file
        type_yaml = type_path.read_text()
        mtype = load_module_from_yaml(type_yaml, sub_registry, param_registry)
        module_type_map[inst_name] = mtype
        module_registry[mtype.name] = mtype

    # ---- Create graph and instantiate modules ----
    g = Graph(512)
    instances: Dict[str, Any] = {}

    for mod_entry in data.get("modules", []):
        inst_name = mod_entry["name"]
        mtype = module_type_map[inst_name]

        # Override the graph for instantiation
        spec_h = mtype._build_spec()
        try:
            ok = g.add_module(inst_name, spec_h)
        finally:
            _b.egress_module_spec_free(spec_h)
        if not ok:
            raise RuntimeError(f"Failed to add module '{inst_name}' to graph.")

        from .module import ModuleInstance
        instances[inst_name] = ModuleInstance(mtype._def, g, inst_name)

    # ---- Apply connections ----
    for conn in data.get("connections", []):
        src_name = conn["src"]
        src_output = conn["src_output"]
        dst_name = conn["dst"]
        dst_input = conn["dst_input"]

        src_def = module_type_map[src_name]._def
        dst_def = module_type_map[dst_name]._def

        src_out_id = src_def["output_names"].index(src_output)
        dst_in_id = dst_def["input_names"].index(dst_input)

        ok = g.connect(src_name, src_out_id, dst_name, dst_in_id)
        if not ok:
            raise RuntimeError(
                f"Failed to connect {src_name}.{src_output} → {dst_name}.{dst_input}"
            )

    # ---- Add outputs ----
    for out_entry in data.get("outputs", []):
        mod_name = out_entry["module"]
        out_name = out_entry["output"]
        mdef = module_type_map[mod_name]._def
        out_id = mdef["output_names"].index(out_name)
        g.add_output(mod_name, out_id)

    return g, instances


# ---- Patch saver ----

def save_patch_to_yaml(
    graph,
    instances: dict,
    type_file_map: dict,
    connections: list = (),
    patch_outputs: list = (),
    uses_map: dict = (),
) -> str:
    """
    Serialize a patch to YAML.

    instances:     dict of instance_name → ModuleInstance
    type_file_map: dict of instance_name → relative YAML file path string
    connections:   list of dicts with keys: src, src_output, dst, dst_input
    patch_outputs: list of dicts with keys: module, output
    uses_map:      dict of instance_name → {TypeName: file_path} for sub-registries
    """
    _yaml = _make_yaml()

    modules_list = []
    for inst_name, inst in instances.items():
        entry: dict = {
            "name": inst_name,
            "type_file": type_file_map[inst_name],
        }
        sub_uses = dict(uses_map).get(inst_name, {}) if uses_map else {}
        if sub_uses:
            entry["uses"] = dict(sub_uses)
        modules_list.append(entry)

    patch = {
        "schema_version": 1,
        "modules": modules_list,
    }

    if connections:
        patch["connections"] = [dict(c) for c in connections]

    if patch_outputs:
        patch["outputs"] = [dict(o) for o in patch_outputs]

    buf = io.StringIO()
    _yaml.dump(patch, buf)
    return buf.getvalue()
