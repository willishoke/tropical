"""
MCP tool server wrapping the egress audio graph Python API.

Run with:
    python -m egress.mcp_server
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

logging.getLogger("mcp").setLevel(logging.WARNING)
from typing import Optional

from mcp.server.fastmcp import FastMCP
from ruamel.yaml import YAML

from .graph import Graph
from .audio import DAC
from .param import Param
from .module import ModuleInstance
from .expr import _coerce
from . import _bindings as _b
from . import module_library
from .yaml_schema import load_module_from_yaml

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class Session:
    graph: Graph
    dac: Optional[DAC]
    type_registry: dict          # type_name -> ModuleType
    type_yaml_registry: dict     # type_name -> original yaml_str
    instance_registry: dict      # instance_name -> ModuleInstance
    # Maps instance_name -> {"library_type": str} or {"user_type": str}
    instance_type_map: dict
    connections: list            # [{src, src_output, dst, dst_input}]
    graph_outputs: list          # [{module, output}]
    param_registry: dict         # param_name -> Param


def _make_session() -> Session:
    return Session(
        graph=Graph(),
        dac=None,
        type_registry={},
        type_yaml_registry={},
        instance_registry={},
        instance_type_map={},
        connections=[],
        graph_outputs=[],
        param_registry={},
    )


session: Session = _make_session()

# ---------------------------------------------------------------------------
# Library type map
# ---------------------------------------------------------------------------

_LIBRARY_TYPES = {
    "vco":          lambda: module_library.vco(),
    "reverb":       lambda: module_library.reverb(),
    "compressor":   lambda: module_library.compressor(),
    "phaser":       lambda: module_library.phaser(),
    "ad_envelope":  lambda: module_library.ad_envelope(),
    "bass_drum":    lambda: module_library.bass_drum(),
    "clock":        lambda: module_library.clock(),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yaml() -> YAML:
    y = YAML()
    y.default_flow_style = False
    return y


def _resolve_output_index(inst: ModuleInstance, name_or_index) -> int:
    """Accept a string name or int index; return the integer output index."""
    output_names = inst._def["output_names"]
    if isinstance(name_or_index, int):
        if name_or_index < 0 or name_or_index >= len(output_names):
            raise ValueError(
                f"Output index {name_or_index} out of range "
                f"(module has {len(output_names)} outputs)."
            )
        return name_or_index
    try:
        return output_names.index(name_or_index)
    except ValueError:
        raise ValueError(
            f"Unknown output '{name_or_index}'. "
            f"Available: {output_names}"
        )


def _resolve_input_index(inst: ModuleInstance, name_or_index) -> int:
    """Accept a string name or int index; return the integer input index."""
    input_names = inst._def["input_names"]
    if isinstance(name_or_index, int):
        if name_or_index < 0 or name_or_index >= len(input_names):
            raise ValueError(
                f"Input index {name_or_index} out of range "
                f"(module has {len(input_names)} inputs)."
            )
        return name_or_index
    try:
        return input_names.index(name_or_index)
    except ValueError:
        raise ValueError(
            f"Unknown input '{name_or_index}'. "
            f"Available: {input_names}"
        )


def _instantiate_with_name(mtype, instance_name: str, graph: Graph) -> ModuleInstance:
    """
    Instantiate mtype onto graph using instance_name as the graph-internal name,
    bypassing the auto-naming in ModuleType._instantiate.

    Replicates the logic from load_patch_from_yaml and ModuleType._instantiate.
    """
    spec_h = mtype._build_spec()
    try:
        ok = graph.add_module(instance_name, spec_h)
    finally:
        _b.egress_module_spec_free(spec_h)
    if not ok:
        raise RuntimeError(f"Failed to add module '{instance_name}' to graph.")

    # Apply input defaults (mirrors ModuleType._instantiate)
    for i, default_expr in enumerate(mtype._def.get("input_defaults", [])):
        if default_expr is not None:
            graph.set_input_expr(instance_name, i, default_expr)

    return ModuleInstance(mtype._def, graph, instance_name)


def _port_info(names: list) -> list:
    return [{"name": n, "index": i} for i, n in enumerate(names)]


def _instance_summary(instance_name: str, inst: ModuleInstance) -> dict:
    return {
        "name": instance_name,
        "type_name": inst._def["type_name"],
        "inputs": [n for n in inst._def["input_names"]],
        "outputs": [n for n in inst._def["output_names"]],
    }


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("egress")


# ---- Tool 1: define_module ------------------------------------------------

@mcp.tool()
def define_module(name: str, yaml_str: str) -> dict:
    """
    Define a reusable module type from a YAML string and register it by name.
    Returns the type name and its input/output port names.
    """
    try:
        mtype = load_module_from_yaml(
            yaml_str,
            module_registry=session.type_registry,
            param_registry=session.param_registry,
        )
        session.type_registry[name] = mtype
        session.type_yaml_registry[name] = yaml_str
        return {
            "ok": True,
            "data": {
                "type_name": name,
                "inputs": list(mtype._def["input_names"]),
                "outputs": list(mtype._def["output_names"]),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 2: instantiate_module -------------------------------------------

@mcp.tool()
def instantiate_module(type_name: str, instance_name: str) -> dict:
    """
    Instantiate a module type (from library or user-defined registry) with
    a specific instance name.
    """
    try:
        if instance_name in session.instance_registry:
            return {
                "ok": False,
                "error": f"An instance named '{instance_name}' already exists.",
            }

        if type_name in session.type_registry:
            mtype = session.type_registry[type_name]
            type_tag = {"user_type": type_name}
        elif type_name in _LIBRARY_TYPES:
            mtype = _LIBRARY_TYPES[type_name]()
            type_tag = {"library_type": type_name}
        else:
            return {
                "ok": False,
                "error": (
                    f"Unknown type '{type_name}'. "
                    f"Library types: {sorted(_LIBRARY_TYPES)}. "
                    f"User types: {sorted(session.type_registry)}."
                ),
            }

        inst = _instantiate_with_name(mtype, instance_name, session.graph)
        session.instance_registry[instance_name] = inst
        session.instance_type_map[instance_name] = type_tag

        return {
            "ok": True,
            "data": _instance_summary(instance_name, inst),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 3: remove_module ------------------------------------------------

@mcp.tool()
def remove_module(instance_name: str) -> dict:
    """
    Remove a module instance from the graph.
    Also removes any connections and graph outputs that reference it.
    """
    try:
        inst = session.instance_registry.get(instance_name)
        if inst is None:
            return {"ok": False, "error": f"No instance named '{instance_name}'."}

        session.graph.remove_module(inst._name)
        del session.instance_registry[instance_name]
        session.instance_type_map.pop(instance_name, None)

        session.connections = [
            c for c in session.connections
            if c["src"] != instance_name and c["dst"] != instance_name
        ]
        session.graph_outputs = [
            o for o in session.graph_outputs
            if o["module"] != instance_name
        ]

        return {"ok": True, "data": {"removed": instance_name}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 4: list_modules -------------------------------------------------

@mcp.tool()
def list_modules() -> dict:
    """
    List all live module instances with their port names.
    """
    try:
        modules = [
            _instance_summary(name, inst)
            for name, inst in session.instance_registry.items()
        ]
        return {"ok": True, "data": modules}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 5: get_module_info ----------------------------------------------

@mcp.tool()
def get_module_info(instance_name: str) -> dict:
    """
    Return detailed info about a module instance including its port indices
    and all connections involving it.
    """
    try:
        inst = session.instance_registry.get(instance_name)
        if inst is None:
            return {"ok": False, "error": f"No instance named '{instance_name}'."}

        conns = [
            c for c in session.connections
            if c["src"] == instance_name or c["dst"] == instance_name
        ]

        return {
            "ok": True,
            "data": {
                "name": instance_name,
                "type_name": inst._def["type_name"],
                "inputs": _port_info(inst._def["input_names"]),
                "outputs": _port_info(inst._def["output_names"]),
                "connections": conns,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 6: connect_modules ----------------------------------------------

@mcp.tool()
def connect_modules(
    src_module: str,
    src_output: str,
    dst_module: str,
    dst_input: str,
) -> dict:
    """
    Connect an output port of one module to an input port of another.
    Ports may be specified by name or integer index.
    """
    try:
        src_inst = session.instance_registry.get(src_module)
        if src_inst is None:
            return {"ok": False, "error": f"No instance named '{src_module}'."}
        dst_inst = session.instance_registry.get(dst_module)
        if dst_inst is None:
            return {"ok": False, "error": f"No instance named '{dst_module}'."}

        # Accept int strings from tool callers
        src_out_arg = int(src_output) if src_output.isdigit() else src_output
        dst_in_arg = int(dst_input) if dst_input.isdigit() else dst_input

        src_out_id = _resolve_output_index(src_inst, src_out_arg)
        dst_in_id = _resolve_input_index(dst_inst, dst_in_arg)

        ok = session.graph.connect(
            src_inst._name, src_out_id,
            dst_inst._name, dst_in_id,
        )
        if not ok:
            return {
                "ok": False,
                "error": (
                    f"graph.connect({src_module}.{src_output} → "
                    f"{dst_module}.{dst_input}) returned false."
                ),
            }

        # Normalise to string names for storage
        src_out_name = src_inst._def["output_names"][src_out_id]
        dst_in_name = dst_inst._def["input_names"][dst_in_id]

        conn = {
            "src": src_module,
            "src_output": src_out_name,
            "dst": dst_module,
            "dst_input": dst_in_name,
        }
        session.connections.append(conn)

        return {"ok": True, "data": conn}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 7: disconnect_modules -------------------------------------------

@mcp.tool()
def disconnect_modules(
    src_module: str,
    src_output: str,
    dst_module: str,
    dst_input: str,
) -> dict:
    """
    Disconnect an output port from an input port.
    """
    try:
        src_inst = session.instance_registry.get(src_module)
        if src_inst is None:
            return {"ok": False, "error": f"No instance named '{src_module}'."}
        dst_inst = session.instance_registry.get(dst_module)
        if dst_inst is None:
            return {"ok": False, "error": f"No instance named '{dst_module}'."}

        src_out_arg = int(src_output) if src_output.isdigit() else src_output
        dst_in_arg = int(dst_input) if dst_input.isdigit() else dst_input

        src_out_id = _resolve_output_index(src_inst, src_out_arg)
        dst_in_id = _resolve_input_index(dst_inst, dst_in_arg)

        ok = session.graph.disconnect(
            src_inst._name, src_out_id,
            dst_inst._name, dst_in_id,
        )
        if not ok:
            return {
                "ok": False,
                "error": (
                    f"graph.disconnect({src_module}.{src_output} → "
                    f"{dst_module}.{dst_input}) returned false."
                ),
            }

        src_out_name = src_inst._def["output_names"][src_out_id]
        dst_in_name = dst_inst._def["input_names"][dst_in_id]

        session.connections = [
            c for c in session.connections
            if not (
                c["src"] == src_module
                and c["src_output"] == src_out_name
                and c["dst"] == dst_module
                and c["dst_input"] == dst_in_name
            )
        ]

        return {
            "ok": True,
            "data": {
                "src": src_module,
                "src_output": src_out_name,
                "dst": dst_module,
                "dst_input": dst_in_name,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 8: list_connections ---------------------------------------------

@mcp.tool()
def list_connections() -> dict:
    """
    List all tracked connections in the current patch.
    """
    try:
        return {"ok": True, "data": list(session.connections)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 9: set_module_input ---------------------------------------------

@mcp.tool()
def set_module_input(instance_name: str, input_name: str, value: float) -> dict:
    """
    Set a module input to a constant float value.
    """
    try:
        inst = session.instance_registry.get(instance_name)
        if inst is None:
            return {"ok": False, "error": f"No instance named '{instance_name}'."}

        in_arg = int(input_name) if input_name.isdigit() else input_name
        idx = _resolve_input_index(inst, in_arg)

        session.graph.set_input_expr(inst._name, idx, _coerce(value))

        resolved_name = inst._def["input_names"][idx]
        return {
            "ok": True,
            "data": {"module": instance_name, "input": resolved_name, "value": value},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 10: add_graph_output --------------------------------------------

@mcp.tool()
def add_graph_output(module_name: str, output_name: str) -> dict:
    """
    Add a module output to the graph's mix output.
    """
    try:
        inst = session.instance_registry.get(module_name)
        if inst is None:
            return {"ok": False, "error": f"No instance named '{module_name}'."}

        out_arg = int(output_name) if output_name.isdigit() else output_name
        idx = _resolve_output_index(inst, out_arg)

        ok = session.graph.add_output(inst._name, idx)
        if not ok:
            return {
                "ok": False,
                "error": f"graph.add_output({module_name}.{output_name}) returned false.",
            }

        resolved_name = inst._def["output_names"][idx]
        entry = {"module": module_name, "output": resolved_name}

        # Avoid duplicate tracking
        if entry not in session.graph_outputs:
            session.graph_outputs.append(entry)

        return {"ok": True, "data": entry}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 11: remove_graph_output -----------------------------------------

@mcp.tool()
def remove_graph_output(module_name: str, output_name: str) -> dict:
    """
    Remove a graph output from session tracking.
    Note: the underlying graph has no remove_output API; this only updates
    the session state used by save_patch. The output will still contribute
    to the mix until the graph is rebuilt.
    """
    try:
        inst = session.instance_registry.get(module_name)
        if inst is None:
            return {"ok": False, "error": f"No instance named '{module_name}'."}

        out_arg = int(output_name) if output_name.isdigit() else output_name
        idx = _resolve_output_index(inst, out_arg)
        resolved_name = inst._def["output_names"][idx]

        entry = {"module": module_name, "output": resolved_name}
        before = len(session.graph_outputs)
        session.graph_outputs = [o for o in session.graph_outputs if o != entry]
        removed = before - len(session.graph_outputs)

        return {
            "ok": True,
            "data": {
                "removed_from_tracking": removed,
                "note": (
                    "The graph has no remove_output API; the output continues "
                    "contributing to the mix until the graph is rebuilt."
                ),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 12: load_patch --------------------------------------------------

@mcp.tool()
def load_patch(yaml_str: str) -> dict:
    """
    Load a patch from a YAML string using the mcp_patch_1 schema.
    Stops audio if running, recreates the graph, and rebuilds the session.
    """
    try:
        _yaml = _make_yaml()
        data = _yaml.load(io.StringIO(yaml_str))

        if data.get("schema_version") != "mcp_patch_1":
            return {
                "ok": False,
                "error": (
                    f"Unsupported schema_version: {data.get('schema_version')!r}. "
                    "Expected 'mcp_patch_1'."
                ),
            }

        # Stop audio before tearing down the graph
        if session.dac is not None and session.dac.is_running:
            session.dac.stop()

        # Rebuild session state
        session.graph = Graph()
        session.dac = None
        session.instance_registry.clear()
        session.instance_type_map.clear()
        session.connections.clear()
        session.graph_outputs.clear()
        # Keep param_registry and type_registry alive across patch loads

        # Register any embedded type definitions
        for type_entry in data.get("types", []):
            type_name = type_entry["name"]
            type_yaml = type_entry["yaml"]
            mtype = load_module_from_yaml(
                type_yaml,
                module_registry=session.type_registry,
                param_registry=session.param_registry,
            )
            session.type_registry[type_name] = mtype
            session.type_yaml_registry[type_name] = type_yaml

        # Instantiate modules
        for mod_entry in data.get("modules", []):
            inst_name = mod_entry["name"]
            library_type = mod_entry.get("library_type")
            user_type = mod_entry.get("user_type")

            if library_type is not None:
                if library_type not in _LIBRARY_TYPES:
                    return {
                        "ok": False,
                        "error": (
                            f"Module '{inst_name}': unknown library_type "
                            f"'{library_type}'. "
                            f"Available: {sorted(_LIBRARY_TYPES)}."
                        ),
                    }
                mtype = _LIBRARY_TYPES[library_type]()
                type_tag = {"library_type": library_type}
            elif user_type is not None:
                if user_type not in session.type_registry:
                    return {
                        "ok": False,
                        "error": (
                            f"Module '{inst_name}': unknown user_type '{user_type}'. "
                            f"Registered types: {sorted(session.type_registry)}."
                        ),
                    }
                mtype = session.type_registry[user_type]
                type_tag = {"user_type": user_type}
            else:
                return {
                    "ok": False,
                    "error": (
                        f"Module '{inst_name}' must have either "
                        "'library_type' or 'user_type'."
                    ),
                }

            inst = _instantiate_with_name(mtype, inst_name, session.graph)
            session.instance_registry[inst_name] = inst
            session.instance_type_map[inst_name] = type_tag

        # Apply connections
        for conn in data.get("connections", []):
            src_name = conn["src"]
            src_out_str = conn["src_output"]
            dst_name = conn["dst"]
            dst_in_str = conn["dst_input"]

            src_inst = session.instance_registry.get(src_name)
            if src_inst is None:
                return {"ok": False, "error": f"Connection: unknown src '{src_name}'."}
            dst_inst = session.instance_registry.get(dst_name)
            if dst_inst is None:
                return {"ok": False, "error": f"Connection: unknown dst '{dst_name}'."}

            src_out_id = _resolve_output_index(src_inst, src_out_str)
            dst_in_id = _resolve_input_index(dst_inst, dst_in_str)

            ok = session.graph.connect(
                src_inst._name, src_out_id,
                dst_inst._name, dst_in_id,
            )
            if not ok:
                return {
                    "ok": False,
                    "error": (
                        f"Failed to connect {src_name}.{src_out_str} → "
                        f"{dst_name}.{dst_in_str}."
                    ),
                }

            src_out_name = src_inst._def["output_names"][src_out_id]
            dst_in_name = dst_inst._def["input_names"][dst_in_id]
            session.connections.append({
                "src": src_name,
                "src_output": src_out_name,
                "dst": dst_name,
                "dst_input": dst_in_name,
            })

        # Add graph outputs
        for out_entry in data.get("outputs", []):
            mod_name = out_entry["module"]
            out_str = out_entry["output"]

            inst = session.instance_registry.get(mod_name)
            if inst is None:
                return {"ok": False, "error": f"Output: unknown module '{mod_name}'."}

            out_id = _resolve_output_index(inst, out_str)
            session.graph.add_output(inst._name, out_id)
            resolved_name = inst._def["output_names"][out_id]
            entry = {"module": mod_name, "output": resolved_name}
            if entry not in session.graph_outputs:
                session.graph_outputs.append(entry)

        # Apply params (create or update)
        for param_entry in data.get("params", []):
            param_name = param_entry["name"]
            param_value = float(param_entry["value"])
            if param_name in session.param_registry:
                session.param_registry[param_name].value = param_value
            else:
                session.param_registry[param_name] = Param(param_value)

        return {
            "ok": True,
            "data": {
                "modules": list(session.instance_registry.keys()),
                "connections": len(session.connections),
                "outputs": len(session.graph_outputs),
                "params": list(session.param_registry.keys()),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 13: save_patch --------------------------------------------------

@mcp.tool()
def save_patch() -> dict:
    """
    Serialize the current session to a mcp_patch_1 YAML string.
    """
    try:
        patch: dict = {"schema_version": "mcp_patch_1"}

        # Embed user-defined types
        types_list = []
        for type_name, yaml_str in session.type_yaml_registry.items():
            types_list.append({"name": type_name, "yaml": yaml_str})
        if types_list:
            patch["types"] = types_list

        # Module instances
        modules_list = []
        for inst_name in session.instance_registry:
            type_tag = session.instance_type_map.get(inst_name, {})
            entry: dict = {"name": inst_name}
            entry.update(type_tag)
            modules_list.append(entry)
        if modules_list:
            patch["modules"] = modules_list

        if session.connections:
            patch["connections"] = [dict(c) for c in session.connections]

        if session.graph_outputs:
            patch["outputs"] = [dict(o) for o in session.graph_outputs]

        # Params
        params_list = [
            {"name": name, "value": p.value}
            for name, p in session.param_registry.items()
        ]
        if params_list:
            patch["params"] = params_list

        _yaml = _make_yaml()
        buf = io.StringIO()
        _yaml.dump(patch, buf)

        return {"ok": True, "data": {"yaml": buf.getvalue()}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 14: start_audio -------------------------------------------------

@mcp.tool()
def start_audio(device_name: str = None) -> dict:
    """
    Start audio output.
    Optionally specify a device by name (partial match against DAC.list_devices()).
    """
    try:
        if session.dac is None:
            session.dac = DAC(session.graph)

        if device_name is not None:
            devices = DAC.list_devices()
            matches = [
                d for d in devices
                if device_name.lower() in d["name"].lower()
            ]
            if not matches:
                return {
                    "ok": False,
                    "error": (
                        f"No device matching '{device_name}'. "
                        f"Available: {[d['name'] for d in devices]}."
                    ),
                }
            target_id = matches[0]["id"]
            if session.dac.is_running:
                session.dac.switch_device(target_id)
            else:
                # Start on the specified device: switch after start
                # (DAC opens the default device on start; then switch)
                session.dac.start()
                session.dac.switch_device(target_id)
                return {
                    "ok": True,
                    "data": {
                        "is_running": session.dac.is_running,
                        "device": matches[0]["name"],
                    },
                }

        if not session.dac.is_running:
            session.dac.start()

        return {"ok": True, "data": {"is_running": session.dac.is_running}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 15: stop_audio --------------------------------------------------

@mcp.tool()
def stop_audio() -> dict:
    """
    Stop audio output.
    """
    try:
        if session.dac is None:
            return {"ok": False, "error": "DAC has not been created yet."}
        session.dac.stop()
        return {"ok": True, "data": {"is_running": session.dac.is_running}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 16: audio_status ------------------------------------------------

@mcp.tool()
def audio_status() -> dict:
    """
    Return the current audio status including callback statistics.
    """
    try:
        if session.dac is None:
            return {"ok": True, "data": {"is_running": False}}
        return {
            "ok": True,
            "data": {
                "is_running": session.dac.is_running,
                "is_reconnecting": session.dac.is_reconnecting,
                "stats": session.dac.callback_stats(),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 17: set_param ---------------------------------------------------

@mcp.tool()
def set_param(name: str, value: float) -> dict:
    """
    Set the value of a named Param (thread-safe).
    """
    try:
        p = session.param_registry.get(name)
        if p is None:
            return {
                "ok": False,
                "error": (
                    f"No param named '{name}'. "
                    f"Registered: {sorted(session.param_registry)}."
                ),
            }
        p.value = value
        return {"ok": True, "data": {"name": name, "value": p.value}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Tool 18: list_params -------------------------------------------------

@mcp.tool()
def list_params() -> dict:
    """
    List all registered Params and their current values.
    """
    try:
        params = [
            {"name": name, "value": p.value}
            for name, p in session.param_registry.items()
        ]
        return {"ok": True, "data": params}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
