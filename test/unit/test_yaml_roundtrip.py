"""
Round-trip tests: load module from YAML → save back to YAML → reload → verify
sample-by-sample equivalence.

Covers all modules in egress/modules/ that are loaded by module_library.py.
"""
import importlib.resources

import egress as eg
from egress import _bindings as _b
from egress.expr import SignalExpr
from egress.yaml_schema import load_module_from_yaml, save_module_to_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lit(v: float) -> SignalExpr:
    return SignalExpr._from_handle(_b.egress_expr_literal_float(v))


def _run_output(module_type, input_vals: dict, n: int = 64) -> list:
    """Instantiate a module, set literal inputs, run, return output[0] buffer."""
    g = eg.Graph(n)
    inst = module_type._instantiate(g, ())
    for port_name, v in input_vals.items():
        idx = module_type._def["input_names"].index(port_name)
        g.set_input_expr(inst.name, idx, _lit(v))
    tap = g.add_output_tap(inst.name, 0)
    g.process()
    return g.output_tap_buffer(tap)


def _load(filename, uses=None):
    text = importlib.resources.files("egress.modules").joinpath(filename).read_text()
    return load_module_from_yaml(text, module_registry=uses or {})


def _roundtrip(module_type, uses=None):
    """Save a YAML module and reload it; return the reloaded ModuleType."""
    yaml_str = save_module_to_yaml(module_type)
    return load_module_from_yaml(yaml_str, module_registry=uses or {})


def _assert_buffers_match(a, b, tol=1e-10, label=""):
    assert len(a) == len(b), f"{label}: buffer length mismatch {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label} sample {i}: {x} vs {y}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_roundtrip_wrap01():
    orig = _load("wrap01.yaml")
    rt = _roundtrip(orig)
    inp = {"x": 1.7}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="Wrap01")


def test_roundtrip_allpass_stage():
    orig = _load("allpass_stage.yaml")
    rt = _roundtrip(orig)
    inp = {"x": 0.5, "a": 0.3}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="AllpassStage")


def test_roundtrip_poly_blep():
    orig = _load("poly_blep.yaml")
    rt = _roundtrip(orig)
    inp = {"t": 0.01, "dt": 0.01}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="PolyBlep")


def test_roundtrip_poly_blamp():
    orig = _load("poly_blamp.yaml")
    rt = _roundtrip(orig)
    inp = {"t": 0.01, "dt": 0.01}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="PolyBlamp")


def test_roundtrip_phaser():
    allpass = _load("allpass_stage.yaml")
    reg = {"AllpassStage": allpass}
    orig = _load("phaser.yaml", uses=reg)
    rt = _roundtrip(orig, uses=reg)
    inp = {"input": 0.5, "feedback": 0.4, "lfo_speed": 0.2}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="Phaser")


def test_roundtrip_phaser16():
    allpass = _load("allpass_stage.yaml")
    reg = {"AllpassStage": allpass}
    orig = _load("phaser16.yaml", uses=reg)
    rt = _roundtrip(orig, uses=reg)
    inp = {"input": 0.5, "feedback": 0.4, "lfo_speed": 0.2}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="Phaser16")


def test_roundtrip_vco():
    wrap01 = _load("wrap01.yaml")
    poly_blep = _load("poly_blep.yaml")
    reg = {"Wrap01": wrap01, "PolyBlep": poly_blep}
    orig = _load("vco.yaml", uses=reg)
    rt = _roundtrip(orig, uses=reg)
    inp = {"freq": 440.0, "fm": 0.0, "fm_index": 5.0}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="VCO")


def test_roundtrip_clock():
    orig = _load("clock.yaml")
    rt = _roundtrip(orig)
    inp = {"freq": 2.0}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="Clock")


def test_roundtrip_ad_envelope():
    poly_blamp = _load("poly_blamp.yaml")
    reg = {"PolyBlamp": poly_blamp}
    orig = _load("ad_envelope.yaml", uses=reg)
    rt = _roundtrip(orig, uses=reg)
    inp = {"gate": 1.0, "attack": 0.01, "decay": 0.1}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="ADEnvelope")


def test_roundtrip_compressor():
    orig = _load("compressor.yaml")
    rt = _roundtrip(orig)
    inp = {"input": 0.5, "sidechain": 0.5, "threshold": -12.0,
           "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0, "makeup": 1.0}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="Compressor")


def test_roundtrip_bass_drum():
    orig = _load("bass_drum.yaml")
    rt = _roundtrip(orig)
    inp = {"gate": 1.0, "freq": 60.0, "punch": 0.5, "decay": 0.35, "tone": 8.0}
    _assert_buffers_match(_run_output(orig, inp), _run_output(rt, inp), label="BassDrum")


def test_save_module_raises_for_dsl_module():
    """save_module_to_yaml() must raise for Python-DSL-defined modules."""
    from egress.module_library import reverb, delay_line
    import pytest
    r = reverb()
    with pytest.raises(ValueError, match="Python DSL"):
        save_module_to_yaml(r)
    d = delay_line(100)
    with pytest.raises(ValueError, match="Python DSL"):
        save_module_to_yaml(d)
