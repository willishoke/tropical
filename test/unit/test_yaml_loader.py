"""
Tests for egress.yaml_schema: load_module_from_yaml, load_patch_from_yaml, save_patch_to_yaml.

Each test verifies a distinct capability:
  1. Simple oscillator YAML → ModuleType (no registers, one output)
  2. Allpass stage YAML → two inputs, two registers
  3. Phaser composition via 'type:' reference in YAML
  4. Module with 'param' expr node (Param smoothing wired correctly)
  5. Module with delay state (one-sample feedback)
  6. Inline nested spec (delay-line built inline)
  7. Patch save/load round-trip
"""
import math

import egress as eg
from egress.yaml_schema import load_module_from_yaml, load_patch_from_yaml, save_patch_to_yaml


# ---------------------------------------------------------------------------
# Helper: run a single-output module for N samples and collect the buffer
# ---------------------------------------------------------------------------

def _run(module_type, n_samples=512, inputs: dict = None) -> list:
    """Instantiate a ModuleType on a fresh graph, set inputs, run, return buffer."""
    g = eg.Graph(n_samples)
    inst = module_type._instantiate(g, ())
    if inputs:
        for port_name, value in inputs.items():
            port_idx = module_type._def["input_names"].index(port_name)
            g.set_input_expr(inst.name, port_idx, eg.SignalExpr._from_handle(
                eg._bindings.egress_expr_literal_float(float(value))
            ))
    # Add first output
    g.add_output(inst.name, 0)
    g.process()
    return g.output_buffer


# ---------------------------------------------------------------------------
# Test 1: Oscillator (no registers, literal-only computation)
#   Computes sin(phase * 2π) where phase increments by freq/sample_rate each tick.
#   We compare a YAML-loaded version against the Python-DSL version.
# ---------------------------------------------------------------------------

OSC_YAML = """\
schema_version: 1
name: Osc
sample_rate: 44100.0
inputs: [freq]
outputs: [sin]

registers:
  - name: phase
    init: 0.0
    update:
      op: add
      args:
        - op: register
          name: phase
        - op: div
          args:
            - op: input
              name: freq
            - op: sample_rate

output_exprs:
  sin:
    op: sin
    args:
      - op: mul
        args:
          - op: register
            name: phase
          - op: literal
            value: 6.283185307179586
"""


def test_osc_yaml_matches_python_dsl():
    """YAML oscillator matches the equivalent Python DSL oscillator."""
    TWO_PI = 2.0 * math.pi

    OscPy = eg.define_module(
        "OscPy",
        inputs=["freq"],
        outputs=["sin"],
        regs={"phase": 0.0},
        process=lambda inp, reg: (
            {"sin": eg.sin(reg["phase"] * TWO_PI)},
            {"phase": reg["phase"] + inp["freq"] / eg.sample_rate()},
        ),
    )

    OscYaml = load_module_from_yaml(OSC_YAML)

    freq = 440.0
    n = 512

    g_py = eg.Graph(n)
    inst_py = OscPy._instantiate(g_py, ())
    from egress._bindings import egress_expr_literal_float
    from egress.expr import SignalExpr
    freq_expr = SignalExpr._from_handle(egress_expr_literal_float(freq))
    g_py.set_input_expr(inst_py.name, 0, freq_expr)
    g_py.add_output(inst_py.name, 0)
    g_py.process()
    buf_py = g_py.output_buffer

    g_yaml = eg.Graph(n)
    inst_yaml = OscYaml._instantiate(g_yaml, ())
    freq_expr2 = SignalExpr._from_handle(egress_expr_literal_float(freq))
    g_yaml.set_input_expr(inst_yaml.name, 0, freq_expr2)
    g_yaml.add_output(inst_yaml.name, 0)
    g_yaml.process()
    buf_yaml = g_yaml.output_buffer

    assert len(buf_py) == len(buf_yaml) == n
    for i, (a, b) in enumerate(zip(buf_py, buf_yaml)):
        assert abs(a - b) < 1e-12, f"Sample {i}: py={a}, yaml={b}"


# ---------------------------------------------------------------------------
# Test 2: Allpass stage — two inputs, two registers
# ---------------------------------------------------------------------------

ALLPASS_YAML = """\
schema_version: 1
name: AllpassStage
sample_rate: 44100.0
inputs: [x, a]
outputs: [y]

registers:
  - name: x_prev
    init: 0.0
    update:
      op: input
      name: x
  - name: y_prev
    init: 0.0
    update:
      op: add
      args:
        - op: add
          args:
            - op: mul
              args:
                - op: neg
                  args:
                    - op: input
                      name: a
                - op: input
                  name: x
            - op: register
              name: x_prev
        - op: mul
          args:
            - op: input
              name: a
            - op: register
              name: y_prev

output_exprs:
  y:
    op: add
    args:
      - op: add
        args:
          - op: mul
            args:
              - op: neg
                args:
                  - op: input
                    name: a
              - op: input
                name: x
          - op: register
            name: x_prev
      - op: mul
        args:
          - op: input
            name: a
          - op: register
            name: y_prev
"""


def test_allpass_yaml_matches_python_dsl():
    """YAML allpass stage matches the Python-DSL version."""
    AllpassPy = eg.define_module(
        "AllpassStagePy",
        inputs=["x", "a"],
        outputs=["y"],
        regs={"x_prev": 0.0, "y_prev": 0.0},
        process=lambda inp, reg: (
            {
                "y": -inp["a"] * inp["x"] + reg["x_prev"] + inp["a"] * reg["y_prev"],
            },
            {
                "x_prev": inp["x"],
                "y_prev": -inp["a"] * inp["x"] + reg["x_prev"] + inp["a"] * reg["y_prev"],
            },
        ),
    )

    AllpassYaml = load_module_from_yaml(ALLPASS_YAML)

    n = 512
    from egress._bindings import egress_expr_literal_float
    from egress.expr import SignalExpr

    def run_allpass(mtype, x_val=0.5, a_val=0.3):
        g = eg.Graph(n)
        inst = mtype._instantiate(g, ())
        g.set_input_expr(
            inst.name, 0,
            SignalExpr._from_handle(egress_expr_literal_float(x_val))
        )
        g.set_input_expr(
            inst.name, 1,
            SignalExpr._from_handle(egress_expr_literal_float(a_val))
        )
        g.add_output(inst.name, 0)
        g.process()
        return g.output_buffer

    buf_py = run_allpass(AllpassPy)
    buf_yaml = run_allpass(AllpassYaml)

    for i, (a, b) in enumerate(zip(buf_py, buf_yaml)):
        assert abs(a - b) < 1e-12, f"Sample {i}: py={a}, yaml={b}"


# ---------------------------------------------------------------------------
# Test 3: Phaser composition via 'type:' reference
# ---------------------------------------------------------------------------

PHASER2_YAML = """\
schema_version: 1
name: Phaser2
sample_rate: 44100.0
inputs: [x, a]
outputs: [y]

nested_modules:
  - id: stage1
    type: AllpassStage
    input_exprs:
      - op: input
        name: x
      - op: input
        name: a
  - id: stage2
    type: AllpassStage
    input_exprs:
      - op: nested_output
        nested_id: stage1
        output_index: 0
      - op: input
        name: a

output_exprs:
  y:
    op: nested_output
    nested_id: stage2
    output_index: 0
"""


def test_phaser_type_reference():
    """YAML phaser using 'type:' references matches cascaded Python allpass modules."""
    AllpassPy = eg.define_module(
        "AllpassStage",
        inputs=["x", "a"],
        outputs=["y"],
        regs={"x_prev": 0.0, "y_prev": 0.0},
        process=lambda inp, reg: (
            {
                "y": -inp["a"] * inp["x"] + reg["x_prev"] + inp["a"] * reg["y_prev"],
            },
            {
                "x_prev": inp["x"],
                "y_prev": -inp["a"] * inp["x"] + reg["x_prev"] + inp["a"] * reg["y_prev"],
            },
        ),
    )

    PhaserPy = eg.define_module(
        "PhaserPy",
        inputs=["x", "a"],
        outputs=["y"],
        regs={},
        process=lambda inp, reg: (
            {"y": AllpassPy(AllpassPy(inp["x"], inp["a"]), inp["a"])},
            {},
        ),
    )

    PhaserYaml = load_module_from_yaml(
        PHASER2_YAML, module_registry={"AllpassStage": AllpassPy}
    )

    n = 512
    from egress._bindings import egress_expr_literal_float
    from egress.expr import SignalExpr

    def run_phaser(mtype, x_val=0.5, a_val=0.3):
        g = eg.Graph(n)
        inst = mtype._instantiate(g, ())
        g.set_input_expr(
            inst.name, 0,
            SignalExpr._from_handle(egress_expr_literal_float(x_val))
        )
        g.set_input_expr(
            inst.name, 1,
            SignalExpr._from_handle(egress_expr_literal_float(a_val))
        )
        g.add_output(inst.name, 0)
        g.process()
        return g.output_buffer

    buf_py = run_phaser(PhaserPy)
    buf_yaml = run_phaser(PhaserYaml)

    for i, (a, b) in enumerate(zip(buf_py, buf_yaml)):
        assert abs(a - b) < 1e-12, f"Sample {i}: py={a}, yaml={b}"


# ---------------------------------------------------------------------------
# Test 4: Module with param_refs → Param smoothing wired correctly
# ---------------------------------------------------------------------------

PARAM_MODULE_YAML = """\
schema_version: 1
name: ParamGain
sample_rate: 44100.0
inputs: [x]
outputs: [y]

output_exprs:
  y:
    op: mul
    args:
      - op: input
        name: x
      - op: param
        name: gain
"""


def test_param_wired_correctly():
    """Module loaded from YAML uses Param smoothing for 'param' expr nodes."""
    gain_param = eg.Param(2.0, time_const=0.005)

    ParamGain = load_module_from_yaml(
        PARAM_MODULE_YAML,
        param_registry={"gain": gain_param},
    )

    n = 512
    from egress._bindings import egress_expr_literal_float
    from egress.expr import SignalExpr

    g = eg.Graph(n)
    inst = ParamGain._instantiate(g, ())
    x_expr = SignalExpr._from_handle(egress_expr_literal_float(1.0))
    g.set_input_expr(inst.name, 0, x_expr)
    # Use a tap to get raw (unscaled) output
    tap_id = g.add_output_tap(inst.name, 0)
    g.process()
    buf = g.output_tap_buffer(tap_id)

    # With x=1.0 and gain=2.0 (init=2.0), output should approach 2.0.
    # The Param starts at its init_value so the last samples should be very close to 2.0.
    assert len(buf) == n
    # Check final samples have converged to 2.0
    for s in buf[-64:]:
        assert abs(s - 2.0) < 0.01, f"Expected ~2.0 (converged), got {s}"
    # Verify the param is connected: output is non-zero (param * x = gain * 1.0)
    assert buf[-1] > 0.0


# ---------------------------------------------------------------------------
# Test 5: Module with delay state (one-sample feedback)
# ---------------------------------------------------------------------------

DELAY_YAML = """\
schema_version: 1
name: Accumulator
sample_rate: 44100.0
inputs: [x]
outputs: [y]

delay_states:
  - id: acc
    init: 0.0
    update:
      op: add
      args:
        - op: delay
          id: acc
        - op: input
          name: x

output_exprs:
  y:
    op: delay
    id: acc
"""


def test_delay_state_feedback():
    """Module with delay state produces same output as Python DSL register accumulator."""
    AccYaml = load_module_from_yaml(DELAY_YAML)

    # Python DSL equivalent using a register (same one-sample delay semantics)
    AccPy = eg.define_module(
        "AccumulatorPy",
        inputs=["x"],
        outputs=["y"],
        regs={"acc": 0.0},
        process=lambda inp, reg: (
            {"y": reg["acc"]},
            {"acc": reg["acc"] + inp["x"]},
        ),
    )

    n = 8
    from egress._bindings import egress_expr_literal_float
    from egress.expr import SignalExpr

    def run_acc(mtype):
        g = eg.Graph(n)
        inst = mtype._instantiate(g, ())
        x_expr = SignalExpr._from_handle(egress_expr_literal_float(1.0))
        g.set_input_expr(inst.name, 0, x_expr)
        tap_id = g.add_output_tap(inst.name, 0)
        g.process()
        return g.output_tap_buffer(tap_id)

    buf_yaml = run_acc(AccYaml)
    buf_py = run_acc(AccPy)

    assert len(buf_yaml) == n
    # Both should match each other exactly
    for i in range(n):
        assert abs(buf_yaml[i] - buf_py[i]) < 1e-12, (
            f"Sample {i}: yaml={buf_yaml[i]}, py={buf_py[i]}"
        )
    # And match expected accumulation: y[k] = k (previous acc)
    for i in range(n):
        assert abs(buf_yaml[i] - float(i)) < 1e-12, (
            f"delay buf[{i}]={buf_yaml[i]}, expected {i}"
        )


# ---------------------------------------------------------------------------
# Test 6: Inline nested spec
# ---------------------------------------------------------------------------

INLINE_DELAY_LINE_YAML = """\
schema_version: 1
name: DelayLine
sample_rate: 44100.0
inputs: [x]
outputs: [y]

nested_modules:
  - id: dl
    input_exprs:
      - op: input
        name: x
    inline:
      input_count: 1
      sample_rate: 44100.0
      registers:
        - name: buf0
          init: 0.0
          update:
            op: nested_input
            index: 0
        - name: buf1
          init: 0.0
          update:
            op: register
            name: buf0
      output_exprs:
        "0":
          op: register
          name: buf1

output_exprs:
  y:
    op: nested_output
    nested_id: dl
    output_index: 0
"""


def test_inline_nested_delay_line():
    """Inline nested spec with two registers acts as a 2-sample delay."""
    DelayLine = load_module_from_yaml(INLINE_DELAY_LINE_YAML)

    # Python DSL equivalent: 2-register delay line
    DelayLinePy = eg.define_module(
        "DelayLinePy",
        inputs=["x"],
        outputs=["y"],
        regs={"buf0": 0.0, "buf1": 0.0},
        process=lambda inp, reg: (
            {"y": reg["buf1"]},
            {"buf0": inp["x"], "buf1": reg["buf0"]},
        ),
    )

    n = 8
    from egress._bindings import egress_expr_literal_float
    from egress.expr import SignalExpr

    def run_delay(mtype):
        g = eg.Graph(n)
        inst = mtype._instantiate(g, ())
        x_expr = SignalExpr._from_handle(egress_expr_literal_float(1.0))
        g.set_input_expr(inst.name, 0, x_expr)
        tap_id = g.add_output_tap(inst.name, 0)
        g.process()
        return g.output_tap_buffer(tap_id)

    buf_py = run_delay(DelayLinePy)
    buf_yaml = run_delay(DelayLine)

    # buf1 follows buf0 by 1 tick; buf0 = x (current input)
    # y = buf1 = x from 2 ticks ago: [0, 0, 1, 1, 1, 1, 1, 1]
    expected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    assert len(buf_py) == n
    for i in range(n):
        assert abs(buf_py[i] - expected[i]) < 1e-12, (
            f"Python DL buf[{i}]={buf_py[i]}, expected {expected[i]}"
        )

    assert len(buf_yaml) == n
    for i in range(n):
        assert abs(buf_yaml[i] - expected[i]) < 1e-12, (
            f"YAML DL buf[{i}]={buf_yaml[i]}, expected {expected[i]}"
        )


# ---------------------------------------------------------------------------
# Test 7: Patch save/load round-trip
# ---------------------------------------------------------------------------

def test_patch_round_trip(tmp_path):
    """Save a patch to YAML files, load it, verify modules instantiate correctly."""
    # Write osc.yaml into tmp_path
    osc_yaml_path = tmp_path / "osc.yaml"
    osc_yaml_path.write_text(OSC_YAML)

    # Write a gain module
    gain_yaml = """\
schema_version: 1
name: Gain
sample_rate: 44100.0
inputs: [x]
outputs: [y]

output_exprs:
  y:
    op: mul
    args:
      - op: input
        name: x
      - op: literal
        value: 0.5
"""
    gain_yaml_path = tmp_path / "gain.yaml"
    gain_yaml_path.write_text(gain_yaml)

    # Load both module types
    OscType = load_module_from_yaml(OSC_YAML)
    GainType = load_module_from_yaml(gain_yaml)

    # Create a simple graph with two modules
    g = eg.Graph(64)
    osc_inst = OscType._instantiate(g, ())
    gain_inst = GainType._instantiate(g, ())
    g.connect(osc_inst.name, 0, gain_inst.name, 0)
    g.add_output(gain_inst.name, 0)

    # Instantiate dicts
    instances = {
        osc_inst.name: osc_inst,
        gain_inst.name: gain_inst,
    }
    type_file_map = {
        osc_inst.name: "osc.yaml",
        gain_inst.name: "gain.yaml",
    }
    connections = [
        {"src": osc_inst.name, "src_output": "sin", "dst": gain_inst.name, "dst_input": "x"}
    ]
    patch_outputs = [{"module": gain_inst.name, "output": "y"}]

    # Save patch
    patch_yaml = save_patch_to_yaml(
        g, instances, type_file_map, connections=connections, patch_outputs=patch_outputs
    )

    assert "schema_version" in patch_yaml
    assert "osc.yaml" in patch_yaml
    assert "gain.yaml" in patch_yaml

    # Write patch YAML to file
    patch_path = tmp_path / "patch.yaml"
    patch_path.write_text(patch_yaml)

    # Reload the patch
    g2, instances2 = load_patch_from_yaml(patch_yaml, base_dir=tmp_path)

    assert len(instances2) == 2
    # Verify the reloaded graph can process audio without errors
    g2.process()
    buf = g2.output_buffer
    assert len(buf) > 0
