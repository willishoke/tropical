import importlib.resources

import egress as eg
from egress.yaml_schema import load_module_from_yaml


_TWO_PI = 6.283185307179586


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load(filename, uses=None, name=None):
    """Load a ModuleType from egress/modules/<filename>."""
    text = importlib.resources.files("egress.modules").joinpath(filename).read_text()
    m = load_module_from_yaml(text, module_registry=uses or {})
    if name is not None:
        m._def = dict(m._def)
        m._def["type_name"] = name
    return m


# ---------------------------------------------------------------------------
# Private helper module types (loaded once at import time)
# ---------------------------------------------------------------------------

_wrap01_type      = _load("wrap01.yaml")
_poly_blep_type   = _load("poly_blep.yaml")
_poly_blamp_type  = _load("poly_blamp.yaml")
_allpass_stage    = _load("allpass_stage.yaml")


# ---------------------------------------------------------------------------
# Public module factories
# ---------------------------------------------------------------------------

def vco(name="VCO"):
    return _load("vco.yaml", uses={"Wrap01": _wrap01_type, "PolyBlep": _poly_blep_type}, name=name)


def vco_instance(freq_hz, fm_index=5.0, name="VCO"):
    osc_type = vco(name=name)
    osc = osc_type()
    osc.freq = freq_hz
    osc.fm_index = fm_index
    return osc


def _phaser(stage_count, name):
    filename = "phaser.yaml" if stage_count == 4 else "phaser16.yaml"
    return _load(filename, uses={"AllpassStage": _allpass_stage}, name=name)


def phaser(name="Phaser"):
    return _phaser(stage_count=4, name=name)


def phaser16(name="Phaser16"):
    return _phaser(stage_count=16, name=name)


def clock(name="Clock"):
    return _load("clock.yaml", name=name)


def ad_envelope(name="ADEnvelope"):
    return _load("ad_envelope.yaml", uses={"PolyBlamp": _poly_blamp_type}, name=name)


def compressor(name="Compressor"):
    return _load("compressor.yaml", name=name)


def bass_drum(name="BassDrum"):
    return _load("bass_drum.yaml", name=name)


# ---------------------------------------------------------------------------
# Parameterized factories — kept in Python DSL (dynamic array sizes)
# ---------------------------------------------------------------------------

def _define_fdn(size=4, name="FDN"):
    def process(inp, reg):
        feedback = eg.matmul(inp["matrix"], reg["state"])
        y = inp["x"] + feedback
        next_state = y * inp["decay"]
        return (
            {"y": y},
            {"state": next_state},
        )

    return eg.define_module(
        name=name,
        inputs=["x", "matrix", "decay"],
        outputs=["y"],
        regs={"state": [0.0] * size},
        process=process,
    )


def delay_line(delay_len, name="Delay"):
    """
    Fixed-length delay line.

    Implemented as a circular buffer register: on each tick, the slot at
    (sample_index % delay_len) is read (returning the value written
    delay_len ticks ago) then overwritten with the current input.

    Inputs : x – scalar input
    Output : y – input delayed by delay_len samples
    """
    def process(inp, reg):
        buf = reg["buf"]
        write_idx = eg.sample_index() % delay_len
        y = buf[write_idx]
        new_buf = eg.array_set(buf, write_idx, inp["x"])
        return (
            {"y": y},
            {"buf": new_buf},
        )

    return eg.define_module(
        name=name,
        inputs=["x"],
        outputs=["y"],
        regs={"buf": [0.0] * delay_len},
        process=process,
        sample_rate=44100.0,
        input_defaults={"x": 0.0},
    )


def _define_comb_filter(delay_len, name="CombFilter"):
    """
    Single comb filter with a one-pole lowpass in the feedback loop.
    Delay buffer is stored directly as a register to avoid nested module
    overhead and keep the module eligible for JIT compilation.

    Inputs : x     – scalar input
             decay – feedback gain (controls RT60)
             damp  – lowpass coefficient [0..1]; 0 = bright, 1 = dark
    Output : y     – comb filter output
    """
    def process(inp, reg):
        x        = inp["x"]
        decay    = eg.clamp(inp["decay"], 0.0, 0.98)
        damp     = eg.clamp(inp["damp"],  0.0, 0.99)
        lpf_prev = reg["lpf"]
        buf      = reg["buf"]

        write_idx = eg.sample_index() % delay_len
        state     = buf[write_idx]
        new_buf   = eg.array_set(buf, write_idx, x + decay * lpf_prev)

        # Update one-pole LPF in the feedback path for next tick
        lpf_out = (1.0 - damp) * state + damp * lpf_prev

        return (
            {"y": state},
            {"lpf": lpf_out, "buf": new_buf},
        )

    return eg.define_module(
        name=name,
        inputs=["x", "decay", "damp"],
        outputs=["y"],
        regs={"lpf": 0.0, "buf": [0.0] * delay_len},
        process=process,
        sample_rate=44100.0,
        input_defaults={"x": 0.0, "decay": 0.84, "damp": 0.4},
    )


def reverb(name="Reverb"):
    """
    Freeverb-inspired reverb.

    Architecture (loosely modelled on Schroeder / Freeverb):
      1. Input diffusion  : 2 cascaded allpass stages
      2. Comb bank        : 4 parallel comb filters with proper delay lines
      3. Output diffusion : 4 cascaded allpass stages
      4. Wet/dry mix

    Parameters
    ----------
    mix    : 0..1   wet/dry blend
    decay  : 0..1   feedback gain (room size feel; try 0.8–0.95)
    damp   : 0..1   high-frequency damping (0 = bright, 1 = very dark/warm)
    """
    # Freeverb-style comb delay lengths (samples at 44100 Hz, ~32–37 ms each).
    _COMB_DELAYS = [1557, 1617, 1491, 1422]
    comb_filters = [
        _define_comb_filter(d, f"{name}_Comb{i}")
        for i, d in enumerate(_COMB_DELAYS)
    ]

    _AP_IN   = [0.70, 0.65]
    _AP_OUT  = [0.60, 0.55, 0.50, 0.45]

    def process(inp, reg):
        x    = inp["input"]
        mix  = eg.clamp(inp["mix"],   0.0, 1.0)
        decay = eg.clamp(inp["decay"], 0.0, 0.99)
        damp  = eg.clamp(inp["damp"],  0.0, 1.0)

        diff = x
        for a in _AP_IN:
            diff = _allpass_stage(diff, a)

        wet = 0.0
        for cf in comb_filters:
            wet = wet + cf(diff, decay, damp)
        wet = wet * 0.25

        for a in _AP_OUT:
            wet = _allpass_stage(wet, a)

        output = (1.0 - mix) * inp["input"] + mix * wet

        return (
            {"output": output},
            {},
        )

    return eg.define_module(
        name=name,
        inputs=["input", "mix", "decay", "damp"],
        outputs=["output"],
        regs={},
        process=process,
        sample_rate=44100.0,
        input_defaults={"input": 0.0, "mix": 0.35, "decay": 0.84, "damp": 0.4},
    )


def topo_waveguide(nx=4, ny=4, name="TopoWaveguide"):
    nx = max(1, int(nx))
    ny = max(1, int(ny))
    node_count = nx * ny
    center_x = 0.5 * (nx - 1)
    center_y = 0.5 * (ny - 1)
    max_radius = max(1.0, (center_x * center_x + center_y * center_y) ** 0.5)
    modal_ratios = [1.0, 2.76, 5.4, 8.93, 13.3, 18.64, 24.97, 32.31]

    def idx(i, j):
        return (i % nx) * ny + (j % ny)

    adjacency = [[] for _ in range(node_count)]
    for i in range(nx):
        for j in range(ny):
            n = idx(i, j)
            neighbors = [
                idx(i - 1, j),
                idx(i + 1, j),
                idx(i, j - 1),
                idx(i, j + 1),
            ]
            for v in neighbors:
                if v != n and v not in adjacency[n]:
                    adjacency[n].append(v)

    fc_values = []
    stiffness_values = []
    phase_offset_values = []
    for i in range(nx):
        for j in range(ny):
            node_id = idx(i, j)
            ratio = modal_ratios[node_id % len(modal_ratios)]
            radial = ((i - center_x) * (i - center_x) + (j - center_y) * (j - center_y)) ** 0.5 / max_radius
            skew = (((node_id * 17) % 9) - 4) / 4.0
            fc = 180.0 * ratio * (1.0 + 0.16 * radial) * (1.0 + 0.035 * skew)
            fc = max(120.0, min(6400.0, fc))
            stiffness = 1.0 + 0.08 * radial + 0.025 * (((node_id * 11) % 7) - 3)
            stiffness = max(0.82, min(1.22, stiffness))
            fc_values.append(fc)
            stiffness_values.append(stiffness)
            phase_offset_values.append(_TWO_PI * (((node_id * 19) % 23) / 23.0))

    def process(inp, reg):
        g = eg.clamp(inp["g"], 0.0, 0.2)
        decay = eg.clamp(inp["decay"], 0.95, 0.999995)
        brightness = eg.clamp(inp["brightness"], 0.0, 1.0)
        amp_prev = reg["amp"]
        phase_prev = reg["phase"]
        outputs = []
        next_amp = []
        next_phase = []

        for i in range(node_count):
            neighbor_energy = 0.0
            for j in adjacency[i]:
                neighbor_energy = neighbor_energy + amp_prev[j]

            neighbor_avg = neighbor_energy / float(max(1, len(adjacency[i])))
            fc = eg.clamp(inp["fc"][i], 120.0, 8000.0)
            normalized_fc = eg.clamp(fc / 8000.0, 0.0, 1.0)
            local_decay = decay * (1.0 - (0.008 + 0.055 * (1.0 - brightness)) * normalized_fc)
            local_decay = eg.clamp(local_decay, 0.94, 0.99997)
            strike = inp["input"][i] * (0.18 + 0.82 * brightness)
            coupling = 0.018 * g * neighbor_avg * (1.0 - 0.5 * normalized_fc)
            amp = local_decay * amp_prev[i] + strike + coupling
            phase = _wrap01_type(phase_prev[i] + fc * stiffness_values[i] / eg.sample_rate())
            fundamental = eg.sin(_TWO_PI * phase + phase_offset_values[i])
            overtone = fundamental * fundamental * fundamental
            y = amp * ((0.9 - 0.26 * brightness) * fundamental + (0.06 + 0.24 * brightness) * overtone)
            outputs.append(y)
            next_amp.append(amp)
            next_phase.append(phase)

        return (
            {"out": eg.array(outputs)},
            {"amp": eg.array(next_amp), "phase": eg.array(next_phase)},
        )

    return eg.define_module(
        name=name,
        inputs=["input", "fc", "g", "decay", "brightness"],
        outputs=["out"],
        regs={"amp": [0.0] * node_count, "phase": [0.0] * node_count},
        process=process,
        sample_rate=44100.0,
        input_defaults={
            "input": eg.array([0.0] * node_count),
            "fc": eg.array(fc_values),
            "g": 0.035,
            "decay": 0.9997,
            "brightness": 0.88,
        },
    )
