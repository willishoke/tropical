import egress as eg


_TWO_PI = 6.283185307179586


def _define_wrap01():
    def process(inp):
        x = inp["x"]
        return {
            "value": ((x % 1.0) + 1.0) % 1.0,
        }

    return eg.define_pure_function(inputs=["x"], outputs=["value"], process=process)


def _define_poly_blep():
    def process(inp):
        t = inp["t"]
        dt = inp["dt"]
        left_t = t / dt
        right_t = (t - 1.0) / dt
        left = left_t + left_t - left_t * left_t - 1.0
        right = right_t * right_t + right_t + right_t + 1.0
        valid = (dt > 0.0) * (dt < 1.0)
        left_mask = t < dt
        right_mask = (t >= dt) * (t > (1.0 - dt))
        return {
            "value": valid * (left_mask * left + right_mask * right),
        }

    return eg.define_pure_function(inputs=["t", "dt"], outputs=["value"], process=process)


_wrap01 = _define_wrap01()
_poly_blep = _define_poly_blep()


def _define_allpass_stage():
    def process(inp, reg):
        output = -inp["a"] * inp["x"] + reg["x_prev"] + inp["a"] * reg["y_prev"]
        return (
            {
                "y": output,
            },
            {
                "x_prev": inp["x"],
                "y_prev": output,
            },
        )

    return eg.define_module(
        name="AllpassStage",
        inputs=["x", "a"],
        outputs=["y"],
        regs={"x_prev": 0.0, "y_prev": 0.0},
        process=process,
    )


_allpass_stage = _define_allpass_stage()


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


def vco(name="VCO"):
    def process(inp, reg):
        fm_ratio = eg.pow(2.0, inp["fm_index"] * inp["fm"] / 5.0)
        freq = inp["freq"] * fm_ratio
        dt = eg.clamp(eg.abs(freq) / eg.sample_rate(), 0.0, 0.5)
        phase = _wrap01(reg["phase"] + freq / eg.sample_rate())

        saw = 2.0 * phase - 1.0
        saw = saw - _poly_blep(phase, dt)

        sqr = 1.0 - 2.0 * (phase >= 0.5)
        sqr = sqr + _poly_blep(phase, dt)
        sqr = sqr - _poly_blep(_wrap01(phase + 0.5), dt)

        tri_state = eg.clamp(reg["tri_state"] + sqr * dt * 4.0, -1.0, 1.0)
        sine = eg.sin(_TWO_PI * phase)

        return (
            {
                "saw": 5.0 * saw,
                "tri": 5.0 * tri_state,
                "sin": 5.0 * sine,
                "sqr": 5.0 * sqr,
            },
            {
                "phase": phase,
                "tri_state": tri_state,
            },
        )

    return eg.define_module(
        name=name,
        inputs=["freq", "fm", "fm_index"],
        outputs=["saw", "tri", "sin", "sqr"],
        regs={"phase": 0.0, "tri_state": 0.0},
        process=process,
        sample_rate=44100.0,
        input_defaults={"freq": 100.0, "fm": 0.0, "fm_index": 5.0},
    )


def vco_instance(freq_hz, fm_index=5.0, name="VCO"):
    osc_type = vco(name=name)
    osc = osc_type()
    osc.freq = freq_hz
    osc.fm_index = fm_index
    return osc


def _phaser(stage_count, name):
    def process(inp, reg):
        lfo = eg.sin(_TWO_PI * eg.sample_index() * inp["lfo_speed"] / eg.sample_rate())

        # Keep the one-pole allpass coefficient in a stable range.
        a = 0.6 + 0.35 * lfo
        stage_input = inp["input"] + inp["feedback"] * reg["fb"]

        for _ in range(stage_count):
            stage_input = _allpass_stage(stage_input, a)

        return (
            {
                "output": 0.5 * inp["input"] + 0.5 * stage_input,
                "lfo": lfo,
            },
            {"fb": stage_input},
        )

    return eg.define_module(
        name=name,
        inputs=["input", "feedback", "lfo_speed"],
        outputs=["output", "lfo"],
        regs={"fb": 0.0},
        process=process,
        sample_rate=44100.0,
        input_defaults={"input": 0.0, "feedback": 0.4, "lfo_speed": 0.2},
    )


def phaser(name="Phaser"):
    return _phaser(stage_count=4, name=name)


def phaser16(name="Phaser16"):
    return _phaser(stage_count=16, name=name)


def clock(name="Clock"):
    def process(inp, reg):
        base_phase = (eg.sample_index() * inp["freq"] / eg.sample_rate()) % 1.0
        base_phase = (base_phase + 1.0) % 1.0
        output = (base_phase < 0.5) * 1.0

        ratio_phase = (eg.sample_index() * inp["freq"] * inp["ratios_in"] / eg.sample_rate()) % 1.0
        ratio_phase = (ratio_phase + 1.0) % 1.0
        ratios_out = (ratio_phase < 0.5) * 1.0

        return (
            {
                "output": output,
                "ratios_out": ratios_out,
            },
            {},
        )

    return eg.define_module(
        name=name,
        inputs=["freq", "ratios_in"],
        outputs=["output", "ratios_out"],
        regs={},
        process=process,
        sample_rate=44100.0,
        input_defaults={"freq": 1.0, "ratios_in": eg.array([1.0])},
    )


def reverb(name="Reverb"):
    fdn = _define_fdn(size=4, name=f"{name}_FDN")
    matrix = eg.matrix(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ]
    )

    def process(inp, reg):
        x = inp["input"]
        x = _allpass_stage(x, 0.7)
        x = _allpass_stage(x, 0.6)
        x = _allpass_stage(x, 0.5)
        x = _allpass_stage(x, 0.4)

        drive = eg.array([0.5, -0.5, 0.5, -0.5]) * x
        decay = eg.clamp(inp["decay"], 0.0, 0.99)
        y = fdn(drive, matrix, decay)

        wet = y[0] + y[1] + y[2] + y[3]
        mix = eg.clamp(inp["mix"], 0.0, 1.0)
        output = (1.0 - mix) * inp["input"] + mix * wet

        return (
            {"output": output},
            {},
        )

    return eg.define_module(
        name=name,
        inputs=["input", "mix", "decay"],
        outputs=["output"],
        regs={},
        process=process,
        sample_rate=44100.0,
        input_defaults={"input": 0.0, "mix": 0.35, "decay": 0.85},
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
            phase = _wrap01(phase_prev[i] + fc * stiffness_values[i] / eg.sample_rate())
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
