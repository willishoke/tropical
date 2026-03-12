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
    )


def vco_instance(freq_hz, fm_index=5.0, name="VCO"):
    osc_type = vco(name=name)
    osc = osc_type()
    osc.freq = freq_hz
    osc.fm_index = fm_index
    return osc


def _phaser(stage_count, name):
    x_regs = [f"x{i}" for i in range(stage_count)]
    y_regs = [f"y{i}" for i in range(stage_count)]

    regs = {"fb": 0.0}
    for reg_name in x_regs + y_regs:
        regs[reg_name] = 0.0

    def process(inp, reg):
        lfo = eg.sin(_TWO_PI * eg.sample_index() * inp["lfo_speed"] / eg.sample_rate())

        # Keep the one-pole allpass coefficient in a stable range.
        a = 0.6 + 0.35 * lfo
        stage_input = inp["input"] + inp["feedback"] * reg["fb"]

        next_regs = {}
        for idx in range(stage_count):
            x_prev = reg[x_regs[idx]]
            y_prev = reg[y_regs[idx]]
            stage_output = -a * stage_input + x_prev + a * y_prev
            next_regs[x_regs[idx]] = stage_input
            next_regs[y_regs[idx]] = stage_output
            stage_input = stage_output

        next_regs["fb"] = stage_input

        return (
            {
                "output": 0.5 * inp["input"] + 0.5 * stage_input,
                "lfo": lfo,
            },
            next_regs,
        )

    return eg.define_module(
        name=name,
        inputs=["input", "feedback", "lfo_speed"],
        outputs=["output", "lfo"],
        regs=regs,
        process=process,
        sample_rate=44100.0,
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
    )
