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
    # Different lengths decorrelate the channels and give the reverb its colour.
    _COMB_DELAYS = [1557, 1617, 1491, 1422]
    comb_filters = [
        _define_comb_filter(d, f"{name}_Comb{i}")
        for i, d in enumerate(_COMB_DELAYS)
    ]

    # Allpass coefficients — slightly detuned to break up flutter
    _AP_IN   = [0.70, 0.65]
    _AP_OUT  = [0.60, 0.55, 0.50, 0.45]

    def process(inp, reg):
        x    = inp["input"]
        mix  = eg.clamp(inp["mix"],   0.0, 1.0)
        decay = eg.clamp(inp["decay"], 0.0, 0.99)
        damp  = eg.clamp(inp["damp"],  0.0, 1.0)

        # --- 1. Input diffusion ---
        diff = x
        for a in _AP_IN:
            diff = _allpass_stage(diff, a)

        # --- 2. Parallel comb bank ---
        wet = 0.0
        for cf in comb_filters:
            wet = wet + cf(diff, decay, damp)
        wet = wet * 0.25

        # --- 3. Output diffusion ---
        for a in _AP_OUT:
            wet = _allpass_stage(wet, a)

        # --- 4. Wet/dry blend ---
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


def _define_poly_blamp():
    """
    PolyBLAMP residual — integral of PolyBLEP — corrects first-derivative
    (slope/kink) discontinuities such as the corners of a linear ramp.

    Inputs : t   – phase in [0, 1)
             dt  – phase increment (normalized frequency)
    Output : value – correction to subtract from a linear ramp

    Left  half: applied near t=0      (start of ramp)
    Right half: applied near t=1-dt   (end of ramp)
    """
    def process(inp):
        t  = inp["t"]
        dt = inp["dt"]
        valid = (dt > 0.0) * (dt < 1.0)

        # --- Near the start of the ramp (t in [0, dt)) ---
        u = t / dt
        # Integral of the polyBLEP left residual:  u^2 - u^3/3 - u  (zero at u=0 and u=1)
        left = u * u - u * u * u / 3.0 - u

        # --- Near the end of the ramp (t in (1-dt, 1]) ---
        v = (1.0 - t) / dt
        right = -(v * v - v * v * v / 3.0 - v)

        left_mask  = t < dt
        right_mask = (1.0 - t) < dt

        return {"value": valid * (left_mask * left + right_mask * right)}

    return eg.define_pure_function(inputs=["t", "dt"], outputs=["value"], process=process)


_poly_blamp = _define_poly_blamp()


def ad_envelope(name="ADEnvelope"):
    """
    Attack-Decay envelope generator with PolyBLAMP anti-kink smoothing.

    A rising linear ramp over 'attack' seconds, then a falling linear ramp
    over 'decay' seconds.  PolyBLAMP residuals are subtracted at the three
    corners (start of attack, attack→decay joint, end of decay) to suppress
    the high-frequency content that would otherwise result from the abrupt
    slope changes.

    Inputs
    ------
    gate    : rising edge (> 0.5) triggers the envelope; re-trigger resets it
    attack  : attack time in seconds
    decay   : decay time in seconds

    Output
    ------
    env : envelope value in [0, 1]
    """
    def process(inp, reg):
        sr      = eg.sample_rate()
        attack  = eg.clamp(inp["attack"], 1e-4, 10.0)
        decay   = eg.clamp(inp["decay"],  1e-4, 10.0)
        gate    = inp["gate"]

        # Rising-edge detection
        prev_gate = eg.delay(gate, init=0.0)
        trig = (gate > 0.5) * (prev_gate <= 0.5)

        stage = reg["stage"]   # 0 = idle, 1 = attack, 2 = decay
        phase = reg["phase"]   # 0..1 within current stage

        in_attack = (stage > 0.5) * (stage < 1.5)   # stage == 1
        in_decay  =  stage > 1.5                     # stage == 2

        dt_a = 1.0 / (attack * sr)
        dt_d = 1.0 / (decay  * sr)
        dt   = in_attack * dt_a + in_decay * dt_d

        new_phase = phase + dt

        a_to_d  = in_attack * (new_phase >= 1.0)
        d_done  = in_decay  * (new_phase >= 1.0)

        # --- Next stage ---
        # gate_rise always restarts attack; otherwise advance the state machine
        next_stage = (
            trig * 1.0
            + (1.0 - trig) * (
                in_attack * (1.0 - a_to_d) * 1.0
                + a_to_d * 2.0
                + in_decay * (1.0 - d_done) * 2.0
                # d_done → idle (0)
            )
        )

        # --- Next phase ---
        # On trigger: reset to 0
        # On attack→decay: carry overshoot into decay stage
        # On decay done / idle: 0
        next_phase = (
            trig * 0.0
            + (1.0 - trig) * (
                a_to_d * eg.clamp(new_phase - 1.0, 0.0, 1.0)
                + (1.0 - a_to_d) * (1.0 - d_done) * eg.clamp(new_phase, 0.0, 1.0)
            )
        )

        # --- Raw envelope value ---
        # Use current (pre-increment) phase so the blamp at phase=0 is visible
        raw_env = in_attack * phase + in_decay * (1.0 - phase)

        # --- PolyBLAMP corrections ---
        # 1. Start of attack: slope jumps from 0 to dt_a
        blamp_start = _poly_blamp(phase, dt_a)
        # 2. End of attack / start of decay: slope jumps from dt_a to -dt_d
        #    Apply to both the tail of attack (near phase=1) and head of decay (near phase=0)
        blamp_atk_end = _poly_blamp(1.0 - phase, dt_a)
        blamp_dec_start = _poly_blamp(phase, dt_d)
        # 3. End of decay: slope jumps from -dt_d to 0
        blamp_end = _poly_blamp(1.0 - phase, dt_d)

        env = (
            raw_env
            - in_attack * dt_a * blamp_start
            + in_attack * dt_a * blamp_atk_end
            + in_decay  * dt_d * blamp_dec_start
            - in_decay  * dt_d * blamp_end
        )

        return (
            {"env": eg.clamp(env, 0.0, 1.0)},
            {"stage": next_stage, "phase": next_phase},
        )

    return eg.define_module(
        name=name,
        inputs=["gate", "attack", "decay"],
        outputs=["env"],
        regs={"stage": 0.0, "phase": 0.0},
        process=process,
        sample_rate=44100.0,
        input_defaults={"gate": 0.0, "attack": 0.01, "decay": 0.3},
    )


def compressor(name="Compressor"):
    """
    Feed-forward compressor with a dedicated sidechain input.

    A full-wave peak detector (with separate attack/release ballistics) feeds
    an RMS-style level estimator.  Gain reduction is computed in dB, smoothed
    with the same ballistics, then applied as a linear gain to the main input.

    Inputs
    ------
    input       : audio signal to be gain-reduced
    sidechain   : control signal for level detection (patch to input for
                  normal compression, or to a separate signal for ducking)
    threshold   : threshold in dBFS (e.g. -12.0)
    ratio       : compression ratio (1 = no compression, ∞ = limiting)
    attack_ms   : attack  time in milliseconds
    release_ms  : release time in milliseconds
    makeup      : linear makeup gain applied after compression

    Outputs
    -------
    output : gain-reduced (and made-up) audio
    gr     : instantaneous gain reduction in dB (≤ 0, useful for metering)
    """
    _LOG10E = 0.4342944819309   # log10(e) = 1/ln(10)
    _20_LN10_INV = 8.68588963807  # 20 / ln(10)  for dBFS conversion

    def process(inp, reg):
        sr = eg.sample_rate()

        threshold  = inp["threshold"]
        ratio      = eg.clamp(inp["ratio"], 1.0, 1000.0)
        attack_ms  = eg.clamp(inp["attack_ms"],  0.01, 2000.0)
        release_ms = eg.clamp(inp["release_ms"], 1.0,  10000.0)
        makeup     = inp["makeup"]

        # --- Level detection (peak follower on sidechain) ---
        sc = eg.abs(inp["sidechain"])
        prev_env = reg["env"]

        atk_coeff = eg.pow(10.0, -_LOG10E / eg.clamp(attack_ms  * 0.001 * sr, 1.0, 1e8))
        rel_coeff = eg.pow(10.0, -_LOG10E / eg.clamp(release_ms * 0.001 * sr, 1.0, 1e8))

        # Attack on rising signal, release on falling
        rising  = sc > prev_env
        new_env = (
            rising       * (atk_coeff * prev_env + (1.0 - atk_coeff) * sc)
            + (1.0 - rising) * (rel_coeff * prev_env + (1.0 - rel_coeff) * sc)
        )

        # --- Gain computer (static curve) ---
        # level_db = 20*log10(env) = (20/ln10) * ln(env)
        level_db   = _20_LN10_INV * eg.log(eg.clamp(new_env, 1e-9, 1e9))
        over_db    = level_db - threshold
        # GR is 0 below threshold, negative above (slope = 1/ratio - 1)
        gr_db      = (over_db > 0.0) * over_db * (1.0 / ratio - 1.0)

        # --- Smooth gain reduction with attack/release ---
        prev_gr = reg["gr"]
        # GR becoming more negative = attack; returning to 0 = release
        gr_attack  = gr_db < prev_gr
        smooth_gr  = (
            gr_attack        * (atk_coeff * prev_gr + (1.0 - atk_coeff) * gr_db)
            + (1.0 - gr_attack) * (rel_coeff * prev_gr + (1.0 - rel_coeff) * gr_db)
        )

        # --- Apply gain ---
        gain   = eg.pow(10.0, smooth_gr / 20.0)  # GR dB → linear
        output = inp["input"] * gain * makeup

        return (
            {"output": output, "gr": smooth_gr},
            {"env": new_env, "gr": smooth_gr},
        )

    return eg.define_module(
        name=name,
        inputs=["input", "sidechain", "threshold", "ratio", "attack_ms", "release_ms", "makeup"],
        outputs=["output", "gr"],
        regs={"env": 0.0, "gr": 0.0},
        process=process,
        sample_rate=44100.0,
        input_defaults={
            "input":      0.0,
            "sidechain":  0.0,
            "threshold": -12.0,
            "ratio":       4.0,
            "attack_ms":  10.0,
            "release_ms": 100.0,
            "makeup":      1.0,
        },
    )


def bass_drum(name="BassDrum"):
    """
    Simple analogue-style bass drum voice.

    An impulse excites a 2-pole resonant state-variable lowpass filter
    (Zavalishin TPT design — unconditionally stable).  A pitch envelope
    sweeps the cutoff frequency downward from (freq × (1 + punch × 4)) to
    freq, giving the characteristic transient 'thump'.  An amplitude envelope
    controls the overall level.

    Inputs
    ------
    gate   : rising edge (> 0.5) fires the drum
    freq   : fundamental / resting pitch in Hz  (default 60 Hz)
    punch  : pitch-sweep depth 0..1             (default 0.5)
    decay  : amplitude decay time in seconds    (default 0.35)
    tone   : filter resonance Q                 (default 8.0)

    Output
    ------
    output : mono audio signal, peak ≈ ±5 V
    """
    _PI = 3.14159265358979

    def process(inp, reg):
        sr = eg.sample_rate()

        freq  = eg.clamp(inp["freq"],  20.0, 500.0)
        punch = eg.clamp(inp["punch"],  0.0,   1.0)
        decay = eg.clamp(inp["decay"], 0.01,   4.0)
        tone  = eg.clamp(inp["tone"],  0.5,   50.0)

        # --- Trigger detection ---
        gate      = inp["gate"]
        prev_gate = eg.delay(gate, init=0.0)
        trig      = (gate > 0.5) * (prev_gate <= 0.5)

        # --- Amplitude envelope (instant attack, exponential decay) ---
        amp_env   = reg["amp_env"]
        amp_coeff = eg.pow(10.0, -0.4342944819 / eg.clamp(decay * sr, 1.0, 1e8))
        new_amp   = trig * 1.0 + (1.0 - trig) * amp_coeff * amp_env

        # --- Pitch envelope (fast exponential decay, ~40 ms) ---
        pitch_env   = reg["pitch_env"]
        pitch_tau   = 0.040 * sr   # 40 ms time constant
        pitch_coeff = eg.pow(10.0, -0.4342944819 / eg.clamp(pitch_tau, 1.0, 1e8))
        new_pitch   = trig * 1.0 + (1.0 - trig) * pitch_coeff * pitch_env

        # Instantaneous cutoff: high at onset, settles to base freq
        fc = freq * (1.0 + punch * 4.0 * new_pitch)

        # --- Zavalishin TPT State Variable Filter (resonant LPF) ---
        # g = tan(π·fc/fs) ≈ π·fc/fs for fc ≪ fs/2 (valid for bass)
        # R = 1/(2Q)  (damping factor)
        g = eg.clamp(_PI * fc / sr, 0.0, 0.98)
        R = 0.5 / tone

        # On trigger: reset integrator states to produce maximum LP amplitude
        # immediately, rather than exciting via an impulse (which has negligible
        # LP output for low fc because the LP gain ∝ g²).
        # ic2 = 2.0 seeds the LP integrator at full amplitude; the filter then
        # rings at fc, decaying at the rate set by R (and the amp_env above).
        ic1 = trig * 0.0 + (1.0 - trig) * reg["ic1"]
        ic2 = trig * 2.0 + (1.0 - trig) * reg["ic2"]

        denom = 1.0 + 2.0 * R * g + g * g
        v3   = (0.0 - ic2 - (2.0 * R + g) * ic1) / denom   # HP output
        v1   = g * v3
        y_bp = v1 + ic1
        v2   = g * y_bp
        y_lp = v2 + ic2

        new_ic1 = y_bp + v1
        new_ic2 = y_lp + v2

        # Scale by 0.5 to compensate for ic2=2.0 seed
        output = new_amp * eg.clamp(y_lp * 0.5, -1.0, 1.0) * 5.0

        return (
            {"output": output},
            {
                "amp_env":   new_amp,
                "pitch_env": new_pitch,
                "ic1":       new_ic1,
                "ic2":       new_ic2,
            },
        )

    return eg.define_module(
        name=name,
        inputs=["gate", "freq", "punch", "decay", "tone"],
        outputs=["output"],
        regs={"amp_env": 0.0, "pitch_env": 0.0, "ic1": 0.0, "ic2": 0.0},
        process=process,
        sample_rate=44100.0,
        input_defaults={"gate": 0.0, "freq": 60.0, "punch": 0.5, "decay": 0.35, "tone": 8.0},
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
