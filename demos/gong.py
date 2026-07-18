"""gong — offline sound design for the struck nonlinear resonator demo.

Everything here is closed-form in τ (no state), i.e. engine-expressible:

  - amplitude bloom  = difference of exponentials  e^{-d1 d} - e^{-d2 d}
    (rises then falls; strictly exp-poly, in-basis)
  - pitch bloom      = analytic warp of the phase:
        φ(d) = 2π f (d + β (1 - e^{-g d}) / g)
    instantaneous freq starts at f(1+β), settles to f as the strike decays
    (this is the per-mode nonlinear clock warp — the warp basis's job)
  - "nonlinearity"   = velocity coupling: bloom depth β and shimmer
    energy scale with strike velocity — soft strikes nearly linear,
    hard strikes bright, sharp, and bloomy

(The tanh/odd-poly drive stage is RETIRED — a static memoryless waveshaper
is not warmth. The gong is linear per strike; velocity coupling carries the
nonlinearity.)

Renders an A/B ladder on a single hard strike, a velocity fan, and a
short score, so the recipe can be tuned by ear before any Lean work.
"""
import numpy as np
import wave
import os

SR = 44100
OUT = os.path.dirname(os.path.abspath(__file__))
rng_global = np.random.default_rng(7)

TAIL = 14.0          # per-strike render window (s)


# ---------------------------------------------------------------- mode bank

def gong_bank(f0, rng):
    """One gong's mode bank. Returns a list of mode dicts.

    Three registers:
      low   — near-harmonic strong partials, slow decay, carry the pitch
      mid   — dense inharmonic cluster, the 'metal' body
      high  — shimmer band, amplitude-BLOOMS after the strike (d1 slow decay,
              d2 fast rise), energy scales with velocity^2 at strike time
    """
    modes = []

    low_ratios = np.array([1.0, 1.51, 2.07, 2.63, 3.21, 3.85])
    low_ratios = low_ratios * (1 + 0.004 * rng.standard_normal(len(low_ratios)))
    for i, r in enumerate(low_ratios):
        modes.append(dict(f=f0 * r, amp=1.0 / (i + 1) ** 0.7,
                          d1=0.18 + 0.10 * i, d2=None,
                          register="low"))

    n_mid = 45
    mid_ratios = np.sort(rng.uniform(2.8, 11.0, n_mid))
    for r in mid_ratios:
        modes.append(dict(f=f0 * r, amp=0.35 / r ** 0.8,
                          d1=rng.uniform(0.4, 1.3), d2=None,
                          register="mid"))

    n_high = 40
    high_ratios = np.sort(rng.uniform(7.0, 26.0, n_high))
    for r in high_ratios:
        rise = rng.uniform(1.2, 4.5)            # 1/d2 ~ bloom-in time
        modes.append(dict(f=f0 * r, amp=0.22 / r ** 0.5,
                          d1=rng.uniform(0.35, 0.9), d2=rise,
                          register="high"))

    # strike transient: broadband, very fast decay — the 'thwack'
    n_hit = 25
    for r in np.sort(rng.uniform(1.5, 30.0, n_hit)):
        modes.append(dict(f=f0 * r, amp=0.5 / r ** 0.3,
                          d1=rng.uniform(18.0, 60.0), d2=None,
                          register="hit"))

    for m in modes:
        m["ph"] = rng.uniform(0, 2 * np.pi, 2)          # stereo phases
        m["pan"] = rng.uniform(0.35, 0.65)
    return modes


# ---------------------------------------------------------------- one strike

def render_strike(modes, vel, d, bloom=True, beta0=0.06, g=1.8):
    """Closed-form render of one strike on local time d (d>=0), stereo.

    vel couplings (the 'nonlinearity'):
      pitch bloom depth  β = beta0 · vel
      shimmer energy     ∝ vel²   (energy cascades UP with harder strikes)
      transient energy   ∝ vel^1.5
    """
    y = np.zeros((len(d), 2))
    beta = beta0 * vel if bloom else 0.0
    # analytic pitch-warp: φ/2πf = d + β (1 - e^{-g d})/g
    warp = d + beta * (1.0 - np.exp(-g * d)) / g if beta > 0 else d
    for m in modes:
        env = np.exp(-m["d1"] * d)
        if m["d2"] is not None and bloom:
            env = env - np.exp(-(m["d1"] + m["d2"]) * d)   # bloom in
        a = m["amp"]
        if m["register"] == "high":
            a *= vel ** 2
        elif m["register"] == "hit":
            a *= vel ** 1.5
        else:
            a *= vel
        # high modes glide a touch less (stiffness rides the low modes)
        w = warp if m["register"] in ("low", "mid") else d + 0.5 * (warp - d)
        arg = 2 * np.pi * m["f"] * w
        for c in range(2):
            g_c = m["pan"] if c == 0 else 1 - m["pan"]
            y[:, c] += a * g_c * env * np.cos(arg + m["ph"][c])
    return y


def strike(modes, vel, bloom=True, dur=TAIL):
    """Render one strike — linear (the drive stage is retired)."""
    d = np.arange(int(dur * SR)) / SR
    return render_strike(modes, vel, d, bloom=bloom)


# ---------------------------------------------------------------- output

def write_wav(path, y, gain=0.9):
    assert np.all(np.isfinite(y)), "non-finite samples"
    y = y / np.max(np.abs(y)) * gain
    f = int(0.01 * SR)
    env = np.ones(len(y))
    env[:f] = np.linspace(0, 1, f)
    env[-f:] = np.linspace(1, 0, f)
    y = y * env[:, None]
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    print(f"wrote {os.path.basename(path)}  ({len(y)/SR:.1f} s, "
          f"rms {np.sqrt(np.mean(y**2)):.3f})")


def place(canvas, y, t0):
    i0 = int(t0 * SR)
    n = min(len(y), len(canvas) - i0)
    if n > 0:
        canvas[i0:i0 + n] += y[:n]


# ---------------------------------------------------------------- renders

def main():
    f0 = 82.0                      # low gong, E-ish
    modes = gong_bank(f0, np.random.default_rng(7))

    # --- A/B ladder: same hard strike, recipe stacking up
    print("— ladder (one hard strike, vel 1.0) —")
    write_wav(f"{OUT}/gong_A_linear.wav", strike(modes, 1.0, bloom=False))
    write_wav(f"{OUT}/gong_B_bloom.wav", strike(modes, 1.0, bloom=True))

    # --- velocity fan: the SAME gong, soft → hard (the aliveness test)
    print("— velocity fan (0.25 / 0.55 / 1.0) —")
    fan = np.zeros((int(24 * SR), 2))
    for i, v in enumerate([0.25, 0.55, 1.0]):
        place(fan, strike(modes, v), 0.5 + 8.0 * i)
    write_wav(f"{OUT}/gong_velocity_fan.wav", fan)

    # --- score: a small phrase across three gongs (melody = the point)
    print("— score —")
    rng = np.random.default_rng(23)
    gongs = {p: gong_bank(p, np.random.default_rng(100 + int(p)))
             for p in [65.4, 82.0, 98.0, 130.8]}          # C2 E2 G2 C3
    phrase = [
        (0.5, 65.4, 1.0), (4.0, 98.0, 0.5), (6.5, 82.0, 0.7),
        (9.0, 130.8, 0.45), (10.5, 98.0, 0.6), (13.0, 65.4, 0.9),
        (17.5, 82.0, 0.35), (19.0, 98.0, 0.4), (20.0, 130.8, 0.55),
        (23.0, 65.4, 1.0),
    ]
    score = np.zeros((int(38 * SR), 2))
    for (t, p, v) in phrase:
        v = v * (1 + 0.06 * rng.standard_normal())
        place(score, strike(gongs[p], v), t)
    write_wav(f"{OUT}/gong_score.wav", score)


# ---------------------------------------------------------------- export
# The engine transcription: per strike, two mode-row tables (full-glide =
# low+mid, half-glide = high bloom pairs + hit transient), velocity folded
# into the amps, bloom pairs expanded to ±amp row pairs. The reference
# renderer below uses ENGINE semantics (one warped clock for envelope AND
# phase, gate on d>0), so gong_engine_ref.wav is the ear-oracle for
# `diffcli render-graph` — recipe drift and engine bugs don't mix.

BETA0, GLOBAL_G = 0.06, 1.8


def strike_rows(modes, vel):
    """Mode dicts -> (full_rows, half_rows), rows [f, sigma, amp, phase]."""
    full, half = [], []
    for m in modes:
        reg = m["register"]
        a = m["amp"] * (vel ** 2 if reg == "high" else
                        vel ** 1.5 if reg == "hit" else vel)
        ph = float(m["ph"][0])
        f, d1 = float(m["f"]), float(m["d1"])
        if reg == "high":
            half.append([f, d1, a, ph])
            half.append([f, d1 + float(m["d2"]), -a, ph])
        elif reg == "hit":
            half.append([f, d1, a, ph])
        else:
            full.append([f, d1, a, ph])
    return full, half


def render_rows(full, half, beta, g, dur):
    """ENGINE semantics: one warped clock per register (envelope and phase
    both read it), causal gate on unwarped d>0. Mono."""
    d = np.arange(int(dur * SR)) / SR
    dpos = np.maximum(d, 0.0)
    y = np.zeros(len(d))
    for rows, scale in ((full, 1.0), (half, 0.5)):
        w = d + scale * beta * (1.0 - np.exp(-g * dpos)) / g
        for (f, s, a, ph) in rows:
            y += a * np.exp(-s * w) * np.cos(2 * np.pi * f * w + ph)
    return y * (d > 0)


def export_graph():
    import json
    rng = np.random.default_rng(23)
    pitches = [65.4, 82.0, 98.0, 130.8]
    gongs = {p: gong_bank(p, np.random.default_rng(100 + int(p)))
             for p in pitches}
    phrase = [
        (0.5, 65.4, 1.0), (4.0, 98.0, 0.5), (6.5, 82.0, 0.7),
        (9.0, 130.8, 0.45), (10.5, 98.0, 0.6), (13.0, 65.4, 0.9),
        (17.5, 82.0, 0.35), (19.0, 98.0, 0.4), (20.0, 130.8, 0.55),
        (23.0, 65.4, 1.0),
    ]
    DUR = 38.0
    nodes, mix_ins = [], []
    ref = np.zeros(int(DUR * SR))
    for i, (t, p, v) in enumerate(phrase):
        v = float(v * (1 + 0.06 * rng.standard_normal()))
        full, half = strike_rows(gongs[p], v)
        beta = BETA0 * v
        rnd = lambda rows: [[round(x, 9) for x in r] for r in rows]
        nodes.append({"id": f"gong{i}", "kind": "gong", "params": {
            "t": t, "beta": round(beta, 9), "g": GLOBAL_G,
            "modes_full": rnd(full), "modes_half": rnd(half)}})
        mix_ins.append(f"gong{i}")
        dur = min(TAIL, DUR - t)
        y = render_rows(full, half, beta, GLOBAL_G, dur)
        place(ref[:, None], y[:, None], t)   # place expects 2D
    nodes.append({"id": "mixer", "kind": "mix", "in": {"in": mix_ins}})
    nodes.append({"id": "master", "kind": "out", "in": {"in": ["mixer"]}})
    graph = {"nodes": nodes, "out": "master"}
    path = f"{OUT}/gong_score_graph.json"
    with open(path, "w") as fh:
        json.dump(graph, fh)
    print(f"wrote {path}  ({len(nodes)} nodes, "
          f"{sum(len(n['params'].get('modes_full', [])) + len(n['params'].get('modes_half', [])) for n in nodes if n['kind'] == 'gong')} mode rows)")
    write_wav(f"{OUT}/gong_engine_ref.wav", np.stack([ref, ref], axis=1))


if __name__ == "__main__":
    import sys
    if "export" in sys.argv:
        export_graph()
    else:
        main()
