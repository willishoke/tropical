"""demo1 extended: 60 s stateless closed-form piece.

Same engine as the 10 s demos — y(tau) = Re sum_i g_i(tau) e^{mu_i tau} C_i(tau)
with C_i a prefix sum over events — plus two ideas the short demo lacked:

  - filter sweeps: filters are DIAGONAL in the mode basis, so a time-varying
    resonant lowpass is a closed-form gain surface g(omega, tau). No biquad
    state; the sweep is addressable and reverses with the master clock.
  - resonance on the verge of blowing up: (a) a high-Q peak (up to +28 dB)
    scanning across the drone's harmonic series, and (b) true double poles —
    tau e^{mu tau} terms, a resonator driven exactly at resonance, swelling
    linearly before its tiny damping wins.

All automation is in SCENE time, so the reverse pass replays the sweep and
the swells backwards, exactly.
"""
import numpy as np
import wave
import os

SR = 44100
DUR = 60.0
N = int(SR * DUR)
OUT = "/Users/willishoke/tropical/demos"


# ---------------------------------------------------------------- engine

def render(modes, ev_t, coup, taus, gain=None, chunk=4096):
    """gain: optional callable (t_chunk (c,), f (M,)) -> (c, M) real —
    a mode-diagonal, time-varying output filter."""
    freqs = modes.imag / (2 * np.pi)
    growth = np.exp(-np.outer(ev_t, modes))
    pref = np.zeros((len(ev_t) + 1, len(modes), 2), complex)
    pref[1:] = np.cumsum(coup * growth[:, :, None], axis=0)
    seg = np.searchsorted(ev_t, taus, side="right")
    out = np.zeros((len(taus), 2))
    for s in np.unique(seg):
        if s == 0:
            continue
        idx = np.where(seg == s)[0]
        C = pref[s]
        for a in range(0, len(idx), chunk):
            ii = idx[a:a + chunk]
            E = np.exp(np.outer(taus[ii], modes))
            if gain is not None:
                E = E * gain(taus[ii], freqs)
            out[ii] = (E @ C).real
    return out


def render_swells(swells, taus, chunk=1 << 16):
    """Double-pole voices: sum of a * (tau-t0) e^{mu (tau-t0)} for tau > t0."""
    out = np.zeros((len(taus), 2))
    for a in range(0, len(taus), chunk):
        t = taus[a:a + chunk]
        acc = np.zeros((len(t), 2))
        for (t0, comps) in swells:
            d = np.maximum(t - t0, 0.0)
            for (f, sg, amp2) in comps:
                env = d * np.exp(-sg * d)
                ph = np.exp(2j * np.pi * f * d)
                for c in range(2):
                    acc[:, c] += (amp2[c] * ph).real * env
        out[a:a + chunk] = acc
    return out


# ---------------------------------------------------------------- helpers

def write_wav(path, y):
    assert np.all(np.isfinite(y)), "non-finite samples"
    y = y / np.max(np.abs(y)) * 0.9
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
    print(f"wrote {path}  ({len(y)/SR:.2f} s, rms {np.sqrt(np.mean(y**2)):.3f})")


def cplx_randn(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def voice_bank(events, npart, sig_fn, amp_fn, rng):
    """events: list of (t, f0, vel). Returns modes, ev_t, coup with one
    exclusive block of npart partials per event."""
    E = len(events)
    modes = np.zeros(E * npart, complex)
    coup = np.zeros((E, E * npart, 2), complex)
    k = np.arange(1, npart + 1)
    for j, (t, f0, vel) in enumerate(events):
        modes[j * npart:(j + 1) * npart] = -sig_fn(k) + 2j * np.pi * f0 * k
        amp = vel * amp_fn(k) * np.exp(2j * np.pi * rng.random(npart))
        for c in range(2):
            coup[j, j * npart:(j + 1) * npart, c] = amp * np.exp(
                2j * np.pi * 0.02 * rng.random(npart))
    ev_t = np.array([t for t, _, _ in events])
    order = np.argsort(ev_t, kind="stable")
    return modes, ev_t[order], coup[order]


# ---------------------------------------------------------------- score

def master_clock():
    # forward build -> brake -> reverse (reverse reverb + reversed sweep)
    # -> 2x blast (octave up) -> settle -> slow-motion ending
    segs = [(0, 30, 1, 1), (30, 32, 1, -1), (32, 37, -1, -1),
            (37, 39, -1, 2), (39, 46, 2, 2), (46, 48, 2, 1),
            (48, 54, 1, 1), (54, 60.01, 1, 0.25)]
    t = np.arange(N) / SR
    v = np.zeros(N)
    for (t0, t1, v0, v1) in segs:
        m = (t >= t0) & (t < t1)
        x = (t[m] - t0) / (t1 - t0)
        v[m] = v0 + (v1 - v0) * (1 - np.cos(np.pi * x)) / 2
    return np.cumsum(v) / SR


# swept resonant lowpass, defined over SCENE time
FC_T = [0, 8, 14, 18, 26, 32, 38, 44, 48, 53]
FC_F = [110, 1100, 650, 300, 4200, 900, 2600, 5000, 400, 130]
RES_A = [1.5, 4, 4, 18, 10, 6, 12, 26, 5, 2]
RES_S = [0.08, 0.07, 0.07, 0.028, 0.035, 0.05, 0.035, 0.025, 0.06, 0.08]


def sweep_gain(t, f):
    fc = np.exp(np.interp(t, FC_T, np.log(FC_F)))
    wob = np.clip((t - 8) / 4, 0, 1) * 0.35
    fc = fc * 2 ** (wob * np.sin(2 * np.pi * 0.22 * t))
    A = np.interp(t, FC_T, RES_A)
    s = np.interp(t, FC_T, RES_S)
    r = f[None, :] / fc[:, None]
    lp = 1.0 / np.sqrt(1.0 + r ** 4)
    pk = A[:, None] * np.exp(-np.log(r) ** 2 / (2 * s[:, None] ** 2))
    return lp + pk


def build_score(rng):
    drone = [(0.2, 55.0, 1.0)]
    bass = [(t, [82.41, 110.0][i % 2], 0.8 + 0.3 * rng.random())
            for i, t in enumerate(np.arange(2.0, 46.1, 4.0))]
    chords = {8: [220.0, 261.63, 329.63],          # Am
              13: [174.61, 220.0, 261.63, 349.23],  # F
              18: [261.63, 329.63, 392.0],          # C
              23: [196.0, 246.94, 293.66, 392.0],   # G
              28: [220.0, 261.63, 329.63],
              33: [174.61, 220.0, 261.63, 349.23],
              38: [261.63, 329.63, 392.0],
              43: [220.0, 261.63, 329.63, 440.0]}   # final Am(add8)
    chord_ev = [(float(t), f, 0.55 + 0.1 * rng.random())
                for t, fs in chords.items() for f in fs]
    pent = [440.0, 523.25, 587.33, 659.26, 783.99, 880.0]
    mel, t, i = [], 10.0, 2
    while t < 47.0:
        if rng.random() > 0.15:
            mel.append((t, pent[i], 0.5 + 0.5 * rng.random()))
        i = int(np.clip(i + rng.choice([-2, -1, -1, 1, 1, 2]), 0, 5))
        t += rng.choice([0.25, 0.35, 0.5, 0.5, 0.7])
    swells = [
        (22.0, [(164.81, 0.32, 0.9 * np.exp(2j * np.pi * rng.random(2))),
                (329.63, 0.40, 0.45 * np.exp(2j * np.pi * rng.random(2)))]),
        (42.0, [(220.0, 0.30, 0.8 * np.exp(2j * np.pi * rng.random(2))),
                (329.63, 0.35, 0.5 * np.exp(2j * np.pi * rng.random(2))),
                (331.10, 0.35, 0.5 * np.exp(2j * np.pi * rng.random(2)))])]
    return drone, bass, chord_ev, mel, swells


def main():
    rng = np.random.default_rng(11)
    w = master_clock()
    print(f"scene-time range along warp: [{w.min():.2f}, {w.max():.2f}] s")

    drone, bass, chord_ev, mel, swells = build_score(rng)

    banks = [
        voice_bank(drone, 16, lambda k: 0.12 + 0.05 * k,
                   lambda k: 0.9 / k ** 0.8, rng),
        voice_bank(bass, 8, lambda k: 1.4 + 0.35 * k, lambda k: 0.5 / k, rng),
        voice_bank(chord_ev, 6, lambda k: 0.8 + 0.2 * k,
                   lambda k: 0.45 / k, rng),
        voice_bank(mel, 5, lambda k: 2.0 + 0.5 * k, lambda k: 0.5 / k, rng),
    ]
    dry = np.zeros((N, 2))
    for modes, ev, coup in banks:
        dry += render(modes, ev, coup, w, gain=sweep_gain)
    print(f"dry voices rendered ({sum(len(b[0]) for b in banks)} modes, "
          f"{sum(len(b[1]) for b in banks)} events)")

    swell = render_swells(swells, w)
    print("double-pole swells rendered")

    # room: all pitched events couple in, spectrally enveloped by pitch
    all_ev = sorted(drone + bass + chord_ev + mel)
    room_M = 500
    f_r = np.sqrt(rng.uniform(60 ** 2, 9000 ** 2, room_M))
    rt = 3.8 + (1.3 - 3.8) * np.log(f_r / 60) / np.log(9000 / 60)
    mu_r = -6.91 / rt + 2j * np.pi * f_r
    coup_r = np.zeros((len(all_ev), room_M, 2), complex)
    for j, (t, f0, vel) in enumerate(all_ev):
        env = vel / (1.0 + (f_r / (8 * f0)) ** 2)
        coup_r[j] = env[:, None] * cplx_randn(rng, room_M, 2)
    wet = render(mu_r, np.array([t for t, _, _ in all_ev]), coup_r, w)
    print("room rendered")

    # measured mix: room ~0.8 of dry; swell peaks ~2.3x dry rms; soft limit
    rms = lambda x: np.sqrt(np.mean(x ** 2))
    wet *= 0.8 * rms(dry) / rms(wet)
    win = SR // 2
    sw_peak = max(rms(swell[a:a + win]) for a in range(0, N - win, win))
    swell *= 2.3 * rms(dry) / sw_peak
    mix = dry + wet + swell
    L = 3.2 * rms(mix)
    mix = L * np.tanh(mix / L)

    # the drone's longest partials (RT60 ~ 40 s) outlive the piece; close
    # with a 4 s cosine fade instead of cutting them at -17 dB
    nf = 4 * SR
    fade = np.cos(np.linspace(0, np.pi / 2, nf)) ** 2
    mix[-nf:] *= fade[:, None]

    os.makedirs(OUT, exist_ok=True)
    write_wav(f"{OUT}/demo1_timewarp_scrub_60s.wav", mix)


if __name__ == "__main__":
    main()
