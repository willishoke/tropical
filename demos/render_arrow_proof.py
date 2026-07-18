"""Audio proof of the Modal arrow algebra: the SAME pipeline the tests verify,
now rendered to sound. `voice >> reverb >> eval`, where `reverb` is the exact
residue-calculus Filter — the reverb's couplings are DERIVED from the voice
(a r_q/(λ-ν_q)), not hand-tuned. Evaluated along a forward-then-reverse master
clock to show the exact composition scrubs.

Also renders an A/B: the identical room modes driven by RANDOM couplings (the
demo's earlier fake). Same poles, same decay — only the connection law differs.
"""
import numpy as np
import wave
import os
from modal_arrow import Obj, Atom, Voice, Filter, LTI, Eval, eval_fast

SR = 44100
OUT = "/Users/willishoke/tropical/demos"


def write_wav(path, y):
    assert np.all(np.isfinite(y)), "non-finite samples"
    y = y / np.max(np.abs(y)) * 0.9
    f = int(0.01 * SR)
    e = np.ones(len(y)); e[:f] = np.linspace(0, 1, f); e[-f:] = np.linspace(1, 0, f)
    y = y * e[:, None]
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    rms = np.sqrt(np.mean(y ** 2))
    print(f"wrote {path}  ({len(y)/SR:.2f}s, rms {rms:.3f}, "
          f"crest {np.max(np.abs(y)) / rms:.1f})")


def room_poles(M, rng, band=(80, 8500), rt=(2.4, 0.8)):
    f = np.sqrt(rng.uniform(band[0] ** 2, band[1] ** 2, M))
    x = np.log(f / band[0]) / np.log(band[1] / band[0])
    sigma = 6.91 / (rt[0] + (rt[1] - rt[0]) * x)
    return -sigma + 2j * np.pi * f, f


def warp_clock():
    # forward build, brake, reverse (reverse reverb), fast-forward, settle
    segs = [(0, 8, 1, 1), (8, 9, 1, -1), (9, 13, -1, -1),
            (13, 14, -1, 1.5), (14, 20.01, 1.5, 1.5)]
    n = int(20 * SR); t = np.arange(n) / SR; v = np.zeros(n)
    for (t0, t1, v0, v1) in segs:
        m = (t >= t0) & (t < t1); x = (t[m] - t0) / (t1 - t0)
        v[m] = v0 + (v1 - v0) * (1 - np.cos(np.pi * x)) / 2
    return np.cumsum(v) / SR


def main():
    rng = np.random.default_rng(11)

    # --- the voice: Events ⇝ Modal ---
    # light modal damping (long ring) with random modal phases at excitation:
    # random phases stop all partials starting coherently (a click), and the
    # light voice damping stays well clear of the room's heavy damping so the
    # residue couplings a·r_q/(λ-ν_q) never blow up on a chance coincidence.
    voice = Voice(lambda f0: [(k, 0.6 + 0.15 * k,
                               (1.0 / k ** 1.1) * np.exp(2j * np.pi * rng.random()))
                              for k in range(1, 9)])
    # polyphony: two triads + a walking bass
    chord = lambda t, fs, v=0.8: [(t, f, v) for f in fs]
    events = (chord(0.3, [110, 164.81, 220]) + chord(2.3, [146.83, 220, 293.66]) +
              chord(4.3, [130.81, 196, 261.63]) + chord(6.3, [110, 164.81, 220, 329.63]) +
              [(t, 55.0 * (1 + (i % 2)), 0.7) for i, t in enumerate(np.arange(1.0, 7.1, 1.0))])
    dry_atoms = voice(events)

    nu, f_r = room_poles(400, rng)
    warp = warp_clock()
    print(f"scene-time along warp: [{warp.min():.2f}, {warp.max():.2f}] s")

    def stereo(atoms):
        return np.stack([eval_fast(atoms, warp), eval_fast(atoms, warp)], 1)

    dry = np.stack([eval_fast(dry_atoms, warp)] * 2, 1)

    # --- exact-residue reverb: the arrow `voice >> Filter` ---
    # room residues per channel give width; the coupling to the voice is the
    # residue law, computed inside Filter — NOT specified here.
    rL = (rng.normal(size=400) + 1j * rng.normal(size=400)) / np.sqrt(400)
    rR = (rng.normal(size=400) + 1j * rng.normal(size=400)) / np.sqrt(400)
    wetL = eval_fast(Filter(LTI(nu, rL))(dry_atoms), warp)
    wetR = eval_fast(Filter(LTI(nu, rR))(dry_atoms), warp)
    wet = np.stack([wetL, wetR], 1)

    def mix(dry, wet, ratio=0.85):
        rms = lambda x: np.sqrt(np.mean(x ** 2))
        g = ratio * rms(dry) / rms(wet)
        m = dry + g * wet
        L = 3.0 * rms(m)
        return L * np.tanh(m / L)

    os.makedirs(OUT, exist_ok=True)
    write_wav(f"{OUT}/arrow_exact_residue_reverb.wav", mix(dry, wet))

    # --- A/B: SAME modes, RANDOM couplings (the earlier fake) ---
    ev_t = np.array(sorted({t for t, _, _ in events}))
    ti = {t: i for i, t in enumerate(ev_t)}
    fake_atoms = []
    for (t, f0, vel) in events:
        env = vel / (1.0 + (f_r / (8 * f0)) ** 2)      # heuristic spectral env
        c = env * (rng.normal(size=400) + 1j * rng.normal(size=400))
        for q in range(400):
            fake_atoms.append(Atom(t, nu[q], c[q], 0))
    fakeL = eval_fast(fake_atoms, warp)
    fakeR = eval_fast([Atom(a.t0, a.pole, a.amp * np.exp(2j * np.pi * rng.random()), 0)
                       for a in fake_atoms], warp)
    fake = np.stack([fakeL, fakeR], 1)
    write_wav(f"{OUT}/arrow_random_couplings.wav", mix(dry, fake))

    # --- verifications ---
    # reverse identity through the exact composition
    full = mix(dry, wet)
    fwd = full[int(4.5 * SR):int(8.0 * SR), 0]
    rev = full[int(9.0 * SR):int(12.5 * SR), 0][::-1]
    n = min(len(fwd), len(rev))
    print(f"reverse-through-reverb identity corr: "
          f"{np.corrcoef(fwd[:n], rev[:n])[0, 1]:.4f}")
    # coupling structure: aggregate energy PER room mode (sum same-pole atoms
    # first), then report the participation ratio. Exact residues shape the
    # room's excitation by physical proximity to the voice's partials; random
    # couplings light every mode equally.
    def per_mode_energy(atoms, poles):
        idx = {p: i for i, p in enumerate(poles)}
        e = np.zeros(len(poles), complex)
        for a in atoms:
            if a.pole in idx:
                e[idx[a.pole]] += a.amp
        return np.abs(e) ** 2

    def pr(e):
        return (e.sum() ** 2) / (e ** 2).sum() / len(e)

    poles = list(nu)
    e_exact = per_mode_energy(Filter(LTI(nu, rL))(dry_atoms), poles)
    e_rand = per_mode_energy(fake_atoms, poles)
    print(f"room-mode participation ratio — exact residues {pr(e_exact):.3f} "
          f"vs random {pr(e_rand):.3f} (fraction of the 400 modes effectively "
          f"active; exact = physically shaped by the voice, random = flat)")


if __name__ == "__main__":
    main()
