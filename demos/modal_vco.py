"""modal_vco — the LFO-wired-to-a-pole canary.

The modal VCO: every LFO is secretly a sigma=0 modal atom. Wire it to another
bank's pole ("cutoff") and the language has to pick a READING of a time-varying
pole. Two candidates, both closed-form f(tau), both random-access:

  snapshot    phi(tau) = omega(tau) * d              (d = tau - t0)
              — what `modalBankSig` computes TODAY if `omega` is a live Sig:
              the pole is read at tau and applied over the whole elapsed d.
  integrated  phi(tau) = omega0*d + depth * INT_{t0}^{tau} lfo(u) du
              — the exact solution of xdot = mu(tau) x (the analog resonator
              actually being modulated). Still closed-form: an LFO bank
              integrates by the residue transform  a -> a/mu  (+ a DC atom),
              so this is PM of the integrated bank — build-time algebra.

The instantaneous frequency of the snapshot reading is

    omega(tau) + omega'(tau) * d

so the modulation DEPTH GROWS with time-since-strike: envelope
sqrt(1 + (w_m d)^2) times the intended depth. A gentle vibrato curdles into
deep FM as the note rings; each strike carries its own d, so overlapping tails
warble at different depths. The integrated reading holds depth constant.
The two agree in the adiabatic limit (error O(w_m^2)) — but the regime is
w_m*d << 1, d bounded by the tail, so the safety criterion is sweep-slow-
relative-to-DECAY (w_m << sigma), not slow in absolute terms. Measured: even
the 0.15 Hz deep sweep decorrelates fully (rel L2 gap 1.39) over ~2 s tails.

Differentials (house style: each against an independent ground truth, with a
convergence RATE so a bug plateaus instead of passing):
  D1 integrated == RK4 solution of the time-varying ODE (error ~ h^4);
     snapshot != ODE (plateaus).
  D2 snapshot -> integrated as w_m -> 0 at rate O(w_m^2).
  D3 the residue-transform integral of the LFO bank == cumulative trapezoid
     of the evaluated LFO (error ~ h^2).

Renders: reference (no mod), snapshot vs integrated at vibrato rate (5.5 Hz,
+-25 cents), and the same pair as a slow deep sweep (0.15 Hz, +-7 semitones)
— the adiabatic regime where the readings converge.
"""
import numpy as np
import wave
from modal_arrow import Atom, eval_complex

SR = 44100
OUT = "/Users/willishoke/tropical/demos"
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------- the modal VCO

def lfo_atoms(f_m, t0=0.0):
    """cos(w_m (tau - t0)) as a single undamped modal atom (Re of e^{i w_m d})."""
    return [Atom(t0, 1j * TWO_PI * f_m, 1.0 + 0j, 0)]


def integrate_bank(atoms):
    """INT_{t0}^{tau} of a deg-0 bank, exactly: a -> a/mu at the same pole,
    plus the -a/mu DC atom that zeroes the value at t0. Pure residue algebra —
    the 'FM = PM after dividing residues by poles' transform."""
    out = []
    for a in atoms:
        assert a.deg == 0 and a.pole != 0
        out.append(Atom(a.t0, a.pole, a.amp / a.pole, 0))
        out.append(Atom(a.t0, 0.0 + 0j, -a.amp / a.pole, 0))
    return out


def eval_real(atoms, taus):
    return eval_complex(atoms, taus).real


# ------------------------------------------------- the two readings of a pole

def strike_snapshot(taus, t0, f0, sigma, amp, phase, p, lfo):
    """One complex mode, pole read at tau, applied over the whole elapsed d."""
    d = np.maximum(taus - t0, 0.0)
    om = TWO_PI * f0 * (1.0 + p * lfo(taus))
    x = amp * np.exp(1j * phase) * np.exp((-sigma + 1j * om) * d)
    return np.where(taus > t0, x, 0.0)


def strike_integrated(taus, t0, f0, sigma, amp, phase, p, lfo_int):
    """Same mode, exact ODE solution: phase advances by the INTEGRAL of the
    modulated frequency. lfo_int(tau) = INT_0^tau lfo — per-strike offset
    subtracts the value at t0 (integral from the strike)."""
    d = np.maximum(taus - t0, 0.0)
    L = lfo_int(taus) - lfo_int(np.full_like(taus, t0))
    ph = TWO_PI * f0 * (d + p * L)
    x = amp * np.exp(1j * phase) * np.exp(-sigma * d + 1j * ph)
    return np.where(taus > t0, x, 0.0)


def render_bank(taus, strikes, partials, p, f_m, reading):
    """A struck metallic bank under one reading. strikes: (t0, f0, vel).
    partials: (ratio, sigma, rel_amp, phase)."""
    lfo_a = lfo_atoms(f_m)
    lfo = lambda ts: eval_real(lfo_a, ts)
    lfo_int = lambda ts: eval_real(integrate_bank(lfo_a), ts)
    y = np.zeros(len(taus), complex)
    for (t0, f0, vel) in strikes:
        for (ratio, sigma, a, ph) in partials:
            if reading == "snapshot":
                y += strike_snapshot(taus, t0, f0 * ratio, sigma, vel * a, ph, p, lfo)
            elif reading == "integrated":
                y += strike_integrated(taus, t0, f0 * ratio, sigma, vel * a, ph, p, lfo_int)
            else:  # reference
                y += strike_snapshot(taus, t0, f0 * ratio, sigma, vel * a, ph, 0.0, lfo)
    return y.real


# ---------------------------------------------------------------- differentials

def rk4(mu, x0, t0, t1, n):
    """xdot = mu(tau) x, complex scalar."""
    h = (t1 - t0) / n
    x, t = x0, t0
    for _ in range(n):
        k1 = mu(t) * x
        k2 = mu(t + h / 2) * (x + h / 2 * k1)
        k3 = mu(t + h / 2) * (x + h / 2 * k2)
        k4 = mu(t + h) * (x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return x


def test_ode_ground_truth():
    """D1: integrated reading == the ODE being modulated; snapshot != it."""
    f0, sigma, p, f_m, T = 5.0, 0.35, 0.08, 0.8, 2.0
    w_m = TWO_PI * f_m
    mu = lambda t: -sigma + 1j * TWO_PI * f0 * (1 + p * np.cos(w_m * t))
    taus = np.array([T])
    lfo = lambda ts: np.cos(w_m * ts)
    lfo_int = lambda ts: np.sin(w_m * ts) / w_m
    xi = strike_integrated(taus, 0.0, f0, sigma, 1.0, 0.0, p, lfo_int)[0]
    xs = strike_snapshot(taus, 0.0, f0, sigma, 1.0, 0.0, p, lfo)[0]

    errs = []
    for n in (500, 1000, 2000):
        gt = rk4(mu, 1.0 + 0j, 0.0, T, n)
        errs.append(abs(xi - gt) / abs(gt))
    print(f"  D1 integrated vs RK4: rel errs {errs[0]:.2e} {errs[1]:.2e} "
          f"{errs[2]:.2e} (ratios {errs[0]/errs[1]:.1f}, {errs[1]/errs[2]:.1f}; "
          f"expect ~16 = h^4)")
    assert 10 < errs[0] / errs[1] < 24 and 10 < errs[1] / errs[2] < 24
    gt = rk4(mu, 1.0 + 0j, 0.0, T, 4000)
    snap_err = abs(xs - gt) / abs(gt)
    print(f"  D1 snapshot vs ODE:   rel err {snap_err:.2e} (plateaus — not a "
          f"solution of the ODE)")
    assert snap_err > 1e-2
    print("D1 PASS: the integrated reading IS the modulated resonator; the "
          "snapshot reading is a different (well-defined) function")


def test_adiabatic_rate():
    """D2: phase gap snapshot-vs-integrated shrinks as O(w_m^2) — IN the
    adiabatic regime w_m*d << 1. The regime matters: for w_m*d >~ 1 the gap
    saturates at ~w0*p*d regardless of w_m, so 'slow sweep' means slow
    relative to the TAIL (w_m << sigma), not slow in absolute terms."""
    f0, p, T = 220.0, 0.0145, 0.5
    gaps = []
    for f_m in (0.1, 0.05, 0.025):
        w_m = TWO_PI * f_m
        taus = np.linspace(1e-4, T, 4000)
        gap = TWO_PI * f0 * p * np.abs(
            np.cos(w_m * taus) * taus - np.sin(w_m * taus) / w_m)
        gaps.append(gap.max())
    print(f"  D2 max phase gap at f_m = 0.1, 0.05, 0.025 Hz over d<=0.5s: "
          f"{gaps[0]:.3e} {gaps[1]:.3e} {gaps[2]:.3e} rad "
          f"(ratios {gaps[0]/gaps[1]:.2f}, {gaps[1]/gaps[2]:.2f}; expect ~4)")
    assert 3.3 < gaps[0] / gaps[1] < 4.7 and 3.3 < gaps[1] / gaps[2] < 4.7
    print("D2 PASS: the readings converge at O(w_m^2) for w_m*d << 1 — the "
          "safety criterion is sweep-slow-relative-to-decay (w_m << sigma), "
          "and outside it the gap saturates at w0*p*d")


def test_residue_integration():
    """D3: integrate_bank (a -> a/mu, + DC atom) == numeric cumulative
    trapezoid of the evaluated LFO, at the trapezoid rate O(h^2)."""
    f_m, T = 3.7, 2.0
    bank = lfo_atoms(f_m)
    ibank = integrate_bank(bank)
    errs = []
    for sr in (2000, 4000, 8000):
        n = int(T * sr)
        t = np.arange(1, n) / sr          # stay past the tau > t0 gate
        y = eval_real(bank, t)
        num = np.concatenate([[0.0], np.cumsum((y[1:] + y[:-1]) / 2)]) / sr
        got = eval_real(ibank, t)
        rel = np.linalg.norm(got - (num + got[0])) / np.linalg.norm(got)
        errs.append(rel)
    print(f"  D3 residue-integral vs trapezoid: rel errs {errs[0]:.2e} "
          f"{errs[1]:.2e} {errs[2]:.2e} (ratios {errs[0]/errs[1]:.1f}, "
          f"{errs[1]/errs[2]:.1f}; expect ~4 = h^2)")
    assert 3.3 < errs[0] / errs[1] < 4.7 and 3.3 < errs[1] / errs[2] < 4.7
    print("D3 PASS: integrating an LFO is the residue transform a -> a/mu — "
          "FM is PM of the transformed bank, at build time")


# ------------------------------------------------------------- depth-growth

def measure_depth(reading, p, f_m, f0=220.0, T=5.0):
    """Empirical vibrato depth (cents) vs time-since-strike, from the phase
    derivative of the complex single-mode signal."""
    taus = np.arange(1, int(T * SR)) / SR
    lfo_a = lfo_atoms(f_m)
    lfo = lambda ts: eval_real(lfo_a, ts)
    lfo_int = lambda ts: eval_real(integrate_bank(lfo_a), ts)
    if reading == "snapshot":
        x = strike_snapshot(taus, 0.0, f0, 0.0, 1.0, 0.0, p, lfo)
    else:
        x = strike_integrated(taus, 0.0, f0, 0.0, 1.0, 0.0, p, lfo_int)
    inst = np.gradient(np.unwrap(np.angle(x)), 1.0 / SR) / TWO_PI
    cents = 1200 * np.log2(np.maximum(inst, 1e-9) / f0)
    out = {}
    for d in (0.25, 0.5, 1.0, 2.0, 4.0):
        w = (taus > d - 0.5 / f_m) & (taus < d + 0.5 / f_m)
        out[d] = np.max(np.abs(cents[w]))
    return out


def depth_table(p, f_m):
    snap = measure_depth("snapshot", p, f_m)
    integ = measure_depth("integrated", p, f_m)
    w_m = TWO_PI * f_m
    print(f"\n  vibrato depth vs time-since-strike (intended +-25 cents, "
          f"f_m = {f_m} Hz):")
    print("    d (s)     snapshot   integrated   predicted-snap "
          "(1731*p*sqrt(1+(w_m d)^2))")
    for d in (0.25, 0.5, 1.0, 2.0, 4.0):
        pred = 1200 * np.log2(1 + p * np.sqrt(1 + (w_m * d) ** 2))
        print(f"    {d:<8}  {snap[d]:>7.1f}    {integ[d]:>7.1f}      {pred:>7.1f}")
    print("    (snapshot entries ~4.5e4 = the instantaneous frequency has been "
          "driven THROUGH ZERO\n     and hit the log floor — the vibrato has "
          "become through-zero FM; prediction is the\n     small-deviation "
          "envelope, so measured > predicted once the swing is large)")


# ---------------------------------------------------------------------- audio

def write_wav(path, y):
    assert np.all(np.isfinite(y)), "non-finite samples"
    y = y / np.max(np.abs(y)) * 0.9
    f = int(0.01 * SR)
    e = np.ones(len(y)); e[:f] = np.linspace(0, 1, f); e[-f:] = np.linspace(1, 0, f)
    y = np.stack([y * e, y * e], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    rms = np.sqrt(np.mean(y ** 2))
    print(f"wrote {path}  ({len(y)/SR:.2f}s, rms {rms:.3f}, "
          f"crest {np.max(np.abs(y)) / rms:.1f})")


def main():
    for t in (test_ode_ground_truth, test_adiabatic_rate,
              test_residue_integration):
        t()

    p_vib = 2 ** (25 / 1200) - 1          # +-25 cents
    depth_table(p_vib, 5.5)

    # a struck metallic voice: bar-like inharmonic partials, long ring
    partials = [(r, s, a, TWO_PI * ph) for (r, s, a, ph) in
                [(1.00, 0.55, 1.00, 0.13), (2.76, 0.80, 0.55, 0.71),
                 (5.40, 1.10, 0.30, 0.29), (8.93, 1.50, 0.16, 0.55)]]
    strikes = [(0.3, 220.0, 1.0), (2.8, 146.83, 0.9), (5.3, 293.66, 0.8)]
    taus = np.arange(int(8.5 * SR)) / SR

    for reading in ("reference", "snapshot", "integrated"):
        y = render_bank(taus, strikes, partials, p_vib, 5.5, reading)
        write_wav(f"{OUT}/modal_vco_vibrato_{reading}.wav", y)

    # the adiabatic regime: slow deep sweep (+-7 semitones at 0.15 Hz)
    p_sweep = 2 ** (700 / 1200) - 1
    for reading in ("snapshot", "integrated"):
        y = render_bank(taus, strikes, partials, p_sweep, 0.15, reading)
        write_wav(f"{OUT}/modal_vco_sweep_{reading}.wav", y)
    d = (render_bank(taus, strikes, partials, p_sweep, 0.15, "snapshot")
         - render_bank(taus, strikes, partials, p_sweep, 0.15, "integrated"))
    r = np.linalg.norm(d) / np.linalg.norm(
        render_bank(taus, strikes, partials, p_sweep, 0.15, "integrated"))
    print(f"\n  sweep-pair rel L2 gap (adiabatic regime): {r:.3f}")


if __name__ == "__main__":
    main()
