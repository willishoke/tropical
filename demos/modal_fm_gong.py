"""modal_fm_gong — FM the struck nonlinear gong. Two meanings, both exact.

heterodyne FM  every mode gets the SAME ABSOLUTE frequency deviation. In the
    analytic (complex) representation the modulation factor is common across
    the bank and factors out of the sum:

        y = Re( e^{i phi_mod(d)} * SUM_k A_k e^{mu_k d} )

    — FM-ing the whole 116-mode gong costs ONE complex rotation per sample,
    independent of bank size. phi_mod comes from `integrate_bank` (static or
    decaying index — exact FM, per modal_fm.py D5). Low partials deviate
    wildly in relative terms (through-zero for modes with f < dev), highs
    barely move. D6 referees: the heterodyne product == the fused bank where
    every mode sprouts the same Bessel sidebands (mu_k + i n w_m, amp
    A_k J_n(b)) — so the FM'd gong is STILL a modal object with poles.

varispeed FM  modulate the CLOCK: w = d + p sin(w_m d)/w_m composed with the
    gong's own pitch-bloom warp (two nonlinear clock transforms stacking —
    the slide law's warp-fuse, in sound). Deviation PROPORTIONAL to each
    partial's frequency: the fundamental barely moves (index p f_k/f_m is
    tiny down low), the shimmer band gets indices >1 and smears. This is
    "FM the tape speed", and it is the Fourier-dual sound of the heterodyne.

Both are closed-form f(tau) — the FM'd gong scrubs. The renders are linear
(the poly drive is retired); the printed budget is just fmax + Bessel
spread vs Nyquist, which the FM'd bank clears by an order of magnitude.

Renders (one hard strike each, vel 1.0):
  gong_fm_reference.wav           the standard driven gong
  gong_fm_heterodyne_static.wav   uniform 287 Hz deviation, constant
  gong_fm_heterodyne_dynamic.wav  deviation 690 Hz -> 0, sigma_b 1.3 (the
                                  FM bite fades as the gong blooms)
  gong_fm_varispeed.wav           8% proportional deviation at f_m = 1.4 f0
"""
import numpy as np
from gong import gong_bank, write_wav, SR, OUT, BETA0, GLOBAL_G
from modal_arrow import Atom, eval_complex
from modal_vco import integrate_bank, TWO_PI
from modal_fm import bessel_j, static_mod, dynamic_mod

TAIL = 12.0


# ------------------------------------------------- the gong, analytically

def render_strike_cx(modes, vel, d, beta0=BETA0, g=GLOBAL_G):
    """gong.py's render_strike with cos -> e^{i.}: the analytic strike,
    stereo complex. Same registers, velocity couplings, bloom envelopes,
    per-register pitch-bloom warp."""
    z = np.zeros((len(d), 2), complex)
    beta = beta0 * vel
    warp = d + beta * (1.0 - np.exp(-g * d)) / g
    for m in modes:
        env = np.exp(-m["d1"] * d)
        if m["d2"] is not None:
            env = env - np.exp(-(m["d1"] + m["d2"]) * d)
        a = m["amp"] * (vel ** 2 if m["register"] == "high" else
                        vel ** 1.5 if m["register"] == "hit" else vel)
        w = warp if m["register"] in ("low", "mid") else d + 0.5 * (warp - d)
        arg = TWO_PI * m["f"] * w
        for c in range(2):
            g_c = m["pan"] if c == 0 else 1 - m["pan"]
            z[:, c] += a * g_c * env * np.exp(1j * (arg + m["ph"][c]))
    return z


def render_strike_varispeed(modes, vel, d, p, f_m):
    """The clock route: the FM term joins the warp itself, so it composes
    with the pitch bloom and every partial deviates proportionally."""
    w_m = TWO_PI * f_m
    fm = p * np.sin(w_m * d) / w_m
    beta = BETA0 * vel
    warp = d + beta * (1.0 - np.exp(-GLOBAL_G * d)) / GLOBAL_G + fm
    z = np.zeros((len(d), 2), complex)
    for m in modes:
        env = np.exp(-m["d1"] * d)
        if m["d2"] is not None:
            env = env - np.exp(-(m["d1"] + m["d2"]) * d)
        a = m["amp"] * (vel ** 2 if m["register"] == "high" else
                        vel ** 1.5 if m["register"] == "hit" else vel)
        w = warp if m["register"] in ("low", "mid") else d + 0.5 * (warp - d + fm)
        arg = TWO_PI * m["f"] * w
        for c in range(2):
            g_c = m["pan"] if c == 0 else 1 - m["pan"]
            z[:, c] += a * g_c * env * np.exp(1j * (arg + m["ph"][c]))
    return z.real


def mod_phase(d, mod_atoms):
    """phi_mod = INT_0^d modulator, via the residue transform (exact FM)."""
    ib = integrate_bank(mod_atoms)
    return eval_complex(ib, np.maximum(d, 1e-12)).real - \
        eval_complex(ib, np.full(1, 1e-12)).real[0]


# ---------------------------------------------------------------- referee

def test_heterodyne_is_a_bank():
    """D6: the one-rotation heterodyne product == the fused bank where every
    mode carries the same Bessel sidebands. The FM'd gong keeps its poles."""
    rng = np.random.default_rng(3)
    f_m, b = 113.0, 2.2
    small = [Atom(0.0, -s + 1j * TWO_PI * f, a * np.exp(2j * np.pi * ph), 0)
             for (f, s, a, ph) in zip(rng.uniform(60, 900, 5),
                                      rng.uniform(0.3, 1.5, 5),
                                      rng.uniform(0.2, 1.0, 5),
                                      rng.random(5))]
    taus = np.arange(1, int(1.5 * 8000)) / 8000
    het = (eval_complex(small, taus)
           * np.exp(1j * b * np.sin(TWO_PI * f_m * taus))).real
    N = 24
    fused = [Atom(0.0, a.pole + 1j * n * TWO_PI * f_m,
                  a.amp * bessel_j(n, b), 0)
             for a in small for n in range(-N, N + 1)]
    rel = np.linalg.norm(eval_complex(fused, taus).real - het) \
        / np.linalg.norm(het)
    print(f"  D6 heterodyne (1 rotation) vs fused bank "
          f"({len(fused)} modes): rel L2 = {rel:.2e}")
    assert rel < 1e-10, rel
    print("D6 PASS: heterodyne FM of a bank is O(1) per sample AND still a "
          "modal bank — the FM'd gong composes with the residue calculus")


# ---------------------------------------------------------------------- audio

def main():
    test_heterodyne_is_a_bank()

    f0 = 82.0
    modes = gong_bank(f0, np.random.default_rng(7))
    f_m = 1.4 * f0                                # 114.8 Hz, the bell ratio
    fmax = max(m["f"] for m in modes)
    d = np.arange(int(TAIL * SR)) / SR

    # bandwidth: the modulation adds ~ (b+2) f_m of Bessel spread
    for (name, spread) in (("heterodyne b=2.5", (2.5 + 2) * f_m),
                           ("varispeed p=0.08", (0.08 * fmax / f_m + 2) * f_m)):
        top = fmax + spread
        ok = "OK" if top < SR / 2 else "ALIASES"
        print(f"  bandwidth [{name}]: {fmax:.0f} + {spread:.0f} = "
              f"{top / 1000:.1f} kHz vs 22.05 kHz  [{ok}]")
    thru = sum(1 for m in modes if m["f"] < 2.5 * f_m)
    print(f"  through-zero: {thru}/{len(modes)} modes have f < the 287 Hz "
          f"deviation — the low register goes through zero (defined: the "
          f"analytic signal folds)")

    z = render_strike_cx(modes, 1.0, d)

    write_wav(f"{OUT}/gong_fm_reference.wav", z.real)

    ph_s = mod_phase(d, static_mod(0.0, 2.5, f_m))
    write_wav(f"{OUT}/gong_fm_heterodyne_static.wav",
              (z * np.exp(1j * ph_s)[:, None]).real)

    ph_d = mod_phase(d, dynamic_mod(0.0, 6.0, 1.3, f_m))
    write_wav(f"{OUT}/gong_fm_heterodyne_dynamic.wav",
              (z * np.exp(1j * ph_d)[:, None]).real)

    write_wav(f"{OUT}/gong_fm_varispeed.wav",
              render_strike_varispeed(modes, 1.0, d, 0.08, f_m))


if __name__ == "__main__":
    main()
