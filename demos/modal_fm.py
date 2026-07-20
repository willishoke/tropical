"""modal_fm — FM in the modal-only language: integrated vs reference.

FM proper, not a PM stand-in: the instantaneous frequency is

    omega(tau) = w_c + A(tau) * cos(w_m tau),    A(tau) = A0 * e^{-sigma_b tau}

and the phase is its INTEGRAL. In a stream language that integral is a phase
accumulator (state); here the modulator-times-index-envelope is a single
exp-poly atom  Re(A0 e^{(-sigma_b + i w_m) tau}),  so `integrate_bank`
(a -> a/mu, + DC atom) gives the phase in closed form — FM with a decaying
index is EXACT, stateless, random-access. "PM if necessary" is not necessary.

Two ground truths referee the integrated reading:

  D4 (static index): Jacobi-Anger. e^{i(w_c tau + b sin w_m tau)}
     = sum_n J_n(b) e^{i(w_c + n w_m) tau} — so the FM voice IS a modal bank
     (poles at w_c + n w_m, amps J_n(b), the strike decay on every sideband).
     Route A (phase route, via integrate_bank) and Route B (the fused Bessel
     bank, an ordinary deg-0 modal object) must agree to machine precision,
     with superexponential convergence as the sideband count N passes b.
     Route B is the 'static-index FM fuses to a modal bank' claim, executable:
     the fused voice has poles, so it can feed the residue calculus (a reverb).

  D5 (dynamic index): no fused bank exists (the Bessel coefficients would be
     time-varying), so the referee is RK4 on  xdot = i omega(tau) x.  The
     integrated reading must converge at the h^4 rate; this is the case where
     FM and PM genuinely differ (int of A(u)cos(w_m u) du != A(tau) sin(w_m
     tau)/w_m once A decays).

Renders (three strikes, inharmonic bell ratio w_m/w_c = 1.4):
  modal_fm_reference.wav   index 0 — the bare carrier bank
  modal_fm_static.wav      index 3.0 — Route A, asserted == Route B on the grid
  modal_fm_dynamic.wav     index 6 -> 0, sigma_b = 1.1 — brightness decays,
                           the classic struck-FM bell, exact FM via the
                           integrated modulator
"""
import numpy as np
from modal_arrow import Atom, eval_complex
from modal_vco import integrate_bank, write_wav, rk4, SR, OUT, TWO_PI


def bessel_j(n, b, quad=4096):
    """J_n(b) by trapezoid on the periodic integrand (spectral accuracy)."""
    th = np.linspace(0, np.pi, quad + 1)
    return np.trapezoid(np.cos(n * th - b * np.sin(th)), th) / np.pi


# --------------------------------------------------------- the two FM routes

def fm_phase(taus, t0, w_c, mod_atoms):
    """Route A: the FM phase w_c*d + INT_{t0}^{tau} modulator, via the residue
    transform. mod_atoms is the modulator-times-index-envelope bank (rad/s)."""
    ibank = integrate_bank(mod_atoms)
    L = eval_complex(ibank, taus).real - eval_complex(
        ibank, np.full_like(taus, t0)).real
    d = np.maximum(taus - t0, 0.0)
    return w_c * d + L


def fm_voice_A(taus, t0, f_c, sigma, amp, mod_atoms):
    d = np.maximum(taus - t0, 0.0)
    ph = fm_phase(taus, t0, TWO_PI * f_c, mod_atoms)
    x = amp * np.exp(-sigma * d) * np.exp(1j * ph)
    return np.where(taus > t0, x, 0.0)


def fm_voice_B(t0, f_c, sigma, amp, b, f_m, N):
    """Route B: the fused Bessel bank — an ordinary deg-0 modal object.
    Negative-frequency sidebands are legal atoms; Re folds them correctly."""
    return [Atom(t0, -sigma + 1j * TWO_PI * (f_c + n * f_m),
                 amp * bessel_j(n, b), 0)
            for n in range(-N, N + 1)]


def static_mod(t0, b, f_m):
    """Constant-index modulator A(tau) = b*w_m * cos(w_m d) as one atom, so
    the phase integral is b*sin(w_m d) — index b in the PM sense."""
    w_m = TWO_PI * f_m
    return [Atom(t0, 1j * w_m, b * w_m + 0j, 0)]


def dynamic_mod(t0, b0, sigma_b, f_m):
    """Decaying-index modulator A(tau) = b0*w_m*e^{-sigma_b d} cos(w_m d):
    STILL one exp-poly atom — the pole just moves off the axis."""
    w_m = TWO_PI * f_m
    return [Atom(t0, -sigma_b + 1j * w_m, b0 * w_m + 0j, 0)]


# ---------------------------------------------------------------- differentials

def test_jacobi_anger():
    """D4: phase route == fused Bessel bank, superexponentially in N."""
    f_c, f_m, b, sigma, T = 220.0, 308.0, 3.0, 0.7, 1.5
    taus = np.arange(1, int(T * 8000)) / 8000
    a = fm_voice_A(taus, 0.0, f_c, sigma, 1.0, static_mod(0.0, b, f_m)).real
    errs = []
    for N in (4, 8, 16, 32):
        bnk = fm_voice_B(0.0, f_c, sigma, 1.0, b, f_m, N)
        rel = np.linalg.norm(eval_complex(bnk, taus).real - a) / np.linalg.norm(a)
        errs.append(rel)
        print(f"  D4 phase route vs Bessel bank  N={N:>2}: rel L2 = {rel:.2e}")
    assert errs[0] > 1e-3 and errs[-1] < 1e-10, errs
    assert all(e1 / e2 > 10 for e1, e2 in zip(errs[:-2], errs[1:-1])), errs
    print("D4 PASS: static-index FM IS a modal bank (Jacobi-Anger), to machine "
          "precision — the fused voice has poles and can feed the residue "
          "calculus")


def test_dynamic_fm_ode():
    """D5: decaying-index FM (the case with no fused bank) == RK4 of the
    modulated oscillator, at the h^4 rate; and it is NOT the PM shortcut."""
    f_c, f_m, b0, sigma_b, T = 8.0, 11.2, 4.0, 1.2, 1.0
    w_c, w_m = TWO_PI * f_c, TWO_PI * f_m
    mod = dynamic_mod(0.0, b0, sigma_b, f_m)
    omega = lambda t: w_c + b0 * w_m * np.exp(-sigma_b * t) * np.cos(w_m * t)
    taus = np.array([T])
    xi = fm_voice_A(taus, 0.0, f_c, 0.0, 1.0, mod)[0]

    errs = []
    for n in (2000, 4000, 8000):
        gt = rk4(lambda t: 1j * omega(t), 1.0 + 0j, 0.0, T, n)
        errs.append(abs(xi - gt) / abs(gt))
    print(f"  D5 integrated FM vs RK4: rel errs {errs[0]:.2e} {errs[1]:.2e} "
          f"{errs[2]:.2e} (ratios {errs[0]/errs[1]:.1f}, "
          f"{errs[1]/errs[2]:.1f}; expect ~16 = h^4)")
    assert 10 < errs[0] / errs[1] < 24 and 10 < errs[1] / errs[2] < 24, errs

    # the PM shortcut (index envelope applied OUTSIDE the integral) is a
    # different function — FM != PM exactly when the index moves
    ph_pm = w_c * T + b0 * np.exp(-sigma_b * T) * np.sin(w_m * T)
    gt = rk4(lambda t: 1j * omega(t), 1.0 + 0j, 0.0, T, 16000)
    pm_err = abs(np.exp(1j * ph_pm) - gt) / abs(gt)
    print(f"  D5 PM shortcut vs ODE:   rel err {pm_err:.2e} (plateaus — "
          f"dynamic-index PM is not FM)")
    assert pm_err > 1e-2
    print("D5 PASS: dynamic-index FM is exact through the integrated "
          "modulator — no accumulator, no PM compromise")


# ---------------------------------------------------------------------- audio

def main():
    test_jacobi_anger()
    test_dynamic_fm_ode()

    # three struck bells, w_m/w_c = 1.4, second partial for body
    strikes = [(0.3, 220.0, 1.0), (2.8, 146.83, 0.9), (5.3, 293.66, 0.8)]
    taus = np.arange(int(8.5 * SR)) / SR

    def render(voice):
        y = np.zeros(len(taus), complex)
        for (t0, f0, vel) in strikes:
            y += voice(t0, f0, vel)
            y += 0.35 * voice(t0, f0 * 2.76, vel)   # inharmonic body partial
        return y.real

    ref = render(lambda t0, f0, vel:
                 fm_voice_A(taus, t0, f0, 0.8, vel, static_mod(t0, 0.0, f0)))
    write_wav(f"{OUT}/modal_fm_reference.wav", ref)

    stat = render(lambda t0, f0, vel:
                  fm_voice_A(taus, t0, f0, 0.8, vel,
                             static_mod(t0, 3.0, 1.4 * f0)))
    write_wav(f"{OUT}/modal_fm_static.wav", stat)

    # the render-grid identity: the static piece re-rendered as Bessel banks
    bank_render = np.zeros(len(taus))
    for (t0, f0, vel) in strikes:
        for (f, g) in ((f0, 1.0), (f0 * 2.76, 0.35)):
            bnk = fm_voice_B(t0, f, 0.8, vel * g, 3.0, 1.4 * f, 40)
            bank_render += eval_complex(bnk, taus).real
    rel = np.linalg.norm(bank_render - stat) / np.linalg.norm(stat)
    print(f"  render-grid identity (phase route vs Bessel banks): "
          f"rel L2 = {rel:.2e}")
    assert rel < 1e-9, rel

    dyn = render(lambda t0, f0, vel:
                 fm_voice_A(taus, t0, f0, 0.8, vel,
                            dynamic_mod(t0, 6.0, 1.1, 1.4 * f0)))
    write_wav(f"{OUT}/modal_fm_dynamic.wav", dyn)


if __name__ == "__main__":
    main()
