#!/usr/bin/env python3
"""WS-LP phase 3a probe: can the CROSSING representation serve the region UNION?

A region-CHANGING pair (|a+1| straddles |kappa| as rt60 sweeps) is dropped
today. The candidate fix is NO union machinery at all: emit the crossing lanes
for the whole interval — at serOnly configs dSwitch = ln(|kappa|/|a+1|)/g goes
negative, the per-sample select sits on the series lane permanently, and the
series-lane constant k1Ser = GammaStar - CF(kappa)/g equals serOnly's
M(1,a+1,kappa)/(nu-mu) EXACTLY (the bridge identity, demos/bridge_verify.py).
The open question is float64 ACCURACY of the crossing representation on the
serOnly side of the boundary, where it has never been exercised.

This probe sweeps sigma_nu over the full clamped rt60 interval for straddling
pairs and compares BOTH f64 representations (replicating the emitted
algorithms: the fixed-depth Kummer Horner, the fixed-depth bottom-up CF, the
Lanczos+reflection lgamma of lgammaB/lgammaE) against a 60-digit mpmath
reference. Encouraging prior bounds, checked here too:
  - a straddling pair has |Im a| < |kappa| <= ~183, far from the
    e^{-pi |Im a|/2} Gamma underflow cliff (~465);
  - non-coincidence (|Im a| >= 1/2) keeps the reflection's log sin(pi a)
    bounded away from the poles (|sin| >= sinh(pi/2) ~ 2.3).
"""

import cmath
import math

import mpmath as mp

mp.mp.dps = 60

G = 1.8
B = 0.05 / 1.8
SIG_LO = 6.91 / 12.0
SIG_HI = 6.91 / 0.2
TP = 2.0 * math.pi

LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
           771.32342877765313, -176.61502916214059, 12.507343278686905,
           -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]


def lgamma_f64(z):
    """The shipped Lanczos(g=7,n=9) + dominant-half reflection (lgammaB/lgammaE)."""
    def core(z):
        zz = z - 1.0
        x = LANCZOS[0]
        for i in range(8):
            x += LANCZOS[i + 1] / (zz + (i + 1))
        t = zz + 7.5
        return (zz + 0.5) * cmath.log(t) - t + cmath.log(x) \
            + 0.5 * math.log(2.0 * math.pi)
    if z.real >= 0.5:
        return core(z)
    if z.imag < 0:
        s = complex(-math.pi * z.imag, math.pi * z.real)
        log2i = complex(math.log(2.0), math.pi / 2.0)
    else:
        s = complex(math.pi * z.imag, -math.pi * z.real)
        log2i = complex(math.log(2.0), -math.pi / 2.0)
    logsin = s + cmath.log(1.0 - cmath.exp(-2.0 * s)) - log2i
    return math.log(math.pi) - logsin - core(1.0 - z)


def m1_forward_depth(a, z, tol=1e-17, cap=4000):
    """bloomM1: forward recurrence, returns (value, depth) — the depth sizer."""
    s = 1.0 + 0j
    t = 1.0 + 0j
    for n in range(1, cap):
        t = t * z / (a + n)
        s += t
        if abs(t) <= tol * max(abs(s), 1.0):
            return s, n
    return s, cap


def m1_horner(a, z, n_depth):
    """The EMITTED fixed-depth Kummer Horner (bloomM1E)."""
    h = 1.0 + 0j
    for k in range(n_depth - 1, -1, -1):
        h = 1.0 + z / (a + k + 1) * h
    return h


def cf_bottomup(a, z, k_depth):
    """The EMITTED fixed-depth bottom-up CF (bloomCFE): CF = Gamma(a,z) e^z z^-a."""
    h = z + (2 * k_depth + 1) - a
    for j in range(k_depth - 1, -1, -1):
        h = (z + (2 * j + 1) - a) - (j + 1) * ((j + 1) - a) / h
    return 1.0 / h


def cf_depth(a, z):
    """Smallest depth where the bottom-up CF is converged to 1e-15 (sizer proxy)."""
    prev = cf_bottomup(a, z, 8)
    k = 16
    while k < 2048:
        cur = cf_bottomup(a, z, k)
        if abs(cur - prev) <= 1e-15 * max(abs(cur), 1e-300):
            return k
        prev = cur
        k += 16
    return 2048


def k1_ref(a, nu_mu, kappa):
    """60-digit reference: M(1, a+1, kappa)/(nu - mu)."""
    am = mp.mpc(a.real, a.imag)
    km = mp.mpc(kappa.real, kappa.imag)
    m = mp.hyp1f1(1, am + 1, km)
    r = m / mp.mpc(nu_mu.real, nu_mu.imag)
    return r


def rel(x, ref):
    d = abs(mp.mpc(x.real, x.imag) - ref)
    return float(d / max(abs(ref), mp.mpf("1e-300")))


def sweep(f_v, sig_v, df, n_pts=60):
    mu = complex(-sig_v, TP * f_v)
    kappa = mu * B
    om_nu = TP * (f_v + df)
    im_a = (om_nu - mu.imag) / G

    # classifier-style depth sizing at the interval samples (endpoints + -1 approach)
    re_lo = (-SIG_HI - mu.real) / G
    re_hi = (-SIG_LO - mu.real) / G
    t_star = max(re_lo, min(re_hi, -1.0))
    n_ship = 0
    k_ship = 0
    for re_a in (re_lo, re_hi, t_star):
        a = complex(re_a, im_a)
        z_bnd = kappa * (abs(a + 1) / abs(kappa))
        z_ser = kappa  # union depth must cover the serOnly side too
        n_ship = max(n_ship, m1_forward_depth(a, z_ser)[1])
        k_ship = max(k_ship, cf_depth(a, z_bnd), cf_depth(a, kappa))
    n_ship += 8
    k_ship += 8

    worst_ser = worst_cross = 0.0
    worst_ser_at = worst_cross_at = None
    n_ser_side = 0
    for i in range(n_pts):
        sig_nu = SIG_LO * (SIG_HI / SIG_LO) ** (i / (n_pts - 1))
        nu = complex(-sig_nu, om_nu)
        a = (nu - mu) / G
        ap1 = a + 1
        ser_side = abs(ap1) >= abs(kappa)
        n_ser_side += ser_side

        ref = k1_ref(a, nu - mu, kappa)
        # serOnly representation: emitted Horner / (nu - mu)
        r_ser = m1_horner(a, kappa, n_ship) / (nu - mu)
        # crossing representation: GammaStar - CF(kappa)/g
        gstar = cmath.exp(lgamma_f64(a) - a * cmath.log(kappa) + kappa) / G
        r_cross = gstar - cf_bottomup(a, kappa, k_ship) / G

        e_ser = rel(r_ser, ref)
        e_cross = rel(r_cross, ref)
        if e_ser > worst_ser:
            worst_ser, worst_ser_at = e_ser, (sig_nu, ser_side)
        if e_cross > worst_cross:
            worst_cross, worst_cross_at = e_cross, (sig_nu, ser_side)

    print(f"  voice f={f_v} sigma={sig_v}  df={df:+.2f} Hz  |kappa|={abs(kappa):.1f}  "
          f"|Im a|={abs(im_a):.1f}  depths ser/cf={n_ship}/{k_ship}  "
          f"serOnly-side {n_ser_side}/{n_pts}")
    print(f"    serOnly rep  worst rel {worst_ser:.3e} at sigma_nu={worst_ser_at[0]:.3f} "
          f"({'ser' if worst_ser_at[1] else 'cross'} side)")
    print(f"    crossing rep worst rel {worst_cross:.3e} at sigma_nu={worst_cross_at[0]:.3f} "
          f"({'ser' if worst_cross_at[1] else 'cross'} side)")
    return worst_cross


def main():
    print("WS-LP phase 3a: crossing-representation accuracy over the WHOLE interval")
    print(f"sigma_nu in [{SIG_LO:.3f}, {SIG_HI:.2f}]  g={G}  B={B:.5f}\n")
    worst = 0.0
    # straddling pairs (|a+1| crosses |kappa| inside the interval)
    worst = max(worst, sweep(220.0, 1.0, 9.8))
    worst = max(worst, sweep(220.0, 1.0, 10.4))
    worst = max(worst, sweep(220.0, 1.0, 10.9))
    worst = max(worst, sweep(1023.0, 1.13, 51.0))
    # control: a comfortably-crossing pair (phase 2 territory) and a deep-serOnly one
    print("\ncontrols:")
    worst = max(worst, sweep(1023.0, 1.13, 43.0))
    sweep(220.0, 1.0, 60.0)
    print(f"\nVERDICT: worst crossing-rep rel err over straddling pairs = {worst:.3e}")
    print("union-collapse viable" if worst < 1e-9 else "union-collapse NEEDS THOUGHT")


if __name__ == "__main__":
    main()
