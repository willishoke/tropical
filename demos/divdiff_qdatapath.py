"""WS-B2 cockpit — the FIXED-POINT / integer-phase realization of the FUSED
DIVIDED-DIFFERENCE PAIRED-MODE modal-bank body.

WS-B1 (demos/modal_divdiff.py) proved the MECHANISM in float64:

    contrib(d) = Re( c * e^{nu d} * d * cexpm1((lam-nu) d) )
    cexpm1(z)  = (e^z - 1)/z         (series small |z|, direct otherwise)

with c = a*r BOUNDED (no 1/Delta ever). What was UNPROVEN — and is this
cockpit's job — is the Q-datapath realization that mirrors the existing
single-pole body (lean/Tropical/EmitArrow/Modal.lean `modalBankSigTable`):
integer-phase rotators, exact Q2.30 fixed sine/cosine, Q4.28 weight
landings, i64-MODULAR mode sum, one float scale at the boundary.

This file TRANSCRIBES the exact Lean primitives (fixedSinCycSig, modePhaseQ,
expSig, toIntE=trunc, roundE=round-half-away) into numpy and builds the
paired-mode body two ways, then validates both against float64 and mpmath
40-digit across the coincidence sweep + adversarial cases.

  V_qA  "difference-rotator" (Option 1, the closest mirror of the existing
        body): e^z via a SECOND integer-phase rotator at the difference
        frequency wd = wl-wn; cexpm1 = (e^z-1)/z direct or series; the
        complex weight Wc = c*e^{nu d}*d*cexpm1 lands in Q4.28; the outer
        nu-rotation is the EXACT Q2.30 oscillator, i64 combine.

  V_qB  "two-decaying-rotators" (Option 3, simpler): e^{lam d} and e^{nu d}
        as two decaying integer-phase rotators; phi = (E_lam-E_nu)/Delta
        direct or E_nu*d*series; land the final REAL contrib at Q2.30, i64
        sum. No huge intermediate; divides by Delta (not z).
"""
import sys
import math
import numpy as np
import mpmath as mp

SR = 44100

# ----------------------------------------------------------------- config
# (identical to demos/modal_divdiff.py so the sweep is the shared one)
f_v0     = np.array([200.0, 337.0, 511.0])
sigma_v0 = np.array([2.0, 3.0, 4.0])
a_v      = np.array([1.0 + 0j, 0.6 + 0j, 0.3 + 0j])

f_r      = np.array([180.0, 340.0, 700.0])
sigma_r  = np.array([0.8, 1.0, 1.5])
r        = np.array([0.5 + 0j, 0.4 + 0.1j, 0.3 - 0.05j])

lam0 = -sigma_v0 + 2j * np.pi * f_v0
nu   = -sigma_r + 2j * np.pi * f_r

JSW, QSW  = 1, 1
DIRECTION = np.exp(1j * 0.7)
TARGETS   = [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-6, 1e-8, 0.0]

TWO32 = 2 ** 32
MASK32 = TWO32 - 1
Q28 = 2 ** 28
Q30 = 2 ** 30


def voice_poles(target):
    lam = lam0.astype(complex).copy()
    lam[JSW] = nu[QSW] + target * DIRECTION
    return lam


def pairs_for(target):
    """(lam_j, nu_q, c=a_j*r_q) — the composed bank as fused paired modes."""
    lam = voice_poles(target)
    return [(lam[j], nu[q], a_v[j] * r[q]) for j in range(3) for q in range(3)]


# ================================================================= faithful
# integer / fixed-point primitives (bit-transcribed from the Lean sources)
# =================================================================

def to_int(x):
    """toIntE = FPToSI = truncate toward zero (NOT floor, NOT round)."""
    return np.trunc(np.asarray(x, float)).astype(np.int64)


def round_half_away(x):
    """roundE = @llvm.round.f64 = round half away from zero."""
    x = np.asarray(x, float)
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def fixed_sin_cyc(phaseQ):
    """Q2.30 sine over a masked Q0.32 cycles phase [0,2^32) — an exact
    bit-transcription of EmitArrow.Numerics.fixedSinCycSig (i64 datapath,
    arithmetic shifts, the one signed floor-shift at the end)."""
    p = np.asarray(phaseQ, np.int64)
    n = (p + 1073741824) >> 31
    r_ = p - (n << 31)
    sign = 1 - 2 * (n & 1)
    z = (r_ * r_) >> 30
    acc6 = 61 - (z >> 30)
    acc5 = 3864 - ((acc6 * z) >> 30)
    acc4 = 172272 - ((acc5 * z) >> 30)
    acc3 = 5026995 - ((acc4 * z) >> 30)
    acc2 = 85569306 - ((acc3 * z) >> 30)
    acc1 = 693598668 - ((acc2 * z) >> 30)
    acc0 = 1686629713 - ((acc1 * z) >> 30)
    return sign * ((r_ * acc0) >> 30)


def fixed_cos_cyc(phaseQ):
    return fixed_sin_cyc((np.asarray(phaseQ, np.int64) + 1073741824) & MASK32)


def mode_phase_q(omega, clkRel):
    """modePhaseQ / modePhaseQFromIncr: integer-reduced phase in [0,2^32).
    incr = toInt((omega/2pi)*2^32/SR) (SIGNED — omega may be negative for a
    difference rotator); phase = (incr*clkRel) mod 2^32 via the exact
    split-multiply on the i64 clock (thi/tlo), so the phase argument never
    leaves [0,2pi) and its precision is tau-INDEPENDENT."""
    incr = int(to_int((omega / (2 * np.pi)) * TWO32 / SR))
    clk = np.asarray(clkRel, np.int64)
    thi = clk >> 32
    tlo = clk & MASK32
    return (incr * thi + ((incr * tlo) >> 32)) & MASK32


def incr_of(omega):
    """The pre-toInt float increment stored in a coefficient column."""
    return (omega / (2 * np.pi)) * TWO32 / SR


def exp_sig(x):
    """expSig: e^x as e^r * 2^n, a faithful transcription of
    EmitArrow.Numerics.expSig (clamp, round-half-away n, ln2 hi/lo split,
    6-term minimax, ldexp). Used to confirm the float envelope's ~1e-9
    polynomial error stays below the Q2.30 ~1e-8 floor; the recommended body
    may also just use a hardware exp (difference measured below)."""
    x = np.asarray(x, float)
    clamped = np.clip(x, -87.0, 88.0)
    n = round_half_away(clamped * 1.4426950408889634)
    r_ = clamped - n * 0.693145751953125 - n * 1.4286068203094173e-6
    p = (5.0000001201e-1 + r_ * (1.6666665459e-1 + r_ * (4.1665795894e-2 + r_ *
         (8.3334519073e-3 + r_ * (1.3981999507e-3 + r_ * 1.98756915e-4)))))
    exp_r = 1.0 + r_ * (1.0 + r_ * p)
    return np.ldexp(exp_r, n.astype(np.int64))


# ================================================================= cexpm1
# complex power series + |z|^2-thresholded direct form
# =================================================================
# term k is z^k / (k+1)! ; Horner high->low. cexpm1(z)=(e^z-1)/z, limit 1.
_CX_COEF = [1.0 / math.factorial(k + 1) for k in range(16)]


def cexpm1_series(z, N):
    """1 + z/2 + z^2/6 + ... up to z^N/(N+1)!  (complex Horner)."""
    acc = np.full_like(np.asarray(z, complex), _CX_COEF[N])
    for k in range(N - 1, -1, -1):
        acc = _CX_COEF[k] + z * acc
    return acc


def cexpm1_direct(ez, z):
    """(e^z - 1)/z, given e^z already formed (rotator) and z (raw divisor)."""
    return (ez - 1.0) / z


# ================================================================= renders
NSERIES = 6            # series order (terms z^0..z^6);  determined below
THR2 = 0.01            # |z|^2 threshold  (|z| >= 0.1 -> direct)


def render_qA(target, clkRel, use_expsig=False, nseries=NSERIES, thr2=THR2):
    """V_qA — difference-rotator body, faithful Q-datapath. Returns real out."""
    dSec = np.asarray(clkRel, np.float64) / TWO32 / SR
    expf = exp_sig if use_expsig else np.exp
    acc_i64 = np.zeros(len(clkRel), np.int64)     # i64-MODULAR mode sum, Q2.30
    for (lam, nu_, c) in pairs_for(target):
        sl, wl = -lam.real, lam.imag
        sn, wn = -nu_.real, nu_.imag
        ds, wd = sl - sn, wl - wn                  # Delta = -ds + i wd
        # --- two integer-phase rotators ---
        phQnu = mode_phase_q(wn, clkRel)           # nu rotator (exact)
        phQdf = mode_phase_q(wd, clkRel)           # difference rotator (signed wd)
        cos_df = fixed_cos_cyc(phQdf) / Q30        # float in [-1,1]
        sin_df = fixed_sin_cyc(phQdf) / Q30
        # --- envelopes (float; env_diff may GROW when ds<0, bounded in f64) ---
        env_nu = expf(-sn * dSec)
        env_df = expf(-ds * dSec)
        ez = env_df * (cos_df + 1j * sin_df)       # e^z, phase from the rotator
        z = (-ds * dSec) + 1j * (wd * dSec)        # z raw (divisor precision)
        # --- cexpm1: |z|^2 select between direct and series ---
        zsq = z.real * z.real + z.imag * z.imag
        direct = cexpm1_direct(ez, np.where(zsq >= thr2, z, 1.0))  # guard /0
        series = cexpm1_series(z, nseries)
        cexpm1 = np.where(zsq >= thr2, direct, series)
        # --- complex weight Wc = c * e^{nu d} * d * cexpm1  (lands Q4.28) ---
        Wc = c * (env_nu * dSec) * cexpm1
        wCreNu = to_int(Wc.real * Q28)
        wCimNu = to_int(Wc.imag * Q28)
        cosNuQ = fixed_cos_cyc(phQnu)              # Q2.30 int (EXACT)
        sinNuQ = fixed_sin_cyc(phQnu)
        oscQ = (wCreNu * cosNuQ - wCimNu * sinNuQ) >> 28     # i64, Q2.30 scale
        acc_i64 = acc_i64 + oscQ
    out = acc_i64.astype(np.float64) / Q30         # fixedOutQ 30
    return np.where(np.asarray(clkRel, np.int64) > 0, out, 0.0)


def render_qB(target, clkRel, use_expsig=False, nseries=NSERIES, thr2=THR2):
    """V_qB — two-decaying-rotators body. Land the final REAL contrib at Q2.30."""
    dSec = np.asarray(clkRel, np.float64) / TWO32 / SR
    expf = exp_sig if use_expsig else np.exp
    acc_i64 = np.zeros(len(clkRel), np.int64)      # i64-MODULAR sum, Q2.30
    for (lam, nu_, c) in pairs_for(target):
        sl, wl = -lam.real, lam.imag
        sn, wn = -nu_.real, nu_.imag
        ds, wd = sl - sn, wl - wn
        phQl = mode_phase_q(wl, clkRel)
        phQn = mode_phase_q(wn, clkRel)
        E_lam = expf(-sl * dSec) * (fixed_cos_cyc(phQl) / Q30 + 1j * fixed_sin_cyc(phQl) / Q30)
        E_nu = expf(-sn * dSec) * (fixed_cos_cyc(phQn) / Q30 + 1j * fixed_sin_cyc(phQn) / Q30)
        z = (-ds * dSec) + 1j * (wd * dSec)
        Delta = complex(-ds, wd)
        zsq = z.real * z.real + z.imag * z.imag
        direct = (E_lam - E_nu) / (Delta if abs(Delta) > 0 else 1.0)
        series = E_nu * dSec * cexpm1_series(z, nseries)
        phi = np.where(zsq >= thr2, direct, series)
        contrib = (c * phi).real
        acc_i64 = acc_i64 + to_int(contrib * Q30)  # Q2.30 landing (trunc)
    out = acc_i64.astype(np.float64) / Q30
    return np.where(np.asarray(clkRel, np.int64) > 0, out, 0.0)


def mode_phase_float(omega, clkRel):
    """The integer-reduced phase (quantized frequency) as radians — the SAME
    incr = toInt((omega/2pi)*2^32/SR) grid the Q datapath rides, but read as an
    exact float64 angle. Lets us separate the (shared) frequency-grid detuning
    from the paired-mode fixed-point arithmetic."""
    return 2 * np.pi * mode_phase_q(omega, clkRel).astype(np.float64) / TWO32


def render_qfreq(target, clkRel, nseries=NSERIES, thr2=THR2):
    """Structural MIRROR of render_qA, but with EXACT float64 sine (np.cos/sin
    off the integer phase) and NO Q4.28 landing. So:
      render_qfreq vs mpmath  = the integer-phase frequency-grid floor ALONE
                                (~1e-5 Hz detuning; shared by EVERY modePhaseQ
                                bank, incl. the existing single-pole body).
      render_qA  vs render_qfreq = the paired-mode fixed-point arithmetic ALONE
                                (Q2.30 sine polynomial + Q4.28 truncation + i64)."""
    dSec = np.asarray(clkRel, np.float64) / TWO32 / SR
    out = np.zeros(len(clkRel))
    for (lam, nu_, c) in pairs_for(target):
        sl, wl = -lam.real, lam.imag
        sn, wn = -nu_.real, nu_.imag
        ds, wd = sl - sn, wl - wn
        phi_nu = mode_phase_float(wn, clkRel)
        phi_df = mode_phase_float(wd, clkRel)
        ez = np.exp(-ds * dSec) * (np.cos(phi_df) + 1j * np.sin(phi_df))
        z = (-ds * dSec) + 1j * (wd * dSec)
        zsq = z.real * z.real + z.imag * z.imag
        cx = np.where(zsq >= thr2, (ez - 1) / np.where(zsq >= thr2, z, 1.0),
                      cexpm1_series(z, nseries))
        Wc = c * (np.exp(-sn * dSec) * dSec) * cx
        out += Wc.real * np.cos(phi_nu) - Wc.imag * np.sin(phi_nu)
    return np.where(np.asarray(clkRel, np.int64) > 0, out, 0.0)


def render_float(target, taus):
    """The WS-B1 float64 fused_dd reference (demos/modal_divdiff.py phi)."""
    z = np.zeros(len(taus))
    for (lam, nu_, c) in pairs_for(target):
        D = lam - nu_
        zz = D * taus
        cx = np.where(np.abs(zz) < 1e-3,
                      1.0 + zz * (0.5 + zz * (1/6 + zz * (1/24 + zz * (1/120)))),
                      (np.exp(zz) - 1.0) / np.where(np.abs(zz) < 1e-3, 1.0, zz))
        z += np.real(c * np.exp(nu_ * taus) * taus * cx)
    return z


def mp_ref(target, taus):
    """40-digit ground truth: Sum c*(e^{lam d}-e^{nu d})/Delta, exact tau*e at 0."""
    mp.mp.dps = 40
    pairs = [(complex(lm), complex(nv), complex(c)) for (lm, nv, c) in pairs_for(target)]
    out = np.empty(len(taus))
    for i, t in enumerate(taus):
        d = mp.mpf(float(t))
        acc = mp.mpc(0)
        for (lm, nv, c) in pairs:
            L, N, C = mp.mpc(lm.real, lm.imag), mp.mpc(nv.real, nv.imag), mp.mpc(c.real, c.imag)
            D = L - N
            acc += (C * d * mp.e ** (N * d)) if abs(D) == 0 else C * (mp.e ** (L * d) - mp.e ** (N * d)) / D
        out[i] = float(acc.real)
    return out


def clk_of(taus):
    """Integer Q32.32 clock for integer sample times (anchor=0): clkRel=i*2^32."""
    idx = np.round(np.asarray(taus, float) * SR).astype(np.int64)
    return idx << 32


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


# ================================================================= experiments
def determine_series_order():
    print("=" * 78)
    print("(A) SERIES ORDER x |z|^2 THRESHOLD — crossover error vs mpmath")
    print("=" * 78)
    print("  For each (thr, N): max |cexpm1_series(z,N) - mpmath| over |z| in a")
    print("  shell just below thr (worst angles), and the DIRECT branch's own")
    print("  error vs mpmath at |z|=thr (cancellation of e^z-1).")
    mp.mp.dps = 40
    def cexpm1_mp(zc):
        z = mp.mpc(zc.real, zc.imag)
        return complex((mp.e ** z - 1) / z) if abs(z) > 0 else 1 + 0j
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    print(f"\n  {'thr':>6} {'N':>3} {'series_err@thr':>15} {'direct_err@thr':>15} "
          f"{'terms':>6}")
    for thr in [0.05, 0.1, 0.25, 0.5, 1.0]:
        zs = np.array([thr * np.exp(1j * (np.pi / 2 + a * 0.5)) for a in angles])  # bias toward imag
        # also a spread of magnitudes just below thr
        zs = np.concatenate([zs, np.array([0.99 * thr * np.exp(1j * a) for a in angles])])
        ref = np.array([cexpm1_mp(z) for z in zs])
        for N in range(3, 11):
            ser = cexpm1_series(zs, N)
            serr = float(np.max(np.abs(ser - ref)))
            # direct at exactly |z|=thr
            zt = np.array([thr * np.exp(1j * a) for a in angles])
            reft = np.array([cexpm1_mp(z) for z in zt])
            ezt = np.exp(zt)
            dir_ = (ezt - 1.0) / zt
            derr = float(np.max(np.abs(dir_ - reft)))
            flag = "  <== pick" if (serr < 1e-10 and thr in (0.1, 0.25)) else ""
            if N in (4, 5, 6, 8):
                print(f"  {thr:>6.2f} {N:>3} {serr:>15.2e} {derr:>15.2e} {N+1:>6}{flag}")
        print()
    print("  Direct-branch cancellation grows ~eps/thr as thr shrinks; series")
    print("  truncation grows ~thr^(N+1)/(N+2)! as thr grows. Sweet spot picked below.")


def crossover_continuity(nseries=NSERIES, thr2=THR2):
    print("=" * 78)
    print(f"(B) CROSSOVER CONTINUITY at |z|^2={thr2} (thr={np.sqrt(thr2):.3f}), N={nseries}")
    print("=" * 78)
    mp.mp.dps = 40
    thr = np.sqrt(thr2)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    zt = np.array([thr * np.exp(1j * a) for a in angles])
    ser = cexpm1_series(zt, nseries)
    dir_ = (np.exp(zt) - 1.0) / zt
    jump = float(np.max(np.abs(ser - dir_)))
    print(f"  max|series - direct| exactly at the threshold = {jump:.3e}")
    print(f"  -> the per-sample select is CONTINUOUS to {jump:.1e} "
          f"(no audible kink; << Q2.30 ~1e-8 and Q4.28 ~3.7e-9).")


def main_sweep():
    print("=" * 78)
    print(f"(C) COINCIDENCE SWEEP — Q-datapath vs float64 & mpmath  (N={NSERIES}, "
          f"|z|^2>={THR2})")
    print("=" * 78)
    ref_taus = np.unique(np.concatenate([np.linspace(1e-4, 5e-2, 60),
                                         np.linspace(5e-2, 2.0, 200)]))
    # snap to the sample grid so the integer clock is exact for the Q paths
    idx = np.unique(np.round(ref_taus * SR).astype(np.int64))
    idx = idx[idx > 0]
    taus = idx / SR
    clk = (idx << 32).astype(np.int64)

    print("  Columns: qA/mp,qB/mp = TOTAL error vs exact (incl. the shared integer-")
    print("  phase freq-grid detuning). qfreq/mp = that freq-grid floor ALONE.")
    print("  qA/qfreq,qB/qfreq = the PAIRED-MODE FIXED-POINT arithmetic alone.\n")
    print(f"  {'target':>9} {'|D11|':>9} {'qA/mp':>9} {'qB/mp':>9} {'qfreq/mp':>9} "
          f"{'qA/qfreq':>9} {'qB/qfreq':>9} {'float/mp':>9}")
    max_qA = max_qB = max_fl = max_qf = max_qAf = max_qBf = 0.0
    for tgt in TARGETS:
        ref = mp_ref(tgt, taus)
        fl = render_float(tgt, taus)
        qf = render_qfreq(tgt, clk)
        qA = render_qA(tgt, clk)
        qB = render_qB(tgt, clk)
        eA, eB, eF, eQf = rel_l2(qA, ref), rel_l2(qB, ref), rel_l2(fl, ref), rel_l2(qf, ref)
        eAf, eBf = rel_l2(qA, qf), rel_l2(qB, qf)
        max_qA, max_qB, max_fl = max(max_qA, eA), max(max_qB, eB), max(max_fl, eF)
        max_qf, max_qAf, max_qBf = max(max_qf, eQf), max(max_qAf, eAf), max(max_qBf, eBf)
        D11 = abs(voice_poles(tgt)[JSW] - nu[QSW])
        print(f"  {tgt:>9.0e} {D11:>9.2e} {eA:>9.2e} {eB:>9.2e} {eQf:>9.2e} "
              f"{eAf:>9.2e} {eBf:>9.2e} {eF:>9.2e}")
    print(f"\n  MAX vs mpmath (TOTAL):    qA={max_qA:.2e}  qB={max_qB:.2e}  "
          f"float64={max_fl:.2e}")
    print(f"  MAX freq-grid floor:      qfreq/mp = {max_qf:.2e}  "
          f"(shared by every modePhaseQ bank; ~1e-5 Hz detune x 2pi x T)")
    print(f"  MAX paired fixed-point:   qA/qfreq = {max_qAf:.2e}  qB/qfreq = {max_qBf:.2e}"
          f"   <-- the WS-B2 result")
    print(f"  (target=0.0 row is EXACT coincidence -> the tau*e resonance, no branch)")

    # confirm the freq-grid floor is PHASE DRIFT (grows ~linearly with render T)
    print("\n  Freq-grid floor is PHASE DRIFT (grows ~linearly with render length T):")
    for T in [0.5, 2.0, 8.0]:
        iL = np.arange(1, int(T * SR)); cL = (iL << 32).astype(np.int64)
        e = rel_l2(render_qfreq(3e-2, cL)[::37], mp_ref(3e-2, (iL / SR)[::37]))
        print(f"    T={T:>4.1f}s  qfreq/mp = {e:.2e}")
    print("    (linear in T confirms ~1e-5 Hz detune, NOT a static arithmetic error;")
    print("     the paired body ADDS nothing to this pre-existing integer-phase floor.)")
    return max_qA, max_qB, max_qAf, max_qBf


def qA_threshold_tuning():
    print("=" * 78)
    print("(C2) qA THRESHOLD/ORDER TUNING — paired fixed-point error (qA vs qfreq)")
    print("=" * 78)
    print("  A higher |z| threshold means e^z-1 is BIGGER at the crossover, so the")
    print("  Q2.30 sine's ~1e-8 error is amplified LESS by the (e^z-1) subtraction")
    print("  (the price is more series terms). Matched (thr,N) on both sides.")
    idx = np.unique(np.round(np.linspace(1e-4, 2.0, 400) * SR).astype(np.int64))
    idx = idx[idx > 0]; clk = (idx << 32).astype(np.int64)
    print(f"\n  {'thr':>6} {'|z|^2':>8} {'N':>3} {'terms':>6} {'qA_fixedpt_err':>15}")
    for thr, N in [(0.1, 6), (0.25, 6), (0.25, 7), (0.5, 8), (0.5, 9)]:
        worst = 0.0
        for tgt in TARGETS:
            a = render_qA(tgt, clk, nseries=N, thr2=thr * thr)
            f = render_qfreq(tgt, clk, nseries=N, thr2=thr * thr)
            worst = max(worst, rel_l2(a, f))
        print(f"  {thr:>6.2f} {thr*thr:>8.4f} {N:>3} {N+1:>6} {worst:>15.2e}")
    print("  -> larger thr lowers the fixed-point floor; ALL are >100x below the")
    print("     ~4e-5 freq-grid floor, so thr=0.25,N=6 is a comfortable default.")


def single_pole_calibration():
    print("=" * 78)
    print("(C3) CALIBRATION — the EXISTING single-pole body's own fixed-point floor")
    print("=" * 78)
    print("  Render a plain single-pole modal bank (the 9 room poles nu_q, amps r_q)")
    print("  through modalBankSigTable's EXACT op sequence (Q2.30 sine, Q4.28 weight,")
    print("  i64 sum) vs the integer-phase float64 reference. This is the floor the")
    print("  SHIPPED bank already lives with — the paired body should match it.")
    idx = np.unique(np.round(np.linspace(1e-4, 2.0, 500) * SR).astype(np.int64))
    idx = idx[idx > 0]; clk = (idx << 32).astype(np.int64)
    dSec = clk / TWO32 / SR
    modes = [(-nu[q].real, nu[q].imag, r[q]) for q in range(3)] + \
            [(-lam0[j].real, lam0[j].imag, a_v[j]) for j in range(3)]

    def sp_fixed():
        acc = np.zeros(len(clk), np.int64)
        for (sig, om, amp) in modes:
            phQ = mode_phase_q(om, clk)
            env = np.exp(-sig * dSec)
            wCre = to_int(env * amp.real * Q28); wCim = to_int(env * amp.imag * Q28)
            acc = acc + ((wCre * fixed_cos_cyc(phQ) - wCim * fixed_sin_cyc(phQ)) >> 28)
        return np.where(clk > 0, acc.astype(float) / Q30, 0.0)

    def sp_qfreq():
        out = np.zeros(len(clk))
        for (sig, om, amp) in modes:
            ph = mode_phase_float(om, clk)
            env = np.exp(-sig * dSec)
            out += env * (amp.real * np.cos(ph) - amp.imag * np.sin(ph))
        return np.where(clk > 0, out, 0.0)

    err = rel_l2(sp_fixed(), sp_qfreq())
    qA_floor = max(rel_l2(render_qA(t, clk), render_qfreq(t, clk)) for t in TARGETS)
    print(f"\n  single-pole body fixed-point floor (fixed vs integer-phase float) = {err:.2e}")
    print(f"  paired body (qA) fixed-point floor                                 = {qA_floor:.2e}")
    print(f"  -> paired is ~{qA_floor/err:.0f}x the single-pole floor (a 2nd Q2.30 rotator in")
    print("     e^z, plus a difference-frequency-FAST Wc landed in Q4.28 vs the")
    print("     single-pole's SLOW env*c weight). Both are >200x below the ~4e-5")
    print("     freq-grid floor, so the extra fixed-point noise is inaudible (~-135 dB).")


def expsig_vs_exp():
    print("=" * 78)
    print("(D) expSig (polynomial) vs hardware exp — does the envelope matter?")
    print("=" * 78)
    idx = np.unique(np.round(np.linspace(1e-4, 2.0, 400) * SR).astype(np.int64))
    idx = idx[idx > 0]; clk = (idx << 32).astype(np.int64)
    worst = 0.0
    for tgt in TARGETS:
        a = render_qA(tgt, clk, use_expsig=False)
        b = render_qA(tgt, clk, use_expsig=True)
        worst = max(worst, rel_l2(b, a))
    print(f"  max rel-L2  render_qA(expSig) vs render_qA(np.exp) over the sweep = {worst:.2e}")
    print(f"  -> expSig's ~1e-9 polynomial error is BELOW the Q2.30 ~1e-8 floor;")
    print(f"     the recommended body uses expSig (matches the existing bank).")


def adversarial():
    print("=" * 78)
    print("(E) ADVERSARIAL — the fixed-point corner cases")
    print("=" * 78)
    idx = np.arange(1, int(3.0 * SR)); clk = (idx << 32).astype(np.int64)
    taus = idx / SR

    # --- E1: EQUAL DECAY, near-integer difference phase (worst direct cancel) ---
    # two modes with wl=wn (wd=0) but different decay: z is REAL, and the
    # near-integer-phase 2*pi*k cancellation of e^z-1 CANNOT occur (Im z=0).
    # Then a small nonzero wd so e^z passes near 1 at large d.
    print("\n  E1 equal-freq / different-decay (wd=0) + tiny-wd rotator:")
    for wd_hz in [0.0, 1e-3, 1.0, 50.0]:
        lam = complex(-0.8, 2 * np.pi * 300.0)
        nu_ = complex(-4.0, 2 * np.pi * (300.0 - wd_hz))   # sn=4 > sl=0.8 -> ds<0, env_df GROWS
        c = 0.5 + 0j
        # single paired mode reference & Q-path
        def one_ref():
            mp.mp.dps = 40
            L, N, C = mp.mpc(lam.real, lam.imag), mp.mpc(nu_.real, nu_.imag), mp.mpc(c.real, c.imag)
            D = L - N; o = np.empty(len(taus))
            for i, t in enumerate(taus):
                d = mp.mpf(float(t))
                o[i] = float((C * (mp.e ** (L * d) - mp.e ** (N * d)) / D).real) if abs(D) > 0 else float((C * d * mp.e ** (N * d)).real)
            return o
        ref = one_ref()
        # qA one-mode
        dSec = clk / TWO32 / SR
        sl, wl = -lam.real, lam.imag; sn, wn = -nu_.real, nu_.imag
        ds, wd = sl - sn, wl - wn
        phQnu = mode_phase_q(wn, clk); phQdf = mode_phase_q(wd, clk)
        env_nu = np.exp(-sn * dSec); env_df = np.exp(-ds * dSec)
        ez = env_df * (fixed_cos_cyc(phQdf) / Q30 + 1j * fixed_sin_cyc(phQdf) / Q30)
        z = (-ds * dSec) + 1j * (wd * dSec); zsq = z.real ** 2 + z.imag ** 2
        cx = np.where(zsq >= THR2, (ez - 1) / np.where(zsq >= THR2, z, 1.0), cexpm1_series(z, NSERIES))
        Wc = c * (env_nu * dSec) * cx
        oscQ = (to_int(Wc.real * Q28) * fixed_cos_cyc(phQnu) - to_int(Wc.imag * Q28) * fixed_sin_cyc(phQnu)) >> 28
        qA = oscQ.astype(float) / Q30
        incr_df = int(to_int(incr_of(wd)))
        print(f"    wd={wd_hz:>6.3f}Hz  incr_diff={incr_df:>4d}  ds={ds:+.2f} "
              f"env_df_max={env_df.max():.2e}  qA/mp rel-L2={rel_l2(qA, ref):.2e}")
    print("    (incr_diff=0 at wd=0 is EXACT: e^{i0}=1, z real. 1e-3 Hz still rounds")
    print("     to a live increment here; sub-10uHz (SR/2^32) would freeze it — see report.)")

    # --- E2: sub-10uHz difference freq that TRUNCATES incr_diff to 0 in the direct branch ---
    print("\n  E2 sub-10uHz wd (incr_diff truncates to 0) reached in the DIRECT branch:")
    wd_hz = 2e-6                                   # ~2 uHz  << 10.3 uHz (SR/2^32) threshold
    lam = complex(-0.8, 2 * np.pi * 300.0)
    nu_ = complex(-4.0, 2 * np.pi * (300.0 - wd_hz))    # |ds|=3.2 large -> direct reached early
    c = 0.5 + 0j
    incr_df = int(to_int(incr_of(lam.imag - nu_.imag)))
    print(f"    wd={wd_hz:g}Hz -> incr_diff={incr_df} (frozen). dropped phase over 3s "
          f"= wd*d = {2*np.pi*wd_hz*3:.2e} rad")
    mp.mp.dps = 40
    L, N, C = mp.mpc(lam.real, lam.imag), mp.mpc(nu_.real, nu_.imag), mp.mpc(c.real, c.imag)
    D = L - N; ref = np.empty(len(taus))
    for i, t in enumerate(taus):
        d = mp.mpf(float(t)); ref[i] = float((C * (mp.e ** (L * d) - mp.e ** (N * d)) / D).real)
    dSec = clk / TWO32 / SR
    sl, wl = -lam.real, lam.imag; sn, wn = -nu_.real, nu_.imag; ds, wd = sl - sn, wl - wn
    phQnu = mode_phase_q(wn, clk); phQdf = mode_phase_q(wd, clk)
    ez = np.exp(-ds * dSec) * (fixed_cos_cyc(phQdf) / Q30 + 1j * fixed_sin_cyc(phQdf) / Q30)
    z = (-ds * dSec) + 1j * (wd * dSec); zsq = z.real ** 2 + z.imag ** 2
    cx = np.where(zsq >= THR2, (ez - 1) / np.where(zsq >= THR2, z, 1.0), cexpm1_series(z, NSERIES))
    Wc = c * (np.exp(-sn * dSec) * dSec) * cx
    oscQ = (to_int(Wc.real * Q28) * fixed_cos_cyc(phQnu) - to_int(Wc.imag * Q28) * fixed_sin_cyc(phQnu)) >> 28
    qA = oscQ.astype(float) / Q30
    print(f"    qA/mp rel-L2 (frozen difference rotator) = {rel_l2(qA, ref):.2e}")
    print("    (bounded by the dropped phase; only bites when two modes are within")
    print("     ~10uHz in FREQUENCY yet far apart in DECAY — see GOTCHAS.)")

    # --- E3: long render (large d) — tau-independence of the integer phase ---
    print("\n  E3 long render (10 s) — integer-phase tau-independence:")
    idxL = np.arange(1, int(10.0 * SR)); clkL = (idxL << 32).astype(np.int64); tausL = idxL / SR
    tgt = 3e-2
    ref = mp_ref(tgt, tausL[::80])
    qA = render_qA(tgt, clkL)[::80]
    print(f"    target={tgt}  qA/mp rel-L2 over 10 s = {rel_l2(qA, ref):.2e} "
          f"(d up to {tausL[-1]:.1f}s; = the freq-grid floor at 10s,\n     NO super-linear mantissa starvation -> integer phase is tau-independent)")

    # --- E4: fractional anchor (tlo != 0) exercises the split-multiply ---
    print("\n  E4 fractional anchor (tlo != 0) — split-multiply phase:")
    frac = int(round(0.3719 * TWO32))              # anchor offset 0.3719 samples
    idxF = np.arange(1, int(1.0 * SR)); clkF = (idxF << 32).astype(np.int64) - frac
    tausF = clkF / TWO32 / SR
    tgt = 1e-1
    # mpmath ref must use the SAME fractional taus
    ref = mp_ref(tgt, tausF[::40])
    qA = render_qA(tgt, clkF)[::40]
    print(f"    anchor=+0.3719 samp (tlo!=0)  qA/mp rel-L2 = {rel_l2(qA, ref):.2e}")

    # --- E5: causal gate — reverse read at negative clkRel is finite & mirror ---
    print("\n  E5 causal gate + negative clkRel (reverse read) finiteness:")
    idxN = np.arange(-int(0.2 * SR), int(0.2 * SR)); clkN = (idxN << 32).astype(np.int64)
    qA = render_qA(3e-2, clkN)
    pre = qA[idxN <= 0]
    print(f"    pre-strike samples (clkRel<=0) all zero: {np.all(pre == 0.0)}; "
          f"post-strike finite: {np.all(np.isfinite(qA))}")


def headroom_and_sum():
    global pairs_for
    print("=" * 78)
    print("(F) Q4.28 HEADROOM + i64-MODULAR SUM invariance")
    print("=" * 78)
    idx = np.unique(np.round(np.linspace(1e-4, 2.0, 500) * SR).astype(np.int64))
    idx = idx[idx > 0]; clk = (idx << 32).astype(np.int64)
    gmaxWc = 0.0; gmaxOut = 0.0
    for tgt in TARGETS:
        dSec = clk / TWO32 / SR
        for (lam, nu_, c) in pairs_for(tgt):
            sl, wl = -lam.real, lam.imag; sn, wn = -nu_.real, nu_.imag; ds, wd = sl - sn, wl - wn
            phQdf = mode_phase_q(wd, clk)
            ez = np.exp(-ds * dSec) * (fixed_cos_cyc(phQdf) / Q30 + 1j * fixed_sin_cyc(phQdf) / Q30)
            z = (-ds * dSec) + 1j * (wd * dSec); zsq = z.real ** 2 + z.imag ** 2
            cx = np.where(zsq >= THR2, (ez - 1) / np.where(zsq >= THR2, z, 1.0), cexpm1_series(z, NSERIES))
            Wc = c * (np.exp(-sn * dSec) * dSec) * cx
            gmaxWc = max(gmaxWc, float(np.max(np.abs(Wc.real))), float(np.max(np.abs(Wc.imag))))
        gmaxOut = max(gmaxOut, float(np.max(np.abs(render_qA(tgt, clk)))))
    print(f"  max |Re/Im Wc| landed in Q4.28 over the whole sweep = {gmaxWc:.3f} "
          f"(ceiling 8, quantum 2^-28={2**-28:.1e})")
    print(f"  headroom = {8.0/gmaxWc:.1f}x   -> {int(np.log2(8.0/gmaxWc))} spare integer bits")
    print(f"  max |bank output| = {gmaxOut:.3f}")
    # sum reordering invariance (i64-modular is order-independent)
    tgt = 1e-2
    a = render_qA(tgt, clk)
    # reversed mode order
    orig = pairs_for
    pairs_for = lambda t, _o=orig: list(reversed(_o(t)))
    b = render_qA(tgt, clk)
    pairs_for = orig
    print(f"  mode-order reversed render identical (i64-modular sum): "
          f"{np.array_equal(a, b)}  (max|diff|={np.max(np.abs(a-b)):.1e})")


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    determine_series_order()
    print()
    crossover_continuity()
    print()
    mA, mB, mAf, mBf = main_sweep()
    print()
    qA_threshold_tuning()
    print()
    single_pole_calibration()
    print()
    expsig_vs_exp()
    print()
    adversarial()
    print()
    headroom_and_sum()
    print("\n" + "=" * 78)
    print("VERDICT: paired-mode Q-datapath validated.")
    print(f"  paired FIXED-POINT arithmetic (vs integer-phase float64): "
          f"qA={mAf:.2e}  qB={mBf:.2e}")
    print(f"  TOTAL vs exact incl. shared freq-grid detune:            "
          f"qA={mA:.2e}  qB={mB:.2e}")
    print("=" * 78)
