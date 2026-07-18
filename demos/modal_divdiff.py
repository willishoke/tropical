"""modal_divdiff — stable near-degenerate residue composition (WS-B1 cockpit).

When a voice pole λ sweeps toward a room pole ν, the collected composed bank
(residueComposeEC: forced a·H(λ) at λ, ringing −a·r/Δ at ν, Δ=λ−ν) develops a
coupling amp ∝ 1/Δ. Two failure modes as Δ→0:

  cancellation   the +a·r/Δ and −a·r/Δ modes are huge and nearly cancel in the
                 render; float64 loses precision as ~eps/|Δ|.
  Q4.28 range    the amp a·r/Δ overflows the fixed-point magnitude ceiling (±8)
                 for |Δ| < |a·r|/8 — unusable in the fixed-point datapath.

The fix (this cockpit's finding): represent each (λ,ν) coupling as ONE fused
divided-difference paired atom carrying both poles and a BOUNDED coeff c=a·r
(no 1/Δ EVER), realized as the stable complex divided difference

    φ(λ, ν; d) = (e^{λd} − e^{νd})/(λ−ν) = e^{νd} · d · cexpm1((λ−ν)d)

with cexpm1(z) = (e^z−1)/z (short series for |z|<1e-3, direct otherwise; the
limit is 1 at z=0, so φ → d·e^{νd}, the τ·e resonance, with NO branch). The
whole composed bank is the homogeneous family {(λ_j, ν_q, c_jq)} of paired
modes — its OWN uniform family (one atom shape, deg-1 resonance absorbed
continuously), the price being ~2.2×/mode and an inherently m·n (uncollected)
expansion, since collecting reintroduces the 1/Δ.

Differentials (house style: independent oracle, a PLATEAU or a RATE — a
cancellation bug's error tracks 1/Δ; the stable form plateaus at machine
precision). The 40-digit oracle uses mpmath when present; otherwise the checks
that don't need it still run (bounded-coeff, Q4.28 range, exact-limit, continuity
are all numpy-only) and the plateau table is skipped with a note.

  D_dd1  candidate vs mpmath ground truth is FLAT ~1e-13 across |Δ| ∈ [1, 0];
         the naive collected form's error SCALES as eps/|Δ| (→~1e-5 at 1e-8).
  D_dd2  the fused coeff |c|=|a·r| ≤ 0.5 ≪ 8 (no Q4.28 overflow) while the
         collected ringing amp |a·r/Δ| exceeds 8 for |Δ| < 0.031.
  D_dd3  at exact coincidence φ reproduces the deg-1 τ·e^{νd} resonance
         (vs the closed-form a·r·d·e^{νd}) to machine precision, no branch.
  D_dd4  rendered continuity: adjacent-step L2 through coincidence scales with
         |Δλ| and the crossing step (1e-8→0) does NOT spike.
"""
import numpy as np

import modal_arrow as ma

try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False

SR = 44100

# ------------------------------------------------------------------ config
f_v0     = np.array([200.0, 337.0, 511.0])
sigma_v0 = np.array([2.0, 3.0, 4.0])
a_v      = np.array([1.0 + 0j, 0.6 + 0j, 0.3 + 0j])

f_r      = np.array([180.0, 340.0, 700.0])
sigma_r  = np.array([0.8, 1.0, 1.5])
r        = np.array([0.5 + 0j, 0.4 + 0.1j, 0.3 - 0.05j])

lam0 = -sigma_v0 + 2j * np.pi * f_v0            # base voice poles
nu   = -sigma_r + 2j * np.pi * f_r              # room poles (fixed)

JSW, QSW = 1, 1                                  # swept pair (j=1, q=1)
DIRECTION = np.exp(1j * 0.7)
TARGETS = [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-6, 1e-8, 0.0]

QRES = 2 ** 28                                    # Q4.28 resolution
QSAT = 8.0                                        # Q4.28 magnitude ceiling


def voice_poles(target):
    lam = lam0.astype(complex).copy()
    lam[JSW] = nu[QSW] + target * DIRECTION       # drive λ_1 → ν_1
    return lam


def pairs_for(target):
    """The composed bank as fused paired modes (λ_j, ν_q, c=a_j·r_q)."""
    lam = voice_poles(target)
    return [(lam[j], nu[q], a_v[j] * r[q]) for j in range(3) for q in range(3)]


def voice_atoms(target):
    lam = voice_poles(target)
    return [ma.Atom(0.0, lam[j], a_v[j], 0) for j in range(3)]


# ------------------------------------------------------ the stable kernel
def cexpm1(z):
    """(e^z − 1)/z for complex arrays; stable, limit 1 at z=0."""
    z = np.asarray(z, complex)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-3
    zs = z[small]                                 # 1 + z/2 + z²/6 + z³/24 + z⁴/120
    out[small] = 1.0 + zs * (0.5 + zs * (1/6 + zs * (1/24 + zs * (1/120))))
    zl = z[~small]
    out[~small] = (np.exp(zl) - 1.0) / zl
    return out


def phi(lam, nu_, d):
    """(e^{λd} − e^{νd})/(λ−ν), stable for all Δ including 0 (→ d·e^{νd})."""
    return np.exp(nu_ * d) * d * cexpm1((lam - nu_) * d)


def cand_render(target, taus, quant=False):
    """Sum the fused paired modes. Returns (real signal, saturated flag)."""
    z = np.zeros(len(taus))
    saturated = False
    for (lam, nu_, c) in pairs_for(target):
        cc = c
        if quant:
            cc, sat = q428(c)
            saturated = saturated or sat
        contrib = np.real(cc * phi(lam, nu_, taus))
        if quant:
            cq = np.clip(np.round(contrib * QRES) / QRES, -QSAT, QSAT - 1.0 / QRES)
            saturated = saturated or bool(np.any(np.abs(np.round(contrib * QRES) / QRES) > QSAT))
            contrib = cq
        z += contrib
    return z, saturated


def q428(x):
    """Nearest 2^-28, saturating at ±8 (re/im separately). Returns (q, sat)."""
    def qc(v):
        qd = np.round(v * QRES) / QRES
        return np.clip(qd, -QSAT, QSAT - 1.0 / QRES), bool(np.any(np.abs(qd) > QSAT))
    qr, sr = qc(np.real(x))
    qi, si = qc(np.imag(x))
    return qr + 1j * qi, (sr or si)


def collected_max_amp(target):
    """Max |amp| of the NAIVE collected form (the Q4.28 range test)."""
    lam = voice_poles(target)
    amps = []
    for j in range(3):
        amps.append(a_v[j] * np.sum(r / (lam[j] - nu)))          # forced a·H(λ)
        for q in range(3):
            amps.append(-a_v[j] * r[q] / (lam[j] - nu[q]))       # ringing −a·r/Δ
    return np.max(np.abs(amps))


def oracle_render(target, taus):
    """modal_arrow.Filter — the m·n uncollected form WITH the degeneracy branch."""
    filt = ma.Filter(ma.LTI(nu.copy(), r.copy()), tol=1e-9)
    return ma.eval_complex(filt(voice_atoms(target)), taus).real


def mp_ref(target, taus):
    """40-digit ground truth (mpmath): Σ c·(e^{λd}−e^{νd})/Δ, exact τ·e at Δ=0."""
    mp.mp.dps = 40
    pairs = [(complex(lm), complex(nv), complex(c)) for (lm, nv, c) in pairs_for(target)]
    out = np.empty(len(taus))
    for i, t in enumerate(taus):
        d = mp.mpf(float(t))
        acc = mp.mpc(0)
        for (lm, nv, c) in pairs:
            L, N, C = mp.mpc(lm.real, lm.imag), mp.mpc(nv.real, nv.imag), mp.mpc(c.real, c.imag)
            D = L - N
            acc += C * d * mp.e ** (N * d) if abs(D) == 0 else C * (mp.e ** (L * d) - mp.e ** (N * d)) / D
        out[i] = float(acc.real)
    return out


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


# ---------------------------------------------------------------- differentials
def test_plateau_vs_cancellation():
    """D_dd1: DD error FLAT ~1e-13 across the sweep; the naive collected/Filter
    form's error SCALES as eps/|Δ| (the house discriminator: correct plateaus,
    buggy tracks 1/Δ)."""
    if not HAVE_MP:
        print("D_dd1 SKIP: mpmath not installed (the 40-digit plateau oracle); "
              "install mpmath to run the plateau table. The other three "
              "differentials below do not need it.")
        return
    ref_taus = np.unique(np.concatenate(
        [np.linspace(1e-4, 5e-2, 60), np.linspace(5e-2, 2.0, 200)]))
    dd, orc = [], []
    for tgt in TARGETS:
        ref = mp_ref(tgt, ref_taus)
        dd.append(rel_l2(cand_render(tgt, ref_taus)[0], ref))
        orc.append(rel_l2(oracle_render(tgt, ref_taus), ref))
    print(f"  D_dd1 fused-DD rel-L2 vs mpmath: max {max(dd):.2e} (flat across |Δ|∈[1,0])")
    print(f"        naive Filter rel-L2: {orc[0]:.2e} @Δ=1 → {orc[-2]:.2e} @Δ=1e-8 "
          f"(grows ~1/Δ, ×{orc[-2]/orc[0]:.0e})")
    assert max(dd) < 1e-11, f"DD not machine-precision flat: {max(dd)}"
    assert orc[-2] > 1e3 * orc[0], "naive form did not degrade near coincidence"
    print("D_dd1 PASS: the fused divided difference plateaus at machine precision; "
          "the naive collected form's error tracks 1/Δ")


def test_q428_range():
    """D_dd2: |c|=|a·r| bounded; the collected ringing amp |a·r/Δ| overflows Q4.28."""
    maxc = max(abs(a_v[j] * r[q]) for j in range(3) for q in range(3))
    overflow_target = next(t for t in TARGETS if collected_max_amp(t) > QSAT)
    onset = abs(a_v[JSW] * r[QSW]) / QSAT                 # |Δ| where a·r/Δ hits ±8
    print(f"  D_dd2 fused |c| max = {maxc:.3f} (Q4.28 ceiling {QSAT}); collected "
          f"maxAmp overflows at |Δ|≈{overflow_target:.0e} (onset |a·r|/8 = {onset:.3f})")
    assert maxc < QSAT, "fused coeff overflows Q4.28"
    assert collected_max_amp(1e-8) > 1e6, "collected amp did not blow up"
    print("D_dd2 PASS: fused coeff is bounded (Q4.28-safe); the collected form "
          "overflows for |Δ| < |a·r|/8 ≈ 0.031")


def test_limit():
    """D_dd3: at exact coincidence φ = the deg-1 τ·e^{νd} resonance, no branch."""
    taus = np.arange(1, int(0.3 * SR)) / SR
    lam = nu[QSW]                                          # λ = ν exactly
    got = phi(lam, nu[QSW], taus)                          # cexpm1 → 1 continuously
    want = taus * np.exp(nu[QSW] * taus)                   # d·e^{νd}, closed form
    err = rel_l2(np.real(got), np.real(want))
    print(f"  D_dd3 φ(ν,ν;d) vs closed-form d·e^(νd): rel-L2 = {err:.2e}")
    assert err < 1e-12, f"limit not the τ·e resonance: {err}"
    print("D_dd3 PASS: the divided difference degenerates smoothly to the τ·e "
          "resonance at coincidence — no branch")


def test_continuity():
    """D_dd4: rendered continuity — adjacent-step L2 through coincidence scales
    with |Δλ|; the crossing step (1e-8→0) does NOT spike."""
    taus = np.arange(int(2.0 * SR)) / SR
    renders = [cand_render(t, taus)[0] for t in TARGETS]
    adj = [float(np.linalg.norm(renders[i] - renders[i - 1]))
           for i in range(1, len(TARGETS))]
    crossing = adj[-1]                                     # 1e-8 → 0.0
    others = max(adj[:-1])
    print(f"  D_dd4 adjacent-step L2: crossing (1e-8→0) = {crossing:.2e}, "
          f"max non-crossing = {others:.2e} — crossing is smallest, no spike")
    assert crossing < others, "crossing step spiked (cancellation kink)"
    # linear scaling: the 1e-6→1e-8 step (100× closer) is ~100× smaller
    assert 50 < adj[-2] / adj[-1] < 200, "adjacent step not linear in |Δλ|"
    print("D_dd4 PASS: rendered output is continuous through coincidence — the "
          "derivative is bounded (a cancellation bug spikes here)")


def main():
    for t in (test_plateau_vs_cancellation, test_q428_range, test_limit,
              test_continuity):
        t()


if __name__ == "__main__":
    main()
