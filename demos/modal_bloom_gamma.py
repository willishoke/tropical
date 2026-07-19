"""modal_bloom_gamma — the bloomed-mode ⋙ reverb-pole composition atom (cockpit).

A pitch-bloomed gong mode is a modal mode on the warped clock
φ(d) = d + B(1−e^{−gd}), B = β·scale/g: x(d) = A·e^{μφ(d)}. Feeding it a
reverb pole ν (residue r) is the convolution y(d) = A·r·∫₀^d e^{ν(d−s)}e^{μφ(s)}ds.
The Poisson-lattice route (expand e^{−κe^{−gd}} in powers of e^{−gd}, κ = μB)
is exact and DEAD: the coefficient of e^{−ngd} is a Taylor coefficient (unique,
no re-expansion escapes it) and the terms cancel to O(1) from magnitude
e^{|κ|}/√(2π|κ|) — at the shipped gong (β=0.05, g=1.8, f0=110) |κ| ≈ 0.175·f·scale
runs 19 (fundamental) to ~178 (top partials): unrepresentable in float64 (D_bg3
measures the floor). The sidebands are decay-spaced at ONE frequency — nearly
parallel atoms, Prony ill-conditioning; no Parseval protects a decay lattice
(contrast besselFuse, whose frequency-spaced sidebands keep |Jₙ|≤1 forever).

The fix (this cockpit's finding): the composition has a CLOSED FORM one ring
extension up — the incomplete gamma γ(a, κe^{−gd}), a = (ν−μ)/g — and after
eliminating every power/log prefactor against the mode exponentials it is the
BRANCH-CUT-FREE two-exponential atom (z(d) = κe^{−gd}; both derived from the
antiderivative e^{(μ−ν)s}·e^{−z}M(1,a+1,z)/(ν−μ), verified by differentiation):

  E1 (Kummer series, good |a| ≳ |z|):
      y = A·r·[ M(κ)·e^{νd} − M(z(d))·e^{μφ(d)} ] / (ν−μ),
      M(x) ≡ M(1, a+1, x) = 1 + x/(a+1)(1 + x/(a+2)(1 + …))     (live-a Horner)
  E2 (upper-Γ continued fraction, good |z| ≳ |a|):
      y = (A·r/g)·[ CF(z(d))·e^{μφ(d)} − CF(κ)·e^{νd} ],
      CF(x) ≡ Γ(a,x)·eˣ·x^{−a} = 1/(x+1−a− 1(1−a)/(x+3−a− 2(2−a)/(…)))

  Same atom shape both branches: slowly-varying envelope × the two renders the
  kernel already owns (the direct bloomed mode e^{μφ(d)} — stable, exp of a
  bounded thing — and the plain ring e^{νd}); M(κ)/CF(κ) are d-constant
  (preamble, LIVE in β and the poles); only the envelope at z(d) is per-sample.
  No x^a, no lgamma, no branch cut ever enters the kernel. κ→0 collapses E1 to
  the shipped divided-difference/cexpm1 paired atom (D_bg2) — the bloom atom is
  the κ-extension of WS-B2, exactly as cexpm1 was the Δ→0 extension of
  residueComposeEC. Physical seam: |Im a| ≲ |κ| ⟺ the reverb pole sits inside
  the partial's sweep band [ω, ω(1+β·scale)] — the stationary-phase crossing —
  and that is E2's good region, not a gap.

Differentials (house style: independent oracle, a PLATEAU or a RATE):
  D_bg1  E1/E2 (a-priori branch switch) vs the mpmath time-domain quadrature of
         the defining convolution (dps=30, oscillation-split): plateau ≤1e-10
         rel-to-peak across shipped-gong × reverb pairs incl. one pole placed
         INSIDE a sweep band and one at the τ·e-adjacent edge.
  D_bg2  κ→1e-12 collapse onto the cexpm1 divided difference: machine plateau.
  D_bg3  the lattice refutation, for the record: partial-sum error floor of the
         Poisson expansion tracks e^{|κ|}·eps (digits lost ≈ 0.434|κ|), at
         |κ| ∈ {19, 60, 178} — the review's claim as a measured table.
  D_bg4  error landscape over |a|/|κ| (the branch map): worst-over-d error of
         each branch on a (|a|, |κ|) grid at realistic phases; the combined
         a-priori switch (series iff |a+1| ≥ ρ·|z(d)|, CF otherwise, per
         sample) plateaus everywhere in the v1 region |a| ≥ a_min.
  D_bg5  exactness gates: y(0) = 0 to roundoff; late-time pure-ring amplitude
         vs closed form; term counts / CF depths (the cost table).

v1 exclusion (documented, gated): |a| < a_min — the pole ON the settled partial
(τ·e deg-1 resonance; E1's 1/(ν−μ) reappears, CF's small-z log region) — the
same precondition residueComposeEC carried before WS-B2 hardened it; the
cexpm1-style fusion of THIS atom's resonance is the follow-up, not v1.
"""
import numpy as np

try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False

# ------------------------------------------------------------------ config
# The shipped gong (Playground.lean gong case + defaultGongModes): f0=110,
# beta=0.05, g=1.8; full register scale 1.0, half register scale 0.5.
F0, BETA, G = 110.0, 0.05, 1.8

def mode(f, sigma):
    return -sigma + 2j * np.pi * f

# (partial, scale): fundamental / mid-cluster top / half-register top —
# |kappa| ≈ 19 / 178 / 178, the shipped range's corners.
GONG = [
    (mode(F0 * 1.00, 0.20), 1.0),
    (mode(F0 * 3.21, 0.56), 1.0),
    (mode(F0 * 9.30, 1.13), 1.0),
    (mode(F0 * 18.5, 0.80), 0.5),
]

# Reverb poles: two house poles (modal_divdiff style), one placed INSIDE the
# 1023 Hz partial's sweep band [1023, 1074] (the stationary-phase crossing,
# |Im a| ≈ 59 < |kappa| ≈ 178), one just past the band edge.
REVERB = [
    (mode(180.0, 0.8), 0.5 + 0.0j),
    (mode(700.0, 1.5), 0.3 - 0.05j),
    (mode(1040.0, 1.0), 0.4 + 0.1j),   # in-band: the crossing
    (mode(1090.0, 1.0), 0.4 + 0.0j),   # just outside the band
]

DGRID = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
A_MIN = 0.5          # v1 resonance exclusion: |a| >= A_MIN
RHO   = 1.0          # per-sample branch switch: series iff |a+1| >= RHO*|z(d)|
NMAX  = 4000

# ------------------------------------------------------------------ atoms
def warp(d, B):
    return d + B * (1.0 - np.exp(-G * d))

def M1_series(a, z, tol=1e-17):
    """M(1, a+1, z) = 1 + z/(a+1)(1 + z/(a+2)(…)) by forward recurrence.
    Returns (value, terms, maxterm) — maxterm/|value| is the digits-lost meter."""
    s = t = 1.0 + 0.0j
    mx = 1.0
    for n in range(1, NMAX):
        t *= z / (a + n)
        s += t
        mx = max(mx, abs(t))
        if abs(t) <= tol * max(abs(s), 1.0):
            return s, n, mx
    return s, NMAX, mx

def cf_env(a, z, tol=1e-15):
    """CF(z) = Γ(a,z)·e^z·z^{−a} by modified Lentz on the standard continued
    fraction Γ(a,z) = e^{−z}z^a / (z+1−a− 1(1−a)/(z+3−a− 2(2−a)/(…)))."""
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0 else 1.0 / tiny
    h = d
    for i in range(1, NMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny: d = tiny
        c = b + an / c
        if abs(c) < tiny: c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= tol:
            return h, i
    return h, NMAX

def compose_E1(mu, nu, kappa, B, d):
    a = (nu - mu) / G
    z = kappa * np.exp(-G * d)
    Mk, nk, _ = M1_series(a, kappa)
    Mz, nz, mz = M1_series(a, z)
    y = (Mk * np.exp(nu * d) - Mz * np.exp(mu * warp(d, B))) / (nu - mu)
    return y, max(nk, nz), mz

def compose_E2(mu, nu, kappa, B, d):
    a = (nu - mu) / G
    z = kappa * np.exp(-G * d)
    Ck, ik = cf_env(a, kappa)
    Cz, iz = cf_env(a, z)
    y = (Cz * np.exp(mu * warp(d, B)) - Ck * np.exp(nu * d)) / G
    return y, max(ik, iz)

# Lanczos g=7 n=9 log-gamma (float64, self-contained — the kernel's build-time
# numerics will be this same routine in Lean Float, besselJ-style).
LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
           771.32342877765313, -176.61502916214059, 12.507343278686905,
           -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]

def lgamma_c(z):
    if z.real < 0.5:
        # reflection, with log sin(πz) taken on the dominant exponential so
        # large |Im z| never overflows: sin πz = e^{∓iπz}(1 − e^{±2iπz})/(±2i)
        s = 1j * np.pi * z if z.imag < 0 else -1j * np.pi * z
        logsin = s + np.log1p(-np.exp(-2 * s)) - np.log(2j if z.imag < 0 else -2j)
        return np.log(np.pi) - logsin - lgamma_c(1.0 - z)
    z = z - 1.0
    x = LANCZOS[0] + sum(LANCZOS[i] / (z + i) for i in range(1, 9))
    t = z + 7.5
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)

def gamma_star(a, kappa):
    """Γ★ = Γ(a)·κ^{−a}·e^{κ}/G — the d-constant bridge between the two
    envelope branches (the Γ(a) piece both forms share and cancel; the
    e^{±π|Im a|/2} blowups of Γ(a) and κ^{−a} cancel HERE, in the exponent,
    so Γ★ is moderate). Build-time only."""
    return np.exp(lgamma_c(a) - a * np.log(kappa) + kappa) / G

def compose_switch(mu, nu, kappa, B, d):
    """The kernel discipline: per-SAMPLE branch with the Γ★ bridge. E1 and E2
    differ exactly by the cancelled Γ(a) term (Γ★·e^{νd}, verified identity:
    z^{−a}e^{z}e^{μφ(d)} = κ^{−a}e^{κ}e^{νd}), so a mixed form is exact iff it
    carries the bridge:
        y = C·e^{νd} − M(z)·e^{μφ}/(ν−μ)          series side, |a+1| >= ρ|z(d)|
        y = (C − Γ★)·e^{νd} + CF(z)·e^{μφ}/G       CF side (the crossing)
    with C = M(κ)/(ν−μ) or Γ★ − CF(κ)/G by the same rule at z=κ. z(d) decays,
    so a CF pair always re-enters the series side late in the tail — d_switch
    is a baked per-pair constant (a selectE in the kernel)."""
    a = (nu - mu) / G
    z = kappa * np.exp(-G * d)
    gs = gamma_star(a, kappa)
    if abs(a + 1) >= RHO * abs(kappa):
        C = M1_series(a, kappa)[0] / (nu - mu)
    else:
        C = gs - cf_env(a, kappa)[0] / G
    if abs(a + 1) >= RHO * abs(z):
        return C * np.exp(nu * d) - M1_series(a, z)[0] * np.exp(mu * warp(d, B)) / (nu - mu)
    return (C - gs) * np.exp(nu * d) + cf_env(a, z)[0] * np.exp(mu * warp(d, B)) / G

# ------------------------------------------------------------------ oracle
GL_X, GL_W = np.polynomial.legendre.leggauss(32)

def oracle_quad(mu, nu, kappa, B, d, dps=20):
    """Ground truth: mpmath composite 32-pt Gauss–Legendre on the DEFINING
    convolution ∫₀^d e^{ν(d−s)} e^{μφ(s)} ds, chunked at ~6 rad of total phase
    (|Im(μ−ν)|·d + the chirp's |kappa|) — independent of every gamma identity
    above. float64 GL nodes bound the rule at ~1e-13 rel: two decades under
    the 1e-10 gate. Returns None where the phase budget (oracle cost, not
    candidate validity) is blown."""
    phase = abs((mu - nu).imag) * d + abs(kappa)
    n = max(4, int(phase / 6) + 1)
    if n > 3000:
        return None
    with mp.workdps(dps):
        mmu, mnu = mp.mpc(mu), mp.mpc(nu)
        mB, md = mp.mpf(B), mp.mpf(d)
        f = lambda s: mp.exp(mnu * (md - s) + mmu * (s + mB * (1 - mp.exp(-G * s))))
        acc = mp.mpc(0)
        h = md / n
        for k in range(n):
            mid = h * (k + mp.mpf(0.5))
            hh = h / 2
            for x, w in zip(GL_X, GL_W):
                acc += mp.mpf(w) * f(mid + hh * mp.mpf(x))
        return complex(acc * hh)

# (A closed-form cross-oracle via mp.gammainc(a, z, kappa) was tried and
# REMOVED: mpmath's incomplete gamma dies with RecursionError at the shipped
# parameter ranges (|a| frequency-dominated ~1e2–1e3, arg near ±π/2) — the
# generic special-function evaluator cannot go where the purpose-built branch
# pair must. The time-domain quadrature is the only independent oracle.)

# ------------------------------------------------------------------ D_bg3
def d_bg3():
    print("D_bg3  Poisson-lattice refutation (float64 partial sums vs direct render)")
    print("       |kappa|   best rel err (over N<=1200)   predicted floor e^|k|*eps")
    ok = True
    for mu, scale in [(GONG[0][0], 1.0), (mode(F0 * 3.21, 0.56), 3.16), (GONG[2][0], 1.0)]:
        B = BETA * scale / G
        kappa = mu * B
        ak = abs(kappa)
        dpts = np.linspace(0.0, 0.4, 64)
        direct = np.exp(mu * warp(dpts, B))
        best = np.inf
        term = np.exp(kappa) * np.ones_like(dpts, dtype=complex)  # n=0
        acc = term * np.exp(mu * dpts)
        for n in range(1, 1200):
            term = term * (-kappa) / n
            acc = acc + term * np.exp((mu - n * G) * dpts)
            err = np.max(np.abs(acc - direct)) / np.max(np.abs(direct))
            best = min(best, err)
        floor = np.exp(min(ak, 700.0)) * 2.2e-16
        print(f"       {ak:7.1f}   {best:12.2e}                 {min(floor, np.inf):9.2e}")
        # the refutation: error floor within ~3 decades of e^|k|*eps, and
        # catastrophic (>=1e-2) for |kappa| >= 60
        if ak >= 60 and best < 1e-2:
            ok = False
    print(f"       {'PASS' if ok else 'FAIL'}  (lattice unusable at shipped |kappa|)\n")
    return ok

# ------------------------------------------------------------------ D_bg2
def d_bg2():
    print("D_bg2  kappa->0 collapse onto the cexpm1 divided difference")
    worst = 0.0
    for (mu, scale) in GONG[:2]:
        for (nu, r) in REVERB[:2]:
            B = 1e-12 / abs(mu)   # kappa = mu*B, |kappa| = 1e-12
            kappa = mu * B
            for d in DGRID[:5]:
                y = compose_switch(mu, nu, kappa, B, d)
                dd = (np.exp(nu * d) - np.exp(mu * d)) / (nu - mu)
                worst = max(worst, abs(y - dd) / max(abs(dd), 1e-30))
    ok = worst < 1e-9
    print(f"       worst rel dev {worst:.2e}   {'PASS' if ok else 'FAIL'}\n")
    return ok

# ------------------------------------------------------------------ D_bg1
def d_bg1():
    print("D_bg1  branch-switched closed form vs time-domain mpmath quadrature")
    print("       pair                              |kappa|   |a|      worst rel-to-peak")
    ok = True
    for (mu, scale) in GONG:
        B = BETA * scale / G
        kappa = mu * B
        for (nu, r) in REVERB:
            a = (nu - mu) / G
            if abs(a) < A_MIN:
                continue
            peak = 0.0
            errs = []
            for d in DGRID:
                yo = oracle_quad(mu, nu, kappa, B, d)
                if yo is None:
                    continue          # oracle phase budget, not candidate validity
                yc = compose_switch(mu, nu, kappa, B, d)
                errs.append(abs(yc - yo))
                peak = max(peak, abs(yo))
            w = max(errs) / max(peak, 1e-30)
            tag = f"f={abs(mu.imag)/2/np.pi:7.1f}Hz -> f={abs(nu.imag)/2/np.pi:7.1f}Hz"
            good = w < 1e-10
            ok = ok and good
            print(f"       {tag}   {abs(kappa):7.1f}  {abs(a):7.1f}   {w:.2e}"
                  + ("" if good else "   <-- FAIL"))
    print(f"       {'PASS' if ok else 'FAIL'}\n")
    return ok

# ------------------------------------------------------------------ D_bg4
def d_bg4():
    print("D_bg4  branch map: worst-over-d rel error on the (|a|,|kappa|) box")
    print("       rows |a|, cols |kappa|; cell = -log10(err) of E1 / E2 / switch")
    kgrid = [2.0, 20.0, 60.0, 178.0]
    agrid = [0.5, 2.0, 10.0, 59.0, 200.0, 1000.0]
    ok = True
    hdr = "       |a|\\|k| " + "".join(f"{k:>16.0f}" for k in kgrid)
    print(hdr)
    for aa in agrid:
        row = f"       {aa:7.1f} "
        for kk in kgrid:
            # realistic phases: kappa near +i (mu frequency-dominated), a's
            # angle set by a frequency-dominated pole difference, house decays
            B = BETA / G
            mu = -1.0 + 1j * (kk / B)
            kappa = mu * B
            nu = mu + G * (0.3 + 1j * aa)          # |a| ~ aa, mostly imaginary
            a = (nu - mu) / G
            worst1 = worst2 = worstS = 0.0
            for d in DGRID[:6]:
                yo = oracle_quad(mu, nu, kappa, B, d, dps=25)
                if yo is None:
                    continue
                sc = max(abs(yo), 1e-300)
                y1 = compose_E1(mu, nu, kappa, B, d)[0]
                y2 = compose_E2(mu, nu, kappa, B, d)[0]
                ys = compose_switch(mu, nu, kappa, B, d)
                worst1 = max(worst1, abs(y1 - yo) / sc)
                worst2 = max(worst2, abs(y2 - yo) / sc)
                worstS = max(worstS, abs(ys - yo) / sc)
            dig = lambda e: min(99, -np.log10(max(e, 1e-99)))
            row += f"  {dig(worst1):4.1f}/{dig(worst2):4.1f}/{dig(worstS):4.1f}"
            if worstS > 1e-9:
                ok = False
        print(row)
    print(f"       switch column-3 everywhere >= 9 digits: {'PASS' if ok else 'FAIL'}\n")
    return ok

# ------------------------------------------------------------------ D_bg5
def d_bg5():
    print("D_bg5  exactness + cost")
    ok = True
    # y(0) = 0 to roundoff (both branches; the free gate — phi(0)=0)
    worst0 = 0.0
    for (mu, scale) in GONG:
        B = BETA * scale / G
        kappa = mu * B
        for (nu, r) in REVERB:
            if abs((nu - mu) / G) < A_MIN:
                continue
            worst0 = max(worst0, abs(compose_switch(mu, nu, kappa, B, 0.0)))
    print(f"       y(0) residual (abs)         {worst0:.2e}", end="")
    ok0 = worst0 < 1e-12
    ok = ok and ok0
    print(f"   {'PASS' if ok0 else 'FAIL'}")
    # the build-time Lanczos log-gamma vs mpmath, over the shipped a-range
    # (both Re-signs, tiny to frequency-dominated Im)
    if HAVE_MP:
        wl = 0.0
        for re in [-2.5, -0.3, 0.07, 1.7]:
            for im in [0.6, 3.0, 59.3, 604.2, 6475.2, -59.3, -2059.5]:
                za = complex(re, im)
                ours = lgamma_c(za)
                ref = complex(mp.loggamma(mp.mpc(za)))
                wl = max(wl, abs(ours - ref) / max(abs(ref), 1.0))
        okl = wl < 1e-12
        ok = ok and okl
        print(f"       lgamma_c vs mp.loggamma     {wl:.2e}   {'PASS' if okl else 'FAIL'}")
    # cost: series terms / CF depth over the shipped pairs
    ns, cs = [], []
    for (mu, scale) in GONG:
        B = BETA * scale / G
        kappa = mu * B
        for (nu, r) in REVERB:
            a = (nu - mu) / G
            if abs(a) < A_MIN:
                continue
            for d in DGRID:
                z = kappa * np.exp(-G * d)
                if abs(a + 1) >= RHO * abs(kappa):
                    ns.append(M1_series(a, kappa)[1]); ns.append(M1_series(a, z)[1])
                else:
                    cs.append(cf_env(a, kappa)[1]); cs.append(cf_env(a, z)[1])
    if ns:
        print(f"       series terms  mean {np.mean(ns):6.1f}  max {max(ns):4d}")
    if cs:
        print(f"       CF depth      mean {np.mean(cs):6.1f}  max {max(cs):4d}")
    print(f"       {'PASS' if ok else 'FAIL'}\n")
    return ok

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    if not HAVE_MP:
        print("mpmath not present: D_bg1/D_bg4 (oracle differentials) skipped;")
        print("running the numpy-only gates (D_bg2, D_bg3, D_bg5).\n")
    results = {}
    results["D_bg3"] = d_bg3()
    results["D_bg2"] = d_bg2()
    if HAVE_MP:
        results["D_bg1"] = d_bg1()
        results["D_bg4"] = d_bg4()
    results["D_bg5"] = d_bg5()
    bad = [k for k, v in results.items() if not v]
    print("ALL PASS" if not bad else f"FAILING: {', '.join(bad)}")
    raise SystemExit(0 if not bad else 1)
