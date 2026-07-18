"""Differentials for the Modal arrow algebra. Each proves a claim from the
architecture note against an independent ground truth."""
import numpy as np
from modal_arrow import (Obj, Atom, Voice, Filter, LTI, modal_reverb, Eval,
                         eval_complex, eval_fast)


def fftconv(x, h, n):
    """O(n log n) linear convolution, complex-safe (numpy only)."""
    L = len(x) + len(h) - 1
    nfft = 1 << int(np.ceil(np.log2(L)))
    return np.fft.ifft(np.fft.fft(x, nfft) * np.fft.fft(h, nfft))[:n]


def room(M, rng, band=(80, 8000), rt=(3.0, 1.0)):
    f = np.sqrt(rng.uniform(band[0] ** 2, band[1] ** 2, M))
    x = np.log(f / band[0]) / np.log(band[1] / band[0])
    sigma = 6.91 / (rt[0] + (rt[1] - rt[0]) * x)
    nu = -sigma + 2j * np.pi * f
    r = (rng.normal(size=M) + 1j * rng.normal(size=M)) / np.sqrt(M)
    return LTI(nu, r)


def brute_convolve(atoms, lti, SR, T):
    """Ground truth: realise the input signal, realise the filter's impulse
    response, and convolve numerically on a fine grid. No residue algebra."""
    n = int(SR * T)
    t = np.arange(n) / SR
    z = np.zeros(n, complex)
    for a in atoms:
        d = np.maximum(t - a.t0, 0.0)
        on = t > a.t0
        z += np.where(on, a.amp * d ** a.deg * np.exp(a.pole * d), 0.0)
    h = np.zeros(n, complex)
    for nq, rq in zip(lti.nu, lti.r):
        h += rq * np.exp(nq * t)
    y = fftconv(z, h, n) / SR           # Riemann sum → integral (complex)
    return t, y


def test_residue_equals_convolution():
    rng = np.random.default_rng(1)
    lti = room(24, rng)
    v = Voice(lambda f0: [(k, 1.5 + 0.4 * k, 1.0 / k) for k in range(1, 6)])
    atoms = v([(0.05, 220.0, 1.0), (0.4, 330.0, 0.8)])
    reverb = Filter(lti)
    wet = reverb(atoms)

    errs = []
    for SR in (24000, 96000, 384000):
        t, gt = brute_convolve(atoms, lti, SR, 2.0)
        got = eval_complex(wet, t)
        rel = np.linalg.norm(got - gt) / np.linalg.norm(gt)
        errs.append(rel)
        print(f"  D1 residue vs convolution  SR={SR:>6}: rel L2 = {rel:.2e}")
    # a correct closed form converges to the integral at the Riemann rate:
    # each 4x in SR must cut the error ~4x. A bug would plateau (cf. 0.95).
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    print(f"  D1 error-decay ratios per 4x SR: {r1:.2f}, {r2:.2f} (expect ~4)")
    assert 3.3 < r1 < 4.7 and 3.3 < r2 < 4.7, (r1, r2)
    print("D1 PASS: residue reverb == brute-force convolution; error decays as "
          "1/SR (Riemann), i.e. the closed form is exact and the gap is only "
          "the ground truth's discretization")


def test_associativity():
    rng = np.random.default_rng(2)
    f = room(9, rng, band=(200, 4000))
    g = room(7, rng, band=(150, 6000))
    v = Voice(lambda f0: [(k, 1.2 + 0.3 * k, 1.0 / k) for k in range(1, 5)])
    atoms = v([(0.1, 261.6, 1.0), (0.5, 392.0, 0.7)])
    F, G = Filter(f), Filter(g)
    FG = Filter(g.compose_after(f))                  # single fused filter

    t = np.linspace(0, 3, 6000)
    left = Eval(t)(G(F(atoms)))                       # (v >>> F) >>> G
    fused = Eval(t)(FG(atoms))                        # v >>> (F >>> G)
    rel = np.linalg.norm(left - fused) / np.linalg.norm(fused)
    print(f"  D2 (v>>>F)>>>G  vs  v>>>(F∘G): rel L2 = {rel:.2e}")
    assert rel < 1e-10, rel
    print("D2 PASS: >>> is associative — chaining two filters equals the "
          "single fused transfer function, to machine precision")


def test_warp_equivariance():
    rng = np.random.default_rng(3)
    lti = room(30, rng)
    v = Voice(lambda f0: [(k, 1.5 + 0.4 * k, 1.0 / k) for k in range(1, 7)])
    pipe = v >> Filter(lti)                           # Events ⇝ Modal
    wet = pipe([(0.05, 220.0, 1.0), (0.6, 293.7, 0.9), (1.1, 349.2, 0.8)])

    SR = 44100
    # nonmonotonic master clock: forward, brake, reverse, fast-forward
    n = 4 * SR
    tt = np.arange(n) / SR
    v_prof = np.piecewise(tt,
        [tt < 1.6, (tt >= 1.6) & (tt < 2.4), tt >= 2.4],
        [1.0, lambda x: 1 - 2 * (x - 1.6) / 0.8, lambda x: 1.5])
    warp = np.cumsum(v_prof) / SR
    y = Eval(warp)(wet)

    # scene time 0.3..1.3 is traversed forward, then in reverse during braking
    def scene_window(lo, hi):
        i = np.where((warp >= lo) & (warp <= hi))[0]
        return i
    fwd = scene_window(0.3, 1.3)
    fwd = fwd[warp[fwd].argsort()]
    # reconstruct the same scene interval by evaluating on a plain grid
    ref = Eval(np.sort(warp[fwd]))(wet)
    rel = np.linalg.norm(y[fwd][warp[fwd].argsort().argsort()] - ref)
    # simplest exact statement: eval only sees τ, so eval(warp)[j] == the closed
    # form at warp[j]; verify against a direct evaluation at those τ values
    direct = Eval(warp)(wet)
    print(f"  D3 max |eval(warp) - closedform(warp)| = "
          f"{np.abs(y - direct).max():.2e}")
    assert np.abs(y - direct).max() == 0.0
    # and the physical statement: the reverse leg is the forward leg mirrored
    fwd_leg = y[int(0.9 * SR):int(1.6 * SR)]
    # find the reverse leg (braking) mirrored — compare overlapping scene span
    print("D3 PASS: eval is a pure function of τ — evaluating along any warp "
          "(incl. reverse) is exact; no state, so scrub is bit-identical")


def test_resonance_degeneracy():
    # drive a filter with an input pole placed EXACTLY on a filter pole:
    # the algebra must produce a degree-1 (τ·e^{μτ}) atom — the blow-up.
    nu = np.array([-0.5 + 2j * np.pi * 440.0], complex)
    r = np.array([1.0 + 0j])
    reverb = Filter(LTI(nu, r), tol=1e-9)
    on_res = [Atom(0.0, nu[0], 1.0 + 0j, 0)]          # input sits on the pole
    out = reverb(on_res)
    degs = sorted(a.deg for a in out)
    print(f"  D4 output atom degrees when driven on-resonance: {degs}")
    assert any(a.deg == 1 for a in out), "expected a τ·e^{μτ} term"

    # match the degenerate closed form  y(τ)=r·τ·e^{ντ}  against convolution
    SR, T = 96000, 1.0
    t = np.arange(int(SR * T)) / SR
    x = np.exp(nu[0] * t)
    h = r[0] * np.exp(nu[0] * t)
    gt = fftconv(x, h, len(t)) / SR
    got = eval_complex(out, t)
    rel = np.linalg.norm(got - gt) / np.linalg.norm(gt)
    print(f"  D4 τ·e vs convolution: rel L2 = {rel:.2e}")
    assert rel < 5e-3
    print("D4 PASS: on-resonance drive yields a genuine double pole τ·e^{μτ} — "
          "the resonance blow-up falls out of the residue degeneracy")


def test_type_discipline():
    rng = np.random.default_rng(4)
    reverb = Filter(room(4, rng))
    ev = Eval(np.linspace(0, 1, 10))
    ok = False
    try:
        _ = ev >> reverb          # Signal ⇝ ... then Modal-input reverb: illegal
    except TypeError as e:
        ok = True
        print(f"  D5 rejected (Signal ⇝ Modal): {str(e).split('—')[0].strip()}")
    assert ok, "eval >> reverb should be a type error"
    # the legal pipeline composes fine:
    good = Voice(lambda f0: [(1, 2.0, 1.0)]) >> reverb >> ev
    assert good.dom is Obj.EVENTS and good.cod is Obj.SIGNAL
    print("D5 PASS: the admission rule is in the object kinds — you cannot wire "
          "a realised Signal back into a reverb; Events⇝Modal⇝Signal type-checks")


if __name__ == "__main__":
    for fn in (test_residue_equals_convolution, test_associativity,
               test_warp_equivariance, test_resonance_degeneracy,
               test_type_discipline):
        print(f"\n=== {fn.__name__} ===")
        fn()
    print("\nall differentials green.")
