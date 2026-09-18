"""
Feasibility check for slice step 3: a CLOSED-FORM per-sample body for a size-k
block (k = 2, 3), i.e. the divided difference exp[z_1..z_k](d), evaluated the
way a kernel would evaluate it (no matrix exponential, no loop-to-convergence),
with a per-sample two-lane switch like the engine's cexpm1:

  lane S (series): exp[z_1..z_k](d) = e^{c d} * sum_n d^{n+k-1} h_n(delta) / (n+k-1)!
     with c = centroid, delta_j = z_j - c, h_n = complete homogeneous symmetric poly.
     Fixed term count M (a kernel body has no data-dependent loop).
  lane R (recurrence): one level of the DD recurrence over size-2 bodies,
     size-2 bodies themselves being cexpm1-style (series/direct switch).

Tested in f64 and in HONEST f32 (every intermediate complex64), vs 60-digit mpmath.
"""
import numpy as np
from mpmath import mp, mpc, mpf, matrix, expm
mp.dps = 60
rng = np.random.default_rng(3)

def ref_row(nodes, d):
    N = len(nodes); Z = matrix(N, N)
    for i in range(N):
        Z[i, i] = mpc(nodes[i]) * d
        if i + 1 < N: Z[i, i + 1] = mpf(d)
    E = expm(Z); return complex(E[0, N - 1])

# ---------------- size-2 body (today's PairedMode shape) ----------------
def dd2(z1, z2, d, dt, M=14, thr=1.0):
    z1, z2, d = dt(z1), dt(z2), dt(d)
    c = (z1 + z2) / dt(2); w = (z1 - z2) / dt(2)      # exp[z1,z2] = e^{cd} * sinh(wd)/w
    x = w * d
    if abs(x) < thr:                                    # series: d * sum x^{2n}/(2n+1)!
        acc = dt(0); term = dt(1)
        for n in range(M):
            acc = acc + term; term = term * x * x / dt((2 * n + 2) * (2 * n + 3))
        val = d * acc
    else:                                               # direct
        val = (np.exp(x) - np.exp(-x)) / (dt(2) * w)
    return np.exp(c * d) * val

# ---------------- size-3 body: series lane ----------------
def h_syms(delta, M, dt):
    """complete homogeneous symmetric polys h_0..h_M of the offsets, by the
       generating recurrence over variables (mult/add only)."""
    h = [dt(1)] + [dt(0)] * M
    for x in delta:
        for n in range(1, M + 1):                       # h_n(x1..xj) = h_n(x1..x_{j-1}) + x_j h_{n-1}(x1..xj)
            h[n] = h[n] + x * h[n - 1]
    return h

def dd3_series(z, d, dt, M=16):
    z = [dt(v) for v in z]; d = dt(d); k = len(z)
    c = sum(z) / dt(k); delta = [v - c for v in z]
    h = h_syms(delta, M, dt)
    # sum_n d^{n+k-1} h_n / (n+k-1)!  -- Horner-free forward accumulation with fixed M
    fact = 1.0
    for i in range(1, k): fact *= i
    acc = dt(0); dp = dt(1)                             # d^n
    for n in range(M + 1):
        acc = acc + dp * h[n] / dt(fact)
        dp = dp * d; fact *= (n + k)
    return np.exp(c * d) * d ** (k - 1) * acc

def dd3_rec(z, d, dt):
    z1, z2, z3 = z
    return (dd2(z1, z2, d, dt) - dd2(z2, z3, d, dt)) / (dt(z1) - dt(z3))

def dd3_body(z, d, dt, thr=1.0):
    z = [dt(v) for v in z]; d = dt(d)
    span = max(abs(z[i] - z[j]) for i in range(3) for j in range(i + 1, 3))
    return dd3_series(z, d, dt) if abs(span * d) < thr else dd3_rec(z, d, dt)

def relerr(x, r): return abs(complex(x) - r) / max(abs(r), 1e-300)

print("size-3 closed-form body vs 60-digit reference; worst rel err over d in [1e-4, 2] s (log grid, 40 pts)")
print("cluster = centroid (-sigma + i omega) + offsets of scale `gap` (rad/s); theta_acc in repo ~= 0.46 rad/s")
print(f"{'gap':>8} {'sigma':>6} {'omega':>7} | {'f64 series-only':>16} {'f64 body':>10} {'f32 body':>10} | lanes(S/R)")
ds = np.logspace(-4, np.log10(2.0), 40)
for gap in [0.0, 1e-9, 1e-6, 1e-3, 0.05, 0.46, 2.0, 40.0]:
    for sigma, omega in [(2.0, 700.0), (20.0, 4000.0)]:
        c = complex(-sigma, omega)
        z = [c, c + gap * (1 + 0.3j), c + gap * (-0.7 + 0.4j)]
        e_s = e_b = e_32 = 0.0; nS = nR = 0
        for d in ds:
            r = ref_row(z, d)
            e_s = max(e_s, relerr(dd3_series(z, d, complex), r))
            b64 = dd3_body(z, d, complex); e_b = max(e_b, relerr(b64, r))
            b32 = dd3_body(z, d, np.complex64); e_32 = max(e_32, relerr(b32, r))
            span = max(abs(z[i] - z[j]) for i in range(3) for j in range(i + 1, 3))
            if abs(span * d) < 1.0: nS += 1
            else: nR += 1
        print(f"{gap:>8.0e} {sigma:>6} {omega:>7} | {e_s:>16.1e} {e_b:>10.1e} {e_32:>10.1e} | {nS}/{nR}")

print()
print("size-2 body (PairedMode shape) sanity, f64 / f32, same grid:")
for gap in [0.0, 1e-6, 0.46, 40.0]:
    c = complex(-5.0, 1000.0); z = [c, c + gap * (1 + 0.3j)]
    e64 = max(relerr(dd2(z[0], z[1], d, complex), ref_row(z, d)) for d in ds)
    e32 = max(relerr(dd2(z[0], z[1], d, np.complex64), ref_row(z, d)) for d in ds)
    print(f"  gap {gap:>8.0e}: f64 {e64:.1e}  f32 {e32:.1e}")

print()
print("NOTE on f32: the large relative errors at big |c d| are the known f32 e^{cd} phase-argument failure")
print("(omega*d up to 8000 rad): the same reason the repo puts phase on the Q0.32 integer clock. The body's")
print("own conditioning is seen in the f64 column; the f32 column bounds what a naive f32 Metal body would do.")
