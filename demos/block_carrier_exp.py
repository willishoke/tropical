"""
Experiments on the nested-Newton / block-partial-fraction carrier for
exponential-polynomial banks (tropical's modal island).

E1  carrier algebra: ordered nodes + numerator in nested Newton basis.
    cascade = concat nodes, multiply numerators (no division by pole gaps).
    Conditioning of three carriers vs a 60-digit reference as poles cluster:
      PF   : collected partial fractions (residueComposeEC style, m+n modes)
      PAIR : pairwise divided-difference atoms (PairedMode style, m*n)
      BLK  : block partial fractions (clusters kept whole, evaluated by
             within-cluster Taylor of the bidiagonal exponential) -- Schur-Parlett shape
E2  runtime evaluation of nested DDs: full-table recurrence vs Opitz Taylor vs hybrid.
E3  term counts for chained rooms: pairwise vs block.
E8  bilateral (future/past) arm: the mixed coupling 1/(lam_f + lam_p) is never hot.
"""
import numpy as np, itertools, math, sys
from mpmath import mp, mpc, mpf, matrix, expm, exp as mexp, fabs
mp.dps = 60
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- polynomials (coeff lists, low->high)
def padd(a, b):
    n = max(len(a), len(b)); out = [0]*n
    for i, x in enumerate(a): out[i] += x
    for i, x in enumerate(b): out[i] += x
    return out
def pmul(a, b):
    out = [0]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b): out[i+j] += x*y
    return out
def pscale(a, c): return [c*x for x in a]
def peval(a, s):
    r = 0
    for c in reversed(a): r = r*s + c
    return r
def psyndiv(a, z):
    """divide polynomial a by (s - z); returns (quotient, remainder). mult/add only."""
    n = len(a)-1; q = [0]*n; r = a[n]
    for i in range(n-1, -1, -1):
        q[i] = r; r = a[i] + r*z
    return q, r
def prod_poly(nodes):
    p = [1]
    for z in nodes: p = pmul(p, [-z, 1])
    return p

# ---------------------------------------------------------------- the carrier
class RF:
    """proper rational function N(s)/prod(s - z_j) with ordered nodes; N stored as plain coeffs.
       nested Newton coefficients n_k (h(d) = sum_k n_k exp[z_1..z_k](d)) derived by synthetic division."""
    def __init__(self, nodes, num):
        self.nodes = list(nodes); self.num = list(num)
        assert len(self.num) <= len(self.nodes)
    @staticmethod
    def from_modes(modes):            # modes: [(amp, pole)] partial fractions
        nodes = [p for _, p in modes]
        num = [0]
        for i, (a, _) in enumerate(modes):
            others = nodes[:i] + nodes[i+1:]
            num = padd(num, pscale(prod_poly(others), a))
        return RF(nodes, num)
    def cascade(self, other):         # convolution: union of nodes, product of numerators. no division.
        return RF(self.nodes + other.nodes, pmul(self.num, other.num))
    def parallel(self, other):        # sum: common denominator. no division.
        return RF(self.nodes + other.nodes,
                  padd(pmul(self.num, prod_poly(other.nodes)), pmul(other.num, prod_poly(self.nodes))))
    def nested(self):
        """n_k, k=1..N: N(s) = sum_k n_k * prod_{j>k}(s - z_j). synthetic division at z_N, z_{N-1}, ..."""
        N = len(self.nodes); num = self.num + [0]*(N - len(self.num)); out = [0]*N
        for k in range(N-1, -1, -1):
            if len(num) == 1: out[k] = num[0]; num = [0]; continue
            q, r = psyndiv(num, self.nodes[k]); out[k] = r; num = q
        return out

# ---------------------------------------------------------------- divided differences of exp at nodes
def opitz_row(nodes, d, prec=None):
    """first row of expm(d*Z), Z bidiagonal: entries exp[z_1..z_k](d), k=1..N. mpmath if prec else float Taylor."""
    N = len(nodes)
    if prec:
        Z = matrix(N, N)
        for i in range(N):
            Z[i, i] = mpc(nodes[i]) * d
            if i+1 < N: Z[i, i+1] = mpf(1) * d
        E = expm(Z)
        return [E[0, k] for k in range(N)]
    # float: scaling & squaring + Taylor on the (upper-triangular) bidiagonal
    Z = np.zeros((N, N), complex)
    for i in range(N):
        Z[i, i] = nodes[i]*d
        if i+1 < N: Z[i, i+1] = d
    s = max(0, int(np.ceil(np.log2(max(1e-300, np.abs(Z).max())))) + 1)
    A = Z / 2**s; E = np.eye(N, dtype=complex); T = np.eye(N, dtype=complex)
    for k in range(1, 30):
        T = T @ A / k; E = E + T
        if np.abs(T).max() < 1e-20: break
    for _ in range(s): E = E @ E
    return list(E[0, :])

def dd_table_row(nodes, d):
    """standard divided-difference recurrence from function values (the 'chart' with 1/(z_j - z_i))."""
    N = len(nodes); f = [np.exp(z*d) for z in nodes]
    tab = [f]
    for lvl in range(1, N):
        prev = tab[-1]
        tab.append([(prev[i+1] - prev[i]) / (nodes[i+lvl] - nodes[i]) for i in range(N-lvl)])
    return [tab[k][0] for k in range(N)]

def clusters(nodes, delta):
    """greedy cluster by proximity (transitive), return list of index lists in original order."""
    N = len(nodes); parent = list(range(N))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(N):
        for j in range(i+1, N):
            if abs(nodes[i] - nodes[j]) < delta: parent[find(i)] = find(j)
    groups = {}
    for i in range(N): groups.setdefault(find(i), []).append(i)
    return list(groups.values())

# ---------------------------------------------------------------- evaluators for h(d)
def h_ref(rf, d):
    """60-digit reference: nested coefficients in mpmath + Opitz row in mpmath."""
    rfm = RF([mpc(z) for z in rf.nodes], [mpc(c) for c in rf.num])
    n = rfm.nested(); row = opitz_row(rfm.nodes, mpf(d), prec=True)
    return sum(nk*rk for nk, rk in zip(n, row))

def h_pf(rf, d):
    """collected partial fractions in float64: residues a_i = N(z_i)/prod_{j!=i}(z_i - z_j)."""
    total = 0
    for i, zi in enumerate(rf.nodes):
        den = 1
        for j, zj in enumerate(rf.nodes):
            if j != i: den *= (zi - zj)
        total += peval(rf.num, zi) / den * np.exp(zi*d)
    return total

def h_nested_table(rf, d):
    n = rf.nested(); row = dd_table_row(rf.nodes, d)
    return sum(nk*rk for nk, rk in zip(n, row))

def h_nested_opitz(rf, d):
    n = rf.nested(); row = opitz_row(rf.nodes, d)
    return sum(nk*rk for nk, rk in zip(n, row))

def block_pf(rf, delta):
    """compile-time block partial fractions, computed in high precision (as the repo does with DyadicI):
       N/D = sum_c N_c/D_c, each block c an RF over its own (close) nodes. Returns list of float RFs."""
    nodesm = [mpc(z) for z in rf.nodes]; numm = [mpc(c) for c in rf.num]
    blocks = []
    for idx in clusters(rf.nodes, delta):
        cn = [nodesm[i] for i in idx]; k = len(idx)
        other = [nodesm[i] for i in range(len(nodesm)) if i not in idx]
        Dother = prod_poly(other)
        # N_c = interpolant of g(s) = N(s)/D_other(s) at the cluster nodes WITH multiplicity.
        # Davies-Higham move: Taylor-expand g about the cluster centroid (g is analytic there,
        # its poles are the other clusters at distance >= delta), then take the polynomial
        # remainder mod D_c(s) = prod(s - z_j): the remainder IS the Hermite interpolant, and
        # polynomial remainder by a monic divisor is division-free. Confluent-safe (eps = 0).
        centroid = sum(cn) / k
        g = lambda s: peval(numm, s) / peval(Dother, s)
        M = k + 24
        tay = mp.taylor(g, centroid, M)                       # coefficients in (s - centroid)
        # shift to plain coefficients in s: P(s) = sum tay_j (s - c)^j
        P = [0]; basis = [1]
        for j in range(M + 1):
            P = padd(P, pscale(basis, tay[j])); basis = pmul(basis, [-centroid, 1])
        num = P
        for z in cn:                                          # reduce mod (s - z) one factor at a time...
            pass
        # ...properly: remainder of P modulo D_c via repeated synthetic division of the *quotient chain*
        Dc = prod_poly(cn); num = list(P)
        while len(num) > k:                                   # long division by monic Dc, division-free
            lead = num[-1]; shift = len(num) - len(Dc)
            for i, c in enumerate(Dc): num[shift + i] -= lead * c
            num.pop()
        blocks.append(RF([complex(z) for z in cn], [complex(c) for c in num]))
    return blocks

def h_block(blocks, d, dtype=complex):
    """per-sample: each block evaluated by its own nested DDs (Opitz Taylor on a tiny matrix)."""
    total = 0
    for b in blocks:
        n = b.nested()
        row = opitz_row(b.nodes, d) if len(b.nodes) > 1 else [np.exp(b.nodes[0]*d)]
        total += sum(dtype(nk)*dtype(rk) for nk, rk in zip(n, row))
    return total

def relerr(x, ref):
    ref = complex(ref); return abs(complex(x) - ref) / max(abs(ref), 1e-300)

# ================================================================ E1
print("=" * 78)
print("E1  chained rooms: voice(4) >> room(6) >> room(6) >> room(6); carriers vs 60-digit reference")
print("    eps = distance between a room-2 pole and a room-3 pole (two independent rooms, runtime-equal)")
def damped(n, smin, smax, wmax):
    return [complex(-rng.uniform(smin, smax), rng.uniform(-wmax, wmax)) for _ in range(n)]
voice = [(complex(rng.uniform(0.3, 1), rng.uniform(-0.5, 0.5)), p) for p in damped(4, 2, 40, 3000)]
room1 = damped(6, 5, 60, 4000); room2 = damped(6, 5, 60, 4000)
d_eval = [0.002, 0.02, 0.2]
print(f"{'eps':>8} {'carrier':>8} " + " ".join(f"{'d='+str(d):>10}" for d in d_eval))
for eps in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 0.0]:
    room3 = damped(6, 5, 60, 4000); room3[2] = room2[4] + eps    # one runtime-(near)-equal pair
    rf = RF.from_modes(voice)
    for room in (room1, room2, room3):
        rf = rf.cascade(RF.from_modes([(1.0, p) for p in room]))   # unit residues per room pole
    blocks = block_pf(rf, delta=1e-2*4000)   # delta ~ 1% of the frequency span
    for name, fn in (("PF", h_pf), ("nestedTab", h_nested_table), ("nestedOpz", h_nested_opitz),
                     ("BLK f64", lambda r, d: h_block(blocks, d)),
                     ("BLK f32", lambda r, d: h_block(blocks, d, dtype=np.complex64))):
        errs = []
        for d in d_eval:
            try:
                with np.errstate(all="ignore"):
                    e = relerr(fn(rf, d), h_ref(rf, d))
            except ZeroDivisionError:
                e = float("inf")
            errs.append(e)
        print(f"{eps:>8.0e} {name:>10} " + " ".join(f"{e:>10.1e}" for e in errs))
    print(f"         (blocks: {[len(b.nodes) for b in blocks]})")

# ================================================================ E2
print("=" * 78)
print("E2  nested DD row exp[z_1..z_k](d), k<=N, random damped nodes with one cluster of size c at spacing eps")
print(f"{'N':>3} {'c':>3} {'eps':>8} {'tableRec':>10} {'opitzTay':>10} {'hybrid':>10}")
def hybrid_row(nodes, d, delta):
    """DD table where within-cluster entries come from Opitz Taylor and cross-cluster from the recurrence."""
    N = len(nodes); cl = clusters(nodes, delta); cid = {}
    for c, idx in enumerate(cl):
        for i in idx: cid[i] = c
    # assume cluster-contiguous ordering (sort by cluster)
    order = [i for idx in cl for i in idx]; nodes = [nodes[i] for i in order]
    cid = {k: cid[i] for k, i in enumerate(order)}
    memo = {}
    def dd(i, j):
        if (i, j) in memo: return memo[(i, j)]
        if cid[i] == cid[j]:
            val = opitz_row(nodes[i:j+1], d)[-1]
        else:
            val = (dd(i+1, j) - dd(i, j-1)) / (nodes[j] - nodes[i])
        memo[(i, j)] = val; return val
    return [dd(0, k) for k in range(N)], nodes
for N, c in [(4, 2), (8, 2), (8, 3), (12, 3), (16, 4)]:
    for eps in [1e-2, 1e-5, 1e-8]:
        base = damped(N, 5, 60, 4000)
        for t in range(1, c): base[t] = base[0] + eps*(1+0.3j)*t
        d = 0.05
        ref = opitz_row([mpc(z) for z in base], mpf(d), prec=True)
        hyb, hn = hybrid_row(base, d, delta=40.0)
        hybref = opitz_row([mpc(z) for z in hn], mpf(d), prec=True)
        e_tab = max(relerr(x, r) for x, r in zip(dd_table_row(base, d), ref))
        e_opz = max(relerr(x, r) for x, r in zip(opitz_row(base, d), ref))
        e_hyb = max(relerr(x, r) for x, r in zip(hyb, hybref))
        print(f"{N:>3} {c:>3} {eps:>8.0e} {e_tab:>10.1e} {e_opz:>10.1e} {e_hyb:>10.1e}")

# ================================================================ E3
print("=" * 78)
print("E3  term counts, voice(m) through r rooms of n poles: pairwise DD atoms vs block/nested carrier")
for m, n, r in [(4, 6, 1), (4, 6, 2), (4, 6, 3), (16, 32, 2), (16, 32, 3)]:
    pair = m * n**r; lin = m + r*n
    print(f"  m={m:>2} n={n:>2} rooms={r}: pairwise {pair:>8}  nested/block {lin:>5}  (collected PF also {lin}, but 1/gap)")

# ================================================================ E8
print("=" * 78)
print("E8  bilateral arm: future pole lam_f = -sf + i wf, past pole (decay in |d|) lam_p = -sp + i wp;")
print("    mixed coupling divisor rho = -(lam_f + lam_p): |rho| >= sf + sp > 0 for damped modes -> never hot.")
worst = min(abs(complex(-sf, wf) + complex(-sp, wp)) / (sf+sp)
            for sf in (1, 10, 100) for sp in (1, 10, 100) for wf in (-5000, 0, 5000) for wp in (-5000, 0, 5000))
print(f"    min |lam_f+lam_p|/(sf+sp) over a grid = {worst:.3f}  (bound is 1)")
# and the closed form of the mixed convolution vs quadrature, one case:
lf, lp, d = complex(-3, 700), complex(-5, -700), 0.01     # mirrored frequency, near the 'hot' shape
s = np.linspace(-1.5, 1.5, 600001); ds = s[1]-s[0]
f_fut = np.where(s >= 0, np.exp(lf*s), 0); g_past = np.where(d - s <= 0, np.exp(lp*np.abs(d - s)), 0)
quad = np.trapezoid(f_fut * g_past, dx=ds)
closed = np.exp(lf*d) / (-(lf + lp))    # for d >= 0
print(f"    mixed convolution at d={d}: quadrature {quad:.6e}  closed-form {closed:.6e}  relerr {abs(quad-closed)/abs(closed):.1e}")
