"""
Block partial fractions computed the way the REPO could compute them: in f64
(Sig const-fold / stage-0 precision), from the FACTORED kernel (cascade of
partial-fraction stages), with NO global numerator polynomial and NO Taylor step.

Per cluster c with nodes z_1..z_k, the block numerator's Newton coefficients are
the divided differences g_c[z_1..z_m], m=1..k, of
    g_c(s) = H(s) * D_c(s),   H = prod_i H_i,  H_i = sum_nu r_nu / (s - z_nu)
computed by the Leibniz rule for divided differences over the factors
    (s - a)           : [i..i] = z_i - a, [i..i+1] = 1, longer = 0        (division-free)
    1/(s - a), a not in c : [i..j] = (-1)^{j-i} / prod_{l=i..j} (z_l - a)  (cross-cluster gaps only)
    r (constant)      : diagonal
    (fg)[i..j] = sum_{l=i..j} f[i..l] g[l..j]
Confluent-safe (z_i == z_j never appears in a divisor). k=1 reduces to the ordinary residue.

Block body per sample: h_c(d) = sum_m n_{c,m} * exp[z_m..z_k](d), the exp-DD by the
closed-form two-lane body (series / one-level recurrence), f64.

Compared against a 60-digit mpmath reference built from the same factored stages.
"""
import numpy as np
from mpmath import mp, mpc, mpf, matrix, expm
mp.dps = 60
rng = np.random.default_rng(7)

# ---------- DD tables (upper-triangular dict (i,j) -> value) over nodes z ----------
def T_const(z, r):
    m = len(z); return {(i, j): (r if i == j else 0) for i in range(m) for j in range(i, m)}
def T_lin(z, a):                        # (s - a)
    m = len(z); T = {}
    for i in range(m):
        for j in range(i, m):
            T[(i, j)] = (z[i] - a) if i == j else (1 if j == i + 1 else 0)
    return T
def T_inv(z, a):                        # 1/(s - a), a not a node of z
    m = len(z); T = {}
    for i in range(m):
        for j in range(i, m):
            p = 1
            for l in range(i, j + 1): p = p * (z[l] - a)
            T[(i, j)] = (-1) ** (j - i) / p
    return T
def T_mul(z, A, B):
    m = len(z); T = {}
    for i in range(m):
        for j in range(i, m):
            T[(i, j)] = sum(A[(i, l)] * B[(l, j)] for l in range(i, j + 1))
    return T
def T_add(z, A, B):
    return {k: A[k] + B[k] for k in A}
def T_scale(A, c):
    return {k: c * v for k, v in A.items()}

# ---------- a factored kernel: list of stages, each a list of (residue, pole) ----------
def block_coeffs(stages, cluster_nodes, cluster_ids):
    """Newton coefficients n_m = g_c[z_1..z_m] for the cluster, g_c = H * D_c.
       cluster_ids: set of (stage index, pole index) that belong to the cluster."""
    z = cluster_nodes; m = len(z)
    G = T_const(z, 1)
    for si, stage in enumerate(stages):
        # H_i * prod_{nu in c, nu in stage_i}(s - z_nu)  =  sum_nu r_nu * prod_{nu' != nu}(s - z_nu') / prod_{nu' notin c}(s - z_nu')
        # Do it as: sum_nu r_nu * [ prod_{nu' in c, nu' != nu} (s - z_nu') ] * [ prod_{nu' notin c, nu' != nu} 1/(s - z_nu') ]
        #   (for nu in c: the (s - z_nu) factor of D_c cancels its own pole; for nu notin c: 1/(s-z_nu) stays)
        Hi = T_const(z, 0)
        for ni, (r, p) in enumerate(stage):
            # term_nu = r_nu * D_{c,i} / (s - z_nu):  D_{c,i} = prod over THIS stage's cluster poles
            term = T_const(z, r)
            for nj, (_, q) in enumerate(stage):
                if nj != ni and (si, nj) in cluster_ids:
                    term = T_mul(z, term, T_lin(z, q))
            if (si, ni) not in cluster_ids:
                term = T_mul(z, term, T_inv(z, p))
            Hi = T_add(z, Hi, term)
        G = T_mul(z, G, Hi)
    return [G[(0, mm)] for mm in range(m)]

# ---------- exp divided differences, closed-form two-lane body (f64) ----------
def h_syms(delta, M):
    h = [1] + [0] * M
    for x in delta:
        for n in range(1, M + 1): h[n] = h[n] + x * h[n - 1]
    return h
def expdd_series(z, d, M=20):
    k = len(z); c = sum(z) / k; delta = [v - c for v in z]
    h = h_syms(delta, M); fact = 1.0
    for i in range(1, k): fact *= i
    acc = 0; dp = 1
    for n in range(M + 1):
        acc = acc + dp * h[n] / fact; dp = dp * d; fact *= (n + k)
    return np.exp(c * d) * d ** (k - 1) * acc
def expdd(z, d, thr=1.0):
    k = len(z)
    if k == 1: return np.exp(z[0] * d)
    span = max(abs(z[i] - z[j]) for i in range(k) for j in range(i + 1, k))
    if abs(span * d) < thr: return expdd_series(z, d)
    return (expdd(z[:-1], d) - expdd(z[1:], d)) / (z[0] - z[-1])

def h_block_f64(stages, clusters, d):
    total = 0
    for ids in clusters:
        z = [stages[si][ni][1] for (si, ni) in ids]
        n = block_coeffs(stages, z, set(ids))
        for mm in range(len(z)):
            total += n[mm] * expdd(z[mm:], d)
    return total

# ---------- reference: 60-digit global nested form from the same factored stages ----------
def padd(a, b):
    n = max(len(a), len(b)); out = [0] * n
    for i, x in enumerate(a): out[i] += x
    for i, x in enumerate(b): out[i] += x
    return out
def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b): out[i + j] += x * y
    return out
def prod_poly(nodes):
    p = [mpc(1)]
    for zz in nodes: p = pmul(p, [-zz, mpc(1)])
    return p
def psyndiv(a, zz):
    n = len(a) - 1; q = [0] * n; r = a[n]
    for i in range(n - 1, -1, -1): q[i] = r; r = a[i] + r * zz
    return q, r
def h_ref(stages, d):
    nodes = []; num = [mpc(1)]
    for stage in stages:
        sn = [mpc(p) for _, p in stage]; snum = [mpc(0)]
        for i, (r, _) in enumerate(stage):
            snum = padd(snum, [mpc(r) * c for c in prod_poly(sn[:i] + sn[i + 1:])])
        nodes += sn; num = pmul(num, snum)
    N = len(nodes); num = num + [mpc(0)] * (N - len(num)); nk = [0] * N
    for k in range(N - 1, -1, -1):
        if len(num) == 1: nk[k] = num[0]; num = [mpc(0)]; continue
        q, r = psyndiv(num, nodes[k]); nk[k] = r; num = q
    Z = matrix(N, N)
    for i in range(N):
        Z[i, i] = nodes[i] * d
        if i + 1 < N: Z[i, i + 1] = mpf(d)
    E = expm(Z)
    return complex(sum(nk[k] * E[0, k] for k in range(N)))

def clusters_of(stages, delta):
    ids = [(si, ni) for si, st in enumerate(stages) for ni in range(len(st))]
    z = {k: stages[k[0]][k[1]][1] for k in ids}
    parent = {k: k for k in ids}
    def find(k):
        while parent[k] != k: parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for a in ids:
        for b in ids:
            if a < b and abs(z[a] - z[b]) < delta: parent[find(a)] = find(b)
    groups = {}
    for k in ids: groups.setdefault(find(k), []).append(k)
    return list(groups.values())

def damped(n, smin, smax, wmax):
    return [complex(-rng.uniform(smin, smax), rng.uniform(-wmax, wmax)) for _ in range(n)]

# sanity: one stage, two separated poles, singleton clusters => plain residues
st=[[(1.0+0.2j, complex(-3,100)), (0.5-0.1j, complex(-7,-250))]]
cl=clusters_of(st, 0.1)
for d in (0.01, 0.3):
    print("sanity 1-stage:", abs(h_block_f64(st, cl, d)-h_ref(st,d))/abs(h_ref(st,d)))
st=[[(1.0, complex(-3,100))],[(1.0, complex(-3,100)), (1.0, complex(-5,900))]]   # exact coincidence across stages
cl=clusters_of(st, 0.1)
for d in (0.01, 0.3):
    print("sanity coincident:", abs(h_block_f64(st, cl, d)-h_ref(st,d))/abs(h_ref(st,d)), [len(c) for c in cl])
print("block PF via DD-Leibniz on the FACTORED kernel, all f64; voice(4) >> room(6) x3; rel err vs 60-digit ref")
voice = [(complex(rng.uniform(0.3, 1), rng.uniform(-0.5, 0.5)), p) for p in damped(4, 2, 40, 3000)]
room1 = damped(6, 5, 60, 4000); room2 = damped(6, 5, 60, 4000)
d_eval = [0.002, 0.02, 0.2, 1.0]
for theta in (0.4642, 40.0):
    print(f"--- blocking tolerance theta = {theta} rad/s ---")
    print(f"{'eps':>8} " + " ".join(f"{'d='+str(d):>10}" for d in d_eval) + "   blocks")
    for eps in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 0.0]:
        room3 = damped(6, 5, 60, 4000); room3[2] = room2[4] + eps
        stages = [voice] + [[(1.0, p) for p in room] for room in (room1, room2, room3)]
        cl = clusters_of(stages, theta)
        errs = []
        for d in d_eval:
            ref = h_ref(stages, d)
            dut = h_block_f64(stages, cl, d)
            errs.append(abs(dut - ref) / abs(ref))
        sizes = sorted([len(c) for c in cl], reverse=True)[:4]
        print(f"{eps:>8.0e} " + " ".join(f"{e:>10.1e}" for e in errs) + f"   {sizes}...")

# deliberate triple unison inside ONE room (three runtime-equal poles), theta = repo value
print("--- deliberate triple + a chance neighbour: room2 has 3 poles within 1e-7 of each other ---")
room3 = damped(6, 5, 60, 4000)
room2b = list(room2); room2b[1] = room2b[4] + 1e-7; room2b[3] = room2b[4] - 2e-7 * 1j
stages = [voice] + [[(1.0, p) for p in room] for room in (room1, room2b, room3)]
cl = clusters_of(stages, 0.4642)
errs = [abs(h_block_f64(stages, cl, d) - h_ref(stages, d)) / abs(h_ref(stages, d)) for d in d_eval]
print("   " + " ".join(f"{e:>10.1e}" for e in errs) + f"   blocks {sorted([len(c) for c in cl], reverse=True)[:3]}")
