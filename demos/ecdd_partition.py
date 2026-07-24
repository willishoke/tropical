"""ecdd_partition — the EC/DD erasure Phase-0 cockpit (fork 3').

Scope: design/ecdd-erasure-scope.local.md. The sprint erases the
residueComposeEC / residueComposeDD choice from every surface: the compiler
partitions PER COUPLING (voice pole λ against room pole ν, Δ = λ−ν) and routes
each coupling to the collected form (cheap, `m+n` modes, carries a 1/Δ) or the
divided-difference paired form (coincidence-stable, bounded c = a·r). This
cockpit produces the partition predicate's constants AS DATA, the config
census that sizes the cost, and the second-order-DD go/no-go for v2.

The predicate under measurement (design decision D1 — dual, cost-only), gated
by the paired range cap `|a·r|·min(2/|Δ|, 1/(e·σ_min)) < 8` (a lens-fired
coupling the cap rejects is REFUSED — stays collected, a stated exclusion):

    route (λ, ν) to DD  ⇔  (|Δ| < θ_acc  ∨  |a·r| / |Δ| > W_rail)  ∧  cap

θ_acc is where the collected form's cancellation error (~eps/|Δ| relative)
crosses the seam gate floor; W_rail is the Q4.28 landing ceiling on the
collected ringing weight (range lens, amp-dependent — |Δ| alone is not the
criterion). Because DD ≡ collected to ~2e-6 away from coincidence, θ is a COST
boundary, not a correctness boundary — the freeze errs generous, toward DD.

Differentials (house style: independent oracle, a PLATEAU or a RATE):

  D_p1  θ_acc as data. Collected-vs-oracle relative L2 error SCALES ~eps/|Δ|
        across amp × Q sweeps (amp-INVARIANT — relative error divides the amp
        out, which is exactly why the rail lens is a separate, amp-dependent
        predicate); the DD form PLATEAUS at machine precision over the same
        sweep. θ_acc = (largest |Δ| where any swept config's collected error
        exceeds the 2e-6 gate floor) × 10 generosity margin.
  D_p1b the CHAIN site (deliverable 2's preview). voice ⋙ room ⋙ room with
        the two room poles near-coincident, lowered exactly as Patch.lean does
        (nested residueComposeEC): the collected chain's error vs the oracle
        scales ~eps/|Δ| with a larger constant than the pair site (this is
        the standing 6.7e-4 folded-fold datum's mechanism); the PARTITIONED
        chain — identical EC algebra except the one hot coupling routed to a
        paired atom at the final fold (sort-hot-last) — plateaus at the
        single-crossing floor. The split is EXACT algebra, not approximation:
        EC's residues decompose coupling-wise, and the hot coupling's two
        ±c/Δ residues sum exactly to the bounded paired atom. θ_acc freezes
        off the WORSE of the two sites.
  D_p1c the DATAPATH — the lens that actually decides θ_acc, measured
        through the faithful Q-datapath transcription (divdiff_qdatapath's
        primitives, the WS-B2 cockpit) with option E's per-bank landing
        exponent. The float64 sweeps above bound the compile-time algebra;
        what the sweep FOUND is that neither float64 cancellation nor the
        Q4.28 landing decides — the FREQUENCY GRID does. Each mode's
        rotator increment quantizes ω to a 2π·SR/2³² ≈ 6.5e-5 rad/s grid,
        so the collected form renders a grid-quantized Δ: below one quantum
        the two increments coincide and the ±c/Δ residues cancel to exact
        SILENCE (rel err 1.0 — a representation failure), and up to ~0.05
        rad/s the mis-rendered beat keeps it decades over the floor. The
        paired body's series branch (|z| < 0.1) carries RAW Δ and holds
        ~2.4e-6 throughout. Above ~0.1 rad/s both representations sit on
        the same shared grid noise and there is nothing to win. θ_acc
        freezes off the ADVANTAGE region (collected worse than 10× DD);
        the landing-conditioning error (the old 300× story) is real but
        subordinate (~5e-5 at worst under option E's exponent).
  D_p2  the rail curve. The collected ringing weight |a·r/Δ| crosses the ±8
        Q4.28 magnitude ceiling at |Δ| = |a·r|/8 — measured, and confirmed
        amp-dependent (so a pure-|Δ| predicate provably under-routes). The DD
        coeff |c| = |a·r| stays bounded across the entire sweep (no 1/Δ).
  D_p3  the census. Realistic configs (log-spaced rooms × harmonic voices,
        room chains, deliberate unison stacks): how many couplings are hot
        under the D_p1/D_p2 predicate, how often >1 coupling shares a pole
        (triple coincidence — the sort-hot-last fallback's only gap), and the
        per-config cost delta at the DD premium (~2.2×/mode).
  D_p4  the v2 probe. The second-order divided difference of e^{·d} over
        three poles, evaluated by the STABLE two-regime candidate (bivariate
        ψ₂ series when all nodes coincide within radius; recursive first-order
        cexpm1 legs with a safe outer division otherwise), PLATEAUS vs the
        40-digit mpmath oracle through a triple-coincidence sweep, while the
        naive Newton recursion scales ~eps/|Δ|². This is the Newton fold's
        kernel: if it plateaus, v2 is numerically fundable and the ladder has
        a top (sprint invariant 3).

Run:  uv run --with numpy --with mpmath python demos/ecdd_partition.py
"""
import numpy as np

try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False

rng = np.random.default_rng(0x7501)

# gate floors on record (seam-sweep data): single crossing 2e-6; the folded-
# fold chain gate currently 6.7e-4 (the 1/Δ artifact this sprint tightens).
GATE_FLOOR = 2e-6
Q428_CEIL = 8.0          # plain-bank Q4.28 magnitude ceiling (modal_divdiff D_dd2)
PAIR_CAP = 8.0           # the paired atom's build-time range cap (ecddPairCap)
DD_PREMIUM = 2.2         # per-mode render cost of the paired atom vs a plain mode

# ---------------------------------------------------------------- evaluators
def cexpm1(z):
    """(e^z − 1)/z, stable: Horner series (N=6, matches cexpm1SeriesE) for
    |z| < 1e-3, direct otherwise. Limit 1 at z = 0."""
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < 1e-3
    out = np.empty_like(z)
    zs = z[small]
    acc = np.full_like(zs, 1.0 / 5040.0)
    for ck in (1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, 1.0 / 2, 1.0):
        acc = ck + zs * acc
    out[small] = acc
    zb = z[~small]
    out[~small] = np.expm1(zb) / zb
    return out


def pair_collected(lam, nu, c, d):
    """The hot coupling rendered as the COLLECTED bank computes it: two modes
    with residues ±c/Δ, summed in float64. The 1/Δ cancellation site."""
    w = c / (lam - nu)
    return w * np.exp(lam * d) - w * np.exp(nu * d)


def pair_dd(lam, nu, c, d):
    """The same coupling as the fused paired atom: c·e^{νd}·d·cexpm1(Δd).
    No 1/Δ is ever formed; the τ·e resonance is the smooth series limit."""
    return c * np.exp(nu * d) * d * cexpm1((lam - nu) * d)


def pair_oracle(lam, nu, c, d, dps=40):
    """40-digit ground truth for c·(e^{λd} − e^{νd})/Δ (→ c·d·e^{νd} at
    coincidence), evaluated pointwise in mpmath."""
    with mp.workdps(dps):
        L, N, C = mp.mpc(lam), mp.mpc(nu), mp.mpc(c)
        out = np.empty(len(d), dtype=complex)
        for i, dv in enumerate(d):
            D = mp.mpf(float(dv))
            if L == N:
                v = C * D * mp.exp(N * D)
            else:
                v = C * (mp.exp(L * D) - mp.exp(N * D)) / (L - N)
            out[i] = complex(v)
    return out


def rel_l2(y, ref):
    n = np.linalg.norm(ref)
    return np.linalg.norm(y - ref) / n if n > 0 else np.linalg.norm(y)

# ------------------------------------------------- D_p1: θ_acc as data
print("=" * 72)
print("D_p1  θ_acc — collected error rate vs DD plateau, amp × Q sweep")
print("=" * 72)

D_GRID = np.linspace(1e-4, 1.0, 160)          # render window: 1 s
DELTAS = np.logspace(-8, 2, 21)               # |Δ| in rad/s (detune along iω)
AMPS = [0.01, 0.5, 4.0]                       # |a·r| — should NOT move rel err
SIGMAS = [0.5, 4.0, 30.0]                     # decay rates s⁻¹ (the Q sweep)
W0 = 2 * np.pi * 440.0

if not HAVE_MP:
    raise SystemExit("mpmath required for the Phase-0 oracle "
                     "(uv run --with numpy --with mpmath ...)")

# worst collected error per |Δ| across the amp × Q sweep, and the DD ceiling
coll_worst = np.zeros(len(DELTAS))
dd_worst = np.zeros(len(DELTAS))
for j, dlt in enumerate(DELTAS):
    for amp in AMPS:
        for sg in SIGMAS:
            nu = -sg + 1j * W0
            lam = nu + 1j * dlt              # detune along ω: |Δ| exactly dlt
            c = amp + 0.0j
            ref = pair_oracle(lam, nu, c, D_GRID)
            e_coll = rel_l2(pair_collected(lam, nu, c, D_GRID).real, ref.real)
            e_dd = rel_l2(pair_dd(lam, nu, c, D_GRID).real, ref.real)
            coll_worst[j] = max(coll_worst[j], e_coll)
            dd_worst[j] = max(dd_worst[j], e_dd)

print(f"{'|Δ| rad/s':>12} {'collected rel L2':>18} {'DD rel L2':>12}")
for j, dlt in enumerate(DELTAS):
    mark = "  <-- over gate floor" if coll_worst[j] > GATE_FLOOR else ""
    print(f"{dlt:12.3e} {coll_worst[j]:18.3e} {dd_worst[j]:12.3e}{mark}")

# amp-invariance of the relative error (rail is the separate amp lens)
j_probe = np.argmin(np.abs(DELTAS - 1e-4))
errs_by_amp = []
for amp in AMPS:
    nu = -4.0 + 1j * W0
    lam = nu + 1j * DELTAS[j_probe]
    ref = pair_oracle(lam, nu, amp + 0j, D_GRID)
    errs_by_amp.append(rel_l2(pair_collected(lam, nu, amp + 0j, D_GRID).real,
                              ref.real))
amp_spread = max(errs_by_amp) / min(errs_by_amp)
print(f"\namp-invariance at |Δ|={DELTAS[j_probe]:.1e}: rel-err spread across "
      f"|c| in {AMPS} = {amp_spread:.2f}x")

# pair-site crossing: largest |Δ| whose worst swept error exceeds the floor
hot = [DELTAS[j] for j in range(len(DELTAS)) if coll_worst[j] > GATE_FLOOR]
theta_pair = max(hot) if hot else 0.0
dd_plateau = dd_worst.max()
print(f"\npair-site crossing: collected > {GATE_FLOOR:.0e} for |Δ| <= "
      f"{theta_pair:.3e} rad/s")
print(f"DD plateau across the ENTIRE sweep: {dd_plateau:.3e}")

assert dd_plateau < GATE_FLOOR, "DD must sit under the gate floor everywhere"
# the rate signature: error at the smallest Δ dwarfs the error at the largest
assert coll_worst[0] > 1e3 * coll_worst[-1], "collected error must SCALE"
assert amp_spread < 3.0, "relative collected error must be ~amp-invariant"

# ------------------------------------- D_p1b: the chain-fold site
print()
print("=" * 72)
print("D_p1b chain site — voice >> room >> room through room coincidence")
print("=" * 72)

def ec(bank, room):
    """residueComposeEC as Patch.lean writes it: forced a·H(λ) per bank pole
    + one ringing mode per room pole with the coupling sum. float64."""
    forced = [(p, a * sum(r / (p - q) for q, r in room)) for p, a in bank]
    ringing = [(q, -r * sum(a / (p - q) for p, a in bank)) for q, r in room]
    return forced + ringing

def ec_partitioned(bank, room, hot):
    """EC with the couplings in `hot` (bank-idx, room-idx) routed OUT of the
    collected amps and into paired atoms (pole_bank, pole_room, c = a·r) —
    the exact coupling-wise split of EC's residue algebra. NOTE the forced
    amp is a·h with h the PARTIAL Cauchy sum (hot terms excluded), so the
    hot voice pole's forced mode loses its 1/Δ term too — both halves of
    the coupling migrate together."""
    plain, paired = [], []
    for i, (p, a) in enumerate(bank):
        h = sum(r / (p - q) for qi, (q, r) in enumerate(room)
                if (i, qi) not in hot)
        plain.append((p, a * h))
    for qi, (q, r) in enumerate(room):
        coup = sum(a / (p - q) for i, (p, a) in enumerate(bank)
                   if (i, qi) not in hot)
        plain.append((q, -r * coup))
    for (i, qi) in hot:
        p, a = bank[i]
        q, r = room[qi]
        paired.append((p, q, a * r))
    return plain, paired

def render_plain(modes, d):
    return sum(a * np.exp(p * d) for p, a in modes)

def render_paired(pairs, d):
    return sum(c * np.exp(q * d) * d * cexpm1((p - q) * d)
               for p, q, c in pairs)

def chain_oracle(lam, a, nu1, r1, nu2, r2, d, dps=40):
    """Partial fractions of a·r1·r2/((s−λ)(s−ν1)(s−ν2)) at 40 digits."""
    with mp.workdps(dps):
        L, N1, N2 = mp.mpc(lam), mp.mpc(nu1), mp.mpc(nu2)
        C = mp.mpc(a) * mp.mpc(r1) * mp.mpc(r2)
        AL = C / ((L - N1) * (L - N2))
        A1 = C / ((N1 - L) * (N1 - N2))
        A2 = C / ((N2 - L) * (N2 - N1))
        out = np.empty(len(d), dtype=complex)
        for i, dv in enumerate(d):
            D = mp.mpf(float(dv))
            out[i] = complex(AL * mp.exp(L * D) + A1 * mp.exp(N1 * D)
                             + A2 * mp.exp(N2 * D))
    return out

lam, a = -5.0 + 2j * np.pi * 220.0, 1.0 + 0j
NU1 = -2.5 + 2j * np.pi * 700.0
r1, r2 = 0.5 + 0j, 0.4 + 0.1j

chain_coll = np.zeros(len(DELTAS))
chain_part = np.zeros(len(DELTAS))
print(f"{'|Δ| rad/s':>12} {'collected chain':>16} {'partitioned':>12}")
for j, dlt in enumerate(DELTAS):
    nu2 = NU1 + 1j * dlt
    ref = chain_oracle(lam, a, NU1, r1, nu2, r2, D_GRID).real
    # collected all the way: EC(EC(voice, room1), room2) in float64 — the hot
    # (ν1, ν2) coupling forms at the second compose as ±c/Δ residues.
    full = ec(ec([(lam, a)], [(NU1, r1)]), [(nu2, r2)])
    y_coll = render_plain(full, D_GRID).real
    # partitioned: same fold order, the one hot coupling (bank idx 1 = the ν1
    # ringing mode of the first compose) routed to a paired atom at the final
    # fold — sort-hot-last's mechanics, exact split.
    bank1 = ec([(lam, a)], [(NU1, r1)])
    plain, paired = ec_partitioned(bank1, [(nu2, r2)], {(1, 0)})
    y_part = (render_plain(plain, D_GRID) + render_paired(paired, D_GRID)).real
    chain_coll[j] = rel_l2(y_coll, ref)
    chain_part[j] = rel_l2(y_part, ref)
    mark = "  <-- over gate floor" if chain_coll[j] > GATE_FLOOR else ""
    print(f"{dlt:12.3e} {chain_coll[j]:16.3e} {chain_part[j]:12.3e}{mark}")

hot_chain = [DELTAS[j] for j in range(len(DELTAS))
             if chain_coll[j] > GATE_FLOOR]
theta_chain = max(hot_chain) if hot_chain else 0.0
part_plateau = chain_part.max()
print(f"\nchain-site crossing: collected > {GATE_FLOOR:.0e} for |Δ| <= "
      f"{theta_chain:.3e} rad/s (pair site: {theta_pair:.3e})")
print(f"partitioned-chain plateau: {part_plateau:.3e}")

assert part_plateau < GATE_FLOOR, \
    "the partitioned chain must sit at the single-crossing floor"
assert chain_coll[0] > 1e3 * chain_coll[-1], "collected chain error must SCALE"

print(f"\nfloat64 crossings (pure-algebra bound): pair {theta_pair:.1e}, "
      f"chain {theta_chain:.1e} rad/s — the datapath (D_p1c) decides θ_acc")

# ------------------------------------- D_p1c: the datapath floor
print()
print("=" * 72)
print("D_p1c datapath floor — Q4.28/Q2.30/i64, collected vs DD paired body")
print("=" * 72)

import divdiff_qdatapath as qd

# a 2 s window on the engine's clock, as the WS-B2 calibration does
_idx = np.unique(np.round(np.linspace(1e-4, 2.0, 500) * qd.SR).astype(np.int64))
_idx = _idx[_idx > 0]
CLK = (_idx << 32).astype(np.int64)
DSEC = CLK / qd.TWO32 / qd.SR

def land_k(max_abs):
    """bankLandExp/landK (option E): k = clamp(0, 28, ⌊log₂ maxAbs⌋ − 4) —
    the per-bank Q exponent that removed the i64 wrap for all |A|."""
    if max_abs <= 0.0:
        return 0
    e = int(np.floor(np.log2(max_abs))) - 4       # floatExpZ = ⌊log₂⌋
    return min(28, max(0, e))

def q_collected(lam, nu, c):
    """The hot coupling through modalBankSigTable's exact op sequence
    (option E): two modes with residues ±c/Δ, per-bank landing exponent k,
    Wc = env·amp landed at 2^(28−k), exact Q2.30 oscillator, per-mode
    >>(28−k), i64 sum."""
    w = c / (lam - nu)
    k = land_k(2.0 * (abs(w.real) + abs(w.imag)))  # both modes, |cre|+|cim|
    sc, sh = float(2 ** (28 - k)), 28 - k
    acc = np.zeros(len(CLK), np.int64)
    for pole, amp in ((lam, w), (nu, -w)):
        sig, om = -pole.real, pole.imag
        phQ = qd.mode_phase_q(om, CLK)
        env = np.exp(-sig * DSEC)
        wre = qd.to_int(env * amp.real * sc)
        wim = qd.to_int(env * amp.imag * sc)
        acc = acc + ((wre * qd.fixed_cos_cyc(phQ)
                      - wim * qd.fixed_sin_cyc(phQ)) >> sh)
    return acc.astype(np.float64) / qd.Q30

def q_paired(lam, nu, c):
    """The same coupling through the paired DD body (qA, the shipped
    modalBankSigTableDD skeleton): ν rotator + difference rotator, cexpm1
    select, Wc = c·e^{νd}·d·cexpm1 landed Q4.28, i64 combine."""
    sl, wl = -lam.real, lam.imag
    sn, wn = -nu.real, nu.imag
    ds, wd = sl - sn, wl - wn
    phQnu = qd.mode_phase_q(wn, CLK)
    phQdf = qd.mode_phase_q(wd, CLK)
    cos_df = qd.fixed_cos_cyc(phQdf) / qd.Q30
    sin_df = qd.fixed_sin_cyc(phQdf) / qd.Q30
    env_nu = np.exp(-sn * DSEC)
    env_df = np.exp(-ds * DSEC)
    ez = env_df * (cos_df + 1j * sin_df)
    z = (-ds * DSEC) + 1j * (wd * DSEC)
    zsq = z.real * z.real + z.imag * z.imag
    direct = (ez - 1.0) / np.where(zsq >= qd.THR2, z, 1.0)
    series = qd.cexpm1_series(z, qd.NSERIES)
    ce = np.where(zsq >= qd.THR2, direct, series)
    Wc = c * (env_nu * DSEC) * ce
    wre = qd.to_int(Wc.real * qd.Q28)
    wim = qd.to_int(Wc.imag * qd.Q28)
    acc = (wre * qd.fixed_cos_cyc(phQnu) - wim * qd.fixed_sin_cyc(phQnu)) >> 28
    return acc.astype(np.float64) / qd.Q30

def ref_collected(lam, nu, c):
    """The collected body's own integer-phase float64 mirror (quantized
    increments, exact arithmetic) — isolates the fixed-point landing error."""
    out = np.zeros(len(CLK))
    for pole, amp in ((lam, c / (lam - nu)), (nu, -c / (lam - nu))):
        sig, om = -pole.real, pole.imag
        ph = qd.mode_phase_float(om, CLK)
        env = np.exp(-sig * DSEC)
        out += env * (amp.real * np.cos(ph) - amp.imag * np.sin(ph))
    return out

def ref_paired(lam, nu, c):
    """The paired body's own float mirror (quantized rotator phases, raw z,
    exact float arithmetic, no landing)."""
    sl, wl = -lam.real, lam.imag
    sn, wn = -nu.real, nu.imag
    ds, wd = sl - sn, wl - wn
    ph_nu = qd.mode_phase_float(wn, CLK)
    ph_df = qd.mode_phase_float(wd, CLK)
    ez = np.exp(-ds * DSEC) * (np.cos(ph_df) + 1j * np.sin(ph_df))
    z = (-ds * DSEC) + 1j * (wd * DSEC)
    zsq = z.real * z.real + z.imag * z.imag
    direct = (ez - 1.0) / np.where(zsq >= qd.THR2, z, 1.0)
    series = qd.cexpm1_series(z, qd.NSERIES)
    ce = np.where(zsq >= qd.THR2, direct, series)
    Wc = c * (np.exp(-sn * DSEC) * DSEC) * ce
    return Wc.real * np.cos(ph_nu) - Wc.imag * np.sin(ph_nu)

def ref_exact(lam, nu, c):
    """RAW-pole float64 truth (no frequency grid): what the listener is owed.
    Computed via the stable DD form (float64 cancellation would re-enter at
    |Δ| below ~1e-7 — negligible over this sweep's range)."""
    return (c * np.exp(nu * DSEC) * DSEC * cexpm1((lam - nu) * DSEC)).real

NU_DP = -3.0 + 2j * np.pi * 700.0
DELTAS_DP = np.logspace(-5, 1, 19)
dp_coll_tot, dp_dd_tot = {}, {}
for c_abs in (0.5, 4.0):
    dp_coll_tot[c_abs] = np.zeros(len(DELTAS_DP))
    dp_dd_tot[c_abs] = np.zeros(len(DELTAS_DP))
    print(f"  |c| = {c_abs}:")
    print(f"{'|Δ| rad/s':>12} {'|c/Δ|':>10} {'coll arith':>11} "
          f"{'coll TOTAL':>11} {'dd arith':>10} {'dd TOTAL':>10}")
    for j, dlt in enumerate(DELTAS_DP):
        lam = NU_DP + 1j * dlt
        c = c_abs + 0j
        exact = ref_exact(lam, NU_DP, c)
        ca = rel_l2(q_collected(lam, NU_DP, c), ref_collected(lam, NU_DP, c))
        ct = rel_l2(q_collected(lam, NU_DP, c), exact)
        da = rel_l2(q_paired(lam, NU_DP, c), ref_paired(lam, NU_DP, c))
        dt = rel_l2(q_paired(lam, NU_DP, c), exact)
        dp_coll_tot[c_abs][j] = ct
        dp_dd_tot[c_abs][j] = dt
        over = "  <-- over gate floor" if ct > GATE_FLOOR else ""
        print(f"{dlt:12.3e} {c_abs/dlt:10.1e} {ca:11.3e} {ct:11.3e} "
              f"{da:10.3e} {dt:10.3e}{over}")

# The datapath verdict is an ADVANTAGE region, not an absolute floor: above
# ~0.1 rad/s BOTH representations sit on the shared frequency-grid noise
# (identical totals — nothing for DD to win); below it the collected form
# degrades (grid-quantized Δ renders the wrong beat; at sub-grid Δ the two
# increments COINCIDE and the ±c/Δ residues cancel to silence — rel err 1.0,
# a representation failure, not a precision one) while DD's series branch
# carries RAW Δ and holds the single-crossing floor. θ_acc = the widest |Δ|
# where collected is materially worse than DD, × 10.
GRID_Q = 2 * np.pi * qd.SR / qd.TWO32
theta_dp = 0.0
for c_abs in dp_coll_tot:
    for j, dlt in enumerate(DELTAS_DP):
        if dp_coll_tot[c_abs][j] > max(10.0 * dp_dd_tot[c_abs][j], GATE_FLOOR):
            theta_dp = max(theta_dp, dlt)
dd_dp_floor = max(v.max() for v in dp_dd_tot.values())
print(f"\nfrequency-grid quantum: {GRID_Q:.3e} rad/s — collected detunes "
      f"below it are UNREPRESENTABLE (exact cancellation to silence)")
print(f"advantage crossing: collected worse than 10x DD for |Δ| <= "
      f"{theta_dp:.3e} rad/s  (float64 algebra said "
      f"{max(theta_pair, theta_chain):.0e})")
print(f"DD TOTAL ceiling across the sweep: {dd_dp_floor:.3e} "
      f"(the shared grid noise; its sub-grid floor is 2.4e-6)")

# DIRECTION invariant only — the magnitude is a finding, not a gate: asserting
# a specific dominance ratio would fail the cockpit on a different datapath
# while the compiler is fine. The measured ratio goes on record instead.
assert theta_dp > max(theta_pair, theta_chain), \
    "the datapath advantage region must dominate the float64 algebra bound"
print(f"datapath dominance ratio: {theta_dp / max(theta_pair, theta_chain):.1e}x "
      f"(the frequency grid, not float64 cancellation — a finding, not a gate)")
assert dd_dp_floor < 2e-4, \
    "the DD body must hold the registered seam snr across the whole sweep"
# the representation-failure witness: sub-grid collected renders ~silence
assert dp_coll_tot[0.5][0] > 0.5 and dp_dd_tot[0.5][0] < 1e-5, \
    "sub-grid detune: collected must fail, DD must serve"

# θ_acc freezes off the datapath site, × 10 generosity
THETA_ACC = theta_dp * 10.0
print(f"\nTHETA_ACC (frozen, datapath-site x10 generous-toward-DD): "
      f"{THETA_ACC:.3e} rad/s")

# ------------------------------------------------- D_p2: the rail curve
print()
print("=" * 72)
print("D_p2  the Q4.28 rail — collected weight |a·r/Δ| vs the ±8 ceiling")
print("=" * 72)

print(f"{'|c|':>8} {'rail crossing |Δ| = |c|/8':>28}")
rail_at = {}
for amp in AMPS:
    rail_at[amp] = amp / Q428_CEIL
    print(f"{amp:8.2f} {rail_at[amp]:28.3e}")

# measured: the weight actually crosses the ceiling there, and DD's c never does
dd_c_max = max(AMPS)
for amp in AMPS:
    dlt = rail_at[amp]
    assert abs(amp / dlt) >= Q428_CEIL - 1e-9
    assert abs(amp / (dlt * 1.5)) < Q428_CEIL
print(f"\nDD coeff |c| = |a·r| max over sweep: {dd_c_max:.2f}  "
      f"(ceiling {Q428_CEIL}; bounded, no 1/Δ)")
assert dd_c_max < Q428_CEIL

# the routing verdict, as the compiler states it (classifyCoupling, Modal.lean):
# lens (accuracy | range) then the paired range cap — the SAME function the
# shipped predicate computes, so every witness below runs on shipped semantics.
def classify_coupling(delta_abs, c_abs, sigma_min,
                      theta_acc=THETA_ACC, rail=Q428_CEIL, rail_margin=2.0,
                      cap=PAIR_CAP):
    """'cold' | 'paired' | 'refused' — D1's dual lens gated by the paired
    range cap |c|·min(2/|Δ|, 1/(e·σ_min)) < cap (min < cap ⟺ either bound
    clears)."""
    lens = (delta_abs < theta_acc) or (
        delta_abs > 0.0 and c_abs / delta_abs > rail / rail_margin)
    if not lens:
        return "cold"
    cap_ok = (delta_abs > 0.0 and c_abs * (2.0 / delta_abs) < cap) or \
             (sigma_min > 0.0 and c_abs * (1.0 / (np.e * sigma_min)) < cap)
    return "paired" if cap_ok else "refused"

def route_to_dd(delta_abs, c_abs, sigma_min):
    return classify_coupling(delta_abs, c_abs, sigma_min) == "paired"

# THE CAP INTERPLAY (the review finding, stated as data): |c|·2/|Δ| < cap is
# EXACTLY the complement of the rail lens (|c|/|Δ| > rail/margin = cap/4·2 ⇔
# cap = 8, margin 2), so a rail-fired coupling routes ONLY through the damping
# arm: |Δ| < 2e·σ_min and |c| < 8e·σ_min. Witnesses on both sides:
# (a) the rail lens's SERVICE region — well-damped, heavy, Δ > θ_acc: routes.
assert 0.6 > THETA_ACC and classify_coupling(0.6, 4.0, 1.5) == "paired", \
    "the rail lens must route a well-damped heavy coupling the θ lens misses"
# (b) the same coupling lightly damped — the damping arm can't clear: REFUSED
#     (stays collected, a stated exclusion, never a certified one).
assert classify_coupling(0.6, 4.0, 0.05) == "refused", \
    "a lightly-damped rail-fired coupling must be refused by the cap"
# (c) the complement identity: with sup = 2/|Δ| binding (σ_min large enough to
#     matter removed), rail-fired ⇒ cap-rejected — the two conditions negate.
assert classify_coupling(1.0, 8.0, 0.0) == "refused", \
    "rail-fired with only the 2/|Δ| arm available must always be refused"
print(f"rail service region: |Δ|=0.6, |c|=4, σ_min=1.5 -> paired (damping arm "
      f"clears); σ_min=0.05 -> refused; the 2/|Δ| cap arm alone NEGATES the "
      f"rail lens exactly (|c|·2/|Δ| < 8 ⟺ |c|/|Δ| < 4)")

# ------------------------------------------------- D_p3: the census
print()
print("=" * 72)
print("D_p3  census — hot couplings in realistic configs")
print("=" * 72)

def room_poles(n, f_lo=100.0, f_hi=8000.0, jitter=0.03, sigma=(1.0, 6.0)):
    """A reverbRoom-shaped bank: log-spaced modes with jitter."""
    f = np.geomspace(f_lo, f_hi, n) * (1 + jitter * rng.standard_normal(n))
    sg = rng.uniform(*sigma, n)
    return -sg + 2j * np.pi * f

def voice_poles(f0, n):
    """Harmonic partials, mild inharmonicity."""
    k = np.arange(1, n + 1)
    return -rng.uniform(1.0, 8.0, n) + 2j * np.pi * f0 * k * (1 + 1e-4 * k * k)

def census(voice, room, amps):
    """Per-coupling routing over the full m·n grid; returns (n_couplings,
    n_hot, n_refused, poles that carry >1 hot coupling). Cap-refused couplings
    render collected (the stated exclusion) — counted so the refusal region's
    real-world bite is on record, not assumed empty."""
    hot_pairs = []
    n_refused = 0
    for i, lam in enumerate(voice):
        for q, nu in enumerate(room):
            verdict = classify_coupling(abs(lam - nu), abs(amps[i, q]),
                                        min(-lam.real, -nu.real))
            if verdict == "paired":
                hot_pairs.append((i, q))
            elif verdict == "refused":
                n_refused += 1
    from collections import Counter
    v_deg = Counter(i for i, _ in hot_pairs)
    r_deg = Counter(q for _, q in hot_pairs)
    shared = [p for p, d in list(v_deg.items()) + list(r_deg.items()) if d > 1]
    return len(voice) * len(room), len(hot_pairs), n_refused, shared

N_TRIALS = 400
tot_hot, tot_coup, tot_refused = 0, 0, 0
multi_hot_cfgs, any_hot_cfgs = 0, 0
for _ in range(N_TRIALS):
    room = room_poles(24)
    f0 = rng.uniform(60.0, 1200.0)
    voice = voice_poles(f0, 12)
    amps = (rng.uniform(0.05, 1.0, (len(voice), len(room)))
            * np.exp(1j * rng.uniform(0, 2 * np.pi, (len(voice), len(room)))))
    nc, nh, nr, shared = census(voice, room, amps)
    tot_coup += nc
    tot_hot += nh
    tot_refused += nr
    any_hot_cfgs += (nh > 0)
    multi_hot_cfgs += (len(shared) > 0)

print(f"random voice x log-spaced room, {N_TRIALS} trials:")
print(f"  hot couplings: {tot_hot}/{tot_coup} "
      f"({100.0 * tot_hot / tot_coup:.3f}% of couplings)")
print(f"  cap-REFUSED couplings (lens fired, collected floor): "
      f"{tot_refused}/{tot_coup}")
print(f"  configs with any hot coupling: {any_hot_cfgs}/{N_TRIALS}")
print(f"  configs with a SHARED-pole multi-hot (triple coincidence, the "
      f"sort-hot-last gap): {multi_hot_cfgs}/{N_TRIALS}")

# a deliberately tuned config: one partial parked on a room mode
room = room_poles(24)
voice = voice_poles(100.0, 12)
voice[3] = room[10] + 1j * 1e-5            # tuned unison, the served case
amps = np.full((12, 24), 0.5 + 0j)
nc, nh, nr, shared = census(voice, room, amps)
cost = (nc - nh + DD_PREMIUM * nh) / nc
print(f"\ntuned-unison config (a partial parked on a room mode): "
      f"{nh}/{nc} hot ({nr} refused), shared-pole poles: {len(shared)}, "
      f"render cost x{cost:.3f} of all-collected")
assert nh >= 1 and nr == 0, "the tuned unison must route, not be cap-refused"

# a deliberate unison STACK: many rooms on one pole (the columnize watch-item)
stack = np.repeat(room[10], 6) + 1j * rng.uniform(-1e-4, 1e-4, 6)
nc, nh, nr, shared = census(voice, stack, np.full((12, 6), 0.5 + 0j))
print(f"unison stack (6 rooms on one mode): {nh}/{nc} hot — the paired "
      f"family wants WS2 columnizing before it wants v2 if this is common")

# chain-fold site: two rooms, room-vs-room couplings under the same predicate
chain_hot = 0
for _ in range(N_TRIALS):
    r1, r2 = room_poles(16), room_poles(16)
    for nu1 in r1:
        for nu2 in r2:
            if route_to_dd(abs(nu1 - nu2), 0.25,
                           min(-nu1.real, -nu2.real)):
                chain_hot += 1
print(f"chain fold (room x room, {N_TRIALS} trials): "
      f"{chain_hot}/{N_TRIALS * 256} couplings hot "
      f"({100.0 * chain_hot / (N_TRIALS * 256):.3f}%)")

# ------------------------------------------------- D_p4: the v2 probe
print()
print("=" * 72)
print("D_p4  second-order DD probe — the Newton fold's kernel (v2 go/no-go)")
print("=" * 72)

def psi2(z0, z1, terms=24):
    """ψ₂(z0,z1) = Σ_{m,n≥0} z0^m z1^n / (m+n+2)!  (Hermite–Genocchi: the
    2-simplex integral of e over the node offsets). Entire, symmetric;
    ψ₂(0,0) = 1/2. The bivariate generalization of cexpm1's series."""
    import math
    tot = 0.0 + 0.0j
    for m in range(terms):
        for n in range(terms - m):
            tot += (z0 ** m) * (z1 ** n) / math.factorial(m + n + 2)
    return tot


def dd2_stable(x0, x1, x2, d):
    """f[x0,x1,x2] of e^{·d}: all-close -> d²·e^{x2 d}·ψ₂((x0−x2)d,(x1−x2)d);
    otherwise recursive Newton with STABLE first-order legs (d·e^{·d}·cexpm1)
    and the outer division taken over the WIDEST node gap (safe by regime)."""
    xs = sorted([x0, x1, x2], key=lambda x: x.imag)
    gaps = [abs(xs[0] - xs[1]), abs(xs[1] - xs[2]), abs(xs[0] - xs[2])]
    if max(g * d for g in gaps) < 0.5:                      # all-close regime
        z0, z1 = (xs[0] - xs[2]) * d, (xs[1] - xs[2]) * d
        return d * d * np.exp(xs[2] * d) * psi2(z0, z1)
    a, b, c_ = xs                                           # widest gap = a..c_
    dd_ab = d * np.exp(b * d) * complex(cexpm1((a - b) * d))
    dd_bc = d * np.exp(c_ * d) * complex(cexpm1((b - c_) * d))
    return (dd_ab - dd_bc) / (a - c_)


def dd2_naive(x0, x1, x2, d):
    """The Newton recursion as float64 writes it — both cancellation sites live."""
    f = [np.exp(x0 * d), np.exp(x1 * d), np.exp(x2 * d)]
    d01 = (f[0] - f[1]) / (x0 - x1)
    d12 = (f[1] - f[2]) / (x1 - x2)
    return (d01 - d12) / (x0 - x2)


def dd2_oracle(x0, x1, x2, d, dps=40):
    with mp.workdps(dps):
        X = [mp.mpc(x0), mp.mpc(x1), mp.mpc(x2)]
        D = mp.mpf(d)
        f = [mp.exp(x * D) for x in X]
        d01 = (f[0] - f[1]) / (X[0] - X[1])
        d12 = (f[1] - f[2]) / (X[1] - X[2])
        return complex((d01 - d12) / (X[0] - X[2]))

NU2 = -3.0 + 2j * np.pi * 440.0
D_PROBE = [0.05, 0.3, 1.0]
print(f"{'|Δ| rad/s':>12} {'naive rel err':>14} {'stable rel err':>15}")
naive_errs, stable_errs = [], []
for dlt in np.logspace(-6, 1, 15):
    nu1 = NU2 + 1j * dlt                 # triple coincidence: all within Δ
    lam = NU2 + 1j * dlt * 0.5
    en = es = 0.0
    for d in D_PROBE:
        ref = dd2_oracle(lam, nu1, NU2, d)
        en = max(en, abs(dd2_naive(lam, nu1, NU2, d) - ref) / abs(ref))
        es = max(es, abs(dd2_stable(lam, nu1, NU2, d) - ref) / abs(ref))
    naive_errs.append(en)
    stable_errs.append(es)
    print(f"{dlt:12.3e} {en:14.3e} {es:15.3e}")

# mixed regime: two coincident, one far (the recursive branch)
lam_far = -5.0 + 2j * np.pi * 2000.0
mixed = max(abs(dd2_stable(lam_far, NU2 + 1j * 1e-7, NU2, d)
                - dd2_oracle(lam_far, NU2 + 1j * 1e-7, NU2, d))
            / abs(dd2_oracle(lam_far, NU2 + 1j * 1e-7, NU2, d))
            for d in D_PROBE)
print(f"mixed regime (two close, one 1.5 kHz away): stable rel err "
      f"{mixed:.3e}")

v2_plateau = max(stable_errs)
assert v2_plateau < 1e-9, "the stable 2nd-order DD must plateau"
assert max(naive_errs[:5]) > 1e2 * v2_plateau, "naive must scale (~eps/Δ²)"
assert mixed < 1e-9, "the mixed-regime recursive branch must hold"

print()
print("=" * 72)
print("Phase-0 verdict")
print("=" * 72)
print(f"  THETA_ACC  = {THETA_ACC:.3e} rad/s  (datapath ADVANTAGE crossing "
      f"{theta_dp:.0e} x 10; the frequency grid, not float64 cancellation, "
      f"owns the accuracy lens — float64 algebra said {theta_pair:.0e}. "
      f"Sub-grid detunes are unrepresentable collected: exact silence.)")
print(f"  RAIL lens  = |c|/|Δ| > {Q428_CEIL}/2 (margin 2x) — amp-dependent; "
      f"comparable in scale to θ_acc (binding for |c| > ~{THETA_ACC * Q428_CEIL / 2:.1f}), "
      f"so BOTH lenses earn their keep: accuracy for moderate amps, range "
      f"for heavy couplings. |Δ| alone is not the criterion (D1 confirmed). "
      f"SERVICE region: the 2/|Δ| cap arm exactly negates this lens, so it "
      f"routes only through the damping arm (|c| < 8e·σ_min) — well-damped "
      f"heavy couplings; the rest are REFUSED (stated, collected floor).")
print(f"  DD plateau = {dd_plateau:.3e} everywhere (θ is a COST boundary)")
print(f"  census     = hot couplings are rare in log-spaced configs, "
      f"~all singletons; sort-hot-last covers everything short of "
      f"deliberate unison stacks")
print(f"  v2 probe   = stable 2nd-order DD plateaus at {v2_plateau:.3e} "
      f"(bivariate ψ₂ + recursive regime); the Newton fold is numerically "
      f"fundable — go, pending the census saying anyone needs it")
print("\nall differentials PASS")
