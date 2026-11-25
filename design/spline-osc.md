# One spline oscillator, alias-bounded at every pitch

Date: 2026-11-25. Code: `lean/Tropical/EmitArrow/SplineOsc.lean`; the types
`SplineOscCatmullRom<K>` and `SplineOscBSpline<K>` in
`lean/Tropical/EmitArrow/Stdlib.lean`. Gates:
`lean/Tropical/Tropicaltest/SplineOsc.lean` (`tropicaltest --spline-osc-only`)
and the `SplineOsc*` cases of `tests/web/wasm_vs_jit.test.ts`.

A periodic spline has two realizations (`design/spline.md`,
`design/spline-spectrum.md`): the time-domain cubic, cheap and aliasing at
high pitch, and its harmonics below Nyquist, alias-free and costing one
partial each. This object plays the cubic at low fundamentals, the bank at
high ones, and crossfades between them over a band of fundamentals derived
from a declared error budget. One object sweeps from LFO rate to audio rate.

## The budget

The budget bounds the error power relative to the in-band signal power, in
dB. The error is the output minus the ideal band-limited curve: every
harmonic strictly below Nyquist, nothing else. The in-band power is that
curve's AC power. The bound holds for every control set and every
fundamental, including rational ones (`f0 = fs·p/N`), where the aliases land
on harmonic bins and add as amplitudes.

The registered types use 60 dB (`splineOscBudgetDb`). The member name does
not carry the budget, so changing that constant changes the band and partial
count of every existing `SplineOsc…<K>` instance. It is part of the
families' contract, not a tuning knob. A second size in the name (budget as
well as K) would need two-size families, which `Family.lean` does not have,
and there is one budget in use. If a second budget is ever wanted, it should
be a second family name.

## In-band harmonics

`M(f0) = ⌈Nyquist/|f0|⌉ − 1` is the number of harmonics strictly below
Nyquist. The bound, the bank's trip count and the gates all use this. A
harmonic exactly at Nyquist is not in band.

## The bound (what is proved)

Write `A_n = 2|F_n|` for harmonic `n`'s amplitude. At every sample, the
cubic's error is the sum of the folded harmonics above `M`, so it is at most
`Σ_{n>M} A_n`. Its mean power over any window is therefore at most
`(Σ_{n>M} A_n)²`. This covers every coincidence of folded components,
including those landing on DC. The in-band power is `Σ_{n≤M} A_n²/2`. That
is exact over whole periods, and is the long-run mean otherwise.

Harmonic `n` is `C[n mod K]·φ̂(n/K)/K`. Group the harmonics into classes
`n ≡ ±m (mod K)`, each with one `|C[m]|`. Cauchy–Schwarz over the classes
gives, for every control set,

    error / in-band  ≤  B(M) = 2·Σ_m a_m(M)² / b_m(M)
    a_m(M) = Σ_{n>M, n≡±m} |φ̂(n/K)|,   b_m(M) = Σ_{1≤n≤M, n≡±m} φ̂(n/K)²

The sum runs over `m = 1 … ⌊K/2⌋`. `B(M) = ∞` while a class has no
harmonic in band. The coupling of classes at rational `f0` (aliases of
different classes landing on one bin) is inside the bound: the numerator
adds every alias coherently.

**How `B` is computed.** All of it is exact carrier or f64 `+ − × ÷`, with no
libm (the `exact-corpse` gate):

- **`φ̂(n/K)` for `n ≤ 4K`** is `kernelTransform`, the exact carrier.
- **Beyond `4K`,** `φ̂` comes from the jump formula. A piecewise cubic with
  knot jumps `J_k(x)` in its `k`-th derivative has, exactly,
  `φ̂(ξ) = Σ_x Σ_k J_k(x)·e^{−iωx}/(iω)^{k+1}`, with `ω = 2πξ`. The jumps are
  integers read off the matrix. `|Re| + |Im|` bounds the modulus.
- **Past `N = 1024·K`,** the tail is bounded by
  `Σ_k S_k·(K/6)^{k+1}/(k·N^k)`, with `S_k = Σ_x |J_k(x)|` and `2π > 6`.
  This term is added to every class.
- **Rounding.** The result is inflated by `1 + 2⁻³⁰` to cover the f64
  rounding of the sums.

**Refused bases.** A basis that is not continuous (`J_0 ≠ 0`; its alias
does not converge), not even, or does not reproduce constants is refused.

**Assumption.** The ideal curve is defined by `M(f0)` of the `f0` value.
The kernel's increment truncates `f0·2³²/rate`, so the rendered frequency
sits up to `rate/2³²` (≈ 10⁻⁵ Hz) below it. That changes which harmonic
is "in band" only within that distance of a band boundary.

The bound is not tight. At K = 8 the exact worst case found by the gate,
an eigenvector over the AC control subspace at rational `f0`, sits several
dB below `B`. Using the exact worst case would shrink the bank. That would
need the eigenvalue maximised over all `f0`, and is not done.

## The band

The bank is exact above its bottom edge, so the mix's error is
`(1 − w)·(cubic error)`. Its ratio is at most `(1 − w(f0))²·B(M(f0))`. The
band is `[Nyquist/capacity, Nyquist/edge]`:

- **`capacity`** is the least `c` with `B(c − 1)` within budget. At the
  bottom edge `w = 0` and `M = c − 1`. Below the band `M` is larger and `B`
  smaller, so the cubic alone meets the budget at every lower `f0`.
- **`edge`** is the least `e` (the widest band) for which
  `(1 − w)²·B(m) ≤ budget` holds on every `m`-interval in the band. On the
  interval where `M = m`, `f0 ∈ [Nyquist/(m+1), Nyquist/m)`. There `B` is
  constant and `(1 − w)²` falls with `f0`, so the worst point is the left
  end, `t = (c − m − 1)·e / ((m + 1)(c − e))`. Every `m` from `e` to
  `c − 1` is checked at that point. `e = c − 1` always passes.
- **Above the band** `w = 1` and the error is the bank's rounding.

So the budget holds at every `f0`, not only at the edges. `worstBound` on the
band is the largest `(1 − w)²·B(m)` over the band.

**Refused at assemble:** fewer than 2 controls (one control is a constant
curve), a capacity below the least, a 0 dB budget, a budget past 120 dB,
and a budget the phase rail's capacity cannot meet.

**At `f0 = Nyquist/edge` exactly,** `t = 1` and `w = 1`. Harmonic `edge`
sits at Nyquist and is not in band, so the bank trips `edge − 1` partials.
**Whenever `t > 0`** (that is, `|f0| > Nyquist/capacity`), the trip count is
at most `capacity − 1`.

**K = 8, 60 dB, coherent bound.** The band in Hz scales with the rate.

| basis | capacity | edge | band at 48 kHz | band at 44.1 kHz |
|---|---|---|---|---|
| Catmull-Rom | 54 | 45 | 444–533 Hz | 408–490 Hz |
| B-spline | 29 | 25 | 828–960 Hz | 760–882 Hz |

The bound is steep in `M`, so the widest band that meets it is narrow:
about a fifth of an octave for Catmull-Rom. The first version of this note
derived the band from an incoherent (power-sum) model, with 44 and 40
partials. That model is wrong at rational `f0`: the Catmull-Rom cubic alone
reaches −59.6 dB at `fs/58`.

Other sizes, Catmull-Rom (B-spline):

| K | capacity | edge |
|---|---|---|
| 2 | 8 (8) | 5 (5) |
| 4 | 20 (15) | 17 (10) |
| 16 | 124 (58) | 112 (53) |
| 64 | 692 (291) | 647 (279) |

## The crossfade

    t = (|f0|·capacity/Nyquist − 1) · edge/(capacity − edge)
    w = smoothstep(clamp(t, 0, 1))
    out = (1 − w)·cubic + w·bank

`w` depends only on `f0`, which is uniform, so it is computed once per call.
Below the band `w = 0` exactly and the output is exactly the cubic; above
it, `w = 1` and the output is exactly the bank.

## The bank's edge

The bank here has no soft Nyquist edge. Its trip count is the number of
harmonics strictly below Nyquist, `⌈Nyquist/|f0|⌉ − 1`, and every one of them
has full weight (`edges := {dc := .off, nyquist := .off}`, with the trip
count as the family's live count). Above the band the output is then the
ideal band-limited curve to rounding. The family's standard edge (the top
10 % of the band) would cost tens of dB against this reference at high
pitch: at `f0` above 0.9·Nyquist the fundamental itself sits in the edge.
The price is a step in `f0` when a harmonic crosses Nyquist, the same step
the ideal curve takes.

## Cost

Below the band the bank's live trip count is 0, so its region runs no
iteration. The plan's family region carries a live count over the capacity
(`ReduceBegin` with a count argument, `loopCount = capacity`). The emitters
trip `clamp(count, 0, loopCount)`, and the kernel's own count is 0 there.
The `spline-osc-trip` gate asserts all three. Nothing in `Spectral.lean`
changed: the general mechanism was already the family's `liveCount`.

Above the band the cubic is still evaluated, because the kernel has no
branch; its weight `1 − w` is 0 there.

Measured on the JIT (wall clock on the dev machine, 512-frame blocks, the
session's 44.1 kHz), so only the ratios mean much:

| type | f0 | regime | partials run | ns per sample |
|---|---|---|---|---|
| `SplineOscCatmullRom8` | 100 Hz | below the band | 0 | 1.9 |
| `SplineOscCatmullRom8` | 412 Hz | bottom of the band | 53 | 203 |
| `SplineOscCatmullRom8` | 450 Hz | in the band | 48 | 186 |
| `SplineOscCatmullRom8` | 1500 Hz | above | 14 | 60 |
| `SplineOscCatmullRom8` | 5000 Hz | above | 4 | 24 |
| `SplineOscCatmullRom16` | 180 Hz | bottom of the band | 122 | 481 |
| `SplineOscBSpline16` | 384 Hz | bottom of the band | 57 | 219 |

About 3.5–4 ns per partial (a sine and a cosine on the fixed-point rail and
a complex weight). The cubic costs about 2 ns, so evaluating it above the
band costs a few percent at most. The worst case is the bottom of the band,
where the bank runs `capacity − 1` partials. At K = 64 that would be 691
partials, about 2.7 µs per sample, which is why the range stops at 16.

## Phase

The cubic reads the family's own fundamental phase: `SplineOsc` builds the
same nodes as `familySig` (`inc = toInt(f0·2³²/rate)`, the clock split into
whole and fraction, the phase offset added on the integer rail) and the
interned graph shares them. Partial `n` is at `n·P` on the integer rail, so
both realizations read one phase and `c_0` sits at phase 0 in both. There is
no float phase anywhere to drift: the `spline-osc-oracle` gate matches an
independent f64 reference at `τ` near 0, 2⁴⁰ and 2⁴⁷ to within half the sine
rail's bound.

## Continuity in f0

A retune is a new call. Its phase offset is re-anchored at the retune time
`C` exactly as the `anchor` discipline does (`design/host-param-dispatch.md`):
`off₁ = off₀ + Φ(inc₀, C) − Φ(inc₁, C)`, so the phase is continuous on the
integer rail and the cubic part does not move. The output steps by

    Δ = (w₁ − w₀)·(bank₀ − cubic) + w₁·(bank₁ − bank₀)
    |Δ| ≤ |Δw|·Σ_{n>M₀} 2|F_n| + w₁·Σ_{harmonics crossing Nyquist} 2|F_n|

`bank − cubic` at one phase is the cubic's alias, at most the harmonic tail
above `M`. `|Δw| ≤ 1.5·Δf/(f_hi − f_lo)` (the smoothstep's largest slope).
A hard switch would step by up to the whole tail at the top edge. The
`spline-osc-continuity` gate steps through the band in 48 anchored retunes
from `τ = 2⁴⁰` and checks each step against the bound.

The session supports this when the host writes `f0` with the `anchor`
discipline and `phase` as its `#phase` companion. Patch-bay types do not
declare disciplines yet; the playground does, per knob. Once patch-bay types
can declare write disciplines, this type should declare `f0` anchored with
`phase` as its `#phase` companion, and the controls `c0 …` raw. Until then a
plain `set_param` on `f0` retunes without re-anchoring, so the phase jumps by
`Φ(inc₁, C) − Φ(inc₀, C)`, as it does for every oscillator.

## What it supports and refuses

- **Uniform fundamental.**
  - **At assemble.** `splineOscSig` refuses an `f0` whose stage signature
    reads τ or an instance output, with the family's reason.
  - **Not refused in a patch.** A program input is uniform at assemble,
    so the registered type itself accepts any wire. The patch bay has no
    rate rule for program inputs: that rule (a frequency inlet accepts only
    uniforms) lives in the playground's wiring layer
    (`design/spectral-family.md` §3). An LFO wired into `f0` of
    `SplineOsc…<K>` in a patch therefore compiles. It is outside this
    object's contract: the weight, the trip count and the bound all assume
    `f0` is constant within the call. The patch bay has no mechanism for a
    registered type to refuse a non-uniform wire, so this is documented,
    not refused.
  - **Audio-rate FM.** For FM of the clock, use the time-domain
    `PeriodicCatmullRom<K>` read by any phase.
- **Clock warps.** `clk` is any clock; the bank reads it the way every
  family does.
- **Negative fundamentals.** The phasor runs backward and both realizations
  read the spline in reverse (`inc < 0`, the same integer rail). `w` and the
  trip count use `|f0|`.
- **Zero.** The phase is frozen. The output is the cubic at that phase
  (`w = 0`, trip 0), which is also the ideal curve's value.
- **No spectral envelope.** A `G(f, τ)` envelope cannot sit on the cubic, so
  the oscillator is not a family to the term algebra.
- **Live controls.** `c0 … c(K−1)` wired from `param:` slots are live, with
  no recompile.
- **Sizes 2 to 16.** One control point is a constant curve. The bank grows
  about linearly with `K`, so the range stops at 16 (see Cost).

## Measured

`tropicaltest --spline-osc-only`, K = 8, 60 dB:

- **Budget (headline).** The test compares the output against the ideal
  band-limited curve over 96 configurations of 8192 samples each, starting
  at τ = 2⁴⁰:
  - a 36-point log sweep from 20 Hz to 20 kHz per basis, with generic
    controls;
  - the exact coherent worst cases at `rate·p/N` (`p ≤ 3`, at 48 and
    44.1 kHz, four sub-sample grid phases). Each uses the worst control set:
    the top eigenvector of the error Gram matrix over the in-band powers on
    the AC subspace. The six worst per basis and rate are rendered.

  The true worst is **−65.2 dB** (B-spline, 857.1 Hz = 48000/56, w = 0.13).
  The model and the render agree to 0.01 dB.
- **Anti-gate.** A band half an octave too high (Catmull-Rom, edge 16,
  capacity 32) measures −57.5 dB at 48000/42 Hz, so the gate fails it. An
  incoherent model passes that band.
- **Derived bands.** For K = 4, 8, 16 and 64, both bases, the derived band
  meets an independent coherent bound at every interval. The independent
  bound uses the closed-form transforms, an explicit class test and its own
  tail. Its capacity agrees exactly for K ≤ 16. At K = 64 Catmull-Rom it is
  692 against an independent least of 690: the core's tail is looser.
- **Jump formula.** Agrees with the closed-form transforms past `4K` to
  5·10⁻¹³ relative.
- **Oracle.** 12 configurations against the f64 mix at τ near 0, 2⁴⁰ and
  2⁴⁷. The worst error is 0.41 of the sine rail's bound.
- **Continuity.** 48 anchored retunes across the band. The worst step is
  1.7·10⁻⁵ (Catmull-Rom) and 4·10⁻⁶ (B-spline), at most 0.97 of the derived
  bound.
