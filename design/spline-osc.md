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

The budget is an error power relative to the in-band signal power, in dB:
the output minus the ideal band-limited curve (every harmonic strictly
below Nyquist, nothing else), over the ideal curve's AC power. It holds for
every choice of control values.

That is possible because the harmonics split into classes. Harmonic `n` is
`C[n mod K]·φ̂(n/K)/K`, so the class `n ≡ ±m (mod K)` scales as one
`|C[m]|²`, and classes are orthogonal. Any control set's error ratio is a
weighted mean of the class ratios, so the worst case over all controls is the
worst class: `max_m (aliased power of class m) / (in-band power of class m)`,
in units of `φ̂²`. Controls with all their power in one class
(`c_j = cos(2πjm/K)`) reach it.

## The band

**Top edge, from the cubic.** With `M = ⌊Nyquist/|f0|⌋` harmonics in band,
the cubic's aliased power in class `m` is the class total minus its first
terms. The total has a closed form by Poisson summation over the kernel's
autocorrelation `r_k = ∫φ(u)φ(u+k)du` (nonzero for `|k| ≤ 3`):

    Σ_{j∈ℤ} φ̂(m/K + j)² = r_0 + 2·Σ_{k=1..3} r_k cos(2πkm/K)

halved at `m = K/2`, where `±m` are one class. `r_k` is an exact integer
over `420·scale²`, from the basis matrix, so this works for any even
kernel. The in-band terms use `kernelTransform` (the exact carrier). `edge`
is the least `M` at which every class meets the budget; the cubic then meets
it for `|f0| ≤ Nyquist/edge`. The gate checks the closed-form class total
against a brute-force sum (agreement to 4·10⁻¹⁵) and that `edge` is the
least such `M`.

**Bottom edge, from the bank.** A bank of `capacity` partials holds every
harmonic below Nyquist when `|f0| ≥ Nyquist/capacity`. So the band is
`[Nyquist/capacity, Nyquist/edge]`, and it exists only if `capacity > edge`.
A capacity at or below `edge` is refused at assemble ("no band meets the
budget with this many partials"), as are a 0 dB budget and a budget past
120 dB (the class tails are differences of f64 sums; past 10⁻¹² they are
rounding).

**Capacity.** Default `2·edge`: an octave band. The capacity is the only
free choice. A wider band costs more partials (the bank must reach lower);
a narrower one makes the crossfade steeper in `f0` (see continuity). It is
an argument of `splineOscBand`; the registered types use the default.

All of this is in fractions of Nyquist, so nothing depends on the rate.

| K = 8, 60 dB | edge `M` | worst class at the edge | capacity | band at 48 kHz |
|---|---|---|---|---|
| Catmull-Rom | 22 | −60.7 dB | 44 | 545–1091 Hz |
| B-spline | 20 | −66.7 dB | 40 | 600–1200 Hz |

(The B-spline's staircase jumps past −60 dB between `M = 19` and 20.) The
edge scales with `K`: Catmull-Rom at 60 dB has `edge = 11, 22, 44, 177` for
`K = 4, 8, 16, 64`; at 80 dB, `K = 8` needs `edge = 61`, at 100 dB 149.

## The crossfade

    t = (|f0|·capacity/Nyquist − 1) · edge/(capacity − edge)
    w = smoothstep(clamp(t, 0, 1))
    out = (1 − w)·cubic + w·bank

`t` is 0 at the bottom edge and 1 at the top. `w` depends only on `f0`, which
is uniform, so it is a per-call value, not per-sample work. Below the band
`w = 0` exactly and the output is exactly the cubic; above it `w = 1` and the
output is exactly the bank.

Why the mix meets the budget: the cubic's error (alias, at non-harmonic
frequencies) and the bank's (none above the bottom edge but rounding) are
uncorrelated, so the mix's error power is at most
`(1−w)²·A + w²·L ≤ max(A, L)`. Inside the band both are within budget.
This is conservative: near the top of the band the cubic's weight is small,
so the band could sit higher (fewer partials) for the same budget. Not done.

The power sum assumes the aliased components do not land on each other,
true for all but rational `fs/f0`; measured below on generic fundamentals.

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
(`ReduceBegin` with a count argument, `loopCount = capacity`), the emitters
trip `clamp(count, 0, loopCount)`, and the kernel's own count is 0 there: the
`spline-osc-trip` gate asserts all three. The general way was already in the
family (`liveCount`, trip-count-as-data); nothing in `Spectral.lean`
changed.

Above the band the cubic is still evaluated: the kernel has no branch, and
its weight `1 − w` is 0. It is about 20 operations.

Measured, `SplineOscCatmullRom8` on the JIT (44 partials, 512-frame
blocks, wall clock on the dev machine, so only the ratios mean much):

| f0 | regime | partials run | ns per sample |
|---|---|---|---|
| 100 Hz | below the band | 0 | 2 |
| 800 Hz | in the band | 29 | 107 |
| 1500 Hz | above | 15 | 59 |
| 5000 Hz | above | 4 | 22 |

About 3.5 ns per partial (a sine and a cosine on the fixed-point rail, a
complex weight). The cubic costs about 2 ns, so evaluating it above the band
is a few percent at most; skipping it would need a branch the kernel does
not have.

The capacity is 44 partials for Catmull-Rom at K = 8, not the 256 of the
`SplineSpectrum*8` fixtures, and the bank never needs more: below the band
the cubic plays. The 2 ns below the band is everything the kernel does per
sample there, including whatever remains of the literal `φ̂` column
(`design/spline-spectrum.md`, "Stage 0").

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
declare disciplines; the playground does, per knob. A plain `set_param` on
`f0` retunes without re-anchoring, so the phase jumps by
`Φ(inc₁, C) − Φ(inc₀, C)`, as it does for every oscillator.

## What it supports and refuses

- **Uniform fundamental only.** `f0` and `phase` must be constant within a
  kernel call (`sigIsUniform`); anything else is refused at assemble with
  the family's reason. The weight `w` and the trip count are per call.
  Audio-rate FM of this oscillator's clock is refused; the time-domain
  `PeriodicCatmullRom<K>` read by any phase is the object for that.
- **Clock warps.** `clk` is any clock; the bank reads it the way every
  family does.
- **Negative fundamentals.** The phasor runs backward and both realizations
  read the spline in reverse (`inc < 0`, the same integer rail); `w` and the
  trip count use `|f0|`. Gated at −0.7 of the band and in WASM≡JIT.
- **Zero.** The phase is frozen, the output is the cubic at that phase
  (`w = 0`, trip 0), which is also the ideal curve's value.
- **No spectral envelope.** A `G(f, τ)` envelope cannot sit on the cubic, so
  the oscillator is not a family to the term algebra. A filtered spline is
  `SplineSpectrum` under an envelope.
- **Live controls.** `c0 … c(K−1)` wired from `param:` slots are live: the
  DFT and the cubic's table read the slots, no recompile.

## Measured

`tropicaltest --spline-osc-only`, K = 8, 60 dB, 48 kHz:

- **Budget (headline).** Worst error against the ideal band-limited curve
  over 152 configurations (a 36-point log sweep 20 Hz–20 kHz per basis, and
  each one-class control set at 10 points across each band), 8192 samples
  each from τ = 2⁴⁰: **−68.7 dB** (Catmull-Rom, class 3, 655 Hz, w = 0.10).
  The sweep with generic controls: −151 dB at 20 Hz, −99 dB at 214 Hz,
  −82 dB at 470 Hz, −96 dB at 1036 Hz (Catmull-Rom); −179 dB (rounding) from
  the top of the band up.
- **Oracle.** 12 configurations against the f64 mix at τ near 0, 2⁴⁰,
  2⁴⁷: worst error 0.51 of the sine rail's bound.
- **Continuity.** 48 anchored retunes across the band: worst step
  5.1·10⁻⁴ (Catmull-Rom, signal `Σ2|F_n|` 1.2) and 3.4·10⁻⁵ (B-spline), at
  most 0.92 of the derived bound. A hard switch at the top edge could step
  by 2.1·10⁻³ and 1.6·10⁻⁴.
- **Trip count.** 0 at 0 Hz, below and at the bottom edge; 29 and 12 (CR)
  inside and above; the region is `loopCount = 44` with a live count.
