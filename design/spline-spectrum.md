# A periodic spline as its harmonics below Nyquist

Date: 2026-11-25. Code: `lean/Tropical/EmitArrow/SplineSpectrum.lean`, on the
partial family (`EmitArrow/Spectral.lean`, `design/spectral-family.md`).
Gates: `lean/Tropical/Tropicaltest/SplineSpectrum.lean`
(`tropicaltest --spline-spectrum-only`).

A periodic uniform cubic spline (`design/spline.md`) has two
realizations. The time-domain cubic costs the same at every pitch and aliases
at high pitch. Its Fourier series, cut at Nyquist, costs one partial per
harmonic, does not alias while the controls are uniform within a kernel
call, and takes the per-partial envelope `G(f, τ)` that a harmonic family
takes. This note covers the second. Controls modulated at audio rate are
evaluated per sample; the sidebands that modulation adds are not culled at
Nyquist.

## The formula

The curve with `K` control values `c_j` is `f(x) = Σ_j c_j φ(Kx − j)`,
periodised, where `x` is the phase in cycles and `φ` is the basis kernel.
Harmonic `n`, `F_n = ∫₀¹ f(x) e^{−2πinx} dx`, is

    F_n = (1/K) · C[n mod K] · φ̂(n/K),    C[m] = Σ_j c_j e^{−2πi·jm/K},
    φ̂(ξ) = ∫ φ(u) e^{−2πiuξ} du,

and `f(x) = F_0 + Σ_{n≥1} Re(2F_n e^{2πinx})`.

- **Kernel transforms.** For the uniform cubic B-spline, `φ̂(ξ) = sinc⁴ξ`.
  For Catmull-Rom, `φ̂(ξ) = sinc³ξ · (3 sinc ξ − 2 cos πξ)`, where
  `sinc ξ = sin(πξ)/(πξ)`. The code does not use either formula. It
  computes `φ̂` at assemble time from the basis matrix (four cubic pieces,
  each integrated against the exponential in closed form), so any basis with
  an even kernel works. The computation runs in the exact carrier
  (`Tropical.Exact`), not libm. A basis without an even kernel is refused, because its
  `φ̂` is complex. The gate checks the matrix route against both closed forms.
  Both kernels give `φ̂(m) = 0` for every nonzero integer `m`, because they
  reproduce constants (checked exactly on the matrix), so harmonics
  `K, 2K, …` are zero. The column holds exact zeros there; the enclosure's
  midpoint would otherwise be a residue near 10⁻³⁸.
- **Weights.** Partial `n` gets the complex weight `2F_n`: a real data column
  `(2/K)·φ̂(n/K)` times one of `K` complex values `C[m]`. Each `C[m]` is a
  fixed linear combination of the live controls. Only `m ≤ K/2` is computed;
  the rest are conjugates.
- **DC.** `F_0 = C[0]/K`, the mean of the controls. The family's new `dc`
  term adds it once per sample, outside the partial loop, as `a₀·Re G(0)`.
  For the SVF envelopes that gives 1 (lowpass) or 0 (bandpass, highpass).
  The family's DC edge is off for a spline, because at LFO rates the low
  partials are the shape. A `dc` term with a DC edge is refused.
- **Phase.** The family reads its own phasor from `freq` and `clk`. Partial
  `n` is at `n·P` on the integer rail, so the band-limited curve equals the
  time-domain curve read at the fundamental's Q0.32 phase. The
  `spline-spectrum-vs-cubic` gate checks this against `splineSig` driven by
  `n·inc`.

## What changed in the family

`PartialWeights.complex tables weight`: a complex weight `a_k + i·b_k`, read
from tables of any length, each materialized once per sample before the loop.
The partial is the family's projection of `(a_k + i·b_k)·G·e^{iθ_k}`: the
weight is one more complex factor next to the envelopes. This is general (any
per-partial phase, any periodic table) and contains nothing spline-specific.
Real-weight families take the same code path as before.

`PartialFamily.dc`: the optional 0 Hz term described above. A family with no
partials (`count = 0`) still plays it, as a live count of 0 does.

## Stage 0

When the controls are `param:` slots, the DFT depends only on slots, so it
is stage 0. Nothing blocks it. In the emitted port kernel of the live gate's
patch (`TROPICAL_STAGE0_DUMP`), the DFT and the packs of both K-entry `C`
tables run once per kernel call, in the prologue before the frame loop.
They do not run per sample. The 256-entry `φ̂` column is literal data
(stage `fold`), and the kernel packs it on every frame: 256 stores per
sample. A harmonic family's literal weight column (`SpectralSaw`) costs the
same. That cost belongs to how literal tables are lowered, not to this
module. Making literal tables constant globals would remove it for every
family.

## Cost

With `T = min(⌊Nyquist/f0⌋, count)` partials evaluated per sample:

| | cost |
|---|---|
| time-domain cubic | per call: the `K + 3`-entry table pack (slot controls); per sample: 4 reads, ~12 multiply-adds, 1 divide, independent of pitch (`design/spline.md`) |
| band-limited | per call: the DFT, `(K/2 + 1)·K` multiply-adds for each of Re and Im, minus exact zeros and ±1 entries; per sample: the `count`-entry literal column pack (above) and the family's invariants; then per partial: 3 table reads, 1 `mod K` (a mask when K is a power of two), 2 multiplies for the weight, one fixed-point sine **and** one cosine, 2 multiplies + 1 add for the projection, the Nyquist edge (~5 ops) |

A harmonic family with real weights and no envelope needs one sine per
partial. A complex weight needs both, so the band-limited spline costs about
twice a harmonic family's per-partial work. At 110 Hz and 48 kHz, `T = 218`.
Under an envelope the projection already needs both sine and cosine, so the
complex weight adds only one complex multiply per partial.

When the controls are wired from a time-varying signal, the DFT and the
`2K` table stores run per sample instead.

## How much a cubic aliases

Rendered at rate `fs`, the time-domain cubic folds every harmonic above
`M = ⌊fs/(2f0)⌋` back into the band. With `‖c‖₁ = Σ|c_j|`, every harmonic
satisfies `|F_n| ≤ (‖c‖₁/K)·|φ̂(n/K)|`, and

    B-spline:     |φ̂(ξ)| ≤ (π|ξ|)⁻⁴
    Catmull-Rom:  |φ̂(ξ)| ≤ 2(π|ξ|)⁻³ + 3(π|ξ|)⁻⁴

So the aliased amplitude, `Σ_{n>M} 2|F_n|`, is bounded by

    B-spline:     2‖c‖₁ · K³ / (3π⁴M³)
    Catmull-Rom:  2‖c‖₁ · (K²/(π³M²) + K³/(π⁴M³))

and the aliased power, `Σ_{n>M} 2|F_n|²`, by `2‖c‖₁²K⁶/(7π⁸M⁷)` for the
B-spline and `4(‖c‖₁/K)²·(4K⁶/(5π⁶M⁵) + 9K⁸/(7π⁸M⁷))` for Catmull-Rom.

So yes, cubics alias by a bounded amount, and the bound is the decay of
`φ̂`. The B-spline is C². Its worst-case alias amplitude falls as `M⁻³`,
and its power as `M⁻⁷`. Catmull-Rom is only C¹. Its amplitude falls as
`M⁻²`, and its power as `M⁻⁵`. Those are envelopes. The alias itself is a
staircase in `M`, not a power law. It depends on `f0` only through
`M = ⌊fs/2f0⌋`. It is flat while `M` crosses the weak harmonics near
multiples of `K` (where `φ̂` vanishes), and it falls in a step as `M` passes
the strong harmonics between them, near odd multiples of `K/2`: 12, 20 and 28
for K = 8. Both bounds are functions of `M/K`.
A curve with `K` points per cycle has most of its energy in harmonics
`n ≲ K`, so once `K·f0` nears Nyquist the cubic aliases about as much as it
plays in band.

The closed-form alias `Σ_{n>M} 2|F_n|² / Σ_{1≤n≤M} 2|F_n|²` against `M`,
for K = 8 and the alias gate's controls, at 48 kHz (`f0` is the top of each
step, `fs/2M`):

| M | f0 (Hz) | Catmull-Rom (dB) | B-spline (dB) |
|---|---|---|---|
| 3 | 8000 | −0.4 | −19.4 |
| 4 | 6000 | −17.7 | −37.0 |
| 5 | 4800 | −31.4 | −46.5 |
| 6 | 4000 | −39.7 | −55.7 |
| 8 | 3000 | −40.0 | −56.0 |
| 10 | 2400 | −40.3 | −56.6 |
| 11 | 2182 | −40.3 | −57.3 |
| 12 | 2000 | −46.2 | −68.3 |
| 14 | 1714 | −55.5 | −73.8 |
| 16 | 1500 | −55.6 | −73.8 |
| 19 | 1263 | −56.2 | −74.7 |
| 20 | 1200 | −59.2 | −82.7 |
| 21 | 1143 | −63.4 | −84.4 |
| 24 | 1000 | −64.9 | −85.2 |
| 28 | 857 | −67.4 | −92.1 |
| 32 | 750 | −71.4 | −93.6 |
| 40 | 600 | −76.5 | −100.2 |

Measured (`spline-spectrum-alias`, K = 8, 48 kHz): the aliased energy of the
time-domain render matches the closed form `Σ_{n>M} 2|F_n|²` to within
0.5 dB. The band-limited render's residual against its reference sits at the
rounding floor.

Measured with K = 8 controls drawn from the gates' LCG (multiples of 1/64
in [−1, 1]). Values are in dB relative to the in-band signal power:

| basis | f0 (Hz) | M | cubic alias, measured | closed form | band-limited residual |
|---|---|---|---|---|---|
| Catmull-Rom | 987.3 | 24 | −64.9 | −64.9 | −179.2 |
| Catmull-Rom | 2966.6 | 8 | −40.0 | −40.0 | −179.2 |
| Catmull-Rom | 6300.8 | 3 | −0.4 | −0.4 | −179.1 |
| B-spline | 987.3 | 24 | −85.2 | −85.2 | −179.4 |
| B-spline | 2966.6 | 8 | −56.0 | −56.0 | −179.4 |
| B-spline | 6300.8 | 3 | −19.4 | −19.4 | −179.4 |

The fundamentals are `fs/f0 ≈ 48.618, 16.18, 7.618`. These are irrational-looking
ratios, so no aliased component lands on a harmonic over the 65536-sample
window, and the measured energy agrees with the closed form to 0.1 dB.

## Not built

- **Crossover.** Built in `design/spline-osc.md`: the cubic below a band of
  fundamentals, this bank above it, a crossfade derived from an alias budget
  between them.
- **Moving fundamental.** The family needs `freq` uniform within a kernel
  call, so the partial count and edges are fixed for the call. A spline read
  by a phase-modulated or swept clock within one call is the time-domain
  realization's job.
- **Odd kernels and non-uniform knots.** These are refused or out of scope.
- **A stdlib type.** The band-limited spline is a `--fixtures` program
  (`SplineSpectrumCatmullRom8`, `SplineSpectrumBSpline8`; inputs `f0`,
  `c0 … c7`, `clk`; 256 partials). No partial family is registered in the
  stdlib yet. A registered band-limited spline would be a type family
  (`Tropical/Family.lean`) like `PeriodicCatmullRom<K>`, sized by K and by its
  partial capacity.

## The step at a crossover (measured, not built)

If a sweep switched realizations at pitch `f0`, the step would be the
difference between the two renders. There are two regimes, and they meet
where the bank's capacity runs out.

**Low pitch: the bank is capacity-limited.** Below `fs/(2·(count + 1))`
(93 Hz at 48 kHz with 256 partials), the bank stops at `count` and drops the
in-band harmonics `count + 1 … M`. At these pitches the cubic aliases far less
than the bank drops. Closed form, K = 8, the alias gate's controls, 256
partials, in dB relative to the in-band power:

| f0 (Hz) | M | Catmull-Rom: bank error | Catmull-Rom: cubic alias | B-spline: bank error | B-spline: cubic alias |
|---|---|---|---|---|---|
| 20 | 1200 | −117.1 | −150.7 | −156.3 | −203.3 |
| 50 | 480 | −117.3 | −130.8 | −156.4 | −175.4 |
| 80 | 300 | −119.8 | −120.6 | −157.9 | −161.4 |
| 93 | 258 | −124.4 | −117.2 | −162.3 | −156.3 |

The bank error is the missing harmonics `count + 1 … M` plus what the soft
Nyquist edge fades from the evaluated ones: `Σ_{n ≤ count} (1 − e_n)²·2|F_n|²`,
with `e_n` the family's edge law, `smooth((Nyquist − n·f0)·10/Nyquist)`. The
edge touches partials above `0.9·Nyquist`, so with 256 partials it matters
only above 84.4 Hz. At 93 Hz it fades partials 233–256. Missing harmonics
alone would be −134.2 dB (Catmull-Rom) and −177.6 dB (B-spline). The fade
dominates: −124.9 and −162.4 dB.

The two errors cross near 82 Hz for Catmull-Rom and near 85 Hz for the
B-spline. The edge loss there is below −195 dB, so it does not move the
crossings. Below that point the cubic is the more accurate realization as
well as the cheaper one. The `spline-spectrum-vs-cubic` gate uses 1024
partials so that its 20 Hz comparison is not capacity-limited.

**High pitch: the cubic is alias-limited.** Above the capacity point, the
band-limited render matches its own reference (the family with its soft
edge) at the rounding floor. Against the ideal hard truncation at Nyquist,
the edge still fades the top in-band harmonics: by −90.4 dB (Catmull-Rom)
and −116.7 dB (B-spline) at 987.3 Hz, and by nothing at 2966.6 and
6300.8 Hz, where no harmonic lies in the top 10% below Nyquist. Both are
well under the cubic's alias, so the step is the alias, which is the
staircase above. Measured `cubic − band-limited` on the same phase:

| basis | f0 (Hz) | step, rms (dB re signal) | step, peak | signal rms |
|---|---|---|---|---|
| Catmull-Rom | 987.3 | −64.9 | 1.3·10⁻³ | 0.57 |
| Catmull-Rom | 2966.6 | −40.0 | 1.2·10⁻² | 0.57 |
| Catmull-Rom | 6300.8 | −0.4 | 0.65 | 0.41 |
| B-spline | 987.3 | −85.2 | 7.8·10⁻⁵ | 0.44 |
| B-spline | 2966.6 | −56.0 | 1.6·10⁻³ | 0.44 |
| B-spline | 6300.8 | −19.4 | 7.5·10⁻² | 0.44 |

Take a step budget of −60 dB rms. From the closed-form table, the cubic
meets it up to `M = 21`, that is up to `f0 = 24000/21 ≈ 1143 Hz`, for
Catmull-Rom (`M = 20` is −59.2 dB). For the B-spline it meets it up to
`M = 12`, `f0 = 2000 Hz`; just above 2 kHz the alias jumps from −68.3 to
−57.3 dB. So the natural layout is the cubic from 0 Hz to the crossover and
the bank above it. The bank then needs only enough capacity for the pitches
above the crossover, which is a few dozen partials, not 256. All of these
figures are for K = 8 and these controls: the crossover scales with `1/K`
and depends on the controls through `C`.
