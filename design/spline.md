# Uniform cubic spline read by a phase

Date: 2026-11-24. Code: `lean/Tropical/EmitArrow/Spline.lean`. Gates:
`lean/Tropical/Tropicaltest/Spline.lean` (`tropicaltest --spline-only`).

A spline is a shape read by a phase. It is a function of one phase
coordinate, not a parametric Bézier curve. At audio rate it is a
single-cycle waveform, at a low rate an LFO, and with a clamped phase an
envelope. The spline owns no clock: its phase is any integer phase signal.

## Decisions

- **Phase format.** The phase is an `int` in Q0.32 (one cycle is 2³²),
  the DDS phase rail that `FixedSin` and the families use. From a
  `ClockPhasor`, wire `toInt(phase · 2³²)`; that product is exact, because
  the phasor's output is a 32-bit integer over 2³².
- **Uniform knots, no search.** With `n` segments, `s = p · n` in `int`,
  `i = s >> 32` and `t = (s mod 2³²) / 2³²` are exact for every `n < 2³¹`.
  For a power-of-two `n` the multiply is a shift and `i` is the phase's top
  bits. Any other `n` costs one integer multiply, with the same exact split.
- **Boundary is structural.**
  - `wrap` (periodic): `n = K`. The kernel reads a table of `K + 3`
    signals extended at assemble time, `T[j] = c[(j − 1) mod K]`, so the
    seam is an ordinary knot by construction. No runtime index wraps.
  - `clamp` (open): the phase clamps to `[0, 2³²]` and `n = K − 1`, so
    `c[0]` is the value at phase 0 and `c[K − 1]` at phase 1. The end
    neighbours are linear extrapolations, `2c[0] − c[1]` and
    `2c[K−1] − c[K−2]`, so both bases pass through the end values. For the
    B-spline this is the natural end condition (zero second derivative at
    the ends). It is not the knot-multiplicity "clamped B-spline".
- **Basis is host data.** A basis is a 4×4 integer matrix and a divisor:
  Catmull-Rom (`/2`, interpolating, C¹) and the uniform cubic B-spline
  (`/6`, approximating, C²). The kernel evaluates
  `((a₃t + a₂)t + a₁)t + a₀` and divides once. Zero coefficients are dropped
  at assemble time.
- **No new IR node.** The spline is `arr` + `index` + integer and float
  arithmetic. No backend needs the spline as data.
- **Registered set.** The patch bay has no structural arguments, so each
  structural choice is a program type. There are four, at K = 8:
  `PeriodicCatmullRom8`, `OpenCatmullRom8`, `PeriodicBSpline8`,
  `OpenBSpline8`. Each has inputs `phase` (int, Q0.32) and `c0 … c7`
  (float, live when wired from `param:` slots) and output `out`. Other K,
  bases or boundaries are one `splineSig` call in a builder.

## Cost

Per sample, per spline:

- the table pack: `K + 3` stores;
- four `index` reads;
- about 12 multiply-adds;
- one divide.

`index` keeps its generic bounds check: two compares and a select per read.
The check never fires, since `i ∈ [0, n]` and `i + 3 ≤ K + 2` hold by
construction. Whether LLVM removes it depends on its range analysis of
`(p · n) >> 32`; nothing here relies on that. The table pack runs per
sample even when the controls are knobs, because arrays are not hoisted
into the coefficient kernel in v1 (`Ir/Stage0.lean`).

## Not built

- Spline integrals, speed-to-position warps.
- Sample playback, table ports.
- Fourier resynthesis of the spline.
- Editors.
- A playground node kind. Playground port tables are static per kind
  (`Playground/VocabularyMetadata.lean`), so a node whose knob count is the
  structural `K` does not fit, and no playground node emits a phase signal
  to read the spline with.
