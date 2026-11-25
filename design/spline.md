# Uniform cubic spline read by a phase

Date: 2026-11-24, envelopes 2026-11-25. Code:
`lean/Tropical/EmitArrow/Spline.lean`, `lean/Tropical/EmitArrow/PhaseClip.lean`.
Gates: `lean/Tropical/Tropicaltest/Spline.lean`,
`lean/Tropical/Tropicaltest/PhaseClip.lean` and, for the families,
`Tropicaltest/TypeFamily.lean` (`tropicaltest --spline-only`).
The browser player ships two spline patches, `web/patches/spline-voice.json`
and `spline-tremolo.json`, and draws one cycle of their output from the
kernel itself (`web/CLAUDE.md`, "The curve display").

A spline is a shape read by a phase. It is a function of one phase
coordinate, not a parametric Bézier curve. Every spline is periodic. At
audio rate it is a single-cycle waveform, at a low rate an LFO, and read by
a clock that stops at the end of its cycle, an envelope. The spline owns no
clock: its phase is any integer phase signal.

## Decisions

- **Phase format.** The phase is an `int` in Q0.32 (one cycle is 2³²),
  the DDS phase rail that `FixedSin` and the families use. From a
  `ClockPhasor`, wire `toInt(phase · 2³²)`. In f64 that product is exact,
  because the phasor's output is a 32-bit integer over 2³².
- **Uniform knots, no search.** With `n` segments, `s = p · n` in `int`,
  `i = s >> 32` is exact on every target and `t = (s mod 2³²) / 2³²` is
  exact in f64, for every `n < 2³¹`. For a power-of-two `n` the multiply is
  a shift and `i` is the phase's top bits. Any other `n` costs one integer
  multiply, with the same split. In an f32 artifact the 32-bit fraction
  rounds to 24 bits and `t` can reach 1.0; that is harmless, because the
  spline is continuous there.
- **Periodic only.** `n = K`. The kernel reads a table of `K + 3` signals
  extended at assemble time, `T[j] = c[(j − 1) mod K]`, so the seam is an
  ordinary knot by construction. No runtime index wraps. The phase is taken
  modulo the cycle, so phase 1 (2³²) reads exactly what phase 0 reads.
- **Basis is host data.** A basis is a 4×4 integer matrix and a divisor:
  Catmull-Rom (`/2`, interpolating, C¹) and the uniform cubic B-spline
  (`/6`, approximating, C²). The kernel evaluates
  `((a₃t + a₂)t + a₁)t + a₀` and divides once. Zero coefficients are dropped
  at assemble time.
- **No new IR node.** The spline is `arr` + `index` + integer and float
  arithmetic. No backend needs the spline as data.
- **Registered set.** The splines are type families over K
  (`design/type-family.md`): `PeriodicCatmullRom<K>` and
  `PeriodicBSpline<K>`, `1 ≤ K ≤ 1024`, e.g. `PeriodicCatmullRom12`.
  Each has inputs `phase` (int, Q0.32) and `c0 … c(K−1)` (float, live when
  wired from `param:` slots) and output `out`. The size-8 members are also
  registered at boot. Other bases are one `splineSig` call in a builder.

## Envelopes: the pulse width is a phase warp

There are no triggers. A gate is a clock with a pulse width `w ∈ (0, 1]`,
and the pulse width is a warp of the clock's phase:

    phaseclip(p, w) = min(p / w, 1)      (p in cycles, reduced to [0, 1))

The read sweeps one full cycle over the first `w` of the input cycle, then
holds at phase 1. Read through a periodic spline, that is an envelope that
always fits its gate: it starts at phase 0, reaches phase 1 exactly when
the gate falls, and holds there; phase 1 is phase 0, so it ends where it
started (at `c₀` for Catmull-Rom, which interpolates; at the curve's
phase-0 value `(c_{K−1} + 4c₀ + c₁)/6` for the B-spline). The value is
continuous at the hold point, and so is the slope when the spline's slope
at phase 0 is zero (`c₁ = c_{K−1}` for both bases: a stationary point, not
necessarily an extremum); otherwise the slope jumps there. A closed C¹
curve leaves phase 0 with the slope it arrives at phase 1 with, so an
envelope from rest whose start is not flat dips past its rest value just
before the gate falls; `c₁ = c_{K−1}` makes the seam flat, which removes
the dip and the slope jump together.

- **Name.** The registered type is `PhaseClip`; the builder is
  `phaseClipSig`. It is the unipolar special case of Csound's `pdclip`
  (linear phase clipping for phase-distortion synthesis) with the window at
  the start of the cycle and `width` the unclipped fraction; `pdclip`'s
  `kWidth` is the clipped fraction, so the two read in opposite senses. No
  single standard term names this special case.
- **Ports.** `PhaseClip` takes `phase` (int, Q0.32, any integer: it is
  reduced to the cycle) and `width` (float, default 1). It outputs `phase`
  (int, `[0, 2³²]`, wired straight into a spline's `phase`) and `gate`
  (float, 1 for the first `w` of the cycle, else 0). The gate is the
  comparison the clip makes anyway; there is no other gate mechanism. The
  held output 2³² reads as phase 0 to anything that reduces its phase mod
  2³²: in a `PhaseClip → PhaseClip` chain the second gate is high for the
  whole of the first one's hold.
- **Width carrier.** The width is a float, so a knob is an ordinary
  `param:` slot and a `set_param` moves it without a recompile. Inside, it
  becomes an integer word `W`, the width in units of 2⁻³² of a cycle.
- **Out of range.** A static width must lie in `[2⁻³³, 1]`: `(0, 1]` less
  the widths narrower than half a phase unit, whose word would round to 0.
  The interval is declared on the port (`InputDecl.interval?`).
  - Refused: a bare numeric literal outside it, when the session compiles
    the patch (`bareLiteralAtPort`, called from `sessionToResolvedRoot`,
    the same crossing rule as a fixed-point format); and
    `PulseWidth.static` outside it, at assemble time in a builder.
  - Clamped: everything else, because only a bare numeric literal is
    checked. `neg(0.25)`, a `paramDecl` of 0, `set_param width 0`, or
    another instance's output all reach the kernel, which clamps
    `W ∈ [1, 2³²]`: a width below half a phase unit is the narrowest gate
    (one phase unit), a width above 1 is the full cycle, where the clip is
    the identity.

### Exactness

With `p = phase mod 2³²`:

    phaseclip = ⌊p · 2³² / W⌋  if p < W,   2³²  otherwise;   gate = p < W

- **Exact on the integer rail, given `W`.** `p · 2³²` overflows an `int`,
  so the quotient is two steps of long division in base 2¹⁶: `q₁ =
  ⌊p·2¹⁶ / W⌋`, `r₁ = p·2¹⁶ − q₁W`, `q₂ = ⌊r₁·2¹⁶ / W⌋`, result `q₁·2¹⁶ +
  q₂`. Every divide is of non-negative integers and, in the lane the final
  select keeps (`p < W`), every operand is below 2⁴⁹, so the result is the
  exact floor on every target. In the discarded lane (`p ≥ W`, e.g.
  `W = 1`, `p = 2³² − 1`) `q₁·2¹⁶` can wrap; that is defined two's-complement
  arithmetic and the value is never read. The gate
  edge `p < W` is exact. The held value 2³² is phase 1, read by the spline
  as phase 0 bit for bit.
- **Where it rounds: the width word.** A static width is `round(w · 2³²)`
  in exact integer arithmetic at assemble time, at least 1 by the interval. A live width is
  `clamp(toInt(round(w · 2³²)), 1, 2³²)` per sample; `w · 2³²` is exact in
  f64 and f32 (a power-of-two scale) and `round` is half away from zero,
  so `|W − w · 2³²| ≤ ½`. Hence the clip is exact for every width that is
  a multiple of 2⁻³² (½, ¼, 0.375, any dyadic knob step), and otherwise
  differs from the real `p / w` by less than `1 + 1/(2w)` phase units (the
  floor plus the word's rounding): at `w = 0.001` about 2⁻²³ of a cycle.
  In an f32 artifact the width is binary32, so `W` keeps 24 significant
  bits; the integer path is unchanged.
- **Cost.** Per sample: one mask, three shifts, two integer divides, a
  multiply, a compare and two selects; for a live width also a multiply,
  a round, a conversion and a clamp. A divide by a static `W` is a
  constant divide, which LLVM strength-reduces.

## Cost

Per sample, per spline:

- four `index` reads;
- about 12 multiply-adds;
- one divide;
- the table pack, `K + 3` stores, only when a control varies within a
  kernel call. When the controls are `param:` slots, the pack is stage 0 and
  runs once per call, in the port kernel's prologue (checked with
  `TROPICAL_STAGE0_DUMP` on the `spline-live` patch, 2026-11-25).

`index` keeps its generic bounds check: two compares and a select per read.
The check never fires, since `i ∈ [0, K)` and `i + 3 ≤ K + 2` hold by
construction. Whether LLVM removes it depends on its range analysis of
`(p · n) >> 32`; nothing here relies on that.

## The band-limited realization

A periodic spline also renders as its Fourier series cut at Nyquist, a
partial family with complex weights: `design/spline-spectrum.md`
(`EmitArrow/SplineSpectrum.lean`).

## Not built

- Spline integrals, speed-to-position warps.
- Absolute-duration envelopes (`w = duration × frequency`, a width set in
  seconds rather than as a fraction of the gate's cycle), and tails that
  overlap the next cycle. Width here is relative only.
- A spline whose two ends differ. The retired phantom-end boundary let
  `c[0]` and `c[K − 1]` be the values at phase 0 and phase 1 (an envelope
  from 1 down to 0), and also gave the two ends independent slopes (a steep
  onset and a flat finish). Every spline is now periodic, so an envelope
  starts and ends at the same value with the same slope.
- Sample playback, table ports.
- Editors.
- A playground node kind. Playground port tables are static per kind
  (`Playground/VocabularyMetadata.lean`), so a node whose knob count is the
  structural `K` does not fit, and no playground node emits a phase signal
  to read the spline with.
