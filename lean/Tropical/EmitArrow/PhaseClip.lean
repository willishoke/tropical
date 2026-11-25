import Tropical.EmitArrow.Sig

/-!
# EmitArrow.PhaseClip — the pulse width of a clock, as a phase warp

A gate is a clock with a pulse width. Given a phase `p` (Q0.32, one cycle is
2³²) and a width `w ∈ (0, 1]`, the clipped phase sweeps one full cycle over
the first `w` of the input cycle, then holds at the end of the cycle:

    phaseclip(p, w) = min(p / w, 1)        (in cycles)

It is the unipolar special case of the linear phase clip that Csound's
`pdclip` applies to a phasor for phase-distortion synthesis, with the window
at the start of the cycle and `width` the unclipped fraction (`pdclip`'s
`kWidth` is the clipped one). Read through a periodic spline it is an envelope that always fits its
gate: the read starts at phase 0 and reaches phase 1 exactly when the gate
falls, and since phase 1 and phase 0 are the same point of a periodic spline,
the held value is the start value. The gate itself, high for the first `w` of
the cycle, is the comparison the clip already makes.

## Arithmetic

The phase is reduced to the cycle, `p = phase & (2³² − 1)`. The width is an
integer word `W ∈ [1, 2³²]`, the width in units of 2⁻³² of a cycle. Then

    phaseclip = ⌊p · 2³² / W⌋   if p < W,     2³²   otherwise
    gate   = p < W

The quotient is exact on the integer rail: `p · 2³²` does not fit an `int`,
so it is two steps of long division in base 2¹⁶ (`x₁ = p · 2¹⁶ < 2⁴⁸`,
`q₁ = ⌊x₁ / W⌋`, `r₁ = x₁ − q₁W < W`, `q₂ = ⌊r₁ · 2¹⁶ / W⌋`, result
`q₁ · 2¹⁶ + q₂`), every divide between non-negative integers. In the lane
the final select keeps (`p < W`) every operand is below 2⁴⁹; in the lane it
discards (`p ≥ W`, e.g. `W = 1`, `p = 2³² − 1`) `q₁ · 2¹⁶` can wrap, which
is defined (two's-complement) and never read. The held value 2³² is phase 1,
which a periodic spline reads as phase 0 (`splineSig` masks its phase), bit
for bit; so does anything else that reduces its phase mod 2³², including a
second `PhaseClip`, whose gate is then high for the whole of the first one's
hold.

The width word is where rounding can happen, and only there:

* a static width (`PulseWidth.static`, a decimal) is `round(w · 2³²)` in
  exact integer arithmetic at assemble time. It is refused outside
  `[2⁻³³, 1]` (`pulseWidthInterval`): at or below 0, above 1, or narrower
  than half a phase unit, where the word would round to 0 and divide by
  zero;
* a live width (`PulseWidth.live`, a float signal) is
  `clamp(toInt(round(w · 2³²)), 1, 2³²)` per sample. `w · 2³²` is exact in
  f64 and f32 (a power-of-two scale), and `round` is half away from zero, so
  `W` is within ½ of `w · 2³²`; a moving width outside `(0, 1]` clamps to
  the narrowest (`W = 1`) or the full cycle (`W = 2³²`, the identity).

So the clipped phase is exact for every width that is a multiple of 2⁻³²
(every dyadic knob value: ½, ¼, 0.375), and otherwise differs from the
real `p / w` by less than `1 + 1/(2w)` phase units (2⁻³² of a cycle; the 1
is the floor). The gate edge `p < W` is likewise exact against `W`. In an
f32 artifact the width itself is binary32, so `W` keeps its 24 significant
bits; the integer path is the same.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- The interval a static pulse width must lie in, `[2⁻³³, 1]`: the widths in
    `(0, 1]` whose word `round(w · 2³²)` is at least 1. A narrower one would
    be less than half a phase unit and its word 0. -/
def pulseWidthInterval : Tropical.PortInterval :=
  { lo := ⟨116415321826934814453125, 33⟩, loOpen := false, hi := 1, hiOpen := false }

/-- A pulse width: a decimal fixed at assemble time, or a float signal. -/
inductive PulseWidth where
  | static (w : Lean.JsonNumber)
  | live (w : Sig)

/-- `round(w · 2³²)` for a non-negative decimal, half away from zero, in exact
    integer arithmetic. -/
def widthWord (w : Lean.JsonNumber) : Int :=
  let num : Int := w.mantissa * 4294967296
  let den : Int := (10 : Int) ^ w.exponent
  (2 * num + den) / (2 * den)

/-- The clipped phase (an `int`, `[0, 2³²]`) and the gate (a float, 1 for the
    first `w` of the cycle, else 0). -/
structure PhaseClipOut where
  phase : Sig
  gate : Sig

/-- `phaseclip(phaseQ, width)` and its gate (see the module doc). A static width
    outside `[2⁻³³, 1]` is refused here. -/
def phaseClipSig (phaseQ : Sig) (width : PulseWidth) : BuildM PhaseClipOut := do
  let cycle ← litI 4294967296
  let word ← match width with
    | .static w =>
      if pulseWidthInterval.contains w then litI (widthWord w)
      else throw s!"phaseclip: width {w} is outside [2^-33, 1]: at or below 0, above 1, or narrower than half a phase unit (its word would round to 0)"
    | .live w =>
      let raw ← toIntE (← roundE (← mul w (← lit 4294967296)))
      clamp raw (← litI 1) cycle
  let p ← bitAnd phaseQ (← lit 4294967295)
  let shift16 ← lit 16
  let x1 ← lshift p shift16
  let q1 ← binary .floorDiv x1 word
  let r1 ← sub x1 (← mul q1 word)
  let q2 ← binary .floorDiv (← lshift r1 shift16) word
  let q ← add (← lshift q1 shift16) q2
  let high ← binary .lt p word
  let phase ← select high q cycle
  let gate ← toFloatE (← select high (← litI 1) (← litI 0))
  pure { phase, gate }

end Tropical.EmitArrow
