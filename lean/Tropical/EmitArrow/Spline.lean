import Tropical.EmitArrow.Sig

/-!
# EmitArrow.Spline — a uniform cubic spline read by a phase

A spline here is a shape read by a phase: a function of one phase coordinate,
not a parametric curve. Every spline is periodic. The same object is a
single-cycle waveform at audio rate, an LFO at a low rate, or an envelope when
the phase that reads it stops at the end of the cycle (`EmitArrow/PhaseClip.lean`):
phase 1 and phase 0 are the same point, so an envelope ends where it starts.
The spline does not own a clock; its phase is any integer phase signal (Q0.32,
one cycle = 2³²), such as the DDS phase every exact oscillator computes.

* **Knots** are uniform over one cycle, so the segment is found by arithmetic,
  with no search.
* **K control values** are signals. Wired from `param:` slots they are live: a
  control move is a parameter write, not a recompile. K is structural.
* **The basis** is structural: a 4×4 integer matrix and a divisor, plain
  host data (`SplineBasis`). Catmull-Rom and the uniform cubic B-spline are
  provided.

## Index arithmetic

With `n = K` segments over the cycle and `p` the phase reduced to
`[0, 2³²)`, `s = p · n` is exact in `int` (`n < 2³¹`), and

    i = s >> 32,   t = (s & (2³² − 1)) / 2³²

are the segment and the local coordinate. `i` is exact on every target; `t`
is exact in f64 (a 32-bit integer over a power of two). When `n` is a power
of two, `p · n` is a shift and `i` is the top bits of the phase. Any other `n`
costs one integer multiply; the result is the same split. In an f32 artifact
the 32-bit fraction rounds to 24 bits, so `t` can round up to 1.0, which is
harmless: at `t = 1` a segment equals the next one at `t = 0`.

## The window

Segment `i` reads four consecutive entries of an extended table `T` of
`K + 3` signals, `T[i] … T[i + 3]`. The extension is built here, at assemble
time, so the kernel does no index wrapping or clamping:

`T[j] = c[(j − 1) mod K]`. The control values on either side of the seam are
the same signals as anywhere else, so phase 1 → 0 is an ordinary knot by
construction.

`i ∈ [0, K)` holds for every phase by construction, so every table read is in
range. The `index` node still carries its generic bounds check; this module
does not change it.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- A uniform cubic basis: on a segment with local coordinate `t ∈ [0, 1)` and
    window values `P₀ … P₃`, the spline is
    `(Σₖ tᵏ · Σⱼ rows[k][j] · Pⱼ) / scale`. Row `k` is the coefficient of `tᵏ`. -/
structure SplineBasis where
  name : String
  scale : Nat
  rows : Array (Array Int)
deriving Repr, Inhabited

/-- Catmull-Rom (tension ½): interpolating, C¹. -/
def catmullRomBasis : SplineBasis where
  name := "Catmull-Rom"
  scale := 2
  rows := #[#[0, 2, 0, 0], #[-1, 0, 1, 0], #[2, -5, 4, -1], #[-1, 3, -3, 1]]

/-- The uniform cubic B-spline: approximating, C². -/
def uniformBSplineBasis : SplineBasis where
  name := "uniform cubic B-spline"
  scale := 6
  rows := #[#[1, 4, 1, 0], #[-3, 0, 3, 0], #[3, -6, 3, 0], #[-1, 3, -3, 1]]

/-- The extended table `T` of `K + 3` signals (see the module doc). -/
def splineTable (controls : Array Sig) : Array Sig :=
  let k := controls.size
  (Array.range (k + 3)).map fun j => controls[(j + k - 1) % k]!

/-- `Σⱼ row[j] · window[j]`, skipping zero coefficients and multiplying by
    no ±1. -/
private def splineCombo (row : Array Int) (window : Array Sig) : BuildM Sig := do
  let mut acc : Option Sig := none
  for j in [0:4] do
    let m := row[j]!
    if m == 0 then continue
    let magnitude ←
      if m.natAbs == 1 then pure window[j]!
      else mul (← lit (Int.ofNat m.natAbs)) window[j]!
    acc ← match acc with
      | none => if m > 0 then pure (some magnitude) else some <$> neg magnitude
      | some a => some <$> (if m > 0 then add a magnitude else sub a magnitude)
  match acc with
  | some a => pure a
  | none => lit 0

/-- The periodic spline of `controls` under `basis`, read at the integer phase
    `phaseQ` (Q0.32: one cycle is 2³², taken modulo the cycle). Refused at
    build time for no control values, too many (`K ≥ 2³¹` would overflow
    `p · K`), or a basis that is not 4×4. -/
def splineSig (basis : SplineBasis) (controls : Array Sig) (phaseQ : Sig) :
    BuildM Sig := do
  let k := controls.size
  if k == 0 then
    throw "spline: no control values"
  let segments := k
  if segments ≥ 2 ^ 31 then
    throw s!"spline: {k} control values; at most 2^31 segments fit the phase product"
  if basis.rows.size != 4 || basis.rows.any (·.size != 4) || basis.scale == 0 then
    throw s!"spline: the {basis.name} basis is not a 4×4 matrix with a positive divisor"
  let mask ← lit 4294967295
  let p ← bitAnd phaseQ mask
  let s ← mul p (← lit segments)
  let i ← rshift s (← lit 32)
  let t ← div (← toFloatE (← bitAnd s mask)) (← lit 4294967296)
  let table ← arr (splineTable controls)
  let window ← (Array.range 4).mapM fun (j : Nat) => do
    if j == 0 then index table i else index table (← add i (← lit (Int.ofNat j)))
  let a ← basis.rows.mapM (splineCombo · window)
  -- Horner in t: ((a₃·t + a₂)·t + a₁)·t + a₀.
  let h3 ← add (← mul a[3]! t) a[2]!
  let h2 ← add (← mul h3 t) a[1]!
  let h1 ← add (← mul h2 t) a[0]!
  if basis.scale == 1 then pure h1 else div h1 (← lit basis.scale)

end Tropical.EmitArrow
