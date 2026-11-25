import Tropical.EmitArrow.Spline
import Tropical.EmitArrow.Spectral
import Tropical.EmitArrow.Term
import Tropical.Exact

/-!
# EmitArrow.SplineSpectrum — a periodic spline as its harmonics below Nyquist

The band-limited realization of a periodic uniform cubic spline
(`EmitArrow.Spline`): the same curve, rendered as a partial
family (`EmitArrow.Spectral`) instead of as a piecewise cubic. It is alias-free
and takes the per-partial envelope `G(f, τ)` a harmonic family takes.
`design/spline-spectrum.md` has the derivation, the alias bound and the cost.

With `K` control values `c_j` and the basis kernel `φ` (the curve is
`f(x) = Σ_j c_j φ(Kx − j)`, periodised, `x` the phase in cycles), harmonic `n`
of the curve is

    F_n = (1/K) · C[n mod K] · φ̂(n/K),   C[m] = Σ_j c_j e^{−2πi·jm/K},

with `φ̂(ξ) = ∫ φ(u) e^{−2πiuξ} du`. So `f(x) = F_0 + Σ_{n≥1} Re(2F_n e^{2πinx})`:

* the partial weights are complex, `2F_n`: a real data column
  `(2/K)·φ̂(n/K)` times one of `K` complex values `C[m]`, each a fixed linear
  combination of the live controls (`PartialWeights.complex`, a `cosine`
  family);
* the 0 Hz term is `F_0 = C[0]/K`, the mean of the controls (the family's
  `dc` term);
* `φ̂` is computed at assemble time from the basis matrix itself, for any
  basis with an even kernel (both provided bases are even), in the exact
  carrier (`Tropical.Exact`), so no platform `libm` touches the column. For a
  basis that reproduces constants, harmonics `K, 2K, …` are baked as exact
  zeros.
* the realization is alias-free when the controls are uniform within a
  kernel call. Controls modulated at audio rate are evaluated per sample, and
  the sidebands they add are not culled at Nyquist.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Exact

/-- The basis kernel on its four unit pieces: `φ(t + 1 − w) = p_w(t)` for
    `t ∈ [0, 1)` and window position `w = 0..3`. Entry `[w][k]` is the
    coefficient of `tᵏ` in `p_w`, over `scale`. -/
def SplineBasis.pieces (basis : SplineBasis) : Array (Array Int) :=
  (Array.range 4).map fun w => (Array.range 4).map fun k => basis.rows[k]![w]!

/-- `Σ_k a_k (1 − t)^k` as coefficients of `t^k` (exact). -/
private def reflect (a : Array Int) : Array Int := Id.run do
  let mut out := Array.replicate a.size (0 : Int)
  for k in [0:a.size] do
    -- (1 − t)^k = Σ_m C(k, m) (−t)^m
    let mut binom : Int := 1
    for m in [0:k + 1] do
      let sign : Int := if m % 2 == 0 then 1 else -1
      out := out.modify m (· + a[k]! * binom * sign)
      binom := binom * ((k - m : Nat) : Int) / ((m + 1 : Nat) : Int)
  return out

/-- Is the basis kernel even (`φ(−u) = φ(u)`)? Exact, on the integer matrix:
    `p_w(t) = p_{3−w}(1 − t)` for every piece. -/
def SplineBasis.kernelIsEven (basis : SplineBasis) : Bool :=
  let p := basis.pieces
  (Array.range 4).all fun w => p[w]! == reflect p[3 - w]!

/-- Does the basis reproduce constants (`Σ_j φ(u − j) = 1`)? Exact, on the
    integer matrix: the pieces sum to `scale` (`Σ_w rows[0][w] = scale`, and
    `Σ_w rows[k][w] = 0` for `k ≥ 1`). By Poisson summation this is
    `φ̂(m) = 0` at every nonzero integer `m`. Both provided bases do. -/
def SplineBasis.reproducesConstants (basis : SplineBasis) : Bool :=
  (Array.range 4).all fun k =>
    basis.rows[k]!.foldl (· + ·) 0 == (if k == 0 then (basis.scale : Int) else 0)

/-- `∫₀¹ tᵏ e^{−iat} dt` for `k = 0..3`, as `(re, im)` enclosures. Small `|a|`:
    the Taylor series `Σ_r (−ia)^r / (r!·(k + r + 1))` (60 terms; at `|a| < 4`
    the tail is below `4⁶⁰/60! < 10⁻⁴⁵`); otherwise the recursion
    `I_k = (k·I_{k−1} − e^{−ia}) / (ia)`, stable for `|a| > k`. -/
private def momentIntegrals (a : DyadicI) : Array (DyadicI × DyadicI) := Id.run do
  if a.toFloat.abs < 4.0 then
    let mut out := #[]
    for k in [0:4] do
      let mut re := DyadicI.zero
      let mut im := DyadicI.zero
      let mut term := DyadicI.one -- a^r / r!
      for r in [0:60] do
        let v := DyadicI.div term (DyadicI.ofNat (k + r + 1))
        match r % 4 with
        | 0 => re := DyadicI.add re v
        | 1 => im := DyadicI.sub im v
        | 2 => re := DyadicI.sub re v
        | _ => im := DyadicI.add im v
        term := DyadicI.div (DyadicI.mul term a) (DyadicI.ofNat (r + 1))
      out := out.push (re, im)
    return out
  else
    let er := DyadicI.cos a
    let ei := DyadicI.neg (DyadicI.sin a)
    -- (x + iy) / (ia) = (y − ix) / a
    let divIa := fun (x y : DyadicI) => (DyadicI.div y a, DyadicI.neg (DyadicI.div x a))
    let mut out := #[divIa (DyadicI.sub DyadicI.one er) (DyadicI.neg ei)]
    for k in [1:4] do
      let (pr, pi) := out[k - 1]!
      let kk := DyadicI.ofNat k
      out := out.push (divIa (DyadicI.sub (DyadicI.mul kk pr) er)
        (DyadicI.sub (DyadicI.mul kk pi) ei))
    return out

/-- `φ̂(ξ) = ∫ φ(u) e^{−2πiuξ} du` of the basis kernel at `ξ = num/den`, from
    its matrix: `Σ_w e^{−ia(1−w)} Σ_k p_w[k]·I_k(a)` with `a = 2πξ`, in the
    exact carrier (`Tropical.Exact`). Real for an even kernel (the imaginary
    part is dropped). At a nonzero integer `ξ` of a basis that reproduces
    constants the value is exactly 0, and it is returned as 0 (the enclosure's
    midpoint would be a residue near 1e-38). -/
def SplineBasis.kernelTransform (basis : SplineBasis) (num den : Nat) : Float := Id.run do
  if num != 0 && den != 0 && num % den == 0 && basis.reproducesConstants then
    return 0.0
  let a := DyadicI.div (DyadicI.mul twoPiI (DyadicI.ofNat num)) (DyadicI.ofNat den)
  let moments := momentIntegrals a
  let p := basis.pieces
  let mut re := DyadicI.zero
  for w in [0:4] do
    let mut sr := DyadicI.zero
    let mut si := DyadicI.zero
    for k in [0:4] do
      let c := DyadicI.ofInt p[w]![k]!
      sr := DyadicI.add sr (DyadicI.mul c moments[k]!.1)
      si := DyadicI.add si (DyadicI.mul c moments[k]!.2)
    -- × e^{−ia(1−w)}: keep the real part
    let angle := DyadicI.mul a (DyadicI.ofInt (1 - (w : Int)))
    re := DyadicI.add re (DyadicI.add (DyadicI.mul sr (DyadicI.cos angle))
      (DyadicI.mul si (DyadicI.sin angle)))
  return (DyadicI.div re (DyadicI.ofNat basis.scale)).toFloat

/-- `cos` and `sin` of `2π·r/K` for an integer `r`, exact at quarter turns. -/
def unitRoot (r k : Nat) : Float × Float :=
  let r := r % k
  if (4 * r) % k == 0 then
    match 4 * r / k with
    | 0 => (1.0, 0.0) | 1 => (0.0, 1.0) | 2 => (-1.0, 0.0) | _ => (0.0, -1.0)
  else
    let angle := DyadicI.div (DyadicI.mul twoPiI (DyadicI.ofNat r)) (DyadicI.ofNat k)
    ((DyadicI.cos angle).toFloat, (DyadicI.sin angle).toFloat)

/-- `Σ_j m_j · x_j` for build-time coefficients, skipping zeros and
    multiplying by no ±1. -/
private def combo (terms : Array (Float × Sig)) : BuildM Sig := do
  let mut acc : Option Sig := none
  for (m, x) in terms do
    if m == 0.0 then continue
    let magnitude ← if m.abs == 1.0 then pure x else mul (← litExact m.abs) x
    acc ← match acc with
      | none => if m > 0.0 then pure (some magnitude) else some <$> neg magnitude
      | some a => some <$> (if m > 0.0 then add a magnitude else sub a magnitude)
  match acc with
  | some a => pure a
  | none => lit 0

/-- The DFT `C[m] = Σ_j c_j e^{−2πi·jm/K}` of the controls, `m = 0..K−1`, as
    `(Re, Im)` signal arrays. Only `m ≤ K/2` is computed; the rest is the
    conjugate `C[K − m]`, since the controls are real. -/
def controlDft (controls : Array Sig) : BuildM (Array Sig × Array Sig) := do
  let k := controls.size
  let half := k / 2
  let mut re : Array Sig := #[]
  let mut im : Array Sig := #[]
  for m in [0:half + 1] do
    let terms := (Array.range k).map fun j => (unitRoot (j * m) k, controls[j]!)
    re := re.push (← combo (terms.map fun ((c, _), x) => (c, x)))
    im := im.push (← combo (terms.map fun ((_, s), x) => (-s, x)))
  for m in [half + 1:k] do
    re := re.push re[k - m]!
    im := im.push (← neg im[k - m]!)
  pure (re, im)

/-- The band-limited spline as a partial family: partials `n = 1..count` of
    the uniform fundamental `freq` on the clock `clk`, weights `2F_n`, and the
    0 Hz term `F_0`. The curve is the periodic spline of `controls`
    under `basis`, read at the family's fundamental phase. The DC edge is off
    (the low partials are the curve's shape at any rate); the Nyquist edge is
    the family's standard one. Refused for fewer than one control, a count
    beyond the phase rail's capacity, or a basis whose kernel is not even. -/
def splineSpectrumFamily (basis : SplineBasis) (controls : Array Sig) (freq : Sig)
    (clk : Clock) (count : Nat) (phase : Option Sig := none) :
    BuildM PartialFamily := do
  let k := controls.size
  if k == 0 then
    throw "spline spectrum: no control values"
  if basis.rows.size != 4 || basis.rows.any (·.size != 4) || basis.scale == 0 then
    throw s!"spline spectrum: the {basis.name} basis is not a 4×4 matrix with a positive divisor"
  unless basis.kernelIsEven do
    throw s!"spline spectrum: the {basis.name} basis has no even kernel, so its harmonics need a complex φ̂; only even kernels are supported"
  let (cre, cim) ← controlDft controls
  let column ← (Array.range count).mapM fun i =>
    litExact (2.0 / k.toFloat * basis.kernelTransform (i + 1) k)
  let dc ← div cre[0]! (← lit k)
  let modK ← lit k
  let weight := fun (tables : Array Sig) (idx n _kf : Sig) => do
    let m ← if k &&& (k - 1) == 0 then bitAnd n (← lit (k - 1)) else binary .mod n modK
    let s ← index tables[0]! idx
    pure (← mul s (← index tables[1]! m), ← mul s (← index tables[2]! m))
  pure {
    name := s!"spline-spectrum ({basis.name}, K = {k})"
    freq, phase, clk, count
    weights := .complex #[column, cre, cim] weight
    wave := .cosine
    edges := { dc := .off }
    dc := some dc }

/-- The band-limited spline as a term: `freq` and the controls are signal
    terms, evaluated in the family's clock context, so an envelope above it
    multiplies its partials. -/
def splineSpectrum (basis : SplineBasis) (count : Nat) (clk : Clock)
    (freq : ArrowTerm) (controls : Array ArrowTerm) : ArrowTerm :=
  .familyC (fun signals =>
      splineSpectrumFamily basis (signals.extract 1 signals.size) signals[0]! clk count)
    (#[freq] ++ controls)

end Tropical.EmitArrow
