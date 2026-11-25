import Tropical.EmitArrow.SplineSpectrum

/-!
# EmitArrow.SplineOsc — one spline oscillator, alias-bounded at every pitch

A periodic uniform cubic spline read by a clock, rendered as the time-domain
cubic (`EmitArrow.Spline`) at low fundamentals and as its harmonics below
Nyquist (`EmitArrow.SplineSpectrum`) at high ones, with a crossfade over a
band of fundamentals between them. `design/spline-osc.md` has the reasoning.

* **The budget** is a worst-case error power relative to the in-band signal
  power, declared in dB. It holds for every choice of control values: the
  harmonics of the spline split into classes `n ≡ ±m (mod K)`, each scaled by
  one `|C[m]|²`, so the worst case over the controls is the worst class.
* **The band** is derived from the budget at assemble time. The cubic's
  aliased power in class `m` with `M` harmonics below Nyquist is the class's
  total `Σ_{j∈ℤ} φ̂(m/K + j)²` (closed form by Poisson summation over the
  kernel's autocorrelation, `r_0 + 2Σ_k r_k cos(2πkm/K)`) minus its first
  `M` terms. `edge` is the least `M` at which every class meets the budget,
  so the cubic meets it for `|f0| ≤ Nyquist/edge`. The bank of `capacity`
  partials holds every harmonic below Nyquist for `|f0| ≥ Nyquist/capacity`.
  The crossfade runs between the two: weight `w = smoothstep(t)`,
  `t = (|f0|·capacity/Nyquist − 1)·edge/(capacity − edge)`, uniform within a
  call. The default capacity is `2·edge` (an octave band).
* **Cost.** Below the band the bank's live trip count is 0, so its region
  runs no iteration. Above it the cubic is still evaluated (branchless; its
  weight is 0).
* **Phase.** The cubic reads the family's own fundamental phase (the same
  nodes), so both realizations render the same waveform at the same phase.

The bank's Nyquist edge is a hard cut (`nyquist := .off`, with the trip count
the harmonics strictly below Nyquist), so above the band the output is the
ideal band-limited curve up to rounding.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- The kernel's autocorrelation at the integers, `r_k = ∫ φ(u)·φ(u + k) du`
    for `k = 0..3` (zero beyond). Each is an exact integer over
    `420·scale²` (420 = lcm(1..7) clears every `∫₀¹ t^j dt`), divided once. -/
def SplineBasis.autocorrelation (basis : SplineBasis) : Array Float := Id.run do
  let p := basis.pieces
  let mut out := #[]
  for k in [0:4] do
    let mut num : Int := 0
    for w in [k:4] do
      for a in [0:4] do
        for b in [0:4] do
          num := num + p[w]![a]! * p[w - k]![b]! * ((420 / (a + b + 1) : Nat) : Int)
    out := out.push (Float.ofInt num / (420.0 * (basis.scale * basis.scale).toFloat))
  return out

/-- The total power of harmonic class `m` (positive harmonics `n ≡ ±m mod K`)
    in units of `φ̂²`: `Σ_{j∈ℤ} φ̂(m/K + j)² = r_0 + 2Σ_k r_k cos(2πkm/K)`,
    halved at `m = K/2`, where `n ≡ m` and `n ≡ −m` are one class. -/
def SplineBasis.classPower (basis : SplineBasis) (k m : Nat) : Float :=
  let r := basis.autocorrelation
  let s := (Array.range 3).foldl (fun acc i =>
    acc + 2.0 * r[i + 1]! * (unitRoot ((i + 1) * m) k).1) r[0]!
  if 2 * m == k then s / 2.0 else s

/-- The worst-case aliased-power ratio of the time-domain cubic with `M`
    harmonics below Nyquist, for every `M = 1..mMax`: entry `M − 1` is
    `max_m (total_m − in_m(M)) / in_m(M)` over the classes `m = 1..⌊K/2⌋`
    (∞ while a class has no harmonic in band). `φ̂` is `kernelTransform`, the
    exact-carrier transform. -/
def SplineBasis.aliasRatios (basis : SplineBasis) (k mMax : Nat) : Array Float := Id.run do
  let classes := k / 2
  let totals := (Array.range classes).map fun i => basis.classPower k (i + 1)
  let mut inBand := Array.replicate classes 0.0
  let mut out := #[]
  for mm in [1:mMax + 1] do
    let r := mm % k
    let cls := if r == 0 then 0 else min r (k - r)
    if cls != 0 then
      let v := basis.kernelTransform mm k
      inBand := inBand.modify (cls - 1) (· + v * v)
    let mut worst := 0.0
    for i in [0:classes] do
      let inb := inBand[i]!
      let tail := max 0.0 (totals[i]! - inb)
      worst := max worst (if inb > 0.0 then tail / inb else (1.0 / 0.0))
    out := out.push worst
  return out

/-- The crossfade band of a spline oscillator: the cubic plays alone while
    `|f0| ≤ Nyquist/capacity`, the bank alone from `Nyquist/edge` up. -/
structure SplineOscBand where
  /-- The least harmonic count below Nyquist at which the cubic's worst-case
      alias meets the budget. -/
  edge : Nat
  /-- The bank's partial count. -/
  capacity : Nat
  /-- The budget, in dB below the in-band power. -/
  budgetDb : Nat
  /-- The cubic's worst-case alias ratio at `edge` (the band's top). -/
  worstAlias : Float
deriving Repr, Inhabited

/-- `10^d` by repeated multiplication (no libm; exact through `10²²`). -/
def powTen (d : Nat) : Float := (List.replicate d 10.0).foldl (· * ·) 1.0

/-- Is the power ratio `r` at least `db` dB down, `r ≤ 10^(−db/10)`? Decided
    as `r¹⁰ · 10^db ≤ 1` with multiplications only, so no libm `pow` or
    `log` touches the band. -/
def withinDb (db : Nat) (r : Float) : Bool :=
  let r2 := r * r
  let r8 := (r2 * r2) * (r2 * r2)
  r ≥ 0.0 && r8 * r2 * powTen db ≤ 1.0

/-- The deepest budget the band derivation resolves. The aliased power of a
    class is its closed-form total minus its in-band terms, in f64; at
    10⁻¹² of the total that difference is still 100× the rounding of the
    sums (a few hundred terms, each within 2⁻⁵³). -/
def splineOscBudgetMaxDb : Nat := 120

/-- The band for `basis`, `K` controls and a budget of `budgetDb` dB below the
    in-band power, from the closed-form alias. `capacity` defaults to
    `2·edge`. Refused for a basis without an even kernel, a budget of 0 dB, a
    budget whose edge does not fit the phase rail, or a capacity that leaves
    no band (`capacity ≤ edge`: the bank would have to start where the cubic
    already misses the budget). -/
def splineOscBand (basis : SplineBasis) (k budgetDb : Nat)
    (capacity? : Option Nat := none) : Except String SplineOscBand := do
  if k == 0 then
    throw "spline oscillator: no control values"
  unless basis.kernelIsEven do
    throw s!"spline oscillator: the {basis.name} basis has no even kernel"
  if budgetDb == 0 then
    throw "spline oscillator: a 0 dB budget bounds nothing; declare the error budget in dB below the signal (e.g. 60)"
  if budgetDb > splineOscBudgetMaxDb then
    throw s!"spline oscillator: a {budgetDb} dB budget is past what the closed form resolves ({splineOscBudgetMaxDb} dB); the alias is a difference of f64 class sums"
  -- the edge must leave room for a bank of at least edge + 1 partials
  let mMax := familyCapacity - 1
  -- search in doubling windows so a modest budget costs a few transforms
  let mut window := 64
  let mut found : Option (Nat × Float) := none
  while found.isNone do
    let ratios := basis.aliasRatios k (min window mMax)
    found := (ratios.findIdx? (withinDb budgetDb)).map fun i => (i + 1, ratios[i]!)
    if found.isNone then
      if window ≥ mMax then
        throw s!"spline oscillator ({basis.name}, K = {k}): no band meets a {budgetDb} dB budget; the cubic would need more than {mMax} harmonics below Nyquist, past the phase rail's {familyCapacity} partials"
      window := window * 4
  let (edge, worst) := found.get!
  let capacity := capacity?.getD (2 * edge)
  if capacity ≤ edge then
    throw s!"spline oscillator ({basis.name}, K = {k}): no band meets a {budgetDb} dB budget with {capacity} partials; the cubic meets it only up to Nyquist/{edge}, and the bank covers only down to Nyquist/{capacity}; give at least {edge + 1} partials"
  if capacity > familyCapacity then
    throw s!"spline oscillator: {capacity} partials exceed the phase rail's capacity of {familyCapacity}"
  pure { edge, capacity, budgetDb, worstAlias := worst }

/-- What the oscillator emits, with its parts for the gates: the mixed output,
    the two realizations, the crossfade weight `w` (bank) and the bank's live
    trip count (a float before the family's clamp). -/
structure SplineOscParts where
  out : Sig
  cubic : Sig
  bank : Sig
  weight : Sig
  trip : Sig

/-- `t²(3 − 2t)` for `t = clamp(v, 0, 1)`. -/
private def smoothstep01 (v : Sig) : BuildM Sig := do
  let t ← clampE v (← lit 0) (← lit 1)
  mul (← mul t t) (← sub (← lit 3) (← mul (← lit 2) t))

/-- The fundamental's Q0.32 phase as the partial family computes it
    (`familySig`: the same nodes, so the interned graph shares them), for the
    cubic to read. -/
private def familyPhase (freq : Sig) (phase : Option Sig) (clock : Clock) :
    BuildM Sig := do
  let twoPow32 ← lit 4294967296
  let inc ← toIntE (← div (← mul freq twoPow32) (← sampleRate))
  let fracBits ← clockFracLit
  let thi ← rshift clock fracBits
  let tlo ← bitAnd clock (← clockFracMaskLit)
  let mask32 ← lit 4294967295
  let turns ← mul inc thi
  let turns ← match phase with
    | some ph => add turns (← toIntE (← mul ph twoPow32))
    | none => pure turns
  let wrapped ← bitAnd turns mask32
  let p ← bitAnd (← add (← lshift wrapped fracBits) (← mul inc tlo))
    (← lit (2 ^ (32 + clockFracBits) - 1))
  rshift p fracBits

/-- The spline oscillator on `clock`: the periodic spline of `controls` under
    `basis`, at the uniform fundamental `freq` (Hz, either sign) with the
    uniform phase offset `phase` (cycles). Refused when `freq` or `phase` is
    not uniform (the family's rule) or the band does not fit the controls. -/
def splineOscSig (basis : SplineBasis) (band : SplineOscBand) (controls : Array Sig)
    (freq : Sig) (clock : Clock) (phase : Option Sig := none) :
    BuildM SplineOscParts := do
  unless ← sigIsUniform freq do
    throw "spline oscillator: the fundamental must be uniform (constant within a kernel call); the crossover weight and the bank's trip count are per call"
  let cap := band.capacity
  let edge := band.edge
  let nyquist ← mul (← sampleRate) (← lit 5 1)
  let f0abs ← absE freq
  -- t = (|f0|·cap/Nyquist − 1)·edge/(cap − edge): 0 at Nyquist/cap, 1 at Nyquist/edge
  let u ← div (← mul f0abs (← lit cap)) nyquist
  let t ← div (← mul (← sub u (← lit 1)) (← lit edge)) (← lit (cap - edge))
  let w ← smoothstep01 t
  -- the harmonics strictly below Nyquist, ⌈Nyquist/|f0|⌉ − 1, while w > 0
  let fFloor ← div nyquist (← lit cap)
  let fSafe ← selectE (← gt f0abs fFloor) f0abs fFloor
  let below ← sub (← unary .ceil (← div nyquist fSafe)) (← lit 1)
  let trip ← selectE (← gt t (← lit 0)) below (← lit 0)
  let fam ← splineSpectrumFamily basis controls freq clock cap phase
  let fam := { fam with
    name := s!"spline oscillator ({basis.name}, K = {controls.size})"
    edges := { dc := .off, nyquist := .off }
    liveCount := some trip }
  let bank ← familySig fam clock #[]
  let cubic ← splineSig basis controls (← familyPhase freq phase clock)
  let out ← add (← mul (← sub (← lit 1) w) cubic) (← mul w bank)
  pure { out, cubic, bank, weight := w, trip }

end Tropical.EmitArrow
