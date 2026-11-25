import Tropical.EmitArrow.SplineSpectrum

/-!
# EmitArrow.SplineOsc — one spline oscillator, alias-bounded at every pitch

A periodic uniform cubic spline read by a clock, rendered as the time-domain
cubic (`EmitArrow.Spline`) at low fundamentals and as its harmonics below
Nyquist (`EmitArrow.SplineSpectrum`) at high ones, with a crossfade over a
band of fundamentals between them. `design/spline-osc.md` has the reasoning.

* **The budget** bounds the error power (output minus the ideal band-limited
  curve: every harmonic strictly below Nyquist) relative to the in-band
  power, for every control set and every fundamental, rational ones
  included, where aliases land on harmonics and add as amplitudes.
* **The bound.** With `M = ⌈Nyquist/|f0|⌉ − 1` harmonics in band and
  `A_n = 2|F_n|`, the cubic's error is at most `Σ_{n>M} A_n` at every sample,
  so its power over any window is at most `(Σ_{n>M} A_n)²`; the in-band power
  is `Σ_{n≤M} A_n²/2`. Harmonic `n` is `C[n mod K]·φ̂(n/K)/K`, so by
  Cauchy–Schwarz over the classes `n ≡ ±m` the ratio is at most
  `B(M) = 2·Σ_m a_m(M)²/b_m(M)`, `a_m = Σ_{n>M, n≡±m} |φ̂(n/K)|`,
  `b_m = Σ_{n≤M, n≡±m} φ̂(n/K)²`, for every control set.
* **The band** `[Nyquist/capacity, Nyquist/edge]`. The bank is exact above its
  bottom edge, so the mix's error is `(1 − w)·(cubic error)` and its ratio is
  at most `(1 − w)²·B(M)`. `capacity` is the least `c` with `B(c − 1)`
  within budget (the bottom edge, where `w = 0`); `edge` is the least `e`
  for which `(1 − w)²·B(m)` is within budget on every `m`-interval of the
  band, checked at each interval's worst point. Every quantity is exact
  carrier or f64 `+ − × ÷`; no libm.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- `p^{(k)}(t)` at `t = 0` or `1` for the coefficient row `a` (`p = Σ a_j t^j`). -/
private def derivAt (a : Array Int) (k : Nat) (atOne : Bool) : Int := Id.run do
  let mut acc : Int := 0
  for j in [k:a.size] do
    -- j!/(j−k)!
    let falling := (List.range k).foldl (fun f i => f * ((j - i : Nat) : Int)) 1
    if atOne then acc := acc + falling * a[j]!
    else if j == k then acc := acc + falling * a[j]!
  return acc

/-- The jumps `J_k(x)·scale` of the kernel's `k`-th derivative at its knots
    `x = −2 … 2` (`k = 0 … 3`), from the matrix: the piece on `[1−w, 2−w)` is
    `p_w`. -/
def SplineBasis.jumps (basis : SplineBasis) : Array (Int × Array Int) := Id.run do
  let p := basis.pieces
  let mut out := #[]
  for xi in [0:5] do
    let x : Int := (xi : Int) - 2
    let right := 1 - x  -- the piece starting at x
    let left := 2 - x   -- the piece ending at x
    let js := (Array.range 4).map fun k =>
      (if 0 ≤ right && right ≤ 3 then derivAt p[right.toNat]! k false else 0) -
      (if 0 ≤ left && left ≤ 3 then derivAt p[left.toNat]! k true else 0)
    out := out.push (x, js)
  return out

/-- `|φ̂(n/K)|`, bounded above, from the jumps: a piecewise cubic has
    `φ̂(ξ) = Σ_x Σ_k J_k(x)·e^{−iωx}/(iω)^{k+1}`, `ω = 2πξ`, exactly. Returned
    as `|Re| + |Im|` of the sum (no square root). Accurate where the terms do
    not cancel, `n ≥ 4K`. -/
def jumpTransformAbs (basis : SplineBasis) (roots : Array (Float × Float))
    (twoPi : Float) (k n : Nat) : Float := Id.run do
  let omega := twoPi * n.toFloat / k.toFloat
  let scale := basis.scale.toFloat
  let mut re := 0.0
  let mut im := 0.0
  for (x, js) in basis.jumps do
    -- e^{−iωx}: the root at −n·x mod K
    let r := ((-(n : Int) * x) % (k : Int)).toNat
    let (cr, ci) := roots[r]!
    let mut inv := 1.0 / omega
    for kk in [0:4] do
      let j := Float.ofInt js[kk]! / scale
      -- 1/(i)^{kk+1}: −i, −1, i, 1
      let (ur, ui) := match kk % 4 with
        | 0 => (0.0, -1.0) | 1 => (-1.0, 0.0) | 2 => (0.0, 1.0) | _ => (1.0, 0.0)
      let tr := j * inv * ur
      let ti := j * inv * ui
      re := re + (cr * tr - ci * ti)
      im := im + (cr * ti + ci * tr)
      inv := inv / omega
  return re.abs + im.abs

/-- `jumpTransformAbs` with its tables built here (for the gates). -/
def SplineBasis.jumpTransformAbsAt (basis : SplineBasis) (k n : Nat) : Float :=
  jumpTransformAbs basis ((Array.range k).map fun r => unitRoot r k)
    (Tropical.Exact.DyadicI.toFloat Tropical.Exact.twoPiI) k n

/-- `x^n` by multiplication. -/
private def ipow (x : Float) (n : Nat) : Float := (List.replicate n x).foldl (· * ·) 1.0

/-- `Σ_x |J_k(x)|` for `k = 0 … 3`, over `scale`. -/
private def jumpMass (basis : SplineBasis) : Array Float :=
  (Array.range 4).map fun kk =>
    basis.jumps.foldl (fun acc (_, js) => acc + (Float.ofInt js[kk]!).abs) 0.0 /
      basis.scale.toFloat

/-- `B(M)` for `M = 1 … mMax`: the coherent worst-case error ratio of the
    cubic with `M` harmonics in band (module doc), ∞ while a class has no
    harmonic in band. `φ̂` is `kernelTransform` (exact carrier) for `n ≤ 4K`,
    the jump formula to `n = N = 1024·K`, then the tail
    `Σ_k S_k (K/6)^{k+1}/(k·N^k)` (`2π > 6`), added to every class. Inflated
    by `1 + 2⁻³⁰` for the f64 rounding of the sums. -/
def SplineBasis.coherentAliasBounds (basis : SplineBasis) (k mMax : Nat) : Array Float := Id.run do
  let classes := k / 2
  let big := 1024 * k
  let roots := (Array.range k).map fun r => unitRoot r k
  let twoPi := Tropical.Exact.DyadicI.toFloat Tropical.Exact.twoPiI
  let classOf := fun (n : Nat) => let r := n % k; min r (k - r)
  let absAt := fun (n : Nat) =>
    if n % k == 0 then 0.0  -- exact zero: the basis reproduces constants
    else if n ≤ 4 * k then (basis.kernelTransform n k).abs
    else jumpTransformAbs basis roots twoPi k n
  let mass := jumpMass basis
  let bigF := big.toFloat
  let kf := k.toFloat / 6.0
  let tail := (List.range 3).foldl (fun acc i =>
    let kk := i + 1
    acc + mass[kk]! * ipow kf (kk + 1) / (kk.toFloat * ipow bigF kk)) 0.0
  -- class totals of |φ̂| to N
  let mut absTotal := Array.replicate (classes + 1) 0.0
  let mut absVals := Array.replicate (mMax + 1) 0.0
  for n in [1:big + 1] do
    let v := absAt n
    absTotal := absTotal.modify (classOf n) (· + v)
    if n ≤ mMax then absVals := absVals.set! n v
  let mut inBandAbs := Array.replicate (classes + 1) 0.0
  let mut inBandSq := Array.replicate (classes + 1) 0.0
  let mut out := #[]
  for mm in [1:mMax + 1] do
    let c := classOf mm
    let v := absVals[mm]!
    inBandAbs := inBandAbs.modify c (· + v)
    inBandSq := inBandSq.modify c (· + v * v)
    let mut sum := 0.0
    for i in [1:classes + 1] do
      let a := max 0.0 (absTotal[i]! - inBandAbs[i]!) + tail
      let b := inBandSq[i]!
      sum := sum + (if b > 0.0 then a * a / b else (1.0 / 0.0))
    out := out.push (2.0 * sum * (1.0 + 1.0 / 1073741824.0))
  return out

/-- The crossfade band of a spline oscillator: the cubic plays alone while
    `|f0| ≤ Nyquist/capacity`, the bank alone from `Nyquist/edge` up. -/
structure SplineOscBand where
  /-- The band's top: `Nyquist/edge`. -/
  edge : Nat
  /-- The bank's partial count; the band's bottom is `Nyquist/capacity`. -/
  capacity : Nat
  /-- The budget, in dB below the in-band power. -/
  budgetDb : Nat
  /-- The proved worst-case error ratio over the whole range (the largest
      `(1 − w)²·B(m)` over the band and below it). -/
  worstBound : Float
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

/-- `(1 − w)²` at the worst point of the `m`-interval of a band `(e, c)`:
    `f0 = Nyquist/(m + 1)`, where `t = (c − m − 1)·e / ((m + 1)(c − e))`. -/
def bandCubicWeightSq (e c m : Nat) : Float :=
  let num := ((c - m - 1) * e).toFloat
  let den := ((m + 1) * (c - e)).toFloat
  let t := max 0.0 (min 1.0 (num / den))
  let w := t * t * (3.0 - 2.0 * t)
  (1.0 - w) * (1.0 - w)

/-- The deepest budget the band derivation resolves (the tails are f64
    differences of class sums). -/
def splineOscBudgetMaxDb : Nat := 120

/-- The band for `basis`, `K` controls and a budget of `budgetDb` dB below the
    in-band power, from the coherent bound (module doc). `capacity` defaults
    to the least that meets the budget at the bottom edge; `edge` is then the
    widest band that meets it everywhere. Refused for fewer than 2 controls
    (one is a constant), a basis without an even, continuous kernel that
    reproduces constants, a budget of 0 dB or past `splineOscBudgetMaxDb`, a
    budget the phase rail's capacity cannot meet, or a capacity below the
    least. -/
def splineOscBand (basis : SplineBasis) (k budgetDb : Nat)
    (capacity? : Option Nat := none) : Except String SplineOscBand := do
  if k < 2 then
    throw s!"spline oscillator: {k} control values; one value is a constant curve with no harmonics, so give at least 2"
  unless basis.kernelIsEven do
    throw s!"spline oscillator: the {basis.name} basis has no even kernel"
  unless basis.reproducesConstants do
    throw s!"spline oscillator: the {basis.name} basis does not reproduce constants, so harmonics at multiples of K carry the mean"
  unless basis.jumps.all (·.2[0]! == 0) do
    throw s!"spline oscillator: the {basis.name} kernel is not continuous, so its alias does not converge"
  if budgetDb == 0 then
    throw "spline oscillator: a 0 dB budget bounds nothing; declare the error budget in dB below the signal (e.g. 60)"
  if budgetDb > splineOscBudgetMaxDb then
    throw s!"spline oscillator: a {budgetDb} dB budget is past what the bound resolves ({splineOscBudgetMaxDb} dB)"
  let mMax := familyCapacity - 1
  let mut window := 64
  let mut bounds : Array Float := #[]
  let mut least : Option Nat := none
  while least.isNone do
    bounds := basis.coherentAliasBounds k (min window mMax)
    least := (bounds.findIdx? (withinDb budgetDb)).map (· + 1)
    if least.isNone then
      if window ≥ mMax then
        throw s!"spline oscillator ({basis.name}, K = {k}): no band meets a {budgetDb} dB budget within the phase rail's {familyCapacity} partials"
      window := window * 4
  let mStar := least.get!
  let capacity := capacity?.getD (mStar + 1)
  if capacity ≤ mStar then
    throw s!"spline oscillator ({basis.name}, K = {k}): no band meets a {budgetDb} dB budget with {capacity} partials; the cubic meets it only with at least {mStar} harmonics in band, so give at least {mStar + 1} partials"
  if capacity > familyCapacity then
    throw s!"spline oscillator: {capacity} partials exceed the phase rail's capacity of {familyCapacity}"
  if bounds.size < capacity then
    bounds := basis.coherentAliasBounds k capacity
  let bAt := fun (m : Nat) => bounds[m - 1]!
  -- the widest band (least e) meeting the budget on every interval
  let mut edge := capacity - 1
  let mut worst := bAt (capacity - 1)
  for e in [1:capacity] do
    let mut ok := true
    let mut w := 0.0
    for m in [e:capacity] do
      let r := bandCubicWeightSq e capacity m * bAt m
      w := max w r
      unless withinDb budgetDb r do ok := false
    if ok then
      edge := e
      worst := w
      break
  pure { edge, capacity, budgetDb, worstBound := worst }

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
