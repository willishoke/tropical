import Tropical.Exact.Const

/-!
# Tropical.Exact.Elementary — the transcendentals, in Lean, over `DyadicI`

`exp`, `log`, `sin`, `cos`, `atan`, `atan2`, `pow` — argument reduction plus a
truncated series, every truncation admitted as an explicit WIDENING of the
enclosure. This is the bake layer's exile of the platform `libm`: the same
purge the runtime already ran (the kernel's own `sinSig`/`expSig` are
polynomials precisely so wasm and the JIT cannot disagree), applied one floor
up, to the constants the kernel is built FROM.

Every function here is total: an argument outside a series' domain returns
`poison`, never a `nan` that keeps flowing.

## How a series is truncated honestly

Each loop accumulates terms `tₙ` until `|tₙ|` is certifiably below the target,
then widens the sum by a bound on the whole remaining tail. Two shapes appear:

* **Alternating with decreasing terms** (`sin`, `cos`, `atan`): the tail is
  bounded by the first omitted term. Widening by `|tₙ|` is already generous.
* **Dominated-ratio** (`exp`, `atanh`): the tail is bounded by a geometric
  series with the ratio the reduction guarantees (`|r|/(n+1) ≤ ½` for `exp`
  after reduction, `s² ≤ 0.03` for `atanh`), so `Σ_{m>n}|tₘ| ≤ |tₙ|`.

In both cases widening by `|tₙ|` — the last term ACCUMULATED — encloses the
truth, and the loop only stops once that is small. The stopping test is a
CERTIFIED comparison on enclosures, so where the loop stops is a deterministic
function of the input, not of a platform's rounding.
-/

namespace Tropical.Exact

namespace DyadicI

/-- Iteration cap for every series here. Reaching it means the argument was
    outside the reduced domain the caller promised — the result poisons rather
    than silently truncating. -/
private def seriesCap : Nat := 4096

/-- The magnitude below which a term stops mattering: `2^{−(workingPrec+8)}`
    relative to the accumulated sum. -/
private def tailTarget (s : DyadicI) : DyadicI :=
  let eps : DyadicI := exact (Dyadic.ofIntWithPrec 1 ((workingPrec + 8 : Nat) : Int))
  mul (max (abs s) one) eps

/-- `e^x`. Reduces `x = k·ln2 + r` with `|r| ≤ ln2/2 ≈ 0.347`, sums the Taylor
    series in `r`, and scales by the EXACT power of two `2^k`. The reduction
    quotient `k` is taken from the enclosure's midpoint — a deterministic
    choice, and any `k` gives a valid reduction (a neighbouring one only makes
    `|r|` slightly larger), so this is an overlap switch with no cliff. -/
def exp (x : DyadicI) : DyadicI :=
  if !x.ok then poison
  else Id.run do
    let k := roundToInt (div x ln2I)
    let r := sub x (mul (ofInt k) ln2I)
    -- Taylor: t₀ = 1, tₙ = tₙ₋₁·r/n
    let mut t : DyadicI := one
    let mut s : DyadicI := one
    let mut converged := false
    for n in [1:seriesCap] do
      t := div (mul t r) (ofNat n)
      s := add s t
      if certLt (abs t) (tailTarget s) then
        converged := true
        break
    if !converged then return poison
    -- tail ≤ |t| (the ratio |r|/(n+1) is well under ½ once the loop stops)
    let tw := abs t
    shift ⟨s.lo - tw.hi, s.hi + tw.hi, s.ok⟩ k

/-- Euler's `e`, certified. A nullary `def`, so it is evaluated once at module
    init and every use is a global read — the series does not re-run per call.
    (The float bake path reaches the same constant two ways, `Float.exp 1.0` in
    one place and the literal `2.718281828459045` in another; here there is one
    of it, and it is the true `e` rather than a rounding of it.) -/
def eulerI : DyadicI := exp one

/-- `atanh s` for `|s| ≤ 0.2` — the kernel of `log`. `Σ s^{2j+1}/(2j+1)`.

    Unlike `sin`/`cos`/`atan` this series does NOT alternate: every term carries
    the sign of `s`, so the truncated tail lies strictly on ONE SIDE of the
    accumulated sum and the interval must be widened symmetrically to keep the
    enclosure. (The tail after term `j` is under `|t|·s²/(1−s²) < |t|/24` at
    `|s| ≤ 0.2`, so widening by `|t|` is generous.) -/
private def atanhSmall (s : DyadicI) : DyadicI :=
  if !s.ok then poison
  else Id.run do
    let s2 := mul s s
    let mut term := s            -- s^{2j+1}
    let mut acc := s
    let mut last := s
    let mut converged := false
    for j in [1:seriesCap] do
      term := mul term s2
      let t := div term (ofNat (2 * j + 1))
      acc := add acc t
      last := t
      if certLt (abs t) (tailTarget acc) then
        converged := true
        break
    if !converged then return poison
    let tw := (abs last).hi
    return ⟨acc.lo - tw, acc.hi + tw, acc.ok⟩

/-- `ln x` for a certifiably POSITIVE `x`; `poison` otherwise. Range-reduces by
    the binary exponent to `m ∈ [1, 2)`, re-centres to `[√½, √2)` when `m > √2`
    (an overlap switch: either side keeps `|s| ≤ 0.1716`, so the branch is
    taken deterministically from the midpoint and neither answer is wrong),
    then `ln m = 2·atanh((m−1)/(m+1))`. -/
def log (x : DyadicI) : DyadicI :=
  if !x.ok || !certGt x zero then poison
  else
    let e0 := (mid x).magBits - 1              -- 2^e0 ≤ |x| < 2^(e0+1) about
    let m0 := x.shift (-e0)                    -- exact: a power of two
    -- √2 ≈ 1.4142135623730951; the comparison only picks a branch
    let big := Dyadic.blt (Dyadic.ofFloat 1.4142135623730951) (mid m0)
    let m := if big then m0.shift (-1) else m0
    let e := if big then e0 + 1 else e0
    let s := div (sub m one) (add m one)
    add (mul (ofInt e) ln2I) ((atanhSmall s).shift 1)

/-- `sin r` for `|r| ≤ π/4` — the alternating Taylor series. -/
private def sinSmall (r : DyadicI) : DyadicI :=
  if !r.ok then poison
  else Id.run do
    let r2 := mul r r
    let mut term := r            -- r^{2j+1}/(2j+1)!
    let mut acc := r
    let mut converged := false
    for j in [1:seriesCap] do
      term := div (mul term r2) (ofNat ((2 * j) * (2 * j + 1)))
      acc := if j % 2 == 1 then sub acc term else add acc term
      if certLt (abs term) (tailTarget acc) then
        converged := true
        break
    if !converged then return poison
    let tw := abs term
    ⟨acc.lo - tw.hi, acc.hi + tw.hi, acc.ok⟩

/-- `cos r` for `|r| ≤ π/4` — the alternating Taylor series. -/
private def cosSmall (r : DyadicI) : DyadicI :=
  if !r.ok then poison
  else Id.run do
    let r2 := mul r r
    let mut term := one          -- r^{2j}/(2j)!
    let mut acc := one
    let mut converged := false
    for j in [1:seriesCap] do
      term := div (mul term r2) (ofNat ((2 * j - 1) * (2 * j)))
      acc := if j % 2 == 1 then sub acc term else add acc term
      if certLt (abs term) (tailTarget acc) then
        converged := true
        break
    if !converged then return poison
    let tw := abs term
    ⟨acc.lo - tw.hi, acc.hi + tw.hi, acc.ok⟩

/-- Quadrant reduction `x = k·(π/2) + r` with `|r| ≤ π/4`, shared by `sin` and
    `cos`. `k` comes from the midpoint — deterministic, and a neighbouring `k`
    only widens `|r|` to `≤ 3π/4`, which the series still handles, so there is no
    cliff at the boundary. Large arguments spend reduction bits: `π` is carried
    to `2^{−300}` and the working precision is 128, so `|x|` up to `2^160` still
    lands a full working mantissa. -/
private def reduceQuad (x : DyadicI) : Int × DyadicI :=
  let k := roundToInt (div x piHalfI)
  (k % 4, sub x (mul (ofInt k) piHalfI))

/-- `sin x`. The quadrant selects WHICH of the two small-argument series to run,
    and only that one is evaluated: these are the carrier's hottest kernels (a
    256-panel Bessel quadrature is 257 of them, `defaultStringModes` three per
    partial), and computing the sibling series to throw it away doubled every
    one of those. -/
def sin (x : DyadicI) : DyadicI :=
  if !x.ok then poison
  else
    let (q, r) := reduceQuad x
    match q with
    | 0 => sinSmall r
    | 1 => cosSmall r
    | 2 => neg (sinSmall r)
    | _ => neg (cosSmall r)

/-- `cos x` — the same reduction, the other quadrant map. -/
def cos (x : DyadicI) : DyadicI :=
  if !x.ok then poison
  else
    let (q, r) := reduceQuad x
    match q with
    | 0 => cosSmall r
    | 1 => neg (sinSmall r)
    | 2 => neg (cosSmall r)
    | _ => sinSmall r

/-- Iteration cap for `atanUnit` alone, and it is a COST bound rather than a
    domain one. After three halvings a convergent argument satisfies
    `|z| ≤ tan(π/16) < 0.2`, so the alternating series needs `(2j+1)·log₂(1/0.2)
    > workingPrec + 8`, i.e. about 30 terms — `seriesCap`'s 4096 is eight
    decades of headroom nothing can use. A NON-convergent argument, on the other
    hand, runs the cap out in full at 128 bits before returning `poison`, and
    that is reachable: the halvings are decorrelated on a WIDE enclosure, so a
    box like `[0, 1e6]` keeps `|z| ≥ 1`, the terms grow, the stopping test can
    never fire, and one call costs ~45 ms. `atan2` routes such a box here since
    its denominator is chosen by certified separation rather than by magnitude
    (before that it poisoned instantly in `inv`, which was fast for the wrong
    reason). 256 keeps every convergent case untouched — eight times the terms
    any of them uses — and makes the divergent one sixteen times cheaper. -/
private def atanCap : Nat := 256

/-- `atan a`. Halves the argument three times by
    `atan a = 2·atan(a / (1 + √(1+a²)))` — after which a convergent `|a|` is
    under `tan(π/16) < 0.2` and the alternating Taylor series converges in ~30
    terms — then undoes the halvings. No π constant enters, so the fold cannot
    drift.

    `|a| ≤ 1` is the cheap case, not a precondition: the halving map is
    `tan(θ/2)` with `|θ| < π/2`, so it lands inside `(−1, 1)` after ONE step
    whatever it started at. That matters because `atan2` no longer bounds the
    quotient it passes — its denominator is picked for certified separation, not
    for magnitude — so what the size tie-break buys is iterations, not validity.
    A WIDE enclosure is the case the halvings cannot help: they shrink a
    magnitude, not a width, so `|z| ≥ 1` can survive all three and the series
    then diverges. That returns `poison` (which `atan2` widens to the full
    range), and `atanCap` is what stops it costing 4096 iterations to say so. -/
private def atanUnit (a : DyadicI) : DyadicI :=
  if !a.ok then poison
  else Id.run do
    let mut z := a
    for _ in [0:3] do
      z := div z (add one (sqrt (add one (mul z z))))
    let z2 := mul z z
    let mut term := z
    let mut acc := z
    let mut converged := false
    for j in [1:atanCap] do
      term := mul term z2
      let t := div term (ofNat (2 * j + 1))
      acc := if j % 2 == 1 then sub acc t else add acc t
      if certLt (abs t) (tailTarget acc) then
        converged := true
        break
    if !converged then return poison
    let tw := abs term
    let enclosed : DyadicI := ⟨acc.lo - tw.hi, acc.hi + tw.hi, acc.ok⟩
    enclosed.shift 3

/-- `atan2 (y, x) ∈ [−π, π]` — the angle of `(x, y)`. First octant by
    `a = min(|x|,|y|)/max(|x|,|y|)`, then the swap and the quadrant placement.

    Three things here are NOT overlap switches and are handled as such:

    * **The origin.** `atan2 (0,0) = 0` is a CONVENTION, and an argument that is
      exactly the origin gets it (matching the float bake path and the emitted
      `atan2E`). But an enclosure that merely CONTAINS the origin without being
      it determines no angle at all, and answering `0` with zero width would be
      a false certification — the one failure mode this carrier exists to
      prevent. That case gets the whole range `[−π, π]`.
    * **The quadrant**, and it is TWO different questions, one per axis, because
      only one of the axes carries the branch cut.

      Across the **y axis** (`x = 0`, `y ≠ 0`) atan2 is CONTINUOUS: the two
      placements `r₁` and `π − r₁` agree in the limit, and whenever `x`'s
      enclosure can vanish, `r₁`'s enclosure already reaches `π/2` — so an
      enclosure of `x` that merely TOUCHES zero picks one arm and loses nothing.

      Across the **negative x axis** (`y = 0`, `x < 0`) it JUMPS by 2π: the
      convention sends `y = +0` to `+π` and `y = −0` to `−π`. A `y` whose
      enclosure touches zero from below therefore admits BOTH, and taking only
      the negative arm — as an earlier cut of this function did, classifying by
      `!hi.isPos` on both axes — EXCLUDES the true `+π`. Only a certifiably
      negative `y` may take a single arm; `y ≥ 0` takes the `+0` convention
      like IEEE.
    * **The swap.** The denominator must be certifiably away from zero or `inv`
      poisons, and a poisoned quotient here would destroy a perfectly
      well-defined angle (`atan2 (1, [−1, 3])` excludes the origin entirely).
      So the denominator is chosen by CERTIFIED SEPARATION first — the guard
      above leaves at most one of `|x|`, `|y|` straddling zero, so a legal
      choice always exists — and only among two legal choices does the midpoint
      break the tie toward the larger one. That tie-break is the sole overlap
      switch, and it is genuinely free: `atanUnit`'s three halvings bring ANY
      finite quotient under `tan(π/16) < 0.2` before the series runs, so a
      wrong-size quotient costs a little width, never the answer.

    Where the series still cannot converge, the answer widens to the full range
    rather than poisoning: a total function that sometimes says `[−π, π]` is
    worth more here than a partial one. The final hull is intersected with
    `[−π, π]`, which the true angle never leaves. -/
def atan2 (y x : DyadicI) : DyadicI :=
  if !y.ok || !x.ok then poison
  else if x.lo.isZero && x.hi.isZero && y.lo.isZero && y.hi.isZero then zero
  else
    let full : DyadicI := ⟨-piI.hi, piI.hi, true⟩
    let ax := abs x
    let ay := abs y
    if !certGt (max ax ay) zero then
      -- the origin is inside but the argument is not the origin: no angle is excluded
      full
    else
      -- `swap = true` divides by `|y|`, `false` by `|x|`; a straddling divisor
      -- overrides the midpoint preference
      let swap :=
        if straddlesZero ay then false
        else if straddlesZero ax then true
        else Dyadic.blt (mid ax) (mid ay)
      let num := if swap then ax else ay
      let den := if swap then ay else ax
      let r0 := atanUnit (div num den)
      if !r0.ok then full
      else
        let r1 := if swap then sub piHalfI r0 else r0
        let place := fun (xNeg yNeg : Bool) =>
          let a := if xNeg then sub piI r1 else r1
          if yNeg then neg a else a
        -- x: one arm on either side of zero, INCLUDING a zero endpoint (the
        -- function is continuous there); y: one arm only when the sign is
        -- certain, because the branch cut lies along `y = 0, x < 0`
        let armsX := fun (v : DyadicI) =>
          if !v.lo.isNeg then #[false]
          else if !v.hi.isPos then #[true]
          else #[true, false]
        let armsY := fun (v : DyadicI) =>
          if !v.lo.isNeg then #[false]
          else if v.hi.isNeg then #[true]
          else #[true, false]
        Id.run do
          let mut acc : Option DyadicI := none
          for xn in armsX x do
            for yn in armsY y do
              let a := place xn yn
              acc := some (match acc with | none => a | some p => hull p a)
          let h := acc.getD full
          -- the true angle is in `[−π, π]`, so trimming to it can only tighten
          let lo := Dyadic.dmax h.lo (-piI.hi)
          let hi := Dyadic.dmin h.hi piI.hi
          return if h.ok && Dyadic.ble lo hi then ⟨lo, hi, true⟩ else h

/-- `x^y` for a certifiably positive `x`, as `exp(y·ln x)`. -/
def pow (x y : DyadicI) : DyadicI :=
  if !x.ok || !y.ok then poison
  else if !certGt x zero then poison
  else exp (mul y (log x))

/-- `x^n` for a natural exponent — exact repeated multiplication, no `log`
    round trip, and valid at `x ≤ 0` where `pow` is not. -/
def powNat (x : DyadicI) : Nat → DyadicI
  | 0 => one
  | 1 => x
  | n + 1 => mul (powNat x n) x

end DyadicI

end Tropical.Exact
