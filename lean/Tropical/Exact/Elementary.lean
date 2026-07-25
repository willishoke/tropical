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

/-- Quadrant-reduced `(sin x, cos x)`: `x = k·(π/2) + r` with `|r| ≤ π/4`, the
    quadrant `k mod 4` selecting which of `±sin r`, `±cos r` each answer is.
    `k` comes from the midpoint — deterministic, and a neighbouring `k` only
    widens `|r|` to `≤ 3π/4`, which the series still handles, so there is no
    cliff at the boundary. Large arguments spend reduction bits: `π` is carried
    to `2^{−300}` and the working precision is 128, so `|x|` up to `2^160` still
    lands a full working mantissa. -/
private def sinCosReduced (x : DyadicI) : DyadicI × DyadicI :=
  if !x.ok then (poison, poison)
  else
    let k := roundToInt (div x piHalfI)
    let r := sub x (mul (ofInt k) piHalfI)
    let sr := sinSmall r
    let cr := cosSmall r
    match k % 4 with
    | 0 => (sr, cr)
    | 1 => (cr, neg sr)
    | 2 => (neg sr, neg cr)
    | _ => (neg cr, sr)

def sin (x : DyadicI) : DyadicI := (sinCosReduced x).1
def cos (x : DyadicI) : DyadicI := (sinCosReduced x).2

/-- `atan a` for `|a| ≤ 1`. Halves the argument three times by
    `atan a = 2·atan(a / (1 + √(1+a²)))` — after which `|a| ≤ 0.0985` and the
    alternating Taylor series converges fast — then undoes the halvings. No π
    constant enters, so the fold cannot drift. -/
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
    for j in [1:seriesCap] do
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

    Two things here are NOT overlap switches and are handled as such:

    * **The origin.** `atan2 (0,0) = 0` is a CONVENTION, and an argument that is
      exactly the origin gets it (matching the float bake path and the emitted
      `atan2E`). But an enclosure that merely CONTAINS the origin without being
      it determines no angle at all, and answering `0` with zero width would be
      a false certification — the one failure mode this carrier exists to
      prevent. That case gets the whole range `[−π, π]`.
    * **The quadrant.** `x < 0 ⇒ π − r` and `y < 0 ⇒ −r` are π-SIZED jumps. A
      midpoint would pick one arm and be wrong about the other half of the
      enclosure, so instead the result is the HULL over every sign combination
      the enclosures still admit. An enclosure that stays on one side of an axis
      — including one whose endpoint IS zero, which takes the `+0` convention
      like IEEE — picks a single arm and loses no width; only a genuine straddle
      widens.

    Only the `swap` is a real overlap switch: `atanUnit`'s halving maps every
    real into `(−1, 1)`, so a wrong swap costs iterations, never correctness. -/
def atan2 (y x : DyadicI) : DyadicI :=
  if !y.ok || !x.ok then poison
  else if x.lo.isZero && x.hi.isZero && y.lo.isZero && y.hi.isZero then zero
  else
    let ax := abs x
    let ay := abs y
    if !certGt (max ax ay) zero then
      -- the origin is inside but the argument is not the origin: no angle is excluded
      ⟨-piI.hi, piI.hi, true⟩
    else
      let swap := Dyadic.blt (mid ax) (mid ay)
      let num := if swap then ax else ay
      let den := if swap then ay else ax
      let r0 := atanUnit (div num den)
      let r1 := if swap then sub piHalfI r0 else r0
      let place := fun (xNeg yNeg : Bool) =>
        let a := if xNeg then sub piI r1 else r1
        if yNeg then neg a else a
      -- one arm whenever the enclosure stays on one side of the axis (a zero
      -- endpoint counts as the `+0` side, as IEEE does); both when it straddles
      let arms := fun (v : DyadicI) =>
        if !v.lo.isNeg then #[false]
        else if !v.hi.isPos then #[true]
        else #[true, false]
      Id.run do
        let mut acc : Option DyadicI := none
        for xn in arms x do
          for yn in arms y do
            let a := place xn yn
            acc := some (match acc with | none => a | some p => hull p a)
        return acc.getD poison

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
