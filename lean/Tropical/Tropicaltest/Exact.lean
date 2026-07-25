import Tropical.Tropicaltest.Harness
import Tropical.Exact.Gamma
import Tropical.EmitArrow.Modal

/-!
# Tropical.Tropicaltest.Exact — the exact-carrier gates (P0)

Five standing gates over `Tropical.Exact`, the bake layer's `libm` exile:

* **`exact-constants`** — π and ln 2 are shipped as literal integer mantissas
  over `2³⁰⁰`. This gate RECOMPUTES both from scratch inside the carrier, at a
  precision well above the literals' own, with rigorous remainder bounds
  (Machin's `16·atan(1/5) − 4·atan(1/239)`; `2·atanh(1/3)`), and checks that the
  recomputed enclosure sits INSIDE the literal one. A wrong digit is a red
  gate, not a silent drift, and the recomputation cannot pass by sharing a bug
  with the literal — the literals are data, the recomputation is code.

* **`exact-elementary`** — the corpus differential against the `Float` path.
  Every transcendental is evaluated at the shapes the bake layer actually
  reaches (the exponent ranges of `bloomGammaStar`, the `|κ|` up to ~178, the
  gong's fractional powers, the Lanczos a-range) and checked two ways: the
  enclosure must be TIGHT (the carrier is not allowed to answer "somewhere in
  this wide interval"), and its midpoint must agree with the platform `libm`
  inside the ulp budget a double can carry. The second direction is the one
  that would catch a carrier that is deterministic but wrong.

* **`exact-quantize`** — `Dyadic.toDecimalMantissa` against `litF`, the emit
  funnel. These must agree on every value in `litF`'s HONEST range, and the
  gate also pins `litF`'s two known edges (flush-to-zero below `5e-13`,
  mantissa saturation above `1.8446744e7`) so that the day the emit boundary
  moves onto the exact quantizer, what changes is written down here.

* **`exact-atan2`** — the one function here that is not a monotone series, and
  the only one whose two open defects survived P0's own review. `atan2` is
  checked as an INTERVAL EXTENSION: every point of a box must land inside that
  box's answer (probing the AXES, not just the corners — both defects lived on
  the boundary), and a box that excludes the origin must produce an answer
  rather than poison.

* **`exact-recip10`** — the decimal reciprocal CACHE against the division it
  replaces. `ofJsonNumber` stopped dividing per literal and started multiplying
  by a precomputed `1/10^e`; because `div` is DEFINED as multiply-by-reciprocal
  that is an identity, and this gate holds it to the bit rather than to a
  tolerance. It compares against the LIVE `div` expression, not a frozen table,
  so the day `divAt` stops being "multiply by the reciprocal" the gate says so.
  It also PRINTS both timings; it does not gate on them, because a wall-clock
  threshold in a test suite is a flake.

The gates are cheap (no render, no JIT) and run in the same `passGate` protocol
as everything else.
-/

open Tropical.Exact
open Tropical.Exact.DyadicI

namespace Tropical.Tropicaltest.ExactGates

/-- Precision the constants are RE-derived at: well above the 300 bits the
    literals carry, so the check has room to be decisive. -/
private def certPrec : Nat := 420

/-- `atan(1/n)` for an integer `n ≥ 2`, as a certified enclosure at `certPrec`.
    The alternating series `Σ (−1)ʲ/((2j+1)·n^{2j+1})` has strictly decreasing
    terms, so the tail is bounded by the first omitted term — the interval is
    widened by exactly that. -/
private def atanInv (n : Nat) : DyadicI := Id.run do
  let x := invAt certPrec (ofNat n)
  let x2 := mulAt certPrec x x
  let mut power := x                      -- x^{2j+1}
  let mut acc := x
  let mut lastTerm := x
  for j in [1:2000] do
    power := mulAt certPrec power x2
    let t := divAt certPrec power (ofNat (2 * j + 1))
    acc := if j % 2 == 1 then subAt certPrec acc t else addAt certPrec acc t
    lastTerm := t
    -- stop once the term is below 2^{-(certPrec+8)} — the tail is then under it
    if certLt (abs t) (exact (Dyadic.ofIntWithPrec 1 ((certPrec + 8 : Nat) : Int))) then
      break
  let w := (abs lastTerm).hi
  return ⟨acc.lo - w, acc.hi + w, acc.ok⟩

/-- π by Machin: `16·atan(1/5) − 4·atan(1/239)`. -/
private def piRecomputed : DyadicI :=
  subAt certPrec (mulAt certPrec (ofNat 16) (atanInv 5))
                 (mulAt certPrec (ofNat 4) (atanInv 239))

/-- `ln 2 = 2·atanh(1/3) = 2·Σ 1/((2k+1)·3^{2k+1})` — all terms positive, and
    the tail after term `k` is under `term·(1/9)/(1−1/9) < term/8`, so widening
    UPWARD by the last term is generous. -/
private def ln2Recomputed : DyadicI := Id.run do
  let x := invAt certPrec (ofNat 3)
  let x2 := mulAt certPrec x x
  let mut power := x
  let mut acc := x
  let mut lastTerm := x
  for k in [1:2000] do
    power := mulAt certPrec power x2
    let t := divAt certPrec power (ofNat (2 * k + 1))
    acc := addAt certPrec acc t
    lastTerm := t
    if certLt (abs t) (exact (Dyadic.ofIntWithPrec 1 ((certPrec + 8 : Nat) : Int))) then
      break
  let w := (abs lastTerm).hi
  return (⟨acc.lo, acc.hi + w, acc.ok⟩ : DyadicI).shift 1

/-- Is `inner` contained in `outer`? -/
private def containedIn (inner outer : DyadicI) : Bool :=
  inner.ok && outer.ok && Dyadic.ble outer.lo inner.lo && Dyadic.ble inner.hi outer.hi

/-- π and ln 2, re-derived from scratch and checked against the shipped
    literals. -/
def runExactConstants : IO Bool := do
  let piR := piRecomputed
  let ln2R := ln2Recomputed
  let piOk := containedIn piR piI
  let ln2Ok := containedIn ln2R ln2I
  -- the recomputation must also be SHARPER than the literal, or the check is vacuous
  let piSharp := Dyadic.blt piR.width piI.width
  let ln2Sharp := Dyadic.blt ln2R.width ln2I.width
  IO.println s!"        recomputed at {certPrec} bits: π {piR.render} · ln2 {ln2R.render}"
  IO.println s!"        shipped literals (2^-{constPrec}): π {piI.render} · ln2 {ln2I.render}"
  if piOk && ln2Ok && piSharp && ln2Sharp then
    passGate "exact-constants"
      s!"π (Machin) and ln2 (atanh ⅓) re-derived inside the carrier at {certPrec} bits land INSIDE the shipped 2^-{constPrec} literals — the constants are gate-covered data, not trusted digits"
  else
    failGate "exact-constants"
      s!"piContained={piOk} ln2Contained={ln2Ok} piSharper={piSharp} ln2Sharper={ln2Sharp}"

-- ── the corpus differential ───────────────────────────────────────────────────

/-- ULP distance between a carrier midpoint and the platform `libm` value,
    measured in units of the double's own precision (`2^-52`). -/
private def ulpDist (m ref : Float) : Float :=
  if !ref.isFinite || !m.isFinite then 1.0e30
  else if ref == 0.0 then (if m == 0.0 then 0.0 else 1.0e30)
  else
    let d := (m - ref) / ref
    (if d < 0.0 then -d else d) * 4503599627370496.0

/-- How many bits of the value the enclosure actually certifies: `−log₂` of the
    relative width, so BIGGER is tighter. `1e9` for an exact interval. -/
private def certifiedBits (x : DyadicI) : Float :=
  if !x.ok then -1.0e9
  else if x.width.isZero then 1.0e9
  else
    let m := (abs x).lo
    if m.isZero then Float.ofInt (-x.width.magBits)
    else Float.ofInt (m.magBits - x.width.magBits)

/-- Halton radical inverse — the deterministic low-discrepancy sampler the
    seam sweep already uses (`Math.random` is banned and would break resume). -/
private def halton (base i : Nat) : Float := Id.run do
  let mut f : Float := 1.0
  let mut r : Float := 0.0
  let mut n : Nat := i
  for _ in [0:32] do
    if n > 0 then
      f := f / base.toFloat
      r := r + f * (n % base).toFloat
      n := n / base
  return r

/-- One differential outcome: worst ulp distance from the `Float` path and the
    LEAST number of bits any enclosure certified. -/
private structure Score where
  worstUlp  : Float := 0.0
  leastBits : Float := 1.0e9
  worstAt   : String := ""
  loosestAt : String := ""
  poisoned  : Nat := 0
deriving Inhabited

private def Score.note (s : Score) (name : String) (iv : DyadicI) (ref : Float) : Score :=
  if !iv.ok then { s with poisoned := s.poisoned + 1 }
  else
    let u := ulpDist iv.toFloat ref
    let b := certifiedBits iv
    { worstUlp  := max s.worstUlp u,
      leastBits := min s.leastBits b,
      worstAt   := if u > s.worstUlp then name else s.worstAt,
      loosestAt := if b < s.leastBits then name else s.loosestAt,
      poisoned  := s.poisoned }

/-- Every transcendental over the shapes the bake layer reaches, against the
    `Float` path — the P0 corpus differential. -/
def runExactElementary : IO Bool := do
  let n := 240
  let mut sc : Score := {}
  for i in [1:n+1] do
    let u1 := halton 2 i
    let u2 := halton 3 i
    let u3 := halton 5 i
    -- exp over the shipped exponent span (κ reaches ~178; expSig clamps at ±87)
    let xe := (u1 * 2.0 - 1.0) * 700.0
    sc := sc.note s!"exp {xe}" (DyadicI.exp (ofFloat xe)) (Float.exp xe)
    -- log over the gauge/bridge span
    let xl := Float.exp ((u2 * 2.0 - 1.0) * 690.0)
    sc := sc.note s!"log {xl}" (DyadicI.log (ofFloat xl)) (Float.log xl)
    -- sin/cos over ω·d spans (ω up to 2π·20k, d up to seconds)
    let xt := (u3 * 2.0 - 1.0) * 125000.0
    sc := sc.note s!"sin {xt}" (DyadicI.sin (ofFloat xt)) (Float.sin xt)
    sc := sc.note s!"cos {xt}" (DyadicI.cos (ofFloat xt)) (Float.cos xt)
    -- atan2 over all four quadrants (the CplxB.log phase)
    let ay := (u1 * 2.0 - 1.0) * 200.0
    let ax := (u2 * 2.0 - 1.0) * 200.0
    sc := sc.note s!"atan2 {ay} {ax}" (DyadicI.atan2 (ofFloat ay) (ofFloat ax)) (Float.atan2 ay ax)
    -- pow: the envelope-peak factor (p/(σe))^p and the gong's r^0.7 / r^0.8
    let pb := 0.05 + u3 * 40.0
    let pe := 0.3 + u1 * 3.0
    sc := sc.note s!"pow {pb} {pe}" (DyadicI.pow (ofFloat pb) (ofFloat pe)) (Float.pow pb pe)
    -- complex log-gamma over the Γ★ bridge's a-range
    let gr := (u2 * 2.0 - 1.0) * 40.0
    let gi := (u3 * 2.0 - 1.0) * 150.0
    let lgE := CplxDI.lgamma (CplxDI.ofFloats gr gi)
    let lgF := Tropical.EmitArrow.lgammaB ⟨gr, gi⟩
    sc := sc.note s!"lgamma.re {gr},{gi}" lgE.re lgF.re
    sc := sc.note s!"lgamma.im {gr},{gi}" lgE.im lgF.im
  IO.println s!"        {n} Halton configs × 9 kernels vs the Float path:"
  IO.println s!"        worst |exact−float| {sc.worstUlp} ulp (at {sc.worstAt}) · tightest-case certified bits {sc.leastBits} (at {sc.loosestAt}) · poisoned {sc.poisoned}"
  -- Two directions, two budgets. The ULP budget is the FLOAT path's error, not
  -- the carrier's: a complex Lanczos chain accumulates hundreds of ulps in f64,
  -- and the carrier is the accurate side of that comparison — a LARGE number
  -- here is evidence about `lgammaB`, not about this module. The BITS budget is
  -- the carrier's own: every enclosure must certify far more than a double's 53.
  if sc.poisoned == 0 && sc.worstUlp < 65536.0 && sc.leastBits > 90.0 then
    passGate "exact-elementary"
      s!"exp/log/sin/cos/atan2/pow/lgamma track the float path to {sc.worstUlp} ulp over the shipped corpus while certifying ≥{sc.leastBits} bits (a double carries 53) — the bake layer's transcendentals are reproducible without libm"
  else
    failGate "exact-elementary"
      s!"worstUlp={sc.worstUlp} (at {sc.worstAt}) leastBits={sc.leastBits} (at {sc.loosestAt}) poisoned={sc.poisoned}"

-- ── the interval extension: atan2 ─────────────────────────────────────────────

/-- The sample points a containment check must probe for one box: the corners,
    the centre, and wherever the box MEETS AN AXIS — which is where atan2's two
    special structures live (the continuous crossing at `x = 0`, and the 2π-wide
    branch cut along `y = 0, x < 0`). Both of the defects this gate exists for
    were exactly there, and neither was reachable from corner samples alone. -/
private def boxProbes (xlo xhi ylo yhi : Float) : Array (Float × Float) := Id.run do
  let xs := #[xlo, xhi, 0.5 * (xlo + xhi)]
             ++ (if xlo ≤ 0.0 && 0.0 ≤ xhi then #[0.0] else #[])
  let ys := #[ylo, yhi, 0.5 * (ylo + yhi)]
             ++ (if ylo ≤ 0.0 && 0.0 ≤ yhi then #[0.0] else #[])
  let mut out : Array (Float × Float) := #[]
  for a in xs do
    for b in ys do
      out := out.push (a, b)
  return out

/-- `atan2` as an INTERVAL EXTENSION, checked the only way an enclosure can be:
    EVERY point of a box must land inside that box's answer, and a box that
    excludes the origin must produce an answer at all.

    The two properties are separable, and the two defects this gate was written
    against are one of each:

    * **Soundness.** `atan2 [−1,0] [−2,−1]` returned `[−3.1468, −2.1415]`, which
      does not contain `atan2 0 (−1) = +π` — a point of its own box. The arm
      classifier read a `hi` of exactly zero as strictly negative, so it took
      only the `−0` side of the branch cut. A containment sweep that probes the
      axes finds this; one that probes corners does not, because the violating
      points are ON the boundary.
    * **Totality.** `atan2 1 [−1,3]` poisoned, though every point of that box has
      a perfectly good angle (the box excludes the origin entirely). The
      midpoint chose a zero-straddling denominator and `inv` refused it.

    Containment is checked carrier-against-carrier — the point evaluation must
    sit inside the box evaluation — rather than against the platform `libm`, so
    no ulp budget enters. The box answer is allowed `2^{−100}` of slack: the two
    evaluations may take different reduction branches (a different swap, a
    different quadrant `k`), and each is only obliged to enclose the truth, not
    to nest in the other bit-for-bit. That slack is still ~2^26 times WIDER than
    a point answer's own width, so it cannot hide anything at the π scale these
    defects live at. The `leastBits` floor is what stops the whole gate from
    being satisfied by a function that always answers `[−π, π]`. -/
def runExactAtan2 : IO Bool := do
  let n := 300
  let slack : Int := -100
  let mut boxes := 0
  let mut probes := 0
  let mut escaped := 0
  let mut escapedAt := ""
  let mut poisoned := 0
  let mut poisonAt := ""
  let mut ptPoison := 0
  let mut leastBits : Float := 1.0e9
  for i in [1:n+1] do
    let u1 := halton 2 i
    let u2 := halton 3 i
    let u3 := halton 5 i
    let u4 := halton 7 i
    let cx := (u1 * 2.0 - 1.0) * 8.0
    let cy0 := (u2 * 2.0 - 1.0) * 8.0
    let cy := if cy0 == 0.0 then 1.0 else cy0
    let rx := u3 * 4.0
    let ry := u4 * 4.0
    -- five constructions: a general box, the two axis-touching classes the open
    -- defects lived in, a straddling denominator over a point numerator, and a
    -- degenerate box (which is what the production bake path actually hands it)
    let (xlo, xhi, ylo, yhi) :=
      match i % 5 with
      | 0 => (cx - rx, cx + rx, cy - ry, cy + ry)
      | 1 => (-1.0 - rx, -0.5, -ry - 0.001, 0.0)      -- y.hi = 0 exactly, x < 0
      | 2 => (-1.0 - rx, -0.5, 0.0, ry + 0.001)       -- y.lo = 0 exactly, x < 0
      | 3 => (cx - rx - 1.0, cx + rx + 1.0, cy, cy)   -- x straddles, y a point
      | _ => (cx, cx, cy, cy)                         -- degenerate
    let xb : DyadicI := ⟨Dyadic.ofFloat xlo, Dyadic.ofFloat xhi, true⟩
    let yb : DyadicI := ⟨Dyadic.ofFloat ylo, Dyadic.ofFloat yhi, true⟩
    let res := DyadicI.atan2 yb xb
    boxes := boxes + 1
    let originIn := straddlesZero xb && straddlesZero yb
    if !originIn && !res.ok then
      poisoned := poisoned + 1
      poisonAt := s!"y=[{ylo},{yhi}] x=[{xlo},{xhi}]"
    if xlo == xhi && ylo == yhi then
      leastBits := min leastBits (certifiedBits res)
    let wide := res.widen slack
    for (px, py) in boxProbes xlo xhi ylo yhi do
      let pt := DyadicI.atan2 (ofFloat py) (ofFloat px)
      probes := probes + 1
      if !pt.ok then ptPoison := ptPoison + 1
      else if !containedIn pt wide then
        escaped := escaped + 1
        escapedAt := s!"({px},{py}) ∉ atan2 [{ylo},{yhi}] [{xlo},{xhi}]"
  -- the two ledger defects, as named regressions
  let boxA : DyadicI := ⟨Dyadic.ofFloat (-2.0), Dyadic.ofFloat (-1.0), true⟩
  let boxAy : DyadicI := ⟨Dyadic.ofFloat (-1.0), 0, true⟩
  let dA := DyadicI.atan2 boxAy boxA
  let dApt := DyadicI.atan2 zero (ofFloat (-1.0))          -- +π, a point of that box
  let soundnessFixed := containedIn dApt (dA.widen slack)
  let dB := DyadicI.atan2 (ofFloat 1.0)
              ⟨Dyadic.ofFloat (-1.0), Dyadic.ofFloat 3.0, true⟩
  let totalityFixed := dB.ok
  IO.println s!"        {boxes} boxes × {probes} probes (corners, centre, axis crossings):"
  IO.println s!"        escaped {escaped} · box-poison-off-origin {poisoned} · point-poison {ptPoison} · tightest point box certifies {leastBits} bits"
  IO.println s!"        ledger regressions — atan2 [-1,0] [-2,-1] = {dA.render} ⊇ atan2 0 -1 = {dApt.render} : {soundnessFixed} · atan2 1 [-1,3] = {dB.render} : {totalityFixed}"
  if escaped == 0 && poisoned == 0 && ptPoison == 0 && soundnessFixed
      && totalityFixed && leastBits > 90.0 then
    passGate "exact-atan2"
      s!"atan2 is a sound interval extension over {probes} probes of {boxes} boxes (axes included, where both open defects lived) and TOTAL wherever the origin is excluded, while still certifying ≥{leastBits} bits on a point argument"
  else
    failGate "exact-atan2"
      s!"escaped={escaped} (at {escapedAt}) boxPoison={poisoned} (at {poisonAt}) pointPoison={ptPoison} soundnessRegression={soundnessFixed} totalityRegression={totalityFixed} leastBits={leastBits}"

-- ── the emit funnel ───────────────────────────────────────────────────────────

/-- Read `litF`'s emitted decimal back as a 12-place mantissa. -/
private def litFMantissa (x : Float) : Int :=
  match Tropical.EmitArrow.litF x with
  | .num jn => if jn.exponent == 12 then jn.mantissa else jn.mantissa * 1000000000000
  | _ => 0

/-- `Dyadic.toDecimalMantissa` against `litF` — the emit funnel, measured.

    `litF` forms `x · 1e12` IN FLOATING POINT and then rounds, so it is only a
    faithful 12-place quantizer while that product stays exactly representable:
    `|x| ≤ 2⁵³/10¹² ≈ 9007.2`. This gate pins the crossover rather than asserting
    a precision `litF` does not have, and states the far edge — above
    `2⁶⁴/10¹² ≈ 1.8446744e7` the `UInt64` cast SATURATES and `litF` emits a
    number unrelated to its input.

    Below `5e-13` both quantizers answer `0`: that is correct rounding at twelve
    places, not a defect — the resolution simply ends there.

    The exact quantizer has none of this: it is `⌊|x|·10¹² + ½⌋` in `Int`
    arithmetic, at any magnitude. The numbers below are what a future
    `litF → litD` cutover would move, so that decision arrives with its cost
    already measured. -/
def runExactQuantize : IO Bool := do
  let n := 400
  let faithfulTop : Float := 9007.199254740992          -- 2⁵³/10¹²
  let mut agreeLow := 0
  let mut offLow := 0
  let mut lowN := 0
  let mut highN := 0
  let mut worstHigh : Int := 0
  let mut worstHighX : Float := 0.0
  for i in [1:n+1] do
    let u := halton 2 i
    let s := halton 3 i
    let mag := Float.exp (-27.0 + u * 43.0)             -- ~2e-12 … ~1e7
    let x := if s < 0.5 then mag else -mag
    let d := (Dyadic.ofFloat x).toDecimalMantissa 12 - litFMantissa x
    let ad := if d < 0 then -d else d
    if mag ≤ faithfulTop then
      lowN := lowN + 1
      if ad == 0 then agreeLow := agreeLow + 1
      else if ad ≤ 1 then offLow := offLow + 1
    else
      highN := highN + 1
      if ad > worstHigh then
        worstHigh := ad
        worstHighX := x
  -- the far edge: the UInt64 cast saturates
  let satX : Float := 1.0e8
  let satM := litFMantissa satX
  let exactSat := (Dyadic.ofFloat satX).toDecimalMantissa 12
  let saturates := satM == 18446744073709551615 && exactSat == 100000000000000000000
  -- the resolution floor: both answer 0, and they agree that they do
  let subResolution := litFMantissa 1.0e-13 == 0 && (Dyadic.ofFloat 1.0e-13).toDecimalMantissa 12 == 0
  IO.println s!"        litF is a faithful 12-place quantizer only while |x|·10¹² < 2⁵³ (|x| ≤ {faithfulTop}):"
  IO.println s!"        below it  ({lowN} samples): identical {agreeLow} · off-by-one {offLow} · worse {lowN - agreeLow - offLow}"
  IO.println s!"        above it  ({highN} samples): worst drift {worstHigh} units of the 12th place (at x={worstHighX})"
  IO.println s!"        far edge — litF(1e8) = {satM} (UInt64 saturation) vs exact {exactSat}; sub-resolution 1e-13 → 0 both sides {subResolution}"
  if lowN - agreeLow - offLow == 0 && saturates && subResolution && worstHigh > 1 then
    passGate "exact-quantize"
      s!"the exact decimal quantizer reproduces litF wherever litF is faithful ({agreeLow}/{lowN} identical, {offLow} off-by-one) and diverges only where litF's own f64 product has run out of bits (drift up to {worstHigh} units past |x|>{faithfulTop}, hard saturation past 1.8446744e7) — the litF→litD cutover's cost, measured"
    else
      failGate "exact-quantize"
        s!"lowMismatch={lowN - agreeLow - offLow} saturates={saturates} subResolution={subResolution} worstHigh={worstHigh}"

-- ── the decimal reciprocal cache ──────────────────────────────────────────────

/-- A synthetic all-literal deg-0 bank — the shape `bankLandExp` takes the
    STATIC path on (`ModalMode.hz` makes `cre` a 12-place `litF` literal and
    leaves `cim` at `lit 0`). Built outside the timer. -/
private def benchBank (n : Nat) : Array Tropical.EmitArrow.ModalMode :=
  (Array.range n).map fun i =>
    Tropical.EmitArrow.ModalMode.hz
      (Tropical.EmitArrow.litF (110.0 * (i + 1).toFloat))
      (Tropical.EmitArrow.litF (2.0 + 0.01 * i.toFloat))
      (Tropical.EmitArrow.litF (1.0 / (1.0 + i.toFloat)))

/-- `recip10Table` is a CACHE, not a second algorithm — held to the BIT.

    `DyadicI.div x y` is DEFINED as `mulAt workingPrec x (invAt workingPrec y)`,
    so multiplying by a cached `inv (ofNat (10^e))` is the same application of
    the same function to the same arguments; the table only decides WHEN the
    reciprocal is computed. This gate pins that identity against the LIVE `div`
    expression rather than a frozen table, so the day `divAt` becomes a genuine
    directed division — a legitimate improvement — this goes red instead of
    `ofJsonNumber` silently keeping the older, wider answer.

    Bit equality is the right bar and a tolerance would be the wrong one. This
    value feeds `landK`'s power-of-two read and every `certLt` in the EC/DD
    router, where an enclosure one ulp wider OR NARROWER is not a rounding
    difference but a different emitted program. Narrower is not the safe
    direction: a tighter interval turns `overlap` into a verdict, and
    `classifyBloomPairLive` DROPS what it cannot certify.

    Exponents past the table's end are included so the division fallback is
    exercised rather than assumed. The two timings are PRINTED, never gated — a
    wall-clock threshold in a test suite is a flake. -/
def runExactRecip10 : IO Bool := do
  let mants : Array Int :=
    #[0, 1, -1, 5, -5, 244140625, 999999999999, -999999999999,
      2718281828459, -3141592653590, 1000000000000, 6283185307179586,
      4611686018427387904, -4611686018427387904]
  let eTop := DyadicI.recip10Max + 3
  let mut checked := 0
  let mut bad := ""
  for e in [0:eTop] do
    for m in mants do
      let got  := DyadicI.ofJsonNumber ⟨m, e⟩
      let want := if e == 0 then DyadicI.ofInt m
                  else DyadicI.div (DyadicI.ofInt m) (DyadicI.ofNat (10 ^ e))
      checked := checked + 1
      if !(got.ok == want.ok && Dyadic.ble got.lo want.lo && Dyadic.ble want.lo got.lo
            && Dyadic.ble got.hi want.hi && Dyadic.ble want.hi got.hi) then
        if bad.isEmpty then
          bad := s!"m={m} e={e}: cache {got.render} vs division {want.render}"
  -- what the change was for. Both forms timed in the SAME run over the SAME
  -- literals, so the comparison needs no second build and no stored number.
  -- PRINTED, never gated: a wall-clock threshold in a test suite is a flake.
  let t0 ← IO.monoMsNow
  let mut sinkOld : Int := 0
  for i in [1:20001] do
    sinkOld := sinkOld +
      (DyadicI.div (DyadicI.ofInt ((i : Int) * 271828182845))
                   (DyadicI.ofNat (10 ^ 12))).hi.magBits
  let t1 ← IO.monoMsNow
  let mut sinkNew : Int := 0
  for i in [1:20001] do
    sinkNew := sinkNew + (DyadicI.ofJsonNumber ⟨(i : Int) * 271828182845, 12⟩).hi.magBits
  let t2 ← IO.monoMsNow
  -- eight DISTINCT banks, so the folds cannot collapse into one
  let banks := (Array.range 8).map fun j => benchBank (512 + j)
  let t3 ← IO.monoMsNow
  let mut ksum := 0
  for b in banks do
    ksum := ksum + (match Tropical.EmitArrow.bankLandExp b with
                    | .static k  => k
                    | .dynamic _ => 0)
  let t4 ← IO.monoMsNow
  IO.println s!"        20000 12-place literals: division {t1 - t0} ms → cached reciprocal {t2 - t1} ms (sinks {sinkOld}/{sinkNew})"
  IO.println s!"        bankLandExp × 8 over ~512-mode banks in {t4 - t3} ms (Σk {ksum})"
  if bad.isEmpty then
    passGate "exact-recip10"
      s!"the 10^-e reciprocal cache reproduces the division form BIT FOR BIT over {checked} (mantissa, exponent) pairs spanning e ∈ [0, {eTop - 1}] — ofJsonNumber got faster without one enclosure getting wider or narrower, so no certLt/certGt verdict and no emitted program can have moved"
  else
    failGate "exact-recip10" bad

end Tropical.Tropicaltest.ExactGates
