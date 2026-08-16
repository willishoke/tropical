import Tropical.Tropicaltest.Harness
import Tropical.Exact.Gamma
import Tropical.Playground.Vocabulary
import Tropical.Testing.ArrowFixtures
import Tropical.Testing.ArrowOracles

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

open Tropical.EmitArrow
open Tropical.Testing.ArrowFixtures

private abbrev CplxB := Tropical.Testing.ArrowOracles.CplxB

private def dutCplx (value : CplxB) : Tropical.EmitArrow.CplxB :=
  ⟨value.re, value.im⟩

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
    let lgF := Tropical.Testing.ArrowOracles.lgammaB ⟨gr, gi⟩
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
  match freezeBuild {} (litF x) with
  | .ok (arena, signal) => match arena.nodes[signal.idx]? with
    | some (.num number) =>
        if number.exponent == 12 then number.mantissa
        else number.mantissa * 1000000000000
    | _ => 0
  | .error _ => 0

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
private def benchBank (n : Nat) : BuildM (Array ModalMode) :=
  (Array.range n).mapM fun i => do
    ModalMode.hz (← litF (110.0 * (i + 1).toFloat))
      (← litF (2.0 + 0.01 * i.toFloat))
      (← litF (1.0 / (1.0 + i.toFloat)))

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
    direction: a tighter interval turns `overlap` into a verdict, and the checked
    live classifier REFUSES what it cannot certify.

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
  let prepared := freezeBuild {} do
    (Array.range 8).mapM fun j => benchBank (512 + j)
  let (frozen, banks) ← match prepared with
    | .error error =>
        return ← failGate "exact-recip10" s!"arena-native bench bank: {firstLine error}"
    | .ok value => pure value
  let t3 ← IO.monoMsNow
  let mut ksum := 0
  for b in banks do
    match runBuild { exprs := frozen } (bankLandExp b) with
    | .ok (_, .static k) => ksum := ksum + k
    | _ => pure ()
  let t4 ← IO.monoMsNow
  IO.println s!"        20000 12-place literals: division {t1 - t0} ms → cached reciprocal {t2 - t1} ms (sinks {sinkOld}/{sinkNew})"
  IO.println s!"        bankLandExp × 8 over ~512-mode banks in {t4 - t3} ms (Σk {ksum})"
  if bad.isEmpty then
    passGate "exact-recip10"
      s!"the 10^-e reciprocal cache reproduces the division form BIT FOR BIT over {checked} (mantissa, exponent) pairs spanning e ∈ [0, {eTop - 1}] — ofJsonNumber got faster without one enclosure getting wider or narrower, so no certLt/certGt verdict and no emitted program can have moved"
  else
    failGate "exact-recip10" bad

-- ── the P2 differential: the flipped values against the pre-flip float path ────

/-- Relative distance between a carrier value and its float twin, complex. -/
private def relC (m : CplxDI) (ref : CplxB) : Float :=
  let (mr, mi) := m.toFloats
  let dr := mr - ref.re
  let di := mi - ref.im
  let num := Float.sqrt (dr * dr + di * di)
  let den := Float.sqrt (ref.re * ref.re + ref.im * ref.im)
  if !num.isFinite then 1.0e30 else if den < 1.0e-300 then num else num / den

/-- The worst relative disagreement seen, and where. -/
private structure Diff where
  worst   : Float := 0.0
  at?     : String := ""
  litSame : Nat := 0
  litOff1 : Nat := 0
  litWorse: Nat := 0
  worstLit: Int := 0
  poisoned: Nat := 0
deriving Inhabited

/-- Score one (exact, float) pair on BOTH axes. The literal axis is the one that
    matters for the emitted program, and both sides go through `litF` so the
    `x·10¹²` f64 product — which itself rounds, disagreeing on ~2e-4 of literals
    even with a perfect carrier — stays INSIDE the funnel and cannot manufacture
    false diffs. (`toDecimalMantissa` is NOT the analogue of `litF` and must not
    appear here.)

    The literal allowance is MAGNITUDE-AWARE, because `litF` is only a faithful
    12-place quantizer while `|x| ≤ 2⁵³/10¹² ≈ 9007.2` (`exact-quantize` measures
    exactly that): above it the emitted mantissa's own last digits are noise, and
    a raw mantissa delta there says nothing about whether a value moved. So a
    literal counts as agreeing when it is within ONE unit of the 12th place OR
    within `1e-11` relatively. -/
private def Diff.note (d : Diff) (name : String) (m : CplxDI) (ref : CplxB) : Diff :=
  if !m.ok then { d with poisoned := d.poisoned + 1 }
  else
    let r := relC m ref
    let (mr, mi) := m.toFloats
    let step := fun (dd : Diff) (a b : Float) =>
      let delta := litFMantissa a - litFMantissa b
      let ad := if delta < 0 then -delta else delta
      let rel := if b == 0.0 then (if a == 0.0 then 0.0 else 1.0)
                 else Float.abs (a - b) / Float.abs b
      if ad == 0 then { dd with litSame := dd.litSame + 1 }
      else if ad ≤ 1 || rel ≤ 1.0e-11 then
        { dd with litOff1 := dd.litOff1 + 1, worstLit := max dd.worstLit ad }
      else { dd with litWorse := dd.litWorse + 1, worstLit := max dd.worstLit ad }
    let d := step (step d mr ref.re) mi ref.im
    if r > d.worst then { d with worst := r, at? := name } else d

/-- THE VALUES-FLIP DIFFERENTIAL (P2). Every bake-time value function whose float
    twin still exists is evaluated BOTH ways over the shipped bloom register, and
    the two are compared on two axes: the VALUE relatively, and the EMITTED
    LITERAL exactly.

    The 1e-9 value budget is the FLOAT path's own error, not the carrier's.
    `exact-elementary` already measured the worst constituent at ~159 ulp
    (3.5e-14 relative, at `lgamma.im`, with the carrier on the ACCURATE side of
    that comparison); Γ★ and Φ apply `exp`/`log` to that, the Horners accumulate
    up to `300·2⁻⁵²` more, and `Φ`'s large-|κ| arm subtracts two big numbers. The
    budget is set to catch a broken branch or a wrong reduction, not to flag the
    accuracy GAIN it exists to see — which is why the LITERAL histogram is the
    second number to read: a literal disagrees only when the true value sits
    within the float path's own error of a half-grid boundary.

    `poisoned == 0` is a hard assertion. A poisoned constant becomes a typed
    `coefficientMaterialization` refusal, and Patch lowering reports it explicitly;
    the shipped register must never take that path. -/
def runExactValues : IO Bool := do
  let tStart ← IO.monoMsNow
  let n := 120
  let mut ser : Diff := {}      -- serOnly: M(1,a+1,κ), the E1 reciprocals
  let mut cro : Diff := {}      -- crossing: Γ★, CF(κ)
  let mut coi : Diff := {}      -- coincident: dCoef, Φ(a,κ)/g, cexpm1
  let mut fol : Diff := {}      -- the fold: Q coefficients, DDa M
  for i in [1:n+1] do
    let u1 := halton 2 i
    let u2 := halton 3 i
    let u3 := halton 5 i
    let u4 := halton 7 i
    let g : Float := 4.0 + 20.0 * u4
    let gD := ofFloat g
    -- κ = μ·B over the shipped span (|κ| reaches ~178 in the register)
    let kappa : CplxB := ⟨-(0.02 + 2.0 * u1), (u2 * 2.0 - 1.0) * 170.0⟩
    let kD := CplxDI.ofFloats kappa.re kappa.im
    let kAbs := kappa.abs
    -- Each family is sampled INSIDE ITS OWN ADMITTED REGION, because that is the
    -- only place the comparison means anything: outside it BOTH paths are
    -- evaluating a divergent series, they disagree by O(1), and the differential
    -- would be measuring the divergence rather than the carrier. `serOnly`
    -- (and the fold) live where `|a+1| ≥ |κ|`; the crossing lives where
    -- `|κ| ≥ |a+1|`; the coincidence lives at `|a| < ½`.
    -- `a = (ν−μ)/g` is IMAGINARY-DOMINATED in production — `Re a = (σ_μ−σ_ν)/g`
    -- is a damping difference over a settle rate, `Im a` a frequency difference
    -- over the same — and that is precisely `bloomM1`'s documented stability
    -- condition: with `|Im a| ≥ |z|`, every `|a+k|` stays above `|z|` and the
    -- terms decay monotonically from the first. A sampler that let `Re a` run
    -- large would walk `a+k` through its own minimum, the terms would grow by
    -- twenty orders before decaying, and the differential would be measuring
    -- the float path's catastrophic cancellation in a configuration no pole pair
    -- can produce.
    let aSer : CplxB :=
      let im := (kAbs + 1.0 + 4.0 * u3) * (if u4 < 0.5 then 1.0 else -1.0)
      ⟨(u3 * 2.0 - 1.0) * 3.0, im⟩
    let aCro : CplxB := ⟨(u3 * 2.0 - 1.0) * 3.0, (u4 * 2.0 - 1.0) * 9.0⟩
    let aNear : CplxB := ⟨(u1 * 2.0 - 1.0) * 0.3, (u2 * 2.0 - 1.0) * 0.3⟩
    let aSerD := CplxDI.ofFloats aSer.re aSer.im
    let aCroD := CplxDI.ofFloats aCro.re aCro.im
    let aNearD := CplxDI.ofFloats aNear.re aNear.im
    -- serOnly arm: the κ-side M constant at the count the point carrier returns
    let nK := bloomM1DepthD (dutCplx aSer).toPoint (dutCplx kappa).toPoint bloomM1TolD
    let (mKf, _) := Tropical.Testing.ArrowOracles.bloomM1 aSer kappa
    ser := ser.note s!"M1 a={aSer.re},{aSer.im} κ={kappa.re},{kappa.im}"
                    (bloomM1D aSerD kD nK) mKf
    -- crossing arm: Γ★ and the Lentz CF
    cro := cro.note s!"Γ★ a={aCro.re},{aCro.im}"
                    (bloomGammaStarD aCroD kD gD)
                    (Tropical.Testing.ArrowOracles.bloomGammaStar aCro kappa g)
    let (cfKf, _) := Tropical.Testing.ArrowOracles.bloomCF aCro kappa
    let (cfKp, _) := bloomCFPointD (dutCplx aCro).toPoint
      (dutCplx kappa).toPoint bloomCFTolD
    cro := cro.note s!"CF a={aCro.re},{aCro.im}" cfKp.asPointI cfKf
    -- coincident arm: cexpm1 on both sides of its 0.01 split, dCoef, Φ
    let wSmall : CplxB := ⟨0.02 * (u1 - 0.5), 0.02 * (u2 - 0.5)⟩
    let wBig : CplxB := ⟨2.0 * (u1 - 0.5), 2.0 * (u2 - 0.5)⟩
    coi := coi.note s!"cexpm1 small {wSmall.re}" (cexpm1D (CplxDI.ofFloats wSmall.re wSmall.im))
                    (Tropical.Testing.ArrowOracles.cexpm1B wSmall)
    coi := coi.note s!"cexpm1 big {wBig.re}" (cexpm1D (CplxDI.ofFloats wBig.re wBig.im))
                    (Tropical.Testing.ArrowOracles.cexpm1B wBig)
    let dcF := Tropical.Testing.ArrowOracles.bloomDCoef aNear 24
    let dcD := bloomDCoefD aNearD 24
    for k in [0:24] do
      coi := coi.note s!"dCoef[{k}] a={aNear.re},{aNear.im}" dcD[k]! dcF[k]!
    coi := coi.note s!"Φ a={aNear.re},{aNear.im} κ={kappa.re},{kappa.im}"
                    (bloomPhiKappaOverGD aNearD kD (CplxDI.ofFloats cfKf.re cfKf.im) dcD gD)
                    (Tropical.Testing.ArrowOracles.bloomPhiKappaOverG
                      aNear kappa cfKf dcF g)
    -- the fold arm (WS-DDF): two nearby a's, the stable Q recurrence, on the
    -- same series-side admission as `bloomFoldCompose` (`|a+1| ≥ |κ|`)
    let a2 : CplxB := ⟨aSer.re + 0.05, aSer.im - 0.03⟩
    let a2D := CplxDI.ofFloats a2.re a2.im
    let qF := Tropical.Testing.ArrowOracles.bloomFoldQCoef aSer a2 24
    let qD := bloomFoldQCoefD aSerD a2D 24
    for k in [0:24] do
      fol := fol.note s!"Q[{k}]" qD[k]! qF[k]!
    fol := fol.note s!"DDaM" (bloomFoldDDaMD qD kD)
      (Tropical.Testing.ArrowOracles.bloomFoldDDaM qF kappa)
  let fams := #[("serOnly M(1,a+1,κ)", ser), ("crossing Γ★/CF", cro),
                ("coincident dCoef/Φ/cexpm1", coi), ("fold Q/DDaM", fol)]
  let mut worst : Float := 0.0
  let mut poisoned := 0
  let mut lits := 0
  let mut litMoved := 0
  IO.println s!"        {n} Halton configs over the shipped register (κ to |170|, both sides of |a|=½):"
  for (label, d) in fams do
    let tot := d.litSame + d.litOff1 + d.litWorse
    IO.println s!"        {label}: worst rel {d.worst} (at {d.at?}) · literals {d.litSame}/{tot} identical, {d.litOff1} off-by-one, {d.litWorse} further (worst {d.worstLit}) · poisoned {d.poisoned}"
    worst := max worst d.worst
    poisoned := poisoned + d.poisoned
    lits := lits + tot
    litMoved := litMoved + d.litOff1 + d.litWorse
  let movedFrac := litMoved.toFloat / (max lits 1).toFloat
  let tEnd ← IO.monoMsNow
  -- the bake layer's cost, running BOTH paths, printed and not gated
  IO.println s!"        emitted literals that moved at all: {litMoved}/{lits} ({movedFrac}) · both paths in {tEnd - tStart} ms"
  if poisoned == 0 && worst < 1.0e-9 && movedFrac < 1.0e-2 then
    passGate "exact-values"
      s!"the flipped bake-time values track the float path they replaced to {worst} relative over the shipped register, moving {litMoved} of {lits} emitted literals ({movedFrac}) — the values flip changed the arithmetic, not the program"
  else
    failGate "exact-values"
      s!"worst={worst} poisoned={poisoned} litMoved={litMoved}/{lits} ({movedFrac})"

-- ── the served bake surface ───────────────────────────────────────────────────

/-- The authored 2π, as the DECIMAL `Playground` spells it (and `twoPiE` with
    it), built from integers so no `JsonNumber` source literal is involved — the
    linux-x86 miscompile bites those. -/
private def twoPiDecimal : DyadicI :=
  DyadicI.div (DyadicI.ofInt 6283185307179586) (DyadicI.ofNat (10 ^ 15))

/-- The first partial's emitted ω for a given loop-transit count `N`:
    `2π·f₁ = 2π·SR/(N + ½) = 2π·88200/(2N+1)`. Built from the definition of the
    cliff rather than from the builder, so it discriminates instead of agreeing
    with a transcription. -/
private def stringOmega1 (n : Nat) : Float :=
  (DyadicI.div (DyadicI.mul twoPiDecimal (DyadicI.ofNat 88200))
               (DyadicI.ofNat (2 * n + 1))).toFloat

/-- One site's differential outcome: literals compared, literals that moved, and
    the worst relative distance. -/
private structure SiteDiff where
  n     : Nat := 0
  moved : Nat := 0
  worst : Float := 0.0
deriving Inhabited

/-- Score one emitted field against the libm expression it replaced. The
    allowance is the magnitude-aware one `exact-values` uses: within ONE unit of
    litF's 12th place, or within 1e-11 relatively (litF is only a faithful
    12-place quantizer while |x| ≤ 9007.2, and a reverb's ω runs past that). -/
private def SiteDiff.note (d : SiteDiff) (emitted : Option Float) (ref : Float) : SiteDiff :=
  match emitted with
  | none => { d with n := d.n + 1, moved := d.moved + 1, worst := 1.0e30 }
  | some v =>
    let delta := litFMantissa v - litFMantissa ref
    let ad := if delta < 0 then -delta else delta
    let rel := if ref == 0.0 then (if v == 0.0 then 0.0 else 1.0)
               else Float.abs (v - ref) / Float.abs ref
    let moved := !(ad ≤ 1 || rel ≤ 1.0e-11)
    { n := d.n + 1, moved := d.moved + (if moved then 1 else 0), worst := max d.worst rel }

/-- THE PLAYGROUND-BAKE gate. `Playground.lean` is the SERVED bake surface —
    `gong`, `string`, `resonator`, `reverb`, `filter` — and it carried eleven
    `libm` calls, two of them structural, while the whole bloom/Γ family the
    campaign was built around sits behind the WITHHELD `bloomgong`. This pins the
    three things a successor needs and nothing else can: the cliffs' verdicts,
    the value differential, and the totality of the emit funnel.

    Every arm here observes what a BUILDER EMITS. An earlier cut of this gate had
    three arms that could not fail: its differential recomputed the builders'
    expressions inline and compared that against libm (testing the
    transcription, not the emitter); its band-edge check was closed-form
    arithmetic evaluated entirely inside the gate, with no dependency on
    `Playground.lean` at all; and its tie-rounding probe read a mode COUNT that
    `min 48` saturates, so the very cliff it documented was invisible to it.
    Each is now taken through `Playground`'s own output.

    1. THE COUNT CLIFF, observed where it is visible. `defaultStringModes`'
       loop-transit count is an exact rational round, not a `Float.round` of an
       f64 quotient. The one MEASURED disagreement is `f0 = 2.24`, where
       `44100/2.24 = 19687.5` exactly and the double is `19687.4999…` — but at
       that f0 the emitted COUNT is 48 either way, because `min 48` saturates
       it. What the cliff actually moves is the whole pole table, so the probe
       reads the first partial's emitted ω and checks it against BOTH candidate
       transit counts: it must equal the N = 19688 value and differ from the
       N = 19687 one.
    2. THE EMIT/SKIP CLIFF. The `g > 0` fork's policy — emit on a CERTIFIED
       positive verdict, drop otherwise — must reproduce the incumbent on every
       reachable input. `ρ = 0` is the only reachable overlap and it must drop
       everything; `ρ < 0` likewise; `f0 ≤ 0` is the one place the carrier FORCES
       a change (an `∞` saturating into 48 undamped DC modes becomes an empty
       bank), asserted so the change is a recorded decision and not a surprise.
    3. THE BAND EDGE, on emitted output. `f_k < SR/2` is provably implied by
       `k ≤ ⌊span/2⌋`, so the conjunct in the emitter never fires — but proving
       that with arithmetic inside the gate proves nothing ABOUT the emitter.
       This reads every ω `defaultStringModes` emits across the served f0 range
       and requires it under `2π·SR/2`, so a future `kmax` change that re-opened
       the band edge turns this red instead of shipping partials above Nyquist.
    4. THE DIFFERENTIAL + TOTALITY, against the real builders. `resonatorBank`,
       `reverbRoom` and `filterPair` are CALLED; the libm expressions they
       replaced are evaluated here, one floor outside the module under test —
       which is also what keeps `Playground.lean` free of platform trig, so
       `exact-corpse` can read its generated C with no exemption. The moved
       counts are frozen: a value that starts moving is a red gate, not a
       printed number nobody reads. -/
def runExactPlayground : IO Bool := do
  let prepared := freezeBuild {} do
    let defaultModes ← Tropical.Playground.Compiler.defaultStringModes
      (196, 0) (996, 3)
    let highModes ← Tropical.Playground.Compiler.defaultStringModes
      (2000, 0) (996, 3)
    let lowModes ← Tropical.Playground.Compiler.defaultStringModes
      (20, 0) (996, 3)
    let quantBank ← Tropical.Playground.Compiler.defaultStringModes
      (224, 2) (996, 3)
    let zeroRho ← Tropical.Playground.Compiler.defaultStringModes
      (196, 0) (0, 0)
    let negativeRho ← Tropical.Playground.Compiler.defaultStringModes
      (196, 0) (-5, 1)
    let zeroFrequency ← Tropical.Playground.Compiler.defaultStringModes
      (0, 0) (996, 3)
    let bandBanks ← (Array.range 40).mapM fun i =>
      Tropical.Playground.Compiler.defaultStringModes
        (20 + 60 * Int.ofNat i, 0) (996, 3)
    let resonator ← Tropical.Playground.Compiler.bakedResonatorProbe 512
    let reverb ← Tropical.Playground.Compiler.bakedReverbProbe 32
    let filterLn80 ← Tropical.Playground.Compiler.bakedFilterLn80
    pure (defaultModes.size, highModes.size, lowModes.size, quantBank,
      zeroRho.size, negativeRho.size, zeroFrequency.size, bandBanks,
      resonator, reverb, filterLn80)
  let (frozen, cDefault, cHigh, cLow, quantBank, cZeroRho, cNegRho,
      cZeroF0, bandBanks, resonator, reverb, filterLn80) ← match prepared with
    | .error error =>
        return ← failGate "exact-playground" s!"arena-native probes: {firstLine error}"
    | .ok (frozen, (cDefault, cHigh, cLow, quantBank, cZeroRho, cNegRho,
        cZeroF0, bandBanks, resonator, reverb, filterLn80)) =>
      pure (frozen, cDefault, cHigh, cLow, quantBank, cZeroRho, cNegRho,
        cZeroF0, bandBanks, resonator, reverb, filterLn80)
  let constants := sigConstTable frozen
  let fold := fun signal => (sigConstDFrom? constants signal).map (·.toFloat)
  -- (1) the count cliff, read where it is observable
  let quantOm := (quantBank.toList.head?).bind (fun mode => fold mode.omega)
  let wantTie := stringOmega1 19688      -- the exact half-away-from-zero answer
  let notTie  := stringOmega1 19687      -- what the f64 quotient rounded to
  let tieOk := match quantOm with
    | some w => litFMantissa w == litFMantissa wantTie && litFMantissa w != litFMantissa notTie
    | none => false
  -- (2) the fork policy
  -- (3) the band edge, on EMITTED ω across the served f0 range
  let nyquist := (DyadicI.mul twoPiDecimal (DyadicI.ofNat 22050)).toFloat
  let mut bandViolations := 0
  let mut bandModes := 0
  for bank in bandBanks do
    for m in bank do
      bandModes := bandModes + 1
      match fold m.omega with
      | some w => if !(w < nyquist) then bandViolations := bandViolations + 1
      | none => bandViolations := bandViolations + 1
  -- (4) the differential, against the builders themselves
  let mut res : SiteDiff := {}
  let mut i := 0
  for m in resonator do
    let k := (i + 1).toFloat
    i := i + 1
    res := res.note (fold m.sigma) (1.0 + 0.4 * k)
    res := res.note (fold m.cre) (1.0 / Float.pow k 1.1)
  let mut rev : SiteDiff := {}
  let twoPiF := 6.283185307179586
  let mut j := 0
  for m in reverb do
    let jf := j.toFloat
    j := j + 1
    let fq := 60.0 * Float.pow (6000.0 / 60.0) (jf / 31.0)
    let ph := twoPiF * (0.6180339887 * jf)
    rev := rev.note (fold m.omega) (twoPiF * fq)
    rev := rev.note (fold m.cre) (Float.cos ph)
    rev := rev.note (fold m.cim) (Float.sin ph)
  -- filterPair's one authored transcendental: the `ln 80` its Q mapping is
  -- written in terms of, against the libm value it replaced
  let mut flt : SiteDiff := {}
  flt := flt.note (fold filterLn80) (Float.log 80.0)
  let mut poison := 0
  for m in resonator do
    if (fold m.cre).isNone then poison := poison + 1
  for m in reverb do
    if (fold m.omega).isNone then poison := poison + 1
  IO.println s!"        string count: f0=196 → {cDefault} (want 48) · f0=2000 → {cHigh} (want 11) · f0=20 → {cLow} (want 48)"
  IO.println s!"        tie cliff   : f0=2.24 first ω {quantOm} — equals the N=19688 answer {wantTie}, differs from N=19687 {notTie} : {tieOk}"
  IO.println s!"        fork policy : ρ=0 → {cZeroRho} · ρ=−0.5 → {cNegRho} · f0=0 → {cZeroF0} (all want 0)"
  IO.println s!"        band edge   : {bandViolations} of {bandModes} EMITTED ω at or above 2π·SR/2 (want 0 — the conjunct is redundant, proved on output)"
  -- printed in units of 1e-18 because Lean's Float formatter shows anything
  -- under ~1e-6 as "0.000000", which is exactly the range this differential
  -- lives in — an unreadable number is not a measurement
  IO.println s!"        differential: resonatorBank {res.moved}/{res.n} moved (worst rel {res.worst * 1.0e18}e-18) · reverbRoom {rev.moved}/{rev.n} ({rev.worst * 1.0e18}e-18) · filterPair {flt.moved}/{flt.n} ({flt.worst * 1.0e18}e-18)"
  IO.println s!"        poison      : {poison} (want 0 — litOfD's lit-0 arm is dead)"
  let countsOk := cDefault == 48 && cHigh == 11 && cLow == 48
  let forkOk := cZeroRho == 0 && cNegRho == 0 && cZeroF0 == 0
  -- FROZEN ON THE EMITTED MANTISSA, not on the relative distance. The relative
  -- figure is printed and deliberately not gated: at small emitted magnitudes it
  -- is dominated by `litF`'s own 12-decimal grid rather than by the carrier —
  -- the resonator's worst, 4.7e-10, is one half-unit of the 12th place on an amp
  -- of 1.3e-3 (`k^{-1.1}` at k = 512), and would be there with a perfect
  -- carrier. Bounding it would be bounding `litF`'s resolution, which
  -- `exact-quantize` already owns. What this gate is for is whether the EMITTED
  -- PROGRAM moved, and that is the mantissa.
  let diffOk := res.moved == 0 && rev.moved == 0 && flt.moved == 0
  if countsOk && tieOk && forkOk && bandViolations == 0 && bandModes > 500
      && diffOk && poison == 0 then
    passGate "exact-playground"
      s!"the served bake surface decides in exact arithmetic: the string's transit count is an Int round (pinned where the cliff is VISIBLE — the emitted pole table, not the mode count `min 48` saturates), the emit/skip fork emits only on a certified verdict, no emitted ω reaches Nyquist over {bandModes} partials, the three baked builders emit LITERALLY what the libm they replaced emitted ({res.n + rev.n + flt.n} coefficients, 0 moved), and none of it came from poison"
  else
    failGate "exact-playground"
      s!"counts={countsOk} ({cDefault}/{cHigh}/{cLow}) tie={tieOk} fork={forkOk} ({cZeroRho}/{cNegRho}/{cZeroF0}) band={bandViolations}/{bandModes} diff={diffOk} (res {res.moved}/{res.n} rev {rev.moved}/{rev.n} flt {flt.moved}/{flt.n}) poison={poison}"

-- ── the one-way door ──────────────────────────────────────────────────────────

/-- Every `libm` entry point Lean's `Float` API exposes, by its C name. The list
    is written against the GENERATED C rather than against Lean source, which is
    the whole point of the change: a source-text scan can only ever recognise the
    spellings its author thought of, and an earlier cut of this gate recognised
    only the `Float.`-qualified ones. Dot notation (`x.exp` on a `Float`) walked
    straight past it, and so did `Float.atan`, `asin`, `acos`, `asinh`, `acosh`,
    `atanh` and `cbrt` — seven `@[extern]` calls, none of which contains any
    banned substring (`Float.tan` is not a substring of `Float.atan`). By the
    time Lean has emitted C there is one spelling left, and it is this one. -/
private def libmSymbols : Array String :=
  #["exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "sqrt", "cbrt", "pow", "hypot", "fmod",
    "floor", "ceil", "round", "trunc", "lgamma", "tgamma"]

/-- Does `line` call `name` as a C function — the identifier immediately
    followed by `(`, and not preceded by an identifier character, so
    `lean_float_pow` and `my_exp(` do not match?

    Written on `String.splitOn` rather than on a `List Char` walk: this runs over
    every line of every production module's generated C, and the char-list form
    cost the suite half a minute by itself. Each split piece ends exactly where
    an occurrence begins, so the preceding character is that piece's last one. -/
private def callsC (line name : String) : Bool := Id.run do
  let parts := line.splitOn (name ++ "(")
  if parts.length == 1 then return false
  for p in parts.dropLast do
    match p.back? with
    | none => return true                    -- occurrence at the start of the line
    | some c => if !(c.isAlphanum || c == '_') then return true
  return false

/-- THE CORPSE GATE (P3). The bake layer's `libm` exile, ENFORCED rather than
    documented. Every other gate in this file says the exact carrier agrees with
    the float path it replaced; this one says the float path is GONE from the
    compiler — so a successor cannot reintroduce one by habit and have every
    behavioural gate stay green (they would: a 1-ulp bake difference is invisible
    to a threshold, which is the whole reason this campaign exists).

    It reads the GENERATED C of each production bake module, not its Lean source.
    That is a deliberate upgrade over the source scan it replaces, which was
    exactly as good as its author's list of spellings and no better — dot
    notation and seven of Lean's own `Float` externs walked past it. After
    elaboration there is one spelling per call, so the check stops depending on
    how the call was written.

    The source is still read for the RETIRED DEFINITIONS: `CplxB`'s three
    transcendental methods and the nine functions that use them must be absent
    from every production module and present in `Tropical.Testing.ArrowOracles`,
    one floor outside the compiler, where they are the INDEPENDENT oracle (the
    DUT is exact arithmetic; the reference is the platform's `libm`).

    A missing `.c` is a FAILURE, never a silent pass: this gate reporting "zero
    libm sites" because it read zero files would be the worst outcome available
    to it. -/
def runExactCorpse : IO Bool := do
  let sourceDirs := #[
    ("EmitArrow", "lean/Tropical/EmitArrow"),
    ("EmitArrow/Modal", "lean/Tropical/EmitArrow/Modal"),
    ("Playground", "lean/Tropical/Playground")]
  let mut mods : Array (String × String) := #[
    ("Playground", "lean/Tropical/Playground.lean")]
  for (modulePrefix, dir) in sourceDirs do
    for entry in ← (System.FilePath.mk dir).readDir do
      if entry.fileName.endsWith ".lean" then
        let stem := (entry.fileName.dropEnd 5).toString
        mods := mods.push (s!"{modulePrefix}/{stem}", s!"{dir}/{stem}.lean")
  mods := mods.qsort fun a b => decide (a.1 < b.1)
  let corpseNames := #["def lgammaB", "def bloomM1 ", "def bloomCF ", "def cexpm1B",
                       "def bloomGammaStar ", "def bloomPhiKappaOverG (",
                       "def bloomDCoef ", "def bloomFoldQCoef ", "def bloomFoldDDaM ",
                       "def sigmaInterval?", "def abs (a : CplxB)",
                       "def exp (z : CplxB)", "def log (z : CplxB)"]
  let mut hits : Array String := #[]
  let mut corpses : Array String := #[]
  let mut missing : Array String := #[]
  let mut oracleMissing : Array String := #[]
  let mut scanned := 0
  for (modName, srcPath) in mods do
    -- the compiler's own answer: what did this module actually call?
    let cPath := s!"lean/.lake/build/ir/Tropical/{modName}.c"
    if !(← System.FilePath.pathExists cPath) then
      missing := missing.push cPath
    else
      for line in ← IO.FS.lines cPath do
        scanned := scanned + 1
        for sym in libmSymbols do
          if callsC line sym then hits := hits.push s!"{modName}.c: {sym}()"
    -- the source, for the retired definitions
    let src ← IO.FS.readFile srcPath
    for name in corpseNames do
      if (src.splitOn name).length != 1 then corpses := corpses.push s!"{srcPath}: {name}"
  let oraclePath := "lean/Tropical/Testing/ArrowOracles.lean"
  let oracleSource ← IO.FS.readFile oraclePath
  for name in corpseNames do
    if name != "def sigmaInterval?" && (oracleSource.splitOn name).length == 1 then
      oracleMissing := oracleMissing.push name
  IO.println s!"        {mods.size} production bake modules, {scanned} lines of GENERATED C (the compiler's own answer, not the source's spelling)"
  IO.println s!"        libm call sites: {hits.size} · retired Float-tier definitions still in production: {corpses.size} · oracle definitions missing: {oracleMissing.size} · unreadable modules: {missing.size}"
  if hits.isEmpty && corpses.isEmpty && oracleMissing.isEmpty && missing.isEmpty && scanned > 1000 then
    passGate "exact-corpse"
      s!"no libm call survives the production bake graph — checked in the EMITTED C across {mods.size} modules and {scanned} lines, so no spelling (dot notation, an unlisted Float extern, an operator) can hide one; and no retired Float-tier definition remains, the oracle tier having moved outside the compiler to Tropical.Testing"
  else
    failGate "exact-corpse"
      s!"libm {hits} · corpses {corpses} · oracleMissing {oracleMissing} · missing {missing} · scanned {scanned}"

end Tropical.Tropicaltest.ExactGates
