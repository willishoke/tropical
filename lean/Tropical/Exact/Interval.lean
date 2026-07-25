import Tropical.Exact.Dyadic
import Lean.Data.Json

/-!
# Tropical.Exact.Interval — the certified enclosure

`DyadicI` is a pair of `Dyadic` endpoints, every operation rounding OUTWARD at
a fixed working precision, so the interval always ENCLOSES the real value of
the expression it was built from. Two things fall out that `Float` cannot give:

* **Determinism.** Every operation is exact `Int` arithmetic plus a directed
  shift. The same expression produces the same endpoints on every platform —
  no `libm`, no fused multiply-add, no compiler reassociation. (The
  `JsonNumber` CI incident is the standing evidence that this layer's platform
  variance bites.)
* **Certified comparison.** `cmp` answers `lt`/`gt` only when the two
  enclosures are DISJOINT; otherwise it answers `overlap`, and the caller must
  say out loud what it does about that. A `Float` comparison near a threshold
  answers confidently and can be wrong; this one cannot be wrong, only silent.

Division by an interval containing zero, and the square root of an interval
lying entirely below zero, produce `poison` — an absorbing state that
propagates and is visible at the end (`ok = false`), rather than an `inf`/`nan`
that keeps flowing and quietly decides something.

Precision is a knob, not a promise: `workingPrec` sets how many mantissa bits
each rounding keeps. Enclosure holds at ANY setting; the setting only decides
how often `cmp` has to answer `overlap`.
-/

namespace Tropical.Exact

/-- Mantissa bits kept by each rounding in `DyadicI`. Generous: these are
    bake-time constants computed once per compile, not a per-sample datapath.
    Correctness (enclosure) is independent of this number — only the frequency
    of `overlap` verdicts depends on it. -/
def workingPrec : Nat := 128

/-- Precision for the module-level transcendental constants (π, ln 2). Above
    `workingPrec` by enough that argument reduction of a large argument still
    lands the reduced value with a full working mantissa. -/
def constPrec : Nat := 300

/-- A certified three-way comparison verdict. `overlap` is not a failure — it
    is the honest answer when two enclosures intersect, and the branch that
    consumes it must state what it does (take the conservative side, widen, or
    drop with a witness). -/
inductive Sep where
  | lt        -- certainly less
  | gt        -- certainly greater
  | overlap   -- enclosures intersect: not separated at this precision
deriving Inhabited, Repr, DecidableEq

/-- An enclosure `[lo, hi]`. `ok = false` is poison: an operation was asked for
    a value that does not exist (a reciprocal across zero) and every downstream
    operation inherits it. `Inhabited`'s default is poison, deliberately. -/
structure DyadicI where
  lo : Dyadic
  hi : Dyadic
  ok : Bool

namespace DyadicI

/-- The absorbing failure state. -/
def poison : DyadicI := ⟨0, 0, false⟩

instance : Inhabited DyadicI := ⟨poison⟩

/-- A degenerate (exact) interval. -/
def exact (d : Dyadic) : DyadicI := ⟨d, d, true⟩

def zero : DyadicI := exact 0
def one  : DyadicI := exact 1

def ofInt (n : Int) : DyadicI := exact (Dyadic.ofInt n)
def ofNat (n : Nat) : DyadicI := exact (Dyadic.ofInt (n : Int))

/-- A finite `Float` enters EXACTLY (it is a dyadic); a non-finite one poisons. -/
def ofFloat (x : Float) : DyadicI :=
  match Dyadic.ofFloat? x with
  | some d => exact d
  | none   => poison

/-- Is this interval a single certified value? -/
def isExact (x : DyadicI) : Bool := x.ok && Dyadic.ble x.lo x.hi && Dyadic.ble x.hi x.lo

/-- Build from two endpoints in either order. -/
def ofPair (lo hi : Dyadic) : DyadicI :=
  if Dyadic.ble lo hi then ⟨lo, hi, true⟩ else ⟨hi, lo, true⟩

private def bin (x y : DyadicI) (f : Unit → DyadicI) : DyadicI :=
  if x.ok && y.ok then f () else poison

private def un (x : DyadicI) (f : Unit → DyadicI) : DyadicI :=
  if x.ok then f () else poison

/-! The arithmetic is written PRECISION-PARAMETERIC (`…At prec`) with the
    everyday operations fixed at `workingPrec`. Enclosure holds at any `prec`;
    the parameter exists for the two places a caller genuinely needs more than
    the working precision — certifying the module constants ABOVE their own
    300 bits, and any future argument reduction that must survive a large
    quotient. -/

private def rdA (prec : Nat) (a : Dyadic) : Dyadic := Dyadic.roundRel .down prec a
private def ruA (prec : Nat) (a : Dyadic) : Dyadic := Dyadic.roundRel .up   prec a

def neg (x : DyadicI) : DyadicI := un x fun _ => ⟨-x.hi, -x.lo, true⟩

def addAt (prec : Nat) (x y : DyadicI) : DyadicI :=
  bin x y fun _ => ⟨rdA prec (x.lo + y.lo), ruA prec (x.hi + y.hi), true⟩

def subAt (prec : Nat) (x y : DyadicI) : DyadicI :=
  bin x y fun _ => ⟨rdA prec (x.lo - y.hi), ruA prec (x.hi - y.lo), true⟩

/-- Interval multiply by sign case analysis — two multiplies in the common
    case (both operands of known sign), four only when both straddle zero. -/
def mulAt (prec : Nat) (x y : DyadicI) : DyadicI :=
  bin x y fun _ =>
    let dn := fun (a b : Dyadic) => rdA prec (a * b)
    let up := fun (a b : Dyadic) => ruA prec (a * b)
    let xl := x.lo; let xh := x.hi; let yl := y.lo; let yh := y.hi
    if !xl.isNeg then                                   -- x ≥ 0
      if !yl.isNeg then      ⟨dn xl yl, up xh yh, true⟩ -- y ≥ 0
      else if !yh.isPos then ⟨dn xh yl, up xl yh, true⟩ -- y ≤ 0
      else                   ⟨dn xh yl, up xh yh, true⟩ -- y straddles
    else if !xh.isPos then                              -- x ≤ 0
      if !yl.isNeg then      ⟨dn xl yh, up xh yl, true⟩
      else if !yh.isPos then ⟨dn xh yh, up xl yl, true⟩
      else                   ⟨dn xl yh, up xl yl, true⟩
    else                                                -- x straddles
      if !yl.isNeg then      ⟨dn xl yh, up xh yh, true⟩
      else if !yh.isPos then ⟨dn xh yl, up xl yl, true⟩
      else ⟨Dyadic.dmin (dn xl yh) (dn xh yl), Dyadic.dmax (up xl yl) (up xh yh), true⟩

def add (x y : DyadicI) : DyadicI := addAt workingPrec x y
def sub (x y : DyadicI) : DyadicI := subAt workingPrec x y
def mul (x y : DyadicI) : DyadicI := mulAt workingPrec x y

/-- Does the enclosure contain zero (so a reciprocal does not exist over it)? -/
def straddlesZero (x : DyadicI) : Bool :=
  !x.ok || (Dyadic.ble x.lo 0 && Dyadic.ble 0 x.hi)

/-- `1/x`. Poison when the enclosure contains zero — the whole point: a
    reciprocal the carrier cannot certify does not silently become `inf`.
    `1/·` is decreasing on either side of zero, so the endpoints swap. -/
def invAt (prec : Nat) (x : DyadicI) : DyadicI :=
  if straddlesZero x then poison
  else
    match Dyadic.divRel? .down prec 1 x.hi, Dyadic.divRel? .up prec 1 x.lo with
    | some lo, some hi => ⟨lo, hi, true⟩
    | _, _ => poison        -- unreachable: `straddlesZero` already excluded 0

def inv (x : DyadicI) : DyadicI := invAt workingPrec x

def divAt (prec : Nat) (x y : DyadicI) : DyadicI := mulAt prec x (invAt prec y)
def div (x y : DyadicI) : DyadicI := divAt workingPrec x y

/-- Exact multiply by a power of two — no rounding, no width growth. -/
def shift (x : DyadicI) (k : Int) : DyadicI :=
  un x fun _ => ⟨x.lo <<< k, x.hi <<< k, true⟩

/-- `|x|` as an enclosure: floored at zero when the interval straddles. -/
def abs (x : DyadicI) : DyadicI :=
  un x fun _ =>
    if !x.lo.isNeg then x
    else if !x.hi.isPos then ⟨-x.hi, -x.lo, true⟩
    else ⟨0, Dyadic.dmax (-x.lo) x.hi, true⟩

/-- `√x`. Poison for an enclosure lying entirely below zero; an interval with
    `lo < 0 ≤ hi` clamps its floor to zero — the honest enclosure of `√` over
    the part that exists. -/
def sqrt (x : DyadicI) : DyadicI :=
  un x fun _ =>
    if x.hi.isNeg then poison
    else
      let lo := if x.lo.isNeg then (0 : Dyadic) else x.lo
      match Dyadic.sqrtRel? .down workingPrec lo, Dyadic.sqrtRel? .up workingPrec x.hi with
      | some a, some b => ⟨a, b, true⟩
      | _, _ => poison      -- unreachable: the negative part was clamped away

def square (x : DyadicI) : DyadicI := mul x x

/-- Certified comparison. Answers `lt`/`gt` ONLY on disjoint enclosures;
    `overlap` otherwise (poison included — an undefined value is not separated
    from anything). -/
def cmp (x y : DyadicI) : Sep :=
  if !x.ok || !y.ok then .overlap
  else if Dyadic.blt x.hi y.lo then .lt
  else if Dyadic.blt y.hi x.lo then .gt
  else .overlap

/-- Certified `x < y` — `false` when the enclosures overlap. -/
def certLt (x y : DyadicI) : Bool := cmp x y == .lt
/-- Certified `x > y` — `false` when the enclosures overlap. -/
def certGt (x y : DyadicI) : Bool := cmp x y == .gt

/-- The sign of the whole enclosure, or `overlap` if it contains zero. -/
def sign (x : DyadicI) : Sep := cmp x zero

def min (x y : DyadicI) : DyadicI :=
  bin x y fun _ => ⟨Dyadic.dmin x.lo y.lo, Dyadic.dmin x.hi y.hi, true⟩
def max (x y : DyadicI) : DyadicI :=
  bin x y fun _ => ⟨Dyadic.dmax x.lo y.lo, Dyadic.dmax x.hi y.hi, true⟩

/-- The exact midpoint — a DETERMINISTIC representative, used where a single
    value must be picked (argument-reduction quotients, the `Float` a literal
    is finally emitted from). Not a claim about accuracy. -/
def mid (x : DyadicI) : Dyadic := (x.lo + x.hi) <<< (-1 : Int)

/-- The enclosure's width `hi − lo`, exact. -/
def width (x : DyadicI) : Dyadic := x.hi - x.lo

/-- The midpoint as a `Float` — the value that reaches `litF`. -/
def toFloat (x : DyadicI) : Float := (mid x).toFloat

/-- Nearest integer to the midpoint (half away from zero) — the deterministic
    quotient of an argument reduction. -/
def roundToInt (x : DyadicI) : Int :=
  let m := mid x
  let half : Dyadic := Dyadic.ofIntWithPrec 1 1
  if m.isNeg then (m - half).toIntCeil else (m + half).toIntFloor

/-- Widen by `2^k` on each side — the honest way to admit an error bound
    reasoned about outside the carrier (a truncated series tail). -/
def widen (x : DyadicI) (k : Int) : DyadicI :=
  un x fun _ =>
    let d : Dyadic := Dyadic.ofIntWithPrec 1 (-k)
    ⟨rdA workingPrec (x.lo - d), ruA workingPrec (x.hi + d), true⟩

/-- Hull of two enclosures (the union's smallest enclosing interval). -/
def hull (x y : DyadicI) : DyadicI :=
  bin x y fun _ => ⟨Dyadic.dmin x.lo y.lo, Dyadic.dmax x.hi y.hi, true⟩

/-- A DECIMAL literal (`mantissa · 10^{−exponent}`, the shape `JsonNumber` and
    every authored `lit` carry) as a tight enclosure. Decimals are not dyadic,
    so this is where the authoring layer's exactness genuinely ends — and the
    enclosure says so, to the working precision, instead of pretending. -/
def ofJsonNumber (n : Lean.JsonNumber) : DyadicI :=
  if n.exponent == 0 then ofInt n.mantissa
  else div (ofInt n.mantissa) (ofNat (10 ^ n.exponent))

/-- Diagnostics: the midpoint as a float plus the enclosure width's binary
    exponent (how many bits are certified). -/
def render (x : DyadicI) : String :=
  if !x.ok then "poison"
  else
    let w := width x
    let bits := if w.isZero then "exact" else s!"2^{w.magBits}"
    s!"{x.toFloat} ±{bits}"

end DyadicI

end Tropical.Exact
