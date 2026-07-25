/-!
# Tropical.Exact.Dyadic — the exact carrier, on core's verified `Dyadic`

Lean 4 core ships `Dyadic` (`Init.Data.Dyadic`): a dyadic rational held as an
ODD `Int` mantissa times a power of two, with `CommRing`/`OrderedRing`
instances, a linear order, and DIRECTED rounding (`roundDown`/`roundUp`) whose
bracketing is proved (`Dyadic.roundDown_le`, `Dyadic.le_roundUp`). That is the
carrier this campaign wanted, already built and already trusted: addition,
subtraction and multiplication are EXACT, and every lossy step names its
direction.

This module adds only what the bake layer needs on top and core does not have:

* **Relative** (floating) rounding. Core's `roundDown x prec` rounds to a
  multiple of `2^{-prec}` — FIXED-POINT. A series whose terms decay past
  `2^{-prec}` would flush to zero, so `roundRel` first reads the value's
  magnitude and rounds to a number of SIGNIFICANT bits instead.
* Exact `Float` ⇆ `Dyadic` (a finite IEEE-754 double *is* a dyadic), the
  nearest-`Float` projection, and the exact decimal quantization `litF` means.
* Directed division and square root, both built from one `Int` operation.

These extend the root `Dyadic` namespace so dot-notation works on core values.
A future core addition of the same name is a loud duplicate-declaration error,
not a silent shadow.
-/

namespace Tropical.Exact

/-- Which way a lossy operation rounds. Every rounding here is DIRECTED —
    never round-to-nearest — because an interval's endpoints must each move
    outward for the enclosure to hold. -/
inductive RoundDir where
  | down    -- toward −∞
  | up      -- toward +∞
deriving Inhabited, Repr, DecidableEq

/-- The opposite direction: `(−x)` rounded `d` is `−(x rounded d.flip)`. -/
def RoundDir.flip : RoundDir → RoundDir
  | .down => .up
  | .up   => .down

/-- Flipping is an involution — the outward-rounding argument for interval
    endpoints is exactly "negating swaps the direction, and swapping twice
    is identity." -/
theorem RoundDir.flip_flip (d : RoundDir) : d.flip.flip = d := by
  cases d <;> rfl

/-- Bit length of an `Int`'s magnitude: `0` for zero, else `⌊log₂|n|⌋ + 1`. -/
def intBitLen (n : Int) : Nat :=
  if n == 0 then 0 else Nat.log2 n.natAbs + 1

private def sqrtGo (n x : Nat) : Nat :=
  let y := (x + n / x) / 2
  if _h : y < x then sqrtGo n y else x
termination_by x
decreasing_by exact _h

/-- Integer square root `⌊√n⌋` by Newton descent from an over-estimate
    (`2^(⌊log₂n⌋/2 + 1) ≥ √n` for every `n ≥ 2`); the iteration decreases
    monotonically from above, so the first non-decrease is the answer. -/
def natSqrt (n : Nat) : Nat :=
  if n < 2 then n else sqrtGo n (1 <<< (Nat.log2 n / 2 + 1))

end Tropical.Exact

namespace Dyadic

open Tropical.Exact

/-- The odd mantissa, or `none` at zero. -/
def mantissa? : Dyadic → Option Int
  | .zero => none
  | .ofOdd n _ _ => some n

/-- `⌈log₂|x|⌉` in the sense `2^(magBits x − 1) ≤ |x| < 2^(magBits x)`; `0` at
    zero. The number of bits ABOVE the binary point, signed. -/
def magBits : Dyadic → Int
  | .zero => 0
  | .ofOdd n k _ => (intBitLen n : Int) - k

def isZero : Dyadic → Bool
  | .zero => true
  | _ => false

def isNeg (x : Dyadic) : Bool := blt x 0
def isPos (x : Dyadic) : Bool := blt 0 x

/-- `⌊x⌋` as an `Int` (core's `roundDown … 0` lands at precision ≤ 0, i.e. an
    integer, so the mantissa shifts back up exactly). -/
def toIntFloor (x : Dyadic) : Int :=
  match roundDown x 0 with
  | .zero => 0
  | .ofOdd n k _ => n <<< (-k).toNat

/-- `⌈x⌉` as an `Int`. -/
def toIntCeil (x : Dyadic) : Int :=
  match roundUp x 0 with
  | .zero => 0
  | .ofOdd n k _ => n <<< (-k).toNat

def dmin (a b : Dyadic) : Dyadic := if blt a b then a else b
def dmax (a b : Dyadic) : Dyadic := if blt a b then b else a
def dabs (a : Dyadic) : Dyadic := if blt a 0 then -a else a

/-- Directed rounding to a fixed number of SIGNIFICANT bits (core's
    `roundDown`/`roundUp` are fixed-point: they round to a multiple of
    `2^{-prec}`). Reading the magnitude first turns them into the floating
    rounding a wide-dynamic-range computation needs — a term at `2^{-900}`
    keeps its `bits` of precision instead of flushing to zero. -/
def roundRel (dir : RoundDir) (bits : Nat) (x : Dyadic) : Dyadic :=
  match x with
  | .zero => .zero
  | .ofOdd n k _ =>
    let bl := intBitLen n
    if bl ≤ bits then x
    else
      let prec := k - ((bl - bits : Nat) : Int)
      match dir with
      | .down => roundDown x prec
      | .up   => roundUp x prec

/-- Round to `bits` significant bits, TO NEAREST, ties to even — the one
    non-directed rounding, used only where a single nearest value is wanted
    (`toFloat`). Exact `Int` arithmetic on the magnitude, so it is
    sign-symmetric. -/
def roundNearestRel (bits : Nat) (x : Dyadic) : Dyadic :=
  match x with
  | .zero => .zero
  | .ofOdd n k _ =>
    let bl := intBitLen n
    if bl ≤ bits || bits == 0 then x
    else
      let s := bl - bits
      let negative := n < 0
      let mag := n.natAbs
      let q := mag >>> s
      let rem := mag - (q <<< s)
      let q' := if rem * 2 > 1 <<< s then q + 1
                else if rem * 2 < 1 <<< s then q
                else if q % 2 == 1 then q + 1 else q
      ofIntWithPrec (if negative then -(q' : Int) else (q' : Int)) (k - (s : Int))

/-- Directed quotient `a / b` at `bits` significant bits. `none` for `b = 0` —
    the value does not exist, and the whole point of this campaign is that a
    nonexistent value is REFUSED rather than fabricated as `0` (the pathology
    the `Float` bake path carries at `sigConstF?`, where `x/0` silently reads
    back as zero and a classifier then certifies a coupling on it).

    Shifts the numerator so the integer quotient carries at least `bits + 2`
    bits, then one flooring `Int.fdiv` — which rounds toward −∞ for either sign,
    i.e. `down` verbatim; `up` is the negated floor of the negated numerator. -/
def divRel? (dir : RoundDir) (bits : Nat) (a b : Dyadic) : Option Dyadic :=
  match a, b with
  | _, .zero => none
  | .zero, _ => some .zero
  | .ofOdd n₁ k₁ _, .ofOdd n₂ k₂ _ =>
    let s : Nat := bits + 2 + intBitLen n₂ - intBitLen n₁
    let num := n₁ <<< s
    let q : Int :=
      match dir with
      | .down => Int.fdiv num n₂
      | .up   => -(Int.fdiv (-num) n₂)
    some (roundRel dir bits (ofIntWithPrec q (k₁ - k₂ + (s : Int))))

/-- Directed square root at `bits` significant bits. `none` for a NEGATIVE
    argument — refused, not fabricated. Aligns the exponent to an even multiple
    with enough guard bits, takes `natSqrt` (a floor), and bumps one unit for
    `up` unless the root was exact. -/
def sqrtRel? (dir : RoundDir) (bits : Nat) (a : Dyadic) : Option Dyadic :=
  match a with
  | .zero => some .zero
  | .ofOdd n k _ =>
    if n < 0 then none
    else some <|
      -- a = n·2^{−k}; shift n up by `s` with `k + s` even and `n <<< s` wide
      let need : Int := ((2 * bits + 4 : Nat) : Int)
      let s0 : Int := max 0 (need - (intBitLen n : Int))
      let s : Nat := (if (k + s0) % 2 == 0 then s0 else s0 + 1).toNat
      let m := (n <<< s).natAbs
      let r := natSqrt m
      let prec := (k + (s : Int)) / 2
      match dir with
      | .down => roundRel .down bits (ofIntWithPrec (r : Int) prec)
      | .up   =>
        let r' := if r * r == m then r else r + 1
        roundRel .up bits (ofIntWithPrec (r' : Int) prec)

-- ── Float ⇆ Dyadic ────────────────────────────────────────────────────────────

/-- A finite `Float` IS a dyadic — this conversion is EXACT and total on finite
    inputs (`none` for NaN/±∞). IEEE-754 binary64: sign bit 63, biased exponent
    bits 52..62, mantissa bits 0..51; a zero biased exponent is the subnormal
    case (no implicit leading one). This is how the Lanczos tables and every
    other literal `Float` coefficient reach the exact carrier UNCHANGED. -/
def ofFloat? (x : Float) : Option Dyadic :=
  if !x.isFinite then none
  else
    let bits := x.toBits
    let sgn  := (bits >>> 63) &&& 1
    let be   := ((bits >>> 52) &&& 0x7ff).toNat
    let frac := (bits &&& 0xfffffffffffff).toNat
    let mag : Int := if be == 0 then (frac : Int) else ((frac + 0x10000000000000 : Nat) : Int)
    -- value = mag · 2^e, i.e. mag · 2^{−(−e)}
    let e   : Int := if be == 0 then (-1074 : Int) else ((be : Int) - 1075)
    some (ofIntWithPrec (if sgn == 1 then -mag else mag) (-e))

/-- Exact `Float` → `Dyadic`; `0` on a non-finite input. Use `ofFloat?` where
    the non-finite case must be handled rather than absorbed. -/
def ofFloat (x : Float) : Dyadic := (ofFloat? x).getD 0

/-- `Dyadic` → the nearest `Float`, ties to even. Rounds to 53 significant bits
    in exact `Int` arithmetic FIRST, so the mantissa reaching `Nat.toFloat` is
    under `2⁵³` and converts exactly, leaving `Float.scaleB` only the exponent.
    (A subnormal result rounds a second time inside `scaleB` — a corner far
    below any bake magnitude.) -/
def toFloat (x : Dyadic) : Float :=
  match roundNearestRel 53 x with
  | .zero => 0.0
  | .ofOdd n k _ =>
    let mag := n.natAbs.toFloat
    Float.scaleB (if n < 0 then -mag else mag) (-k)

/-- `⌊|x|·10^places + ½⌋` signed — the exact analogue of `litF`'s decimal
    quantization, with no `Float` anywhere (half away from zero, which is what
    `litF`'s `(s + 0.5).toUInt64` intends). -/
def toDecimalMantissa (x : Dyadic) (places : Nat) : Int :=
  match x with
  | .zero => 0
  | .ofOdd n k _ =>
    let negative := n < 0
    let num : Nat := n.natAbs * 10 ^ places
    -- value·10^places = num · 2^{−k}
    let v : Nat :=
      if k ≤ 0 then num <<< (-k).toNat
      else
        let s := k.toNat
        let q := num >>> s
        let rem := num - (q <<< s)
        if rem * 2 ≥ 1 <<< s then q + 1 else q
    if negative then -(v : Int) else (v : Int)

end Dyadic
