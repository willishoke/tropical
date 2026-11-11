import Lean.Data.Json
import Tropical.Plan

/-!
# Shared emitter validation

Backend-neutral classification and validation used by the textual LLVM and
Metal emitters. Code generation remains backend-specific; this module only
owns facts about the plan being consumed.
-/

namespace Tropical.Ir.EmitCommon

open Lean (JsonNumber)
open Tropical.Plan

/-- The distinctions an emitter needs before resolving an operand. Scalar
    operands share the ordinary typed-value path; array and loop operands
    require dedicated handling. -/
inductive OperandClass where
  | scalar
  | arrayReg (slot : Nat)
  | sessionArrayReg (slot : Nat)
  | loopIndex (id : Nat)
deriving BEq, Repr

def classifyOperand : NOperand → OperandClass
  | .arrayReg slot => .arrayReg slot
  | .sessionArrayReg slot => .sessionArrayReg slot
  | .loopIdx id => .loopIndex id
  | _ => .scalar

def arrayRegSlot? (operand : NOperand) : Option Nat :=
  match classifyOperand operand with
  | .arrayReg slot => some slot
  | _ => none

/-- Require the operand form shared by Index, SetElement, and elementwise
    instructions while retaining the backend name in diagnostics. -/
def expectArrayReg (backend : String) (operand : NOperand) : Except String Nat :=
  match arrayRegSlot? operand with
  | some slot => .ok slot
  | none => .error s!"{backend}: expected an arrayReg operand"

/-- Arity is part of the scalar-op signature, independent of its lowering. -/
def planOpArity : PlanOp → Nat
  | .select | .clamp => 3
  | .neg | .abs | .sqrt | .floor | .ceil | .round
  | .not | .bitNot | .floatExponent | .toInt | .toInt32 | .toBool | .toFloat => 1
  | _ => 2

/-- Decode a scalar-op tag and reject malformed instruction arity before a
    backend indexes its operand array. -/
def validateScalarOp
    (backend tag : String) (actualArity : Nat) : Except String PlanOp := do
  let some op := PlanOp.ofString? tag
    | throw s!"{backend}: unsupported op '{tag}'"
  if actualArity == planOpArity op then
    pure op
  else
    throw s!"{backend}: op '{tag}' expects {planOpArity op} operands, got {actualArity}"

/-- Integer value of a JSON numeric constant. The scalar const path carries
    integers with exponent zero, but division keeps malformed input total. -/
def jsonNumberToInt (n : JsonNumber) : Int :=
  let rec divPow (m : Int) : Nat → Int
    | 0 => m
    | k + 1 => divPow (m / 10) k
  divPow n.mantissa n.exponent

/-- The IEEE 754 binary32 nearest the decimal `n = mantissa · 10^-exponent`,
    ties to even, as its bit pattern; `none` when it rounds past the largest
    finite binary32 (a literal a binary32 target cannot hold). Exact rational
    arithmetic, one rounding: a plan literal at binary32 is the binary32
    rounding of the literal as written, never the binary32 rounding of its
    binary64 rounding (the two differ when the binary64 value is a binary32
    tie). Subnormal results are produced as such. -/
def binary32BitsOfDecimal? (n : JsonNumber) : Option UInt32 := Id.run do
  let num : Nat := n.mantissa.natAbs
  let den : Nat := 10 ^ n.exponent
  let sign : UInt32 := if n.mantissa < 0 then 0x80000000 else 0
  if num == 0 then return some sign
  -- `q k = ⌊num / (den · 2^k)⌋` with its remainder and divisor (exact).
  let qr (k : Int) : Nat × Nat × Nat :=
    if k ≥ 0 then
      let d := den * 2 ^ k.toNat
      (num / d, num % d, d)
    else
      let nn := num * 2 ^ (-k).toNat
      (nn / den, nn % den, den)
  -- The scale that puts the quotient in [2^23, 2^24): the log2 estimate is
  -- within one of it, so a few steps settle it.
  let mut k : Int := (Nat.log2 num : Int) - (Nat.log2 den : Int) - 23
  for _ in [0:4] do
    let q := (qr k).1
    if q ≥ 2 ^ 24 then k := k + 1
    else if q < 2 ^ 23 then k := k - 1
  -- Below the normal range the scale stops at the subnormal step, 2^-149.
  if k < -149 then k := -149
  let (q0, r, d) := qr k
  let up := 2 * r > d || (2 * r == d && q0 % 2 == 1)
  let mut q := if up then q0 + 1 else q0
  if q == 2 ^ 24 then
    q := 2 ^ 23
    k := k + 1
  if q ≥ 2 ^ 23 then
    let biased := k + 150
    if biased ≥ 255 then return none
    return some (sign ||| (biased.toNat.toUInt32 <<< 23) ||| (q - 2 ^ 23).toUInt32)
  else
    return some (sign ||| q.toUInt32)

end Tropical.Ir.EmitCommon
