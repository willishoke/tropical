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
  | .not | .bitNot | .floatExponent | .toInt | .toBool | .toFloat => 1
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

end Tropical.Ir.EmitCommon
