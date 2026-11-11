import Tropical.Semantics.Algebra
import Tropical.Ir.ConstFold
import Tropical.Ir.EmitCommon

/-!
# The concrete scalar algebra — the reference integer semantics

`Algebra CVal` over the constant folder's values: f64 floats, `int` on
ℤ/2⁶⁴, `int32` on ℤ/2³², bools. Each Plan operation is `ConstFold.foldOp` at
the instruction's result sort, so the reference semantics and the folder are
one definition, and `ConstFoldLaws` states their integer equations. The
integer semantics is total:

* `div`/`mod` by zero give 0; `MIN / -1` wraps to `MIN` and `MIN % -1` is 0;
* a shift amount is taken modulo the width (32 or 64);
* float → int conversion saturates to the word, NaN → 0 (`truncSatToInt`).

Every backend implements exactly this (`EmitLlvm`'s safe divisor, masked
shifts and `fptosi.sat`; `EmitMsl`'s guarded quotient, masked shifts and
explicit clamp), and `tropicaltest`'s `int-edge` gate checks the JIT against
this algebra on boundary values.

Operations the IR leaves unsorted (`denoteExpr`) infer their sort from their
operands' values (`PlanOp.resultType`); an unsorted literal is a float. The
Plan path is the sorted one: there every constant, instruction and slot store
names its kind.
-/

namespace Tropical.Semantics.Scalar

open Tropical.Ir
open Tropical.Plan (PlanOp ScalarType)

private def planOpOfBinaryTag : BinaryOpTag → PlanOp
  | .add => .add | .sub => .sub | .mul => .mul | .div => .div | .mod => .mod
  | .lt => .less | .lte => .lessEq | .gt => .greater | .gte => .greaterEq
  | .eq => .equal | .neq => .notEqual | .and => .and | .or => .or
  | .bitAnd => .bitAnd | .bitOr => .bitOr | .bitXor => .bitXor
  | .lshift => .lshift | .rshift => .rshift
  | .floorDiv => .floorDiv | .ldexp => .ldexp

private def planOpOfUnaryTag : UnaryOpTag → PlanOp
  | .neg => .neg | .not => .not | .bitNot => .bitNot
  | .sqrt => .sqrt | .abs => .abs | .floor => .floor | .ceil => .ceil
  | .round => .round | .floatExponent => .floatExponent
  | .toInt => .toInt | .toBool => .toBool | .toFloat => .toFloat
  | .toInt32 => .toInt32

private def scalarArgs (operation : String) (args : Array (Value CVal)) :
    Outcome (Array CVal) :=
  args.mapM fun
    | .scalar v => .ok v
    | .array _ => .error { operation, detail := "array operand to a scalar op" }

/-- `op` at sort `sort` over scalar values: `foldOp`, refusing only what it
    declines (Ldexp/FloatExponent, which this reference leaves to the
    backends). -/
def applyOp (op : PlanOp) (sort : ScalarType) (args : Array (Value CVal)) :
    Outcome (Value CVal) := do
  let xs ← scalarArgs s!"scalar.{op.name}" args
  match foldOp op.name sort xs with
  | some v => .ok (.scalar v)
  | none => refusal s!"scalar.{op.name}" "outside the reference algebra"

/-- The sort an unsorted op takes from its operand values. -/
private def inferSort (op : PlanOp) (args : Array (Value CVal)) : ScalarType :=
  op.resultType (args.map fun | .scalar v => v.ty | .array _ => .float)

/-- A constant of a declared kind, read exactly as the emitters read it. -/
def literalAt : ScalarType → Lean.JsonNumber → CVal
  | .float, n => .f n.toFloat
  | .int, n => .i (wrap64 (EmitCommon.jsonNumberToInt n))
  | .int32, n => .i32 (wrap32 (EmitCommon.jsonNumberToInt n))
  | .bool, n => .b (EmitCommon.jsonNumberToInt n != 0)

/-- Convert to a sort (`CVal.coerce?`, total). -/
def toSort (sort : ScalarType) : Value CVal → Result CVal
  | .scalar v =>
    match v.coerce? sort with
    | some w => .ok (.scalar w)
    | none => refusal "scalar.atSort" s!"no conversion to {sort.wire}"
  | .array _ => refusal "scalar.atSort" "an array has no scalar sort"

private def toIntValue (operation : String) : Value CVal → Except Refusal Int
  | .scalar v =>
    match v.coerce? .int with
    | some (.i n) => .ok n
    | _ => .error { operation, detail := "no integer view" }
  | .array _ => .error { operation, detail := "array index" }

def algebra : Algebra CVal where
  literal := fun n => .ok (.scalar (.f n.toFloat))
  unary := fun tag v => let op := planOpOfUnaryTag tag; applyOp op (inferSort op #[v]) #[v]
  binary := fun tag a b =>
    let op := planOpOfBinaryTag tag; applyOp op (inferSort op #[a, b]) #[a, b]
  clamp := fun v lo hi => applyOp .clamp (inferSort .clamp #[v, lo, hi]) #[v, lo, hi]
  select := fun c t e => applyOp .select (inferSort .select #[c, t, e]) #[c, t, e]
  -- `Index`: the element at the saturated integer index, 0.0 out of range
  -- (`EmitLlvm.emitIndex`)
  index := fun array index => (do
    let .array items := array
      | refusal "scalar.index" "not an array"
    let i : Int ← toIntValue "scalar.index" index
    if 0 ≤ i then
      match items[i.toNat]? with
      | some v => .ok v
      | none => .ok (.scalar (.f 0.0))
    else .ok (.scalar (.f 0.0)) : Outcome (Value CVal))
  loopIndex := fun n => .ok (.scalar (.i n))
  dynamicCount := toIntValue "scalar.count"
  zero := .ok (.scalar (.f 0.0))
  sortedLiteral := fun sort n => .ok (.scalar (literalAt sort n))
  sortedUnary := fun sort tag v => applyOp (planOpOfUnaryTag tag) sort #[v]
  sortedBinary := fun sort tag a b => applyOp (planOpOfBinaryTag tag) sort #[a, b]
  sortedClamp := fun sort v lo hi => applyOp .clamp sort #[v, lo, hi]
  sortedSelect := fun sort c t e => applyOp .select sort #[c, t, e]
  atSort := toSort

end Tropical.Semantics.Scalar
