import Tropical.Semantics.Plan

/-!
# Focused Plan-semantics fixtures

These examples pin namespace lookup, scalar execution, and immutable array
updates independently of LLVM/MSL execution.
-/

namespace Tropical.Testing.PlanSemantics

open Tropical.Ir
open Tropical.Plan
open Tropical.Semantics

private def fixtureAlgebra : Algebra Int where
  literal := fun n => .ok (.scalar n.mantissa)
  unary := fun tag value =>
    match tag, value with
    | .neg, .scalar n => .ok (.scalar (-n))
    | _, _ => refusal "fixture.unary" "unsupported fixture operation"
  binary := fun tag lhs rhs =>
    match tag, lhs, rhs with
    | .add, .scalar a, .scalar b => .ok (.scalar (a + b))
    | .mul, .scalar a, .scalar b => .ok (.scalar (a * b))
    | _, _, _ => refusal "fixture.binary" "unsupported fixture operation"
  clamp := fun _ _ _ => refusal "fixture.clamp" "outside fixture"
  select := fun _ _ _ => refusal "fixture.select" "outside fixture"
  index := fun array index =>
    match array, index with
    | .array values, .scalar i =>
        if i < 0 then refusal "fixture.index" "negative index"
        else lookupValue "fixture.index" values i.toNat
    | _, _ => refusal "fixture.index" "ill-typed index"
  loopIndex := fun index => .ok (.scalar (Int.ofNat index))
  dynamicCount := fun
    | .scalar value => .ok value
    | .array _ => .error { operation := "fixture.count", detail := "array count" }
  zero := .ok (.scalar 0)

private def fixtureInputs : PlanInputs Int := {
  inputs := #[.scalar 7]
  params := fun name => if name == "gain" then some (.scalar 3) else none
  sources := #[.scalar 11, .scalar 48000]
}

private def fixtureState : PlanState Int := {
  temps := #[.scalar 2, .scalar 5]
  slots := #[.scalar 13]
  arrays := #[#[.scalar 17, .scalar 19]]
}

example :
    evalOperand fixtureAlgebra fixtureInputs fixtureState (.source 0 .int) =
      .ok (.scalar 11) := rfl

private def scalarResult? : Result Int → Option Int
  | .ok (.scalar value) => some value
  | _ => none

private def tempScalar? (index : Nat) : Outcome (PlanState Int) → Option Int
  | .ok state => scalarResult? (lookupValue "fixture.temp" state.temps index)
  | .error _ => none

private def arrayScalars? (index : Nat) : Outcome (PlanState Int) → Option (Array Int)
  | .ok state => do
      let values ← state.arrays[index]?
      values.mapM fun
        | .scalar value => some value
        | .array _ => none
  | .error _ => none

example : scalarResult?
    (evalOperand fixtureAlgebra fixtureInputs fixtureState (.param "gain" .float)) =
      some 3 := by native_decide

example :
    (match evalOperand fixtureAlgebra fixtureInputs fixtureState (.sessionArrayReg 0) with
    | .error error => error.operation
    | .ok _ => "unexpected success") = "operand.sessionArrayReg" := rfl

private def addInstr : NInstr :=
  instrScalar "Add" 0 #[.reg 0 .int, .reg 1 .int] .int

example : tempScalar? 0
    (execSmallInstr fixtureAlgebra fixtureInputs fixtureState addInstr) = some 7 := by
  native_decide

private def packInstr : NInstr := instrPack 0 #[.reg 0 .int, .slot 0 .int]

example : arrayScalars? 0
    (execSmallInstr fixtureAlgebra fixtureInputs fixtureState packInstr) =
      some #[2, 13] := by native_decide

private def setInstr : NInstr :=
  instrSetElement 0 #[.arrayReg 0, .const ⟨1, 0⟩ .int, .reg 1 .int]

example : arrayScalars? 0
    (execSmallInstr fixtureAlgebra fixtureInputs fixtureState setInstr) =
      some #[17, 5] := by native_decide

private def setOutOfRangeInstr : NInstr :=
  instrSetElement 0 #[.arrayReg 0, .const ⟨8, 0⟩ .int, .reg 1 .int]

example : arrayScalars? 0
    (execSmallInstr fixtureAlgebra fixtureInputs fixtureState setOutOfRangeInstr) =
      some #[17, 19] := by native_decide

private def elementwiseInstr : NInstr :=
  instrArray "Add" 0 #[.arrayReg 0, .slot 0 .int] 2 #[1, 0] .int

example : arrayScalars? 0
    (execSmallInstr fixtureAlgebra fixtureInputs fixtureState elementwiseInstr) =
      some #[30, 32] := by native_decide

private def stereoPlan : FlatPlan := {
  arraySlotNames := #[]
  registerCount := 0
  arraySlotCount := 0
  arraySlotSizes := #[]
  instanceFunctions := #[]
  sinks := #[
    { inputs := #[0, 1], gain := ⟨1, 0⟩, target := 0 },
    { inputs := #[2], gain := ⟨2, 0⟩, target := 1 }]
  outputChannelCount := 2
  slotCount := 3
  slotNames := #["left-a", "left-b", "right"]
  slotDefaults := #[]
}

private def stereoState : PlanState Int := {
  slots := #[.scalar 3, .scalar 5, .scalar 7]
}

private def imageScalars? : Outcome (SinkImage Int) → Option (Array Int)
  | .ok values => values.mapM fun
      | .scalar value => some value
      | .array _ => none
  | .error _ => none

example : imageScalars? (denoteSinks fixtureAlgebra stereoPlan stereoState) =
    some #[8, 14] := by native_decide

end Tropical.Testing.PlanSemantics
