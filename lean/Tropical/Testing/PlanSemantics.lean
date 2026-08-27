import Tropical.Semantics.PlanWellFormed

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

private def sourceImage : PlanSourceImage Int := {
  tick := .scalar 11
  rate := .scalar 48000
  tilePhase := .scalar 2
  tileTick := .scalar 128
}

example : (PlanInputs.withDeclaredSources ({} : PlanInputs Int)
    defaultSources sourceImage).sources = #[.scalar 11, .scalar 48000] := by
  simp [PlanInputs.withDeclaredSources, defaultSources, PlanSourceImage.value,
    sourceImage]

example : (PlanInputs.withDeclaredSources ({} : PlanInputs Int)
    tileSources sourceImage).sources =
      #[.scalar 11, .scalar 48000, .scalar 2, .scalar 128] := by
  simp [PlanInputs.withDeclaredSources, tileSources, PlanSourceImage.value,
    sourceImage]

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
  slotDefaults := #[.num ⟨0, 0⟩, .num ⟨0, 0⟩, .num ⟨0, 0⟩]
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

private def stereoInputs : PlanInputs Int := {
  sources := #[.scalar 11, .scalar 48000]
  initialSlots := #[.scalar 3, .scalar 5, .scalar 7]
}

private def observedSlotScalar? (index : Nat) :
    Outcome (FlatPlanObservation Int) → Option Int
  | .ok observation => scalarResult?
      (lookupValue "fixture.observedSlot" observation.state.slots index)
  | .error _ => none

private def observedImageScalars? :
    Outcome (FlatPlanObservation Int) → Option (Array Int)
  | .ok observation => observation.sinks.mapM fun
      | .scalar value => some value
      | .array _ => none
  | .error _ => none

example : observedSlotScalar? 2
    (observeFlatPlan fixtureAlgebra stereoInputs stereoPlan) = some 7 := by
  native_decide

example : observedImageScalars?
    (observeFlatPlan fixtureAlgebra stereoInputs stereoPlan) = some #[8, 14] := by
  native_decide

example (observation : FlatPlanObservation Int)
    (hrun : observeFlatPlan fixtureAlgebra stereoInputs stereoPlan = .ok observation) :
    observation.sinks.size = 2 := by
  simpa [stereoPlan] using observeFlatPlan_sinks_size_of_ok
    fixtureAlgebra stereoInputs stereoPlan observation hrun

example (image : SinkImage Int)
    (hrun : denoteFlatPlan fixtureAlgebra stereoInputs stereoPlan = .ok image) :
    image.size = 2 := by
  simpa [stereoPlan] using denoteFlatPlan_size_of_ok
    fixtureAlgebra stereoInputs stereoPlan image hrun

example : denoteFlatPlan fixtureAlgebra stereoInputs stereoPlan =
    (observeFlatPlan fixtureAlgebra stereoInputs stereoPlan).map
      FlatPlanObservation.sinks :=
  denoteFlatPlan_eq_observeFlatPlan_map _ _ _

private def reducedState : PlanState Int := {
  temps := #[.scalar 99]
}

private def reduceBlock : Array NInstr := #[
  instrReduceBegin 0 (.const ⟨0, 0⟩ .int) 4 .int none 7,
  instrScalar "Add" 0 #[.reg 0 .int, .loopIdx 7] .int,
  instrReduceEnd 0 .int
]

example : tempScalar? 0
    (execBlocks fixtureAlgebra fixtureInputs reducedState reduceBlock) = some 6 := by
  native_decide

private def dynamicReduceState : PlanState Int := {
  temps := #[.scalar 0]
  slots := #[.scalar 2]
}

private def dynamicReduceBlock : Array NInstr := #[
  instrReduceBegin 0 (.const ⟨0, 0⟩ .int) 8 .int (some (.slot 0 .int)) 3,
  instrScalar "Add" 0 #[.reg 0 .int, .loopIdx 3] .int,
  instrReduceEnd 0 .int
]

example : tempScalar? 0
    (execBlocks fixtureAlgebra fixtureInputs dynamicReduceState dynamicReduceBlock) =
      some 1 := by native_decide

private def nestedReduceState : PlanState Int := {
  temps := #[.scalar 0, .scalar 0]
}

private def nestedReduceBlock : Array NInstr := #[
  instrReduceBegin 0 (.const ⟨0, 0⟩ .int) 3 .int none 10,
  instrReduceBegin 1 (.loopIdx 10) 2 .int none 11,
  instrScalar "Add" 1 #[.reg 1 .int, .const ⟨1, 0⟩ .int] .int,
  instrReduceEnd 1 .int,
  instrScalar "Add" 0 #[.reg 0 .int, .reg 1 .int] .int,
  instrReduceEnd 0 .int
]

example : tempScalar? 0
    (execBlocks fixtureAlgebra fixtureInputs nestedReduceState nestedReduceBlock) =
      some 9 := by native_decide

private def routedState : PlanState Int := {
  temps := #[.scalar 0, .scalar 5]
  arrays := #[#[.scalar 99, .scalar 99]]
}

private def routedBlock : Array NInstr := #[
  instrRoutedSumBegin 0 3 2
    #[some 0, none, some 1, some 0, some 0, some 1] none 23,
  instrScalar "Add" 1 #[.loopIdx 23, .const ⟨10, 0⟩ .int] .int,
  instrRoutedSumYield 0 #[.loopIdx 23, .reg 1 .int],
  instrRoutedSumEnd 0
]

example : arrayScalars? 0
    (execBlocks fixtureAlgebra fixtureInputs routedState routedBlock) =
      some #[13, 13] := by native_decide

example : tempScalar? 1
    (execBlocks fixtureAlgebra fixtureInputs routedState routedBlock) = some 5 := by
  native_decide

example :
    (match execBlocks fixtureAlgebra fixtureInputs reducedState
      #[instrReduceEnd 0 .int] with
    | .error error => error.operation
    | .ok _ => "unexpected success") = "instruction.region" := by native_decide

example : FlatPlanWellFormed stereoPlan := by native_decide

example : (stereoPlan.sinks.map SinkSpec.target).toList.Nodup :=
  (by native_decide : FlatPlanWellFormed stereoPlan).sinkTargetsNodup

example : stereoPlan.sinks[1].target < stereoPlan.outputChannelCount :=
  (by native_decide : FlatPlanWellFormed stereoPlan).sinkTargetInRange 1 (by decide)

example : stereoPlan.sinks[0].inputs[1] < stereoPlan.slotCount :=
  (by native_decide : FlatPlanWellFormed stereoPlan).sinkInputInRange
    0 (by decide) 1 (by decide)

private def balancedChild : InstanceFunction :=
  .mk "child" "child" #[] #[] #[] 0 0 0 #[]

private def balancedRoot : InstanceFunction :=
  .mk "root" "root" #[] #[] #[] 0 0 0 #[balancedChild]

private def balancedPlan : FlatPlan := {
  stereoPlan with instanceFunctions := #[balancedRoot]
}

example : InstanceRegionsBalanced balancedPlan balancedPlan.instanceFunctions[0] :=
  planWellFormed_regions_balanced (by native_decide) 0 (by decide)

example : ¬FlatPlanWellFormed {
    stereoPlan with
    sinks := #[
      { inputs := #[0], gain := ⟨1, 0⟩, target := 0 },
      { inputs := #[1], gain := ⟨1, 0⟩, target := 0 }]
  } := by native_decide

example : ¬BlocksWellFormed {
    stereoPlan with registerCount := 1
  } #[instrSessionSetElement 0 #[.sessionArrayReg 0,
      .const ⟨0, 0⟩ .int, .const ⟨1, 0⟩ .float]] := by native_decide

example : ¬BlocksWellFormed {
    stereoPlan with registerCount := 1
  } #[instrReduceBegin 0 (.const ⟨0, 0⟩ .int) 2 .int] := by native_decide

private def routedWfPlan : FlatPlan := {
  stereoPlan with
  arraySlotCount := 1
  arraySlotNames := #["routed"]
  arraySlotSizes := #[2]
}

example : ¬BlocksWellFormed routedWfPlan #[
    instrRoutedSumBegin 0 2 2 #[some 0] none 1,
    instrRoutedSumYield 0 #[.const ⟨1, 0⟩ .float],
    instrRoutedSumEnd 0
  ] := by native_decide

end Tropical.Testing.PlanSemantics
