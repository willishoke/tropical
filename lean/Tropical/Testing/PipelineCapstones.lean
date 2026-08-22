import Tropical.EmitArrow.ClockPlanLaws
import Tropical.Ir.BankPlanLaws
import Tropical.Ir.RoutedSumLaws
import Tropical.Testing.ClockLaws

/-!
# Executable clock, bank, and routed pipeline capstones

The observations here compare independent source formulas with the public Plan
executor.  They do not run the Plan interpreter twice and they make no claim
about LLVM/MSL execution.
-/

namespace Tropical.Testing.PipelineCapstones

open Tropical.EmitArrow
open Tropical.Ir
open Tropical.Ir.Emit
open Tropical.Plan
open Tropical.Semantics

def capstoneAlgebra : Algebra Int where
  literal := fun n => .ok (.scalar n.mantissa)
  unary := fun tag value =>
    match tag, value with
    | .neg, .scalar n => .ok (.scalar (-n))
    | _, _ => refusal "capstone.unary" "unsupported operation"
  binary := fun tag lhs rhs =>
    match tag, lhs, rhs with
    | .add, .scalar a, .scalar b => .ok (.scalar (a + b))
    | .sub, .scalar a, .scalar b => .ok (.scalar (a - b))
    | .mul, .scalar a, .scalar b => .ok (.scalar (a * b))
    | _, _, _ => refusal "capstone.binary" "unsupported operation"
  clamp := fun _ _ _ => refusal "capstone.clamp" "outside fixture"
  select := fun _ _ _ => refusal "capstone.select" "outside fixture"
  index := fun array index =>
    match array, index with
    | .array values, .scalar i =>
        if i < 0 then refusal "capstone.index" "negative index"
        else lookupValue "capstone.index" values i.toNat
    | _, _ => refusal "capstone.index" "ill-typed index"
  loopIndex := fun index => .ok (.scalar (Int.ofNat index))
  dynamicCount := fun
    | .scalar value => .ok value
    | .array _ => .error { operation := "capstone.count", detail := "array count" }
  zero := .ok (.scalar 0)

def capstoneInputs : PlanInputs Int := {}

def tempScalar? (index : Nat) : Outcome (PlanState Int) → Option Int
  | .ok state =>
      match state.temps[index]? with
      | some (.scalar value) => some value
      | _ => none
  | .error _ => none

def arrayScalars? (index : Nat) : Outcome (PlanState Int) → Option (Array Int)
  | .ok state => do
      let values ← state.arrays[index]?
      values.mapM fun
        | .scalar value => some value
        | .array _ => none
  | .error _ => none

def resultArrayScalars? : Result Int → Option (Array Int)
  | .ok (.array values) => values.mapM fun
      | .scalar value => some value
      | .array _ => none
  | _ => none

/-- The production-authored seven-node inverse clock fixture remains valid
    before taking its signed-i64 image. -/
def sevenNodeClock_source_identity :=
  Tropical.Testing.ClockLaws.inverseFixture_denotes

/-- Its independent executable construction remains exactly seven nodes. -/
example : Tropical.Testing.ClockLaws.fixturePasses = true := by native_decide

def staticBankState : PlanState Int := { temps := #[.scalar 99] }

def staticBankBlock : Array NInstr := #[
  instrReduceBegin 0 (.const ⟨0, 0⟩ .int) 4 .int none 7,
  instrScalar "Add" 0 #[.reg 0 .int, .loopIdx 7] .int,
  instrReduceEnd 0 .int
]

def staticBankSource : Int := refFold (fun a b => a + b) 0 Int.ofNat 4

theorem staticBank_source_to_plan :
    tempScalar? 0
      (execBlocks capstoneAlgebra capstoneInputs staticBankState staticBankBlock) =
      some staticBankSource := by
  native_decide

def liveBankState : PlanState Int := {
  temps := #[.scalar 0]
  slots := #[.scalar 2]
}

def liveBankBlock : Array NInstr := #[
  instrReduceBegin 0 (.const ⟨0, 0⟩ .int) 8 .int (some (.slot 0 .int)) 3,
  instrScalar "Add" 0 #[.reg 0 .int, .loopIdx 3] .int,
  instrReduceEnd 0 .int
]

def liveBankSource : Int :=
  refFold (fun a b => a + b) 0 Int.ofNat (regionTrips 8 (some 2))

theorem liveBank_source_to_plan :
    tempScalar? 0
      (execBlocks capstoneAlgebra capstoneInputs liveBankState liveBankBlock) =
      some liveBankSource := by
  native_decide

def nestedBankState : PlanState Int := { temps := #[.scalar 0, .scalar 0] }

def nestedBankBlock : Array NInstr := #[
  instrReduceBegin 0 (.const ⟨0, 0⟩ .int) 3 .int none 10,
  instrReduceBegin 1 (.loopIdx 10) 2 .int none 11,
  instrScalar "Add" 1 #[.reg 1 .int, .const ⟨1, 0⟩ .int] .int,
  instrReduceEnd 1 .int,
  instrScalar "Add" 0 #[.reg 0 .int, .reg 1 .int] .int,
  instrReduceEnd 0 .int
]

def nestedBankSource : Int :=
  refFold (fun a b => a + b) 0
    (fun outer => refFold (fun a b => a + b) outer (fun _ => 1) 2) 3

theorem nestedBank_source_to_plan :
    tempScalar? 0
      (execBlocks capstoneAlgebra capstoneInputs nestedBankState nestedBankBlock) =
      some nestedBankSource := by
  native_decide

def routedRoutes : Array (Option Nat) :=
  #[some 0, none, some 1, some 0, some 0, some 1]

def routedMapped : Value Int → Array (Result Int)
  | .scalar item => #[.ok (.scalar item), .ok (.scalar (item + 10))]
  | .array _ => #[]

def routedState : PlanState Int := {
  temps := #[.scalar 0, .scalar 5]
  arrays := #[#[.scalar 99, .scalar 99]]
}

def routedBlock (count? : Option NOperand := none) : Array NInstr := #[
  instrRoutedSumBegin 0 3 2 routedRoutes count? 23,
  instrScalar "Add" 1 #[.loopIdx 23, .const ⟨10, 0⟩ .int] .int,
  instrRoutedSumYield 0 #[.loopIdx 23, .reg 1 .int],
  instrRoutedSumEnd 0
]

def staticRoutedSource : Result Int :=
  denoteRoutedSum capstoneAlgebra 3 2 2 routedRoutes #[] routedMapped none

/-- Static routed execution agrees with direct routed denotation, including
    the authored inactive route at `(item, emit) = (0, 1)`. -/
theorem staticRouted_source_to_plan :
    arrayScalars? 0
      (execBlocks capstoneAlgebra capstoneInputs routedState (routedBlock none)) =
      resultArrayScalars? staticRoutedSource := by
  native_decide

def liveRoutedState : PlanState Int := {
  temps := #[.scalar 0, .scalar 5]
  slots := #[.scalar 2]
  arrays := #[#[.scalar 99, .scalar 99]]
}

def liveRoutedSource : Result Int :=
  denoteRoutedSum capstoneAlgebra 3 2 2 routedRoutes #[] routedMapped
    (some (.ok (.scalar 2)))

theorem liveRouted_source_to_plan :
    arrayScalars? 0 (execBlocks capstoneAlgebra capstoneInputs liveRoutedState
      (routedBlock (some (.slot 0 .int)))) =
      resultArrayScalars? liveRoutedSource := by
  native_decide

/-- Mapped temporaries and binders are restored after a routed region. -/
theorem routed_scope_is_local :
    tempScalar? 1
      (execBlocks capstoneAlgebra capstoneInputs routedState (routedBlock none)) =
      some 5 := by
  native_decide

end Tropical.Testing.PipelineCapstones
