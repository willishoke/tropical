import Tropical.Semantics.Staging
import Tropical.Ir.Stage0Laws

/-! Executable witnesses for the staging proof surface. -/

namespace Tropical.Testing.StagingLaws

open Tropical.Ir
open Tropical.Ir.Staging
open Tropical.Semantics.Staging

example : Stage.fold.join .s0 = .s0 := by rfl
example : Stage.s0.join .s1 = .s1 := by rfl
example : Stage.s0.le Stage.s1 = true := by rfl
example : Stage.s1.le Stage.s0 = false := by rfl

example (a b c : Stage) : (a.join b).join c = a.join (b.join c) :=
  stage_join_assoc a b c

example (arena : ExprArena) (id : ExprId)
    (h : arena.sig? id = none) :
    stageOf arena {} id = .s1 :=
  stageOf_dangling h

end Tropical.Testing.StagingLaws

namespace Tropical.Testing.Stage0Laws

open Tropical.Ir
open Tropical.Ir.Stage0
open Tropical.Plan

private def emptyPlan : FlatPlan :=
  { arraySlotNames := #[]
    registerCount := 0
    arraySlotCount := 0
    arraySlotSizes := #[]
    instanceFunctions := #[]
    sinks := #[]
    slotCount := 0
    slotNames := #[]
    slotDefaults := #[] }

example : hoistTyped emptyPlan #[] =
    .ok { audio := emptyPlan, coeff? := none } := by
  exact hoistTyped_identity_of_no_selection _ _
    (by native_decide)
    (by native_decide)

example : hoistTyped emptyPlan #[#[none]] =
    .error "Stage0.hoistTyped: typed stage blocks do not align with emitter blocks" := by
  exact hoistTyped_refuses_misaligned _ _
    (by native_decide)

end Tropical.Testing.Stage0Laws
