import Tropical.Ir.Stage0
import Tropical.Semantics.PlanWellFormed

/-!
# Structural laws for typed Stage0 splitting

These laws cover the refusal and identity boundaries before the deeper
publication refinement.  In particular, alignment is per emitter block, not
merely a comparison of flattened totals.
-/

namespace Tropical.Ir.Stage0

open Tropical.Plan

theorem hoistTyped_refuses_misaligned (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Stage)))
    (h : typedStagesAligned (collectPlanBlocks plan) stageBlocks = false) :
    hoistTyped plan stageBlocks =
      .error "Stage0.hoistTyped: typed stage blocks do not align with emitter blocks" := by
  simp [hoistTyped, h]
  rfl

theorem hoistTyped_identity_of_no_selection (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Stage)))
    (haligned : typedStagesAligned (collectPlanBlocks plan) stageBlocks = true)
    (hempty : noTypedSelection stageBlocks = true) :
    hoistTyped plan stageBlocks = .ok { audio := plan, coeff? := none } := by
  simp [hoistTyped, haligned, hempty]
  rfl

theorem hoistTyped_identity_audio (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Stage)))
    (haligned : typedStagesAligned (collectPlanBlocks plan) stageBlocks = true)
    (hempty : noTypedSelection stageBlocks = true) :
    (hoistTyped plan stageBlocks).map Split.audio = .ok plan := by
  rw [hoistTyped_identity_of_no_selection plan stageBlocks haligned hempty]
  rfl

end Tropical.Ir.Stage0
