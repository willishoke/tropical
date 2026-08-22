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
open Tropical.Semantics

/-- Execute a plan through its state boundary without observing sinks. -/
def runPlanState (alg : Algebra α) (inputs : PlanInputs α)
    (plan : FlatPlan) : Outcome (PlanState α) := do
  let initial ← initialPlanState alg inputs plan
  execPlanFunctions alg inputs initial plan

/-- Publish one completed coefficient image into the initial image used by an
    audio sample.  Scalar slots always share the coefficient plan's complete
    image.  Array storage is shared only when the coefficient plan declares
    the audio plan's array shape; scalar-only coefficient plans preserve the
    sample environment's arrays. -/
def publishCoefficientImage (audio : FlatPlan) (sampleInputs : PlanInputs α)
    (coefficientState : PlanState α) : PlanInputs α :=
  { sampleInputs with
    initialSlots := coefficientState.slots
    initialArrays := if coefficientState.arrays.size == audio.arraySlotCount then
      coefficientState.arrays
    else sampleInputs.initialArrays }

/-- Denotation of the documented two-step Stage0 protocol for one consistent
    generation: finish the control-write coefficient plan, publish its image,
    then execute and observe one audio sample.  The runtime's atomic generation
    flip is deliberately a host obligation, not an in-memory Lean claim. -/
def denoteStaged (alg : Algebra α) (controlInputs sampleInputs : PlanInputs α)
    (split : Split) : Outcome (SinkImage α) :=
  match split.coeff? with
  | none => denoteFlatPlan alg sampleInputs split.audio
  | some coefficient => do
    let coefficientState ← runPlanState alg controlInputs coefficient
    denoteFlatPlan alg
      (publishCoefficientImage split.audio sampleInputs coefficientState)
      split.audio

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

theorem denoteStaged_without_coeff (alg : Algebra α)
    (controlInputs sampleInputs : PlanInputs α) (audio : FlatPlan) :
    denoteStaged alg controlInputs sampleInputs
      { audio := audio, coeff? := none } =
      denoteFlatPlan alg sampleInputs audio := by
  rfl

/-- The first semantic refinement boundary: an empty typed selection is
    exactly the original Plan denotation under the explicit publication
    protocol. -/
theorem hoistTyped_refines_no_selection (alg : Algebra α)
    (controlInputs sampleInputs : PlanInputs α) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Stage))) (split : Split)
    (haligned : typedStagesAligned (collectPlanBlocks plan) stageBlocks = true)
    (hempty : noTypedSelection stageBlocks = true)
    (hsplit : hoistTyped plan stageBlocks = .ok split) :
    denoteStaged alg controlInputs sampleInputs split =
      denoteFlatPlan alg sampleInputs plan := by
  rw [hoistTyped_identity_of_no_selection plan stageBlocks haligned hempty]
    at hsplit
  cases hsplit
  rfl

end Tropical.Ir.Stage0
