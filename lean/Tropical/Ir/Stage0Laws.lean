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

/-- Execute the same publication protocol up to its final audio state, before
    observing any sinks.  This isolates instruction-state simulation from the
    independent stereo/multi-sink observation layer. -/
def runStagedState (alg : Algebra α)
    (controlInputs sampleInputs : PlanInputs α) (split : Split) :
    Outcome (PlanState α) :=
  match split.coeff? with
  | none => runPlanState alg sampleInputs split.audio
  | some coefficient => do
    let coefficientState ← runPlanState alg controlInputs coefficient
    runPlanState alg
      (publishCoefficientImage split.audio sampleInputs coefficientState)
      split.audio

/-- The exact remaining simulation obligation for a nontrivial Stage0 split:
    coefficient execution followed by atomic-image publication reaches the
    same completed Plan state as direct execution.  It is deliberately a state
    relation, not the sink-image conclusion restated as a premise. -/
def StatePublicationRefines (alg : Algebra α)
    (controlInputs sampleInputs : PlanInputs α) (plan : FlatPlan)
    (split : Split) : Prop :=
  runStagedState alg controlInputs sampleInputs split =
    runPlanState alg sampleInputs plan

private theorem outcome_bind_assoc (x : Outcome α) (f : α → Outcome β)
    (g : β → Outcome γ) :
    (x.bind f).bind g = x.bind fun value => (f value).bind g := by
  cases x <;> rfl

theorem denoteFlatPlan_via_state (alg : Algebra α)
    (sampleInputs : PlanInputs α) (plan : FlatPlan) :
    denoteFlatPlan alg sampleInputs plan = (do
      let state ← runPlanState alg sampleInputs plan
      denoteSinks alg plan state) := by
  unfold denoteFlatPlan runPlanState
  cases hi : initialPlanState alg sampleInputs plan <;> rfl

theorem denoteStaged_via_state (alg : Algebra α)
    (controlInputs sampleInputs : PlanInputs α) (split : Split) :
    denoteStaged alg controlInputs sampleInputs split = (do
      let state ← runStagedState alg controlInputs sampleInputs split
      denoteSinks alg split.audio state) := by
  cases h : split.coeff? with
  | none =>
    simp [denoteStaged, runStagedState, h, denoteFlatPlan_via_state]
  | some coefficient =>
    rw [denoteStaged, h, denoteFlatPlan_via_state]
    simp only [runStagedState, h]
    simp only [denoteFlatPlan_via_state]
    exact (outcome_bind_assoc (runPlanState alg controlInputs coefficient)
      (fun coefficientState => runPlanState alg
        (publishCoefficientImage split.audio sampleInputs coefficientState)
        split.audio)
      (denoteSinks alg split.audio)).symm

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

/-- Full observable refinement for any successful, including nontrivial,
    typed split once the explicit coefficient-publication state simulation is
    discharged.  `hoistTyped` itself supplies the orthogonal interface fact:
    every original sink and output channel survives, so independent stereo and
    wider layouts are observed without collapsing them into one sink. -/
theorem hoistTyped_refines_of_state_publication (alg : Algebra α)
    (controlInputs sampleInputs : PlanInputs α) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Stage))) (split : Split)
    (hsplit : hoistTyped plan stageBlocks = .ok split)
    (hpublication : StatePublicationRefines alg controlInputs sampleInputs
      plan split) :
    denoteStaged alg controlInputs sampleInputs split =
      denoteFlatPlan alg sampleInputs plan := by
  have hinterface := hoistTyped_preserves_audio_interface
    plan stageBlocks split hsplit
  have hlayout : split.audio.outputLayoutWellFormed =
      plan.outputLayoutWellFormed := by
    unfold FlatPlan.outputLayoutWellFormed
    rw [hinterface.1, hinterface.2.1]
  have hobserve (state : PlanState α) :
      denoteSinks alg split.audio state = denoteSinks alg plan state := by
    unfold denoteSinks
    rw [hlayout, hinterface.1, hinterface.2.1]
  rw [denoteStaged_via_state, denoteFlatPlan_via_state]
  unfold StatePublicationRefines at hpublication
  rw [hpublication]
  simp only [hobserve]

end Tropical.Ir.Stage0
