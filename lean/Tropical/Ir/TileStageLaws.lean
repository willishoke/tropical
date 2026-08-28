import Tropical.Ir.TileStage
import Tropical.Ir.Stage0Laws

/-!
# Structural laws for TileStage

Tile residualization is opt-in through `tileArraySlots`.  The empty case is a
proved structural and semantic identity; it is important because exact/JIT
plans and unadmitted shapes use that path as their reference behavior.

No statement in this module equates polynomial interpolation over a complete
tile with exact expression evaluation.  That comparison remains a measured
tolerance gate.
-/

namespace Tropical.Ir.TileStage

open Tropical.Plan
open Tropical.Semantics

/-- Publish a completed endpoint image into the arrays observed by one audio
    sample.  Tile materialization intentionally publishes arrays only: shared
    scalar support is duplicated, while the ordinary audio clock remains an
    independent runtime signal. -/
def publishEndpointImage (audio : FlatPlan) (sampleInputs : PlanInputs α)
    (endpointState : PlanState α) : PlanInputs α :=
  { sampleInputs with
    initialArrays := if endpointState.arrays.size == audio.arraySlotCount then
      endpointState.arrays
    else sampleInputs.initialArrays }

/-- Execute endpoint materialization and publication up to the completed audio
    state.  The materializer inputs may carry an independent absolute endpoint
    coordinate; the audio inputs retain their own sample clock. -/
def runTiledState (alg : Algebra α)
    (materializerInputs sampleInputs : PlanInputs α) (split : Split) :
    Outcome (PlanState α) :=
  match split.tile? with
  | none => Stage0.runPlanState alg sampleInputs split.audio
  | some materializer => do
    let endpointState ← Stage0.runPlanState alg materializerInputs materializer
    Stage0.runPlanState alg
      (publishEndpointImage split.audio sampleInputs endpointState) split.audio

/-- Observe all independent audio sinks after endpoint publication. -/
def denoteTiled (alg : Algebra α)
    (materializerInputs sampleInputs : PlanInputs α) (split : Split) :
    Outcome (SinkImage α) := do
  let state ← runTiledState alg materializerInputs sampleInputs split
  denoteSinks alg split.audio state

/-- The nontrivial tile-refinement obligation, stated below the observation
    layer: endpoint-array publication must reach the same final audio state as
    direct execution.  Polynomial approximation over the rest of the tile is
    intentionally outside this exact left-endpoint theorem. -/
def EndpointPublicationRefines (alg : Algebra α)
    (materializerInputs sampleInputs : PlanInputs α) (plan : FlatPlan)
    (split : Split) : Prop :=
  runTiledState alg materializerInputs sampleInputs split =
    Stage0.runPlanState alg sampleInputs plan

theorem split_identity_of_no_tile_arrays (plan : FlatPlan)
    (h : plan.tileArraySlots.isEmpty = true) :
    split plan = .ok { audio := plan, tile? := none } := by
  simp [split, h]
  rfl

theorem split_identity_audio_of_no_tile_arrays (plan : FlatPlan)
    (h : plan.tileArraySlots.isEmpty = true) :
    (split plan).map Split.audio = .ok plan := by
  rw [split_identity_of_no_tile_arrays plan h]
  rfl

theorem denoteTiled_without_materializer (alg : Algebra α)
    (materializerInputs sampleInputs : PlanInputs α) (audio : FlatPlan) :
    denoteTiled alg materializerInputs sampleInputs
      { audio := audio, tile? := none } =
      denoteFlatPlan alg sampleInputs audio := by
  rw [Stage0.denoteFlatPlan_via_state]
  rfl

/-- Exact observable left-endpoint refinement for any successful nontrivial
    tile split once endpoint publication's state simulation is discharged.
    The split theorem supplies the orthogonal guarantee that stereo and wider
    sink layouts are preserved verbatim. -/
theorem split_refines_of_endpoint_publication (alg : Algebra α)
    (materializerInputs sampleInputs : PlanInputs α) (plan : FlatPlan)
    (result : Split) (hsplit : split plan = .ok result)
    (hpublication : EndpointPublicationRefines alg materializerInputs
      sampleInputs plan result) :
    denoteTiled alg materializerInputs sampleInputs result =
      denoteFlatPlan alg sampleInputs plan := by
  have hinterface := split_preserves_audio_interface plan result hsplit
  have hlayout : result.audio.outputLayoutWellFormed =
      plan.outputLayoutWellFormed := by
    unfold FlatPlan.outputLayoutWellFormed
    rw [hinterface.1, hinterface.2]
  have hobserve (state : PlanState α) :
      denoteSinks alg result.audio state = denoteSinks alg plan state := by
    unfold denoteSinks
    rw [hlayout, hinterface.1, hinterface.2]
  rw [Stage0.denoteFlatPlan_via_state]
  unfold denoteTiled EndpointPublicationRefines at *
  rw [hpublication]
  simp only [hobserve]

end Tropical.Ir.TileStage
