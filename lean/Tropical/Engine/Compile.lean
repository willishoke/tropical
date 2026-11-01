import Tropical.Engine.Core

/-!
# Engine.Compile — compile the session and publish it

Every graph mutation ends here. `syncCompile` runs the pure session compile
(`Tropical.Session.compileStaged`: lift, resolved root, Core check,
partition, plan) and hot-swaps the result into the runtime through the host
port; a failed compile leaves the mutated graph in place and the previous
kernel playing. `loadKernel` emits the artifact and publishes.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)
open Tropical.Session (compileStaged)

private def buildLoadArtifacts (env : Env) (plan : Tropical.Plan.FlatPlan)
    (stages? : Option (Array (Array (Option Tropical.Ir.Stage))))
    (emitMsl : Bool) : EngineM Tropical.Artifact.KernelArtifact := do
  let split ← match stages? with
    | some blocks =>
      match Tropical.Artifact.splitTyped env.artifact plan blocks with
      | .ok s => pure s
      | .error e => internalError e
    | none => pure (Tropical.Artifact.split env.artifact plan)
  -- Shapes are static: the artifact is compiled for the frames the host
  -- renders per call (`design/typed-kernel.md`).
  let emitted := if emitMsl then Tropical.Artifact.emitMetal split env.host.frames
    else Tropical.Artifact.emitJit split env.host.frames
  match emitted with
  | .error msg => internalError s!"Emit artifact: {msg}"
  | .ok artifact => pure artifact

/-- Emit the plan's artifact and publish it through the host port: the
    stage-0 split first (under the root's `Artifact.Options`), then the
    port kernel with the coefficient plan as its prologue; on the metal
    backend also the MSL kernel and the CPU pre-pass, dual-loaded (audio on
    the GPU, render_window/the scope on the JIT). Any emit failure errors
    BEFORE the load, so the previous kernel keeps playing. -/
def loadKernelPublished (env : Env) (plan : Tropical.Plan.FlatPlan)
    (stages? : Option (Array (Array (Option Tropical.Ir.Stage))) := none) :
    EngineM Tropical.Host.PublishedGeneration := do
  let artifact ← buildLoadArtifacts env plan stages? (env.host.backend == .metal)
  env.host.publish artifact

/-- Compatibility form for compile paths whose replies do not expose the
    publication identity. -/
def loadKernel (env : Env) (plan : Tropical.Plan.FlatPlan)
    (stages? : Option (Array (Array (Option Tropical.Ir.Stage))) := none) :
    EngineM Unit := do
  let _ ← loadKernelPublished env plan stages?
  pure ()

/-- Compile and atomically publish separate audio and read-only observation
    realizations. The observation artifact is always JIT-only and owns no
    realtime storage; controls project into it by matching slot names. -/
def loadKernelWithObservationPublished (env : Env)
    (audioPlan observationPlan : Tropical.Plan.FlatPlan)
    (audioStages observationStages : Array (Array (Option Tropical.Ir.Stage))) :
    EngineM Tropical.Host.PublishedGeneration := do
  let audio ← buildLoadArtifacts env audioPlan (some audioStages) (env.host.backend == .metal)
  let observation ←
    buildLoadArtifacts env observationPlan (some observationStages) false
  env.host.publishWithObservation audio observation

/-- The session lowering + compile: lift free wires if needed, build the session
    input from the mirror (the shared prologue), compile it to a plan with typed
    stage blocks, and hot-swap the result into the Lean-owned runtime. A failed
    compile leaves the mutated graph in place and the previous kernel playing. -/
def syncCompile (env : Env) : EngineM Unit := do
  let (plan, stageBlocks) ← env.session (compileStaged "syncCompile")
  -- Lean owns codegen: emit the kernel artifacts from the in-memory plan
  -- and hand them to the host (planJson is the metadata manifest). The
  -- split is TYPED here — the session compile is in hand.
  loadKernel env plan (some stageBlocks)

end Tropical.Engine
