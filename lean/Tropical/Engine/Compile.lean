import Tropical.Engine.Core

/-!
# Engine.Compile — compile the session and publish it

Every graph mutation ends here. `syncCompile` runs the pure session compile
(`Tropical.Session.compileStaged`: lift, resolved root, Core check,
partition, plan) and hot-swaps the result into the runtime through the host
port; a failed compile leaves the mutated graph in place and the previous
kernel playing. `buildKernelIr`/`loadKernel` emit the artifact and publish.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)
open Tropical.Session (compileStaged)

/-- Emit the (LLVM IR text, manifest JSON) for a compiled session plan
    without loading it. Shared by the audio compile (`syncCompile`) and the
    lazy inspection compile (Part II): both realizations go through the same
    `toWire`/`emitKernel` path; only the originating `FlatPlan` differs. -/
def buildKernelIr (plan : Tropical.Plan.FlatPlan) : EngineM (String × String) := do
  let planJson ← match plan.toWire with
    | .error msg => internalError msg
    | .ok j => pure j.compress
  let ir ← match Tropical.Ir.EmitLlvm.emitKernel plan with
    | .error msg => internalError s!"EmitLlvm: {msg}"
    | .ok s => pure s
  pure (ir, planJson)

private def buildLoadArtifacts (plan : Tropical.Plan.FlatPlan)
    (stages? : Option (Array (Array (Option Tropical.Ir.Stage))))
    (emitMsl : Bool) : EngineM Tropical.Artifact.KernelArtifact := do
  let split ← match stages? with
    | some blocks => Tropical.Artifact.splitTyped plan blocks
    | none => Tropical.Artifact.split plan
  if emitMsl then
    match Tropical.Artifact.emitMetal split with
      | .error msg => internalError s!"Emit Metal artifacts: {msg}"
      | .ok artifact => pure artifact
  else
    let (ir, manifest) ← buildKernelIr split.audio
    let coefficientIr ← match Tropical.Artifact.coeffIr split with
      | .error msg => internalError s!"EmitLlvm (coeff): {msg}"
      | .ok s => pure s
    pure { ir, coefficientIr, manifest }

/-- Emit the plan's kernel artifacts and load them into the runtime: the
    stage-0 split first (Stage0.hoist, gated by `TROPICAL_STAGE0`), then
    LLVM IR for the audio kernel — and the coefficient kernel when
    anything hoisted — always; on the metal backend also the MSL kernel,
    dual-loaded (audio on the GPU, render_window/the scope on the JIT;
    coefficients run on the CPU JIT in f64 and cross to the GPU as
    host-written slots, coefficient COLUMNS as the packed `coeff_columns`
    buffer). Any emit failure errors BEFORE the load, so the
    previous kernel keeps playing — the same recoverable contract as the
    IR path. -/
def loadKernelPublished (env : Env) (plan : Tropical.Plan.FlatPlan)
    (stages? : Option (Array (Array (Option Tropical.Ir.Stage))) := none) :
    EngineM Tropical.Host.PublishedGeneration := do
  let artifact ← buildLoadArtifacts plan stages? env.metalBackend
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
  let audio ← buildLoadArtifacts audioPlan (some audioStages) env.metalBackend
  let observation ←
    buildLoadArtifacts observationPlan (some observationStages) false
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
