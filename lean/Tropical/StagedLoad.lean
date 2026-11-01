import Tropical.Ffi
import Tropical.Artifact

/-!
# StagedLoad — the one place a `FlatPlan` becomes running kernels

Every load path (the engine session compile, the golden/equiv runner,
the diffcli render verbs) funnels through here so the stage-0 split is
gated by the same suites that gate everything else: hoist (unless
`TROPICAL_STAGE0=0`), emit the audio kernel — and the coefficient
kernel when anything hoisted — through the same `EmitLlvm`, and load
both over `Runtime.loadIrStaged`. The split and emission are the
compiler's (`Tropical.Artifact`); this module is only the loading side.
-/

namespace Tropical.StagedLoad

open Tropical.Plan (FlatPlan)
open Tropical.Artifact (KernelArtifact emitJit emitMetal)

/-- The emission options this host's environment selects:
    `TROPICAL_STAGE0=0` disables the stage-0 split. The core never reads
    the environment; the roots that boot on libtropical read it here. -/
def optionsFromEnv : IO Tropical.Artifact.Options := do
  pure { stage0 := (← IO.getEnv "TROPICAL_STAGE0") != some "0" }

private def split (plan : FlatPlan) : IO Tropical.Ir.Stage0.Split := do
  pure (Tropical.Artifact.split (← optionsFromEnv) plan)

private def splitTyped (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Tropical.Ir.Stage0.Split := do
  match Tropical.Artifact.splitTyped (← optionsFromEnv) plan stageBlocks with
  | .ok s => pure s
  | .error e => throw <| IO.userError s!"StagedLoad: {e}"

/-- Debug: `TROPICAL_STAGE0_DUMP=<dir>` writes the artifact. -/
private def dumpParts (parts : KernelArtifact) : IO Unit := do
  if let some dir ← IO.getEnv "TROPICAL_STAGE0_DUMP" then
    IO.FS.writeFile s!"{dir}/kernel.ll" parts.ir
    IO.FS.writeFile s!"{dir}/manifest.json" parts.manifest
    unless parts.msl.isEmpty do IO.FS.writeFile s!"{dir}/kernel.metal" parts.msl
    unless parts.stage0Ir.isEmpty do IO.FS.writeFile s!"{dir}/stage0.ll" parts.stage0Ir

private def publish (rt : Tropical.Ffi.Runtime) (parts : KernelArtifact) : IO Unit := do
  dumpParts parts
  rt.load parts.ir parts.msl parts.stage0Ir parts.manifest

/-- Split + emit + load, JIT-only, for the runtime's own frame count. -/
def load (rt : Tropical.Ffi.Runtime) (plan : FlatPlan) : IO Unit := do
  let frames := (← rt.bufferLength).toNat
  match emitJit (← split plan) frames with
  | .error e => throw <| IO.userError s!"StagedLoad: {e}"
  | .ok parts => publish rt parts

/-- Split + emit + dual load (JIT + Metal). -/
def loadMsl (rt : Tropical.Ffi.Runtime) (plan : FlatPlan) : IO Unit := do
  let frames := (← rt.bufferLength).toNat
  match emitMetal (← split plan) frames with
  | .error e => throw <| IO.userError s!"StagedLoad (msl): {e}"
  | .ok parts => publish rt parts

/-- Typed split + emit + load, JIT-only. -/
def loadTyped (rt : Tropical.Ffi.Runtime) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Unit := do
  let frames := (← rt.bufferLength).toNat
  match emitJit (← splitTyped plan stageBlocks) frames with
  | .error e => throw <| IO.userError s!"StagedLoad: {e}"
  | .ok parts => publish rt parts

/-- Typed split + emit + dual load (JIT + Metal). The MSL kernel reads the
    slot vector and the hoisted coefficient columns the host computes with
    the artifact's stage-0 pre-pass, f64 on the CPU, narrowed per dispatch. -/
def loadMslTyped (rt : Tropical.Ffi.Runtime) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Unit := do
  let frames := (← rt.bufferLength).toNat
  match emitMetal (← splitTyped plan stageBlocks) frames with
  | .error e => throw <| IO.userError s!"StagedLoad (msl): {e}"
  | .ok parts => publish rt parts

end Tropical.StagedLoad
