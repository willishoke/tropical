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
open Tropical.Artifact (KernelArtifact split splitTyped emitJit emitMetal)

/-- Debug: `TROPICAL_STAGE0_DUMP=<dir>` writes the split artifacts. -/
private def dumpParts (parts : KernelArtifact) : IO Unit := do
  if let some dir ← IO.getEnv "TROPICAL_STAGE0_DUMP" then
    IO.FS.writeFile s!"{dir}/audio.ll" parts.ir
    IO.FS.writeFile s!"{dir}/coeff.ll" parts.coefficientIr
    IO.FS.writeFile s!"{dir}/manifest.json" parts.manifest

/-- The Metal sibling of the opt-in stage dump. Normal output and wire schemas
    are untouched when `TROPICAL_STAGE0_DUMP` is absent. -/
private def dumpMetalParts (parts : KernelArtifact) : IO Unit := do
  if let some dir ← IO.getEnv "TROPICAL_STAGE0_DUMP" then
    IO.FS.writeFile s!"{dir}/audio.ll" parts.ir
    IO.FS.writeFile s!"{dir}/coeff.ll" parts.coefficientIr
    IO.FS.writeFile s!"{dir}/manifest.json" parts.manifest
    IO.FS.writeFile s!"{dir}/audio.metal" parts.msl

/-- Split + emit + load, JIT-only. -/
def load (rt : Tropical.Ffi.Runtime) (plan : FlatPlan) : IO Unit := do
  let s ← split plan
  match emitJit s with
  | .error e => throw <| IO.userError s!"StagedLoad: {e}"
  | .ok parts =>
    dumpParts parts
    rt.loadIrStaged parts.ir "" parts.coefficientIr parts.manifest

/-- Split + emit + dual load (JIT + Metal): MSL is emitted from the
    *audio* plan — coefficients are computed host-side in f64 by the JIT
    coefficient kernel and cross to the GPU as host-written slots. -/
def loadMsl (rt : Tropical.Ffi.Runtime) (plan : FlatPlan) : IO Unit := do
  let s ← split plan
  match emitMetal s with
  | .error e => throw <| IO.userError s!"StagedLoad (msl): {e}"
  | .ok parts =>
    dumpMetalParts parts
    rt.loadIrStaged parts.ir parts.msl parts.coefficientIr parts.manifest

/-- Typed split + emit + load, JIT-only. -/
def loadTyped (rt : Tropical.Ffi.Runtime) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Unit := do
  let s ← splitTyped plan stageBlocks
  match emitJit s with
  | .error e => throw <| IO.userError s!"StagedLoad: {e}"
  | .ok parts =>
    dumpParts parts
    rt.loadIrStaged parts.ir "" parts.coefficientIr parts.manifest

/-- Typed split + emit + dual load (JIT + Metal). MSL is emitted from the
    *audio* plan: scalar coefficients cross to the GPU as host-written
    slots, and hoisted coefficient COLUMNS (banks-as-data) cross via the
    packed `coeff_columns` buffer — the runtime uploads the captured
    generation per dispatch, f64→f32. The same typed split a live
    session runs on `TROPICAL_BACKEND=metal`. -/
def loadMslTyped (rt : Tropical.Ffi.Runtime) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Unit := do
  let s ← splitTyped plan stageBlocks
  match emitMetal s with
  | .error e => throw <| IO.userError s!"StagedLoad (msl): {e}"
  | .ok parts =>
    dumpMetalParts parts
    rt.loadIrStaged parts.ir parts.msl parts.coefficientIr parts.manifest

end Tropical.StagedLoad
