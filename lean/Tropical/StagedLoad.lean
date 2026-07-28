import Tropical.Ffi
import Tropical.Ir.Stage0
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl

/-!
# StagedLoad — the one place a `FlatPlan` becomes running kernels

Every load path (the engine session compile, the golden/equiv runner,
the diffcli render verbs) funnels through here so the stage-0 split is
gated by the same suites that gate everything else: hoist (unless
`TROPICAL_STAGE0=0`), emit the audio kernel — and the coefficient
kernel when anything hoisted — through the same `EmitLlvm`, and load
both over `Runtime.loadIrStaged`. The manifest is the *audio* plan's
wire form: same schema, its slot table extended with the `coef:<n>`
slots the two kernels share.
-/

namespace Tropical.StagedLoad

open Tropical.Plan (FlatPlan)
open Tropical.Ir.Stage0 (Split hoist)

/-- Kill-switch for A/B and debugging: `TROPICAL_STAGE0=0` disables the
    split (plans load exactly as before, one kernel). -/
def enabled : IO Bool := do
  pure ((← IO.getEnv "TROPICAL_STAGE0") != some "0")

/-- The env-gated split (flow classification — the only splitter where
    the arena is gone, i.e. plans parsed from JSON). -/
def split (plan : FlatPlan) : IO Split := do
  if ← enabled then pure (hoist plan)
  else pure { audio := plan, coeff? := none }

/-- The env-gated TYPED split: classification from the intern-time
    stage attribute (the partitioner's emit-order stage blocks), the
    production path wherever the session compile is in hand. -/
def splitTyped (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Split := do
  if ← enabled then
    match Tropical.Ir.Stage0.hoistTyped plan stageBlocks with
    | .ok s => pure s
    | .error e => throw <| IO.userError s!"StagedLoad (typed split): {e}"
  else pure { audio := plan, coeff? := none }

/-- The coefficient kernel's IR — empty string when nothing hoisted
    (the FFI treats empty as "no coefficient kernel"). -/
def coeffIr (s : Split) : Except String String :=
  match s.coeff? with
  | none => .ok ""
  | some c => Tropical.Ir.EmitLlvm.emitKernel c

private structure EmittedParts where
  audioIr : String
  coefficientIr : String
  manifest : String

private def emitParts (s : Split) : Except String EmittedParts := do
  let manifestJson ← s.audio.toWire
  let ir ← Tropical.Ir.EmitLlvm.emitKernel s.audio
  let cir ← coeffIr s
  return {
    audioIr := ir
    coefficientIr := cir
    manifest := manifestJson.compress }

/-- Debug: `TROPICAL_STAGE0_DUMP=<dir>` writes the split artifacts. -/
private def dumpParts (parts : EmittedParts) : IO Unit := do
  if let some dir ← IO.getEnv "TROPICAL_STAGE0_DUMP" then
    IO.FS.writeFile s!"{dir}/audio.ll" parts.audioIr
    IO.FS.writeFile s!"{dir}/coeff.ll" parts.coefficientIr
    IO.FS.writeFile s!"{dir}/manifest.json" parts.manifest

/-- The Metal sibling of the opt-in stage dump. Normal output and wire schemas
    are untouched when `TROPICAL_STAGE0_DUMP` is absent. -/
private def dumpMsl (msl : String) : IO Unit := do
  if let some dir ← IO.getEnv "TROPICAL_STAGE0_DUMP" then
    IO.FS.writeFile s!"{dir}/audio.metal" msl

/-- Split + emit + load, JIT-only. -/
def load (rt : Tropical.Ffi.Runtime) (plan : FlatPlan) : IO Unit := do
  let s ← split plan
  match emitParts s with
  | .error e => throw <| IO.userError s!"StagedLoad: {e}"
  | .ok parts =>
    dumpParts parts
    rt.loadIrStaged parts.audioIr "" parts.coefficientIr parts.manifest

/-- Split + emit + dual load (JIT + Metal): MSL is emitted from the
    *audio* plan — coefficients are computed host-side in f64 by the JIT
    coefficient kernel and cross to the GPU as host-written slots. -/
def loadMsl (rt : Tropical.Ffi.Runtime) (plan : FlatPlan) : IO Unit := do
  let s ← split plan
  match emitParts s, Tropical.Ir.EmitMsl.emitKernel s.audio with
  | .error e, _ => throw <| IO.userError s!"StagedLoad: {e}"
  | _, .error e => throw <| IO.userError s!"StagedLoad (msl): {e}"
  | .ok parts, .ok msl =>
    dumpParts parts
    dumpMsl msl
    rt.loadIrStaged parts.audioIr msl parts.coefficientIr parts.manifest

/-- Typed split + emit + load, JIT-only. -/
def loadTyped (rt : Tropical.Ffi.Runtime) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Unit := do
  let s ← splitTyped plan stageBlocks
  match emitParts s with
  | .error e => throw <| IO.userError s!"StagedLoad: {e}"
  | .ok parts =>
    dumpParts parts
    rt.loadIrStaged parts.audioIr "" parts.coefficientIr parts.manifest

/-- Typed split + emit + dual load (JIT + Metal). MSL is emitted from the
    *audio* plan: scalar coefficients cross to the GPU as host-written
    slots, and hoisted coefficient COLUMNS (banks-as-data) cross via the
    packed `coeff_columns` buffer — the runtime uploads the captured
    generation per dispatch, f64→f32. The same typed split a live
    session runs on `TROPICAL_BACKEND=metal`. -/
def loadMslTyped (rt : Tropical.Ffi.Runtime) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : IO Unit := do
  let s ← splitTyped plan stageBlocks
  match emitParts s, Tropical.Ir.EmitMsl.emitKernel s.audio with
  | .error e, _ => throw <| IO.userError s!"StagedLoad: {e}"
  | _, .error e => throw <| IO.userError s!"StagedLoad (msl): {e}"
  | .ok parts, .ok msl =>
    dumpParts parts
    dumpMsl msl
    rt.loadIrStaged parts.audioIr msl parts.coefficientIr parts.manifest

end Tropical.StagedLoad
