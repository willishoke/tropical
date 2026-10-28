import Tropical.Ir.Stage0
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl

/-!
# Artifact — what the compiler hands a runtime

A `KernelArtifact` is the compiler's product for one `FlatPlan`: the kernel
code a runtime loads plus the manifest that describes it. Producing one is
pure emission (stage-0 split, `EmitLlvm`, `EmitMsl`, the plan's wire form);
loading one is a host concern and lives outside the compiler core
(`Tropical.StagedLoad`, the engine's host port).

The manifest is the plan's wire form: same schema, its slot table extended
with the `coef:<n>` slots the main and coefficient kernels share.
-/

namespace Tropical.Artifact

open Tropical.Plan (FlatPlan)
open Tropical.Ir.Stage0 (Split hoist)

/-- One loadable kernel image. Empty strings mean "absent": no coefficient
    kernel (nothing hoisted), no MSL (JIT-only). -/
structure KernelArtifact where
  ir : String
  coefficientIr : String
  msl : String := ""
  manifest : String

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

/-- The JIT-only artifact: main kernel, coefficient kernel, manifest. -/
def emitJit (s : Split) : Except String KernelArtifact := do
  let manifestJson ← s.audio.toWire
  let ir ← Tropical.Ir.EmitLlvm.emitKernel s.audio
  let cir ← coeffIr s
  return {
    ir := ir
    coefficientIr := cir
    manifest := manifestJson.compress }

/-- The dual artifact: the JIT kernels plus the MSL realization of the same
    audio plan.  JIT-only callers never enter this function. -/
def emitMetal (s : Split) : Except String KernelArtifact := do
  let manifestJson ← s.audio.toWire
  let ir ← Tropical.Ir.EmitLlvm.emitKernel s.audio
  let coefficientIr ← coeffIr s
  let msl ← Tropical.Ir.EmitMsl.emitKernel s.audio
  return {
    ir := ir
    coefficientIr := coefficientIr
    manifest := manifestJson.compress
    msl := msl }

end Tropical.Artifact
