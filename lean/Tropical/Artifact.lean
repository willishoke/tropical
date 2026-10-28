import Tropical.Ir.Stage0
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.Kernel

/-!
# Artifact — what the compiler hands a runtime

A `KernelArtifact` is the compiler's product for one `FlatPlan`: the kernel
code a runtime loads plus the manifest that describes it. Producing one is
pure emission (stage-0 split, `EmitLlvm`, `EmitMsl`, the plan's wire form);
loading one is a host concern and lives outside the compiler core
(`Tropical.StagedLoad`, the engine's host port).

The manifest is the plan's wire form: same schema, its slot table extended
with the `coef:<n>` slots the main and coefficient kernels share. Beside it
rides `kernel`, the `tropical_kernel_1` type of the same plan
(`Tropical.Kernel`, `design/typed-kernel.md`): emitted for every artifact,
consumed by no host yet (slice T1).
-/

namespace Tropical.Artifact

open Tropical.Plan (FlatPlan)
open Tropical.Ir.Stage0 (Split hoist)

/-- One loadable kernel image. Empty strings mean "absent": no coefficient
    kernel (nothing hoisted), no MSL (JIT-only). `kernel` is the typed-kernel
    manifest of the same plan. -/
structure KernelArtifact where
  ir : String
  coefficientIr : String
  msl : String := ""
  manifest : String
  kernel : String := ""

/-- The frames-per-call the typed kernel is derived for when the caller has
    no host to ask: the runtime's own device default (`TROPICAL_BUFFER_LENGTH`
    absent). Shapes are static, so a host with a different buffer length is
    a different kernel; slice T3 threads the host's value through. -/
def defaultFrames : Nat := 512

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
def emitJit (s : Split) (frames : Nat := defaultFrames) :
    Except String KernelArtifact := do
  let manifestJson ← s.audio.toWire
  let ir ← Tropical.Ir.EmitLlvm.emitKernel s.audio
  let cir ← coeffIr s
  let kernel ← Tropical.Kernel.manifest s.audio frames
  return {
    ir := ir
    coefficientIr := cir
    manifest := manifestJson.compress
    kernel }

/-- The dual artifact: the JIT kernels plus the MSL realization of the same
    audio plan.  JIT-only callers never enter this function. -/
def emitMetal (s : Split) (frames : Nat := defaultFrames) :
    Except String KernelArtifact := do
  let manifestJson ← s.audio.toWire
  let ir ← Tropical.Ir.EmitLlvm.emitKernel s.audio
  let coefficientIr ← coeffIr s
  let msl ← Tropical.Ir.EmitMsl.emitKernel s.audio
  let kernel ← Tropical.Kernel.manifest s.audio frames
  return {
    ir := ir
    coefficientIr := coefficientIr
    manifest := manifestJson.compress
    msl := msl
    kernel }

end Tropical.Artifact
