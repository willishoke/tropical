import Tropical.Ir.Stage0
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.Kernel

/-!
# Artifact — what the compiler hands a runtime

A `KernelArtifact` is the compiler's product for one `FlatPlan`: the kernel
code a runtime loads plus the manifest that types it. Producing one is
pure emission (stage-0 split, `EmitLlvm`, `EmitMsl`, the kernel type);
loading one is a host concern and lives outside the compiler core
(`Tropical.StagedLoad`, the engine's host port).

The manifest is the kernel's type, `tropical_kernel_1` (`Tropical.Kernel`,
`design/typed-kernel.md`), with the plan's wire form nested for the gates.
-/

namespace Tropical.Artifact

open Tropical.Plan (FlatPlan)
open Tropical.Ir.Stage0 (Split hoist)

/-- One loadable kernel: the port function as LLVM IR, the manifest that
    types it (`tropical_kernel_1`, the plan nested inside), and the Metal
    realization when asked for — its MSL plus `stage0Ir`, the CPU pre-pass
    that computes the slot vector and coefficient columns the GPU kernel
    reads (empty for a JIT-only artifact: the JIT kernel carries that
    prologue inside itself). -/
structure KernelArtifact where
  ir : String
  manifest : String
  msl : String := ""
  stage0Ir : String := ""

/-- The frames-per-call an artifact is compiled for when the caller has no
    host to ask: the runtime's own device default (`TROPICAL_BUFFER_LENGTH`
    absent). Shapes are static, so a host with a different buffer length is
    a different kernel; every host path passes its own value. -/
def defaultFrames : Nat := 512

/-- Emission options a composition root chooses. The core reads no
    environment: a root that honours `TROPICAL_STAGE0=0` (the kill-switch
    for A/B and debugging — the whole plan runs in the sample loop) maps it
    to `stage0 := false` here. -/
structure Options where
  stage0 : Bool := true
deriving Repr, Inhabited

/-- The split by flow classification — the only splitter where the arena
    is gone, i.e. plans parsed from JSON. -/
def split (opts : Options) (plan : FlatPlan) : Split :=
  if opts.stage0 then hoist plan else { audio := plan, coeff? := none }

/-- The TYPED split: classification from the intern-time stage attribute
    (the partitioner's emit-order stage blocks), the production path
    wherever the session compile is in hand. -/
def splitTyped (opts : Options) (plan : FlatPlan)
    (stageBlocks : Array (Array (Option Tropical.Ir.Stage))) : Except String Split :=
  if opts.stage0 then
    (Tropical.Ir.Stage0.hoistTyped plan stageBlocks).mapError (s!"Artifact (typed split): {·}")
  else .ok { audio := plan, coeff? := none }

/-- The JIT-only artifact: the port kernel and its manifest. -/
def emitJit (s : Split) (frames : Nat) : Except String KernelArtifact := do
  let ir ← Tropical.Ir.EmitLlvm.emitKernel s frames
  let manifest ← Tropical.Kernel.manifest s.audio frames
  return { ir, manifest }

/-- The dual artifact: the port kernel, the MSL realization of the same
    audio plan, and the CPU pre-pass for its τ-independent part. -/
def emitMetal (s : Split) (frames : Nat) : Except String KernelArtifact := do
  let ir ← Tropical.Ir.EmitLlvm.emitKernel s frames
  let manifest ← Tropical.Kernel.manifest s.audio frames
  let msl ← Tropical.Ir.EmitMsl.emitKernel s.audio frames
  let stage0Ir ← match s.coeff? with
    | none => pure ""
    | some c => Tropical.Ir.EmitLlvm.emitStage0 c
  return { ir, manifest, msl, stage0Ir }

end Tropical.Artifact
