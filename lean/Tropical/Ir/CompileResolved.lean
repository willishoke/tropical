import Tropical.Ir.Emit

/-!
# compileResolved — per-program emit boundary

Port of `compiler/ir/compile_resolved.ts` (and the trivial
`compiler/ir/slots.ts`, whose decl-table projections are already
`CoreProgram.regs` / `.instances` — slot index IS table position).

Takes a post-strata `CoreProgram` and produces a
`Plan.PerInstancePlan`. The session compiler packs one per kernel into
`instance_functions[]`; this function does not produce a runnable plan
on its own.
-/

namespace Tropical.Ir.CompileResolved

open Lean (JsonNumber)
open Tropical.Ir
open Tropical.Ir.Core
open Tropical.Ir.Emit (EmitSlots ArraySlotInfo ScalarType inputDeclScalarType)

/-- Param-handle / slot bindings plus the nested slot maps for the
    fractal compile path (port of `CompileResolvedContext`). -/
structure Context where
  paramHandles : Array (Nat × String) := #[]
  paramSlots : Array (Nat × Nat) := #[]
  nestedOutputSlots? : Option (Array (Nat × Array (Nat × Nat))) := none
  nestedInputSlots : Array (Nat × Array (Nat × Nat)) := #[]
  inputSlotOverride : Array (Nat × Nat) := #[]
  inputArraySlots : Array (Nat × ArraySlotInfo) := #[]
  nestedInputArraySlots : Array (Nat × Array (Nat × ArraySlotInfo)) := #[]
  nestedOutputArraySlots : Array (Nat × Array (Nat × ArraySlotInfo)) := #[]
  /-- Staging: the partitioner supplies input-wire stages and (having
      recursed into children first) each child's per-output stages. -/
  staging : Tropical.Ir.Emit.StagingInfo := {}
deriving Inhabited

/-- TS `inputPortTypes` derivation (compile_resolved.ts): scalar → it,
    array → its element kind. -/
private def inputPortType (t? : Option CorePortType) : ScalarType :=
  match t? with
  | none => .float
  | some (.scalar k) => k
  | some (.array k _) => k

private def shapeDimNat (n : JsonNumber) : Nat :=
  n.toFloat.toUInt64.toNat

/-- TS `outputPortScalarCount`: scalar → 1; array → product of shape
    dims (literal — type-param dims were retired with generics). -/
private def outputPortScalarCount (decl : CoreOutputDecl) : Except String Nat :=
  match decl.type? with
  | none | some (.scalar _) => .ok 1
  | some (.array _ shape) => do
    let mut total := 1
    for dim in shape do
      total := total * shapeDimNat dim
    .ok total

/-- Compile a post-strata `CoreProgram` to a `PerInstancePlan`. The
    `arena` is the shared hash-consed DAG the program's leaf `ExprId`s
    index into (Phase B: one arena for the whole root + registry). -/
def compileResolved (prog : CoreProgram) (arena : Tropical.Ir.ExprArena)
    (ctx : Context := {}) :
    Except String Tropical.Plan.PerInstancePlan := do
  -- ── Output expressions: map output position → expr, in port order ──
  let outputExprs ← do
    let mut exprs : Array Tropical.Ir.ExprId := #[]
    for i in [0:prog.outputs.size] do
      let out := prog.outputs[i]!
      let assign := prog.assigns.find? fun a =>
        match a.target with
        | .port idx => idx.idx == i
        | .dac => false
      match assign with
      | some a => exprs := exprs.push a.expr
      | none =>
        throw s!"compileResolved: program '{prog.name}' output '{out.name}' has no outputAssign."
    pure exprs

  let inputPortTypes := prog.inputs.map fun d => inputPortType d.type?
  let outputPortScalarCounts ← prog.outputs.mapM outputPortScalarCount

  let emitSlots : EmitSlots := {
    paramHandles := ctx.paramHandles
    paramSlots := ctx.paramSlots
    nestedOutputSlots? := ctx.nestedOutputSlots?
    nestedInputSlots := ctx.nestedInputSlots
    inputSlotOverride := ctx.inputSlotOverride
    inputArraySlots := ctx.inputArraySlots
    nestedInputArraySlots := ctx.nestedInputArraySlots
    nestedOutputArraySlots := ctx.nestedOutputArraySlots }

  let program ← Tropical.Ir.Emit.emitResolvedProgram
    outputExprs outputPortScalarCounts
    inputPortTypes emitSlots arena
    { instances := prog.instances, enclosing := prog }
    ctx.staging

  return {
    registerCount := program.registerCount
    arraySlotCount := program.arraySlotCount
    arraySlotSizes := program.arraySlotSizes
    instructions := program.instructions
    perChildPreInput := program.perChildPreInput
    outputTargets := program.outputTargets
    arraySlotNames := #[]
    instrStages := program.instrStages
    perChildPreInputStages := program.perChildPreInputStages }

end Tropical.Ir.CompileResolved
