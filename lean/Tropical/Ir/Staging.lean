import Tropical.Ir.Core

/-!
# Staging — resolving stage signatures against a binding context

A `StageSig` (computed once at `intern`, see `ExprArena`) is symbolic in
the two parametric leaves: input ports and nested-instance outputs. This
module resolves a signature to a concrete `Stage` given a `StageCtx` —
the binding context one kernel is emitted under.

Availability follows the KERNEL EXECUTION ORDER, not just the graph:
a nested output is resolvable only if that child's body has already run
this sample (children emit in decl order, pre-input block k sees only
siblings j < k, the parent body sees all). A `nestedOut` dep on a
not-yet-run sibling reads the previous sample's slot — genuinely
per-sample, `s1`. This is the graph-level form of the plan pass's
availability rule (a slot read is stage-0 only if its writer precedes
it), with the same conservative answer in the same corner.
-/

namespace Tropical.Ir.Staging

open Tropical.Ir

/-- The binding context a program's expressions resolve under. -/
structure StageCtx where
  /-- Per input port: the stage of the wire expression bound to it
      (in the parent's context). Missing → `s1` (conservative). -/
  inputStages : Array Stage := #[]
  /-- Per instance (by `InstanceIdx`): the child's per-output stages,
      `none` while the child hasn't run yet this sample. -/
  childOut : Array (Option (Array Stage)) := #[]
deriving Inhabited

/-- Resolve one nested-output dependency. -/
def resolveNested (ctx : StageCtx) (dependency : Nat × Nat) : Stage :=
  match ctx.childOut[dependency.1]? with
  | some (some outputs) => outputs[dependency.2]?.getD .s1
  | _ => .s1

/-- Resolve a signature against a context.  The declarative folds are the
    proof-facing form of the original emitter-order loops. -/
def resolve (ctx : StageCtx) (sig : StageSig) : Stage :=
  let afterInputs := sig.inputs.foldl (fun stage inputIdx =>
    stage.join (ctx.inputStages[inputIdx]?.getD .s1)) sig.base
  sig.nested.foldl (fun stage dependency =>
    stage.join (resolveNested ctx dependency)) afterInputs

/-- The stage of an interned node under a context. Dangling id → `s1`. -/
def stageOf (arena : ExprArena) (ctx : StageCtx) (id : ExprId) : Stage :=
  match arena.sig? id with
  | some sig => resolve ctx sig
  | none => .s1

end Tropical.Ir.Staging
