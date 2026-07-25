import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena

/-!
# inlineInstances

Splice each `InstanceDecl` into its parent: recursively inline its own
sub-instances (depth-first, bottom-up), substitute wired-in input
expressions, shift surviving Param/Binding refs by the lift offsets
(CF-only: there are no reg decls to lift or rename), then resolve every
`nestedOut` against the recorded per-instance output expressions.
(Inners are monomorphic and sum-free by contract — generics and the
sum-type lowering are retired.)

The TS pass memoizes the nestedOut substitution walk on node identity
to preserve DAG sharing; the id-form walks here memoize the same way
(`mapExprIdGo` under `MapM`). Phase B threads the DAG straight to emit
with no tree encoding anywhere downstream, so an unmemoized walk would
be the only place paying O(expanded tree) — super-linear on composed
patches (the residue-composition compile wall).

Substituted outer expressions pass through WITHOUT rewriting (TS
returns them by reference; offsets must not touch outer-coordinate
refs). Errors are comparable outputs (byte-exact TS messages).
-/

namespace Tropical.Ir.Strata.InlineInstances

open Lean (JsonNumber)
open Tropical.Ir

-- ─────────────────────────────────────────────────────────────
-- Id-form (#190 native-DAG) — the substitution stores shared ExprIds,
-- so the modulated-clock duplication never materializes.
-- ─────────────────────────────────────────────────────────────

private def isParamDeclE : BodyDecl → Bool
  | .param .. => true | _ => false

/-- Clone hooks: `inputRef → wired id` (SHARED — the outer expression passes
    through untouched and is never re-walked, so the bloat never forms); every
    surviving Param/Binding ref + binder idx shifts by the lift offsets. -/
private def inlineSubstHooksE (inputSubst : Array (Nat × ExprId))
    (paramOffset binderOffset : Nat) : MapHooksId := {
  node := fun e => match e with
    | .inputRef i =>
      match inputSubst.find? (·.1 == i.idx) with
      | some (_, v) => pure (some v)
      | none => pure none
    | .paramRef i => do pure (some (← einternP (.paramRef ⟨i.idx + paramOffset⟩)))
    | .bindingRef i => do pure (some (← einternP (.bindingRef ⟨i.idx + binderOffset⟩)))
    | _ => pure none
  binder := fun b => { b with idx := ⟨b.idx.idx + binderOffset⟩ }
}

private def inlineSubstProgramE (inner : Program)
    (inputSubst : Array (Nat × ExprId))
    (paramOffset binderOffset : Nat) : PassM Program :=
  -- One memo across all of the clone's roots — the inner's inputs, decls, and
  -- assigns share subgraphs, and the rewrite is root-independent.
  StateT.run' (s := {}) do
  let rw := mapExprIdGo (inlineSubstHooksE inputSubst paramOffset binderOffset)
  let inputs ← inner.inputs.mapM fun i => do
    pure ({ i with default? := ← i.default?.mapM rw } : InputDecl)
  let decls ← inner.decls.mapM fun d => do
    match d with
    | .param name value? => pure (BodyDecl.param name value?)
    | .inst name typeKey tArgs ins =>
      pure (BodyDecl.inst name typeKey tArgs
        (← ins.mapM fun i => do pure ({ i with value := ← rw i.value } : InstanceInput)))
    | .prog name p => pure (BodyDecl.prog name p)
  let assigns ← inner.assigns.mapM fun a => do
    pure ({ a with expr := ← rw a.expr } : OutputAssign)
  pure { inner with inputs := inputs, decls := decls, assigns := assigns, binderCount := inner.binderCount + binderOffset }

private def liftClonedBodyE (cloned : Program) : PassM (Array BodyDecl) := do
  let mut out : Array BodyDecl := #[]
  for d in cloned.decls do
    match d with
    | .param .. | .prog .. => out := out.push d
    | .inst dname .. =>
      failP (s!"inlineInstances: post-recurse: cloned inner '{cloned.name}' still has " ++
        s!"instanceDecl '{dname}' — depth-first invariant violated")
  return out

private def recordOutputsE (instName : String) (declType cloned : Program) :
    PassM (Array ExprId) := do
  let mut out : Array ExprId := #[]
  for i in [0:declType.outputs.size] do
    let some clonedOut := cloned.outputs[i]?
      | failP (s!"inlineInstances: instance '{instName}' output arity mismatch " ++
          s!"(template: {declType.outputs.size}, cloned: {cloned.outputs.size})")
    match (cloned.assigns.filter (·.target == OutputTarget.port ⟨i⟩)).back? with
    | some a => out := out.push a.expr
    | none =>
      failP (s!"inlineInstances: instance '{instName}': program '{cloned.name}' has no " ++
        s!"output_assign for output '{clonedOut.name}' (idx {i})")
  return out

private def buildInputSubstE (instName : String) (declType flattened : Program)
    (inputs : Array InstanceInput) : PassM (Array (Nat × ExprId)) := do
  let mut subst : Array (Nat × ExprId) := #[]
  for i in [0:declType.inputs.size] do
    let some innerPort := flattened.inputs[i]?
      | failP (s!"inlineInstances: instance '{instName}' input arity mismatch " ++
          s!"(template: {declType.inputs.size}, specialized: {flattened.inputs.size})")
    match wiredForE inputs i with
    | some w => subst := subst.push (i, w)
    | none =>
      match innerPort.default? with
      | some d => subst := subst.push (i, d)
      | none => pure ()
  return subst

/-- `table[instanceIdx][outputIdx]` = recorded cloned output id. -/
private abbrev NestedOutTableE := Array (Array ExprId)

private partial def substExprNestedGoE (table : NestedOutTableE) (id : ExprId) : MapM ExprId :=
  mapExprIdGo {
    node := fun e => match e with
      | .nestedOut inst out =>
        match table[inst.idx]? with
        | none =>
          failP (s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
            "— instance not inlined?")
        | some perInstance =>
          match perInstance[out.idx]? with
          | some v => do pure (some (← substExprNestedGoE table v))
          | none =>
            failP (s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
              "has no resolved expression for that output")
      | _ => pure none
  } id

private def substExprNestedE (table : NestedOutTableE) (id : ExprId) : PassM ExprId :=
  (substExprNestedGoE table id).run' {}

private def substDeclNestedE (d : BodyDecl) : PassM BodyDecl := do
  match d with
  | .param .. | .prog .. => pure d
  | .inst name .. => failP s!"inlineInstances: substDecl on surviving InstanceDecl '{name}'"

mutual

/-- Inline one instance: returns its lifted decls, recorded output ids, and the
    flattened inner's binderCount (the caller's running binder offset advances by
    it). The shared expr DAG threads through `PassM`. -/
private partial def inlineOneE (enclosing : Program)
    (instName typeKey : String) (typeArgs : Array InstanceTypeArg)
    (inputs : Array InstanceInput)
    (paramOffset binderOffset : Nat) :
    PassM (Array BodyDecl × Array ExprId × Nat) := do
  let (declTypeIdx, declType) ← getInstanceTypeE enclosing instName typeKey
  -- 1. Generics are retired: a concrete instance carries no type args and
  --    its target declares no type params. (The specialize/sumLower passes
  --    that ran here were retired 2026-07-25 with their producers.)
  unless declType.typeParams.isEmpty && typeArgs.isEmpty do
    failP s!"inlineInstances: instance '{instName}' (target '{declType.name}') carries type args — generics are retired"
  -- 2. Recursively inline sub-instances (depth-first, bottom-up).
  let flatIdx ← runE declTypeIdx
  let flattened ← getEProgram flatIdx "inlineInstances"
  -- 3. Input substitution map (wired > default > unsubstituted).
  let inputSubst ← buildInputSubstE instName declType flattened inputs
  -- 4. Clone with input substitution + idx shifting.
  let cloned ← inlineSubstProgramE flattened inputSubst paramOffset binderOffset
  -- 5. Lift body decls into the outer.
  let lifted ← liftClonedBodyE cloned
  -- 6. Record output exprs for the nestedOut substitution.
  let outputs ← recordOutputsE instName declType cloned
  return (lifted, outputs, flattened.binderCount)

partial def runE (rootIdx : ProgramIdx) : PassM ProgramIdx := do
  let prog ← getEProgram rootIdx "inlineInstances"
  unless prog.decls.any (fun d => match d with | .inst .. => true | _ => false) do
    return rootIdx

  let outerParamCount := (prog.decls.filter isParamDeclE).size
  let mut survivingDecls : Array BodyDecl := #[]
  let mut liftedDecls : Array BodyDecl := #[]
  let mut nestedOutSubst : NestedOutTableE := #[]
  let mut liftedBinderCount := 0
  for decl in prog.decls do
    match decl with
    | .inst name typeKey typeArgs inputs =>
      let paramOffset := outerParamCount + (liftedDecls.filter isParamDeclE).size
      let binderOffset := prog.binderCount + liftedBinderCount
      let (lifted, outputs, innerBinders) ←
        inlineOneE prog name typeKey typeArgs inputs paramOffset binderOffset
      liftedDecls := liftedDecls ++ lifted
      nestedOutSubst := nestedOutSubst.push outputs
      liftedBinderCount := liftedBinderCount + innerBinders
    | _ => survivingDecls := survivingDecls.push decl

  -- Surviving + lifted decls are param/prog only (CF-only: no regs to carry a
  -- sibling's output), so substDecl is the identity-plus-assertion here.
  let newDecls ← (survivingDecls ++ liftedDecls).mapM substDeclNestedE
  let newAssigns ← prog.assigns.mapM fun a => do
    pure ({ a with expr := ← substExprNestedE nestedOutSubst a.expr } : OutputAssign)

  pushEProgram { prog with
    decls := newDecls
    assigns := newAssigns
    binderCount := prog.binderCount + liftedBinderCount
    registry := #[] }

end

end Tropical.Ir.Strata.InlineInstances
