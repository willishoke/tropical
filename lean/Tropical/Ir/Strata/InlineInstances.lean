import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.Specialize
import Tropical.Ir.Strata.SumLower
import Tropical.Ir.Strata.EArena

/-!
# inlineInstances — port of compiler/ir/inline_instances.ts (Phase 5 pass 3)

Splice each `InstanceDecl` into its parent: specialize the inner
(identity-keyed typeArgs), sumLower it, recursively inline its own
sub-instances (depth-first, bottom-up), substitute wired-in input
expressions, shift surviving Param/Binding refs by the lift offsets
(CF-only: there are no reg decls to lift or rename), then resolve every
`nestedOut` against the recorded per-instance output expressions.

The TS pass memoizes the nestedOut substitution walk on node identity
to preserve DAG sharing. Expr sharing is invisible to the
`tropical_resolved_1` codec (only the three arena pools carry
identity), and the unmemoized walk is O(output tree) — the same bound
as encoding — so the Lean port walks structurally.

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

private def isParamDeclE : EBodyDecl → Bool
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

private def inlineSubstProgramE (inner : EProgram)
    (inputSubst : Array (Nat × ExprId))
    (paramOffset binderOffset : Nat) : PassM EProgram := do
  let rw := mapExprId (inlineSubstHooksE inputSubst paramOffset binderOffset)
  let inputs ← inner.inputs.mapM fun i => do
    pure ({ i with default? := ← i.default?.mapM rw } : EInputDecl)
  let decls ← inner.decls.mapM fun d => do
    match d with
    | .param name value? => pure (EBodyDecl.param name value?)
    | .inst name typeKey tArgs ins =>
      pure (EBodyDecl.inst name typeKey tArgs
        (← ins.mapM fun i => do pure ({ i with value := ← rw i.value } : EInstanceInput)))
    | .prog name p => pure (EBodyDecl.prog name p)
  let assigns ← inner.assigns.mapM fun a => do
    pure ({ a with expr := ← rw a.expr } : EOutputAssign)
  pure { inner with inputs := inputs, decls := decls, assigns := assigns, binderCount := inner.binderCount + binderOffset }

private def liftClonedBodyE (cloned : EProgram) : PassM (Array EBodyDecl) := do
  let mut out : Array EBodyDecl := #[]
  for d in cloned.decls do
    match d with
    | .param .. | .prog .. => out := out.push d
    | .inst dname .. =>
      failP (s!"inlineInstances: post-recurse: cloned inner '{cloned.name}' still has " ++
        s!"instanceDecl '{dname}' — depth-first invariant violated")
  return out

private def recordOutputsE (instName : String) (declType cloned : EProgram) :
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

private def wiredForE (inputs : Array EInstanceInput) (i : Nat) : Option ExprId :=
  ((inputs.filter (·.port.idx == i)).back?).map (·.value)

private def buildInputSubstE (instName : String) (declType flattened : EProgram)
    (inputs : Array EInstanceInput) : PassM (Array (Nat × ExprId)) := do
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

private partial def substExprNestedE (table : NestedOutTableE) (id : ExprId) : PassM ExprId :=
  mapExprId {
    node := fun e => match e with
      | .nestedOut inst out =>
        match table[inst.idx]? with
        | none =>
          failP (s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
            "— instance not inlined?")
        | some perInstance =>
          match perInstance[out.idx]? with
          | some v => do pure (some (← substExprNestedE table v))
          | none =>
            failP (s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
              "has no resolved expression for that output")
      | _ => pure none
  } id

private def substDeclNestedE (d : EBodyDecl) : PassM EBodyDecl := do
  match d with
  | .param .. | .prog .. => pure d
  | .inst name .. => failP s!"inlineInstances: substDecl on surviving InstanceDecl '{name}'"

mutual

/-- Inline one instance: returns its lifted decls, recorded output ids, and the
    flattened inner's binderCount (the caller's running binder offset advances by
    it). The shared expr DAG threads through `PassM`. -/
private partial def inlineOneE (enclosing : EProgram)
    (instName typeKey : String) (typeArgs : Array InstanceTypeArg)
    (inputs : Array EInstanceInput)
    (paramOffset binderOffset : Nat) :
    PassM (Array EBodyDecl × Array ExprId × Nat) := do
  let (declTypeIdx, declType) ← getInstanceTypeE enclosing instName typeKey
  -- 1. Specialize (identity-keyed typeArgs; no-op for concrete inners).
  let specializedIdx ←
    if declType.typeParams.isEmpty && typeArgs.isEmpty then
      pure declTypeIdx
    else do
      let args ← typeArgs.mapM fun a => do
        let some pd := declType.typeParams[a.param.idx]?
          | failP (s!"inlineInstances: instance '{instName}' typeArg idx={a.param.idx} out of range " ++
              s!"(target '{declType.name}' has {declType.typeParams.size} typeParams)")
        let some tp ← typeParamP? pd
          | failP s!"inlineInstances: instance '{instName}': typeParam pool index {pd.idx} out of range (internal)"
        pure { poolIdx? := some pd, name := tp.name, value := a.value : Specialize.ArgEntry }
      Specialize.runCoreE declTypeIdx args
  -- 2a. Lower sums before recursing/lifting.
  let summedIdx ← SumLower.runE specializedIdx
  -- 2b. Recursively inline sub-instances (depth-first, bottom-up).
  let flatIdx ← runE summedIdx
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
  let mut survivingDecls : Array EBodyDecl := #[]
  let mut liftedDecls : Array EBodyDecl := #[]
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
    pure ({ a with expr := ← substExprNestedE nestedOutSubst a.expr } : EOutputAssign)

  pushEProgram { prog with
    decls := newDecls
    assigns := newAssigns
    binderCount := prog.binderCount + liftedBinderCount
    registry := #[] }

end

end Tropical.Ir.Strata.InlineInstances
