import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena

/-!
# inlineInstances

Splice each `InstanceDecl` into its parent: recursively inline its own
sub-instances (depth-first, bottom-up), substitute wired-in input
expressions, shift surviving Param refs by the lift offset (CF-only:
there are no reg decls to lift or rename), then resolve every
`nestedOut` against the recorded per-instance output expressions.

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
    surviving Param ref shifts by the lift offset. -/
private def inlineSubstHooksE (inputSubst : Array (Nat × ExprId))
    (paramOffset : Nat) : MapHooksId := {
  node := fun e => match e with
    | .inputRef i =>
      match inputSubst.find? (·.1 == i.idx) with
      | some (_, v) => pure (some v)
      | none => pure none
    | .paramRef i => do pure (some (← einternP (.paramRef ⟨i.idx + paramOffset⟩)))
    | _ => pure none
}

private def inlineSubstProgramE (inner : Program)
    (inputSubst : Array (Nat × ExprId))
    (paramOffset : Nat) : PassM Program :=
  -- One memo across all of the clone's roots — the inner's inputs, decls, and
  -- assigns share subgraphs, and the rewrite is root-independent.
  withFrozenSrc "inlineInstances" fun src hw =>
  StateT.run' (s := {}) do
  let rw := mapExprIdGo src hw (inlineSubstHooksE inputSubst paramOffset)
  let inputs ← inner.inputs.mapM fun i => do
    pure ({ i with default? := ← i.default?.mapM rw } : InputDecl)
  let decls ← inner.decls.mapM fun d => do
    match d with
    | .param name value? => pure (BodyDecl.param name value?)
    | .inst name typeKey ins =>
      pure (BodyDecl.inst name typeKey
        (← ins.mapM fun i => do pure ({ i with value := ← rw i.value } : InstanceInput)))
    | .prog name p => pure (BodyDecl.prog name p)
  let assigns ← inner.assigns.mapM fun a => do
    pure ({ a with expr := ← rw a.expr } : OutputAssign)
  pure { inner with inputs := inputs, decls := decls, assigns := assigns }

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

/-- ONE hop through the recorded-output table: a `nestedOut` to an
    inlined sibling becomes that sibling's recorded output id as
    recorded (not re-walked; the fixpoint loop below supplies the
    collapse). -/
private def nestedHopHooks (table : NestedOutTableE) : MapHooksId := {
  node := fun e => match e with
    | .nestedOut inst out =>
      match table[inst.idx]? with
      | none =>
        failP (s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
          "— instance not inlined?")
      | some perInstance =>
        match perInstance[out.idx]? with
        | some v => pure (some v)
        | none =>
          failP (s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
            "has no resolved expression for that output")
    | _ => pure none
}

/-- Post-fixpoint verification: every instance was inlined, so a
    `nestedOut` REMAINING in a final table value means the sibling
    graph had a cycle the upstream contract should have rejected (a
    bare forwarding cycle even converges to self-reference, so the
    fixpoint alone is not the check) — an internal error, never an
    unbounded recursion. -/
private def nestedVerifyHooks : MapHooksId := {
  node := fun e => match e with
    | .nestedOut inst out =>
      failP (s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
        "did not resolve — the instance graph must be acyclic (upstream contract)")
    | _ => pure none
}

private def substDeclNestedE (d : BodyDecl) : PassM BodyDecl := do
  match d with
  | .param .. | .prog .. => pure d
  | .inst name .. => failP s!"inlineInstances: substDecl on surviving InstanceDecl '{name}'"

/-- The recursive flatten + splice, TOTAL by descent on `rootIdx.idx`
    over the frozen pool snapshot: the only recursion follows a
    registry edge of a frozen-pool program, and `hwp` says those edges
    point strictly down. Programs pushed by inner calls are read only
    through `getEProgram` on their returned indices (never recursed
    on), so the snapshot never goes stale for the recursion.

    Phase 1 flattens every instance target (recursion, order-
    independent); phase 2 splices sequentially with the running param
    offsets (no recursion). Same pushes in the same order as the old
    interleaved mutual — only expression-interning order shifts, which
    the `toResolved` GC renumbers away. -/
private def goE (pool : Array Program) (hwp : progPoolWf pool = true)
    (rootIdx : ProgramIdx) : PassM ProgramIdx := do
  match hp : pool[rootIdx.idx]? with
  | none => failP s!"inlineInstances: program pool index {rootIdx.idx} out of range"
  | some prog => do
    unless prog.decls.any (fun d => match d with | .inst .. => true | _ => false) do
      return rootIdx

    -- ── Phase 1: recursively flatten every instance target. ──
    let flats : Array (Option (Program × Program)) ← prog.decls.mapM fun d => do
      match d with
      | .inst name typeKey _ =>
        match hr : prog.registryGet? typeKey with
        | some declTypeIdx =>
          match pool[declTypeIdx.idx]? with
          | none =>
            failP s!"inlineInstances: registry target {declTypeIdx.idx} out of range (internal)"
          | some declType => do
            let flatIdx ← goE pool hwp declTypeIdx
            let flattened ← getEProgram flatIdx "inlineInstances"
            pure (some (declType, flattened))
        | none =>
          let keys := ", ".intercalate (prog.registry.toList.map (·.1))
          failP (s!"getInstanceType: instance '{name}' typeKey '{typeKey}' " ++
            s!"not found in enclosing program '{prog.name}' registry " ++
            s!"(keys: {keys}). This is a registry-build bug; check buildProgramRegistry call sites.")
      | _ => pure none

    -- ── Phase 2: splice sequentially with the running offsets. ──
    let outerParamCount := (prog.decls.filter isParamDeclE).size
    let mut survivingDecls : Array BodyDecl := #[]
    let mut liftedDecls : Array BodyDecl := #[]
    let mut nestedOutSubst : NestedOutTableE := #[]
    for (decl, flat?) in prog.decls.zip flats do
      match decl, flat? with
      | .inst name _ inputs, some (declType, flattened) =>
        let paramOffset := outerParamCount + (liftedDecls.filter isParamDeclE).size
        let inputSubst ← buildInputSubstE name declType flattened inputs
        let cloned ← inlineSubstProgramE flattened inputSubst paramOffset
        let lifted ← liftClonedBodyE cloned
        let outputs ← recordOutputsE name declType cloned
        liftedDecls := liftedDecls ++ lifted
        nestedOutSubst := nestedOutSubst.push outputs
      | .inst name .., none =>
        failP s!"inlineInstances: internal phase mismatch for instance '{name}'"
      | d, _ => survivingDecls := survivingDecls.push d

    -- ── Normalize the recorded-output table (bounded Jacobi passes). ──
    -- Entries reference sibling instances through the outer wiring; the
    -- sibling graph is acyclic (elaborator/session contract), so
    -- `n` one-hop passes collapse every chain — the quotient walk that
    -- used to be the hook's own recursion is now table construction. A
    -- fixpoint pass rewrites to the same ids (dedup), so `==` early-exits.
    for _ in [0:nestedOutSubst.size] do
      let prev := nestedOutSubst
      nestedOutSubst ← withFrozenSrc "inlineInstances" fun src hw =>
        prev.mapM fun outs =>
          (outs.mapM (mapExprIdGo src hw (nestedHopHooks prev))).run' {}
      if nestedOutSubst == prev then break
    let table := nestedOutSubst
    _ ← withFrozenSrc "inlineInstances" fun src hw =>
      table.mapM fun outs =>
        (outs.mapM (mapExprIdGo src hw nestedVerifyHooks)).run' {}

    -- Surviving + lifted decls are param/prog only (CF-only: no regs to carry a
    -- sibling's output), so substDecl is the identity-plus-assertion here.
    let newDecls ← (survivingDecls ++ liftedDecls).mapM substDeclNestedE
    let newAssigns ← withFrozenSrc "inlineInstances" fun src hw =>
      prog.assigns.mapM fun a => do
        pure ({ a with expr := ← mapExprId src hw (nestedHopHooks table) a.expr } : OutputAssign)

    pushEProgram { prog with
      decls := newDecls
      assigns := newAssigns
      registry := #[] }
termination_by rootIdx.idx
decreasing_by exact progPool_registry_lt hwp hp hr

/-- The pass entry: snapshot the program pool, check pool-wf once (an
    O(edges) sweep that buys the whole recursion's termination measure
    — every pool is child-descending by construction, so the failure
    arm is a construction-order bug, never a user error), and descend. -/
def runE (rootIdx : ProgramIdx) : PassM ProgramIdx := do
  let pool := (← get).programs
  if hwp : progPoolWf pool then goE pool hwp rootIdx
  else failP "inlineInstances: program pool is not child-descending (internal construction-order bug)"

end Tropical.Ir.Strata.InlineInstances
