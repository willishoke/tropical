import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena

/-!
# identityElim

The categorical identity-law rewrite: an `InstanceDecl` whose program
body is the identity morphism (no decls; every output assigned exactly
one `inputRef`) is a no-op kernel. Every `nestedOut(I, o)` to it
becomes the expression wired into the forwarded input at the consumer
site, and the decl is dropped. Surviving instances' positions shift,
so survivor `nestedOut` refs remap to their new InstanceIdx in the
same walk. Mostly fires on the fractal path (trivial lifted
wire-programs); idempotent; pure.

The unit law is an equation of the theory, not of the free structure:
collapsing `id ∘ id ∘ f` walks the QUOTIENT, whose termination fact is
the instance graph's acyclicity (the elaborator/session contract), not
the arena's child-descending ids. That walk happens HERE, at table
construction, as bounded Jacobi passes over the fates table — one hop
per pass, ≤ #eliminated passes for acyclic chains — so the expression
walks themselves stay structural (total, frozen-src descent) with
one-hop hooks. A chain that fails to resolve (an upstream-acyclicity
bug; a bare forwarding cycle even CONVERGES to self-reference, so
convergence is not the check) dies in `finalizeHooks` with an internal
error, never an unbounded recursion.
-/

namespace Tropical.Ir.Strata.IdentityElim

open Tropical.Ir

private inductive InstFateE where
  | eliminated (outputs : Array ExprId)
  | survivor (newIdx : Nat)
deriving Inhabited, BEq

private def detectIdentityE (enclosing : Program)
    (instName typeKey : String) (inputs : Array InstanceInput) :
    PassM (Option (Array ExprId)) := do
  let (_, target) ← getInstanceTypeE enclosing instName typeKey
  if target.decls.size > 0 then return none
  let mut outs : Array ExprId := #[]
  for oi in [0:target.outputs.size] do
    let assigns := target.assigns.filter (·.target == OutputTarget.port ⟨oi⟩)
    if assigns.size != 1 then return none
    let .inputRef i ← derefP assigns[0]!.expr | return none
    match wiredForE inputs i.idx with
    | some w => outs := outs.push w
    | none => return none
  if outs.isEmpty then return none
  return some outs

/-- Phase-A hook: ONE hop of identity forwarding — an eliminated
    instance's `nestedOut` becomes that instance's recorded output id
    as recorded (not re-walked; the fixpoint loop supplies the
    collapse). Survivors keep their ORIGINAL indices through phase A,
    so re-walking a pass-k result in pass k+1 never misreads a
    remapped index. -/
private def hopHooks (fates : Array InstFateE) : MapHooksId := {
  node := fun e => match e with
    | .nestedOut inst out =>
      match fates[inst.idx]? with
      | some (.eliminated outs) => pure outs[out.idx]?
      | _ => pure none
    | _ => pure none
}

/-- Phase-B hook, applied to the chain-collapsed table values exactly
    once (they are never re-walked, so the remap cannot double-apply):
    survivor indices remap; an eliminated ref surviving phase A means
    the instance graph had a cycle the upstream contract should have
    rejected — an internal error, not a user error. -/
private def finalizeHooks (fates : Array InstFateE) : MapHooksId := {
  node := fun e => match e with
    | .nestedOut inst out =>
      match fates[inst.idx]? with
      | some (.eliminated _) =>
        failP (s!"identityElim: identity chain through instance idx={inst.idx} did not " ++
          "resolve — the instance graph must be acyclic (upstream contract)")
      | some (.survivor newIdx) =>
        if newIdx != inst.idx then do pure (some (← einternP (.nestedOut ⟨newIdx⟩ out)))
        else pure none
      | none => pure none
    | _ => pure none
}

/-- Consumer-site hook (table values are fully final): eliminated →
    the recorded output id; survivor → remapped index. One hop. -/
private def substHooks (fates : Array InstFateE) : MapHooksId := {
  node := fun e => match e with
    | .nestedOut inst out =>
      match fates[inst.idx]? with
      | some (.eliminated outs) =>
        match outs[out.idx]? with
        | some v => pure (some v)
        | none => pure none
      | some (.survivor newIdx) =>
        if newIdx != inst.idx then do pure (some (← einternP (.nestedOut ⟨newIdx⟩ out)))
        else pure none
      | none => pure none
    | _ => pure none
}

def runE (rootIdx : ProgramIdx) : PassM ProgramIdx := do
  let prog ← getEProgram rootIdx "identityElim"
  let mut any := false
  for d in prog.decls do
    if !any then
      if let .inst name typeKey inputs := d then
        if (← detectIdentityE prog name typeKey inputs).isSome then
          any := true
  unless any do return rootIdx

  let mut fates : Array InstFateE := #[]
  let mut newPos := 0
  let mut elimCount := 0
  for d in prog.decls do
    if let .inst name typeKey inputs := d then
      match ← detectIdentityE prog name typeKey inputs with
      | some outs =>
        fates := fates.push (.eliminated outs)
        elimCount := elimCount + 1
      | none =>
        fates := fates.push (.survivor newPos)
        newPos := newPos + 1

  -- ── Phase A: collapse identity→identity chains (Jacobi passes) ──
  -- Each pass hops once through the PREVIOUS pass's table, so a chain
  -- of length L is final after L−1 passes and `elimCount` passes
  -- always suffice (acyclic chains cannot repeat an instance). A
  -- fixpoint pass rewrites to the same ids (dedup), so `==` early-exits.
  for _ in [0:elimCount] do
    let prev := fates
    fates ← withFrozenSrc "identityElim" fun src hw =>
      prev.mapM fun f => match f with
        | .eliminated outs =>
          (InstFateE.eliminated <$>
            outs.mapM (mapExprIdGo src hw (hopHooks prev))).run' {}
        | s => pure s
    if fates == prev then break

  -- ── Phase B: remap survivors inside the collapsed values, once. ──
  let tbl := fates
  fates ← withFrozenSrc "identityElim" fun src hw =>
    tbl.mapM fun f => match f with
      | .eliminated outs =>
        (InstFateE.eliminated <$>
          outs.mapM (mapExprIdGo src hw (finalizeHooks tbl))).run' {}
      | s => pure s

  -- ── Consumer sites: survivor inputs + assigns, one-hop hooks. ──
  let fatesF := fates
  withFrozenSrc "identityElim" fun src hw => do
    let mut newDecls : Array BodyDecl := #[]
    let mut instPos := 0
    for d in prog.decls do
      match d with
      | .inst name typeKey inputs =>
        let fate := fatesF[instPos]!
        instPos := instPos + 1
        match fate with
        | .eliminated _ => pure ()
        | .survivor _ =>
          let ins ← inputs.mapM fun i => do
            pure ({ port := i.port,
                    value := ← mapExprId src hw (substHooks fatesF) i.value } : InstanceInput)
          newDecls := newDecls.push (.inst name typeKey ins)
      | .param .. | .prog .. =>
        newDecls := newDecls.push d
    let newAssigns ← prog.assigns.mapM fun a => do
      pure ({ target := a.target,
              expr := ← mapExprId src hw (substHooks fatesF) a.expr } : OutputAssign)
    pushEProgram { prog with decls := newDecls, assigns := newAssigns }

end Tropical.Ir.Strata.IdentityElim
