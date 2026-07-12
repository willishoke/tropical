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
-/

namespace Tropical.Ir.Strata.IdentityElim

open Tropical.Ir

-- ─────────────────────────────────────────────────────────────
-- Id-form (#190 native-DAG) — mirrors run on EArena
-- ─────────────────────────────────────────────────────────────

private inductive InstFateE where
  | eliminated (outputs : Array ExprId)
  | survivor (newIdx : Nat)
deriving Inhabited

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

/-- nestedOut(eliminated, o) → wired id (walked recursively so chains collapse);
    nestedOut(survivor) → remapped idx. -/
private partial def substExprGoE (fates : Array InstFateE) (id : ExprId) : MapM ExprId :=
  mapExprIdGo {
    node := fun e => match e with
      | .nestedOut inst out =>
        match fates[inst.idx]? with
        | some (.eliminated outs) =>
          match outs[out.idx]? with
          | some v => do pure (some (← substExprGoE fates v))
          | none => pure none
        | some (.survivor newIdx) =>
          if newIdx != inst.idx then do pure (some (← einternP (.nestedOut ⟨newIdx⟩ out)))
          else pure none
        | none => pure none
      | _ => pure none
  } id

private def substExprE (fates : Array InstFateE) (id : ExprId) : PassM ExprId :=
  (substExprGoE fates id).run' {}

def runE (rootIdx : ProgramIdx) : PassM ProgramIdx := do
  let prog ← getEProgram rootIdx "identityElim"
  let mut any := false
  for d in prog.decls do
    if !any then
      if let .inst name typeKey _ inputs := d then
        if (← detectIdentityE prog name typeKey inputs).isSome then
          any := true
  unless any do return rootIdx

  let mut fates : Array InstFateE := #[]
  let mut newPos := 0
  for d in prog.decls do
    if let .inst name typeKey _ inputs := d then
      match ← detectIdentityE prog name typeKey inputs with
      | some outs => fates := fates.push (.eliminated outs)
      | none =>
        fates := fates.push (.survivor newPos)
        newPos := newPos + 1

  let mut newDecls : Array BodyDecl := #[]
  let mut instPos := 0
  for d in prog.decls do
    match d with
    | .inst name typeKey tArgs inputs =>
      let fate := fates[instPos]!
      instPos := instPos + 1
      match fate with
      | .eliminated _ => pure ()
      | .survivor _ =>
        let ins ← inputs.mapM fun i => do
          pure ({ port := i.port, value := ← substExprE fates i.value } : InstanceInput)
        newDecls := newDecls.push (.inst name typeKey tArgs ins)
    | .param .. | .prog .. =>
      newDecls := newDecls.push d
  let newAssigns ← prog.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← substExprE fates a.expr } : OutputAssign)
  pushEProgram { prog with decls := newDecls, assigns := newAssigns }

end Tropical.Ir.Strata.IdentityElim
