import Tropical.Ir.Nodes
import Tropical.Ir.Recursion
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena

/-!
# identityElim — port of compiler/ir/identity_elim.ts (Phase 5 pass 5)

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

private inductive InstFate where
  | eliminated (outputs : Array Expr)
  | survivor (newIdx : Nat)
deriving Inhabited

/-- The wired expression for an input position (TS Map insertion:
    LAST wire to a port wins). -/
private def wiredFor (inputs : Array InstanceInput) (i : Nat) : Option Expr :=
  ((inputs.filter (·.port.idx == i)).back?).map (·.value)

/-- Recognize an identity instance: target body has no decls, and each
    output forwards exactly one wired input. Returns the per-output
    substitution expressions (dense over output positions). -/
private def detectIdentity (arena : Arena) (enclosing : Program)
    (instName typeKey : String) (inputs : Array InstanceInput) :
    Except Error (Option (Array Expr)) := do
  let (_, target) ← getInstanceType arena enclosing instName typeKey
  -- No state, no nested instances, no parameters in the body.
  if target.decls.size > 0 then return none
  let mut outs : Array Expr := #[]
  for oi in [0:target.outputs.size] do
    let assigns := target.assigns.filter (·.target == OutputTarget.port ⟨oi⟩)
    if assigns.size != 1 then return none   -- missing or duplicated
    let .inputRef i := assigns[0]!.expr | return none
    match wiredFor inputs i.idx with
    | some w => outs := outs.push w
    | none => return none   -- unwired forwarded input — defensive
  -- Must have at least one output to be a meaningful identity.
  if outs.isEmpty then return none
  return some outs

/-- nestedOut(eliminated, o) → wired expr (walked recursively so
    identity chains collapse); nestedOut(survivor) → remapped idx. -/
private partial def substExpr (fates : Array InstFate) (e : Expr) : Expr :=
  mapExpr {
    expr := fun e => match e with
      | .nestedOut inst out =>
        match fates[inst.idx]? with
        | some (.eliminated outs) =>
          match outs[out.idx]? with
          | some v => some (substExpr fates v)
          | none => some e
        | some (.survivor newIdx) =>
          if newIdx != inst.idx then some (.nestedOut ⟨newIdx⟩ out) else some e
        | none => some e
      | _ => none
  } e

def run (arena : Arena) (rootIdx : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"identityElim: program pool index {rootIdx.idx} out of range"⟩

  -- Fast path: no identities on the input — pass through.
  let mut any := false
  for d in prog.decls do
    if !any then
      if let .inst name typeKey _ inputs := d then
        if (← detectIdentity arena prog name typeKey inputs).isSome then
          any := true
  unless any do return (arena, rootIdx)

  -- Position-keyed fates: eliminated outputs or survivor remap.
  let mut fates : Array InstFate := #[]
  let mut newPos := 0
  for d in prog.decls do
    if let .inst name typeKey _ inputs := d then
      match ← detectIdentity arena prog name typeKey inputs with
      | some outs => fates := fates.push (.eliminated outs)
      | none =>
        fates := fates.push (.survivor newPos)
        newPos := newPos + 1

  let sub := substExpr fates
  let mut newDecls : Array BodyDecl := #[]
  let mut instPos := 0
  for d in prog.decls do
    match d with
    | .inst name typeKey tArgs inputs =>
      let fate := fates[instPos]!
      instPos := instPos + 1
      match fate with
      | .eliminated _ => pure ()   -- dropped
      | .survivor _ =>
        newDecls := newDecls.push (.inst name typeKey tArgs
          (inputs.map fun i => { i with value := sub i.value }))
    | .param .. | .prog .. =>
      newDecls := newDecls.push d
  let newAssigns := prog.assigns.map fun a => { a with expr := sub a.expr }

  let fresh : Program := { prog with decls := newDecls, assigns := newAssigns }
  return ({ arena with programs := arena.programs.push fresh },
          ⟨arena.programs.size⟩)

-- ─────────────────────────────────────────────────────────────
-- Id-form (#190 native-DAG) — mirrors run on EArena
-- ─────────────────────────────────────────────────────────────

private inductive InstFateE where
  | eliminated (outputs : Array ExprId)
  | survivor (newIdx : Nat)
deriving Inhabited

private def wiredForE (inputs : Array EInstanceInput) (i : Nat) : Option ExprId :=
  ((inputs.filter (·.port.idx == i)).back?).map (·.value)

private def detectIdentityE (enclosing : EProgram)
    (instName typeKey : String) (inputs : Array EInstanceInput) :
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
private partial def substExprE (fates : Array InstFateE) (id : ExprId) : PassM ExprId :=
  mapExprId {
    node := fun e => match e with
      | .nestedOut inst out =>
        match fates[inst.idx]? with
        | some (.eliminated outs) =>
          match outs[out.idx]? with
          | some v => do pure (some (← substExprE fates v))
          | none => pure none
        | some (.survivor newIdx) =>
          if newIdx != inst.idx then do pure (some (← einternP (.nestedOut ⟨newIdx⟩ out)))
          else pure none
        | none => pure none
      | _ => pure none
  } id

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

  let mut newDecls : Array EBodyDecl := #[]
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
          pure ({ port := i.port, value := ← substExprE fates i.value } : EInstanceInput)
        newDecls := newDecls.push (.inst name typeKey tArgs ins)
    | .param .. | .prog .. =>
      newDecls := newDecls.push d
  let newAssigns ← prog.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← substExprE fates a.expr } : EOutputAssign)
  pushEProgram { prog with decls := newDecls, assigns := newAssigns }

end Tropical.Ir.Strata.IdentityElim
