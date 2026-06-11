import Tropical.Ir.Nodes
import Tropical.Ir.Recursion
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.Specialize
import Tropical.Ir.Strata.SumLower

/-!
# inlineInstances — port of compiler/ir/inline_instances.ts (Phase 5 pass 3)

Splice each `InstanceDecl` into its parent: specialize the inner
(identity-keyed typeArgs), sumLower it, recursively inline its own
sub-instances (depth-first, bottom-up), substitute wired-in input
expressions, shift surviving Reg/Param/Binding refs by the lift
offsets, lift regs (renamed `${instance}_${inner}`, `_liftedFrom`
OVERWRITTEN with the current instance — each lift re-stamps, so the
post-strata tag is the outermost instance), then resolve every
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

/-- Port of decl_tables.ts `getInstanceType`. -/
private def getInstanceType (arena : Arena) (enclosing : Program)
    (instName typeKey : String) : Except Error (ProgramIdx × Program) := do
  match enclosing.registryGet? typeKey with
  | some pIdx =>
    let some p := arena.program? pIdx
      | throw ⟨s!"getInstanceType: instance '{instName}' typeKey '{typeKey}' program pool index {pIdx.idx} out of range (internal)"⟩
    return (pIdx, p)
  | none =>
    let keys := ", ".intercalate (enclosing.registry.toList.map (·.1))
    throw ⟨s!"getInstanceType: instance '{instName}' typeKey '{typeKey}' " ++
      s!"not found in enclosing program '{enclosing.name}' registry " ++
      s!"(keys: {keys}). This is a registry-build bug; check buildProgramRegistry call sites."⟩

-- ─────────────────────────────────────────────────────────────
-- Functional input-substitution + offset shifting
-- ─────────────────────────────────────────────────────────────

/-- Fresh `Program` from `inner` with every substituted `InputRef`
    replaced by the outer-site expression (passed through unrewritten)
    and every surviving RegRef/ParamRef/BindingRef + binder idx
    shifted by the lift offsets. -/
private def inlineSubstProgram (inner : Program)
    (inputSubst : Array (Nat × Expr))
    (regOffset paramOffset binderOffset : Nat) : Program :=
  let hooks : MapHooks := {
    expr := fun e => match e with
      | .inputRef i => (inputSubst.find? (·.1 == i.idx)).map (·.2)
      | .regRef i => some (.regRef ⟨i.idx + regOffset⟩)
      | .paramRef i => some (.paramRef ⟨i.idx + paramOffset⟩)
      | .bindingRef i => some (.bindingRef ⟨i.idx + binderOffset⟩)
      | _ => none
    binder := fun b => { b with idx := ⟨b.idx.idx + binderOffset⟩ }
  }
  let rw := mapExpr hooks
  let mapDecl : BodyDecl → BodyDecl := fun
    | .reg name init update? type? liftedFrom? =>
      .reg name (rw init) (update?.map rw) type? liftedFrom?
    | .param name value? => .param name value?   -- session-scoped; preserved
    | .inst name typeKey tArgs inputs =>
      -- Post-recurse there should be none; defensive pass-through.
      .inst name typeKey tArgs (inputs.map fun i => { i with value := rw i.value })
    | .prog name p => .prog name p
  { inner with
    inputs := inner.inputs.map fun i => { i with default? := i.default?.map rw }
    outputs := inner.outputs
    decls := inner.decls.map mapDecl
    assigns := inner.assigns.map fun a => { a with expr := rw a.expr }
    binderCount := inner.binderCount + binderOffset }

/-- Lift the cloned inner's body decls: regs renamed
    `${instance}_${name}` with provenance re-stamped to THIS instance;
    params and programDecls as-is. -/
private def liftClonedBody (instanceName : String) (cloned : Program) :
    Except Error (Array BodyDecl) := do
  let mut out : Array BodyDecl := #[]
  for d in cloned.decls do
    match d with
    | .reg name init update? type? _ =>
      out := out.push (.reg s!"{instanceName}_{name}" init update? type? (some instanceName))
    | .param .. | .prog .. => out := out.push d
    | .inst dname .. =>
      throw ⟨s!"inlineInstances: post-recurse: cloned inner '{cloned.name}' still has " ++
        s!"instanceDecl '{dname}' — depth-first invariant violated"⟩
  return out

/-- Record the cloned inner's output expressions by output position
    (TS keys the per-instance table by the template's OutputIdx). -/
private def recordOutputs (instName : String) (declType cloned : Program) :
    Except Error (Array Expr) := do
  let mut out : Array Expr := #[]
  for i in [0:declType.outputs.size] do
    let some clonedOut := cloned.outputs[i]?
      | throw ⟨s!"inlineInstances: instance '{instName}' output arity mismatch " ++
          s!"(template: {declType.outputs.size}, cloned: {cloned.outputs.size})"⟩
    -- TS builds a Map per assign in order: the LAST assign to a
    -- position wins.
    match (cloned.assigns.filter (·.target == OutputTarget.port ⟨i⟩)).back? with
    | some a => out := out.push a.expr
    | none =>
      throw ⟨s!"inlineInstances: instance '{instName}': program '{cloned.name}' has no " ++
        s!"output_assign for output '{clonedOut.name}' (idx {i})"⟩
  return out

/-- The wired expression for an input position; the TS `wiredByIdx`
    Map is built by insertion, so the LAST wire to a port wins. -/
private def wiredFor (inputs : Array InstanceInput) (i : Nat) : Option Expr :=
  ((inputs.filter (·.port.idx == i)).back?).map (·.value)

/-- Port of `buildInputSubst`: wired expressions take priority, then
    the (specialized) inner's declared defaults; an unwired,
    default-less input stays unsubstituted (harmless if unused). -/
private def buildInputSubst (instName : String) (declType flattened : Program)
    (inputs : Array InstanceInput) : Except Error (Array (Nat × Expr)) := do
  let mut subst : Array (Nat × Expr) := #[]
  for i in [0:declType.inputs.size] do
    let some innerPort := flattened.inputs[i]?
      | throw ⟨s!"inlineInstances: instance '{instName}' input arity mismatch " ++
          s!"(template: {declType.inputs.size}, specialized: {flattened.inputs.size})"⟩
    match wiredFor inputs i with
    | some w => subst := subst.push (i, w)
    | none =>
      match innerPort.default? with
      | some d => subst := subst.push (i, d)
      | none => pure ()
  return subst

-- ─────────────────────────────────────────────────────────────
-- NestedOut substitution — exhaustive expression walker
-- ─────────────────────────────────────────────────────────────

/-- `table[instanceIdx][outputIdx]` = recorded cloned output expr. -/
private abbrev NestedOutTable := Array (Array Expr)

private partial def substExpr (table : NestedOutTable) (e : Expr) :
    Except Error Expr := do
  match e with
  | .nestedOut inst out =>
    let some perInstance := table[inst.idx]?
      | throw ⟨s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
          "— instance not inlined?"⟩
    let some v := perInstance[out.idx]?
      | throw ⟨s!"inlineInstances: nestedOut to instance idx={inst.idx} output idx={out.idx} " ++
          "has no resolved expression for that output"⟩
    -- The recorded expression may itself contain nestedOut refs to
    -- outer-scope instances (chained stages); walk it too.
    substExpr table v
  | .num _ | .bool _
  | .inputRef _ | .regRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
  | .sampleRate | .sampleIndex => return e
  | .arr items => return .arr (← items.mapM (substExpr table))
  | .binary tag a b => return .binary tag (← substExpr table a) (← substExpr table b)
  | .unary tag a => return .unary tag (← substExpr table a)
  | .clamp a b c =>
    return .clamp (← substExpr table a) (← substExpr table b) (← substExpr table c)
  | .select a b c =>
    return .select (← substExpr table a) (← substExpr table b) (← substExpr table c)
  | .arraySet a b c =>
    return .arraySet (← substExpr table a) (← substExpr table b) (← substExpr table c)
  | .index a b => return .index (← substExpr table a) (← substExpr table b)
  | .zeros count => return .zeros (← substExpr table count)
  | .fold over init acc elem body =>
    return .fold (← substExpr table over) (← substExpr table init) acc elem
      (← substExpr table body)
  | .scan over init acc elem body =>
    return .scan (← substExpr table over) (← substExpr table init) acc elem
      (← substExpr table body)
  | .generate count iter body =>
    return .generate (← substExpr table count) iter (← substExpr table body)
  | .iterate count init iter body =>
    return .iterate (← substExpr table count) (← substExpr table init) iter
      (← substExpr table body)
  | .chain count init iter body =>
    return .chain (← substExpr table count) (← substExpr table init) iter
      (← substExpr table body)
  | .map2 over elem body =>
    return .map2 (← substExpr table over) elem (← substExpr table body)
  | .zipWith a b x y body =>
    return .zipWith (← substExpr table a) (← substExpr table b) x y
      (← substExpr table body)
  | .letIn binders body =>
    return .letIn
      (← binders.mapM fun b => do pure (.mk b.binder (← substExpr table b.value)))
      (← substExpr table body)
  | .tag d v payload =>
    return .tag d v
      (← payload.mapM fun p => do pure (.mk p.field (← substExpr table p.value)))
  | .match_ d scrutinee arms =>
    return .match_ d (← substExpr table scrutinee)
      (← arms.mapM fun arm => do
        pure (.mk arm.variant arm.binders (← substExpr table arm.body)))

private def substDecl (table : NestedOutTable) : BodyDecl → Except Error BodyDecl
  | .reg name init update? type? liftedFrom? => do
    return .reg name (← substExpr table init)
      (← match update? with
         | some u => some <$> substExpr table u
         | none => pure none)
      type? liftedFrom?
  | .param name value? => return .param name value?
  | .prog name p => return .prog name p
  | .inst name .. =>
    throw ⟨s!"inlineInstances: substDecl on surviving InstanceDecl '{name}'"⟩

-- ─────────────────────────────────────────────────────────────
-- Per-instance inlining + public entry (mutually recursive)
-- ─────────────────────────────────────────────────────────────

private def isRegDecl : BodyDecl → Bool
  | .reg .. => true | _ => false

private def isParamDecl : BodyDecl → Bool
  | .param .. => true | _ => false

mutual

/-- Inline one instance: returns the threaded arena, this instance's
    lifted decls, its recorded output exprs, and the flattened inner's
    binderCount (the caller's running binder offset advances by it). -/
private partial def inlineOne (arena : Arena) (enclosing : Program)
    (instName typeKey : String) (typeArgs : Array InstanceTypeArg)
    (inputs : Array InstanceInput)
    (regOffset paramOffset binderOffset : Nat) :
    Except Error (Arena × Array BodyDecl × Array Expr × Nat) := do
  let (declTypeIdx, declType) ← getInstanceType arena enclosing instName typeKey
  -- 1. Specialize (identity-keyed typeArgs; no-op for concrete inners).
  let (arena, specializedIdx) ←
    if declType.typeParams.isEmpty && typeArgs.isEmpty then
      pure (arena, declTypeIdx)
    else do
      let args ← typeArgs.mapM fun a => do
        let some pd := declType.typeParams[a.param.idx]?
          | throw (⟨s!"inlineInstances: instance '{instName}' typeArg idx={a.param.idx} out of range " ++
              s!"(target '{declType.name}' has {declType.typeParams.size} typeParams)"⟩ : Error)
        let some tp := arena.typeParam? pd
          | throw (⟨s!"inlineInstances: instance '{instName}': typeParam pool index {pd.idx} out of range (internal)"⟩ : Error)
        pure { poolIdx? := some pd, name := tp.name, value := a.value : Specialize.ArgEntry }
      Specialize.runCore arena declTypeIdx args
  -- 2a. Lower sums in the specialized inner BEFORE recursing/lifting
  --     (per-instance ordering must match the outer pipeline's).
  let (arena, summedIdx) ← SumLower.run arena specializedIdx
  -- 2b. Recursively inline sub-instances (depth-first, bottom-up).
  let (arena, flatIdx) ← run arena summedIdx
  let some flattened := arena.program? flatIdx
    | throw ⟨s!"inlineInstances: flattened program pool index {flatIdx.idx} out of range (internal)"⟩
  -- 3. Input substitution map (wired > default > unsubstituted).
  let inputSubst ← buildInputSubst instName declType flattened inputs
  -- 4. Clone with input substitution + idx shifting.
  let cloned := inlineSubstProgram flattened inputSubst regOffset paramOffset binderOffset
  -- 5. Lift body decls into the outer.
  let lifted ← liftClonedBody instName cloned
  -- 6. Record output exprs for the nestedOut substitution.
  let outputs ← recordOutputs instName declType cloned
  return (arena, lifted, outputs, flattened.binderCount)

/-- Public entry — the recursive pass. -/
partial def run (arena : Arena) (rootIdx : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"inlineInstances: program pool index {rootIdx.idx} out of range"⟩
  -- Fast path: no instances at this level — pass through.
  unless prog.decls.any (fun d => match d with | .inst .. => true | _ => false) do
    return (arena, rootIdx)

  let outerRegCount := prog.regs.size
  let outerParamCount := prog.params.size
  let mut arena := arena
  let mut survivingDecls : Array BodyDecl := #[]
  let mut liftedDecls : Array BodyDecl := #[]
  let mut nestedOutSubst : NestedOutTable := #[]
  let mut liftedBinderCount := 0
  for decl in prog.decls do
    match decl with
    | .inst name typeKey typeArgs inputs =>
      let regOffset := outerRegCount + (liftedDecls.filter isRegDecl).size
      let paramOffset := outerParamCount + (liftedDecls.filter isParamDecl).size
      let binderOffset := prog.binderCount + liftedBinderCount
      let (arena', lifted, outputs, innerBinders) ←
        inlineOne arena prog name typeKey typeArgs inputs
          regOffset paramOffset binderOffset
      arena := arena'
      liftedDecls := liftedDecls ++ lifted
      nestedOutSubst := nestedOutSubst.push outputs
      liftedBinderCount := liftedBinderCount + innerBinders
    | _ => survivingDecls := survivingDecls.push decl

  -- Substitute nestedOut refs across surviving AND lifted decls (a
  -- lifted reg's init/update may reference a sibling instance's
  -- output) plus the assigns.
  let newDecls ← (survivingDecls ++ liftedDecls).mapM (substDecl nestedOutSubst)
  let newAssigns ← prog.assigns.mapM fun a => do
    pure { a with expr := ← substExpr nestedOutSubst a.expr }

  let fresh : Program := { prog with
    decls := newDecls
    assigns := newAssigns
    binderCount := prog.binderCount + liftedBinderCount
    -- Post-inline: zero InstanceDecls ⇒ an empty registry suffices.
    registry := #[] }
  return ({ arena with programs := arena.programs.push fresh },
          ⟨arena.programs.size⟩)

end

end Tropical.Ir.Strata.InlineInstances
