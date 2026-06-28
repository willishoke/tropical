import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena

/-!
# sumLower — port of compiler/ir/sum_lower.ts (Phase 5 pass 2)

Lowers every `match`/`tag` expression to scalar select-chains and
variant-index literals. CF-only removed sum-typed registers — the only
construct that carried variant payloads across samples — so there is no
register decomposition left, just the expression-level sum lowering.

TS identity discipline, mirrored exactly where it is observable:
  - variant/field `===` checks → (typeDef pool idx, position) equality
  - `slotKey` map lookups → NAME-keyed (`variant__field`), because TS
    keys those maps by name strings — a cross-type lookup (match arms
    over a different sum type than the scrutinee reg's) matches by
    name in TS, and must here too.

Errors are comparable outputs (byte-exact TS messages).
-/

namespace Tropical.Ir.Strata.SumLower

open Lean (JsonNumber)
open Tropical.Ir

private def nat0 (n : Nat) : Expr := .num ⟨Int.ofNat n, 0⟩

private def zeroE : Expr := nat0 0

-- ─────────────────────────────────────────────────────────────
-- Rewriting context
-- ─────────────────────────────────────────────────────────────

private structure Ctx where
  arena : Arena
  /-- Active per-binder substitutions from match arms, innermost
      first (lookup is find-first, so inner shadows outer). -/
  bindings : List (BinderIdx × Expr) := []

private def Ctx.withBindings (ctx : Ctx) (extra : List (BinderIdx × Expr)) : Ctx :=
  if extra.isEmpty then ctx else { ctx with bindings := extra ++ ctx.bindings }

/-- Resolve a match's sum def: variants + name. -/
private def sumDef (ctx : Ctx) (d : TypeDefIdx) :
    Except Error (String × Array SumVariant) :=
  match ctx.arena.typeDef? d with
  | some (.sum name variants) => .ok (name, variants)
  | _ => .error ⟨s!"sumLower: typeDef pool index {d.idx} is not a sum type (internal)"⟩

-- ─────────────────────────────────────────────────────────────
-- Expression rewriting (mutually recursive with match lowering)
-- ─────────────────────────────────────────────────────────────

mutual

private partial def rewriteExpr (ctx : Ctx) (e : Expr) : Except Error Expr := do
  match e with
  | .num _ | .bool _ => return e
  | .arr items => return .arr (← items.mapM (rewriteExpr ctx))
  -- Bindings: substitute when the binder is in the active map.
  | .bindingRef i =>
    match ctx.bindings.find? (·.1 == i) with
    | some (_, sub) => return sub
    | none => return e
  -- Tag in expression position (no payload) → variant index.
  | .tag d variant payload =>
    let (_, variants) ← sumDef ctx d
    let some v := variants[variant]?
      | throw ⟨s!"sumLower: tag variant index {variant} out of range (internal)"⟩
    if payload.isEmpty then
      return nat0 variant
    throw ⟨s!"sumLower: bare tag with payload (variant '{v.name}') in non-update context"⟩
  -- Match: lower to a scalar select chain.
  | .match_ d scrutinee arms => lowerMatchToSelectChain ctx d scrutinee arms
  -- Pass-through references / leaves.
  | .inputRef _ | .paramRef _ | .typeParamRef _
  | .sampleRate | .sampleIndex | .nestedOut _ _ => return e
  -- Operators.
  | .binary tag lhs rhs => return .binary tag (← rewriteExpr ctx lhs) (← rewriteExpr ctx rhs)
  | .unary tag arg => return .unary tag (← rewriteExpr ctx arg)
  | .clamp a b c => return .clamp (← rewriteExpr ctx a) (← rewriteExpr ctx b) (← rewriteExpr ctx c)
  | .select a b c => return .select (← rewriteExpr ctx a) (← rewriteExpr ctx b) (← rewriteExpr ctx c)
  | .index a b => return .index (← rewriteExpr ctx a) (← rewriteExpr ctx b)
  | .arraySet a b c => return .arraySet (← rewriteExpr ctx a) (← rewriteExpr ctx b) (← rewriteExpr ctx c)
  | .zeros count => return .zeros (← rewriteExpr ctx count)
  -- Combinators (binders pass through; bodies get rewritten).
  | .fold over init acc elem body =>
    return .fold (← rewriteExpr ctx over) (← rewriteExpr ctx init) acc elem
      (← rewriteExpr ctx body)
  | .scan over init acc elem body =>
    return .scan (← rewriteExpr ctx over) (← rewriteExpr ctx init) acc elem
      (← rewriteExpr ctx body)
  | .generate count iter body =>
    return .generate (← rewriteExpr ctx count) iter (← rewriteExpr ctx body)
  | .iterate count init iter body =>
    return .iterate (← rewriteExpr ctx count) (← rewriteExpr ctx init) iter
      (← rewriteExpr ctx body)
  | .chain count init iter body =>
    return .chain (← rewriteExpr ctx count) (← rewriteExpr ctx init) iter
      (← rewriteExpr ctx body)
  | .map2 over elem body =>
    return .map2 (← rewriteExpr ctx over) elem (← rewriteExpr ctx body)
  | .zipWith a b x y body =>
    return .zipWith (← rewriteExpr ctx a) (← rewriteExpr ctx b) x y
      (← rewriteExpr ctx body)
  | .letIn binders body =>
    return .letIn
      (← binders.mapM fun b => do pure (.mk b.binder (← rewriteExpr ctx b.value)))
      (← rewriteExpr ctx body)

/-- Lower a `match` to a select chain over the scrutinee's tag read. -/
private partial def lowerMatchToSelectChain (ctx : Ctx) (d : TypeDefIdx)
    (scrutinee : Expr) (arms : Array MatchArm) : Except Error Expr := do
  let tagRead ← rewriteExpr ctx scrutinee
  let (typeName, variants) ← sumDef ctx d
  let armFor (vi : Nat) (v : SumVariant) : Except Error MatchArm :=
    match arms.find? (·.variant == vi) with
    | some arm => .ok arm
    | none => .error ⟨s!"sumLower: match on '{typeName}' missing arm for '{v.name}'"⟩
  let lowerArmBody (arm : MatchArm) : Except Error Expr := do
    if arm.binders.isEmpty then
      rewriteExpr ctx arm.body
    else
      let subs ← bindingsForArm ctx scrutinee d arm
      rewriteExpr (ctx.withBindings subs) arm.body
  if variants.isEmpty then
    throw ⟨s!"sumLower: match on '{typeName}' has no variants (internal)"⟩
  let lastIdx := variants.size - 1
  let mut chain : Expr ← lowerArmBody (← armFor lastIdx variants[lastIdx]!)
  for k in [1:variants.size] do
    let i := lastIdx - k
    let arm ← armFor i variants[i]!
    chain := .select (.binary .eq tagRead (nat0 i)) (← lowerArmBody arm) chain
  return chain

/-- Per-arm payload bindings. Under CF-only the only scrutinee that
    could supply payload slots was a sum-typed `reg` (now unrepresentable),
    so a match with payload binders has no slot source to read from. -/
private partial def bindingsForArm (ctx : Ctx) (scrutinee : Expr)
    (matchDef : TypeDefIdx) (arm : MatchArm) :
    Except Error (List (BinderIdx × Expr)) := do
  let _ := scrutinee
  if arm.binders.isEmpty then return []
  let (_, matchVariants) ← sumDef ctx matchDef
  let some armVariant := matchVariants[arm.variant]?
    | throw ⟨s!"sumLower: match arm variant index {arm.variant} out of range (internal)"⟩
  throw ⟨s!"sumLower: match arm '{armVariant.name}' has payload bindings, but " ++
    "CF-only removed sum-typed registers — there is no slot source to bind from."⟩

end

-- ─────────────────────────────────────────────────────────────
-- Phase 2 — decl construction
-- ─────────────────────────────────────────────────────────────

private def buildNewDecls (ctx : Ctx) (decls : Array BodyDecl) :
    Except Error (Array BodyDecl) := do
  let mut out : Array BodyDecl := #[]
  for decl in decls do
    match decl with
    | .inst name typeKey tArgs inputs =>
      out := out.push (.inst name typeKey tArgs
        (← inputs.mapM fun i => do pure { i with value := ← rewriteExpr ctx i.value }))
    | .param .. | .prog .. =>
      out := out.push decl
  return out

-- ─────────────────────────────────────────────────────────────
-- Fast-path detection: skip if no work to do
-- ─────────────────────────────────────────────────────────────

private partial def exprHasSumExpr : Expr → Bool
  | .tag .. | .match_ .. => true
  | .num _ | .bool _
  | .inputRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
  | .sampleRate | .sampleIndex | .nestedOut _ _ => false
  | .arr items => items.any exprHasSumExpr
  | .binary _ a b | .index a b => exprHasSumExpr a || exprHasSumExpr b
  | .unary _ a => exprHasSumExpr a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    exprHasSumExpr a || exprHasSumExpr b || exprHasSumExpr c
  | .zeros count => exprHasSumExpr count
  | .fold over init _ _ body | .scan over init _ _ body =>
    exprHasSumExpr over || exprHasSumExpr init || exprHasSumExpr body
  | .iterate count init _ body | .chain count init _ body =>
    exprHasSumExpr count || exprHasSumExpr init || exprHasSumExpr body
  | .generate count _ body => exprHasSumExpr count || exprHasSumExpr body
  | .map2 over _ body => exprHasSumExpr over || exprHasSumExpr body
  | .zipWith a b _ _ body =>
    exprHasSumExpr a || exprHasSumExpr b || exprHasSumExpr body
  | .letIn binders body =>
    binders.any (fun b => exprHasSumExpr b.value) || exprHasSumExpr body

private def declHasSumExpr : BodyDecl → Bool
  | .inst _ _ _ inputs => inputs.any (fun i => exprHasSumExpr i.value)
  | .param .. | .prog .. => false

private def progHasAnySumWork (_arena : Arena) (prog : Program) : Bool :=
  prog.decls.any declHasSumExpr
  || prog.assigns.any (fun a => exprHasSumExpr a.expr)

-- ─────────────────────────────────────────────────────────────
-- Public entry
-- ─────────────────────────────────────────────────────────────

def run (arena : Arena) (rootIdx : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"sumLower: program pool index {rootIdx.idx} out of range"⟩
  if !progHasAnySumWork arena prog then
    return (arena, rootIdx)
  let ctx : Ctx := { arena }
  let newDecls ← buildNewDecls ctx prog.decls
  let newAssigns ← prog.assigns.mapM fun a => do
    pure { a with expr := ← rewriteExpr ctx a.expr }
  let fresh : Program := { prog with decls := newDecls, assigns := newAssigns }
  return ({ arena with programs := arena.programs.push fresh },
          ⟨arena.programs.size⟩)

-- ─────────────────────────────────────────────────────────────
-- Id-form (#190 native-DAG) — mirrors the walker on EArena
-- ─────────────────────────────────────────────────────────────

private structure CtxE where
  bindings : List (BinderIdx × ExprId) := []

private def CtxE.withBindings (ctx : CtxE) (extra : List (BinderIdx × ExprId)) : CtxE :=
  if extra.isEmpty then ctx else { ctx with bindings := extra ++ ctx.bindings }

private def sumDefE (d : TypeDefIdx) : PassM (String × Array SumVariant) := do
  match ← typeDefP? d with
  | some (.sum name variants) => pure (name, variants)
  | _ => failP s!"sumLower: typeDef pool index {d.idx} is not a sum type (internal)"

private def nat0E (n : Nat) : PassM ExprId := einternP (.num ⟨Int.ofNat n, 0⟩)

mutual

private partial def rewriteExprE (ctx : CtxE) (id : ExprId) : PassM ExprId := do
  match ← derefP id with
  | .num _ | .bool _ => pure id
  | .arr items => einternP (.arr (← items.mapM (rewriteExprE ctx)))
  | .bindingRef i =>
    match ctx.bindings.find? (·.1 == i) with
    | some (_, sub) => pure sub
    | none => pure id
  | .tag d variant payload =>
    let (_, variants) ← sumDefE d
    let some v := variants[variant]?
      | failP s!"sumLower: tag variant index {variant} out of range (internal)"
    if payload.isEmpty then
      nat0E variant
    else
      failP s!"sumLower: bare tag with payload (variant '{v.name}') in non-update context"
  | .match_ d scrutinee arms => lowerMatchToSelectChainE ctx d scrutinee arms
  | .inputRef _ | .paramRef _ | .typeParamRef _
  | .sampleRate | .sampleIndex | .nestedOut _ _ => pure id
  | .binary tag a b => einternP (.binary tag (← rewriteExprE ctx a) (← rewriteExprE ctx b))
  | .unary tag a => einternP (.unary tag (← rewriteExprE ctx a))
  | .clamp a b c => einternP (.clamp (← rewriteExprE ctx a) (← rewriteExprE ctx b) (← rewriteExprE ctx c))
  | .select a b c => einternP (.select (← rewriteExprE ctx a) (← rewriteExprE ctx b) (← rewriteExprE ctx c))
  | .index a b => einternP (.index (← rewriteExprE ctx a) (← rewriteExprE ctx b))
  | .arraySet a b c => einternP (.arraySet (← rewriteExprE ctx a) (← rewriteExprE ctx b) (← rewriteExprE ctx c))
  | .zeros count => einternP (.zeros (← rewriteExprE ctx count))
  | .fold over init acc elem body =>
    einternP (.fold (← rewriteExprE ctx over) (← rewriteExprE ctx init) acc elem (← rewriteExprE ctx body))
  | .scan over init acc elem body =>
    einternP (.scan (← rewriteExprE ctx over) (← rewriteExprE ctx init) acc elem (← rewriteExprE ctx body))
  | .generate count iter body =>
    einternP (.generate (← rewriteExprE ctx count) iter (← rewriteExprE ctx body))
  | .iterate count init iter body =>
    einternP (.iterate (← rewriteExprE ctx count) (← rewriteExprE ctx init) iter (← rewriteExprE ctx body))
  | .chain count init iter body =>
    einternP (.chain (← rewriteExprE ctx count) (← rewriteExprE ctx init) iter (← rewriteExprE ctx body))
  | .map2 over elem body =>
    einternP (.map2 (← rewriteExprE ctx over) elem (← rewriteExprE ctx body))
  | .zipWith a b x y body =>
    einternP (.zipWith (← rewriteExprE ctx a) (← rewriteExprE ctx b) x y (← rewriteExprE ctx body))
  | .letIn binders body =>
    let bs ← binders.mapM fun b => do
      pure ({ binder := b.binder, value := ← rewriteExprE ctx b.value } : ELetBinder)
    einternP (.letIn bs (← rewriteExprE ctx body))

private partial def lowerMatchToSelectChainE (ctx : CtxE) (d : TypeDefIdx)
    (scrutinee : ExprId) (arms : Array EMatchArm) : PassM ExprId := do
  let tagRead ← rewriteExprE ctx scrutinee
  let (typeName, variants) ← sumDefE d
  let armFor (vi : Nat) (v : SumVariant) : PassM EMatchArm :=
    match arms.find? (·.variant == vi) with
    | some arm => pure arm
    | none => failP s!"sumLower: match on '{typeName}' missing arm for '{v.name}'"
  let lowerArmBody (arm : EMatchArm) : PassM ExprId := do
    if arm.binders.isEmpty then
      rewriteExprE ctx arm.body
    else
      let subs ← bindingsForArmE ctx scrutinee d arm
      rewriteExprE (ctx.withBindings subs) arm.body
  if variants.isEmpty then
    failP s!"sumLower: match on '{typeName}' has no variants (internal)"
  let lastIdx := variants.size - 1
  let mut chain : ExprId ← lowerArmBody (← armFor lastIdx variants[lastIdx]!)
  for k in [1:variants.size] do
    let i := lastIdx - k
    let arm ← armFor i variants[i]!
    let body ← lowerArmBody arm
    let cond ← einternP (.binary .eq tagRead (← nat0E i))
    chain ← einternP (.select cond body chain)
  return chain

private partial def bindingsForArmE (ctx : CtxE) (scrutinee : ExprId)
    (matchDef : TypeDefIdx) (arm : EMatchArm) :
    PassM (List (BinderIdx × ExprId)) := do
  let _ := ctx
  let _ := scrutinee
  if arm.binders.isEmpty then return []
  let (_, matchVariants) ← sumDefE matchDef
  let some armVariant := matchVariants[arm.variant]?
    | failP s!"sumLower: match arm variant index {arm.variant} out of range (internal)"
  failP (s!"sumLower: match arm '{armVariant.name}' has payload bindings, but " ++
    "CF-only removed sum-typed registers — there is no slot source to bind from.")

end

private partial def exprHasSumE (id : ExprId) : PassM Bool := do
  match ← derefP id with
  | .tag .. | .match_ .. => pure true
  | .num _ | .bool _
  | .inputRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
  | .sampleRate | .sampleIndex | .nestedOut _ _ => pure false
  | .arr items => items.anyM exprHasSumE
  | .binary _ a b | .index a b => pure ((← exprHasSumE a) || (← exprHasSumE b))
  | .unary _ a => exprHasSumE a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    pure ((← exprHasSumE a) || (← exprHasSumE b) || (← exprHasSumE c))
  | .zeros count => exprHasSumE count
  | .fold over init _ _ body | .scan over init _ _ body =>
    pure ((← exprHasSumE over) || (← exprHasSumE init) || (← exprHasSumE body))
  | .iterate count init _ body | .chain count init _ body =>
    pure ((← exprHasSumE count) || (← exprHasSumE init) || (← exprHasSumE body))
  | .generate count _ body => pure ((← exprHasSumE count) || (← exprHasSumE body))
  | .map2 over _ body => pure ((← exprHasSumE over) || (← exprHasSumE body))
  | .zipWith a b _ _ body =>
    pure ((← exprHasSumE a) || (← exprHasSumE b) || (← exprHasSumE body))
  | .letIn binders body =>
    pure ((← binders.anyM (fun b => exprHasSumE b.value)) || (← exprHasSumE body))

private def progHasAnySumWorkE (prog : EProgram) : PassM Bool := do
  for d in prog.decls do
    if let .inst _ _ _ inputs := d then
      if ← inputs.anyM (fun i => exprHasSumE i.value) then return true
  for a in prog.assigns do
    if ← exprHasSumE a.expr then return true
  return false

def runE (rootIdx : ProgramIdx) : PassM ProgramIdx := do
  let prog ← getEProgram rootIdx "sumLower"
  if !(← progHasAnySumWorkE prog) then
    return rootIdx
  let ctx : CtxE := {}
  let newDecls ← prog.decls.mapM fun decl => do
    match decl with
    | .inst name typeKey tArgs inputs =>
      pure (EBodyDecl.inst name typeKey tArgs
        (← inputs.mapM fun i => do pure ({ port := i.port, value := ← rewriteExprE ctx i.value } : EInstanceInput)))
    | .param .. | .prog .. => pure decl
  let newAssigns ← prog.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← rewriteExprE ctx a.expr } : EOutputAssign)
  pushEProgram { prog with decls := newDecls, assigns := newAssigns }

end Tropical.Ir.Strata.SumLower
