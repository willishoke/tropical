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

/-- Pass-wide memo of the has-sum predicate per source id (the arena is
    append-only, so a node never changes under an id). Keeps the walks
    DAG-shaped — without it they cost O(expanded tree), which is super-linear
    on composed patches. -/
private abbrev SumM := StateT (Std.HashMap Nat Bool) PassM

private partial def exprHasSumE (id : ExprId) : SumM Bool := do
  if let some r := (← get).get? id.idx then return r
  let r ← match ← derefP id with
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
  modify (·.insert id.idx r)
  return r

mutual

private partial def rewriteExprE (ctx : CtxE) (id : ExprId) : SumM ExprId := do
  -- No tag/match anywhere below and no pending arm bindings: the rebuild
  -- would re-intern to the same id — skip it.
  if ctx.bindings.isEmpty then
    unless ← exprHasSumE id do return id
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
    (scrutinee : ExprId) (arms : Array EMatchArm) : SumM ExprId := do
  let tagRead ← rewriteExprE ctx scrutinee
  let (typeName, variants) ← sumDefE d
  let armFor (vi : Nat) (v : SumVariant) : SumM EMatchArm :=
    match arms.find? (·.variant == vi) with
    | some arm => pure arm
    | none => failP s!"sumLower: match on '{typeName}' missing arm for '{v.name}'"
  let lowerArmBody (arm : EMatchArm) : SumM ExprId := do
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
    SumM (List (BinderIdx × ExprId)) := do
  let _ := ctx
  let _ := scrutinee
  if arm.binders.isEmpty then return []
  let (_, matchVariants) ← sumDefE matchDef
  let some armVariant := matchVariants[arm.variant]?
    | failP s!"sumLower: match arm variant index {arm.variant} out of range (internal)"
  failP (s!"sumLower: match arm '{armVariant.name}' has payload bindings, but " ++
    "CF-only removed sum-typed registers — there is no slot source to bind from.")

end

private def progHasAnySumWorkE (prog : EProgram) : SumM Bool := do
  for d in prog.decls do
    if let .inst _ _ _ inputs := d then
      if ← inputs.anyM (fun i => exprHasSumE i.value) then return true
  for a in prog.assigns do
    if ← exprHasSumE a.expr then return true
  return false

private def runGoE (rootIdx : ProgramIdx) : SumM ProgramIdx := do
  let prog ← getEProgram rootIdx "sumLower"
  if !(← progHasAnySumWorkE prog) then
    return rootIdx
  let ctx : CtxE := {}
  let newDecls ← prog.decls.mapM fun decl => do
    match decl with
    | .inst name typeKey tArgs inputs =>
      pure (BodyDecl.inst name typeKey tArgs
        (← inputs.mapM fun i => do pure ({ port := i.port, value := ← rewriteExprE ctx i.value } : EInstanceInput)))
    | .param .. | .prog .. => pure decl
  let newAssigns ← prog.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← rewriteExprE ctx a.expr } : EOutputAssign)
  pushEProgram { prog with decls := newDecls, assigns := newAssigns }

/-- Run the pass with a fresh has-sum memo (one per pass application). -/
def runE (rootIdx : ProgramIdx) : PassM ProgramIdx :=
  (runGoE rootIdx).run' {}

end Tropical.Ir.Strata.SumLower
