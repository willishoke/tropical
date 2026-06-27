import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic

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

end Tropical.Ir.Strata.SumLower
