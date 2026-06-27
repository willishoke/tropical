import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic

/-!
# arrayLower — port of compiler/ir/array_lower.ts (Phase 5 pass 4)

Combinator unrolling and array-op lowering. After this pass no
`let`/`fold`/`scan`/`generate`/`iterate`/`chain`/`map2`/`zipWith`
survives, every `bindingRef` is substituted away, and literal-count
`zeros` materializes to inline `[0, …, 0]`. `index`/`arraySet` and
inline arrays survive as scalar ops over slot bundles.

Substitution is BinderIdx-keyed (unique-per-program integers), so
shadowing is structurally impossible. The TS per-iteration WeakMap
memos are sharing-only (invisible to the structural codec) and are
dropped; the unmemoized walk is O(output tree), the same bound as
encoding.

Fractal: surviving InstanceDecls keep their typeKey; registry entries
are lowered recursively PER ENTRY (no idx-level dedup — TS lowers each
map entry independently, so two keys aliasing one program stay aliased
only on the no-op fast path, and the pool reflects that).

JS loop semantics on counts are mirrored: `for (i = 0; i < n; i++)`
over a float n runs ⌈n⌉ times (0 for n ≤ 0).

Errors are comparable outputs (byte-exact TS messages).
-/

namespace Tropical.Ir.Strata.ArrayLower

open Lean (JsonNumber)
open Tropical.Ir

private def nat0 (n : Nat) : Expr := .num ⟨Int.ofNat n, 0⟩

/-- JS `for (let i = 0; i < n; i++)` iteration count. -/
private def loopCount (n : JsonNumber) : Nat :=
  let f := n.toFloat
  if f <= 0 then 0 else f.ceil.toUInt64.toNat

-- ─────────────────────────────────────────────────────────────
-- Detection — fast path when there's nothing to lower
-- ─────────────────────────────────────────────────────────────

private partial def exprNeedsLowering : Expr → Bool
  | .letIn .. | .fold .. | .scan .. | .generate ..
  | .iterate .. | .chain .. | .map2 .. | .zipWith .. => true
  -- A residual bindingRef indicates a combinator scope not yet
  -- entered; treat as "needs lowering" so the walk substitutes it.
  | .bindingRef _ => true
  | .zeros _ => true
  | .num _ | .bool _
  | .inputRef _ | .paramRef _ | .typeParamRef _
  | .nestedOut _ _ | .sampleRate | .sampleIndex => false
  | .arr items => items.any exprNeedsLowering
  | .tag _ _ payload => payload.any (fun p => exprNeedsLowering p.value)
  | .match_ _ scrutinee arms =>
    exprNeedsLowering scrutinee || arms.any (fun a => exprNeedsLowering a.body)
  | .binary _ a b | .index a b => exprNeedsLowering a || exprNeedsLowering b
  | .unary _ a => exprNeedsLowering a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    exprNeedsLowering a || exprNeedsLowering b || exprNeedsLowering c

mutual

private partial def progNeedsLowering (arena : Arena) (prog : Program) :
    Except Error Bool := do
  for i in prog.inputs do
    if i.default?.any exprNeedsLowering then return true
  for decl in prog.decls do
    if ← declNeedsLowering arena prog decl then return true
  for a in prog.assigns do
    if exprNeedsLowering a.expr then return true
  return false

private partial def declNeedsLowering (arena : Arena) (enclosing : Program) :
    BodyDecl → Except Error Bool
  | .param .. | .prog .. => return false
  | .inst name typeKey _ inputs => do
    -- Fractal: also check the sub-program for surviving combinators.
    let (_, subType) ← getInstanceType arena enclosing name typeKey
    if inputs.any (fun i => exprNeedsLowering i.value) then return true
    progNeedsLowering arena subType

end

-- ─────────────────────────────────────────────────────────────
-- Substitution map — keyed by BinderIdx, innermost-first lookup
-- ─────────────────────────────────────────────────────────────

private abbrev SubstMap := List (BinderIdx × Expr)

-- ─────────────────────────────────────────────────────────────
-- Expression lowering + combinator unrolling
-- ─────────────────────────────────────────────────────────────

mutual

private partial def lowerExpr (subst : SubstMap) (e : Expr) : Except Error Expr := do
  match e with
  | .num _ | .bool _ => return e
  | .arr items => return .arr (← items.mapM (lowerExpr subst))
  -- BindingRef: substitute when bound; residuals survive for the
  -- wrapping iteration to resolve.
  | .bindingRef i =>
    match subst.find? (·.1 == i) with
    | some (_, v) => return v
    | none => return e
  -- Combinators: unroll.
  | .letIn binders body =>
    let mut cur := subst
    for b in binders do
      let value ← lowerExpr cur b.value
      cur := (b.binder.idx, value) :: cur
    lowerExpr cur body
  | .fold over init acc elem body =>
    let .arr items ← lowerExpr subst over
      | throw ⟨"arrayLower: fold's 'over' did not lower to a static array"⟩
    let mut accV ← lowerExpr subst init
    for item in items do
      accV ← lowerExpr ((acc.idx, accV) :: (elem.idx, item) :: subst) body
    return accV
  | .scan over init acc elem body =>
    let .arr items ← lowerExpr subst over
      | throw ⟨"arrayLower: scan's 'over' did not lower to a static array"⟩
    let mut out : Array Expr := #[]
    let mut accV ← lowerExpr subst init
    for item in items do
      accV ← lowerExpr ((acc.idx, accV) :: (elem.idx, item) :: subst) body
      out := out.push accV
    return .arr out
  | .generate count iter body =>
    let .num n ← lowerExpr subst count
      | throw ⟨"arrayLower: generate's 'count' did not lower to a numeric literal"⟩
    let mut out : Array Expr := #[]
    for i in [0:loopCount n] do
      out := out.push (← lowerExpr ((iter.idx, nat0 i) :: subst) body)
    return .arr out
  | .iterate count init iter body =>
    let .num n ← lowerExpr subst count
      | throw ⟨"arrayLower: iterate's 'count' did not lower to a numeric literal"⟩
    let mut out : Array Expr := #[]
    let mut cur ← lowerExpr subst init
    for _ in [0:loopCount n] do
      out := out.push cur
      cur ← lowerExpr ((iter.idx, cur) :: subst) body
    return .arr out
  | .chain count init iter body =>
    let .num n ← lowerExpr subst count
      | throw ⟨"arrayLower: chain's 'count' did not lower to a numeric literal"⟩
    let mut cur ← lowerExpr subst init
    for _ in [0:loopCount n] do
      cur ← lowerExpr ((iter.idx, cur) :: subst) body
    return cur
  | .map2 over elem body =>
    let .arr items ← lowerExpr subst over
      | throw ⟨"arrayLower: map2's 'over' did not lower to a static array"⟩
    return .arr (← items.mapM fun item => lowerExpr ((elem.idx, item) :: subst) body)
  | .zipWith a b x y body =>
    let aL ← lowerExpr subst a
    let bL ← lowerExpr subst b
    let (.arr aItems, .arr bItems) := (aL, bL)
      | throw ⟨"arrayLower: zipWith's 'a' and 'b' did not both lower to static arrays"⟩
    let n := Nat.min aItems.size bItems.size
    let mut out : Array Expr := #[]
    for i in [0:n] do
      out := out.push
        (← lowerExpr ((x.idx, aItems[i]!) :: (y.idx, bItems[i]!) :: subst) body)
    return .arr out
  -- zeros{count}: materialize when count lowers to a literal.
  | .zeros count =>
    let countL ← lowerExpr subst count
    match countL with
    | .num n =>
      let f := n.toFloat
      if f == f.floor && f >= 0 then
        return .arr (Array.replicate f.toUInt64.toNat (nat0 0))
      else
        -- TS `new Array(count)` throws RangeError on an invalid
        -- length — a harness crash, unreachable on valid input.
        throw ⟨s!"arrayLower: zeros count {n} is not a valid array length"⟩
    | _ => return .zeros countL
  -- References / leaves.
  | .inputRef _ | .paramRef _ | .typeParamRef _
  | .nestedOut _ _ | .sampleRate | .sampleIndex => return e
  -- Tag / match: recurse structurally (match survivors are defensive —
  -- sum_lower runs first in the strata order).
  | .tag d v payload =>
    return .tag d v
      (← payload.mapM fun p => do pure (.mk p.field (← lowerExpr subst p.value)))
  | .match_ d scrutinee arms =>
    return .match_ d (← lowerExpr subst scrutinee)
      (← arms.mapM fun arm => do
        pure (.mk arm.variant arm.binders (← lowerExpr subst arm.body)))
  -- Operators.
  | .binary tag a b => return .binary tag (← lowerExpr subst a) (← lowerExpr subst b)
  | .unary tag a => return .unary tag (← lowerExpr subst a)
  | .clamp a b c =>
    return .clamp (← lowerExpr subst a) (← lowerExpr subst b) (← lowerExpr subst c)
  | .select a b c =>
    return .select (← lowerExpr subst a) (← lowerExpr subst b) (← lowerExpr subst c)
  | .arraySet a b c =>
    return .arraySet (← lowerExpr subst a) (← lowerExpr subst b) (← lowerExpr subst c)
  | .index a b => return .index (← lowerExpr subst a) (← lowerExpr subst b)

end

-- ─────────────────────────────────────────────────────────────
-- Decl / assign rewriting + public entry
-- ─────────────────────────────────────────────────────────────

private def lowerDecl (decl : BodyDecl) : Except Error BodyDecl := do
  match decl with
  | .param .. | .prog .. => return decl
  | .inst name typeKey tArgs inputs =>
    -- Fractal: typeKey stays; the lowered sub-program now lives
    -- behind that key in the new registry.
    return .inst name typeKey tArgs
      (← inputs.mapM fun i => do pure { i with value := ← lowerExpr [] i.value })

partial def run (arena : Arena) (rootIdx : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"arrayLower: program pool index {rootIdx.idx} out of range"⟩
  unless ← progNeedsLowering arena prog do
    return (arena, rootIdx)

  -- Recursively lower each registry entry (per entry, preserving
  -- TS's per-map-entry behavior on aliased keys).
  let mut arena := arena
  let mut newRegistry : Array (String × ProgramIdx) := #[]
  for (key, subIdx) in prog.registry do
    let (arena', subIdx') ← run arena subIdx
    arena := arena'
    newRegistry := newRegistry.push (key, subIdx')

  let newInputs ← prog.inputs.mapM fun i => do
    pure { i with
      default? := ← match i.default? with
        | some d => some <$> lowerExpr [] d
        | none => pure none }
  let newDecls ← prog.decls.mapM lowerDecl
  let newAssigns ← prog.assigns.mapM fun a => do
    pure { a with expr := ← lowerExpr [] a.expr }

  let fresh : Program := { prog with
    inputs := newInputs
    decls := newDecls
    assigns := newAssigns
    registry := newRegistry }
  return ({ arena with programs := arena.programs.push fresh },
          ⟨arena.programs.size⟩)

end Tropical.Ir.Strata.ArrayLower
