import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena

/-!
# arrayLower — port of compiler/ir/array_lower.ts (Phase 5 pass 4)

Combinator unrolling and array-op lowering. After this pass no
`let`/`fold`/`scan`/`generate`/`iterate`/`chain`/`map2`/`zipWith`
survives, every `bindingRef` is substituted away, and literal-count
`zeros` materializes to inline `[0, …, 0]`. `index`/`arraySet` and
inline arrays survive as scalar ops over slot bundles.

Substitution is BinderIdx-keyed (unique-per-program integers), so
shadowing is structurally impossible. Phase B threads the DAG straight
to emit with no tree encoding downstream, so the walk must be
DAG-shaped: `LowerM` carries a pass-wide memo of the needs-lowering
predicate (a subgraph with no combinator/bindingRef/zeros anywhere
below is returned by id, unchanged — id-identical because `eintern`
hash-conses) plus a rewrite memo for closed (empty-subst) roots.
Without it the walk is O(expanded tree) — the residue-composition
compile wall.

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

/-- JS `for (let i = 0; i < n; i++)` iteration count. -/
private def loopCount (n : JsonNumber) : Nat :=
  let f := n.toFloat
  if f <= 0 then 0 else f.ceil.toUInt64.toNat

-- ─────────────────────────────────────────────────────────────
-- Id-form (#190 native-DAG) — mirrors the unroller on EArena
-- ─────────────────────────────────────────────────────────────

private abbrev SubstMapE := List (BinderIdx × ExprId)

/-- Pass-wide memos. `needs` caches the needs-lowering predicate per source id
    (valid for the whole pass: the arena is append-only, so a node never
    changes under an id). `rw` caches rewrites of closed roots — sound only
    for empty substs, where the rewrite is a pure function of the subgraph. -/
private structure LowerSt where
  needs : Std.HashMap Nat Bool := {}
  rw : Std.HashMap Nat ExprId := {}

private abbrev LowerM := StateT LowerSt PassM

private def nat0E (n : Nat) : PassM ExprId := einternP (.num ⟨Int.ofNat n, 0⟩)

partial def exprNeedsLoweringE (id : ExprId) : LowerM Bool := do
  if let some r := (← get).needs.get? id.idx then return r
  let r ← match ← derefP id with
    | .letIn .. | .fold .. | .scan .. | .generate ..
    | .iterate .. | .chain .. | .map2 .. | .zipWith .. => pure true
    | .bindingRef _ => pure true
    | .zeros _ => pure true
    | .num _ | .bool _
    | .inputRef _ | .paramRef _ | .typeParamRef _
    | .nestedOut _ _ | .sampleRate | .sampleIndex => pure false
    | .arr items => items.anyM exprNeedsLoweringE
    | .tag _ _ payload => payload.anyM (fun p => exprNeedsLoweringE p.value)
    | .match_ _ scrutinee arms =>
      pure ((← exprNeedsLoweringE scrutinee) || (← arms.anyM (fun a => exprNeedsLoweringE a.body)))
    | .binary _ a b | .index a b => pure ((← exprNeedsLoweringE a) || (← exprNeedsLoweringE b))
    | .unary _ a => exprNeedsLoweringE a
    | .clamp a b c | .select a b c | .arraySet a b c =>
      pure ((← exprNeedsLoweringE a) || (← exprNeedsLoweringE b) || (← exprNeedsLoweringE c))
  modify fun s => { s with needs := s.needs.insert id.idx r }
  return r

mutual

partial def lowerExprE (subst : SubstMapE) (id : ExprId) : LowerM ExprId := do
  -- Nothing to lower anywhere below (in particular no bindingRef, so `subst`
  -- is irrelevant): the rebuild would re-intern to the same id — skip it.
  unless ← exprNeedsLoweringE id do return id
  if subst.isEmpty then
    if let some r := (← get).rw.get? id.idx then return r
  let r ← lowerExprCoreE subst id
  if subst.isEmpty then
    modify fun s => { s with rw := s.rw.insert id.idx r }
  return r

private partial def lowerExprCoreE (subst : SubstMapE) (id : ExprId) : LowerM ExprId := do
  match ← derefP id with
  | .num _ | .bool _ => pure id
  | .arr items => einternP (.arr (← items.mapM (lowerExprE subst)))
  | .bindingRef i =>
    match subst.find? (·.1 == i) with
    | some (_, v) => pure v
    | none => pure id
  | .letIn binders body =>
    let mut cur := subst
    for b in binders do
      let value ← lowerExprE cur b.value
      cur := (b.binder.idx, value) :: cur
    lowerExprE cur body
  | .fold over init acc elem body =>
    let .arr items ← derefP (← lowerExprE subst over)
      | failP "arrayLower: fold's 'over' did not lower to a static array"
    let mut accV ← lowerExprE subst init
    for item in items do
      accV ← lowerExprE ((acc.idx, accV) :: (elem.idx, item) :: subst) body
    pure accV
  | .scan over init acc elem body =>
    let .arr items ← derefP (← lowerExprE subst over)
      | failP "arrayLower: scan's 'over' did not lower to a static array"
    let mut out : Array ExprId := #[]
    let mut accV ← lowerExprE subst init
    for item in items do
      accV ← lowerExprE ((acc.idx, accV) :: (elem.idx, item) :: subst) body
      out := out.push accV
    einternP (.arr out)
  | .generate count iter body =>
    let .num n ← derefP (← lowerExprE subst count)
      | failP "arrayLower: generate's 'count' did not lower to a numeric literal"
    let mut out : Array ExprId := #[]
    for i in [0:loopCount n] do
      out := out.push (← lowerExprE ((iter.idx, ← nat0E i) :: subst) body)
    einternP (.arr out)
  | .iterate count init iter body =>
    let .num n ← derefP (← lowerExprE subst count)
      | failP "arrayLower: iterate's 'count' did not lower to a numeric literal"
    let mut out : Array ExprId := #[]
    let mut cur ← lowerExprE subst init
    for _ in [0:loopCount n] do
      out := out.push cur
      cur ← lowerExprE ((iter.idx, cur) :: subst) body
    einternP (.arr out)
  | .chain count init iter body =>
    let .num n ← derefP (← lowerExprE subst count)
      | failP "arrayLower: chain's 'count' did not lower to a numeric literal"
    let mut cur ← lowerExprE subst init
    for _ in [0:loopCount n] do
      cur ← lowerExprE ((iter.idx, cur) :: subst) body
    pure cur
  | .map2 over elem body =>
    let .arr items ← derefP (← lowerExprE subst over)
      | failP "arrayLower: map2's 'over' did not lower to a static array"
    einternP (.arr (← items.mapM fun item => lowerExprE ((elem.idx, item) :: subst) body))
  | .zipWith a b x y body =>
    let .arr aItems ← derefP (← lowerExprE subst a)
      | failP "arrayLower: zipWith's 'a' and 'b' did not both lower to static arrays"
    let .arr bItems ← derefP (← lowerExprE subst b)
      | failP "arrayLower: zipWith's 'a' and 'b' did not both lower to static arrays"
    let n := Nat.min aItems.size bItems.size
    let mut out : Array ExprId := #[]
    for i in [0:n] do
      out := out.push (← lowerExprE ((x.idx, aItems[i]!) :: (y.idx, bItems[i]!) :: subst) body)
    einternP (.arr out)
  | .zeros count =>
    let countL ← lowerExprE subst count
    match ← derefP countL with
    | .num n =>
      let f := n.toFloat
      if f == f.floor && f >= 0 then
        einternP (.arr (Array.replicate f.toUInt64.toNat (← nat0E 0)))
      else
        failP s!"arrayLower: zeros count {n} is not a valid array length"
    | _ => einternP (.zeros countL)
  | .inputRef _ | .paramRef _ | .typeParamRef _
  | .nestedOut _ _ | .sampleRate | .sampleIndex => pure id
  | .tag d v payload =>
    einternP (.tag d v
      (← payload.mapM fun p => do pure ({ field := p.field, value := ← lowerExprE subst p.value } : ETagPayload)))
  | .match_ d scrutinee arms =>
    einternP (.match_ d (← lowerExprE subst scrutinee)
      (← arms.mapM fun arm => do
        pure ({ variant := arm.variant, binders := arm.binders, body := ← lowerExprE subst arm.body } : EMatchArm)))
  | .binary tag a b => einternP (.binary tag (← lowerExprE subst a) (← lowerExprE subst b))
  | .unary tag a => einternP (.unary tag (← lowerExprE subst a))
  | .clamp a b c => einternP (.clamp (← lowerExprE subst a) (← lowerExprE subst b) (← lowerExprE subst c))
  | .select a b c => einternP (.select (← lowerExprE subst a) (← lowerExprE subst b) (← lowerExprE subst c))
  | .arraySet a b c => einternP (.arraySet (← lowerExprE subst a) (← lowerExprE subst b) (← lowerExprE subst c))
  | .index a b => einternP (.index (← lowerExprE subst a) (← lowerExprE subst b))

end

mutual

private partial def progNeedsLoweringE (prog : EProgram) : LowerM Bool := do
  for i in prog.inputs do
    if let some d := i.default? then
      if ← exprNeedsLoweringE d then return true
  for decl in prog.decls do
    if ← declNeedsLoweringE prog decl then return true
  for a in prog.assigns do
    if ← exprNeedsLoweringE a.expr then return true
  return false

private partial def declNeedsLoweringE (enclosing : EProgram) : EBodyDecl → LowerM Bool
  | .param .. | .prog .. => pure false
  | .inst name typeKey _ inputs => do
    let (_, subType) ← getInstanceTypeE enclosing name typeKey
    if ← inputs.anyM (fun i => exprNeedsLoweringE i.value) then return true
    progNeedsLoweringE subType

end

private def lowerDeclE (decl : EBodyDecl) : LowerM EBodyDecl := do
  match decl with
  | .param .. | .prog .. => pure decl
  | .inst name typeKey tArgs inputs =>
    pure (.inst name typeKey tArgs
      (← inputs.mapM fun i => do pure ({ port := i.port, value := ← lowerExprE [] i.value } : EInstanceInput)))

private partial def runGoE (rootIdx : ProgramIdx) : LowerM ProgramIdx := do
  let prog ← getEProgram rootIdx "arrayLower"
  unless ← progNeedsLoweringE prog do
    return rootIdx
  let mut newRegistry : Array (String × ProgramIdx) := #[]
  for (key, subIdx) in prog.registry do
    let subIdx' ← runGoE subIdx
    newRegistry := newRegistry.push (key, subIdx')
  let newInputs ← prog.inputs.mapM fun i => do
    pure ({ name := i.name, type? := i.type?, default? := ← i.default?.mapM (lowerExprE []) } : EInputDecl)
  let newDecls ← prog.decls.mapM lowerDeclE
  let newAssigns ← prog.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← lowerExprE [] a.expr } : EOutputAssign)
  pushEProgram { prog with
    inputs := newInputs, decls := newDecls, assigns := newAssigns, registry := newRegistry }

/-- Run the pass with fresh memos (one per pass application; the registry
    recursion in `runGoE` shares them). -/
def runE (rootIdx : ProgramIdx) : PassM ProgramIdx :=
  (runGoE rootIdx).run' {}

end Tropical.Ir.Strata.ArrayLower
