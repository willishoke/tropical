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

/-- Loop-everything escape hatch (banks-as-data, the trunk move):
    `TROPICAL_BANKS_UNROLL` reverts summing folds to the unrolled realization —
    the same env var the modal-bank dispatch honors; one knob restores the
    whole naive ladder for bisection. -/
initialize banksLoopEnabled : Bool ← do
  return (← IO.getEnv "TROPICAL_BANKS_UNROLL").isNone

/-- The immediate children of a node — the generic walk the bank-eligibility
    predicates use. -/
private def childrenE : ENode → Array ExprId
  | .num _ | .bool _ | .inputRef _ | .paramRef _ | .typeParamRef _
  | .nestedOut _ _ | .sampleRate | .sampleIndex | .loopIdx _ | .bindingRef _ => #[]
  | .arr items => items
  | .zeros c => #[c]
  | .letIn binders body => (binders.map (·.value)).push body
  | .fold over init _ _ body => #[over, init, body]
  | .scan over init _ _ body => #[over, init, body]
  | .generate count _ body => #[count, body]
  | .iterate count init _ body => #[count, init, body]
  | .chain count init _ body => #[count, init, body]
  | .map2 over _ body => #[over, body]
  | .zipWith a b _ _ body => #[a, b, body]
  | .binary _ a b => #[a, b]
  | .index a b => #[a, b]
  | .unary _ a => #[a]
  | .clamp a b c | .select a b c | .arraySet a b c => #[a, b, c]
  | .bankSum _ ts b dc _ => (ts.push b) ++ dc.toArray
  | .tag _ _ payload => payload.map (·.value)
  | .match_ _ s arms => #[s] ++ arms.map (·.body)

/-- Does any node satisfying `p` occur at or below `root`? Iterative DAG walk
    with a visited set (subgraphs share; without it this is O(expanded tree)). -/
private partial def anyNodeE (p : ENode → Bool) (root : ExprId) : PassM Bool := do
  let mut stack : Array ExprId := #[root]
  let mut seen : Std.HashSet Nat := {}
  while !stack.isEmpty do
    let id := stack.back!
    stack := stack.pop
    if seen.contains id.idx then continue
    seen := seen.insert id.idx
    let n ← derefP id
    if p n then return true
    stack := stack ++ childrenE n
  return false

/-- Column layout for a bankable fold, decided by the ELEMENT SHAPE:
    - all scalar → one column, the element binder substitutes to
      `index(col, loopIdx)` (today's path);
    - all `.arr` of ONE arity n ≥ 1 with scalar sub-elements → n per-position
      columns (columnize-over-shapes: the AoS→SoA iso
      `Array (A×B) ≅ Array A × Array B`, done generically). The element binder
      substitutes to a SYMBOLIC tuple `[index(col_0, loopIdx), …]` whose
      literal projections the static-index fold resolves; `residual?` carries
      that tuple node so the caller can verify it did not survive lowering
      (`.index` is the only projection — a survivor means a non-literal index
      or the element escaping whole, and the fold must unroll: a materialized
      array-of-arrays would silently zero-fill in emit's `compilePack`);
    - anything else (tags, ragged arities, nested tuples, mixed shapes) →
      `none`, and the fold unrolls exactly as before.
    `idxId` is the binder id of the region being built (nested banks): the
    projections read `loopIdx idxId`, so an inner bank constructed later inside
    the body leaves them untouched. -/
private def bankColumnsE (idxId : Nat) (items : Array ExprId) :
    PassM (Option (Array ExprId × ExprId × Option ENode)) := do
  let derefed ← items.mapM derefP
  let isScalarShape : ENode → Bool := fun n => match n with
    | .arr _ | .tag .. => false
    | _ => true
  let loopIdxE ← einternP (.loopIdx idxId)
  if derefed.all isScalarShape then
    let col ← einternP (.arr items)
    let elemSym ← einternP (.index col loopIdxE)
    return some (#[col], elemSym, none)
  let .arr subs0 := derefed[0]! | return none
  let n := subs0.size
  if n == 0 then return none
  let mut tuples : Array (Array ExprId) := #[]
  for d in derefed do
    let .arr subs := d | return none
    if subs.size != n then return none
    for s in subs do
      match ← derefP s with
      | .arr _ | .tag .. | .zeros _ => return none
      | _ => pure ()
    tuples := tuples.push subs
  let mut cols : Array ExprId := #[]
  let mut projs : Array ExprId := #[]
  for j in [0:n] do
    let col ← einternP (.arr (tuples.map (·[j]!)))
    cols := cols.push col
    projs := projs.push (← einternP (.index col loopIdxE))
  let tupleNode : ENode := .arr projs
  let elemSym ← einternP tupleNode
  return some (cols, elemSym, some tupleNode)

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
  /-- Binder ids of the banks whose bodies are CURRENTLY being lowered (the
      open nesting chain, innermost first). A fold banking inside one of these
      bodies will nest inside them, so its own id must avoid all of them. -/
  openBankIds : List Nat := []

private abbrev LowerM := StateT LowerSt PassM

partial def exprNeedsLoweringE (id : ExprId) : LowerM Bool := do
  if let some r := (← get).needs.get? id.idx then return r
  let r ← match ← derefP id with
    | .letIn .. | .fold .. | .scan .. | .generate ..
    | .iterate .. | .chain .. | .map2 .. | .zipWith .. => pure true
    | .bindingRef _ => pure true
    | .zeros _ => pure true
    | .num _ | .bool _
    | .inputRef _ | .paramRef _ | .typeParamRef _
    | .nestedOut _ _ | .sampleRate | .sampleIndex | .loopIdx _ => pure false
    -- A bankSum is NOT unrolled — it survives to post-strata as an indexed
    -- reduction. Its tables/body are ordinary subgraphs, so descend only if one
    -- genuinely contains a combinator/bindingRef/zeros to lower.
    | .bankSum _ ts b dc _ =>
      pure ((← ts.anyM exprNeedsLoweringE) || (← exprNeedsLoweringE b)
        || (← dc.mapM exprNeedsLoweringE).getD false)
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
    match ← tryBankFoldE subst items init acc elem body with
    | some banked => pure banked
    | none =>
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
  | .nestedOut _ _ | .sampleRate | .sampleIndex | .loopIdx _ => pure id
  | .bankSum c ts b dc ii =>
    -- Preserve the region; lower its subgraphs (never unroll). `subst` threads
    -- through in case the bank is nested in an enclosing combinator being unrolled.
    einternP (.bankSum c (← ts.mapM (lowerExprE subst)) (← lowerExprE subst b)
      (← dc.mapM (lowerExprE subst)) ii)
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
  | .index a b =>
    let a' ← lowerExprE subst a
    let b' ← lowerExprE subst b
    -- Static-index fold: a literal, integral, in-bounds index into a literal
    -- array IS the element. Columnize-over-shapes leans on this to resolve the
    -- symbolic tuple's projections (`index(elem, j)` → `index(col_j, loopIdx)`);
    -- out-of-bounds / non-integral indices keep today's runtime `Index`
    -- semantics untouched. Fires only on nodes being rewritten anyway (the
    -- needs-lowering fast path returns untouched subgraphs by id).
    match ← derefP a', ← derefP b' with
    | .arr items, .num n =>
      let f := n.toFloat
      if f == f.floor && f >= 0 && f.toUInt64.toNat < items.size then
        pure items[f.toUInt64.toNat]!
      else
        einternP (.index a' b')
    | _, _ => einternP (.index a' b')

/-- Collect the binder ids of every `bankSum` reachable from `root`, following
    `bindingRef`s through `subst` (the values a later lowering of `root` under
    `subst` can pull in). Used to pick a fresh binder id for a new bank: a
    pre-existing bank reachable this way can end up NESTED inside the new
    region (e.g. a `letIn`-bound banked fold referenced by the body — the
    Cauchy-with-shared-subsum shape), so the new id must avoid all of them.
    Over-approximate is fine (an avoided id that never nests costs nothing);
    banks reachable only through TABLES are included too, harmlessly — tables
    materialize before their region, so they never join the nesting chain. -/
private partial def collectBankIdsE (subst : SubstMapE) (root : ExprId) :
    PassM (Std.HashSet Nat) := do
  let mut stack : Array ExprId := #[root]
  let mut seen : Std.HashSet Nat := {}
  let mut out : Std.HashSet Nat := {}
  while !stack.isEmpty do
    let id := stack.back!
    stack := stack.pop
    if seen.contains id.idx then continue
    seen := seen.insert id.idx
    let n ← derefP id
    match n with
    | .bankSum _ _ _ _ ii => out := out.insert ii
    | .bindingRef i =>
      if let some (_, v) := subst.find? (·.1 == i) then
        stack := stack.push v
    | _ => pure ()
    stack := stack ++ childrenE n
  return out

/-- LOOP-EVERYTHING (banks-as-data, the trunk move): a SUMMING fold survives as
    an indexed reduction (`bankSum`) instead of unrolling — the realization
    below post-strata owns the loop; nothing per-effect, no size threshold.
    Eligibility (anything else unrolls exactly as before):
    - body ≡ `acc + rhs` with the accumulator free ONLY at that head. The loop
      then visits elements in exactly the order the unroll nests its adds, so
      the render is bit-identical for ANY element type (order preservation, not
      associativity — the typed-accumulator emit carries floats). A Horner fold
      (`acc·x + c`) is not a Σ and keeps unrolling, as it should.
    - init ≡ literal 0 (the reduce seed `ReduceBegin` provides).
    - element SHAPE decides the columns (`bankColumnsE`): scalar elements →
      one column; uniform-arity tuple elements → per-position columns
      (columnize-over-shapes); tags / ragged / nested / mixed → unroll.
    The element binder substitutes to `index(col, loopIdx idxId)` (or the
    symbolic tuple of such projections) — the same substitution machinery the
    unroll uses, pointed at a symbolic element.

    NESTED banks (WS5): a summing fold inside the body banks TOO, as a nested
    region with its own binder id. Lowering is inside-out — by the time this
    fold's `rhs'` exists, any inner summing fold has already banked with an id
    chosen while this fold's id was on `openBankIds` — so no rewriting of inner
    bodies ever happens; ids are stable under nesting. Freshness needs only
    chain-uniqueness: the id chosen here is the smallest Nat avoiding (a) every
    enclosing bank currently being lowered (`openBankIds` — this fold nests
    inside them) and (b) every bank already reachable from the body under
    `subst` (`collectBankIdsE` — those can nest inside THIS fold). Plain
    non-nested folds therefore still get id 0, keeping their wire form (and
    hash-consing of identical folds) byte-identical to the pre-nesting world. -/
private partial def tryBankFoldE (subst : SubstMapE) (items : Array ExprId)
    (init : ExprId) (acc elem : Binder) (body : ExprId) : LowerM (Option ExprId) := do
  unless banksLoopEnabled do return none
  if items.isEmpty then return none
  match ← derefP (← lowerExprE subst init) with
  | .num n => unless n.mantissa == 0 do return none
  | _ => return none
  let .binary .add accRef rhs ← derefP body | return none
  let .bindingRef i ← derefP accRef | return none
  unless i == acc.idx do return none
  if ← anyNodeE (· == .bindingRef acc.idx) rhs then return none
  -- Fresh binder id for this region: smallest Nat unused along the chain.
  let used ← (collectBankIdsE subst rhs : PassM _)
  let openIds := (← get).openBankIds
  let mut idxId := 0
  while used.contains idxId || openIds.contains idxId do
    idxId := idxId + 1
  let some (cols, elemSym, residual?) ← (bankColumnsE idxId items : PassM _) | return none
  modify fun s => { s with openBankIds := idxId :: s.openBankIds }
  let rhs' ← lowerExprE ((elem.idx, elemSym) :: subst) rhs
  modify fun s => { s with openBankIds := s.openBankIds.tail }
  -- The banked body must be SCALAR. If the symbolic tuple survived lowering
  -- (a non-literal index into the element, or the element escaping whole),
  -- bail to unroll — a materialized array-of-arrays would hit compilePack's
  -- silent zero-fill.
  if let some node := residual? then
    if ← anyNodeE (· == node) rhs' then return none
  some <$> einternP (.bankSum items.size cols rhs' none idxId)

end

mutual

private partial def progNeedsLoweringE (prog : Program) : LowerM Bool := do
  for i in prog.inputs do
    if let some d := i.default? then
      if ← exprNeedsLoweringE d then return true
  for decl in prog.decls do
    if ← declNeedsLoweringE prog decl then return true
  for a in prog.assigns do
    if ← exprNeedsLoweringE a.expr then return true
  return false

private partial def declNeedsLoweringE (enclosing : Program) : BodyDecl → LowerM Bool
  | .param .. | .prog .. => pure false
  | .inst name typeKey _ inputs => do
    let (_, subType) ← getInstanceTypeE enclosing name typeKey
    if ← inputs.anyM (fun i => exprNeedsLoweringE i.value) then return true
    progNeedsLoweringE subType

end

private def lowerDeclE (decl : BodyDecl) : LowerM BodyDecl := do
  match decl with
  | .param .. | .prog .. => pure decl
  | .inst name typeKey tArgs inputs =>
    pure (.inst name typeKey tArgs
      (← inputs.mapM fun i => do pure ({ port := i.port, value := ← lowerExprE [] i.value } : InstanceInput)))

private partial def runGoE (rootIdx : ProgramIdx) : LowerM ProgramIdx := do
  let prog ← getEProgram rootIdx "arrayLower"
  unless ← progNeedsLoweringE prog do
    return rootIdx
  let mut newRegistry : Array (String × ProgramIdx) := #[]
  for (key, subIdx) in prog.registry do
    let subIdx' ← runGoE subIdx
    newRegistry := newRegistry.push (key, subIdx')
  let newInputs ← prog.inputs.mapM fun i => do
    pure ({ name := i.name, type? := i.type?, default? := ← i.default?.mapM (lowerExprE []) } : InputDecl)
  let newDecls ← prog.decls.mapM lowerDeclE
  let newAssigns ← prog.assigns.mapM fun a => do
    pure ({ target := a.target, expr := ← lowerExprE [] a.expr } : OutputAssign)
  pushEProgram { prog with
    inputs := newInputs, decls := newDecls, assigns := newAssigns, registry := newRegistry }

/-- Run the pass with fresh memos (one per pass application; the registry
    recursion in `runGoE` shares them). -/
def runE (rootIdx : ProgramIdx) : PassM ProgramIdx :=
  (runGoE rootIdx).run' {}

end Tropical.Ir.Strata.ArrayLower
