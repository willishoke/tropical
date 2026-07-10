import Std.Data.HashMap
import Tropical.Ir.ExprArena
import Tropical.Ir.Strata.Basic

/-!
# EArena — the strata pipeline's id-form working state (#190 native-DAG)

The tree passes thread an `Arena` (program pool of tree `Program`s) and rebuild
`Expr` trees; the bloat (the modulated clock duplicated into every oscillator
partial) is born in `inlineInstances`' substitution. This is the id-form
counterpart: an `EArena` carries a pool of `EProgram`s (id-valued) plus the
shared `ExprArena`, and every pass rewrites by interning into that one DAG. When
`inlineInstances` substitutes a wired expression at many use sites it stores the
same `ExprId` — the duplication never materializes.

`PassM = StateT EArena (Except Error)` is the shared pass monad: `einternP`
builds nodes, `derefP` reads them, `pushEProgram` appends a rewritten program,
and `typeParam?/typeDef?` read the (unchanged) identity pools off `base`.

Phase A keeps `Strata.run`'s signature: `ofArena` converts the input `Arena` in
(interning every program's expressions once, pre-bloat), the id-passes run, and
`materialize` converts the post-strata root back to a tree `Program`. The
re-materialization is temporary — Phase B threads the DAG straight to emit — but
it lets the id-passes land behind the bit-identical audio gate first.
-/

namespace Tropical.Ir.Strata

open Tropical.Ir

/-- The id-form pipeline state IS the `Arena` now: identity pools, an id-valued
    program pool, and the shared expression DAG (`exprs`) all in one. `base` is
    the arena itself (kept as an accessor so the pass helpers read unchanged). -/
abbrev EArena := Arena

/-- `EArena.base` — the identity pools live directly on the arena. -/
def _root_.Tropical.Ir.Arena.base (a : Arena) : Arena := a

/-- The shared pass monad. -/
abbrev PassM := StateT EArena (Except Error)

/-- Intern a node into the shared DAG, returning its (possibly shared) id. -/
def einternP (n : ENode) : PassM ExprId := do
  let ea ← get
  let (id, ex) := (eintern n).run ea.exprs
  set { ea with exprs := ex }
  pure id

/-- Dereference an id to its node (dangling id is an internal bug). -/
def derefP (id : ExprId) : PassM ENode := do
  match (← get).exprs.deref id with
  | some n => pure n
  | none => throw ⟨s!"EArena: dangling ExprId {id.idx} (internal)"⟩

/-- An `EProgram` by pool index, with a contextual error on miss. -/
def getEProgram (i : ProgramIdx) (ctx : String) : PassM EProgram := do
  match (← get).programs[i.idx]? with
  | some p => pure p
  | none => throw ⟨s!"{ctx}: program pool index {i.idx} out of range"⟩

/-- Append a rewritten program; returns its fresh pool index. -/
def pushEProgram (p : EProgram) : PassM ProgramIdx := do
  let ea ← get
  set { ea with programs := ea.programs.push p }
  pure ⟨ea.programs.size⟩

def typeParamP? (i : TypeParamPoolIdx) : PassM (Option TypeParamDecl) := do
  pure ((← get).base.typeParam? i)

def typeDefP? (i : TypeDefIdx) : PassM (Option TypeDef) := do
  pure ((← get).base.typeDef? i)

/-- Lift a pure `Except Error` (e.g. a shared validation helper) into `PassM`. -/
def liftE {α} (e : Except Error α) : PassM α :=
  match e with
  | .ok a => pure a
  | .error err => throw err

/-- Throw a strata `Error` in `PassM` (concrete return type for inference). -/
def failP {α} (msg : String) : PassM α := throw ⟨msg⟩

/-- Id-form `getInstanceType`: resolve an instance's target `EProgram` through
    the enclosing program's registry. -/
def getInstanceTypeE (enclosing : EProgram) (instName typeKey : String) :
    PassM (ProgramIdx × EProgram) := do
  match enclosing.registryGet? typeKey with
  | some pIdx =>
    let p ← getEProgram pIdx
      s!"getInstanceType: instance '{instName}' typeKey '{typeKey}' program pool index"
    pure (pIdx, p)
  | none =>
    let keys := ", ".intercalate (enclosing.registry.toList.map (·.1))
    throw ⟨s!"getInstanceType: instance '{instName}' typeKey '{typeKey}' " ++
      s!"not found in enclosing program '{enclosing.name}' registry " ++
      s!"(keys: {keys}). This is a registry-build bug; check buildProgramRegistry call sites."⟩

-- ─────────────────────────────────────────────────────────────
-- Entry/exit — both identities now
--
-- `Arena` IS the id-form (`Program` is id-valued, `Arena.exprs` is the shared
-- DAG), so the old tree↔id bridges collapse: `ofArena` is the identity, and
-- there is no tree to `materialize` back to — the strata passes push their
-- rewritten programs into the same arena, so the post-strata root is already a
-- valid `(Arena, ProgramIdx)`.
-- ─────────────────────────────────────────────────────────────

/-- Identity: an input `Arena` is already the id-form pipeline state. -/
def EArena.ofArena (a : Arena) : EArena := a

/-- Identity: the post-strata root is already an `(Arena, ProgramIdx)`. -/
def EArena.materialize (ea : EArena) (root : ProgramIdx) :
    Except Error (Arena × ProgramIdx) :=
  pure (ea, root)

-- ─────────────────────────────────────────────────────────────
-- Exit (Phase B): id-form EArena → (CoreArena × CoreProgram)
--
-- Replaces `materialize` + `Core.check` in one pass: the strata DAG is
-- threaded straight to emit as a `CoreArena` instead of flattened to a
-- tree and re-interned three times (the modulated-clock blowup). Each
-- reachable expression node is converted once and equal subtrees stay
-- one node, so O(unique nodes), never O(expanded tree).
--
-- REACHABLE-only: `ea.exprs` is append-only and still holds the
-- pre-lowering combinator nodes (`letIn`/`fold`/`tag`/…) that later
-- passes rewrote away, so a whole-arena fold would falsely reject them
-- as "survived <pass>". We convert only what the post-strata root (and
-- its instance-referenced registry) actually references — exactly the
-- evaluator-reachable graph the old `Core.check` walked.
-- ─────────────────────────────────────────────────────────────

open Tropical.Ir.Core (CoreProgram CoreInputDecl CoreOutputDecl CoreOutputAssign
  CoreBodyDecl CoreInstanceInput)

/-- Conversion state: the emit `CoreArena` under construction plus a memo
    mapping an `ExprArena` id (`.idx`) to its interned `CoreArena` id. -/
private abbrev ConvM := StateT (CoreArena × Std.HashMap Nat ExprId) (Except Error)

/-- Convert one reachable `ExprArena` node (and its children) into the
    `CoreArena`, memoized. Rejects any node a strata pass should have
    removed — the id-form analogue of `Core.checkExpr`, but visited only
    when actually referenced. -/
private partial def convExprId (ea : ExprArena) (eid : ExprId) : ConvM ExprId := do
  match (← get).2.get? eid.idx with
  | some cid => return cid
  | none =>
    let some n := ea.deref eid
      | throw ⟨s!"toResolved: dangling ExprId {eid.idx} (internal)"⟩
    let cn : CNode ← match n with
      | .num x          => pure (.num x)
      | .bool b         => pure (.bool b)
      | .arr items      => pure (.arr (← items.mapM (convExprId ea)))
      | .binary t a b   => pure (.binary t (← convExprId ea a) (← convExprId ea b))
      | .unary t a      => pure (.unary t (← convExprId ea a))
      | .clamp a b c    => pure (.clamp (← convExprId ea a) (← convExprId ea b) (← convExprId ea c))
      | .select a b c   => pure (.select (← convExprId ea a) (← convExprId ea b) (← convExprId ea c))
      | .arraySet a b c => pure (.arraySet (← convExprId ea a) (← convExprId ea b) (← convExprId ea c))
      | .index a b      => pure (.index (← convExprId ea a) (← convExprId ea b))
      | .inputRef i     => pure (.inputRef i)
      | .paramRef i     => pure (.paramRef i)
      | .nestedOut i o  => pure (.nestedOut i o)
      | .sampleRate     => pure .sampleRate
      | .sampleIndex    => pure .sampleIndex
      | .loopIdx        => pure .loopIdx
      | .bankSum c ts b dc => pure (.bankSum c (← ts.mapM (convExprId ea)) (← convExprId ea b) (← dc.mapM (convExprId ea)))
      | .zeros _        => throw ⟨"toResolved: zeros survived arrayLower"⟩
      | .typeParamRef _ => throw ⟨"toResolved: typeParamRef survived specialize"⟩
      | .bindingRef _   => throw ⟨"toResolved: bindingRef survived arrayLower"⟩
      | .letIn ..       => throw ⟨"toResolved: let survived arrayLower"⟩
      | .fold ..        => throw ⟨"toResolved: fold survived arrayLower"⟩
      | .scan ..        => throw ⟨"toResolved: scan survived arrayLower"⟩
      | .generate ..    => throw ⟨"toResolved: generate survived arrayLower"⟩
      | .iterate ..     => throw ⟨"toResolved: iterate survived arrayLower"⟩
      | .chain ..       => throw ⟨"toResolved: chain survived arrayLower"⟩
      | .map2 ..        => throw ⟨"toResolved: map2 survived arrayLower"⟩
      | .zipWith ..     => throw ⟨"toResolved: zipWith survived arrayLower"⟩
      | .tag ..         => throw ⟨"toResolved: tag survived sumLower"⟩
      | .match_ ..      => throw ⟨"toResolved: match survived sumLower"⟩
    let st ← get
    let (cid, ca') := (intern cn).run st.1
    set (ca', st.2.insert eid.idx cid)
    return cid

/-- Convert the reachable `EProgram` subgraph rooted at `eIdx` into a
    `CoreProgram`, remapping every leaf id into the `CoreArena` and
    following instance-referenced registry entries recursively (the
    id-form `Core.check`). Port types resolve against the identity pools
    (`base`). -/
private partial def convProgram (ea : EArena) (eIdx : ProgramIdx) : ConvM CoreProgram := do
  let some ep := ea.programs[eIdx.idx]?
    | throw ⟨s!"toResolved: program pool index {eIdx.idx} out of range (internal)"⟩
  unless ep.typeParams.isEmpty do
    throw ⟨s!"core check ('{ep.name}'): {ep.typeParams.size} typeParam decl(s) (specialize) survived strata"⟩
  let decls : Array CoreBodyDecl ← ep.decls.mapM fun d => do
    match d with
    | .param name value? => pure (.param name value?)
    | .inst name typeKey tArgs inputs =>
      let inputs' ← inputs.mapM fun i => do
        pure ({ port := i.port, value := ← convExprId ea.exprs i.value } : CoreInstanceInput)
      pure (.inst name typeKey tArgs inputs')
    | .prog name _ => pure (.progDecl name)
  let assigns : Array CoreOutputAssign ← ep.assigns.mapM fun a => do
    pure { target := a.target, expr := ← convExprId ea.exprs a.expr }
  let inputs : Array CoreInputDecl ← ep.inputs.mapM fun i => do
    pure { name := i.name, type? := Core.resolveOptPortType ea.base i.type?,
           default? := ← i.default?.mapM (convExprId ea.exprs) }
  let outputs : Array CoreOutputDecl := ep.outputs.map fun o =>
    { name := o.name, type? := Core.resolveOptPortType ea.base o.type? }
  -- Registry: follow only instance-referenced entries (evaluator-reachable),
  -- recursively — matching `Core.check`'s first-use dedup and tree duplication.
  let mut registry : Array (String × CoreProgram) := #[]
  for d in ep.decls do
    if let .inst name typeKey _ _ := d then
      unless registry.any (·.1 == typeKey) do
        let some tIdx := ep.registryGet? typeKey
          | throw ⟨s!"core check ('{ep.name}'): instance '{name}' typeKey '{typeKey}' missing from registry"⟩
        registry := registry.push (typeKey, ← convProgram ea tIdx)
  return .mk ep.name inputs outputs decls assigns registry

/-- The Phase B strata-exit reify: post-strata `EArena` → `(CoreArena ×
    CoreProgram)`, sharing preserved, reachable-only from `root`. -/
def EArena.toResolved (ea : EArena) (root : ProgramIdx) :
    Except Error (CoreArena × CoreProgram) := do
  let (core, (ca, _)) ← (convProgram ea root).run ({}, {})
  return (ca, core)

/-- Downcast an elaborated `Arena` (a session root / per-program compile
    boundary that never ran the strata passes, but IS the id-form) to
    `(CoreArena × CoreProgram)`. A thin `Except String` wrapper over
    `toResolved` for the compile call sites; reachable-only, so it validates
    only the evaluator-reachable graph and never touches the whole pool. -/
def _root_.Tropical.Ir.checkResolvedArena (a : Arena) (root : ProgramIdx) :
    Except String (Tropical.Ir.CoreArena × Tropical.Ir.Core.CoreProgram) :=
  (EArena.toResolved a root).mapError (·.message)

-- ─────────────────────────────────────────────────────────────
-- mapExprId — id-form structural walker (mirrors Recursion.mapExpr)
-- ─────────────────────────────────────────────────────────────

/-- Rewrite monad for one hook-set application: `PassM` plus a memo from
    source `ExprId.idx` to rewritten id. The arena is a DAG (equal subtrees
    share one node); the memo keeps the walk O(unique nodes) instead of
    O(expanded tree). Sound because hooks are functions of the node alone —
    there is no path context to invalidate a cached rewrite. -/
abbrev MapM := StateT (Std.HashMap Nat ExprId) PassM

/-- Hook set for `mapExprId`: `node n` may replace a node (returning its id,
    possibly freshly interned via `einternP`); `none` recurses structurally.
    `binder` transforms binders at every binding site. Hooks run in `MapM` so
    a hook that recurses (e.g. `nestedOut`-chain substitution) shares the
    walk's memo. -/
structure MapHooksId where
  node : ENode → MapM (Option ExprId) := fun _ => pure none
  binder : Binder → Binder := id

/-- Structural map over an id-rooted expression, re-interning the result and
    memoizing per source id. Equal subtrees collapse on intern, so the rewrite
    produces a DAG — and the memo makes the walk itself DAG-shaped too. -/
partial def mapExprIdGo (h : MapHooksId) (id : ExprId) : MapM ExprId := do
  if let some r := (← get).get? id.idx then return r
  let n ← derefP id
  let r ← do
    match ← h.node n with
    | some r => pure r
    | none =>
      match n with
      | .num _ | .bool _
      | .inputRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
      | .nestedOut _ _ | .sampleRate | .sampleIndex | .loopIdx => pure id
      | .bankSum c ts b dc =>
        einternP (.bankSum c (← ts.mapM (mapExprIdGo h)) (← mapExprIdGo h b)
          (← dc.mapM (mapExprIdGo h)))
      | .arr items => einternP (.arr (← items.mapM (mapExprIdGo h)))
      | .binary t a b => einternP (.binary t (← mapExprIdGo h a) (← mapExprIdGo h b))
      | .unary t a => einternP (.unary t (← mapExprIdGo h a))
      | .clamp a b c => einternP (.clamp (← mapExprIdGo h a) (← mapExprIdGo h b) (← mapExprIdGo h c))
      | .select a b c => einternP (.select (← mapExprIdGo h a) (← mapExprIdGo h b) (← mapExprIdGo h c))
      | .arraySet a b c => einternP (.arraySet (← mapExprIdGo h a) (← mapExprIdGo h b) (← mapExprIdGo h c))
      | .index a b => einternP (.index (← mapExprIdGo h a) (← mapExprIdGo h b))
      | .zeros c => einternP (.zeros (← mapExprIdGo h c))
      | .fold o i ac e b =>
        einternP (.fold (← mapExprIdGo h o) (← mapExprIdGo h i) (h.binder ac) (h.binder e) (← mapExprIdGo h b))
      | .scan o i ac e b =>
        einternP (.scan (← mapExprIdGo h o) (← mapExprIdGo h i) (h.binder ac) (h.binder e) (← mapExprIdGo h b))
      | .generate c it b =>
        einternP (.generate (← mapExprIdGo h c) (h.binder it) (← mapExprIdGo h b))
      | .iterate c i it b =>
        einternP (.iterate (← mapExprIdGo h c) (← mapExprIdGo h i) (h.binder it) (← mapExprIdGo h b))
      | .chain c i it b =>
        einternP (.chain (← mapExprIdGo h c) (← mapExprIdGo h i) (h.binder it) (← mapExprIdGo h b))
      | .map2 o e b =>
        einternP (.map2 (← mapExprIdGo h o) (h.binder e) (← mapExprIdGo h b))
      | .zipWith a b x y bd =>
        einternP (.zipWith (← mapExprIdGo h a) (← mapExprIdGo h b) (h.binder x) (h.binder y) (← mapExprIdGo h bd))
      | .letIn bs b =>
        let bs' ← bs.mapM fun lb => do
          pure ({ binder := h.binder lb.binder, value := ← mapExprIdGo h lb.value } : ELetBinder)
        einternP (.letIn bs' (← mapExprIdGo h b))
      | .tag d v p =>
        let p' ← p.mapM fun tp => do
          pure ({ field := tp.field, value := ← mapExprIdGo h tp.value } : ETagPayload)
        einternP (.tag d v p')
      | .match_ d s arms =>
        let arms' ← arms.mapM fun arm => do
          pure ({ variant := arm.variant, binders := arm.binders.map h.binder,
                  body := ← mapExprIdGo h arm.body } : EMatchArm)
        einternP (.match_ d (← mapExprIdGo h s) arms')
  modify (·.insert id.idx r)
  pure r

/-- One hook-set application of `mapExprIdGo` with a fresh memo. -/
def mapExprId (h : MapHooksId) (id : ExprId) : PassM ExprId :=
  (mapExprIdGo h id).run' {}

end Tropical.Ir.Strata
