import Std.Data.HashMap
import Tropical.Ir.Nodes
import Tropical.Ir.CoreArena
import Tropical.Ir.Strata.Basic

/-!
# EArena — the lowering's id-form working state (#190 native-DAG)

The tree passes thread an `Arena` (program pool of tree `Program`s) and rebuild
`Expr` trees; the bloat (the modulated clock duplicated into every oscillator
partial) is born in `inlineInstances`' substitution. This is the id-form
counterpart: an `EArena` carries a pool of `Program`s (id-valued) plus the
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

/-- Intern a nat as an integer literal id (`.num ⟨n, 0⟩`) — the loop-index /
    tag constant the array and sum lowerings emit. -/
def nat0E (n : Nat) : PassM ExprId := einternP (.num ⟨Int.ofNat n, 0⟩)

/-- The id wired to input port `i` — the last assignment wins (later `wire`
    calls override earlier ones). `none` when the port is unwired. Shared by
    the passes that substitute wired expressions at inline sites. -/
def wiredForE (inputs : Array InstanceInput) (i : Nat) : Option ExprId :=
  ((inputs.filter (·.port.idx == i)).back?).map (·.value)

/-- Dereference an id to its node (dangling id is an internal bug). -/
def derefP (id : ExprId) : PassM ENode := do
  match (← get).exprs.deref id with
  | some n => pure n
  | none => throw ⟨s!"EArena: dangling ExprId {id.idx} (internal)"⟩

/-- A `Program` by pool index, with a contextual error on miss. -/
def getEProgram (i : ProgramIdx) (ctx : String) : PassM Program := do
  match (← get).programs[i.idx]? with
  | some p => pure p
  | none => throw ⟨s!"{ctx}: program pool index {i.idx} out of range"⟩

/-- Append a rewritten program; returns its fresh pool index. -/
def pushEProgram (p : Program) : PassM ProgramIdx := do
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

/-- Id-form `getInstanceType`: resolve an instance's target `Program` through
    the enclosing program's registry. -/
def getInstanceTypeE (enclosing : Program) (instName typeKey : String) :
    PassM (ProgramIdx × Program) := do
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
-- there is no tree to `materialize` back to — the rewrites push their
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
-- REACHABLE-only: `ea.exprs` is append-only, so it can hold nodes no
-- longer referenced by the lowered root (rewritten-away identities,
-- dead JSON-interned structure), and a whole-arena fold would falsely
-- reject them. We convert only what the root (and its
-- instance-referenced registry) actually references — exactly the
-- evaluator-reachable graph the old `Core.check` walked. A REACHABLE
-- retired constructor (a JSON-loaded `fold`/`tag`/…) is refused here:
-- this boundary is the front-door contract now that no pass lowers
-- combinators or sum types.
-- ─────────────────────────────────────────────────────────────

open Tropical.Ir.Core (CoreProgram CoreInputDecl CoreOutputDecl CoreOutputAssign
  CoreBodyDecl CoreInstanceInput)

/-- Conversion state: the emit `CoreArena` under construction plus a memo
    mapping an `ExprArena` id (`.idx`) to its interned `CoreArena` id. -/
private abbrev ConvM := StateT (CoreArena × Std.HashMap Nat ExprId) (Except Error)

/-- The retired-constructor rejection: the front-door contract, stated at
    the type boundary. Combinator/sum-type lowering left with the surface
    language and generics that produced it (2026-07-25); a JSON program
    that still spells one of these ops is refused here, not silently
    miscompiled. Summing indexed families are authored as `bankSum`. -/
private def retired (op : String) : Error :=
  ⟨s!"toResolved: '{op}' is not a trunk construct — combinator/sum-type lowering was retired with its producers (the literate surface language and generics); a summing indexed family is authored as bankSum"⟩

/-- Convert one reachable `ExprArena` node (and its children) into the
    `CoreArena`, memoized. Rejects every retired constructor — the
    id-form analogue of `Core.checkExpr`, but visited only when actually
    referenced.

    TOTAL, by descent on `eid.idx`: the source arena is a frozen
    parameter and `hw` says every edge points down
    (`ExprArena.forall_children_lt`), so each recursive call is on a
    strictly smaller id. The pilot of the arena-termination survey. -/
private def convExprId (ea : ExprArena) (hw : ea.wf = true) (eid : ExprId) :
    ConvM ExprId := do
  match (← get).2.get? eid.idx with
  | some cid => return cid
  | none =>
    let cn : CNode ← match _hd : ea.deref eid with
      | none => throw ⟨s!"toResolved: dangling ExprId {eid.idx} (internal)"⟩
      | some (.num x)          => pure (.num x)
      | some (.bool b)         => pure (.bool b)
      | some (.arr items)      =>
        pure (.arr (← items.attach.mapM fun ⟨c, _⟩ => convExprId ea hw c))
      | some (.binary t a b)   => pure (.binary t (← convExprId ea hw a) (← convExprId ea hw b))
      | some (.unary t a)      => pure (.unary t (← convExprId ea hw a))
      | some (.clamp a b c)    => pure (.clamp (← convExprId ea hw a) (← convExprId ea hw b) (← convExprId ea hw c))
      | some (.select a b c)   => pure (.select (← convExprId ea hw a) (← convExprId ea hw b) (← convExprId ea hw c))
      | some (.arraySet a b c) => pure (.arraySet (← convExprId ea hw a) (← convExprId ea hw b) (← convExprId ea hw c))
      | some (.index a b)      => pure (.index (← convExprId ea hw a) (← convExprId ea hw b))
      | some (.inputRef i)     => pure (.inputRef i)
      | some (.paramRef i)     => pure (.paramRef i)
      | some (.nestedOut i o)  => pure (.nestedOut i o)
      | some .sampleRate       => pure .sampleRate
      | some .sampleIndex      => pure .sampleIndex
      | some (.loopIdx id)     => pure (.loopIdx id)
      | some (.bankSum c ts b dc ii) => do
        let ts' ← ts.attach.mapM fun ⟨t, _⟩ => convExprId ea hw t
        let b' ← convExprId ea hw b
        let dc' ← match _hdc : dc with
          | none => pure none
          | some d => pure (some (← convExprId ea hw d))
        pure (.bankSum c ts' b' dc' ii)
      | some (.zeros _)        => throw (retired "zeros")
      | some (.typeParamRef _) => throw (retired "typeParamRef")
      | some (.bindingRef _)   => throw (retired "bindingRef")
      | some (.letIn ..)       => throw (retired "let")
      | some (.fold ..)        => throw (retired "fold")
      | some (.scan ..)        => throw (retired "scan")
      | some (.generate ..)    => throw (retired "generate")
      | some (.iterate ..)     => throw (retired "iterate")
      | some (.chain ..)       => throw (retired "chain")
      | some (.map2 ..)        => throw (retired "map2")
      | some (.zipWith ..)     => throw (retired "zipWith")
      | some (.tag ..)         => throw (retired "tag")
      | some (.match_ ..)      => throw (retired "match")
    let st ← get
    let (cid, ca') := (intern cn).run st.1
    set (ca', st.2.insert eid.idx cid)
    return cid
termination_by eid.idx
decreasing_by
  all_goals
    apply ExprArena.forall_children_lt hw ‹ExprArena.deref _ _ = some _›
    simp_all [ENode.children]

/-- Convert the reachable `Program` subgraph rooted at `eIdx` into a
    `CoreProgram`, remapping every leaf id into the `CoreArena` and
    following instance-referenced registry entries recursively (the
    id-form `Core.check`). Port types resolve against the identity pools
    (`base`). -/
private partial def convProgram (ea : EArena) (hw : ea.exprs.wf = true)
    (eIdx : ProgramIdx) : ConvM CoreProgram := do
  let some ep := ea.programs[eIdx.idx]?
    | throw ⟨s!"toResolved: program pool index {eIdx.idx} out of range (internal)"⟩
  unless ep.typeParams.isEmpty do
    throw ⟨s!"core check ('{ep.name}'): {ep.typeParams.size} typeParam decl(s) — generics are retired; the trunk accepts only monomorphic programs"⟩
  let decls : Array CoreBodyDecl ← ep.decls.mapM fun d => do
    match d with
    | .param name value? => pure (.param name value?)
    | .inst name typeKey tArgs inputs =>
      let inputs' ← inputs.mapM fun i => do
        pure ({ port := i.port, value := ← convExprId ea.exprs hw i.value } : CoreInstanceInput)
      pure (.inst name typeKey tArgs inputs')
    | .prog name _ => pure (.progDecl name)
  let assigns : Array CoreOutputAssign ← ep.assigns.mapM fun a => do
    pure { target := a.target, expr := ← convExprId ea.exprs hw a.expr }
  let inputs : Array CoreInputDecl ← ep.inputs.mapM fun i => do
    pure { name := i.name, type? := Core.resolveOptPortType ea.base i.type?,
           default? := ← i.default?.mapM (convExprId ea.exprs hw) }
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
        registry := registry.push (typeKey, ← convProgram ea hw tIdx)
  return .mk ep.name inputs outputs decls assigns registry

/-- The Phase B strata-exit reify: post-strata `EArena` → `(CoreArena ×
    CoreProgram)`, sharing preserved, reachable-only from `root`. -/
def EArena.toResolved (ea : EArena) (root : ProgramIdx) :
    Except Error (CoreArena × CoreProgram) := do
  -- One O(edges) sweep buys the conversion's termination measure: every
  -- arena built through `eintern` is child-descending by construction,
  -- so a failure here is an interning-order bug, not a user error.
  if hw : ea.exprs.wf then
    let (core, (ca, _)) ← (convProgram ea hw root).run ({}, {})
    return (ca, core)
  else
    throw ⟨"toResolved: expression arena is not child-descending (internal interning-order bug)"⟩

/-- Downcast an elaborated `Arena` (a session root / per-program compile
    boundary that never ran the lowering rewrites, but IS the id-form) to
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
      | .nestedOut _ _ | .sampleRate | .sampleIndex | .loopIdx _ => pure id
      | .bankSum c ts b dc ii =>
        einternP (.bankSum c (← ts.mapM (mapExprIdGo h)) (← mapExprIdGo h b)
          (← dc.mapM (mapExprIdGo h)) ii)
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
