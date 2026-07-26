import Std.Data.HashMap
import Tropical.Ir.Nodes
import Tropical.Ir.Core
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
builds nodes, `derefP` reads them, `pushEProgram` appends a rewritten program.
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
-- Exit: id-form EArena → (ExprArena × CoreProgram) — reachability GC
--
-- The strata DAG is threaded straight to emit as a fresh `ExprArena`
-- (there is ONE arena vocabulary now — src and dst differ only in
-- which ids are live). Each reachable expression node is copied once
-- and equal subtrees stay one node, so O(unique nodes), never
-- O(expanded tree). REACHABLE-only: `ea.exprs` is append-only, so it
-- can hold nodes no longer referenced by the lowered root
-- (rewritten-away identities); we copy only what the root (and its
-- instance-referenced registry) actually references. There is no
-- refusal here any more — retired constructors are refused at the JSON
-- front doors and are unspellable in `ENode`.
-- ─────────────────────────────────────────────────────────────

open Tropical.Ir.Core (CoreProgram CoreInputDecl CoreOutputDecl CoreOutputAssign
  CoreBodyDecl CoreInstanceInput)

/-- GC state: the fresh dst `ExprArena` under construction plus a memo
    mapping a src id (`.idx`) to its interned dst id. -/
private abbrev ConvM := StateT (ExprArena × Std.HashMap Nat ExprId) (Except Error)

/-- Copy one reachable src node (and its children) into the dst arena,
    memoized — a pure structure-preserving copy; the reachability GC.

    TOTAL, by descent on `eid.idx`: the source arena is a frozen
    parameter and `hw` says every edge points down
    (`ExprArena.forall_children_lt`), so each recursive call is on a
    strictly smaller id. The pilot of the arena-termination survey. -/
private def convExprId (ea : ExprArena) (hw : ea.wf = true) (eid : ExprId) :
    ConvM ExprId := do
  match (← get).2.get? eid.idx with
  | some cid => return cid
  | none =>
    let cn : ENode ← match _hd : ea.deref eid with
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
    let st ← get
    let (cid, ca') := (eintern cn).run st.1
    set (ca', st.2.insert eid.idx cid)
    return cid
termination_by eid.idx
decreasing_by
  all_goals
    apply ExprArena.forall_children_lt hw ‹ExprArena.deref _ _ = some _›
    simp_all [ENode.children]

/-- Convert the reachable `Program` subgraph rooted at `eIdx` into a
    `CoreProgram`, remapping every leaf id into the `ExprArena` and
    following instance-referenced registry entries recursively (the
    id-form `Core.check`).

    TOTAL, by descent on `eIdx.idx`: `hwp` says every registry edge
    points strictly below its program, so the follow is a frozen-pool
    descent — the same recipe as the expression walks, one level up.
    The key collection (first-use dedup, matching `Core.check`) runs
    first as a plain loop, carrying each target's decrease fact as a
    subtype so the recursive `mapM` discharges its measure directly. -/
private def convProgram (ea : EArena) (hw : ea.exprs.wf = true)
    (hwp : progPoolWf ea.programs = true)
    (eIdx : ProgramIdx) : ConvM CoreProgram := do
  match hp : ea.programs[eIdx.idx]? with
  | none => throw ⟨s!"toResolved: program pool index {eIdx.idx} out of range (internal)"⟩
  | some ep => do
    let decls : Array CoreBodyDecl ← ep.decls.mapM fun d => do
      match d with
      | .param name value? => pure (.param name value?)
      | .inst name typeKey inputs =>
        let inputs' ← inputs.mapM fun i => do
          pure ({ port := i.port, value := ← convExprId ea.exprs hw i.value } : CoreInstanceInput)
        pure (.inst name typeKey inputs')
      | .prog name _ => pure (.progDecl name)
    let assigns : Array CoreOutputAssign ← ep.assigns.mapM fun a => do
      pure { target := a.target, expr := ← convExprId ea.exprs hw a.expr }
    let inputs : Array CoreInputDecl ← ep.inputs.mapM fun i => do
      pure { name := i.name, type? := i.type?,
             default? := ← i.default?.mapM (convExprId ea.exprs hw) }
    let outputs : Array CoreOutputDecl := ep.outputs.map fun o =>
      { name := o.name, type? := o.type? }
    -- Registry: follow only instance-referenced entries (evaluator-
    -- reachable), first-use dedup order, each carrying its decrease fact.
    let mut keys : Array (String × {t : ProgramIdx // t.idx < eIdx.idx}) := #[]
    for d in ep.decls do
      if let .inst name typeKey _ := d then
        unless keys.any (·.1 == typeKey) do
          match hr : ep.registryGet? typeKey with
          | some tIdx =>
            keys := keys.push (typeKey, ⟨tIdx, progPool_registry_lt hwp hp hr⟩)
          | none =>
            throw ⟨s!"core check ('{ep.name}'): instance '{name}' typeKey '{typeKey}' missing from registry"⟩
    let registry ← keys.mapM fun kt => do
      pure (kt.1, ← convProgram ea hw hwp kt.2.1)
    return .mk ep.name inputs outputs decls assigns registry
termination_by eIdx.idx
decreasing_by exact kt.2.2

/-- The Phase B strata-exit reify: post-strata `EArena` → `(ExprArena ×
    CoreProgram)`, sharing preserved, reachable-only from `root`. -/
def EArena.toResolved (ea : EArena) (root : ProgramIdx) :
    Except Error (ExprArena × CoreProgram) := do
  -- Two O(edges) sweeps buy the conversion's termination measures: the
  -- expression arena's child-descending ids and the program pool's —
  -- both hold by construction, so a failure here is a construction-
  -- order bug, not a user error.
  if hw : ea.exprs.wf then
    if hwp : progPoolWf ea.programs then
      let (core, (ca, _)) ← (convProgram ea hw hwp root).run ({}, {})
      return (ca, core)
    else
      throw ⟨"toResolved: program pool is not child-descending (internal construction-order bug)"⟩
  else
    throw ⟨"toResolved: expression arena is not child-descending (internal interning-order bug)"⟩

/-- Downcast an elaborated `Arena` (a session root / per-program compile
    boundary that never ran the lowering rewrites, but IS the id-form) to
    `(ExprArena × CoreProgram)`. A thin `Except String` wrapper over
    `toResolved` for the compile call sites; reachable-only, so it validates
    only the evaluator-reachable graph and never touches the whole pool. -/
def _root_.Tropical.Ir.checkResolvedArena (a : Arena) (root : ProgramIdx) :
    Except String (Tropical.Ir.ExprArena × Tropical.Ir.Core.CoreProgram) :=
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
    Hooks are ONE-HOP: a replacement id is returned whole, never re-walked —
    a hook that needs collapsed values consults a pre-normalized table
    (see the pass-side fixpoint loops), it does not recurse through the
    walk. That contract is what makes the walker total. -/
structure MapHooksId where
  node : ENode → MapM (Option ExprId) := fun _ => pure none

/-- Freeze the current expression arena as a walk group's read source.
    The walks only ever deref pre-walk ids — rewrites intern (write) but
    are never read back — so one snapshot plus one O(edges) `wf` check
    buys `mapExprIdGo`'s termination measure for every walk in the
    group. The failure arm is an interning-order bug, never a user
    error (every arena built through `eintern` is child-descending by
    construction). -/
def withFrozenSrc {α} (ctx : String)
    (k : (src : ExprArena) → src.wf = true → PassM α) : PassM α := do
  let src := (← get).exprs
  if hw : src.wf then k src hw
  else failP s!"{ctx}: arena is not child-descending (internal interning-order bug)"

/-- Structural map over an id-rooted expression, re-interning the result and
    memoizing per source id. Equal subtrees collapse on intern, so the rewrite
    produces a DAG — and the memo makes the walk itself DAG-shaped too.

    TOTAL, by descent on `id.idx` over the frozen source `src`: every
    deref reads the snapshot (the walk never reads what it interns), and
    `hw` says the snapshot's edges point down. A hook may start a FRESH
    walk (its own root, its own measure); it must not re-enter this one. -/
def mapExprIdGo (src : ExprArena) (hw : src.wf = true) (h : MapHooksId)
    (id : ExprId) : MapM ExprId := do
  if let some r := (← get).get? id.idx then return r
  let r ← do
    match _hd : src.deref id with
    | none => failP s!"mapExprId: dangling ExprId {id.idx} (internal)"
    | some n =>
      match ← h.node n with
      | some r => pure r
      | none =>
        match _hn : n with
        | .num _ | .bool _ | .inputRef _ | .paramRef _
        | .nestedOut _ _ | .sampleRate | .sampleIndex | .loopIdx _ => pure id
        | .bankSum c ts b dc ii =>
          einternP (.bankSum c
            (← ts.attach.mapM fun ⟨t, _⟩ => mapExprIdGo src hw h t)
            (← mapExprIdGo src hw h b)
            (← match _hdc : dc with
                | none => pure none
                | some d => some <$> mapExprIdGo src hw h d) ii)
        | .arr items =>
          einternP (.arr (← items.attach.mapM fun ⟨x, _⟩ => mapExprIdGo src hw h x))
        | .binary t a b => einternP (.binary t (← mapExprIdGo src hw h a) (← mapExprIdGo src hw h b))
        | .unary t a => einternP (.unary t (← mapExprIdGo src hw h a))
        | .clamp a b c => einternP (.clamp (← mapExprIdGo src hw h a) (← mapExprIdGo src hw h b) (← mapExprIdGo src hw h c))
        | .select a b c => einternP (.select (← mapExprIdGo src hw h a) (← mapExprIdGo src hw h b) (← mapExprIdGo src hw h c))
        | .arraySet a b c => einternP (.arraySet (← mapExprIdGo src hw h a) (← mapExprIdGo src hw h b) (← mapExprIdGo src hw h c))
        | .index a b => einternP (.index (← mapExprIdGo src hw h a) (← mapExprIdGo src hw h b))
  modify (·.insert id.idx r)
  pure r
termination_by id.idx
decreasing_by
  all_goals
    apply ExprArena.forall_children_lt hw ‹ExprArena.deref _ _ = some _›
    simp_all [ENode.children]

/-- One hook-set application of `mapExprIdGo` with a fresh memo. -/
def mapExprId (src : ExprArena) (hw : src.wf = true) (h : MapHooksId)
    (id : ExprId) : PassM ExprId :=
  (mapExprIdGo src hw h id).run' {}

end Tropical.Ir.Strata
