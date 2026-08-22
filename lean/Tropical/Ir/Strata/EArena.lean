import Std.Data.HashMap
import Std.Data.HashMap.Lemmas
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

instance : ReflBEq OutputTarget where
  rfl := by
    intro target
    cases target with
    | port idx =>
      change (idx == idx) = true
      exact BEq.rfl
    | dac => rfl

instance : LawfulBEq OutputTarget where
  rfl := by
    intro target
    cases target with
    | port idx =>
      change (idx == idx) = true
      exact BEq.rfl
    | dac => rfl
  eq_of_beq := by
    intro lhs rhs h
    cases lhs with
    | port lhs =>
      cases rhs with
      | port rhs =>
        change (lhs == rhs) = true at h
        exact congrArg OutputTarget.port (eq_of_beq h)
      | dac =>
        change false = true at h
        contradiction
    | dac =>
      cases rhs with
      | port rhs =>
        change false = true at h
        contradiction
      | dac => rfl

/-- The representation-neutral, evaluator-visible part of an input. -/
structure ProgramCopyInput (ρ : Type) where
  name : String
  default? : Option ρ := none
deriving BEq, ReflBEq, LawfulBEq

/-- The representation-neutral, evaluator-visible part of an instance input. -/
structure ProgramCopyInstanceInput (ρ : Type) where
  port : InputIdx
  value : ρ
deriving BEq, ReflBEq, LawfulBEq

/-- The representation-neutral, evaluator-visible body declarations. -/
inductive ProgramCopyDecl (ρ : Type) where
  | param (name : String) (value? : Option Lean.JsonNumber)
  | inst (name typeKey : String) (inputs : Array (ProgramCopyInstanceInput ρ))
  | progDecl (name : String)
deriving BEq, ReflBEq, LawfulBEq

/-- The representation-neutral, evaluator-visible part of an assignment. -/
structure ProgramCopyAssign (ρ : Type) where
  target : OutputTarget
  expr : ρ
deriving BEq, ReflBEq, LawfulBEq

/-- A common structural view of source and core programs.  It deliberately
    omits port types and registry storage: neither is inspected by the small
    per-sample evaluator, while output arity and authored declaration order
    remain observable. -/
structure ProgramCopyView (ρ : Type) where
  name : String
  inputs : Array (ProgramCopyInput ρ)
  outputCount : Nat
  decls : Array (ProgramCopyDecl ρ)
  assigns : Array (ProgramCopyAssign ρ)
deriving BEq, ReflBEq, LawfulBEq

def Program.copyView (program : Program) : ProgramCopyView ExprId :=
  { name := program.name
    inputs := program.inputs.map fun input =>
      { name := input.name, default? := input.default? }
    outputCount := program.outputs.size
    decls := program.decls.map fun
      | .param name value? => .param name value?
      | .inst name typeKey inputs => .inst name typeKey <|
          inputs.map fun input => { port := input.port, value := input.value }
      | .prog name _ => .progDecl name
    assigns := program.assigns.map fun assign =>
      { target := assign.target, expr := assign.expr } }

def CoreProgram.copyView (program : CoreProgram) : ProgramCopyView ExprId :=
  { name := program.name
    inputs := program.inputs.map fun input =>
      { name := input.name, default? := input.default? }
    outputCount := program.outputs.size
    decls := program.decls.map fun
      | .param name value? => .param name value?
      | .inst name typeKey inputs => .inst name typeKey <|
          inputs.map fun input => { port := input.port, value := input.value }
      | .progDecl name => .progDecl name
    assigns := program.assigns.map fun assign =>
      { target := assign.target, expr := assign.expr } }

/-- GC state: the fresh dst `ExprArena` under construction plus a memo
    mapping a src id (`.idx`) to its interned dst id. -/
private abbrev ConvM := StateT (ExprArena × Std.HashMap Nat ExprId) (Except Error)

/-- Proof-facing source-to-destination id memo produced by reachability copy. -/
abbrev ExprCopyMemo := Std.HashMap Nat ExprId

def remapExprId? (memo : ExprCopyMemo) (id : ExprId) : Option ExprId :=
  memo[id.idx]?

def remapProgramCopyInput? (memo : ExprCopyMemo)
    (input : ProgramCopyInput ExprId) : Option (ProgramCopyInput ExprId) := do
  pure { input with default? := ← input.default?.mapM (remapExprId? memo) }

def remapProgramCopyInstanceInput? (memo : ExprCopyMemo)
    (input : ProgramCopyInstanceInput ExprId) : Option (ProgramCopyInstanceInput ExprId) := do
  pure { input with value := ← remapExprId? memo input.value }

def remapProgramCopyDecl? (memo : ExprCopyMemo) :
    ProgramCopyDecl ExprId → Option (ProgramCopyDecl ExprId)
  | .param name value? => pure (.param name value?)
  | .progDecl name => pure (.progDecl name)
  | .inst name typeKey inputs => do
      pure (.inst name typeKey
        (← inputs.mapM (remapProgramCopyInstanceInput? memo)))

def remapProgramCopyAssign? (memo : ExprCopyMemo)
    (assign : ProgramCopyAssign ExprId) : Option (ProgramCopyAssign ExprId) := do
  pure { assign with expr := ← remapExprId? memo assign.expr }

/-- Rename every evaluator-reachable expression root in a program view. -/
def remapProgramCopyView? (memo : ExprCopyMemo)
    (view : ProgramCopyView ExprId) : Option (ProgramCopyView ExprId) := do
  pure { view with
    inputs := ← view.inputs.mapM (remapProgramCopyInput? memo)
    decls := ← view.decls.mapM (remapProgramCopyDecl? memo)
    assigns := ← view.assigns.mapM (remapProgramCopyAssign? memo) }

def remapExprIdList? (memo : ExprCopyMemo) :
    List ExprId → Option (List ExprId)
  | [] => pure []
  | id :: rest =>
    match memo[id.idx]? with
    | none => none
    | some mapped =>
      match remapExprIdList? memo rest with
      | none => none
      | some mappedRest => some (mapped :: mappedRest)

def remapExprIds? (memo : ExprCopyMemo) (ids : Array ExprId) :
    Option (Array ExprId) :=
  (·.toArray) <$> remapExprIdList? memo ids.toList

/-- Rename every child of a source node through a completed conversion memo.
    Metadata (including binders, dynamic counts, routes, and tile identity) is
    copied verbatim. -/
def remapENode? (memo : ExprCopyMemo) : ENode → Option ENode
  | .num value => pure (.num value)
  | .bool value => pure (.bool value)
  | .arr items => .arr <$> remapExprIds? memo items
  | .tileArray items => .tileArray <$> remapExprIds? memo items
  | .binary tag lhs rhs =>
      match memo[lhs.idx]?, memo[rhs.idx]? with
      | some lhs, some rhs => some (.binary tag lhs rhs)
      | _, _ => none
  | .unary tag arg =>
      match memo[arg.idx]? with
      | some arg => some (.unary tag arg)
      | none => none
  | .clamp value lo hi =>
      match memo[value.idx]?, memo[lo.idx]?, memo[hi.idx]? with
      | some value, some lo, some hi => some (.clamp value lo hi)
      | _, _, _ => none
  | .select cond then_ else_ =>
      match memo[cond.idx]?, memo[then_.idx]?, memo[else_.idx]? with
      | some cond, some then_, some else_ => some (.select cond then_ else_)
      | _, _, _ => none
  | .arraySet array index value =>
      match memo[array.idx]?, memo[index.idx]?, memo[value.idx]? with
      | some array, some index, some value => some (.arraySet array index value)
      | _, _, _ => none
  | .index array index =>
      match memo[array.idx]?, memo[index.idx]? with
      | some array, some index => some (.index array index)
      | _, _ => none
  | .inputRef index => pure (.inputRef index)
  | .paramRef index => pure (.paramRef index)
  | .nestedOut instanceIdx outputIdx =>
      pure (.nestedOut instanceIdx outputIdx)
  | .sampleRate => pure .sampleRate
  | .sampleIndex => pure .sampleIndex
  | .tileSampleIndex => pure .tileSampleIndex
  | .tilePhase => pure .tilePhase
  | .loopIdx binderId => pure (.loopIdx binderId)
  | .bankSum capacity tables body dynCount? binderId => do
      let tables ← remapExprIds? memo tables
      let body ← memo[body.idx]?
      let dynCount? ← dynCount?.mapM fun count => memo[count.idx]?
      pure (.bankSum capacity tables body dynCount? binderId)
  | .routedSum capacity outputCount routes tables values dynCount? binderId => do
      let tables ← remapExprIds? memo tables
      let values ← remapExprIds? memo values
      let dynCount? ← dynCount?.mapM fun count => memo[count.idx]?
      pure (.routedSum capacity outputCount routes tables values dynCount? binderId)

/-- Executable validation of the completed copy memo.  Every memo entry must
    dereference on both sides to the same node after child renaming. -/
def checkExprCopyMemo (src dst : ExprArena) (memo : ExprCopyMemo) : Bool :=
  memo.toList.all fun entry =>
    match src.deref ⟨entry.1⟩ with
    | none => false
    | some sourceNode =>
      match remapENode? memo sourceNode with
      | none => false
      | some destNode => dst.deref entry.2 == some destNode

/-- Executable form of the complete semantic arena invariant.  Unlike
    `ExprArena.wf`, this also checks the hash-cons index and stage alignment. -/
def semanticWfCheck (arena : ExprArena) : Bool :=
  arena.wf &&
  arena.sigs.size == arena.nodes.size &&
  arena.dedup.toList.all fun entry => arena.deref entry.2 == some entry.1

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
      | some (.tileArray items) =>
        pure (.tileArray (← items.attach.mapM fun ⟨c, _⟩ => convExprId ea hw c))
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
      | some .tileSampleIndex  => pure .tileSampleIndex
      | some .tilePhase        => pure .tilePhase
      | some (.loopIdx id)     => pure (.loopIdx id)
      | some (.bankSum c ts b dc ii) => do
        let ts' ← ts.attach.mapM fun ⟨t, _⟩ => convExprId ea hw t
        let b' ← convExprId ea hw b
        let dc' ← match _hdc : dc with
          | none => pure none
          | some d => pure (some (← convExprId ea hw d))
        pure (.bankSum c ts' b' dc' ii)
      | some (.routedSum c oc rs ts vs dc ii) => do
        let ts' ← ts.attach.mapM fun ⟨t, _⟩ => convExprId ea hw t
        let vs' ← vs.attach.mapM fun ⟨v, _⟩ => convExprId ea hw v
        let dc' ← match _hdc : dc with
          | none => pure none
          | some d => pure (some (← convExprId ea hw d))
        pure (.routedSum c oc rs ts' vs' dc' ii)
    let st ← get
    let (cid, ca') := (eintern cn).run st.1
    set (ca', st.2.insert eid.idx cid)
    return cid
termination_by eid.idx
decreasing_by
  all_goals
    apply ExprArena.forall_children_lt hw ‹ExprArena.deref _ _ = some _›
    simp_all [ENode.children]

/-- Instance-referenced registry entries in evaluator first-use order.  The
    target subtype retains the pool-descent fact used by both conversion and
    proof-witness checking. -/
def referencedPrograms (ea : EArena)
    (hwp : progPoolWf ea.programs = true) (eIdx : ProgramIdx)
    (ep : Program) (hp : ea.programs[eIdx.idx]? = some ep) :
    Except Error (Array (String × {t : ProgramIdx // t.idx < eIdx.idx})) := do
  let mut keys : Array (String × {t : ProgramIdx // t.idx < eIdx.idx}) := #[]
  for d in ep.decls do
    if let .inst name typeKey _ := d then
      unless keys.any (·.1 == typeKey) do
        match hr : ep.registryGet? typeKey with
        | some tIdx =>
          keys := keys.push (typeKey, ⟨tIdx, progPool_registry_lt hwp hp hr⟩)
        | none =>
          throw ⟨s!"core check ('{ep.name}'): instance '{name}' typeKey '{typeKey}' missing from registry"⟩
  pure keys

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
    let keys ← referencedPrograms ea hwp eIdx ep hp
    let registry ← keys.mapM fun kt => do
      pure (kt.1, ← convProgram ea hw hwp kt.2.1)
    return .mk ep.name inputs outputs decls assigns registry
termination_by eIdx.idx
decreasing_by exact kt.2.2

/-- Executable validation of the complete reachable program copy.  Node views
    check all evaluator-visible fields and expression-root renaming; recursive
    checks follow exactly the first-use registry spine used by `convProgram`. -/
def checkProgramCopy (ea : EArena) (hwp : progPoolWf ea.programs = true)
    (memo : ExprCopyMemo) (root : ProgramIdx) (core : CoreProgram) : Bool :=
  match hp : ea.programs[root.idx]? with
  | none => false
  | some program =>
    match remapProgramCopyView? memo
        (Tropical.Ir.Strata.Program.copyView program) with
    | none => false
    | some copiedView =>
      match referencedPrograms ea hwp root program hp with
      | .error _ => false
      | .ok keys =>
        copiedView == Tropical.Ir.Strata.CoreProgram.copyView core &&
        keys.map (·.1) == core.registry.map (·.1) &&
        program.decls.all fun decl =>
          match decl with
          | .param .. | .prog .. => true
          | .inst _ typeKey _ =>
            match hs : program.registryGet? typeKey with
            | none => false
            | some sourceChild =>
              match hr : core.registryGet? typeKey with
              | none => false
              | some child => checkProgramCopy ea hwp memo sourceChild child
termination_by root.idx
decreasing_by exact progPool_registry_lt hwp hp hs

/-- The Phase B strata-exit reify together with the checks that make its
    source/destination correspondence proof-visible. -/
structure ResolvedCopy (ea : EArena) (root : ProgramIdx) where
  exprs : ExprArena
  program : CoreProgram
  memo : ExprCopyMemo
  sourceExprDescends : ea.exprs.wf = true
  sourceProgramsDescend : progPoolWf ea.programs = true
  destinationChecked : semanticWfCheck exprs = true
  expressionsChecked : checkExprCopyMemo ea.exprs exprs memo = true
  programChecked : checkProgramCopy ea sourceProgramsDescend memo root program = true

/-- Proof-facing variant retaining the exact source/destination root memo. -/
def EArena.toResolvedWithWitness (ea : EArena) (root : ProgramIdx) :
    Except Error (ResolvedCopy ea root) := do
  -- Two O(edges) sweeps buy the conversion's termination measures: the
  -- expression arena's child-descending ids and the program pool's —
  -- both hold by construction, so a failure here is a construction-
  -- order bug, not a user error.
  if hw : ea.exprs.wf then
    if hwp : progPoolWf ea.programs then
      let (core, (ca, memo)) ← (convProgram ea hw hwp root).run ({}, {})
      if hArena : semanticWfCheck ca then
        if hExprCopy : checkExprCopyMemo ea.exprs ca memo then
          if hProgramCopy : checkProgramCopy ea hwp memo root core then
            return ResolvedCopy.mk ca core memo hw hwp hArena hExprCopy hProgramCopy
          else
            throw ⟨"toResolved: program copy witness check failed (internal)"⟩
        else
          throw ⟨"toResolved: expression copy witness check failed (internal)"⟩
      else
        throw ⟨"toResolved: destination arena invariant check failed (internal)"⟩
    else
      throw ⟨"toResolved: program pool is not child-descending (internal construction-order bug)"⟩
  else
    throw ⟨"toResolved: expression arena is not child-descending (internal interning-order bug)"⟩

/-- The Phase B strata-exit reify: post-strata `EArena` → `(ExprArena ×
    CoreProgram)`, sharing preserved, reachable-only from `root`. -/
def EArena.toResolved (ea : EArena) (root : ProgramIdx) :
    Except Error (ExprArena × CoreProgram) := do
  let result ← ea.toResolvedWithWitness root
  pure (result.exprs, result.program)

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
        | .nestedOut _ _ | .sampleRate | .sampleIndex | .tileSampleIndex
        | .tilePhase
        | .loopIdx _ => pure id
        | .bankSum c ts b dc ii =>
          einternP (.bankSum c
            (← ts.attach.mapM fun ⟨t, _⟩ => mapExprIdGo src hw h t)
            (← mapExprIdGo src hw h b)
            (← match _hdc : dc with
                | none => pure none
                | some d => some <$> mapExprIdGo src hw h d) ii)
        | .routedSum c oc rs ts vs dc ii =>
          einternP (.routedSum c oc rs
            (← ts.attach.mapM fun ⟨t, _⟩ => mapExprIdGo src hw h t)
            (← vs.attach.mapM fun ⟨v, _⟩ => mapExprIdGo src hw h v)
            (← match _hdc : dc with
                | none => pure none
                | some d => some <$> mapExprIdGo src hw h d) ii)
        | .arr items =>
          einternP (.arr (← items.attach.mapM fun ⟨x, _⟩ => mapExprIdGo src hw h x))
        | .tileArray items =>
          einternP (.tileArray
            (← items.attach.mapM fun ⟨x, _⟩ => mapExprIdGo src hw h x))
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
