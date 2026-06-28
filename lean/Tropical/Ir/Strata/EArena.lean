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

/-- Instance-type lookup against an `EProgram`'s registry. -/
def _root_.Tropical.Ir.EProgram.registryGet? (p : EProgram) (key : String) :
    Option ProgramIdx :=
  (p.registry.find? (·.1 == key)).map (·.2)

/-- The id-form pipeline state: identity pools (`base`), an id-valued program
    pool, and the shared expression DAG. -/
structure EArena where
  base : Arena
  programs : Array EProgram := #[]
  exprs : ExprArena := {}
deriving Inhabited

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
-- Entry: tree Arena → id-form EArena
-- ─────────────────────────────────────────────────────────────

/-- Convert an input `Arena` into id form: intern every program's expressions
    into one shared DAG, parallel to the tree pool (same `ProgramIdx`es, so
    registry / `prog`-decl references stay valid). -/
def EArena.ofArena (a : Arena) : EArena :=
  let (programs, exprs) := (a.programs.mapM toEProgram).run {}
  { base := a, programs, exprs }

-- ─────────────────────────────────────────────────────────────
-- Exit: id-form EProgram → tree Program (Phase A only; thrown away in B)
-- ─────────────────────────────────────────────────────────────

/-- Convert one `EProgram` to a tree `Program`, derefing every id through the
    DAG and applying the registry / `prog`-decl index remap. -/
private def eprogramToProgram (ex : ExprArena) (ep : EProgram)
    (newReg : Array (String × ProgramIdx))
    (progRemap : ProgramIdx → ProgramIdx) : Program :=
  { name := ep.name
    typeParams := ep.typeParams
    inputs := ep.inputs.map fun d =>
      { name := d.name, type? := d.type?, default? := d.default?.map ex.toExpr }
    outputs := ep.outputs
    typeDefs := ep.typeDefs
    decls := ep.decls.map fun d => match d with
      | .param n v => .param n v
      | .prog n pi => .prog n (progRemap pi)
      | .inst n k ta ins =>
        .inst n k ta (ins.map fun i => { port := i.port, value := ex.toExpr i.value })
    assigns := ep.assigns.map fun a => { target := a.target, expr := ex.toExpr a.expr }
    binderCount := ep.binderCount
    registry := newReg }

/-- Recursively materialize the reachable `EProgram` subgraph rooted at `eIdx`
    into `acc` (a fresh tree `Arena` carrying the identity pools), children
    before parents so the acyclic pool invariant (refs point at lower indices)
    holds. `memo` maps an id-pool index to its tree-pool index. -/
private partial def materializeInto (ea : EArena) (eIdx : ProgramIdx)
    (acc : Arena) (memo : Std.HashMap Nat ProgramIdx) :
    Except Error (Arena × ProgramIdx × Std.HashMap Nat ProgramIdx) := do
  match memo.get? eIdx.idx with
  | some tIdx => pure (acc, tIdx, memo)
  | none =>
    let some ep := ea.programs[eIdx.idx]?
      | throw ⟨s!"materialize: program pool index {eIdx.idx} out of range (internal)"⟩
    -- Every child program reference: registry values + prog-decl targets.
    let progChildren : Array ProgramIdx := ep.decls.filterMap fun d =>
      match d with | .prog _ pi => some pi | _ => none
    let childIdxs : Array ProgramIdx := ep.registry.map (·.2) ++ progChildren
    let mut acc := acc
    let mut memo := memo
    for c in childIdxs do
      let (acc', _, memo') ← materializeInto ea c acc memo
      acc := acc'; memo := memo'
    let remap : ProgramIdx → ProgramIdx := fun pi =>
      (memo.get? pi.idx).getD pi
    let newReg := ep.registry.map fun (k, pi) => (k, remap pi)
    let treeProg := eprogramToProgram ea.exprs ep newReg remap
    let tIdx : ProgramIdx := ⟨acc.programs.size⟩
    acc := { acc with programs := acc.programs.push treeProg }
    memo := memo.insert eIdx.idx tIdx
    pure (acc, tIdx, memo)

/-- Materialize the post-strata root back to a tree `(Arena, ProgramIdx)`,
    appending the reachable subgraph onto the original program pool (like the
    tree passes, which push and never remove) so callers that still index
    unreachable originals keep working. -/
def EArena.materialize (ea : EArena) (root : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let (acc, tRoot, _) ← materializeInto ea root ea.base {}
  pure (acc, tRoot)

-- ─────────────────────────────────────────────────────────────
-- mapExprId — id-form structural walker (mirrors Recursion.mapExpr)
-- ─────────────────────────────────────────────────────────────

/-- Hook set for `mapExprId`: `node n` may replace a node (returning its id,
    possibly freshly interned via `einternP`); `none` recurses structurally.
    `binder` transforms binders at every binding site. -/
structure MapHooksId where
  node : ENode → PassM (Option ExprId) := fun _ => pure none
  binder : Binder → Binder := id

/-- Structural map over an id-rooted expression, re-interning the result. Equal
    subtrees collapse on intern, so the rewrite produces a DAG. -/
partial def mapExprId (h : MapHooksId) (id : ExprId) : PassM ExprId := do
  let n ← derefP id
  match ← h.node n with
  | some r => pure r
  | none =>
    match n with
    | .num _ | .bool _
    | .inputRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
    | .nestedOut _ _ | .sampleRate | .sampleIndex => pure id
    | .arr items => einternP (.arr (← items.mapM (mapExprId h)))
    | .binary t a b => einternP (.binary t (← mapExprId h a) (← mapExprId h b))
    | .unary t a => einternP (.unary t (← mapExprId h a))
    | .clamp a b c => einternP (.clamp (← mapExprId h a) (← mapExprId h b) (← mapExprId h c))
    | .select a b c => einternP (.select (← mapExprId h a) (← mapExprId h b) (← mapExprId h c))
    | .arraySet a b c => einternP (.arraySet (← mapExprId h a) (← mapExprId h b) (← mapExprId h c))
    | .index a b => einternP (.index (← mapExprId h a) (← mapExprId h b))
    | .zeros c => einternP (.zeros (← mapExprId h c))
    | .fold o i ac e b =>
      einternP (.fold (← mapExprId h o) (← mapExprId h i) (h.binder ac) (h.binder e) (← mapExprId h b))
    | .scan o i ac e b =>
      einternP (.scan (← mapExprId h o) (← mapExprId h i) (h.binder ac) (h.binder e) (← mapExprId h b))
    | .generate c it b =>
      einternP (.generate (← mapExprId h c) (h.binder it) (← mapExprId h b))
    | .iterate c i it b =>
      einternP (.iterate (← mapExprId h c) (← mapExprId h i) (h.binder it) (← mapExprId h b))
    | .chain c i it b =>
      einternP (.chain (← mapExprId h c) (← mapExprId h i) (h.binder it) (← mapExprId h b))
    | .map2 o e b =>
      einternP (.map2 (← mapExprId h o) (h.binder e) (← mapExprId h b))
    | .zipWith a b x y bd =>
      einternP (.zipWith (← mapExprId h a) (← mapExprId h b) (h.binder x) (h.binder y) (← mapExprId h bd))
    | .letIn bs b =>
      let bs' ← bs.mapM fun lb => do
        pure ({ binder := h.binder lb.binder, value := ← mapExprId h lb.value } : ELetBinder)
      einternP (.letIn bs' (← mapExprId h b))
    | .tag d v p =>
      let p' ← p.mapM fun tp => do
        pure ({ field := tp.field, value := ← mapExprId h tp.value } : ETagPayload)
      einternP (.tag d v p')
    | .match_ d s arms =>
      let arms' ← arms.mapM fun arm => do
        pure ({ variant := arm.variant, binders := arm.binders.map h.binder,
                body := ← mapExprId h arm.body } : EMatchArm)
      einternP (.match_ d (← mapExprId h s) arms')

end Tropical.Ir.Strata
