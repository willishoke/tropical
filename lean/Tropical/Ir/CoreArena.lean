import Std.Data.HashMap
import Tropical.Ir.Core

/-!
# CoreArena — hash-consed (DAG) form of the post-strata expression IR

The native-DAG representation for issue #190. A `CNode` is a post-strata
expression node whose children are `ExprId`s into an arena instead of inlined
subtrees — so a node is **flat** (O(1) to hash and compare) and equal subtrees
are one arena entry referenced by many ids. Interning at construction
(`intern`) makes duplication impossible: the bloat a tree representation forces
(the modulated clock copied into every oscillator partial) collapses to a
single node.

This is now the ONLY post-strata expression representation — the tree twin
`CoreExpr` is gone (Phase B). Both the strata exit (`EArena.toResolved`) and the
elaborated-tree downcast (`checkResolvedArena`, below) intern straight into a
`CoreArena`; `Core`'s program leaves are `ExprId`s.

Soundness: interning merges two nodes iff they have the same constructor and the
same child ids. Every tropical op is pure and deterministic, so a merge never
changes a computed value — the rendered audio is identical (the goldens hash the
audio, not the plan, so register relabeling from a DAG walk is free).

The remaining tree twin `Expr` (the elaborator's output, still consumed by the
strata passes and `materialize`) is the next strangler target — deleting it
means the elaborator constructs interned nodes directly.
-/

namespace Tropical.Ir

open Lean (JsonNumber)

/-- A post-strata expression node with children referenced by `ExprId`.
    The 14-constructor post-strata subset; flat (no inlined subtrees). -/
inductive CNode where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array ExprId)
  | binary (tag : BinaryOpTag) (lhs rhs : ExprId)
  | unary (tag : UnaryOpTag) (arg : ExprId)
  | clamp (value lo hi : ExprId)
  | select (cond then_ else_ : ExprId)
  | arraySet (arr idx value : ExprId)
  | index (arr idx : ExprId)
  | inputRef (idx : InputIdx)
  | paramRef (idx : ParamIdx)
  | nestedOut (instance_ : InstanceIdx) (output : OutputIdx)
  | sampleRate
  | sampleIndex
deriving BEq, Repr, Inhabited

/-- O(1) structural hash — children are ids, so no subtree recursion. Op tags
    fold through their `.wire` string (no `Hashable` needed on the tag types). -/
def cnodeHash : CNode → UInt64
  | .num n          => mixHash 1 (hash n)
  | .bool b         => mixHash 2 (hash b)
  | .arr items      => mixHash 3 (hash (items.map (·.idx)))
  | .binary t a b   => mixHash (mixHash (mixHash 4 (hash t.wire)) (hash a.idx)) (hash b.idx)
  | .unary t a      => mixHash (mixHash 5 (hash t.wire)) (hash a.idx)
  | .clamp a b c    => mixHash (mixHash (mixHash 6 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .select a b c   => mixHash (mixHash (mixHash 7 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .arraySet a b c => mixHash (mixHash (mixHash 8 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .index a b      => mixHash (mixHash 9 (hash a.idx)) (hash b.idx)
  | .inputRef i     => mixHash 10 (hash i.idx)
  | .paramRef i     => mixHash 11 (hash i.idx)
  | .nestedOut i o  => mixHash (mixHash 12 (hash i.idx)) (hash o.idx)
  | .sampleRate     => 13
  | .sampleIndex    => 14

instance : Hashable CNode := ⟨cnodeHash⟩

/-- Interned node store: `nodes[id]` is the node; `dedup` maps a node back to its
    id so equal nodes collapse. Append-only; ids are assigned in first-seen order. -/
structure CoreArena where
  nodes : Array CNode := #[]
  dedup : Std.HashMap CNode ExprId := {}
deriving Inhabited

abbrev ArenaM := StateM CoreArena

/-- Intern a flat node, returning its (shared) id. O(1). -/
def intern (n : CNode) : ArenaM ExprId := do
  let a ← get
  match a.dedup.get? n with
  | some id => pure id
  | none =>
    let id : ExprId := ⟨a.nodes.size⟩
    set { a with nodes := a.nodes.push n, dedup := a.dedup.insert n id }
    pure id

/-- Dereference an id to its node. -/
def CoreArena.deref (a : CoreArena) (id : ExprId) : Option CNode :=
  a.nodes[id.idx]?

-- ─────────────────────────────────────────────────────────────
-- checkResolvedArena — the tree downcast (reachable-only), interning
--
-- The Phase B replacement for `Core.check` on the paths that hand a
-- freshly-*elaborated* tree `Arena` straight to the compiler (a
-- synthetic session root, a per-program emit) — those never run the
-- strata passes, so there is no `EArena` to reify. It validates the
-- post-strata 14-constructor subset (byte-exact `Core.check` messages)
-- AND interns each leaf into a fresh `CoreArena`, so the emit consumes
-- `ExprId`s like the strata path. Reachable-only (body exprs, then the
-- instance-referenced registry recursively): it touches just the
-- evaluator-reachable graph, never the whole program pool — so it does
-- NOT intern every stored program the way `EArena.ofArena` would.
-- ─────────────────────────────────────────────────────────────

open Tropical.Ir.Core (CoreProgram CoreInputDecl CoreOutputDecl CoreOutputAssign
  CoreBodyDecl CoreInstanceInput)

private abbrev CheckM := StateT CoreArena (Except String)

private def internM (cn : CNode) : CheckM ExprId := fun a => .ok ((intern cn).run a)

private def checkFail {α} (prog what : String) : CheckM α :=
  fun _ => .error s!"core check ('{prog}'): {what} survived strata"

/-- Validate + intern a reachable tree `Expr` into the arena. Mirrors the
    old `Core.checkExpr`, but the accepted leaves land in the `CoreArena`. -/
private partial def checkInternExpr (progName : String) : Expr → CheckM ExprId
  | .num n          => internM (.num n)
  | .bool b         => internM (.bool b)
  | .arr items      => do internM (.arr (← items.mapM (checkInternExpr progName)))
  | .binary t a b   => do internM (.binary t (← checkInternExpr progName a) (← checkInternExpr progName b))
  | .unary t a      => do internM (.unary t (← checkInternExpr progName a))
  | .clamp a b c    => do internM (.clamp (← checkInternExpr progName a) (← checkInternExpr progName b) (← checkInternExpr progName c))
  | .select a b c   => do internM (.select (← checkInternExpr progName a) (← checkInternExpr progName b) (← checkInternExpr progName c))
  | .arraySet a b c => do internM (.arraySet (← checkInternExpr progName a) (← checkInternExpr progName b) (← checkInternExpr progName c))
  | .index a b      => do internM (.index (← checkInternExpr progName a) (← checkInternExpr progName b))
  | .inputRef i     => internM (.inputRef i)
  | .paramRef i     => internM (.paramRef i)
  | .nestedOut i o  => internM (.nestedOut i o)
  | .sampleRate     => internM .sampleRate
  | .sampleIndex    => internM .sampleIndex
  | .typeParamRef _ => checkFail progName "a typeParamRef (specialize)"
  | .bindingRef _   => checkFail progName "a bindingRef (arrayLower)"
  | .tag ..         => checkFail progName "a tag (sumLower)"
  | .match_ ..      => checkFail progName "a match (sumLower)"
  | .zeros _        => checkFail progName "a zeros (arrayLower)"
  | .fold ..        => checkFail progName "a fold (arrayLower)"
  | .scan ..        => checkFail progName "a scan (arrayLower)"
  | .generate ..    => checkFail progName "a generate (arrayLower)"
  | .iterate ..     => checkFail progName "an iterate (arrayLower)"
  | .chain ..       => checkFail progName "a chain (arrayLower)"
  | .map2 ..        => checkFail progName "a map2 (arrayLower)"
  | .zipWith ..     => checkFail progName "a zipWith (arrayLower)"
  | .letIn ..       => checkFail progName "a let (arrayLower)"

private partial def checkProgram (arena : Arena) (rootIdx : ProgramIdx) : CheckM CoreProgram := do
  let some prog := arena.program? rootIdx
    | fun _ => .error s!"core check: program pool index {rootIdx.idx} out of range"
  unless prog.typeParams.isEmpty do
    checkFail prog.name s!"{prog.typeParams.size} typeParam decl(s) (specialize)"
  let decls ← prog.decls.mapM fun d => do
    match d with
    | .param name value? => pure (CoreBodyDecl.param name value?)
    | .inst name typeKey tArgs inputs =>
      pure (CoreBodyDecl.inst name typeKey tArgs
        (← inputs.mapM fun i => do
          pure { port := i.port, value := ← checkInternExpr prog.name i.value : CoreInstanceInput }))
    | .prog name _ => pure (CoreBodyDecl.progDecl name)
  let assigns ← prog.assigns.mapM fun a => do
    pure { target := a.target, expr := ← checkInternExpr prog.name a.expr : CoreOutputAssign }
  let inputs ← prog.inputs.mapM fun i => do
    pure { name := i.name, type? := Core.resolveOptPortType arena i.type?,
           default? := ← i.default?.mapM (checkInternExpr prog.name) : CoreInputDecl }
  let outputs := prog.outputs.map fun o =>
    { name := o.name, type? := Core.resolveOptPortType arena o.type? : CoreOutputDecl }
  let mut registry : Array (String × CoreProgram) := #[]
  for d in prog.decls do
    if let .inst name typeKey _ _ := d then
      unless registry.any (·.1 == typeKey) do
        let some tIdx := prog.registryGet? typeKey
          | fun _ => .error s!"core check ('{prog.name}'): instance '{name}' typeKey '{typeKey}' missing from registry"
        registry := registry.push (typeKey, ← checkProgram arena tIdx)
  return .mk prog.name inputs outputs decls assigns registry

/-- Downcast an elaborated tree `Arena` (that never ran the strata passes)
    to `(CoreArena × CoreProgram)`, reachable-only. The Phase B analogue of
    `Core.check` for the session-root / per-program compile boundaries. -/
def checkResolvedArena (arena : Arena) (rootIdx : ProgramIdx) :
    Except String (CoreArena × CoreProgram) := do
  let (core, ca) ← (checkProgram arena rootIdx).run {}
  return (ca, core)


end Tropical.Ir
