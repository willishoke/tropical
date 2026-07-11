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
  /-- Iteration index inside the enclosing `bankSum` region (→ `NOperand.loopIdx`). -/
  | loopIdx
  /-- An indexed reduction `Σ_{k<count} body(k)`, i64-modular. `tables` are the
      loop-invariant coefficient columns the body indexes at `loopIdx`; emit
      materializes them once before the `ReduceBegin`/`ReduceEnd` region.
      `count` is the static CAPACITY; `dynCount?` (trip-count-as-data) is an
      optional runtime effective count, clamped to `[0, count]` at the loop
      head by the emitters (`none` = today's static path). -/
  | bankSum (count : Nat) (tables : Array ExprId) (body : ExprId)
      (dynCount? : Option ExprId := none)
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
  | .loopIdx        => 15
  | .bankSum c ts b dc => mixHash (mixHash (mixHash (mixHash 16 (hash c)) (hash (ts.map (·.idx)))) (hash b.idx)) (hash (dc.map (·.idx)))

instance : Hashable CNode := ⟨cnodeHash⟩

-- ─────────────────────────────────────────────────────────────
-- Binding-time stage — computed at intern, "inference you do once"
-- ─────────────────────────────────────────────────────────────

/-- Binding time of a value, ordered `fold < s0 < s1`:

    - `fold` — const/rate-only. Bound at elaboration; both emitters
      constant-fold it in f64, so it must never be demoted to a slot
      crossing (the Metal-precision rule the plan-level pass learned
      at ~109 dB).
    - `s0` — τ-independent but control-derived (params): bound at
      control-write time; the stage-0 coefficient kernel's territory.
    - `s1` — per-sample (τ below). -/
inductive Stage where
  | fold | s0 | s1
deriving BEq, Repr, Inhabited

def Stage.join : Stage → Stage → Stage
  | .s1, _ | _, .s1 => .s1
  | .s0, _ | _, .s0 => .s0
  | .fold, .fold => .fold

/-- `fold ≤ s0 ≤ s1` — "bound no later than". -/
def Stage.le : Stage → Stage → Bool
  | .fold, _ => true
  | .s0, .fold => false
  | .s0, _ => true
  | .s1, s => s == .s1

/-- The stage *signature* of a node: its intrinsic stage joined with
    symbolic dependencies on the two parametric leaves — input ports
    (bound per instance by the wiring) and nested-instance outputs
    (bound by the child program's own signature). Resolution against a
    binding context happens in `Staging`; the signature itself is a
    birth attribute, computed once at `intern` from the (already
    interned) children — never re-derived by a later walk. -/
structure StageSig where
  base : Stage := .fold
  /-- `InputIdx.idx` deps, strictly ascending. -/
  inputs : Array Nat := #[]
  /-- `(InstanceIdx.idx, OutputIdx.idx)` deps, strictly ascending. -/
  nested : Array (Nat × Nat) := #[]
deriving BEq, Repr, Inhabited

/-- Merge two strictly-ascending dep arrays (dedup). -/
private def mergeAsc {α : Type} [Ord α] [Inhabited α] (a b : Array α) : Array α := Id.run do
  if a.isEmpty then return b
  if b.isEmpty then return a
  let mut out : Array α := #[]
  let mut i := 0
  let mut j := 0
  while i < a.size && j < b.size do
    match compare a[i]! b[j]! with
    | .lt => out := out.push a[i]!; i := i + 1
    | .gt => out := out.push b[j]!; j := j + 1
    | .eq => out := out.push a[i]!; i := i + 1; j := j + 1
  while i < a.size do out := out.push a[i]!; i := i + 1
  while j < b.size do out := out.push b[j]!; j := j + 1
  return out

private instance : Ord (Nat × Nat) := ⟨fun a b =>
  match compare a.1 b.1 with
  | .eq => compare a.2 b.2
  | o => o⟩

def StageSig.join (a b : StageSig) : StageSig :=
  { base := a.base.join b.base
    inputs := mergeAsc a.inputs b.inputs
    nested := mergeAsc a.nested b.nested }

/-- Interned node store: `nodes[id]` is the node; `dedup` maps a node back to its
    id so equal nodes collapse; `sigs[id]` is the node's stage signature,
    computed at intern (children precede parents, so the join is O(children)).
    Append-only; ids are assigned in first-seen order. -/
structure CoreArena where
  nodes : Array CNode := #[]
  dedup : Std.HashMap CNode ExprId := {}
  sigs : Array StageSig := #[]
deriving Inhabited

/-- A child's signature during intern (parents intern after children). -/
private def sigAt (sigs : Array StageSig) (id : ExprId) : StageSig :=
  sigs[id.idx]?.getD { base := .s1 }

/-- The stage signature of a node given its children's (leaf rules +
    join). `arr`/`arraySet`/`index` join like everything else — a stage
    is a property of the VALUE; whether an array-valued node is
    *hoistable* is the residualizer's placement decision, not the
    attribute's. -/
def cnodeSig (sigs : Array StageSig) : CNode → StageSig
  | .num _ | .bool _ | .sampleRate => { base := .fold }
  | .sampleIndex => { base := .s1 }
  -- `loopIdx` is the join IDENTITY (`fold`), not `s1`: as a VALUE attribute,
  -- the iteration index is defined by the enclosing reduce region, so it
  -- contributes no binding time of its own. This is what lets a `bankSum`
  -- whose tables/body/count are all s0 BE an s0 value (the whole loop runs
  -- in the coefficient kernel at control-write time — region-aware Stage0);
  -- pinning the leaf s1 would drag every region to the audio kernel forever.
  -- SAFETY: the relaxation is sound only because per-instruction PLACEMENT
  -- never trusts the attribute alone — `Stage0.overlayS1` pins every
  -- `loopIdx`-reading instruction (and both delimiters) to s1 for INDIVIDUAL
  -- moves, and `placementFromStages`' availability walk keeps everything
  -- downstream of a pinned instruction in place (a loop-dependent temp's
  -- reaching def is pinned, so its readers fail availability). Loop-dependent
  -- code therefore leaves the audio kernel only as a WHOLE delimiter-matched
  -- region (the separate `tryRegion` decision), never one instruction at a
  -- time.
  | .loopIdx => { base := .fold }
  | .paramRef _ => { base := .s0 }
  | .inputRef i => { base := .fold, inputs := #[i.idx] }
  | .nestedOut i o => { base := .fold, nested := #[(i.idx, o.idx)] }
  | .arr items => items.foldl (fun acc id => acc.join (sigAt sigs id)) { base := .fold }
  | .binary _ a b => (sigAt sigs a).join (sigAt sigs b)
  | .unary _ a => sigAt sigs a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    ((sigAt sigs a).join (sigAt sigs b)).join (sigAt sigs c)
  | .index a b => (sigAt sigs a).join (sigAt sigs b)
  | .bankSum _ ts b dc =>
    let s := (ts.foldl (fun acc id => acc.join (sigAt sigs id)) ({ base := .fold } : StageSig)).join (sigAt sigs b)
    match dc with
    | some d => s.join (sigAt sigs d)
    | none => s

abbrev ArenaM := StateM CoreArena

/-- Intern a flat node, returning its (shared) id. O(1) (+O(children)
    for the stage signature on first intern). -/
def intern (n : CNode) : ArenaM ExprId := do
  let a ← get
  match a.dedup.get? n with
  | some id => pure id
  | none =>
    let id : ExprId := ⟨a.nodes.size⟩
    set { a with nodes := a.nodes.push n, dedup := a.dedup.insert n id,
                 sigs := a.sigs.push (cnodeSig a.sigs n) }
    pure id

def CoreArena.sig? (a : CoreArena) (id : ExprId) : Option StageSig :=
  a.sigs[id.idx]?

/-- Dereference an id to its node. -/
def CoreArena.deref (a : CoreArena) (id : ExprId) : Option CNode :=
  a.nodes[id.idx]?

-- The elaborated-arena → `(CoreArena × CoreProgram)` downcast is now just
-- `EArena.toResolved` (the arena is already the id-form). `checkResolvedArena`
-- (a thin `Except String` wrapper for the compile boundaries) lives in
-- `Strata/EArena.lean`, alongside `toResolved`.

end Tropical.Ir
