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

-- The elaborated-arena → `(CoreArena × CoreProgram)` downcast is now just
-- `EArena.toResolved` (the arena is already the id-form). `checkResolvedArena`
-- (a thin `Except String` wrapper for the compile boundaries) lives in
-- `Strata/EArena.lean`, alongside `toResolved`.

end Tropical.Ir
