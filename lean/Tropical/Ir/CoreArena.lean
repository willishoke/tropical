import Std.Data.HashMap
import Tropical.Ir.Core

/-!
# CoreArena — hash-consed (DAG) form of the post-strata `CoreExpr`

The native-DAG representation for issue #190 (the first boundary of the
strangler migration). A `CNode` is a `CoreExpr` node whose children are
`ExprId`s into an arena instead of inlined subtrees — so a node is **flat**
(O(1) to hash and compare) and equal subtrees are one arena entry referenced by
many ids. Interning at construction (`intern`) makes duplication impossible: the
bloat a tree representation forces (the modulated clock copied into every
oscillator partial) collapses to a single node.

Soundness: interning merges two nodes iff they have the same constructor and the
same child ids. Every tropical op is pure and deterministic, so a merge never
changes a computed value — the rendered audio is identical (the goldens hash the
audio, not the plan, so register relabeling from a DAG walk is free).

This boundary lives just before emit for now; later strangler steps move it
earlier (before `Core.check`, before `inlineInstances`) until the elaborator
constructs interned nodes directly and the tree `Expr` is gone.
-/

namespace Tropical.Ir

open Lean (JsonNumber)
open Tropical.Ir.Core (CoreExpr)

/-- Dense index into a `CoreArena`. -/
structure ExprId where
  idx : Nat
deriving BEq, Hashable, Repr, Inhabited

/-- A post-strata expression node with children referenced by `ExprId`.
    Mirrors `CoreExpr`'s 14 constructors; flat (no inlined subtrees). -/
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

/-- Lower a tree `CoreExpr` into the arena, interning bottom-up. Each occurrence
    is visited once (O(N_tree)); equal subtrees share one id (O(N_unique) nodes). -/
partial def toArena : CoreExpr → ArenaM ExprId
  | .num n          => intern (.num n)
  | .bool b         => intern (.bool b)
  | .arr items      => do intern (.arr (← items.mapM toArena))
  | .binary t a b   => do intern (.binary t (← toArena a) (← toArena b))
  | .unary t a      => do intern (.unary t (← toArena a))
  | .clamp a b c    => do intern (.clamp (← toArena a) (← toArena b) (← toArena c))
  | .select a b c   => do intern (.select (← toArena a) (← toArena b) (← toArena c))
  | .arraySet a b c => do intern (.arraySet (← toArena a) (← toArena b) (← toArena c))
  | .index a b      => do intern (.index (← toArena a) (← toArena b))
  | .inputRef i     => intern (.inputRef i)
  | .paramRef i     => intern (.paramRef i)
  | .nestedOut i o  => intern (.nestedOut i o)
  | .sampleRate     => intern .sampleRate
  | .sampleIndex    => intern .sampleIndex

/-- Rebuild a tree `CoreExpr` from an id (round-trip witness for tests). -/
partial def CoreArena.toCore (a : CoreArena) (id : ExprId) : CoreExpr :=
  match a.deref id with
  | none => .num 0  -- unreachable for well-formed ids
  | some n => match n with
    | .num x          => .num x
    | .bool b         => .bool b
    | .arr items      => .arr (items.map a.toCore)
    | .binary t l r   => .binary t (a.toCore l) (a.toCore r)
    | .unary t x      => .unary t (a.toCore x)
    | .clamp x y z    => .clamp (a.toCore x) (a.toCore y) (a.toCore z)
    | .select x y z   => .select (a.toCore x) (a.toCore y) (a.toCore z)
    | .arraySet x y z => .arraySet (a.toCore x) (a.toCore y) (a.toCore z)
    | .index x y      => .index (a.toCore x) (a.toCore y)
    | .inputRef i     => .inputRef i
    | .paramRef i     => .paramRef i
    | .nestedOut i o  => .nestedOut i o
    | .sampleRate     => .sampleRate
    | .sampleIndex    => .sampleIndex

end Tropical.Ir
