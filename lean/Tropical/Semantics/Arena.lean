import Tropical.Semantics.Sig

/-!
# Arena-side contract for the first semantic slice

The production `ExprArena.wf` predicate checks descending child ids, but the
current IR does not expose a proposition connecting `dedup` hits to `nodes`
lookups.  That missing invariant blocks a sound total arena evaluator over an
arbitrary admitted arena: a malformed dedup map can make `eintern` return an id
whose node is unrelated to the requested node.

This sprint therefore takes the handoff's approved relational fallback in
`LowerSig`.  The definitions here name the production invariant the later
functional evaluator must receive rather than silently assuming it.
-/

namespace Tropical.Semantics

open Tropical.Ir

/-- The already-checked descending-edge condition exposed by production. -/
def ChildrenDescend (arena : ExprArena) : Prop :=
  arena.wf = true

/-- The local premise under which appending `node` preserves descending
    children: every referenced child already belongs to the frozen arena
    prefix.  An arbitrary `ENode` does not satisfy this premise. -/
def ChildrenInPrefix (arena : ExprArena) (node : ENode) : Prop :=
  ∀ child ∈ node.children, child.idx < arena.nodes.size

/-- The missing hash-cons soundness invariant.  It is data about the production
    `ExprArena`, not a second IR. -/
def DedupSound (arena : ExprArena) : Prop :=
  ∀ node id, arena.dedup.get? node = some id → arena.deref id = some node

/-- Exact arena precondition required by the future functional evaluator. -/
structure ArenaWellFormed (arena : ExprArena) : Prop where
  childrenDescend : ChildrenDescend arena
  dedupSound : DedupSound arena
  signaturesAligned : arena.sigs.size = arena.nodes.size

end Tropical.Semantics
