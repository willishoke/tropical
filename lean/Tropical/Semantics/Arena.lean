import Tropical.Semantics.Sig
import Std.Data.HashMap.Lemmas

/-!
# Semantic invariants of the production expression arena

`ExprArena` stores a child-descending expression DAG together with a hash-cons
index.  The production representation keeps those invariants computational;
this module exposes the propositions needed by denotational proofs and proves
that a qualified `eintern` step preserves them.
-/

namespace Tropical.Semantics

open Tropical.Ir

/-- Every child of a stored node has a strictly smaller arena index. -/
def ChildrenDescend (arena : ExprArena) : Prop :=
  ∀ ⦃id node⦄, arena.deref id = some node →
    ∀ child ∈ node.children, child.idx < id.idx

/-- Every child referenced by a node already belongs to the frozen prefix. -/
def ChildrenInPrefix (arena : ExprArena) (node : ENode) : Prop :=
  ∀ child ∈ node.children, child.idx < arena.nodes.size

/-- A hash-cons hit names the node used as its key. -/
def DedupSound (arena : ExprArena) : Prop :=
  ∀ node id, arena.dedup.get? node = some id → arena.deref id = some node

/-- The invariants required by the semantic evaluator and qualified interning. -/
structure ArenaWellFormed (arena : ExprArena) : Prop where
  childrenDescend : ChildrenDescend arena
  dedupSound : DedupSound arena
  signaturesAligned : arena.sigs.size = arena.nodes.size

/-- `after` retains every node addressable in `before`. -/
def Extends (before after : ExprArena) : Prop :=
  ∀ ⦃id node⦄, before.deref id = some node → after.deref id = some node

theorem Extends.refl (arena : ExprArena) : Extends arena arena :=
  by
    intro id node h
    exact h

theorem Extends.trans {a b c : ExprArena} (hab : Extends a b)
    (hbc : Extends b c) : Extends a c :=
  by
    intro id node h
    exact hbc (hab h)

theorem deref_index_lt {arena : ExprArena} {id : ExprId} {node : ENode}
    (h : arena.deref id = some node) : id.idx < arena.nodes.size := by
  exact (Array.getElem?_eq_some_iff.mp h).choose

theorem deref_of_index_lt {arena : ExprArena} {id : ExprId}
    (h : id.idx < arena.nodes.size) :
    ∃ node, arena.deref id = some node := by
  exact ⟨arena.nodes[id.idx], Array.getElem?_eq_getElem h⟩

theorem emptyArena_wellFormed : ArenaWellFormed ({} : ExprArena) := by
  constructor
  · intro id node h
    simp [ExprArena.deref] at h
  · intro node id h
    simp at h
  · rfl

private theorem deref_push_old {arena : ExprArena} {node oldNode : ENode}
    {id : ExprId} (h : arena.deref id = some oldNode) :
    ({ arena with nodes := arena.nodes.push node }).deref id = some oldNode := by
  have hi := deref_index_lt h
  rw [ExprArena.deref] at h ⊢
  calc
    (arena.nodes.push node)[id.idx]? = arena.nodes[id.idx]? := by
      rw [Array.getElem?_push_lt hi, Array.getElem?_eq_getElem hi]
    _ = some oldNode := h

/-- The executable behavior of one production interning step. -/
theorem eintern_run (arena : ExprArena) (node : ENode) :
    eintern node arena =
      match arena.dedup.get? node with
      | some id => (id, arena)
      | none =>
        let id : ExprId := ⟨arena.nodes.size⟩
        (id, {
          arena with
          nodes := arena.nodes.push node
          dedup := arena.dedup.insert node id
          sigs := arena.sigs.push (enodeSig arena.sigs node) }) := by
  simp only [eintern, StateT.bind, bind, get, set, getThe,
    MonadStateOf.get, StateT.get]
  split <;>
    simp_all [StateT.bind, StateT.set, StateT.pure, pure, Pure.pure,
      Bind.bind]

/-- Qualified interning preserves all semantic arena invariants, extends the
    old node prefix, and returns an id that dereferences to the requested node. -/
theorem eintern_preserves {arena : ExprArena} {node : ENode}
    (hArena : ArenaWellFormed arena)
    (hChildren : ChildrenInPrefix arena node) :
    let result := eintern node arena
    ArenaWellFormed result.2 ∧
      Extends arena result.2 ∧
      result.2.deref result.1 = some node := by
  rw [eintern_run]
  simp only
  split
  next id hHit =>
    exact ⟨hArena, Extends.refl arena, hArena.dedupSound node id hHit⟩
  next hMiss =>
    let id : ExprId := ⟨arena.nodes.size⟩
    let arena' : ExprArena := {
      arena with
      nodes := arena.nodes.push node
      dedup := arena.dedup.insert node id
      sigs := arena.sigs.push (enodeSig arena.sigs node) }
    have hNewDeref : arena'.deref id = some node := by
      simp [arena', id, ExprArena.deref]
    have hExtends : Extends arena arena' := by
      intro oldId oldNode hOld
      exact deref_push_old hOld
    have hDescend : ChildrenDescend arena' := by
      intro queryId queryNode hQuery child hChild
      by_cases hq : queryId.idx = arena.nodes.size
      · have hId : queryId = id := by
          cases queryId
          simp_all [id]
        rw [hId] at hQuery
        have hNode : queryNode = node := by
          rw [hNewDeref] at hQuery
          exact Option.some.inj hQuery.symm
        subst hNode
        simpa [hId, id] using hChildren child hChild
      · have hOldBound : queryId.idx < arena.nodes.size := by
          have hi := deref_index_lt hQuery
          simp [arena'] at hi
          omega
        have hOldQuery : arena.deref queryId = some queryNode := by
          change (arena.nodes.push node)[queryId.idx]? =
            some queryNode at hQuery
          rw [Array.getElem?_push_lt hOldBound] at hQuery
          rw [ExprArena.deref, Array.getElem?_eq_getElem hOldBound]
          exact hQuery
        exact hArena.childrenDescend hOldQuery child hChild
    have hDedup : DedupSound arena' := by
      intro queryNode queryId hQuery
      rw [Std.HashMap.get?_eq_getElem?,
        Std.HashMap.getElem?_insert] at hQuery
      split at hQuery
      next hEq =>
        have hNodeEq : node = queryNode := eq_of_beq hEq
        subst queryNode
        have hIdEq : queryId = id := Option.some.inj hQuery.symm
        simpa [hIdEq] using hNewDeref
      next hNe =>
        exact hExtends (hArena.dedupSound queryNode queryId hQuery)
    have hSigs : arena'.sigs.size = arena'.nodes.size := by
      simp [arena', hArena.signaturesAligned]
    exact ⟨⟨hDescend, hDedup, hSigs⟩, hExtends, hNewDeref⟩

end Tropical.Semantics
