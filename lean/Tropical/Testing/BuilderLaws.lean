import Tropical.EmitArrow.BuilderLaws

/-!
# Compile-time fixtures for builder preservation

These examples exercise the public qualified laws.  The foreign-id fixture is
intentionally about addressability: `Sig = ExprId` carries no nominal arena
token, so a numerically colliding id is indistinguishable by design.
-/

namespace Tropical.Testing.BuilderLaws

open Tropical.Ir
open Tropical.Semantics
open Tropical.EmitArrow

private def zero : Lean.JsonNumber := ⟨0, 0⟩
private def one : Lean.JsonNumber := ⟨1, 0⟩

def oneNodeBuilder : Builder :=
  { exprs :=
    { nodes := #[.num zero]
      dedup := {}
      sigs := #[{}] } }

def twoNodeForeignBuilder : Builder :=
  { exprs :=
    { nodes := #[.num zero, .num one]
      dedup := {}
      sigs := #[{}, {}] } }

theorem oneNodeBuilder_wellFormed : BuilderWellFormed oneNodeBuilder := by
  constructor
  · constructor
    · apply childrenDescend_of_wf
      native_decide
    · intro node id h
      simp [oneNodeBuilder] at h
    · rfl
  · simp [BuilderDeclsWellFormed, oneNodeBuilder]

theorem twoNodeForeignBuilder_wellFormed :
    BuilderWellFormed twoNodeForeignBuilder := by
  constructor
  · constructor
    · apply childrenDescend_of_wf
      native_decide
    · intro node id h
      simp [twoNodeForeignBuilder] at h
    · rfl
  · simp [BuilderDeclsWellFormed, twoNodeForeignBuilder]

def localRoot : Sig := ⟨0⟩
def foreignOnlyRoot : Sig := ⟨1⟩
def danglingRoot : Sig := ⟨7⟩

example : SigIn oneNodeBuilder localRoot := by
  simp [SigIn, ExprIdIn, oneNodeBuilder, localRoot]

example : ¬ SigIn oneNodeBuilder danglingRoot := by
  simp [SigIn, ExprIdIn, oneNodeBuilder, danglingRoot]

/-- The id is owned by the foreign arena but rejected by the smaller local one. -/
example : SigIn twoNodeForeignBuilder foreignOnlyRoot ∧
    ¬ SigIn oneNodeBuilder foreignOnlyRoot := by
  simp [SigIn, ExprIdIn, twoNodeForeignBuilder, oneNodeBuilder,
    foreignOnlyRoot]

example : SigBuildResultWellFormed twoNodeForeignBuilder
    ((add localRoot foreignOnlyRoot).run twoNodeForeignBuilder) := by
  exact add_preserves twoNodeForeignBuilder_wellFormed
    (by simp [SigIn, ExprIdIn, twoNodeForeignBuilder, localRoot])
    (by simp [SigIn, ExprIdIn, twoNodeForeignBuilder, foreignOnlyRoot])

example : ProducesSig (nestedOut ⟨0⟩ ⟨0⟩) :=
  nestedOut_preserves ⟨0⟩ ⟨0⟩

example : SigBuildResultWellFormed twoNodeForeignBuilder
    ((bankSum 2 #[localRoot] foreignOnlyRoot none 3).run
      twoNodeForeignBuilder) := by
  apply bankSum_preserves twoNodeForeignBuilder_wellFormed 2 3
  · intro id h
    simp only [Array.mem_singleton] at h
    subst id
    simp [SigIn, ExprIdIn, twoNodeForeignBuilder, localRoot]
  · simp [SigIn, ExprIdIn, twoNodeForeignBuilder, foreignOnlyRoot]
  · simp [OptionalSigIn]

example : SigBuildResultWellFormed twoNodeForeignBuilder
    ((routedSum 2 1 #[some 0, none] #[localRoot]
      #[foreignOnlyRoot] none 4).run twoNodeForeignBuilder) := by
  apply routedSum_preserves twoNodeForeignBuilder_wellFormed 2 1
    #[some 0, none] 4
  · intro id h
    simp only [Array.mem_singleton] at h
    subst id
    simp [SigIn, ExprIdIn, twoNodeForeignBuilder, localRoot]
  · intro id h
    simp only [Array.mem_singleton] at h
    subst id
    simp [SigIn, ExprIdIn, twoNodeForeignBuilder, foreignOnlyRoot]
  · simp [OptionalSigIn]

example : SigBuildResultWellFormed twoNodeForeignBuilder
    ((tileArray #[localRoot, foreignOnlyRoot]).run twoNodeForeignBuilder) := by
  apply tileArray_preserves twoNodeForeignBuilder_wellFormed
  intro id h
  simp at h
  rcases h with rfl | rfl
  · simp [SigIn, ExprIdIn, twoNodeForeignBuilder, localRoot]
  · simp [SigIn, ExprIdIn, twoNodeForeignBuilder, foreignOnlyRoot]

example : ProducesSig tilePhase := tilePhase_preserves
example : ProducesSig tileSampleIndex := tileSampleIndex_preserves

example : ShiftResultWellFormed twoNodeForeignBuilder
    #[localRoot, foreignOnlyRoot]
    ((shiftSampleIndex #[localRoot, foreignOnlyRoot] 64).run
      twoNodeForeignBuilder) := by
  apply shiftSampleIndex_preserves twoNodeForeignBuilder_wellFormed
  intro id h
  simp at h
  rcases h with rfl | rfl
  · simp [SigIn, ExprIdIn, twoNodeForeignBuilder, localRoot]
  · simp [SigIn, ExprIdIn, twoNodeForeignBuilder, foreignOnlyRoot]

example : match
    (inst "child" "Child" #[{ port := ⟨0⟩, value := localRoot }]).run
      oneNodeBuilder with
  | .error _ => True
  | .ok (idx, after) => BuilderWellFormed after ∧
    BuilderExtends oneNodeBuilder after ∧ idx.idx < after.decls.size := by
  exact inst_preserves oneNodeBuilder_wellFormed "child" "Child"
    #[{ port := ⟨0⟩, value := localRoot }] (by
      intro input h
      simp only [Array.mem_singleton] at h
      subst input
      simp [SigIn, ExprIdIn, oneNodeBuilder, localRoot])

end Tropical.Testing.BuilderLaws
