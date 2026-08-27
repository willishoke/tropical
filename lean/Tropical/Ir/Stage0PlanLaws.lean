import Tropical.Ir.Stage0
import Tropical.Semantics.Plan

/-!
# Stage0 block-order laws

This proof module bridges the production Stage0 linearization to the Plan
reference interpreter without introducing an `Ir.Stage0` dependency into the
semantic waist.
-/

namespace Tropical.Ir.Stage0PlanLaws

open Tropical.Plan
open Tropical.Semantics

/-- Execute an authored array of independently delimited instruction blocks. -/
def execCollectedBlocks (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (blocks : Array (Array NInstr)) :
    Outcome (PlanState α) :=
  List.foldlM (execBlocks alg inputs) state blocks.toList

private def collectedChildBlocks (children : List InstanceFunction) :
    List (Array NInstr) :=
  children.flatMap fun child =>
    child.preInputInstructions :: (Stage0.collectBlocks child).toList

private theorem collectChildrenFold (children : Array InstanceFunction)
    (out : Array (Array NInstr)) :
    (children.foldl (fun blocks child =>
      blocks.push child.preInputInstructions ++ Stage0.collectBlocks child) out).toList =
      out.toList ++ collectedChildBlocks children.toList := by
  cases children with
  | mk children =>
      induction children generalizing out with
      | nil => simp [collectedChildBlocks]
      | cons child rest ih =>
          rw [← Array.foldl_toList]
          simp only [List.foldl_cons]
          rw [Array.foldl_toList]
          rw [ih]
          simp [collectedChildBlocks, List.append_assoc]

theorem collectBlocks_toList (inst : InstanceFunction) :
    (Stage0.collectBlocks inst).toList =
      inst.preambleInstructions ::
        (collectedChildBlocks inst.children.toList ++ [inst.instructions]) := by
  rw [Stage0.collectBlocks.eq_1]
  simp [collectChildrenFold]

private theorem execCollectedChildBlocks
    (alg : Algebra α) (inputs : PlanInputs α)
    (children : List InstanceFunction)
    (hagrees : ∀ child, child ∈ children → ∀ state,
      execCollectedBlocks alg inputs state (Stage0.collectBlocks child) =
        execInstanceFunction alg inputs state child)
    (state : PlanState α) :
    List.foldlM (execBlocks alg inputs) state (collectedChildBlocks children) =
      List.foldlM (fun current child => do
          let current ← execBlocks alg inputs current child.preInputInstructions
          execInstanceFunction alg inputs current child)
        state children := by
  induction children generalizing state with
  | nil => simp [collectedChildBlocks]
  | cons child rest ih =>
      have hchild := hagrees child (by simp)
      have hrest : ∀ nested, nested ∈ rest → ∀ current,
          execCollectedBlocks alg inputs current (Stage0.collectBlocks nested) =
            execInstanceFunction alg inputs current nested := by
        intro nested hnested
        exact hagrees nested (by simp [hnested])
      simp only [collectedChildBlocks, List.flatMap_cons, List.foldlM_append,
        List.foldlM_cons]
      cases hpre : execBlocks alg inputs state child.preInputInstructions with
      | error error => rfl
      | ok current =>
          have hrun := hchild current
          unfold execCollectedBlocks at hrun
          change (List.foldlM (execBlocks alg inputs) current
              (Stage0.collectBlocks child).toList >>= fun next =>
                List.foldlM (execBlocks alg inputs) next
                  (collectedChildBlocks rest)) =
            (execInstanceFunction alg inputs current child >>= fun next =>
              List.foldlM (fun current child => do
                  let current ← execBlocks alg inputs current child.preInputInstructions
                  execInstanceFunction alg inputs current child)
                next rest)
          rw [hrun]
          cases hchildRun : execInstanceFunction alg inputs current child with
          | error error => rfl
          | ok next =>
              exact ih hrest next

private theorem execChildrenForIn
    (alg : Algebra α) (inputs : PlanInputs α)
    (children : Array InstanceFunction) (state : PlanState α) :
    (forIn' children state fun child _h current => do
        let current ← execBlocks alg inputs current child.preInputInstructions
        let current ← execInstanceFunction alg inputs current child
        pure PUnit.unit
        pure (.yield current)) =
      List.foldlM (fun current child => do
          let current ← execBlocks alg inputs current child.preInputInstructions
          execInstanceFunction alg inputs current child)
        state children.toList := by
  simp only [forIn'_eq_forIn]
  have hfold := Array.forIn_yield_eq_foldlM
    (xs := children)
    (fun child current => do
      let current ← execBlocks alg inputs current child.preInputInstructions
      execInstanceFunction alg inputs current child)
    (fun _child _current next => next) state
  simpa using hfold

/-- `Stage0.collectBlocks` linearizes one recursive instance in exactly the
order consumed by `execInstanceFunction`: preamble, then each child's
pre-input block and recursive body, then the parent body. -/
theorem collectBlocks_agrees_with_execution_order
    (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (inst : InstanceFunction) :
    execCollectedBlocks alg inputs state (Stage0.collectBlocks inst) =
      execInstanceFunction alg inputs state inst := by
  fun_induction execInstanceFunction alg inputs state inst with
  | case1 state inst ih =>
      unfold execCollectedBlocks
      rw [collectBlocks_toList]
      simp only [List.foldlM_cons, List.foldlM_append, List.foldlM_nil, bind_pure]
      cases hpreamble : execBlocks alg inputs state inst.preambleInstructions with
      | error error => rfl
      | ok current =>
          have hagrees : ∀ child, child ∈ inst.children.toList → ∀ next,
              execCollectedBlocks alg inputs next (Stage0.collectBlocks child) =
                execInstanceFunction alg inputs next child := by
            intro child hchild next
            exact ih child (by simpa using hchild) next
          change (List.foldlM (execBlocks alg inputs) current
                (collectedChildBlocks inst.children.toList) >>= fun next =>
                  execBlocks alg inputs next inst.instructions) =
            ((forIn' inst.children current fun child _h next => do
                let next ← execBlocks alg inputs next child.preInputInstructions
                let next ← execInstanceFunction alg inputs next child
                pure PUnit.unit
                pure (.yield next)) >>= fun next =>
              execBlocks alg inputs next inst.instructions)
          rw [execCollectedChildBlocks alg inputs inst.children.toList hagrees current]
          rw [execChildrenForIn alg inputs inst.children current]

end Tropical.Ir.Stage0PlanLaws
