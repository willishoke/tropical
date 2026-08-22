import Tropical.Semantics.Staging

/-! Executable witnesses for the staging proof surface. -/

namespace Tropical.Testing.StagingLaws

open Tropical.Ir
open Tropical.Ir.Staging
open Tropical.Semantics.Staging

example : Stage.fold.join .s0 = .s0 := by rfl
example : Stage.s0.join .s1 = .s1 := by rfl
example : Stage.s0.le Stage.s1 = true := by rfl
example : Stage.s1.le Stage.s0 = false := by rfl

example (a b c : Stage) : (a.join b).join c = a.join (b.join c) :=
  stage_join_assoc a b c

example (arena : ExprArena) (id : ExprId)
    (h : arena.sig? id = none) :
    stageOf arena {} id = .s1 :=
  stageOf_dangling h

end Tropical.Testing.StagingLaws
