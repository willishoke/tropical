import Tropical.Ir.Staging
import Tropical.Semantics.Expr

/-!
# Semantic laws for binding-time staging

This module keeps the proof-facing staging contract separate from the
executable classifier.  In particular, `Stage.le` is a Boolean decision
procedure, so the lattice laws below state both its order theory and the
least-upper-bound property implemented by `Stage.join`.
-/

namespace Tropical.Semantics.Staging

open Tropical.Ir
open Tropical.Ir.Staging

theorem stage_le_refl (stage : Stage) : stage.le stage = true := by
  cases stage <;> rfl

private theorem stage_beq_s1 (stage : Stage) :
    (stage == Stage.s1) = true ↔ stage = Stage.s1 := by
  cases stage with
  | fold =>
    constructor <;> intro h <;> cases h
  | s0 =>
    constructor <;> intro h <;> cases h
  | s1 =>
    constructor <;> intro <;> rfl

theorem stage_le_trans {a b c : Stage}
    (hab : a.le b = true) (hbc : b.le c = true) : a.le c = true := by
  cases a <;> cases b <;> cases c <;>
    simp_all [Stage.le, stage_beq_s1]

theorem stage_le_antisymm {a b : Stage}
    (hab : a.le b = true) (hba : b.le a = true) : a = b := by
  cases a <;> cases b <;> simp_all [Stage.le, stage_beq_s1]

theorem stage_join_assoc (a b c : Stage) :
    (a.join b).join c = a.join (b.join c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem stage_join_comm (a b : Stage) : a.join b = b.join a := by
  cases a <;> cases b <;> rfl

theorem stage_join_idem (a : Stage) : a.join a = a := by
  cases a <;> rfl

theorem stage_le_join_left (a b : Stage) : a.le (a.join b) = true := by
  cases a <;> cases b <;> rfl

theorem stage_le_join_right (a b : Stage) : b.le (a.join b) = true := by
  cases a <;> cases b <;> rfl

theorem stage_join_least {a b upper : Stage}
    (ha : a.le upper = true) (hb : b.le upper = true) :
    (a.join b).le upper = true := by
  cases a <;> cases b <;> cases upper <;> simp_all [Stage.le, Stage.join]

/-- Resolve one nested-output dependency.  Missing children and outputs are
    maximally dynamic. -/
def childStage (ctx : StageCtx) (instanceIdx outputIdx : Nat) : Stage :=
  match ctx.childOut[instanceIdx]? with
  | some (some outputs) => outputs[outputIdx]?.getD Stage.s1
  | _ => Stage.s1

/-- Pointwise order on staging contexts.  Missing entries remain maximally
    dynamic, so extending a context with earlier-stage bindings can only lower
    the resolved stage. -/
def StageCtxLe (a b : StageCtx) : Prop :=
  (∀ (i : Nat), (a.inputStages[i]?.getD Stage.s1).le
      (b.inputStages[i]?.getD Stage.s1) = true) ∧
  (∀ (instanceIdx outputIdx : Nat),
    (childStage a instanceIdx outputIdx).le
      (childStage b instanceIdx outputIdx) = true)

/-- A dangling expression id is classified maximally dynamically. -/
theorem stageOf_dangling {arena : ExprArena} {ctx : StageCtx} {id : ExprId}
    (h : arena.sig? id = none) : stageOf arena ctx id = .s1 := by
  simp [stageOf, h]

/-- Any successful signature lookup resolves exactly that stored signature. -/
theorem stageOf_resolves {arena : ExprArena} {ctx : StageCtx} {id : ExprId}
    {sig : StageSig} (h : arena.sig? id = some sig) :
    stageOf arena ctx id = resolve ctx sig := by
  simp [stageOf, h]

/-- The absolute tile clock has exactly the same direct denotation as the
    ordinary sample clock.  TileStage changes the leaf used by a rebuilt DAG,
    not the coordinate carried by the exact environment. -/
theorem denoteExpr_tileSampleIndex (alg : Algebra α) (env : SigEnv α)
    (arena : ExprArena) (hArena : ArenaWellFormed arena) {id : ExprId}
    (hDeref : arena.deref id = some .tileSampleIndex) :
    denoteExpr alg env arena hArena id = .ok env.sampleIndex := by
  rw [denoteExpr_of_deref alg env arena hArena hDeref]
  rfl

/-- Direct/JIT semantics observes the exact left endpoint: `tilePhase` is the
    literal zero.  This deliberately says nothing about nonzero interpolated
    Metal lanes. -/
theorem denoteExpr_tilePhase_zero (alg : Algebra α) (env : SigEnv α)
    (arena : ExprArena) (hArena : ArenaWellFormed arena) {id : ExprId}
    (hDeref : arena.deref id = some .tilePhase) :
    denoteExpr alg env arena hArena id = alg.literal (0 : Nat) := by
  rw [denoteExpr_of_deref alg env arena hArena hDeref]
  rfl

/-- Marking `tilePhase` as per-sample is conservative: its direct expression
    denotation is nevertheless the constant left endpoint. -/
theorem tilePhase_signature_is_conservative :
    (enodeSig #[] .tilePhase).base = .s1 := by
  rfl

end Tropical.Semantics.Staging
