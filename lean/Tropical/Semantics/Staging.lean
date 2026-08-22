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

/-- Strict lexicographic order used by nested-output dependency arrays. -/
def NestedDependencyLt (a b : Nat × Nat) : Prop :=
  a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

/-- The representation invariant promised by `StageSig`: dependency arrays
    are strictly ascending and contain no duplicates.  `Nodup` is stated
    explicitly even though it follows mathematically from strict order, so
    downstream audits need not rely on that derived fact. -/
structure StageSigShape (sig : StageSig) : Prop where
  inputsAscending : sig.inputs.toList.Pairwise (· < ·)
  inputsDeduplicated : sig.inputs.toList.Nodup
  nestedAscending : sig.nested.toList.Pairwise NestedDependencyLt
  nestedDeduplicated : sig.nested.toList.Nodup

/-- Every stored signature has the representation invariant.  This is
    intentionally separate from `ArenaWellFormed`: arena termination/dedup
    soundness and staging classification are distinct proof obligations. -/
def SignaturesShaped (arena : ExprArena) : Prop :=
  ∀ sig ∈ arena.sigs, StageSigShape sig

theorem empty_signaturesShaped : SignaturesShaped ({} : ExprArena) := by
  intro sig h
  simp at h

theorem signaturesShaped_push {arena : ExprArena} {sig : StageSig}
    (hArena : SignaturesShaped arena) (hSig : StageSigShape sig) :
    SignaturesShaped { arena with sigs := arena.sigs.push sig } := by
  intro query hQuery
  rw [Array.mem_push] at hQuery
  rcases hQuery with hOld | rfl
  · exact hArena query hOld
  · exact hSig

/-- Qualified interning preserves the signature-shape invariant.  The
    qualification is local to the newly computed signature; hash-cons hits
    retain the already-shaped arena unchanged. -/
theorem eintern_preserves_signaturesShaped {arena : ExprArena} {node : ENode}
    (hArena : SignaturesShaped arena)
    (hNode : StageSigShape (enodeSig arena.sigs node)) :
    SignaturesShaped (eintern node arena).2 := by
  rw [Tropical.Semantics.eintern_run]
  split
  · exact hArena
  · exact signaturesShaped_push hArena hNode

theorem stageSigShape_fold :
    StageSigShape ({ base := .fold } : StageSig) := by
  constructor <;> simp

theorem stageSigShape_s1 :
    StageSigShape ({ base := .s1 } : StageSig) := by
  constructor <;> simp

theorem stageSigShape_input (inputIdx : Nat) :
    StageSigShape ({ base := .fold, inputs := #[inputIdx] } : StageSig) := by
  constructor <;> simp

theorem stageSigShape_nested (instanceIdx outputIdx : Nat) :
    StageSigShape
      ({ base := .fold, nested := #[(instanceIdx, outputIdx)] } : StageSig) := by
  constructor <;> simp

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

/-- Semantic environment agreement through one binding time.  Rate and open
    loop binders are structural/fold inputs.  Control parameters become fixed
    at `s0`; the sample clock only becomes fixed at `s1`.  Symbolic input and
    nested-output dependencies agree exactly when their resolved stage is no
    later than the requested boundary.  Agreement is phrased through the
    production lookup operations, preserving refusal behavior as well as
    successful values. -/
def EnvAgreesThrough (stage : Stage) (ctx : StageCtx)
    (a b : SigEnv α) : Prop :=
  a.sampleRate = b.sampleRate ∧
  a.loops = b.loops ∧
  (Stage.s0.le stage = true →
    ∀ i, lookupValue "paramRef" a.params i =
      lookupValue "paramRef" b.params i) ∧
  (Stage.s1.le stage = true → a.sampleIndex = b.sampleIndex) ∧
  (∀ i, (ctx.inputStages[i]?.getD Stage.s1).le stage = true →
    lookupValue "inputRef" a.inputs i =
      lookupValue "inputRef" b.inputs i) ∧
  (∀ instanceIdx outputIdx,
    (childStage ctx instanceIdx outputIdx).le stage = true →
    lookupNested a instanceIdx outputIdx =
      lookupNested b instanceIdx outputIdx)

theorem EnvAgreesThrough.sampleRate {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b) :
    a.sampleRate = b.sampleRate := h.1

theorem EnvAgreesThrough.loops {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b) :
    a.loops = b.loops := h.2.1

theorem EnvAgreesThrough.params {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (hstage : Stage.s0.le stage = true) (i : Nat) :
    lookupValue "paramRef" a.params i = lookupValue "paramRef" b.params i :=
  h.2.2.1 hstage i

theorem EnvAgreesThrough.sampleIndex {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (hstage : Stage.s1.le stage = true) : a.sampleIndex = b.sampleIndex :=
  h.2.2.2.1 hstage

theorem EnvAgreesThrough.input {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b) (i : Nat)
    (hstage : (ctx.inputStages[i]?.getD Stage.s1).le stage = true) :
    lookupValue "inputRef" a.inputs i = lookupValue "inputRef" b.inputs i :=
  h.2.2.2.2.1 i hstage

theorem EnvAgreesThrough.nested {stage : Stage} {ctx : StageCtx}
    {a b : SigEnv α} (h : EnvAgreesThrough stage ctx a b)
    (instanceIdx outputIdx : Nat)
    (hstage : (childStage ctx instanceIdx outputIdx).le stage = true) :
    lookupNested a instanceIdx outputIdx = lookupNested b instanceIdx outputIdx :=
  h.2.2.2.2.2 instanceIdx outputIdx hstage

/-- Pointwise order on staging contexts.  Missing entries remain maximally
    dynamic, so extending a context with earlier-stage bindings can only lower
    the resolved stage. -/
def StageCtxLe (a b : StageCtx) : Prop :=
  (∀ (i : Nat), (a.inputStages[i]?.getD Stage.s1).le
      (b.inputStages[i]?.getD Stage.s1) = true) ∧
  (∀ (instanceIdx outputIdx : Nat),
    (childStage a instanceIdx outputIdx).le
      (childStage b instanceIdx outputIdx) = true)

theorem resolve_le_s1 (ctx : StageCtx) (sig : StageSig) :
    (resolve ctx sig).le .s1 = true := by
  cases h : resolve ctx sig <;> rfl

theorem stageOf_le_s1 (arena : ExprArena) (ctx : StageCtx) (id : ExprId) :
    (stageOf arena ctx id).le .s1 = true := by
  cases h : stageOf arena ctx id <;> rfl

/-- A dangling expression id is classified maximally dynamically. -/
theorem stageOf_dangling {arena : ExprArena} {ctx : StageCtx} {id : ExprId}
    (h : arena.sig? id = none) : stageOf arena ctx id = .s1 := by
  simp [stageOf, h]

/-- Any successful signature lookup resolves exactly that stored signature. -/
theorem stageOf_resolves {arena : ExprArena} {ctx : StageCtx} {id : ExprId}
    {sig : StageSig} (h : arena.sig? id = some sig) :
    stageOf arena ctx id = resolve ctx sig := by
  simp [stageOf, h]

theorem stageSig_sound_sampleRate (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .sampleRate)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact congrArg Except.ok henv.sampleRate

theorem stageSig_sound_paramRef (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} {paramIdx : Nat}
    (hDeref : arena.deref id = some (.paramRef ⟨paramIdx⟩))
    (hstage : Stage.s0.le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact henv.params hstage paramIdx

theorem stageSig_sound_inputRef (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} {inputIdx : Nat}
    (hDeref : arena.deref id = some (.inputRef ⟨inputIdx⟩))
    (hstage : (ctx.inputStages[inputIdx]?.getD Stage.s1).le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact henv.input inputIdx hstage

theorem stageSig_sound_nestedOut (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} {instanceIdx outputIdx : Nat}
    (hDeref : arena.deref id =
      some (.nestedOut ⟨instanceIdx⟩ ⟨outputIdx⟩))
    (hstage : (childStage ctx instanceIdx outputIdx).le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact henv.nested instanceIdx outputIdx hstage

theorem stageSig_sound_sampleIndex (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .sampleIndex)
    (hstage : Stage.s1.le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact congrArg Except.ok (henv.sampleIndex hstage)

theorem stageSig_sound_tileSampleIndex (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .tileSampleIndex)
    (hstage : Stage.s1.le stage = true)
    (henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  exact congrArg Except.ok (henv.sampleIndex hstage)

theorem stageSig_sound_tilePhase (alg : Algebra α) (ctx : StageCtx)
    (a b : SigEnv α) (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {id : ExprId} (hDeref : arena.deref id = some .tilePhase)
    (_henv : EnvAgreesThrough stage ctx a b) :
    denoteExpr alg a arena hArena id = denoteExpr alg b arena hArena id := by
  rw [denoteExpr_of_deref alg a arena hArena hDeref,
    denoteExpr_of_deref alg b arena hArena hDeref]
  rfl

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
