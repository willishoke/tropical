import Tropical.Semantics.ShiftSampleIndex

/-!
# Production-shaped absolute-clock substitution fixture

The frozen DAG includes shared scalar nodes, an ordinary array, a bank, a
routed bank, and pre-existing tile-clock/tile-phase leaves. The fixture uses
the public production rewrite and instantiates its semantic capstone at a
nonzero frame offset.
-/

namespace Tropical.Testing.ShiftSampleIndex

open Tropical.Ir
open Tropical.Semantics
open Tropical.EmitArrow

private def fixtureArena : ExprArena := {
  nodes := #[
    .sampleIndex,
    .tileSampleIndex,
    .tilePhase,
    .num ⟨1, 0⟩,
    .binary .add ⟨0⟩ ⟨3⟩,
    .arr #[⟨4⟩, ⟨1⟩, ⟨2⟩],
    .loopIdx 7,
    .index ⟨5⟩ ⟨6⟩,
    .bankSum 2 #[⟨5⟩] ⟨7⟩ none 7,
    .routedSum 2 1 #[some 0, some 0] #[⟨5⟩] #[⟨4⟩] none 9,
    .arr #[⟨4⟩, ⟨1⟩, ⟨2⟩, ⟨8⟩, ⟨9⟩]]
  sigs := Array.replicate 11 {}
}

private def fixtureBuilder : Builder := { exprs := fixtureArena }
private def fixtureRoots : Array ExprId := #[⟨10⟩]

private theorem fixtureBuilder_wellFormed :
    BuilderWellFormed fixtureBuilder := by
  constructor
  · constructor
    · apply childrenDescend_of_wf
      native_decide
    · simp [DedupSound, fixtureBuilder, fixtureArena]
    · simp [fixtureBuilder, fixtureArena]
  · simp [BuilderDeclsWellFormed, fixtureBuilder]

private theorem fixtureRoots_owned : SigsIn fixtureBuilder fixtureRoots := by
  simp [SigsIn, SigIn, ExprIdIn, fixtureBuilder, fixtureRoots, fixtureArena]

private def fixtureAlgebra : Algebra Int where
  literal := fun number => .ok (.scalar number.mantissa)
  unary := fun tag value => match tag, value with
    | .toInt, .scalar n => .ok (.scalar n)
    | _, _ => refusal "shift-fixture.unary" "unsupported operation"
  binary := fun tag lhs rhs => match tag, lhs, rhs with
    | .add, .scalar a, .scalar b => .ok (.scalar (a + b))
    | _, _, _ => refusal "shift-fixture.binary" "unsupported operation"
  clamp := fun _ _ _ => refusal "shift-fixture.clamp" "outside fixture"
  select := fun _ _ _ => refusal "shift-fixture.select" "outside fixture"
  index := fun array index => match array, index with
    | .array values, .scalar i =>
      if i < 0 then refusal "shift-fixture.index" "negative index"
      else lookupValue "shift-fixture.index" values i.toNat
    | _, _ => refusal "shift-fixture.index" "ill-typed index"
  loopIndex := fun index => .ok (.scalar (Int.ofNat index))
  dynamicCount := fun
    | .scalar value => .ok value
    | .array _ => .error {
        operation := "shift-fixture.count", detail := "array count" }
  zero := .ok (.scalar 0)

private def sourceEnv : SigEnv Int := {
  sampleRate := .scalar 48000
  sampleIndex := .scalar 104
  tileSampleIndex := .scalar 100
}

private def shiftedEnv : SigEnv Int := {
  sampleRate := .scalar 48000
  sampleIndex := .scalar 0
  tileSampleIndex := .scalar 100
}

private theorem fixtureEnvs_agree :
    AgreesOutsideSampleIndex sourceEnv shiftedEnv := by
  constructor <;> rfl

example {shiftedRoots : Array ExprId} {after : Builder}
    (hRun : (shiftSampleIndex fixtureRoots 4).run fixtureBuilder =
      .ok (shiftedRoots, after)) :
    ∀ (i : Nat) (root shiftedRoot : ExprId),
      fixtureRoots[i]? = some root → shiftedRoots[i]? = some shiftedRoot →
      denoteExpr fixtureAlgebra sourceEnv fixtureBuilder.exprs
          fixtureBuilder_wellFormed.arena root =
        denoteExpr fixtureAlgebra shiftedEnv after.exprs
          (shiftSampleIndex_run_certificate fixtureBuilder_wellFormed
            fixtureRoots_owned hRun).after_wellFormed.arena shiftedRoot := by
  exact shiftSampleIndex_denotes fixtureBuilder_wellFormed fixtureRoots_owned
    hRun fixtureAlgebra fixtureEnvs_agree (by rfl)

end Tropical.Testing.ShiftSampleIndex
