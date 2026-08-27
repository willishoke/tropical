import Tropical.Trust
import Tropical.Semantics.Expr

/-!
# Production arena semantics and trust audit

Phase 4 removes the recursive source/lowering differential from the active
semantic gate.  These fixtures exercise the direct `ExprArena` semantics and
the bank-count boundary used by both bank and routed reductions.
-/

namespace Tropical.Testing.Semantics

open Tropical.Ir
open Tropical.Semantics

/-- Minimal carrier used to pin the production bank-count clamp at both
    boundaries. The other operations are irrelevant to these checks. -/
private def countAlgebra : Algebra Int where
  literal := fun _ => refusal "count-test" "literal is outside this fixture"
  unary := fun _ _ => refusal "count-test" "unary is outside this fixture"
  binary := fun _ _ _ => refusal "count-test" "binary is outside this fixture"
  clamp := fun _ _ _ => refusal "count-test" "clamp is outside this fixture"
  select := fun _ _ _ => refusal "count-test" "select is outside this fixture"
  index := fun _ _ => refusal "count-test" "index is outside this fixture"
  loopIndex := fun index => .ok (.scalar (Int.ofNat index))
  dynamicCount := fun
    | .scalar count => .ok count
    | .array _ => .error { operation := "count-test", detail := "array count" }
  zero := .ok (.scalar 0)

example : bankTrips countAlgebra 4 (some (.ok (.scalar (-3)))) = .ok 0 := rfl

example : bankTrips countAlgebra 4 (some (.ok (.scalar 9))) = .ok 4 := rfl

/-- A tiny frozen production DAG used to instantiate direct denotation and
    extension stability without any recursive authoring syntax. -/
private def fixtureArena : ExprArena := {
  nodes := #[.sampleIndex, .num ⟨1, 0⟩, .binary .add ⟨0⟩ ⟨1⟩]
  sigs := #[{ base := .s1 }, {}, { base := .s1 }]
}

private theorem fixtureWellFormed : ArenaWellFormed fixtureArena := by
  constructor
  · intro id node hDeref child hChild
    obtain ⟨i⟩ := id
    have hi : i < 3 := by
      exact deref_index_lt hDeref
    have hCases : i = 0 ∨ i = 1 ∨ i = 2 := by omega
    rcases hCases with rfl | rfl | rfl <;>
      simp [fixtureArena, ExprArena.deref] at hDeref <;>
      subst node <;> simp_all [ENode.children]
    rcases hChild with rfl | rfl <;> decide
  · simp [DedupSound, fixtureArena]
  · rfl

private def extendedFixtureArena : ExprArena := {
  fixtureArena with
  nodes := fixtureArena.nodes.push (.unary .neg ⟨2⟩)
  sigs := fixtureArena.sigs.push { base := .s1 }
}

private theorem extendedFixtureWellFormed : ArenaWellFormed extendedFixtureArena := by
  constructor
  · intro id node hDeref child hChild
    obtain ⟨i⟩ := id
    have hi : i < 4 := by
      exact deref_index_lt hDeref
    have hCases : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
    rcases hCases with rfl | rfl | rfl | rfl <;>
      simp [extendedFixtureArena, fixtureArena, ExprArena.deref] at hDeref <;>
      subst node <;> simp_all [ENode.children]
    rcases hChild with rfl | rfl <;> decide
  · simp [DedupSound, extendedFixtureArena, fixtureArena]
  · rfl

private theorem fixtureExtends : Extends fixtureArena extendedFixtureArena := by
  intro id node hDeref
  obtain ⟨i⟩ := id
  have hi : i < 3 := deref_index_lt hDeref
  have hCases : i = 0 ∨ i = 1 ∨ i = 2 := by omega
  rcases hCases with rfl | rfl | rfl <;>
    simp [fixtureArena, extendedFixtureArena, ExprArena.deref] at hDeref ⊢ <;>
    exact hDeref

example (alg : Algebra α) (env : SigEnv α) :
    denoteExpr alg env fixtureArena fixtureWellFormed ⟨2⟩ =
      denoteExpr alg env extendedFixtureArena extendedFixtureWellFormed ⟨2⟩ :=
  denoteExpr_extends fixtureWellFormed extendedFixtureWellFormed fixtureExtends
    alg env (by rfl)

/-- The ordinary and tile coordinates are independent semantic rails. Exact
    callers may bind them equally; materializer proofs can vary them
    independently without changing the expression language. -/
example (alg : Algebra α) (env : SigEnv α) :
    denoteNode alg env .sampleIndex (fun _ _ _ => .ok env.sampleIndex) =
      .ok env.sampleIndex := rfl

example (alg : Algebra α) (env : SigEnv α) :
    denoteNode alg env .tileSampleIndex (fun _ _ _ => .ok env.sampleIndex) =
      .ok env.tileSampleIndex := rfl

def runTrustAudit : IO Bool := do
  let ledgerErrors := Tropical.Trust.auditLedger
  let reportPath := System.FilePath.mk "design/trust-boundary.md"
  let reportMatches ←
    if ← reportPath.pathExists then
      let contents ← IO.FS.readFile reportPath
      pure (contents == Tropical.Trust.renderMarkdown)
    else
      pure false
  if ledgerErrors.isEmpty && reportMatches then
    IO.println "  PASS  trust-ledger"
    return true
  for error in ledgerErrors do IO.println s!"  FAIL  trust-ledger  {error}"
  if !reportMatches then IO.println "  FAIL  trust-ledger  generated report drift"
  return false

end Tropical.Testing.Semantics
