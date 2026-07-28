import Tropical.Trust
import Tropical.EmitArrow.Term
import Tropical.EmitArrow.Modal.Residue

/-!
# Production semantic fixtures and trust audit

The four fixtures instantiate the checked relational fallback on the requested
production shapes.  The differential calls `lowerSig` (therefore its compiled
pointer implementation) and `lowerSigTree` independently; it is evidence for,
not a theorem about, pointer identity.
-/

namespace Tropical.Testing.Semantics

open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Semantics

def modalColumns : BankCols where
  count := 2
  idxId := 3
  incr := .arr #[lit 1, lit 2]
  sigma := .arr #[lit (-1), lit (-2)]
  cre := .arr #[lit 3, lit 4]
  cim := .arr #[lit 5, lit 6]

def modalBankFixture : Sig :=
  bankFold modalColumns fun mode => add mode.cre mode.cim

def nestedBankFixture : Sig :=
  .bankSum 2 #[.arr #[lit 1, lit 2]]
    (.bankSum 3 #[.arr #[lit 3, lit 4, lit 5]]
      (.binary .add (.loopIdx 7) (.loopIdx 11)) none 11)
    none 7

def parameterSelectFixture : Sig :=
  .select (.binary .gt (.paramRef ⟨0⟩) (lit 0))
    (.binary .mul (.inputRef ⟨0⟩) (.paramRef ⟨0⟩))
    (lit 0)

def productionFixtures : Array Sig :=
  #[clockLit, modalBankFixture, nestedBankFixture, parameterSelectFixture]

/-- Minimal carrier used to pin the production bank-count clamp at both
    boundaries.  The other operations are irrelevant to these checks. -/
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

example (arena : ExprArena) :
    LowersTo clockLit arena (lowerSigTree clockLit arena).1
      (lowerSigTree clockLit arena).2 :=
  lowerSigTree_lowersTo _ _

example (arena : ExprArena) :
    LowersTo modalBankFixture arena (lowerSigTree modalBankFixture arena).1
      (lowerSigTree modalBankFixture arena).2 :=
  lowerSigTree_lowersTo _ _

example (arena : ExprArena) :
    LowersTo nestedBankFixture arena (lowerSigTree nestedBankFixture arena).1
      (lowerSigTree nestedBankFixture arena).2 :=
  lowerSigTree_lowersTo _ _

example (arena : ExprArena) :
    LowersTo parameterSelectFixture arena
      (lowerSigTree parameterSelectFixture arena).1
      (lowerSigTree parameterSelectFixture arena).2 :=
  lowerSigTree_lowersTo _ _

private def reinternObservesDedup (arena : ExprArena) : Bool :=
  (Array.range arena.nodes.size).all fun index =>
    match arena.nodes[index]? with
    | none => false
    | some node =>
      let result := eintern node arena
      result.1.idx == index &&
        result.2.nodes == arena.nodes &&
        result.2.sigs == arena.sigs

private def sameLoweringResult (sig : Sig) : Bool :=
  let tree := lowerSigTree sig {}
  let optimized := lowerSig sig {}
  tree.1.idx == optimized.1.idx &&
    tree.2.nodes == optimized.2.nodes &&
    tree.2.sigs == optimized.2.sigs &&
    reinternObservesDedup tree.2 &&
    reinternObservesDedup optimized.2

def pointerDifferentialPasses : Bool :=
  productionFixtures.all sameLoweringResult

def runTrustAudit : IO Bool := do
  let ledgerErrors := Tropical.Trust.auditLedger
  let reportPath := System.FilePath.mk "design/trust-boundary.md"
  let reportMatches ←
    if ← reportPath.pathExists then
      let contents ← IO.FS.readFile reportPath
      pure (contents == Tropical.Trust.renderMarkdown)
    else
      pure false
  let pointerMatches := pointerDifferentialPasses
  if ledgerErrors.isEmpty && reportMatches && pointerMatches then
    IO.println "  PASS  trust-ledger"
    return true
  for error in ledgerErrors do IO.println s!"  FAIL  trust-ledger  {error}"
  if !reportMatches then IO.println "  FAIL  trust-ledger  generated report drift"
  if !pointerMatches then IO.println "  FAIL  trust-ledger  lowerSig pointer differential"
  return false

end Tropical.Testing.Semantics
