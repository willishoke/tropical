import Tropical.EmitArrow.ClockAlgebra

/-!
# Arena-native clock-law witness

This focused phase-4 fixture authors the inverse law with the production
`BuildM` smart constructors, freezes the resulting builder arena, constructs
the rail derivations from dereference facts, and instantiates the universal
arena-native theorem.  No recursive `Sig` or lowering function participates.

The clock is the production clock: τ itself, the IR's Q32.32 time leaf
(`clockLit`/`clockSig` are `tau`). The fixture warps it by a boundary delta
and back.
-/

namespace Tropical.Testing.ClockLaws

open Tropical.Ir
open Tropical.EmitArrow

private structure InverseFixture where
  clock : ExprId
  deltaSource : ExprId
  delta : ExprId
  sum : ExprId
  root : ExprId
deriving Inhabited

private def buildInverseFixture : BuildM InverseFixture := do
  let clock ← tau
  let deltaSource ← lit 1
  let delta ← toIntE deltaSource
  let sum ← add clock delta
  let root ← sub sum delta
  pure { clock, deltaSource, delta, sum, root }

private def built : InverseFixture × Builder :=
  match buildInverseFixture.run {} with
  | .ok result => result
  | .error _ => default

private theorem clockNode :
    built.2.exprs.deref built.1.clock = some .tau := by
  apply eq_of_beq
  native_decide

private theorem deltaNode :
    built.2.exprs.deref built.1.delta =
      some (.unary .toInt built.1.deltaSource) := by
  apply eq_of_beq
  native_decide

private theorem sumNode :
    built.2.exprs.deref built.1.sum =
      some (.binary .add built.1.clock built.1.delta) := by
  apply eq_of_beq
  native_decide

private theorem rootNode :
    built.2.exprs.deref built.1.root =
      some (.binary .sub built.1.sum built.1.delta) := by
  apply eq_of_beq
  native_decide

private def clockRail : OnClockRail built.2.exprs built.1.clock :=
  .tau built.1.clock clockNode

private def deltaRail : OnClockRail built.2.exprs built.1.delta :=
  .boundary built.1.delta built.1.deltaSource deltaNode

private def sumRail : OnClockRail built.2.exprs built.1.sum :=
  .add sumNode clockRail deltaRail

private def rootRail : OnClockRail built.2.exprs built.1.root :=
  .sub rootNode sumRail deltaRail

/-- The production-authored native DAG is an instance of arena-native law 1. -/
theorem inverseFixture_denotes (env : ClockEnv) :
    denoteClock rootRail env = denoteClock clockRail env :=
  warp_inv clockRail deltaRail sumNode rootNode env

/-- Executable companion to the theorem witness: the production builder made
    one child-descending five-node graph and returned its final root. -/
def fixturePasses : Bool :=
  built.2.exprs.wf &&
    built.2.exprs.nodes.size == 5 &&
    built.1.root.idx == 4 &&
    built.2.exprs.deref built.1.root ==
      some (.binary .sub built.1.sum built.1.delta)

def runPhase4Gate : IO Bool := do
  if fixturePasses then
    IO.println "  PASS  arena-native-phase4  clock-rail-nodes=5"
    pure true
  else
    IO.println "  FAIL  arena-native-phase4  production clock-law witness drifted"
    pure false

end Tropical.Testing.ClockLaws
