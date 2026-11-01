import Tropical.Proofs
import Tropical.Testing.EmitArrow
import Tropical.Testing.ClockLaws
import Tropical.Testing.Semantics
import Tropical.Tropicaltest.Exact
import Tropical.Tropicaltest.Oriented
import Tropical.Tropicaltest.Block
import Tropical.Tropicaltest.KernelType

/-!
# tropicaltest-core — the gates that need no runtime

The core tier of the test suite: the trusted-boundary ledger, the EmitArrow
authoring qualification, the exact bake carrier, and the oriented and block
modal algebra. Each of these is a statement about the compiler alone, so
this binary links no libtropical and tools/check_layers.py keeps its import
closure inside TropicalCore and TropicalTest. The same gates also run inside
`tropicaltest`, the conformance tier, next to the goldens and the
realization-variant equivalences that need a host.
-/

/-- The sections of the core tier: a header line and its gates, in the
    order `tropicaltest` runs them. -/
def sections : List (String × List (IO Bool)) := [
  ("trusted boundary (typed ledger, report, production fixtures):", [
    Tropical.Testing.Semantics.runTrustAudit]),
  ("EmitArrow authoring foundation:", [
    Tropical.Testing.EmitArrow.runPhase1Gate,
    Tropical.Testing.EmitArrow.runPhase2Gate,
    Tropical.Testing.EmitArrow.runPhase3Gate,
    Tropical.Testing.ClockLaws.runPhase4Gate,
    Tropical.Testing.EmitArrow.runPhase5Gate]),
  ("exact bake carrier (dyadic/interval — the bake layer's libm exile):", [
    Tropical.Tropicaltest.ExactGates.runExactCorpse,
    Tropical.Tropicaltest.ExactGates.runExactConstants,
    Tropical.Tropicaltest.ExactGates.runExactElementary,
    Tropical.Tropicaltest.ExactGates.runExactAtan2,
    Tropical.Tropicaltest.ExactGates.runExactRecip10,
    Tropical.Tropicaltest.ExactGates.runExactValues,
    Tropical.Tropicaltest.ExactGates.runExactPlayground,
    Tropical.Tropicaltest.ExactGates.runExactQuantize]),
  ("oriented modal convolution (local room direction):", [
    Tropical.Tropicaltest.Oriented.runOriented]),
  ("block carrier (divided-difference Leibniz over the factor tree):", [
    Tropical.Tropicaltest.Block.runBlockAlgebra]),
  ("typed kernel (the port: tau/params/out, tropical_kernel_1 round trip):", [
    Tropical.Tropicaltest.KernelType.runKernelType])]

def main (_ : List String) : IO UInt32 := do
  let mut failed := 0
  let mut total := 0
  for (header, gates) in sections do
    IO.println header
    for gate in gates do
      total := total + 1
      if !(← gate) then failed := failed + 1
  IO.println ""
  IO.println s!"{total - failed}/{total} passed"
  return if failed == 0 then 0 else 1
