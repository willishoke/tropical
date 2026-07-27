import Tropical.Testing.ArrowFixtures
import Tropical.EmitArrow.ClockAlgebra

/-!
# ClockLaws — the ArrowLaws fixture clocks, as theorems (slice 3b, WS-3/4)

`Tropicaltest/ArrowLaws.lean` certifies laws 1/2/4/5 by rendering both
sides of each fixture pair and comparing SHA256. This module states the
same four equations as theorems over the SAME fixture `Sig` values — a
universal statement where the gates sample. The generic laws
(`EmitArrow/ClockAlgebra.lean`) quantify over every rail expression; these
instances pin the actual `Testing/ArrowFixtures.lean` trees to them, so
the audio gates and the theorems provably speak about the same objects.

What this changes about the golden set: the per-law-instance audio gates
were carrying BOTH the algebra claim (LHS clock ≡ RHS clock as integers)
and the backend claim (the engine renders a given plan as frozen). The
algebra half is now these theorems, for all clocks; the gates' remaining
job is the backend half, for which one gate per backend path suffices.
Nothing is deleted here — shrinking the gate set is a separate,
now-licensed move (see the handoff's "what this does not replace").

Laws 3 and 6 are OUT of this fragment, deliberately:
* **Law 3 (diagonal)** is a program-graph property (shared vs duplicated
  dry instance under CSE), not a clock-expression identity.
* **Law 6 (reverse/flanger commute)** is a value-datapath law — the ±δ
  tap sum reassociates in float (the slice-5 finding,
  `ArrowLaws.lean`). The proof boundary landing exactly on the
  empirical boundary is evidence the fragment is carved right.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

-- Rail derivations for the fixture ingredients: the deltas are `toInt`
-- boundaries (their float seconds→samples math is behind the door), the
-- clock is the root clock.

/-- `deltaLit` (hence `delta1`/`delta2`) is a boundary: a closed-form
    Q32.32 integer sample count entering the rail through `toInt`. -/
def deltaLit_rail (mantissa : Int) (exponent : Nat) :
    OnClockRail (deltaLit mantissa exponent) := .boundary _

def delta1_rail : OnClockRail delta1 := .boundary _
def delta2_rail : OnClockRail delta2 := .boundary _

-- The four law pairs, as derivations of the fixture trees…

def invLawLhs_rail : OnClockRail invLawLhsClock :=
  .sub (.add rootClock_rail delta1_rail) delta1_rail
def invLawRhs_rail : OnClockRail invLawRhsClock := rootClock_rail

def addLawLhs_rail : OnClockRail addLawLhsClock :=
  .sub (.sub rootClock_rail delta1_rail) delta2_rail
def addLawRhs_rail : OnClockRail addLawRhsClock :=
  .sub rootClock_rail (.add delta1_rail delta2_rail)

def revInvolutionLhs_rail : OnClockRail revInvolutionLhsClock :=
  .neg (.neg rootClock_rail)
def revInvolutionRhs_rail : OnClockRail revInvolutionRhsClock := rootClock_rail

def revSwapLhs_rail : OnClockRail revSwapLhsClock :=
  .neg (.sub rootClock_rail delta1_rail)
def revSwapRhs_rail : OnClockRail revSwapRhsClock :=
  .add (.neg rootClock_rail) delta1_rail

-- …and the four laws, each an instance of its universal form. Under
-- CLOCK_RAIL_IS_EXACT (the one named hypothesis,
-- `EmitArrow/ClockAlgebra.lean`), each equation pushes forward to a
-- bit-identical i64 clock on the wire — which is exactly what the audio
-- gates observe as byte-identical renders.

/-- **Law 1 (inverse/cancellation)**: `(clk+δ₁)−δ₁ = clk`. -/
theorem invLaw_denote (env : ClockEnv) :
    denoteClock invLawLhs_rail env = denoteClock invLawRhs_rail env :=
  warp_inv rootClock_rail delta1_rail env

/-- **Law 2 (additive delay)**: `(clk−δ₁)−δ₂ = clk−(δ₁+δ₂)`. -/
theorem addLaw_denote (env : ClockEnv) :
    denoteClock addLawLhs_rail env = denoteClock addLawRhs_rail env :=
  warp_assoc rootClock_rail delta1_rail delta2_rail env

/-- **Law 4 (reverse involution)**: `−(−clk) = clk`. -/
theorem revInvolution_denote (env : ClockEnv) :
    denoteClock revInvolutionLhs_rail env = denoteClock revInvolutionRhs_rail env :=
  rev_involution rootClock_rail env

/-- **Law 5 (reverse conjugates delay)**: `−(clk−δ₁) = (−clk)+δ₁`. -/
theorem revSwap_denote (env : ClockEnv) :
    denoteClock revSwapLhs_rail env = denoteClock revSwapRhs_rail env :=
  rev_swap rootClock_rail delta1_rail env

end Tropical.EmitArrow
