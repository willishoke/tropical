import Tropical.EmitArrow.Modal.Oriented

/-!
# Focused tests for per-kernel oriented modal convolution

These gates stay at the symbolic modal layer.  They fold constant `Sig`
expressions back to floats and sample the resulting analytic expansions on the
negative axis, at exactly zero, and on the positive axis.  No patch lowering or
modal realizer participates.
-/

namespace Tropical.Tropicaltest.Oriented

open Tropical
open Tropical.EmitArrow
open Tropical.EmitArrow.Oriented

private def passGate (label detail : String) : IO Bool := do
  IO.println s!"  PASS  {label}  {detail}"
  pure true

private def failGate (label detail : String) : IO Bool := do
  IO.println s!"  FAIL  {label}  {detail}"
  pure false

private def foldCplx (value : CplxE) : Option (Float × Float) := do
  let re ← sigConstF? value.1
  let im ← sigConstF? value.2
  pure (re, im)

private def powNat (base : Float) : Nat → Float
  | 0 => 1.0
  | n + 1 => powNat base n * base

private def evalAtom (orientation : Orientation) (atom : Atom)
    (coordinate : Float) : Option Float := do
  let (poleRe, poleIm) ← foldCplx atom.pole
  let (ampRe, ampIm) ← foldCplx atom.amp
  let axis := match orientation with
    | .future => coordinate
    | .past => -coordinate
  let phase := poleIm * axis
  let carrier := ampRe * Float.cos phase - ampIm * Float.sin phase
  pure (powNat axis atom.deg * Float.exp (poleRe * axis) * carrier)

private def evalAtoms (orientation : Orientation) (atoms : Array Atom)
    (coordinate : Float) : Option Float := do
  let mut total := 0.0
  for atom in atoms do
    total := total + (← evalAtom orientation atom coordinate)
  pure total

private def evalExpansion (expansion : Expansion) (coordinate : Float) :
    Option Float :=
  if coordinate > 0.0 then
    evalAtoms .future expansion.future coordinate
  else if coordinate < 0.0 then
    evalAtoms .past expansion.past coordinate
  else do
    let (atZero, _) ← foldCplx expansion.atZero
    pure atZero

private def evalModes (orientation : Orientation) (modes : Array ModalMode)
    (coordinate : Float) : Option Float :=
  evalAtoms orientation (modes.map Atom.ofMode) coordinate

private def evalBank (bank : Bank) (coordinate : Float) : Option Float :=
  if coordinate > 0.0 then
    evalModes .future bank.future coordinate
  else if coordinate < 0.0 then
    evalModes .past bank.past coordinate
  else do
    let (atZero, _) ← foldCplx bank.atZero
    pure atZero

private def close (actual expected : Float) (tolerance : Float := 1.0e-10) :
    Bool :=
  (actual - expected).abs < tolerance

private def sourceMode : ModalMode :=
  { sigma := lit 2, omega := lit 0, cre := lit 3 }

private def roomMode : ModalMode :=
  { sigma := lit 5, omega := lit 0, cre := lit 7 }

private def alwaysDistinct : SameSideClassifier :=
  fun _ _ _ => .distinct

/-- One analytic `F(-2,3,0) * P(-5,7,0)` pair is `3*exp(-2t)` for `t>0`,
`3*exp(5t)` for `t<0`, and exactly `3` at zero. -/
private def futurePastAxisCheck : Bool :=
  let expansion := convolveFuturePast (Atom.ofMode sourceMode) (Atom.ofMode roomMode)
  let cases : Array (Float × Float) := #[
    (-0.25, 3.0 * Float.exp (-1.25)),
    (0.0, 3.0),
    (0.25, 3.0 * Float.exp (-0.5))]
  cases.all fun (coordinate, expected) =>
    match evalExpansion expansion coordinate with
    | some actual => close actual expected
    | none => false

private def directedOracle (direction : Sig) : Option Bank := do
  let kernel := Bank.kernel #[roomMode] direction
  let futureMode ← kernel.future[0]?
  let pastMode ← kernel.past[0]?
  let source : DirectedAtom :=
    { Atom.ofMode sourceMode with orientation := .future }
  let future : DirectedAtom :=
    { Atom.ofMode futureMode with orientation := .future }
  let past : DirectedAtom :=
    { Atom.ofMode pastMode with orientation := .past }
  pure ((convolve .distinct source future).add
    (convolve .distinct source past) |>.toBank)

private def expectedDirected (direction coordinate : Float) : Float :=
  if coordinate > 0.0 then
    (1.0 - direction) * 7.0 *
        (Float.exp (-2.0 * coordinate) - Float.exp (-5.0 * coordinate)) +
      direction * 3.0 * Float.exp (-2.0 * coordinate)
  else if coordinate < 0.0 then
    direction * 3.0 * Float.exp (5.0 * coordinate)
  else
    direction * 3.0

/-- Direction 0, 1/4, and 1 each orient only the room kernel.  The production
collected result is checked on all three time regions against the direct form. -/
private def directionCheck : Bool :=
  let directions : Array (Sig × Float) := #[
    (lit 0, 0.0), (lit 25 2, 0.25), (lit 1, 1.0)]
  let coordinates : Array Float := #[-0.2, 0.0, 0.2]
  directions.all fun (directionSig, direction) =>
    let bank := (Bank.ofFuture #[sourceMode]).convolveKernel #[roomMode]
      directionSig alwaysDistinct
    coordinates.all fun coordinate =>
      match evalBank bank coordinate with
      | some actual => close actual (expectedDirected direction coordinate)
      | none => false

/-- On a one-mode degree-zero input, the collected `m+n` carrier agrees with
the literal FF+FP pairwise oracle across both axes and the zero seam. -/
private def collectedPairwiseCheck : Bool :=
  let direction := lit 25 2
  let collected := (Bank.ofFuture #[sourceMode]).convolveKernel #[roomMode]
    direction alwaysDistinct
  match directedOracle direction with
  | none => false
  | some pairwise =>
      let coordinates : Array Float := #[-0.4, -0.1, 0.0, 0.1, 0.4]
      collected.future.size == 2 && collected.past.size == 1 &&
        coordinates.all fun coordinate =>
          match evalBank collected coordinate, evalBank pairwise coordinate with
          | some actual, some expected => close actual expected
          | _, _ => false

/-- The repeated-pole beta limit for degrees 1 and 2 is one degree-4 atom with
amplitude `3*5*1!*2!/4! = 5/4`, and remains zero at the strict-axis seam. -/
private def coincidenceCheck : Bool :=
  let left : Atom :=
    { pole := (neg (lit 2), lit 0), amp := (lit 3, lit 0), deg := 1 }
  let right : Atom :=
    { pole := (neg (lit 2), lit 0), amp := (lit 5, lit 0), deg := 2 }
  let expansion := convolveSameSideCoincident .future left right
  match expansion.future[0]?, foldCplx expansion.atZero with
  | some atom, some (atZero, _) =>
      expansion.future.size == 1 && expansion.past.isEmpty && atom.deg == 4 &&
        match foldCplx atom.amp with
        | some (amp, imag) => close amp 1.25 && close imag 0.0 && close atZero 0.0
        | none => false
  | _, _ => false

def runOriented : IO Bool := do
  let fp := futurePastAxisCheck
  let direction := directionCheck
  let collected := collectedPairwiseCheck
  let coincidence := coincidenceCheck
  if fp && direction && collected && coincidence then
    passGate "modal-oriented"
      "FP axes + exact zero seam; local direction endpoints/interior; collected=pairwise; repeated-pole beta limit"
  else
    failGate "modal-oriented"
      s!"futurePast={fp} direction={direction} collectedPairwise={collected} coincidence={coincidence}"

end Tropical.Tropicaltest.Oriented
