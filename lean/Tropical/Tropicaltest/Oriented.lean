import Tropical.EmitArrow.Modal.OrientedRealize

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

private abbrev CplxF := Float × Float

private def caddF (a b : CplxF) : CplxF := (a.1 + b.1, a.2 + b.2)
private def cmulF (a b : CplxF) : CplxF :=
  (a.1 * b.1 - a.2 * b.2, a.1 * b.2 + a.2 * b.1)
private def cdivF (a b : CplxF) : CplxF :=
  let d := b.1 * b.1 + b.2 * b.2
  ((a.1 * b.1 + a.2 * b.2) / d,
   (a.2 * b.1 - a.1 * b.2) / d)
private def cscaleF (scale : Float) (a : CplxF) : CplxF :=
  (scale * a.1, scale * a.2)
private def cdistF (a b : CplxF) : Float :=
  Float.sqrt ((a.1 - b.1) * (a.1 - b.1) + (a.2 - b.2) * (a.2 - b.2))

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

private def phaserPoleValues : Array Float :=
  let ratios : Array Float := #[
    0.42044820762685725, 0.5946035575013605, 0.8408964152537145,
    1.189207115002721, 1.681792830507429, 2.378414230005442]
  ratios.map fun ratio => 6.283185307179586 * 700.0 * ratio

private def phaserTails : Array ModalMode :=
  phaserPoleValues.map fun a => allpassTail (litF a)

private def bankTransfer (bank : Bank) (omega : Float) : Option CplxF :=
  foldCplx (bank.transferAt (litF omega))

/-- Independent direct rational evaluator for the fixed all-pass product.  It
    does not inspect modal residues emitted by `Bank.phaser`. -/
private def phaserOracle (omega mix : Float) : CplxF :=
  let s : CplxF := (0.0, omega)
  let source := cdivF (3.0, 0.0) (2.0, omega)
  let allpass := phaserPoleValues.foldl (fun value a =>
    cmulF value (cdivF (s.1 - a, s.2) (s.1 + a, s.2))) (1.0, 0.0)
  cmulF source (caddF (1.0 - mix, 0.0) (cscaleF mix allpass))

/-- Whole-bank scale/add/blend keep authored arm order and one exact-zero
    scalar. -/
private def linearBankCheck : Bool :=
  let left : Bank := { future := #[sourceMode], atZero := (lit 1, lit 2) }
  let right : Bank := { future := #[roomMode], atZero := (lit 3, lit 4) }
  let sum := left.add right
  let scaled := sum.scale (lit 2, lit (-1))
  match sum.future[0]?, sum.future[1]?, foldCplx sum.atZero,
      foldCplx scaled.atZero with
  | some first, some second, some atZero, some scaledZero =>
      sigConstF? first.sigma == some 2.0 && sigConstF? second.sigma == some 5.0 &&
        atZero == (4.0, 6.0) && scaledZero == (14.0, 8.0)
  | _, _, _, _ => false

/-- One identity-plus-tail section agrees in time with a direct convolution;
    the direct path is observable in the `-7e^-2t + 10e^-5t` result. -/
private def oneSectionCheck : Bool :=
  let filtered := (Bank.ofFuture #[sourceMode]).allpassSection (allpassTail (lit 5))
  let cases : Array Float := #[0.0, 0.01, 0.1, 0.5]
  cases.all fun coordinate =>
    let expected := if coordinate == 0.0 then 0.0 else
      -7.0 * Float.exp (-2.0 * coordinate) + 10.0 * Float.exp (-5.0 * coordinate)
    match evalBank filtered coordinate with
    | some actual => close actual expected 2.0e-10
    | none => false

/-- Six sequential sections and the compact degree-zero decorator both agree
    with the independent rational product over mix endpoints, cancellation
    neighborhoods, and the audible frequency grid. -/
private def phaserRationalCheck : Bool :=
  let dry := Bank.ofFuture #[sourceMode]
  let frequencies : Array Float := #[
    20.0, 162.211, 699.9, 700.0, 700.1, 3020.766, 20000.0]
  let mixes : Array Float := #[0.0, 0.5, 1.0]
  mixes.all fun mix =>
    let mixE := litF mix
    let reference := dry.phaser phaserTails mixE
    let compact := Bank.ofFuture
      (decorateDegreeZeroCausalPhaser #[sourceMode] phaserTails mixE)
    frequencies.all fun frequency =>
      let omega := 6.283185307179586 * frequency
      match bankTransfer reference omega, bankTransfer compact omega with
      | some actual, some decorated =>
          let expected := phaserOracle omega mix
          cdistF actual expected < 2.0e-9 && cdistF decorated expected < 2.0e-9
      | _, _ => false

private def wetUnitMagnitudeCheck : Bool :=
  let frequencies : Array Float := #[20.0, 40.0, 100.0, 700.0, 4000.0, 20000.0]
  frequencies.all fun frequency =>
    let omega := 6.283185307179586 * frequency
    let s : CplxF := (0.0, omega)
    let wet := phaserPoleValues.foldl (fun value a =>
      cmulF value (cdivF (s.1 - a, s.2) (s.1 + a, s.2))) (1.0, 0.0)
    close (Float.sqrt (wet.1 * wet.1 + wet.2 * wet.2)) 1.0 2.0e-15

/-- Frozen LTI room/phaser commutation across FF/interior/PP room orientation.
    This is the numerical law used by the exact two-room canonicalizer; no such
    rewrite is defined for gauge or bloom stages. -/
private def phaserRoomCommutationCheck : Bool :=
  let room : Array ModalMode := #[
    { sigma := lit 11, omega := lit 37, cre := lit 7 1, cim := lit (-2) 1 }]
  let directions : Array Sig := #[lit 0, lit 4 1, lit 1]
  let frequencies : Array Float := #[31.0, 700.0, 4700.0]
  directions.all fun direction =>
    let source := Bank.ofFuture #[sourceMode]
    let roomThenPhaser :=
      (source.convolveKernel room direction syntacticSameSideClassifier).phaser
        phaserTails (lit 5 1)
    let phaserThenRoom :=
      (source.phaser phaserTails (lit 5 1)).convolveKernel room direction
        syntacticSameSideClassifier
    frequencies.all fun frequency =>
      let omega := 6.283185307179586 * frequency
      match bankTransfer roomThenPhaser omega, bankTransfer phaserThenRoom omega with
      | some left, some right => cdistF left right < 5.0e-8
      | _, _ => false

def runOriented : IO Bool := do
  let fp := futurePastAxisCheck
  let direction := directionCheck
  let collected := collectedPairwiseCheck
  let coincidence := coincidenceCheck
  let linear := linearBankCheck
  let oneSection := oneSectionCheck
  let rational := phaserRationalCheck
  let unitMagnitude := wetUnitMagnitudeCheck
  let commutation := phaserRoomCommutationCheck
  if fp && direction && collected && coincidence && linear && oneSection &&
      rational && unitMagnitude && commutation then
    passGate "modal-oriented"
      "FP axes + exact zero seam; local direction; collected=pairwise; repeated-pole beta limit; bank linearity; current-universe all-pass cascade=independent rational oracle; wet |A|=1; compact decoration exact; frozen room/phaser commutation"
  else
    failGate "modal-oriented"
      s!"futurePast={fp} direction={direction} collectedPairwise={collected} coincidence={coincidence} linear={linear} section={oneSection} rational={rational} unit={unitMagnitude} commute={commutation}"

end Tropical.Tropicaltest.Oriented
