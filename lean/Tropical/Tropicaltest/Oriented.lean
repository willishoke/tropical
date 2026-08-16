import Tropical.EmitArrow.Modal.OrientedRealize
import Tropical.Testing.ArrowFixtures

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
open Tropical.Ir

private def passGate (label detail : String) : IO Bool := do
  IO.println s!"  PASS  {label}  {detail}"
  pure true

private def failGate (label detail : String) : IO Bool := do
  IO.println s!"  FAIL  {label}  {detail}"
  pure false

/-- Inspect coefficients from the actual frozen arena produced by a native
    build.  No recursive expression evaluator participates. -/
private def foldCplx (constants : Array (Option Tropical.Exact.DyadicI))
    (value : CplxE) : Option (Float × Float) := do
  let re ← (sigConstDFrom? constants value.1).map Tropical.Exact.DyadicI.toFloat
  let im ← (sigConstDFrom? constants value.2).map Tropical.Exact.DyadicI.toFloat
  pure (re, im)

private def powNat (base : Float) : Nat → Float
  | 0 => 1.0
  | n + 1 => powNat base n * base

private def evalAtom (constants : Array (Option Tropical.Exact.DyadicI))
    (orientation : Orientation) (atom : Atom)
    (coordinate : Float) : Option Float := do
  let (poleRe, poleIm) ← foldCplx constants atom.pole
  let (ampRe, ampIm) ← foldCplx constants atom.amp
  let axis := match orientation with
    | .future => coordinate
    | .past => -coordinate
  let phase := poleIm * axis
  let carrier := ampRe * Float.cos phase - ampIm * Float.sin phase
  pure (powNat axis atom.deg * Float.exp (poleRe * axis) * carrier)

private def evalAtoms (constants : Array (Option Tropical.Exact.DyadicI))
    (orientation : Orientation) (atoms : Array Atom)
    (coordinate : Float) : Option Float := do
  let mut total := 0.0
  for atom in atoms do
    total := total + (← evalAtom constants orientation atom coordinate)
  pure total

private def evalExpansion (constants : Array (Option Tropical.Exact.DyadicI))
    (expansion : Expansion) (coordinate : Float) :
    Option Float :=
  if coordinate > 0.0 then
    evalAtoms constants .future expansion.future coordinate
  else if coordinate < 0.0 then
    evalAtoms constants .past expansion.past coordinate
  else do
    let (atZero, _) ← foldCplx constants expansion.atZero
    pure atZero

private def evalModes (constants : Array (Option Tropical.Exact.DyadicI))
    (orientation : Orientation) (modes : Array ModalMode)
    (coordinate : Float) : Option Float := do
  let values ← modes.mapM fun mode => do
    let sigma ← (sigConstDFrom? constants mode.sigma).map Tropical.Exact.DyadicI.toFloat
    let omega ← (sigConstDFrom? constants mode.omega).map Tropical.Exact.DyadicI.toFloat
    let ampRe ← (sigConstDFrom? constants mode.cre).map Tropical.Exact.DyadicI.toFloat
    let ampIm ← (sigConstDFrom? constants mode.cim).map Tropical.Exact.DyadicI.toFloat
    pure (sigma, omega, ampRe, ampIm, mode.deg)
  let axis := match orientation with | .future => coordinate | .past => -coordinate
  pure (values.foldl (fun total item =>
    let (sigma, omega, ampRe, ampIm, degree) := item
    let phase := omega * axis
    total + powNat axis degree * Float.exp (-sigma * axis) *
      (ampRe * Float.cos phase - ampIm * Float.sin phase)) 0.0)

private def evalBank (constants : Array (Option Tropical.Exact.DyadicI))
    (bank : Bank) (coordinate : Float) : Option Float :=
  if coordinate > 0.0 then
    evalModes constants .future bank.future coordinate
  else if coordinate < 0.0 then
    evalModes constants .past bank.past coordinate
  else do
    let (atZero, _) ← foldCplx constants bank.atZero
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

private def sourceMode : BuildM ModalMode := do
  pure { sigma := ← lit 2, omega := ← lit 0, cre := ← lit 3, cim := ← lit 0 }

private def roomMode : BuildM ModalMode := do
  pure { sigma := ← lit 5, omega := ← lit 0, cre := ← lit 7, cim := ← lit 0 }

private def alwaysDistinct : SameSideClassifier :=
  fun _ _ _ => .distinct

/-- One analytic `F(-2,3,0) * P(-5,7,0)` pair is `3*exp(-2t)` for `t>0`,
`3*exp(5t)` for `t<0`, and exactly `3` at zero. -/
private def futurePastAxisCheck : Bool :=
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      convolveFuturePast (← Atom.ofMode (← sourceMode)) (← Atom.ofMode (← roomMode)) with
  | .error _ => false
  | .ok (arena, expansion) =>
    let constants := sigConstTable arena
    let cases : Array (Float × Float) := #[
      (-0.25, 3.0 * Float.exp (-1.25)),
      (0.0, 3.0),
      (0.25, 3.0 * Float.exp (-0.5))]
    cases.all fun (coordinate, expected) =>
      match evalExpansion constants expansion coordinate with
      | some actual => close actual expected
      | none => false

private def directedOracle (direction : Sig) : BuildM Bank := do
  let kernel ← Bank.kernel #[← roomMode] direction
  let some futureMode := kernel.future[0]? | throw "directed oracle: missing future mode"
  let some pastMode := kernel.past[0]? | throw "directed oracle: missing past mode"
  let sourceAtom ← Atom.ofMode (← sourceMode)
  let futureAtom ← Atom.ofMode futureMode
  let pastAtom ← Atom.ofMode pastMode
  let source : DirectedAtom :=
    { sourceAtom with orientation := .future }
  let future : DirectedAtom :=
    { futureAtom with orientation := .future }
  let past : DirectedAtom :=
    { pastAtom with orientation := .past }
  let combined ← (← convolve .distinct source future).add
    (← convolve .distinct source past)
  combined.toBank

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
  let directions : Array (Int × Nat × Float) := #[(0, 0, 0.0), (25, 2, 0.25), (1, 0, 1.0)]
  directions.all fun (mantissa, exponent, direction) =>
    match Tropical.Testing.ArrowFixtures.freezeBuild {} do
        let bank ← Bank.ofFuture #[← sourceMode]
        bank.convolveKernel #[← roomMode] (← lit mantissa exponent) alwaysDistinct with
    | .error _ => false
    | .ok (arena, bank) =>
      let constants := sigConstTable arena
      (#[-0.2, 0.0, 0.2] : Array Float).all fun coordinate =>
        match evalBank constants bank coordinate with
        | some actual => close actual (expectedDirected direction coordinate)
        | none => false

/-- On a one-mode degree-zero input, the collected `m+n` carrier agrees with
the literal FF+FP pairwise oracle across both axes and the zero seam. -/
private def collectedPairwiseCheck : Bool :=
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      let direction ← lit 25 2
      let collected ← (← Bank.ofFuture #[← sourceMode]).convolveKernel
        #[← roomMode] direction alwaysDistinct
      let pairwise ← directedOracle direction
      pure (collected, pairwise) with
  | .error _ => false
  | .ok (arena, (collected, pairwise)) =>
      let constants := sigConstTable arena
      let coordinates : Array Float := #[-0.4, -0.1, 0.0, 0.1, 0.4]
      collected.future.size == 2 && collected.past.size == 1 &&
        coordinates.all fun coordinate =>
          match evalBank constants collected coordinate,
              evalBank constants pairwise coordinate with
          | some actual, some expected => close actual expected
          | _, _ => false

/-- The repeated-pole beta limit for degrees 1 and 2 is one degree-4 atom with
amplitude `3*5*1!*2!/4! = 5/4`, and remains zero at the strict-axis seam. -/
private def coincidenceCheck : Bool :=
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      let pole := (← neg (← lit 2), ← lit 0)
      let left : Atom := { pole, amp := (← lit 3, ← lit 0), deg := 1 }
      let right : Atom := { pole, amp := (← lit 5, ← lit 0), deg := 2 }
      convolveSameSideCoincident .future left right with
  | .error _ => false
  | .ok (arena, expansion) =>
    let constants := sigConstTable arena
    match expansion.future[0]?, foldCplx constants expansion.atZero with
  | some atom, some (atZero, _) =>
      expansion.future.size == 1 && expansion.past.isEmpty && atom.deg == 4 &&
        match foldCplx constants atom.amp with
        | some (amp, imag) => close amp 1.25 && close imag 0.0 && close atZero 0.0
        | none => false
  | _, _ => false

private def phaserPoleValues : Array Float :=
  let ratios : Array Float := #[
    0.42044820762685725, 0.5946035575013605, 0.8408964152537145,
    1.189207115002721, 1.681792830507429, 2.378414230005442]
  ratios.map fun ratio => 6.283185307179586 * 700.0 * ratio

private def phaserTails : BuildM (Array ModalMode) :=
  phaserPoleValues.mapM fun a => do allpassTail (← litF a)

private def bankTransfer (bank : Bank) (omega : Float) : BuildM CplxE := do
  bank.transferAt (← litF omega)

/-- Independent direct rational evaluator for one fixed all-pass product.  It
    does not inspect modal residues emitted by either phaser construction. -/
private def phaserOracleFor (poles : Array Float) (omega mix : Float) : CplxF :=
  let s : CplxF := (0.0, omega)
  let source := cdivF (3.0, 0.0) (2.0, omega)
  let allpass := poles.foldl (fun value a =>
    cmulF value (cdivF (s.1 - a, s.2) (s.1 + a, s.2))) (1.0, 0.0)
  cmulF source (caddF (1.0 - mix, 0.0) (cscaleF mix allpass))

/-- Whole-bank scale/add/blend keep authored arm order and one exact-zero
    scalar. -/
private def linearBankCheck : Bool :=
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      let left : Bank := { future := #[← sourceMode], atZero := (← lit 1, ← lit 2) }
      let right : Bank := { future := #[← roomMode], atZero := (← lit 3, ← lit 4) }
      let sum ← left.add right
      let scaled ← sum.scale (← lit 2, ← lit (-1))
      pure (sum, scaled) with
  | .error _ => false
  | .ok (arena, (sum, scaled)) =>
    let constants := sigConstTable arena
    match sum.future[0]?, sum.future[1]?, foldCplx constants sum.atZero,
      foldCplx constants scaled.atZero with
  | some first, some second, some atZero, some scaledZero =>
      (sigConstDFrom? constants first.sigma).map Tropical.Exact.DyadicI.toFloat == some 2.0 &&
        (sigConstDFrom? constants second.sigma).map Tropical.Exact.DyadicI.toFloat == some 5.0 &&
        atZero == (4.0, 6.0) && scaledZero == (14.0, 8.0)
  | _, _, _, _ => false

/-- One identity-plus-tail section agrees in time with a direct convolution;
    the direct path is observable in the `-7e^-2t + 10e^-5t` result. -/
private def oneSectionCheck : Bool :=
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      (← Bank.ofFuture #[← sourceMode]).allpassSection (← allpassTail (← lit 5)) with
  | .error _ => false
  | .ok (arena, filtered) =>
    let constants := sigConstTable arena
    (#[(0.0 : Float), 0.01, 0.1, 0.5]).all fun coordinate =>
      let expected := if coordinate == 0.0 then 0.0 else
        -7.0 * Float.exp (-2.0 * coordinate) + 10.0 * Float.exp (-5.0 * coordinate)
      match evalBank constants filtered coordinate with
      | some actual => close actual expected 2.0e-10
      | none => false

/-- A two-section generic cascade (the induction witness) and the full
    six-section compact decorator each agree with their independent rational
    products over mix endpoints, cancellation neighborhoods, and the audible
    frequency grid.  The generic reference intentionally preserves duplicate
    identity-plus-tail rows and therefore doubles in size at each section; the
    separate Phaser gate covers the complete six-section product terminal. -/
private def phaserRationalCheck : Bool :=
  let referencePoles := phaserPoleValues.extract 0 2
  let frequencies : Array Float := #[
    20.0, 162.211, 699.9, 700.0, 700.1, 3020.766, 20000.0]
  let mixes : Array Float := #[0.0, 0.5, 1.0]
  mixes.all fun mix =>
    match Tropical.Testing.ArrowFixtures.freezeBuild {} do
        let source ← sourceMode
        let tails ← phaserTails
        let mixE ← litF mix
        let reference ← (← Bank.ofFuture #[source]).phaser (tails.extract 0 2) mixE
        let decorated ← decorateDegreeZeroCausalPhaser #[source] tails mixE
        let compact ← Bank.ofFuture decorated
        let referenceValues ← frequencies.mapM fun frequency =>
          bankTransfer reference (6.283185307179586 * frequency)
        let compactValues ← frequencies.mapM fun frequency =>
          bankTransfer compact (6.283185307179586 * frequency)
        pure (referenceValues, compactValues) with
    | .error _ => false
    | .ok (arena, (referenceValues, compactValues)) =>
      let constants := sigConstTable arena
      (Array.range frequencies.size).all fun index =>
        let omega := 6.283185307179586 * frequencies[index]!
        match foldCplx constants referenceValues[index]!,
            foldCplx constants compactValues[index]! with
        | some actual, some decorated =>
            let referenceExpected := phaserOracleFor referencePoles omega mix
            let compactExpected := phaserOracleFor phaserPoleValues omega mix
            cdistF actual referenceExpected < 2.0e-9 &&
              cdistF decorated compactExpected < 2.0e-9
        | _, _ => false

private def wetUnitMagnitudeCheck : Bool :=
  let frequencies : Array Float := #[20.0, 40.0, 100.0, 700.0, 4000.0, 20000.0]
  frequencies.all fun frequency =>
    let omega := 6.283185307179586 * frequency
    let s : CplxF := (0.0, omega)
    let wet := phaserPoleValues.foldl (fun value a =>
      cmulF value (cdivF (s.1 - a, s.2) (s.1 + a, s.2))) (1.0, 0.0)
    close (Float.sqrt (wet.1 * wet.1 + wet.2 * wet.2)) 1.0 2.0e-15

/-- Frozen LTI room/phaser commutation across FF/interior/PP room orientation,
    witnessed with two sequential sections so the intentionally uncollected
    generic reference remains small.  The section law composes inductively;
    full six-section product-terminal behavior is gated separately.  This is
    the numerical law used by the exact two-room canonicalizer; no such rewrite
    is defined for gauge or bloom stages. -/
private def phaserRoomCommutationCheck : Bool :=
  let directions : Array (Int × Nat) := #[(0, 0), (4, 1), (1, 0)]
  let frequencies : Array Float := #[31.0, 700.0, 4700.0]
  directions.all fun (mantissa, exponent) =>
    match Tropical.Testing.ArrowFixtures.freezeBuild {} do
        let room : Array ModalMode := #[{
          sigma := ← lit 11, omega := ← lit 37,
          cre := ← lit 7 1, cim := ← lit (-2) 1 }]
        let direction ← lit mantissa exponent
        let source ← Bank.ofFuture #[← sourceMode]
        let witnessTails := (← phaserTails).extract 0 2
        let roomThenPhaser ← (← source.convolveKernel room direction
          syntacticSameSideClassifier).phaser witnessTails (← lit 5 1)
        let phaserThenRoom ← (← source.phaser witnessTails (← lit 5 1)).convolveKernel
          room direction syntacticSameSideClassifier
        let left ← frequencies.mapM fun frequency =>
          bankTransfer roomThenPhaser (6.283185307179586 * frequency)
        let right ← frequencies.mapM fun frequency =>
          bankTransfer phaserThenRoom (6.283185307179586 * frequency)
        pure (left, right) with
    | .error _ => false
    | .ok (arena, (left, right)) =>
      let constants := sigConstTable arena
      (Array.range frequencies.size).all fun index =>
        match foldCplx constants left[index]!, foldCplx constants right[index]! with
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
