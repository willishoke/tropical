import Tropical.EmitArrow.ArenaModal.Residue

/-!
# Arena-native per-kernel oriented modal convolution

The modal carrier remains factored by future and past support.  Scalar
construction is sequenced in `BuildM`; topology, degrees, and route choices
remain pure and retain authored order.
-/

namespace Tropical.EmitArrow.ArenaNative.Oriented

open Tropical.Ir

inductive Orientation where
  | future
  | past
deriving BEq, Repr

structure Atom where
  pole : CplxE
  amp : CplxE
  deg : Nat := 0

structure DirectedAtom extends Atom where
  orientation : Orientation

structure Expansion where
  future : Array Atom := #[]
  past : Array Atom := #[]
  atZero : CplxE

structure Bank where
  future : Array ModalMode := #[]
  past : Array ModalMode := #[]
  atZero : CplxE

inductive SameSideRoute where
  | distinct
  | coincident
deriving BEq, Repr

def natE (n : Nat) : BuildM CplxE := do
  let value ← lit (Int.ofNat n)
  let zero ← lit 0
  pure (value, zero)

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

def cpowE (base : CplxE) : Nat → BuildM CplxE
  | 0 => natE 1
  | n + 1 => do
      let power ← cpowE base n
      cmulE power base

def scaledPoleQuotient (amp : CplxE) (numerator factorial : Nat)
    (pole : CplxE) (power : Nat) (negative : Bool := false) : BuildM CplxE := do
  let numerator ← natE numerator
  let signed ← if negative then cnegE numerator else pure numerator
  let scaledAmp ← cmulE amp signed
  let factorial ← natE factorial
  let polePower ← cpowE pole power
  let denominator ← cmulE factorial polePower
  cdivE scaledAmp denominator

def scaledNatQuotient (amp : CplxE) (numerator denominator : Nat) : BuildM CplxE := do
  let numerator ← natE numerator
  let scaled ← cmulE amp numerator
  let denominator ← natE denominator
  cdivE scaled denominator

def Atom.ofMode (mode : ModalMode) : BuildM Atom := do
  pure { pole := ← mode.poleE, amp := mode.ampE, deg := mode.deg }

def Atom.toMode (atom : Atom) : BuildM ModalMode :=
  modeOfE atom.pole atom.amp atom.deg

def Bank.ofFuture (modes : Array ModalMode) : BuildM Bank := do
  let zero ← natE 0
  pure { future := modes, atZero := zero }

def scaleModeAmp (scale : CplxE) (mode : ModalMode) : BuildM ModalMode := do
  let amp ← cmulE mode.ampE scale
  pure { mode with cre := amp.1, cim := amp.2 }

def Bank.kernel (modes : Array ModalMode) (direction : Sig) : BuildM Bank := do
  let one ← lit 1
  let forward ← sub one direction
  let zero ← lit 0
  let future ← modes.mapM (scaleModeAmp (forward, zero))
  let past ← modes.mapM (scaleModeAmp (direction, zero))
  pure { future, past, atZero := (zero, zero) }

def Atom.orientKernel (atom : Atom) (direction : Sig) : BuildM (Array DirectedAtom) := do
  let one ← lit 1
  let forward ← sub one direction
  let zero ← lit 0
  let futureAmp ← cmulE atom.amp (forward, zero)
  let pastAmp ← cmulE atom.amp (direction, zero)
  pure #[{ atom with amp := futureAmp, orientation := .future },
    { atom with amp := pastAmp, orientation := .past }]

private def onSide (orientation : Orientation) (atoms : Array Atom)
    (atZero : CplxE) : Expansion :=
  match orientation with
  | .future => { future := atoms, atZero }
  | .past => { past := atoms, atZero }

def Expansion.add (left right : Expansion) : BuildM Expansion := do
  let atZero ← caddE left.atZero right.atZero
  pure { future := left.future ++ right.future, past := left.past ++ right.past, atZero }

def Expansion.toBank (expansion : Expansion) : BuildM Bank := do
  let future ← expansion.future.mapM Atom.toMode
  let past ← expansion.past.mapM Atom.toMode
  pure { future, past, atZero := expansion.atZero }

def convolveSameSideDistinct (orientation : Orientation) (left right : Atom) :
    BuildM Expansion := do
  let p := left.deg
  let q := right.deg
  let ab ← cmulE left.amp right.amp
  let delta ← csubE left.pole right.pole
  let atLeft ← (Array.range (p + 1)).mapM fun r => do
    let distance := p - r
    let numerator := factorial p * factorial q * choose (q + p - r) distance
    let amp ← scaledPoleQuotient ab numerator (factorial r) delta
      (q + p - r + 1) (distance % 2 == 1)
    pure ({ pole := left.pole, amp, deg := r } : Atom)
  let atRight ← (Array.range (q + 1)).mapM fun s => do
    let distance := q - s
    let numerator := factorial p * factorial q * choose (p + q - s) distance
    let amp ← scaledPoleQuotient ab numerator (factorial s) delta
      (p + q - s + 1) ((p + 1) % 2 == 1)
    pure ({ pole := right.pole, amp, deg := s } : Atom)
  let zero ← natE 0
  pure (onSide orientation (atLeft ++ atRight) zero)

def convolveSameSideCoincident (orientation : Orientation)
    (left right : Atom) : BuildM Expansion := do
  let p := left.deg
  let q := right.deg
  let ab ← cmulE left.amp right.amp
  let amp ← scaledNatQuotient ab (factorial p * factorial q)
    (factorial (p + q + 1))
  let zero ← natE 0
  pure (onSide orientation #[{ pole := left.pole, amp, deg := p + q + 1 }] zero)

def convolveSameSide (route : SameSideRoute) (orientation : Orientation)
    (left right : Atom) : BuildM Expansion :=
  match route with
  | .distinct => convolveSameSideDistinct orientation left right
  | .coincident => convolveSameSideCoincident orientation left right

def convolveFuturePast (future past : Atom) : BuildM Expansion := do
  let p := future.deg
  let q := past.deg
  let ab ← cmulE future.amp past.amp
  let poleSum ← caddE future.pole past.pole
  let rho ← cnegE poleSum
  let futureTerms ← (Array.range (p + 1)).mapM fun i => do
    let numerator := choose p i * factorial (q + i)
    let amp ← scaledPoleQuotient ab numerator 1 rho (q + i + 1)
    pure ({ pole := future.pole, amp, deg := p - i } : Atom)
  let pastTerms ← (Array.range (q + 1)).mapM fun j => do
    let numerator := choose q j * factorial (p + j)
    let amp ← scaledPoleQuotient ab numerator 1 rho (p + j + 1)
    pure ({ pole := past.pole, amp, deg := q - j } : Atom)
  let atZero ← scaledPoleQuotient ab (factorial (p + q)) 1 rho (p + q + 1)
  pure { future := futureTerms, past := pastTerms, atZero }

def convolvePastFuture (past future : Atom) : BuildM Expansion :=
  convolveFuturePast future past

def convolve (sameSideRoute : SameSideRoute) (left right : DirectedAtom) :
    BuildM Expansion :=
  match left.orientation, right.orientation with
  | .future, .future => convolveSameSide sameSideRoute .future left.toAtom right.toAtom
  | .past, .past => convolveSameSide sameSideRoute .past left.toAtom right.toAtom
  | .future, .past => convolveFuturePast left.toAtom right.toAtom
  | .past, .future => convolvePastFuture left.toAtom right.toAtom

abbrev SameSideClassifier :=
  Orientation → ModalMode → ModalMode → SameSideRoute

def syntacticSameSideClassifier : SameSideClassifier := fun _ left right =>
  if left.sigma == right.sigma && left.omega == right.omega then .coincident
  else .distinct

private def convolveSameSideModes (classify : SameSideClassifier)
    (orientation : Orientation) (left right : ModalMode) : BuildM Expansion := do
  match classify orientation left right with
  | .distinct =>
      if left.deg == 0 && right.deg == 0 then
        let modes ← residueComposeE #[left] #[right]
        let atoms ← modes.mapM Atom.ofMode
        let zero ← natE 0
        pure (onSide orientation atoms zero)
      else
        convolveSameSideDistinct orientation (← Atom.ofMode left) (← Atom.ofMode right)
  | .coincident =>
      convolveSameSideCoincident orientation (← Atom.ofMode left) (← Atom.ofMode right)

private def emptyExpansion : BuildM Expansion := do
  pure { atZero := ← natE 0 }

private def convolvePairs (left right : Array ModalMode)
    (pair : ModalMode → ModalMode → BuildM Expansion) : BuildM Expansion := do
  let initial ← emptyExpansion
  left.foldlM (fun accumulated leftMode =>
    right.foldlM (fun accumulated rightMode => do
      let expansion ← pair leftMode rightMode
      accumulated.add expansion) accumulated) initial

private def sumAtDifference (pole : CplxE)
    (modes : Array ModalMode) : BuildM CplxE := do
  let zero ← natE 0
  modes.foldlM (fun total mode => do
    let modePole ← mode.poleE
    let difference ← csubE pole modePole
    let quotient ← cdivE mode.ampE difference
    caddE total quotient) zero

private def sumAtPhysicalSum (pole : CplxE)
    (modes : Array ModalMode) : BuildM CplxE := do
  let zero ← natE 0
  modes.foldlM (fun total mode => do
    let modePole ← mode.poleE
    let sum ← caddE pole modePole
    let quotient ← cdivE mode.ampE sum
    caddE total quotient) zero

private def mixedAtZero (future past : Array ModalMode) : BuildM CplxE := do
  let zero ← natE 0
  future.foldlM (fun total futureMode =>
    past.foldlM (fun total pastMode => do
      let numerator ← cmulE futureMode.ampE pastMode.ampE
      let futurePole ← futureMode.poleE
      let pastPole ← pastMode.poleE
      let denominator ← caddE futurePole pastPole
      let quotient ← cdivE numerator denominator
      csubE total quotient) total) zero

private def convolveKernelDegZeroCollected (input kernel : Bank) : BuildM Bank := do
  let inputFuture ← input.future.mapM fun mode => do
    let pole ← mode.poleE
    let difference ← sumAtDifference pole kernel.future
    let physical ← sumAtPhysicalSum pole kernel.past
    scaleModeAmp (← csubE difference physical) mode
  let kernelFuture ← kernel.future.mapM fun mode => do
    let pole ← mode.poleE
    let difference ← sumAtDifference pole input.future
    let physical ← sumAtPhysicalSum pole input.past
    scaleModeAmp (← csubE difference physical) mode
  let inputPast ← input.past.mapM fun mode => do
    let pole ← mode.poleE
    let difference ← sumAtDifference pole kernel.past
    let physical ← sumAtPhysicalSum pole kernel.future
    scaleModeAmp (← csubE difference physical) mode
  let kernelPast ← kernel.past.mapM fun mode => do
    let pole ← mode.poleE
    let difference ← sumAtDifference pole input.past
    let physical ← sumAtPhysicalSum pole input.future
    scaleModeAmp (← csubE difference physical) mode
  let leftZero ← mixedAtZero input.future kernel.past
  let rightZero ← mixedAtZero kernel.future input.past
  let atZero ← caddE leftZero rightZero
  pure { future := inputFuture ++ kernelFuture, past := inputPast ++ kernelPast, atZero }

private def modesDegreeZero (modes : Array ModalMode) : Bool :=
  modes.all fun mode => mode.deg == 0

private def sameSidePairsDistinct (classify : SameSideClassifier)
    (orientation : Orientation) (left right : Array ModalMode) : Bool :=
  left.all fun leftMode => right.all fun rightMode =>
    classify orientation leftMode rightMode == .distinct

private def canCollectDegreeZero (classify : SameSideClassifier)
    (input kernel : Bank) : Bool :=
  modesDegreeZero input.future && modesDegreeZero input.past &&
    modesDegreeZero kernel.future && modesDegreeZero kernel.past &&
    sameSidePairsDistinct classify .future input.future kernel.future &&
    sameSidePairsDistinct classify .past input.past kernel.past

private def convolveKernelPairwise (classify : SameSideClassifier)
    (input kernel : Bank) : BuildM Bank := do
  let ff ← convolvePairs input.future kernel.future
    (convolveSameSideModes classify .future)
  let pp ← convolvePairs input.past kernel.past
    (convolveSameSideModes classify .past)
  let fp ← convolvePairs input.future kernel.past fun future past => do
    convolveFuturePast (← Atom.ofMode future) (← Atom.ofMode past)
  let pf ← convolvePairs input.past kernel.future fun past future => do
    convolvePastFuture (← Atom.ofMode past) (← Atom.ofMode future)
  let combined ← (← (← (← ff.add pp).add fp).add pf).toBank
  pure combined

def Bank.convolveKernel (input : Bank) (room : Array ModalMode)
    (direction : Sig) (classify : SameSideClassifier) : BuildM Bank := do
  let kernel ← Bank.kernel room direction
  if canCollectDegreeZero classify input kernel then
    convolveKernelDegZeroCollected input kernel
  else
    convolveKernelPairwise classify input kernel

def Bank.scale (bank : Bank) (scale : CplxE) : BuildM Bank := do
  let future ← bank.future.mapM (scaleModeAmp scale)
  let past ← bank.past.mapM (scaleModeAmp scale)
  let atZero ← cmulE scale bank.atZero
  pure { future, past, atZero }

protected def Bank.add (left right : Bank) : BuildM Bank := do
  let atZero ← caddE left.atZero right.atZero
  pure { future := left.future ++ right.future, past := left.past ++ right.past, atZero }

def Bank.blend (dry wet : Bank) (mix : Sig) : BuildM Bank := do
  let one ← lit 1
  let dryAmount ← sub one mix
  let zero ← lit 0
  let dry ← dry.scale (dryAmount, zero)
  let wet ← wet.scale (mix, zero)
  dry.add wet

def allpassTail (a : Sig)
    (sigmaRange : Option (Float × Float) := none) : BuildM ModalMode := do
  let zero ← lit 0
  let two ← lit 2
  let doubled ← mul two a
  let cre ← neg doubled
  pure { sigma := a, omega := zero, cre, cim := zero, sigmaRange }

def Bank.allpassSection (bank : Bank) (tail : ModalMode) : BuildM Bank := do
  let zero ← lit 0
  let convolved ← bank.convolveKernel #[tail] zero syntacticSameSideClassifier
  bank.add convolved

def Bank.phaser (bank : Bank) (tails : Array ModalMode) (mix : Sig) : BuildM Bank := do
  let wet ← tails.foldlM (fun value tail => value.allpassSection tail) bank
  bank.blend wet mix

def decorateDegreeZeroCausalPhaser (source tails : Array ModalMode)
    (mix : Sig) : BuildM (Array ModalMode) := do
  let wet ← tails.foldlM (fun modes tail => do
    let tailPole ← tail.poleE
    let zero ← natE 0
    let atTail ← modes.foldlM (fun total mode => do
      let modePole ← mode.poleE
      let difference ← csubE tailPole modePole
      let quotient ← cdivE mode.ampE difference
      caddE total quotient) zero
    let existing ← modes.mapM fun mode => do
      let one ← natE 1
      let modePole ← mode.poleE
      let difference ← csubE modePole tailPole
      let quotient ← cdivE tail.ampE difference
      let gain ← caddE one quotient
      scaleModeAmp gain mode
    let tail ← scaleModeAmp atTail tail
    pure (existing.push tail)) source
  let one ← lit 1
  let dryAmount ← sub one mix
  let zero ← lit 0
  let dryScale : CplxE := (dryAmount, zero)
  let wetScale : CplxE := (mix, zero)
  let sourceRows ← (wet.extract 0 source.size).zip source |>.mapM fun (wetMode, dryMode) => do
    let wetAmp ← cmulE wetScale wetMode.ampE
    let dryAmp ← cmulE dryScale dryMode.ampE
    let amp ← caddE wetAmp dryAmp
    pure { wetMode with cre := amp.1, cim := amp.2 }
  let sectionRows ← (wet.extract source.size wet.size).mapM (scaleModeAmp wetScale)
  pure (sourceRows ++ sectionRows)

end Tropical.EmitArrow.ArenaNative.Oriented
