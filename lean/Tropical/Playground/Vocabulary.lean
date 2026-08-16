import Tropical.Playground.VocabularyMetadata
import Tropical.EmitArrow.Gong

/-!
# Playground.Vocabulary

Arena-native scalar and modal builders for the production playground decoder.
The public vocabulary tables remain pure in `Playground.VocabularyMetadata`;
this file only owns constructions that allocate expression nodes.
-/

namespace Tropical.Playground.Compiler

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Exact (DyadicI)

def jExpr (obj : Json) (key : String) (dflt : Sig) : BuildM Sig :=
  match Tropical.Playground.Metadata.jNum? obj key with
  | some n => lit n.mantissa n.exponent
  | none => pure dflt

def litOfD? (x : DyadicI) : BuildM (Option Sig) :=
  if x.ok then return some (← litF x.toFloat) else pure none

def litOfD (x : DyadicI) : BuildM Sig := do
  match ← litOfD? x with
  | some value => pure value
  | none => lit 0

def lnEightyLit : BuildM Sig := litOfD Tropical.Playground.Metadata.ln80D

/-- Decode a baked modal table directly into arena IDs. -/
def jModes (obj : Json) (key : String) : BuildM (Array ModalMode) := do
  let some rows := (obj.getObjVal? key).toOption.bind (·.getArr?.toOption)
    | return #[]
  let mut modes : Array ModalMode := #[]
  for row in rows do
    let some cells := row.getArr?.toOption | continue
    if cells.size < 4 then continue
    let numD := fun (i : Nat) =>
      match cells[i]!.getNum?.toOption with
      | some n => DyadicI.ofJsonNumber n
      | none => DyadicI.zero
    let amplitude := numD 2
    let phase := numD 3
    let sigma ← litOfD (numD 1)
    let omega ← litOfD (DyadicI.mul Tropical.Playground.Metadata.twoPiD (numD 0))
    let cre ← litOfD (DyadicI.mul amplitude (DyadicI.cos phase))
    let cim ← litOfD (DyadicI.mul amplitude (DyadicI.sin phase))
    modes := modes.push { sigma, omega, cre, cim }
  pure modes

def phaseCorr (pitchE freqInit : Sig) : Clock → BuildM Sig :=
  fun shift => do
    let difference ← sub pitchE freqInit
    let shiftFloat ← toFloatE shift
    let numerator ← mul difference shiftFloat
    let twoPow32 ← lit 4294967296
    let sr ← sampleRate
    let denominator ← mul twoPow32 sr
    div numerator denominator

abbrev Anchor := Sig × Sig

def sineVoiceE (pitchE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "FixedSinOsc"
      wire := fun clkE => pure #[⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩, ⟨⟨2⟩, phaseE⟩]
      phaseAnchor := some (⟨2⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "FixedSinOsc"
      wire := fun clkE => pure #[⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩] }

def morphVoiceE (pitchE morphE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "MorphOsc"
      wire := fun clkE => pure #[⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩,
        ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, phaseE⟩]
      phaseAnchor := some (⟨3⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "MorphOsc"
      wire := fun clkE => pure #[⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩] }

def voiceOf (pitchE morphE : Sig) (anchor : Option Anchor) : Voice :=
  morphVoiceE pitchE morphE anchor

def pluckedVoiceE (pitchE morphE eventRateE : Sig)
    (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "PluckedMorphOsc"
      wire := fun clkE => pure #[⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩,
        ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, eventRateE⟩, ⟨⟨4⟩, phaseE⟩]
      phaseAnchor := some (⟨4⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "PluckedMorphOsc"
      wire := fun clkE => pure #[⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩,
        ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, eventRateE⟩] }

def deltaOf (secondsE : Sig) : BuildM Sig := do
  let sr ← sampleRate
  let samples ← mul secondsE sr
  let twoPow32 ← lit 4294967296
  let fixed ← mul samples twoPow32
  toIntE fixed

def gPow (g : Sig) (k : Nat) : BuildM Sig := do
  let one ← lit 1
  (Array.range k).foldlM (fun acc _ => mul acc g) one

def resonatorBank (f0 decay : Sig) (npart : Nat) : BuildM (Array ModalMode) :=
  (Array.range npart).mapM fun j => do
    let k := j + 1
    let kD := DyadicI.ofNat k
    let sigFac := DyadicI.add DyadicI.one
      (DyadicI.mul (Tropical.Playground.Metadata.decD (4, 1)) kD)
    let ampD := DyadicI.div DyadicI.one
      (DyadicI.pow kD (Tropical.Playground.Metadata.decD (11, 1)))
    let kSig ← lit (Int.ofNat k)
    let frequency ← mul kSig f0
    let sigmaFactor ← litOfD sigFac
    let sigma ← mul decay sigmaFactor
    let amplitude ← litOfD ampD
    ModalMode.hz frequency sigma amplitude

def defaultStringModes (f0 rho : Int × Nat) : BuildM (Array ModalMode) := do
  let srI : Int := 44100
  let (mantissa, exponent) := f0
  if mantissa ≤ 0 then return #[]
  let numerator : Int := srI * (10 : Int) ^ exponent
  let denominator : Int := mantissa
  let n : Int := (2 * numerator + denominator) / (2 * denominator)
  if n ≤ 0 then return #[]
  let span : DyadicI := (DyadicI.ofInt (2 * n + 1)).shift (-1 : Int)
  let kmax : Nat := min 48 (n / 2).toNat
  let rhoD := Tropical.Playground.Metadata.decD rho
  let srD := DyadicI.ofInt srI
  let halfSR := srD.shift (-1 : Int)
  let mut modes : Array ModalMode := #[]
  for j in [0:kmax] do
    let k := DyadicI.ofNat (j + 1)
    let frequency := DyadicI.div (DyadicI.mul k srD) span
    let gain := DyadicI.mul rhoD (DyadicI.cos
      (DyadicI.div (DyadicI.mul Tropical.Playground.Metadata.piD k) span))
    if DyadicI.certLt frequency halfSR && DyadicI.certGt gain DyadicI.zero then
      let sigmaD := DyadicI.neg
        (DyadicI.div (DyadicI.mul srD (DyadicI.log gain)) span)
      let amplitude := DyadicI.inv k
      let phase := DyadicI.mul Tropical.Playground.Metadata.goldenAngleD k
      let sigma ← litOfD sigmaD
      let omega ← litOfD (DyadicI.mul Tropical.Playground.Metadata.twoPiD frequency)
      let cre ← litOfD (DyadicI.mul amplitude (DyadicI.cos phase))
      let cim ← litOfD (DyadicI.mul amplitude (DyadicI.sin phase))
      modes := modes.push { sigma, omega, cre, cim }
  pure modes

def reverbRoom (rt60 : Sig) (rtRange : Option (Float × Float))
    (nmode : Nat) (flo fhi : Int × Nat) : BuildM (Array ModalMode) := do
  let c691Sig ← lit 691 2
  let sigma ← div c691Sig rt60
  let c691 : DyadicI := Tropical.Playground.Metadata.decD (691, 2)
  let sigmaRange := rtRange.map fun (lo, hi) =>
    ((DyadicI.div c691 (DyadicI.ofFloat hi)).toFloat,
     (DyadicI.div c691 (DyadicI.ofFloat lo)).toFloat)
  let floD := Tropical.Playground.Metadata.decD flo
  let fhiD := Tropical.Playground.Metadata.decD fhi
  let ratio := DyadicI.div fhiD floD
  let denominator : DyadicI := DyadicI.ofNat (if nmode ≤ 1 then 1 else nmode - 1)
  (Array.range nmode).mapM fun j => do
    let frequency := DyadicI.mul floD
      (DyadicI.pow ratio (DyadicI.div (DyadicI.ofNat j) denominator))
    let phase := DyadicI.mul Tropical.Playground.Metadata.twoPiD
      (DyadicI.mul Tropical.Playground.Metadata.goldenRatioD (DyadicI.ofNat j))
    let frequencySig ← litOfD frequency
    let twoPi ← twoPiE
    let omega ← mul twoPi frequencySig
    let cre ← litOfD (DyadicI.cos phase)
    let cim ← litOfD (DyadicI.sin phase)
    pure { sigma, omega, cre, cim, sigmaRange }

/-- Arena-native coefficient probes used by the independent exact/numeric
    suites.  The returned IDs are meaningful only with the builder/frozen arena
    that owns them. -/
def bakedResonatorProbe (npart : Nat) : BuildM (Array ModalMode) := do
  resonatorBank (← lit 1) (← lit 1) npart

def bakedReverbProbe (nmode : Nat) : BuildM (Array ModalMode) := do
  reverbRoom (← lit 1) none nmode (60, 0) (6000, 0)

def bakedFilterLn80 : BuildM Sig := lnEightyLit

def filterPair (fc res : Sig) : BuildM (Array ModalMode) := do
  let twoPi ← twoPiE
  let w0 ← mul twoPi fc
  let qBase ← lit 55 2
  let ln80 ← lnEightyLit
  let qPower ← mul res ln80
  let qExp ← expSig qPower
  let q ← mul qBase qExp
  let two ← lit 2
  let twoQ ← mul two q
  let alpha ← div w0 twoQ
  let one ← lit 1
  let four ← lit 4
  let qSquared ← mul q q
  let fourQSquared ← mul four qSquared
  let reciprocal ← div one fourQSquared
  let radicand ← sub one reciprocal
  let root ← unary .sqrt radicand
  let wd ← mul w0 root
  let w0Squared ← mul w0 w0
  let twoWd ← mul two wd
  let rim ← div w0Squared twoWd
  let zero ← lit 0
  let negativeRim ← neg rim
  let negativeWd ← neg wd
  pure #[{ sigma := alpha, omega := wd, cre := zero, cim := negativeRim },
    { sigma := alpha, omega := negativeWd, cre := zero, cim := rim }]

def pref (pidx : String → Option Nat) (name : String) (dflt : Sig) : BuildM Sig :=
  match pidx name with
  | some i => paramRef ⟨i⟩
  | none => pure dflt

def fallbackOf (kind kname : String) : BuildM Sig :=
  match (Tropical.Playground.Metadata.portOf kind kname).bind (·.knob) with
  | some (mantissa, exponent) => lit mantissa exponent
  | none => lit 0

def exactI64Companion (pidx : String → Option Nat) (base : String) : BuildM Sig := do
  let sample ← sampleIndex
  let sampleZero ← sub sample sample >>= toFloatE
  let limb (i : Nat) : BuildM Sig := do
    let zero ← lit 0
    let slot ← pref pidx s!"{base}#u{i}" zero
    let pinned ← add slot sampleZero
    toIntE pinned
  let l0 ← limb 0
  let l1 ← limb 1
  let l2 ← limb 2
  let l3 ← limb 3
  let sixteen ← lit 16
  let thirtyTwo ← lit 32
  let fortyEight ← lit 48
  let s1 ← lshift l1 sixteen
  let s2 ← lshift l2 thirtyTwo
  let s3 ← lshift l3 fortyEight
  let low ← bitOr l0 s1
  let high ← bitOr s2 s3
  bitOr low high

def glideExprAt (pidx : String → Option Nat) (base : String)
    (dflt coordinate : Sig) : BuildM Sig := do
  let v0 ← pref pidx s!"{base}#v0" dflt
  let v1 ← pref pidx s!"{base}#v1" dflt
  let zero ← lit 0
  let t0 ← pref pidx s!"{base}#t0" zero
  let exactNames := (Array.range 4).map fun i => s!"{base}#t0#u{i}"
  let elapsed ← if exactNames.all (fun name => (pidx name).isSome) then do
      let exact ← exactI64Companion pidx s!"{base}#t0"
      let delta ← sub coordinate exact
      toFloatE delta
    else do
      let coordinate ← toFloatE coordinate
      sub coordinate t0
  let durationFactor ← lit 5 3
  let sr ← sampleRate
  let duration ← mul durationFactor sr
  let fraction ← div elapsed duration
  let one ← lit 1
  let s ← clampE fraction zero one
  let square ← mul s s
  let two ← lit 2
  let twiceS ← mul two s
  let three ← lit 3
  let shoulder ← sub three twiceS
  let smooth ← mul square shoulder
  let delta ← sub v1 v0
  let ramp ← mul delta smooth
  add v0 ramp

def q32DeltaSamples (coordinateQ originQ : Sig) : BuildM Sig := do
  let delta ← sub coordinateQ originQ
  let delta ← toFloatE delta
  let scale ← lit 4294967296
  div delta scale

def glideExprQAt (pidx : String → Option Nat) (base : String)
    (dflt coordinateQ : Sig) : BuildM Sig := do
  let v0 ← pref pidx s!"{base}#v0" dflt
  let v1 ← pref pidx s!"{base}#v1" dflt
  let zero ← lit 0
  let t0 ← pref pidx s!"{base}#t0" zero
  let exactNames := (Array.range 4).map fun i => s!"{base}#t0#u{i}"
  let qScale ← lit 4294967296
  let elapsed ← if exactNames.all (fun name => (pidx name).isSome) then do
      let exact ← exactI64Companion pidx s!"{base}#t0"
      let thirtyTwo ← lit 32
      let origin ← lshift exact thirtyTwo
      q32DeltaSamples coordinateQ origin
    else do
      let coordinate ← toFloatE coordinateQ
      let samples ← div coordinate qScale
      sub samples t0
  let durationFactor ← lit 5 3
  let sr ← sampleRate
  let duration ← mul durationFactor sr
  let fraction ← div elapsed duration
  let one ← lit 1
  let s ← clampE fraction zero one
  let square ← mul s s
  let two ← lit 2
  let twiceS ← mul two s
  let three ← lit 3
  let shoulder ← sub three twiceS
  let smooth ← mul square shoulder
  let delta ← sub v1 v0
  let ramp ← mul delta smooth
  add v0 ramp

def glideExpr (pidx : String → Option Nat) (base : String) (dflt : Sig) : BuildM Sig := do
  let coordinate ← sampleIndex
  glideExprAt pidx base dflt coordinate

end Tropical.Playground.Compiler
