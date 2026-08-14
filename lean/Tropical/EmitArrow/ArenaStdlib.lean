import Tropical.EmitArrow.ArenaNumerics

/-!
# EmitArrow.ArenaStdlib — arena-native standard-library authoring

The production stdlib builders allocate directly in the `ExprArena` owned by
`ArenaNative.Builder`.  Inputs, instance wiring, and output assignments carry
stable IDs from the same build action; no recursive `Sig` or `lowerSig` path is
involved.
-/

namespace Tropical.EmitArrow.ArenaNative

open Tropical.Ir

abbrev StdBuilder :=
  Arena → Array (String × ProgramIdx) → Except String (Arena × ProgramIdx)

def freqDecl (name : String) (mantissa : Int) (exponent : Nat := 0) :
    BuildM AInputDecl := do
  let hz ← lit mantissa exponent
  let zero ← lit 0
  let positive ← gt hz zero
  let default ← selectE positive hz zero
  pure { name, type? := some (.scalar .float), defaultSig := some default }

def unipolarDecl (name : String) (default? : Option Sig := none) :
    BuildM AInputDecl := do
  let value ← match default? with
    | some value => pure value
    | none => lit 0
  let zero ← lit 0
  let one ← lit 1
  let default ← clampE value zero one
  pure { name, type? := some (.scalar .float), defaultSig := some default }

def signalDecl (name : String) (default? : Option Sig := none) :
    BuildM AInputDecl := do
  let value ← match default? with
    | some value => pure value
    | none => lit 0
  let lo ← lit (-1)
  let hi ← lit 1
  let default ← clampE value lo hi
  pure { name, type? := some (.scalar .float), defaultSig := some default }

def floatDecl (name : String) (default : Sig) : AInputDecl :=
  { name, type? := some (.scalar .float), defaultSig := some default }

def clockInDecl (name : String) (default : Sig) : AInputDecl :=
  { name, type? := some (.scalar .int), defaultSig := some default }

def clockSig : BuildM Sig := do
  let sample ← sampleIndex
  let shift ← lit 32
  lshift sample shift

def floatOut (name : String := "out") : OutputDecl :=
  { name, type? := some (.scalar .float) }

def buildSin (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  assemble arena "Sin" #[floatOut] #[] do
    let x ← inputRef ⟨0⟩
    let invPi ← lit 3183098861837907 16
    let xInvPi ← mul x invPi
    let n ← roundE xInvPi
    let pi ← lit 3141592653589793 15
    let nPi ← mul n pi
    let r ← sub x nPi
    let one ← lit 1
    let two ← lit 2
    let parity ← bitAnd n one
    let twiceParity ← mul two parity
    let sign ← sub one twiceParity
    let r2 ← mul r r
    let step := fun (acc coeff : Sig) => do
      let product ← mul acc r2
      add coeff product
    let zero ← lit 0
    let c5 ← lit (-2505210838544172) 23
    let a ← step zero c5
    let c4 ← lit 27557319223985893 22
    let a ← step a c4
    let c3 ← lit (-1984126984126984) 19
    let a ← step a c3
    let c2 ← lit 8333333333333333 18
    let a ← step a c2
    let c1 ← lit (-16666666666666666) 17
    let a ← step a c1
    let poly ← step a one
    let rPoly ← mul r poly
    let body ← mul sign rPoly
    let input := floatDecl "x" zero
    pure { inputs := #[input], assigns := #[(.port ⟨0⟩, body)] }

def buildTanh (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  assemble arena "Tanh" #[floatOut] #[] do
    let x ← inputRef ⟨0⟩
    let lo ← lit (-3)
    let hi ← lit 3
    let c ← clampE x lo hi
    let c2 ← mul c c
    let twentySeven ← lit 27
    let numeratorInner ← add twentySeven c2
    let numerator ← mul c numeratorInner
    let nine ← lit 9
    let nineC2 ← mul nine c2
    let denominator ← add twentySeven nineC2
    let body ← div numerator denominator
    let zero ← lit 0
    pure {
      inputs := #[floatDecl "x" zero]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildScrubClock (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  assemble arena "ScrubClock"
      #[{ name := "clk", type? := some (.scalar .int) }] #[] do
    let tauBase ← inputRef ⟨0⟩
    let velocity ← inputRef ⟨1⟩
    let sr ← sampleRate
    let baseSamples ← mul tauBase sr
    let twoPow32 ← lit 4294967296
    let baseFixed ← mul baseSamples twoPow32
    let base ← toIntE baseFixed
    let velocityFixed ← mul velocity twoPow32
    let velocityInt ← toIntE velocityFixed
    let sample ← sampleIndex
    let ramp ← mul velocityInt sample
    let body ← add base ramp
    let zero ← lit 0
    let one ← lit 1
    pure {
      inputs := #[floatDecl "tau_base" zero, floatDecl "velocity" one]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildVCA (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  assemble arena "VCA" #[floatOut] #[] do
    let audio ← inputRef ⟨0⟩
    let cv ← inputRef ⟨1⟩
    let body ← mul audio cv
    let zero ← lit 0
    pure {
      inputs := #[floatDecl "audio" zero, floatDecl "cv" zero]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildClockPhasor (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  assemble arena "ClockPhasor"
      #[{ name := "phase", type? := some (.scalar .float) }] #[] do
    let clk ← inputRef ⟨0⟩
    let freq ← inputRef ⟨1⟩
    let offset ← inputRef ⟨2⟩
    let phase ← phasorPhaseSig freq offset clk
    let zero ← lit 0
    let one ← lit 1
    let body ← clampE phase zero one
    let clkInput := clockInDecl "clk" zero
    let freqInput ← freqDecl "freq" 440
    let offsetInput ← unipolarDecl "offset"
    pure {
      inputs := #[clkInput, freqInput, offsetInput]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildSoftClip (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["Tanh"]
  assemble arena "SoftClip"
      #[{ name := "out", type? := some (.scalar .float) }] registry do
    let input ← inputRef ⟨0⟩
    let drive ← inputRef ⟨1⟩
    let driven ← mul drive input
    let tanhIdx ← inst "tanh" "Tanh" #[{ port := ⟨0⟩, value := driven }]
    let tanhOut ← nestedOut tanhIdx ⟨0⟩
    let lo ← lit (-1)
    let hi ← lit 1
    let body ← clampE tanhOut lo hi
    let inputDecl ← signalDecl "input"
    let one ← lit 1
    pure {
      inputs := #[inputDecl, floatDecl "drive" one]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildModalVoice (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "Sin"]
  assemble arena "ModalVoice" #[floatOut] registry do
    let clk ← inputRef ⟨0⟩
    let f0 ← inputRef ⟨1⟩
    let phasor := fun (name : String) (frequency : Sig) =>
      inst name "ClockPhasor" #[
        { port := ⟨0⟩, value := clk }, { port := ⟨1⟩, value := frequency }]
    let p1 ← phasor "p1" f0
    let ratio2 ← lit 2414213562373095 15
    let f2 ← mul f0 ratio2
    let p2 ← phasor "p2" f2
    let ratio3 ← lit 423606797749979 14
    let f3 ← mul f0 ratio3
    let p3 ← phasor "p3" f3
    let ratio4 ← lit 6854101966249685 15
    let f4 ← mul f0 ratio4
    let p4 ← phasor "p4" f4
    let twoPi ← twoPiE
    let sine := fun (name : String) (phasorIdx : InstanceIdx) => do
      let phase ← nestedOut phasorIdx ⟨0⟩
      let angle ← mul twoPi phase
      inst name "Sin" #[{ port := ⟨0⟩, value := angle }]
    let s1 ← sine "s1" p1
    let s2 ← sine "s2" p2
    let s3 ← sine "s3" p3
    let s4 ← sine "s4" p4
    let weighted := fun (weightMantissa : Int) (weightExponent : Nat)
        (instanceIdx : InstanceIdx) => do
      let weight ← lit weightMantissa weightExponent
      let signal ← nestedOut instanceIdx ⟨0⟩
      mul weight signal
    let w1 ← weighted 4 1 s1
    let w2 ← weighted 24 2 s2
    let w3 ← weighted 16 2 s3
    let w4 ← weighted 1 1 s4
    let sum12 ← add w1 w2
    let sum123 ← add sum12 w3
    let body ← add sum123 w4
    let defaultClock ← clockSig
    let freqInput ← freqDecl "f0" 110
    pure {
      inputs := #[clockInDecl "clk" defaultClock, freqInput]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildPluckedMorphOsc (arena : Arena)
    (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["MorphOsc", "ClockPhasor"]
  assemble arena "PluckedMorphOsc" #[floatOut] registry do
    let freq ← inputRef ⟨0⟩
    let morph ← inputRef ⟨1⟩
    let clk ← inputRef ⟨2⟩
    let eventRate ← inputRef ⟨3⟩
    let phase ← inputRef ⟨4⟩
    let osc ← inst "osc" "MorphOsc" #[
      { port := ⟨0⟩, value := freq }, { port := ⟨1⟩, value := morph },
      { port := ⟨2⟩, value := clk }, { port := ⟨3⟩, value := phase }]
    let event ← inst "ev" "ClockPhasor" #[
      { port := ⟨0⟩, value := clk }, { port := ⟨1⟩, value := eventRate }]
    let f ← nestedOut event ⟨0⟩
    let one ← lit 1
    let u ← sub one f
    let u2 ← mul u u
    let scale ← lit 176 1
    let scaled ← mul scale f
    let env2 ← mul scaled u2
    let env4 ← mul env2 u2
    let env ← mul env4 u2
    let oscOut ← nestedOut osc ⟨0⟩
    let body ← mul oscOut env
    let freqInput ← freqDecl "freq" 220
    let morphInput ← unipolarDecl "morph"
    let defaultClock ← clockSig
    let eventInput ← freqDecl "event_rate" 1
    let phaseInput ← unipolarDecl "phase"
    pure {
      inputs := #[freqInput, morphInput, clockInDecl "clk" defaultClock,
        eventInput, phaseInput]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildReverseReverb (arena : Arena)
    (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ModalVoice"]
  assemble arena "ReverseReverb" #[floatOut] registry do
    let clk ← inputRef ⟨0⟩
    let f0 ← inputRef ⟨1⟩
    let spacing ← inputRef ⟨2⟩
    let decay ← inputRef ⟨3⟩
    let amount ← inputRef ⟨4⟩
    let tapClock := fun (coefficient? : Option Int) => do
      let scaledSpacing ← match coefficient? with
        | none => pure spacing
        | some coefficient => do
            let coefficient ← lit coefficient
            mul coefficient spacing
      let sr ← sampleRate
      let samples ← mul scaledSpacing sr
      let twoPow32 ← lit 4294967296
      let fixed ← mul samples twoPow32
      let delta ← toIntE fixed
      add clk delta
    let voice := fun (name : String) (voiceClock : Sig) =>
      inst name "ModalVoice" #[
        { port := ⟨0⟩, value := voiceClock }, { port := ⟨1⟩, value := f0 }]
    let dry ← voice "dry" clk
    let tap1Clock ← tapClock none
    let tap1 ← voice "tap1" tap1Clock
    let tap2Clock ← tapClock (some 2)
    let tap2 ← voice "tap2" tap2Clock
    let tap3Clock ← tapClock (some 3)
    let tap3 ← voice "tap3" tap3Clock
    let tap4Clock ← tapClock (some 4)
    let tap4 ← voice "tap4" tap4Clock
    let d2 ← mul decay decay
    let d3 ← mul d2 decay
    let d4 ← mul d3 decay
    let tap1Out ← nestedOut tap1 ⟨0⟩
    let tap2Out ← nestedOut tap2 ⟨0⟩
    let tap3Out ← nestedOut tap3 ⟨0⟩
    let tap4Out ← nestedOut tap4 ⟨0⟩
    let weighted1 ← mul decay tap1Out
    let weighted2 ← mul d2 tap2Out
    let weighted3 ← mul d3 tap3Out
    let weighted4 ← mul d4 tap4Out
    let taps12 ← add weighted1 weighted2
    let taps123 ← add taps12 weighted3
    let taps ← add taps123 weighted4
    let wet ← mul amount taps
    let dryOut ← nestedOut dry ⟨0⟩
    let body ← add dryOut wet
    let defaultClock ← clockSig
    let f0Input ← freqDecl "f0" 110
    let spacingDefault ← lit 45 3
    let decayDefault ← lit 72 2
    let amountDefault ← lit 7 1
    pure {
      inputs := #[clockInDecl "clk" defaultClock, f0Input,
        floatDecl "spacing" spacingDefault, floatDecl "decay" decayDefault,
        floatDecl "amount" amountDefault]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildThroughZeroFlanger (arena : Arena)
    (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved
    #["ClockPhasor", "Sin", "ReversibleComb"]
  assemble arena "ThroughZeroFlanger" #[floatOut] registry do
    let clk ← inputRef ⟨0⟩
    let f0 ← inputRef ⟨1⟩
    let depth ← inputRef ⟨2⟩
    let rate ← inputRef ⟨3⟩
    let phasor ← inst "lfoph" "ClockPhasor" #[
      { port := ⟨0⟩, value := clk }, { port := ⟨1⟩, value := rate }]
    let phase ← nestedOut phasor ⟨0⟩
    let twoPi ← twoPiE
    let angle ← mul twoPi phase
    let lfo ← inst "lfo" "Sin" #[{ port := ⟨0⟩, value := angle }]
    let lfoOut ← nestedOut lfo ⟨0⟩
    let sweptDepth ← mul depth lfoOut
    let comb ← inst "comb" "ReversibleComb" #[
      { port := ⟨0⟩, value := clk }, { port := ⟨1⟩, value := f0 },
      { port := ⟨2⟩, value := sweptDepth }]
    let body ← nestedOut comb ⟨0⟩
    let defaultClock ← clockSig
    let f0Input ← freqDecl "f0" 110
    let depthDefault ← lit 7 4
    let rateInput ← freqDecl "rate" 3 1
    pure {
      inputs := #[clockInDecl "clk" defaultClock, f0Input,
        floatDecl "depth" depthDefault, rateInput]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildFixedSin (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  assemble arena "FixedSin"
      #[{ name := "out", type? := some (.scalar .int) }] #[] do
    let phase ← inputRef ⟨0⟩
    let body ← fixedSinCycSig phase
    let zero ← lit 0
    pure {
      inputs := #[{
        name := "phase", type? := some (.scalar .int), defaultSig := some zero }]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildFixedSinOsc (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  assemble arena "FixedSinOsc"
      #[{ name := "sine", type? := some (.scalar .float) }] #[] do
    let freq ← inputRef ⟨0⟩
    let clk ← inputRef ⟨1⟩
    let offset ← inputRef ⟨2⟩
    let twoPow32 ← lit 4294967296
    let scaledFreq ← mul freq twoPow32
    let sr ← sampleRate
    let perSample ← div scaledFreq sr
    let inc ← toIntE perSample
    let shift32 ← lit 32
    let thi ← rshift clk shift32
    let mask ← lit 4294967295
    let tlo ← bitAnd clk mask
    let scaledOffset ← mul offset twoPow32
    let off ← toIntE scaledOffset
    let highProduct ← mul inc thi
    let lowProduct ← mul inc tlo
    let lowHigh ← rshift lowProduct shift32
    let accumulated ← add highProduct lowHigh
    let acc ← add accumulated off
    let wrapped ← bitAnd acc mask
    let wrappedFloat ← toFloatE wrapped
    let phaseRaw ← div wrappedFloat twoPow32
    let zero ← lit 0
    let one ← lit 1
    let phase ← clampE phaseRaw zero one
    let phaseScaled ← mul phase twoPow32
    let phaseQ ← toIntE phaseScaled
    let fixedSine ← fixedSinCycSig phaseQ
    let sineFloat ← toFloatE fixedSine
    let q30 ← lit 1073741824
    let body ← div sineFloat q30
    let freqInput ← freqDecl "freq" 440
    let defaultClock ← clockSig
    let phaseInput ← unipolarDecl "phase"
    pure {
      inputs := #[freqInput, clockInDecl "clk" defaultClock, phaseInput]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildMorphOsc (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "FixedSin"]
  assemble arena "MorphOsc" #[floatOut] registry do
    let freq ← inputRef ⟨0⟩
    let morph ← inputRef ⟨1⟩
    let clk ← inputRef ⟨2⟩
    let phase ← inputRef ⟨3⟩
    let phasor ← inst "ph" "ClockPhasor" #[
      { port := ⟨0⟩, value := clk }, { port := ⟨1⟩, value := freq },
      { port := ⟨2⟩, value := phase }]
    let phaseOut ← nestedOut phasor ⟨0⟩
    let twoPow32 ← lit 4294967296
    let fixedPhaseRaw ← mul phaseOut twoPow32
    let fixedPhase ← toIntE fixedPhaseRaw
    let sine ← inst "sin" "FixedSin" #[{ port := ⟨0⟩, value := fixedPhase }]
    let two ← lit 2
    let sawScaled ← mul two phaseOut
    let one ← lit 1
    let saw ← sub sawScaled one
    let sineOut ← nestedOut sine ⟨0⟩
    let sineFloat ← toFloatE sineOut
    let q30 ← lit 1073741824
    let sineScaled ← div sineFloat q30
    let inverseMorph ← sub one morph
    let sawMix ← mul inverseMorph saw
    let sineMix ← mul morph sineScaled
    let body ← add sawMix sineMix
    let freqInput ← freqDecl "freq" 220
    let morphInput ← unipolarDecl "morph"
    let defaultClock ← clockSig
    let phaseInput ← unipolarDecl "phase"
    pure {
      inputs := #[freqInput, morphInput, clockInDecl "clk" defaultClock, phaseInput]
      assigns := #[(.port ⟨0⟩, body)]
    }

private def warpBank3 (arena : Arena) (name voiceName : String)
    (registry : Array (String × ProgramIdx)) (freqDefault : Int) :
    Except String (Arena × ProgramIdx) :=
  assemble arena name #[floatOut] registry do
    let clk ← inputRef ⟨0⟩
    let pitch ← inputRef ⟨1⟩
    let offset ← inputRef ⟨2⟩
    let sr ← sampleRate
    let offsetSamples ← mul offset sr
    let twoPow32 ← lit 4294967296
    let offsetFixed ← mul offsetSamples twoPow32
    let delta ← toIntE offsetFixed
    let wire := fun (clock : Sig) =>
      if voiceName == "ModalVoice" then
        #[{ port := ⟨0⟩, value := clock }, { port := ⟨1⟩, value := pitch }]
      else
        #[{ port := ⟨0⟩, value := pitch }, { port := ⟨1⟩, value := clock }]
    let dry ← inst "dry" voiceName (wire clk)
    let pastClock ← sub clk delta
    let past ← inst "past" voiceName (wire pastClock)
    let aheadClock ← add clk delta
    let ahead ← inst "ahead" voiceName (wire aheadClock)
    let weighted := fun (mantissa : Int) (exponent : Nat)
        (instanceIdx : InstanceIdx) => do
      let weight ← lit mantissa exponent
      let signal ← nestedOut instanceIdx ⟨0⟩
      mul weight signal
    let dryWeighted ← weighted 5 1 dry
    let pastWeighted ← weighted 25 2 past
    let aheadWeighted ← weighted 25 2 ahead
    let firstTwo ← add dryWeighted pastWeighted
    let body ← add firstTwo aheadWeighted
    let defaultClock ← clockSig
    let pitchName := if voiceName == "ModalVoice" then "f0" else "freq"
    let pitchInput ← freqDecl pitchName freqDefault
    let offsetDefault ← lit 7 4
    let offsetName := if voiceName == "ModalVoice" then "delta" else "depth"
    pure {
      inputs := #[clockInDecl "clk" defaultClock, pitchInput,
        floatDecl offsetName offsetDefault]
      assigns := #[(.port ⟨0⟩, body)]
    }

def buildFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  warpBank3 arena "FlangeSin" "FixedSinOsc" registry 220

def buildReversibleComb (arena : Arena)
    (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ModalVoice"]
  warpBank3 arena "ReversibleComb" "ModalVoice" registry 110

def stdlibBuilders : Array (String × StdBuilder) := #[
  ("ClockPhasor", buildClockPhasor), ("FixedSin", buildFixedSin),
  ("FixedSinOsc", buildFixedSinOsc), ("MorphOsc", buildMorphOsc),
  ("Sin", buildSin), ("Tanh", buildTanh), ("VCA", buildVCA),
  ("FlangeSin", buildFlanger), ("ModalVoice", buildModalVoice),
  ("PluckedMorphOsc", buildPluckedMorphOsc),
  ("ReverseReverb", buildReverseReverb),
  ("ReversibleComb", buildReversibleComb), ("ScrubClock", buildScrubClock),
  ("ThroughZeroFlanger", buildThroughZeroFlanger),
  ("SoftClip", buildSoftClip)]

def buildStdlibChain : Except String (Arena × Array (String × ProgramIdx)) := do
  let mut arena : Arena := {}
  let mut chain : Array (String × ProgramIdx) := #[]
  for (name, build) in stdlibBuilders do
    let (nextArena, index) ← build arena chain
    arena := nextArena
    chain := chain.push (name, index)
  pure (arena, chain)

end Tropical.EmitArrow.ArenaNative
