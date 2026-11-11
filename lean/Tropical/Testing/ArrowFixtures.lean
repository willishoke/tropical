import Tropical.EmitArrow
import Tropical.Ir.Strata
import Tropical.Ir.CompileResolved
import Tropical.Testing.PlanWire
import Tropical.Testing.ArrowOracles
import Tropical.Ir.ConstFold

/-!
# Arena-native EmitArrow test fixtures

This module is the test-only boundary for constructing and inspecting programs
through the ID-native authoring API.  It deliberately accepts only `BuildM`
actions and stable expression IDs; recursive authoring expressions never cross
this surface.
-/

namespace Tropical.Testing.ArrowFixtures

open Tropical.Ir
open Tropical.EmitArrow

/-- Run an arena-native construction against an existing arena without
publishing a program.  The returned builder owns the frozen expression arena
and ordered declaration effects produced by the action. -/
def runBuild {α : Type} (arena : Arena) (build : BuildM α) :
    Except String (Builder × α) := do
  let (result, builder) ← build.run { exprs := arena.exprs }
  pure (builder, result)

/-- Execute a construction and retain just its frozen expression arena and
result.  This is the classifier-inspection boundary used by exact tests. -/
def freezeBuild {α : Type} (arena : Arena) (build : BuildM α) :
    Except String (ExprArena × α) := do
  let (builder, result) ← runBuild arena build
  pure (builder.exprs, result)

/-- Assemble a program atomically and return its stable index.  Failure returns
no arena, so neither a partial program nor partially-authored expressions can
escape. -/
def assembleProgram (arena : Arena) (name : String)
    (outputs : Array OutputDecl) (registry : Array (String × ProgramIdx))
    (build : BuildM ProgramBody) (extraDecls : Array BodyDecl := #[]) :
    Except String (Arena × ProgramIdx) :=
  assemble arena name outputs registry build extraDecls

/-- Resolve and compile a returned program through the ordinary production
lowering boundary. -/
def compileProgram (arena : Arena) (program : ProgramIdx)
    (options : Tropical.Ir.Strata.Options := { inlineNested := true }) :
    Except String (ExprArena × Tropical.Plan.PerInstancePlan) := do
  let (exprs, core) ← (Tropical.Ir.Strata.runResolved options arena program).mapError
    (fun failure => failure.message)
  let plan ← Tropical.Ir.CompileResolved.compileResolved core exprs
  pure (exprs, plan)

/-- Compile and render a plan to its stable JSON wire representation without
accepting a recursive expression at any point. -/
def renderProgramWire (arena : Arena) (program : ProgramIdx)
    (options : Tropical.Ir.Strata.Options := { inlineNested := true }) :
    Except String (ExprArena × String) := do
  let (exprs, plan) ← compileProgram arena program options
  pure (exprs, (← plan.toWire).compress)

/-- A scalar smoke fixture used to qualify the common boundary. -/
def scalarFixture : Except String (Arena × ProgramIdx) :=
  assembleProgram {} "ArenaScalarFixture"
      #[{ name := "out", type? := some (.scalar .float) }] #[] do
    let two ← lit 2
    let three ← lit 3
    let output ← add two three
    pure { assigns := #[(.port ⟨0⟩, output)] }

/-- A multi-declaration smoke fixture.  Declaration order is observable and is
therefore checked at construction time. -/
def multiDeclarationFixture (arena : Arena) (leaf : ProgramIdx) :
    Except String (Arena × ProgramIdx) :=
  assembleProgram arena "ArenaMultiDeclarationFixture"
      #[{ name := "out", type? := some (.scalar .float) }]
      #[("ArenaScalarFixture", leaf)] do
    let zero ← lit 0
    let first ← inst "first" "ArenaScalarFixture"
    let firstOut ← nestedOut first ⟨0⟩
    let second ← inst "second" "ArenaScalarFixture"
    let secondOut ← nestedOut second ⟨0⟩
    let output ← add firstOut secondOut
    let builder ← get
    unless builder.decls.map (·.name) == #["first", "second"] do
      throw "ArrowFixtures: declaration order changed"
    pure {
      inputs := #[{ name := "unused", type? := some (.scalar .float), defaultSig := some zero }]
      assigns := #[(.port ⟨0⟩, output)]
    }

/-- Compile-time/executable refusal evidence: a failed build cannot mutate the
source arena because the updated builder state is returned only by `.ok`. -/
def failedBuildLeavesSourceUnchanged : Bool :=
  let source : Arena := {}
  let attempted : Except String (Arena × ProgramIdx) :=
    assembleProgram source "ArenaRefusalFixture"
        #[{ name := "out", type? := some (.scalar .float) }] #[] do
      let _ ← lit 1
      throw "intentional fixture refusal"
  match attempted with
  | .error message =>
      message == "intentional fixture refusal" && source.programs.isEmpty &&
        source.exprs.nodes.isEmpty
  | .ok _ => false

/-- The two substrate smoke fixtures both compile and render through the shared
helpers, and the refusal fixture demonstrates atomic publication. -/
def substratePasses : Except String Bool := do
  let (arena, leaf) ← scalarFixture
  let (_, scalarWire) ← renderProgramWire arena leaf
  let (arena, root) ← multiDeclarationFixture arena leaf
  let (_, multiWire) ← renderProgramWire arena root
  pure (failedBuildLeavesSourceUnchanged && !scalarWire.isEmpty && !multiWire.isEmpty)

end Tropical.Testing.ArrowFixtures

/-! Native DUT carriers used by the arrow, slide, and stress suites.  These
live beside the shared runner so every allocating callback is visibly monadic. -/

namespace Tropical.EmitArrow

open Tropical.Ir

private def fixtureOutput (type : ScalarKind := .float) : Array OutputDecl :=
  #[{ name := "out", type? := some (.scalar type) }]

def clkIn : BuildM Clock := inputRef ⟨0⟩
def pitchIn : BuildM Sig := inputRef ⟨1⟩
def offsetIn : BuildM Sig := inputRef ⟨2⟩

def deltaSamples : BuildM Sig := do
  let offset ← offsetIn
  let sr ← sampleRate
  let seconds ← mul offset sr
  let scale ← clockOneLit
  toIntE (← mul seconds scale)

def fixedSinOscVoice : Voice := {
  programName := "FixedSinOsc"
  wire := fun clock => do
    let pitch ← pitchIn
    pure #[{ port := ⟨0⟩, value := pitch }, { port := ⟨1⟩, value := clock }]
}

def modalVoice : Voice := {
  programName := "ModalVoice"
  wire := fun clock => do
    let pitch ← pitchIn
    pure #[{ port := ⟨0⟩, value := clock }, { port := ⟨1⟩, value := pitch }]
}

structure Tap where
  name : String
  warp : Clock → BuildM Clock
  weight : BuildM Sig

def warpBank (voice : Voice) (taps : Array Tap) (clock : Clock) : BuildM Sig := do
  let summands ← taps.mapM fun tap => do
    let warped ← tap.warp clock
    let signal ← Builder.osc voice tap.name warped
    let weight ← tap.weight
    mul weight signal
  sumLeft summands

def flangerTaps : Array Tap := #[
  { name := "dry", warp := pure, weight := lit 5 1 },
  { name := "past", warp := fun clock => do sub clock (← deltaSamples), weight := lit 25 2 },
  { name := "ahead", warp := fun clock => do add clock (← deltaSamples), weight := lit 25 2 }
]

def clkInputDecl : BuildM AInputDecl := do
  let sample ← sampleIndex
  let shift ← clockFracLit
  let defaultSig ← lshift sample shift
  pure { name := "clk", type? := some (.scalar .int), defaultSig := some defaultSig }

def pitchInputDecl (name : String) (hz : Int) : BuildM AInputDecl := do
  let hz ← lit hz
  let zero ← lit 0
  let positive ← gt hz zero
  let defaultSig ← select positive hz zero
  pure { name, type? := some (.scalar .float), defaultSig := some defaultSig }

def offsetInputDecl (name : String) : BuildM AInputDecl := do
  let defaultSig ← lit 7 4
  pure { name, type? := some (.scalar .float), defaultSig := some defaultSig }

structure WarpBankProgram where
  name : String
  voice : Voice
  inputs : BuildM (Array AInputDecl)
  taps : Array Tap

def buildWarpBank (spec : WarpBankProgram) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #[spec.voice.programName]
  assemble arena spec.name (fixtureOutput .float) registry do
    let inputs ← spec.inputs
    let clock ← clkIn
    let output ← warpBank spec.voice spec.taps clock
    pure { inputs, assigns := #[(.port ⟨0⟩, output)] }

def flangeSinSpec : WarpBankProgram := {
  name := "FlangeSin", voice := fixedSinOscVoice, taps := flangerTaps
  inputs := do
    pure #[← clkInputDecl, ← pitchInputDecl "freq" 220, ← offsetInputDecl "depth"]
}

def reversibleCombSpec : WarpBankProgram := {
  name := "ReversibleComb", voice := modalVoice, taps := flangerTaps
  inputs := do
    pure #[← clkInputDecl, ← pitchInputDecl "f0" 110, ← offsetInputDecl "delta"]
}

def clockPhasorPorts : Array InputIdx := #[⟨0⟩, ⟨1⟩, ⟨2⟩]
def phasorMor : Mor := instMor "ph" "ClockPhasor" clockPhasorPorts 1

def sawMor : Mor := arrMor fun values => do
  let two ← lit 2
  let scaled ← mul two values[0]!
  let one ← lit 1
  pure #[← sub scaled one]

def sinMor : Mor :=
  seq (arrMor fun values => do
    let scale ← lit 4294967296
    pure #[← toIntE (← mul values[0]! scale)])
    (seq (instMor "sin" "FixedSin" #[⟨0⟩] 1)
      (arrMor fun values => do
        let value ← toFloatE values[0]!
        let scale ← lit 1073741824
        pure #[← div value scale]))

def crossfadeMor : Mor := arrMor fun values => do
  let one ← lit 1
  let dryWeight ← sub one values[2]!
  let dry ← mul dryWeight values[0]!
  let wet ← mul values[2]! values[1]!
  pure #[← add dry wet]

def morphOscMor : Mor :=
  seq (arrMor fun values => pure #[values[2]!, values[0]!, values[3]!, values[1]!])
    (seq (first 3 phasorMor) (seq (first 1 (fan sawMor sinMor)) crossfadeMor))

def buildMorphOscLit (name : String) (freqHz : Int) (morph : BuildM Sig)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "FixedSin"]
  assemble arena name (fixtureOutput .float) registry do
    let freq ← lit freqHz
    let morph ← morph
    let clock ← clockLit
    let zero ← lit 0
    let outputs ← morphOscMor #[freq, morph, clock, zero]
    pure { assigns := #[(.port ⟨0⟩, outputs[0]!)] }

def deltaLit (mantissa : Int) (exponent : Nat) : BuildM Sig := do
  let seconds ← lit mantissa exponent
  let sr ← sampleRate
  let samples ← mul seconds sr
  let scale ← clockOneLit
  toIntE (← mul samples scale)

def delta1 : BuildM Sig := deltaLit 7 4
def delta2 : BuildM Sig := deltaLit 11 4

def litPitchVoice (hz : Int) : Voice := {
  programName := "FixedSinOsc"
  wire := fun clock => do
    let pitch ← lit hz
    pure #[{ port := ⟨0⟩, value := pitch }, { port := ⟨1⟩, value := clock }]
}

def litPitchSinOscVoice : Voice := litPitchVoice 220
def litPitch12kVoice : Voice := litPitchVoice 12000

def buildVoiceProgram (name : String) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) (build : BuildM Sig) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  assemble arena name (fixtureOutput .float) registry do
    let output ← build
    pure { assigns := #[(.port ⟨0⟩, output)] }

def buildClockCarrier (name : String) (clock : BuildM Clock) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) :=
  buildVoiceProgram name arena resolved do
    Builder.osc litPitchSinOscVoice "voice" (← clock)

def buildTapCarrier (name : String) (voice : Voice) (taps : Array Tap)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram name arena resolved do
    warpBank voice taps (← clockLit)

def buildFmCarrier (name : String) (carHz modHz depthSamples : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram name arena resolved do
    let clock ← clockLit
    let modSignal ← Builder.osc (litPitchVoice modHz) "mod" clock
    let depth ← lit depthSamples
    let scaled ← mul depth modSignal
    let qScale ← clockOneLit
    let shift ← toIntE (← mul scaled qScale)
    let warped ← sub clock shift
    Builder.osc (litPitchVoice carHz) "car" warped

def buildPmPmCarrier (name : String) (carHz modHz mod2Hz depth1 depth2 : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram name arena resolved do
    let clock ← clockLit
    let mod2 ← Builder.osc (litPitchVoice mod2Hz) "mod2" clock
    let depth2 ← lit depth2
    let qScale ← clockOneLit
    let modShift ← toIntE (← mul (← mul depth2 mod2) qScale)
    let modClock ← sub clock modShift
    let modSignal ← Builder.osc (litPitchVoice modHz) "mod" modClock
    let depth1 ← lit depth1
    let carShift ← toIntE (← mul (← mul depth1 modSignal) qScale)
    let carClock ← sub clock carShift
    Builder.osc (litPitchVoice carHz) "car" carClock

def invLawLhsClock : BuildM Clock := do
  let clock ← clockLit
  let delta ← delta1
  sub (← add clock delta) delta

def invLawRhsClock : BuildM Clock := clockLit

def addLawLhsClock : BuildM Clock := do
  let clock ← clockLit
  sub (← sub clock (← delta1)) (← delta2)

def addLawRhsClock : BuildM Clock := do
  let clock ← clockLit
  sub clock (← add (← delta1) (← delta2))

def flangerSum (dry past ahead : Sig) : BuildM Sig := do
  let half ← lit 5 1
  let quarter ← lit 25 2
  let dry ← mul half dry
  let past ← mul quarter past
  let ahead ← mul quarter ahead
  add (← add dry past) ahead

private def buildFixtureFlanger (voice : Voice) (baseClock delta : Sig)
    (tag : String) : BuildM Sig := do
  let dry ← Builder.osc voice ("dry" ++ tag) baseClock
  let pastClock ← sub baseClock delta
  let past ← Builder.osc voice ("past" ++ tag) pastClock
  let aheadClock ← add baseClock delta
  let ahead ← Builder.osc voice ("ahead" ++ tag) aheadClock
  flangerSum dry past ahead

private def buildFixtureFlangerSharedDry (voice : Voice) (baseClock delta dry : Sig)
    (tag : String) : BuildM Sig := do
  let past ← Builder.osc voice ("past" ++ tag) (← sub baseClock delta)
  let ahead ← Builder.osc voice ("ahead" ++ tag) (← add baseClock delta)
  flangerSum dry past ahead

def buildSharedDiagonal (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "DiagonalShared" arena resolved do
    let clock ← clockLit
    let dry ← Builder.osc litPitchSinOscVoice "dry" clock
    let first ← buildFixtureFlangerSharedDry litPitchSinOscVoice clock (← delta1) dry "1"
    let second ← buildFixtureFlangerSharedDry litPitchSinOscVoice clock (← delta2) dry "2"
    add first second

def buildIndependentDiagonal (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "DiagonalIndependent" arena resolved do
    let clock ← clockLit
    let first ← buildFixtureFlanger litPitchSinOscVoice clock (← delta1) "1"
    let second ← buildFixtureFlanger litPitchSinOscVoice clock (← delta2) "2"
    add first second

def revInvolutionLhsClock : BuildM Clock := do neg (← neg (← clockLit))
def revInvolutionRhsClock : BuildM Clock := clockLit

def revSwapLhsClock : BuildM Clock := do
  neg (← sub (← clockLit) (← delta1))

def revSwapRhsClock : BuildM Clock := do
  add (← neg (← clockLit)) (← delta1)

def buildReverseThenFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "ReverseThenFlanger" arena resolved do
    let clock ← clockLit
    let delta ← delta1
    let dry ← Builder.osc litPitchSinOscVoice "dry" (← neg clock)
    let past ← Builder.osc litPitchSinOscVoice "past" (← neg (← sub clock delta))
    let ahead ← Builder.osc litPitchSinOscVoice "ahead" (← neg (← add clock delta))
    flangerSum dry past ahead

def buildFlangerThenReverse (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "FlangerThenReverse" arena resolved do
    let reversed ← neg (← clockLit)
    let delta ← delta1
    let dry ← Builder.osc litPitchSinOscVoice "dry" reversed
    let past ← Builder.osc litPitchSinOscVoice "past" (← sub reversed delta)
    let ahead ← Builder.osc litPitchSinOscVoice "ahead" (← add reversed delta)
    flangerSum dry past ahead

def fixedPhase (clock : Clock) : BuildM Sig := do
  let shift ← clockFracLit
  let high ← rshift clock shift
  let low ← bitAnd clock (← clockFracMaskLit)
  let increment ← lit 21426140
  let highProduct ← mul increment high
  let lowProduct ← mul increment low
  let lowShifted ← rshift lowProduct shift
  let mask ← lit 4294967295
  bitAnd (← add highProduct lowShifted) mask

def fixedFlangerSum (dry past ahead : Sig) : BuildM Sig := do
  let one ← lit 1
  let two ← lit 2
  let dry ← rshift dry one
  let past ← rshift past two
  let ahead ← rshift ahead two
  add (← add dry past) ahead

def fixedOut (mix : Sig) : BuildM Sig := do
  let value ← toFloatE mix
  div value (← lit 4294967296)

def buildExprCarrier (name : String) (out : BuildM Sig) (arena : Arena) :
    Except String (Arena × ProgramIdx) :=
  assemble arena name (fixtureOutput .float) #[] do
    pure { assigns := #[(.port ⟨0⟩, ← out)] }

def buildFixedSourceCarrier (name : String) (clock : BuildM Clock) (arena : Arena) :
    Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do fixedOut (← fixedPhase (← clock))) arena

def buildReverseThenFixedFlanger (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier "ReverseThenFixedFlanger" (do
    let clock ← clockLit
    let delta ← delta1
    let dry ← fixedPhase (← neg clock)
    let past ← fixedPhase (← neg (← sub clock delta))
    let ahead ← fixedPhase (← neg (← add clock delta))
    fixedOut (← fixedFlangerSum dry past ahead)) arena

def buildFixedFlangerThenReverse (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier "FixedFlangerThenReverse" (do
    let reversed ← neg (← clockLit)
    let delta ← delta1
    let dry ← fixedPhase reversed
    let past ← fixedPhase (← sub reversed delta)
    let ahead ← fixedPhase (← add reversed delta)
    fixedOut (← fixedFlangerSum dry past ahead)) arena

def slideBack : Clock → BuildM Clock := fun clock => do sub clock (← deltaLit 7 4)
def slideFwd : Clock → BuildM Clock := fun clock => do add clock (← deltaLit 7 4)

def flangeEffect (signal : ArrowTerm) : BuildM ArrowTerm := do
  let delta ← deltaSamples
  flangeEffectWith (fun clock => sub clock delta) (fun clock => add clock delta) signal

def buildFlangerViaSlide (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  assemble arena "FlangeSin" (fixtureOutput .float) registry do
    let term ← flangeEffect (.gen fixedSinOscVoice "osc" (← clkIn))
    let output ← emitTerm (normalize term)
    let inputs : Array AInputDecl := #[
      ← clkInputDecl, ← pitchInputDecl "freq" 220, ← offsetInputDecl "depth"]
    pure { inputs, assigns := #[(.port ⟨0⟩, output)] }

def litOscGen : BuildM ArrowTerm := do
  pure (.gen litPitchSinOscVoice "osc" (← clockLit))

def buildSlideShaperDownstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "SlideShaperDown" arena resolved do
    let shaped : ArrowTerm := .arrUn (fun signal => mul signal signal) (← litOscGen)
    let term ← flangeEffectWith slideBack slideFwd shaped
    emitTerm (normalize term)

def buildSlideShaperUpstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "SlideShaperUp" arena resolved do
    let clock ← clockLit
    let shaped := fun (warp : Clock → BuildM Clock) => do
      pure (.arrUn (fun signal => mul signal signal)
        (.gen litPitchSinOscVoice "osc" (← warp clock)))
    let half ← lit 5 1
    let quarter ← lit 25 2
    let term : ArrowTerm := .sum #[
      .scale half (← shaped pure),
      .scale quarter (← shaped slideBack),
      .scale quarter (← shaped slideFwd)]
    emitTerm term

def buildSlideSingleFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "SlideSingleFlange" arena resolved do
    let term ← flangeEffectWith slideBack slideFwd (← litOscGen)
    emitTerm (normalize term)

def buildSlideDoubleFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "SlideDoubleFlange" arena resolved do
    let inner ← flangeEffectWith slideBack slideFwd (← litOscGen)
    let outer ← flangeEffectWith slideBack slideFwd inner
    emitTerm (normalize outer)

def buildFlangeFromGraph (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  assemble arena "FlangeSin" (fixtureOutput .float) registry do
    let back : Clock → BuildM Clock := fun clock => do
      sub clock (← deltaSamples)
    let forward : Clock → BuildM Clock := fun clock => do
      add clock (← deltaSamples)
    let graph : PatchGraph := {
      nodes := #[
        { id := "osc", node := .source fixedSinOscVoice (← clkIn) },
        { id := "fl", node := .flange "osc" back forward }]
      output := "fl" }
    let output ← emitTerm (normalize (← lowerGraph graph))
    let inputs : Array AInputDecl := #[
      ← clkInputDecl, ← pitchInputDecl "freq" 220, ← offsetInputDecl "depth"]
    pure { inputs, assigns := #[(.port ⟨0⟩, output)] }

def buildDoubleFlangeFromGraph (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "GraphDoubleFlange" arena resolved do
    let graph : PatchGraph := {
      nodes := #[
        { id := "osc", node := .source litPitchSinOscVoice (← clockLit) },
        { id := "f1", node := .flange "osc" slideBack slideFwd },
        { id := "f2", node := .flange "f1" slideBack slideFwd }]
      output := "f2" }
    emitTerm (normalize (← lowerGraph graph))

def buildFanOutFromGraph (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "GraphFanOut" arena resolved do
    let back := fun clock => do sub clock (← deltaLit 11 4)
    let forward := fun clock => do add clock (← deltaLit 11 4)
    let graph : PatchGraph := {
      nodes := #[
        { id := "osc", node := .source litPitchSinOscVoice (← clockLit) },
        { id := "fa", node := .flange "osc" slideBack slideFwd },
        { id := "fb", node := .flange "osc" back forward },
        { id := "mix", node := .mix #["fa", "fb"] }]
      output := "mix" }
    emitTerm (normalize (← lowerGraph graph))

def buildFmFromGraph (carHz modHz depth : Int) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "GraphFm" arena resolved do
    let clock ← clockLit
    let graph : PatchGraph := {
      nodes := #[
        { id := "mod", node := .source (litPitchVoice modHz) clock },
        { id := "car", node := .fm "mod" (litPitchVoice carHz) clock (← lit depth) }]
      output := "car" }
    emitTerm (normalize (← lowerGraph graph))

def buildPmPmFromGraph (carHz modHz mod2Hz depth1 depth2 : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "GraphPmPm" arena resolved do
    let clock ← clockLit
    let graph : PatchGraph := {
      nodes := #[
        { id := "mod2", node := .source (litPitchVoice mod2Hz) clock },
        { id := "mod", node := .fm "mod2" (litPitchVoice modHz) clock (← lit depth2) },
        { id := "car", node := .fm "mod" (litPitchVoice carHz) clock (← lit depth1) }]
      output := "car" }
    emitTerm (normalize (← lowerGraph graph))

def buildSlideProdDownstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "SlideProdDown" arena resolved do
    let clock ← clockLit
    let product : ArrowTerm := .prod (.gen (litPitchVoice 220) "a" clock)
      (.gen (litPitchVoice 330) "b" clock)
    emitTerm (normalize (.warp slideBack product))

def buildSlideProdUpstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildVoiceProgram "SlideProdUp" arena resolved do
    let clock ← clockLit
    let x : ArrowTerm := .gen (litPitchVoice 220) "a" clock
    let y : ArrowTerm := .gen (litPitchVoice 330) "b" clock
    emitTerm (normalize (.prod (.warp slideBack x) (.warp slideBack y)))

def fixedSinOscTerm (freq offset clock : Sig) : ArrowTerm :=
  .arrUn (fun clock => do
    let phase ← phasorPhaseSig freq offset clock
    let qScale ← lit 4294967296
    let phaseQ ← toIntE (← mul phase qScale)
    let sine ← fixedSinCycSig phaseQ
    let value ← toFloatE sine
    div value (← lit 1073741824)) (.clk clock)

def buildBootstrapSinOsc (name : String) (arena : Arena) :
    Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let term := fixedSinOscTerm (← lit 220) (← lit 0) (← clockLit)
    emitTerm (normalize term)) arena

private def sampleNumber : BuildM Sig := do
  let clock ← clockLit
  let shifted ← rshift clock (← clockFracLit)
  toFloatE shifted

def buildExpProbe (name : String) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let x ← sub (← mul (← sampleNumber) (← lit 9765625 9)) (← lit 10)
    expSig x) arena

def buildLogProbe (name : String) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let x ← add (← lit 2 2) (← mul (← sampleNumber) (← lit 9765625 8))
    logSig x) arena

def buildAtan2Probe (name : String) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let theta ← sub (← mul (← sampleNumber) (← lit 302734375 11)) (← lit 31 1)
    atan2E (← sinSig theta) (← cosSig theta)) arena

/-- The rare-op expression coverage fixture used by wasm≡JIT.  Allocation is
    written in recursive postorder so its frozen wire remains identical to the
    retired tree fixture while sharing the single `seed` ID explicitly. -/
def opZooSig : BuildM Sig := do
  let seed ← bitAnd (← toIntE (← sampleIndex)) (← lit 255)
  let shifted ← lshift seed (← lit 1)
  let ored ← bitOr seed (← lit 1)
  let right ← rshift ored (← lit 1)
  let inverted ← unary .bitNot seed
  let masked ← bitAnd inverted right
  let noise ← toFloatE (← binary .bitXor shifted masked)
  let quotient ← binary .floorDiv seed (← lit 3)
  let remainder ← binary .mod quotient (← lit 7)
  let modTerm ← toFloatE remainder
  let afterStart ← gt (← toFloatE (← sampleIndex)) (← lit 0)
  let rateOk ← binary .lte (← sampleRate) (← lit 1000000)
  let sampleBool ← unary .toBool (← toFloatE (← sampleIndex))
  let sampleIsZero ← binary .eq sampleBool (← lit 0)
  let sampleNonzero ← unary .not sampleIsZero
  let rateOrSample ← binary .or rateOk sampleNonzero
  let condition ← binary .and afterStart rateOrSample
  let ceilValue ← unary .ceil (← lit 22 1)
  let floorValue ← unary .floor (← lit 37 1)
  let difference ← sub ceilValue floorValue
  let root ← unary .sqrt (← absE (← neg difference))
  let alternate ← toFloatE (← toIntE (← lit 37 1))
  let branch ← selectE condition root alternate
  let exponent ← floatExponentE (← lit 1 1)
  let scaled ← ldexpE exponent (← lit 2)
  let rounded ← roundE scaled
  let less ← binary .lt (← lit 1) (← lit 2)
  let atLeastZero ← binary .gte less (← lit 0)
  let tail ← mul rounded (← div (← toFloatE atLeastZero) (← lit 1000))
  let value ← add noise (← add modTerm (← add branch tail))
  clampE value (← lit (-1)) (← lit 1)

def buildOpZoo (arena : Arena) : Except String (Arena × ProgramIdx) :=
  assemble arena "OpZoo" #[{ name := "out" }] #[] do
    pure { assigns := #[(.port ⟨0⟩, ← opZooSig)] }

/-- The int32 coverage word: every int32 op at runtime, seeded from `x` so
    nothing folds — the wrapping ring ops (an LCG step, a shift into the sign
    bit, `-(x ∣ INT32_MIN)`, which is `INT32_MIN` itself at `x = 0`), the
    arithmetic right shift of negative words, division and remainder (by a
    nonzero divisor, and by a RUNTIME divisor that is 0 for odd `x` and -1 for
    even `x` over the numerator `x ∣ INT32_MIN`, so `INT32_MIN / -1` happens
    at `x = 0`), shifts by a runtime amount in `[0, 63]` (reduced modulo 32),
    comparison and select, clamp, abs, bitNot, and the casts both ways (`int`
    product truncated to its low word, `float` truncated toward zero, and
    `x · 2¹⁷` saturating for `|x| ≥ 2¹⁴`). `tropicaltest`'s `int32-model`
    gate checks the JIT against the reference semantics
    (`Semantics.Scalar.algebra`) on this word. -/
def int32ZooWord (x : Sig) : BuildM Sig := do
  let lcg ← add (← mul x (← lit 1103515245)) (← lit 12345)
  let mixed ← binary .bitXor lcg (← rshift lcg (← lit 16))
  let high ← lshift x (← lit 27)
  let minned ← neg (← bitOr x (← litI32 (-2147483648)))
  let quot ← div lcg (← bitOr (← bitAnd x (← lit 7)) (← lit 1))
  let rem ← binary .mod lcg (← lit 1000)
  let sel ← selectE (← binary .lt lcg (← lit 0)) (← litI32 3) (← litI32 5)
  let clamped ← clampE mixed (← lit (-1000000)) (← lit 1000000)
  let ab ← absE (← sub x (← lit 100))
  let inv ← unary .bitNot x
  let viaInt ← toInt32E (← mul (← toIntE lcg) (← litI 3000000000))
  let viaFloat ← toInt32E (← mul (← toFloatE x) (← lit 5 1))
  -- the total edges, at runtime
  let edgeNum ← bitOr x (← litI32 (-2147483648))
  let edgeDiv ← sub (← bitAnd x (← lit 1)) (← lit 1)
  let edgeQ ← div edgeNum edgeDiv
  let edgeR ← binary .mod edgeNum edgeDiv
  let edgeShl ← lshift lcg (← bitAnd x (← lit 63))
  let edgeShr ← rshift edgeNum (← bitAnd (← rshift x (← lit 3)) (← lit 63))
  let edgeSat ← toInt32E (← mul (← toFloatE x) (← lit 131072))
  let a ← binary .bitXor mixed high
  let b ← add a minned
  let c ← sub b quot
  let d ← binary .bitXor c rem
  let e ← add d (← mul clamped sel)
  let f ← binary .bitXor e ab
  let g ← add f inv
  let h ← binary .bitXor g viaInt
  let i ← add h viaFloat
  let j ← binary .bitXor i edgeQ
  let k ← add j edgeR
  let l ← binary .bitXor k edgeShl
  let m ← add l edgeShr
  binary .bitXor m edgeSat

/-- `Int32Zoo`: input `x : int32` (default: the sample index's low word),
    output `out = int32ZooWord x / 2³¹`, a float in `[-1, 1)`. Registered as a
    test fixture so wasm≡JIT can drive it by name, unwired or with `x` wired
    from an int32 output (the slot crossing). -/
def buildInt32Zoo (arena : Arena) : Except String (Arena × ProgramIdx) :=
  assemble arena "Int32Zoo" #[{ name := "out", type? := some (.scalar .float) }] #[] do
    let word ← int32ZooWord (← inputRef ⟨0⟩)
    let out ← div (← toFloatE word) (← lit 2147483648)
    let x0 ← toInt32E (← sampleIndex)
    pure {
      inputs := #[{ name := "x", type? := some (.scalar .int32), defaultSig := some x0 }]
      assigns := #[(.port ⟨0⟩, out)] }

/-! ### Integer edge values — the programs of the `int-edge` gate

Each program computes every integer op of one sort on boundary operands, at
runtime, and reads the result out exactly. Operand `j` of a pair comes from
Pack tables of 16-bit pieces indexed by the frame (`(p₃ ≪ 48) ∣ … ∣ p₀`, so
`INT64_MIN` and friends arrive exactly on every target, f32 Metal included),
and a result leaves as four 16-bit pieces of its sign-extended 64-bit word,
one per frame, which every target's output carries exactly. Frame
`k = o·F + 4·p + q` is piece `q` of op `o` on pair `p` (`F = 4 · pairs`). -/

/-- Boundary values of an integer sort. -/
def intEdgeValues : ScalarKind → Array Int
  | .int32 => #[-2147483648, 2147483647, -1, 0, 1, 2, -2, -2147483647, 65536, 46341]
  | _ => #[-9223372036854775808, 9223372036854775807, -1, 0, 1, 2, -2,
      -9223372036854775807, 2147483647, 4294967296]

/-- Shift amounts around both widths, negative and past the word included. -/
def intEdgeShifts : Array Int := #[-1, 0, 1, 31, 32, 33, 63, 64, 65]

/-- Knuth's MMIX LCG on ℤ/2⁶⁴: the seeded random sample. -/
def edgeLcg (s : Nat) : Nat :=
  (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616

/-- `count` words of a sort from the seed `seed` (signed, full range). -/
def edgeRandomWords (k : ScalarKind) (seed count : Nat) : Array Int := Id.run do
  let mut s := seed
  let mut out := #[]
  for _ in [0:count] do
    s := edgeLcg s
    let w : Int := if k == .int32 then Int.ofNat (s / 4294967296) - 2147483648
      else Int.ofNat s - 9223372036854775808
    out := out.push w
  return out

/-- The operand pairs of the arithmetic, bitwise and comparison ops: every
    boundary value against every other (runtime zero and -1 divisors under
    `MIN` included), then six seeded random pairs. -/
def intEdgePairs (k : ScalarKind) : Array (Int × Int) :=
  let v := intEdgeValues k
  let r := edgeRandomWords k 0x7e57 12
  v.flatMap (fun a => v.map (a, ·)) ++ (List.range 6).toArray.map fun i => (r[2*i]!, r[2*i+1]!)

/-- The operand pairs of the shifts: every boundary value by every edge
    amount, then sixteen random words by random amounts in `[-70, 70]`. -/
def intEdgeShiftPairs (k : ScalarKind) : Array (Int × Int) :=
  let v := intEdgeValues k
  let r := edgeRandomWords k 0x5ee5 16
  let amounts := (edgeRandomWords .int 0xa11 16).map fun w => w % 141 - 70
  v.flatMap (fun a => intEdgeShifts.map (a, ·)) ++
    (List.range 16).toArray.map fun i => (r[i]!, amounts[i]!)

/-- The ops of the edge zoo for sort `k`: name, result kind, and the builder
    over the pair `(a, b)` and the shift pair `(a₂, s)`. -/
def intEdgeOps (k : ScalarKind) :
    Array (String × ScalarKind × (Sig → Sig → Sig → Sig → BuildM Sig)) := #[
  ("add", k, fun a b _ _ => add a b),
  ("sub", k, fun a b _ _ => sub a b),
  ("mul", k, fun a b _ _ => mul a b),
  ("div", k, fun a b _ _ => div a b),
  ("mod", k, fun a b _ _ => binary .mod a b),
  ("floorDiv", k, fun a b _ _ => binary .floorDiv a b),
  ("bitAnd", k, fun a b _ _ => bitAnd a b),
  ("bitOr", k, fun a b _ _ => bitOr a b),
  ("bitXor", k, fun a b _ _ => binary .bitXor a b),
  ("lshift", k, fun _ _ a s => lshift a s),
  ("rshift", k, fun _ _ a s => rshift a s),
  ("less", .bool, fun a b _ _ => binary .lt a b),
  ("lessEq", .bool, fun a b _ _ => binary .lte a b),
  ("greater", .bool, fun a b _ _ => binary .gt a b),
  ("greaterEq", .bool, fun a b _ _ => binary .gte a b),
  ("equal", .bool, fun a b _ _ => binary .eq a b),
  ("notEqual", .bool, fun a b _ _ => binary .neq a b),
  ("neg", k, fun a _ _ _ => neg a),
  ("abs", k, fun a _ _ _ => absE a),
  ("bitNot", k, fun a _ _ _ => unary .bitNot a),
  ("clamp", k, fun a b _ _ => do clampE a b (← neg b)),
  ("select", k, fun a b _ _ => do selectE a b (← neg b)),
  ("toBool", .bool, fun a _ _ _ => unary .toBool a),
  ("not", .bool, fun a _ _ _ => unary .not a),
  ("toFloat", .float, fun a _ _ _ => toFloatE a),
  (if k == .int32 then "toInt" else "toInt32",
    (if k == .int32 then .int else .int32),
    fun a _ _ _ => if k == .int32 then toIntE a else toInt32E a)]

/-- The 16-bit piece `i` of a word's two's-complement image. -/
def edgePiece (v : Int) (i : Nat) : Nat :=
  ((v % 18446744073709551616).toNat / (65536 ^ i)) % 65536

/-- An operand of sort `k` read at pair index `p` from piece tables. -/
def edgeOperand (k : ScalarKind) (vals : Array Int) (p : Sig) : BuildM Sig := do
  let n := if k == .int32 then 2 else 4
  let mut acc ← toIntE (← index (← arr (← vals.mapM fun v => lit (edgePiece v 0))) p)
  for i in [1:n] do
    let piece ← toIntE (← index (← arr (← vals.mapM fun v => lit (edgePiece v i))) p)
    acc ← bitOr acc (← lshift piece (← lit (16 * i)))
  if k == .int32 then toInt32E acc else pure acc

/-- Piece `q` of a result of kind `kind`: a 16-bit piece of its sign-extended
    64-bit word, or (a float result) the value itself at `q = 0`. -/
def edgeReadout (kind : ScalarKind) (r q : Sig) : BuildM Sig := do
  if kind == .float then
    return ← selectE (← binary .eq q (← lit 0)) r (← lit 0)
  let r64 ← if kind == .int then pure r else toIntE r
  let piece (i : Nat) : BuildM Sig := do
    toFloatE (← bitAnd (← rshift r64 (← lit (16 * i))) (← lit 65535))
  selectE (← binary .eq q (← lit 0)) (← piece 0)
    (← selectE (← binary .eq q (← lit 1)) (← piece 1)
      (← selectE (← binary .eq q (← lit 2)) (← piece 2) (← piece 3)))

/-- Frames per op of the edge program of sort `k`. -/
def intEdgeFramesPerOp (k : ScalarKind) : Nat := 4 * (intEdgePairs k).size

/-- The edge program's output: every op of `intEdgeOps k`, multiplexed by
    the frame. The pair tables have one size (`intEdgePairs` and
    `intEdgeShiftPairs` both hold 106). -/
def intEdgeSig (k : ScalarKind) : BuildM Sig := do
  let pairs := intEdgePairs k
  let shiftPairs := intEdgeShiftPairs k
  let fpo := intEdgeFramesPerOp k
  -- `litI`: an int divisor, so the frame arithmetic is integer (a bare
  -- literal beside an `int` under no expectation would be a float)
  let t ← sampleIndex
  let o ← div t (← litI fpo)
  let r ← binary .mod t (← litI fpo)
  let p ← rshift r (← lit 2)
  let q ← bitAnd r (← lit 3)
  let a ← edgeOperand k (pairs.map (·.1)) p
  let b ← edgeOperand k (pairs.map (·.2)) p
  let a2 ← edgeOperand k (shiftPairs.map (·.1)) p
  let s ← edgeOperand k (shiftPairs.map (·.2)) p
  let ops := intEdgeOps k
  let mut out ← lit 0
  for i' in [0:ops.size] do
    let i := ops.size - 1 - i'
    let (_, kind, build) := ops[i]!
    let readout ← edgeReadout kind (← build a b a2 s) q
    out ← selectE (← binary .eq o (← litI i)) readout out
  pure out

/-- `IntEdge64` / `IntEdge32`: the edge program of a sort, a float output. -/
def buildIntEdge (k : ScalarKind) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  assemble arena (if k == .int32 then "IntEdge32" else "IntEdge64")
      #[{ name := "out", type? := some (.scalar .float) }] #[] do
    pure { assigns := #[(.port ⟨0⟩, ← intEdgeSig k)] }

/-- The float inputs of the conversion program, as `lit mantissa exponent`
    (value `mantissa · 10^-exponent`), each exact in f32 or saturating either
    way, then the three specials by code (1 = +∞, 2 = -∞, 3 = NaN, built at
    runtime from `10³⁰⁰²`, which is ∞ in f64 and f32 alike). -/
def intEdgeFloats : Array (Int × Nat × Nat) := #[
  (0, 0, 0), (5, 1, 0), (-5, 1, 0), (15, 1, 0), (-25, 1, 0),
  (2147483648, 0, 0), (-2147483648, 0, 0), (-2147483904, 0, 0),
  (4294967296, 0, 0), (1099511627776, 0, 0), (-1099511627776, 0, 0),
  (4611686018427387904, 0, 0), (9223372036854775808, 0, 0),
  (-9223372036854775808, 0, 0), (18446744073709551616, 0, 0),
  ((10 : Int) ^ 30, 0, 0), (-(10 : Int) ^ 30, 0, 0), (12345675, 2, 0), (-775, 2, 0),
  ((10 : Int) ^ 300, 0, 1), ((10 : Int) ^ 300, 0, 2), ((10 : Int) ^ 300, 0, 3)]

/-- The conversion ops: `toInt` and `toInt32` of a float, saturating. -/
def intEdgeConvOps : Array (String × ScalarKind) := #[("toInt", .int), ("toInt32", .int32)]

def intEdgeConvFramesPerOp : Nat := 4 * intEdgeFloats.size

def intEdgeConvSig : BuildM Sig := do
  let fpo := intEdgeConvFramesPerOp
  -- `litI`: an int divisor, so the frame arithmetic is integer (a bare
  -- literal beside an `int` under no expectation would be a float)
  let t ← sampleIndex
  let o ← div t (← litI fpo)
  let r ← binary .mod t (← litI fpo)
  let p ← rshift r (← lit 2)
  let q ← bitAnd r (← lit 3)
  let f ← index (← arr (← intEdgeFloats.mapM fun (m, e, _) => lit m e)) p
  let code ← index (← arr (← intEdgeFloats.mapM fun (_, _, c) => lit c)) p
  let inf ← mul f f
  let x ← selectE (← binary .eq code (← lit 1)) inf
    (← selectE (← binary .eq code (← lit 2)) (← neg inf)
      (← selectE (← binary .eq code (← lit 3)) (← sub inf inf) f))
  let toI ← edgeReadout .int (← toIntE x) q
  let toI32 ← edgeReadout .int32 (← toInt32E x) q
  selectE (← binary .eq o (← litI 0)) toI
    (← selectE (← binary .eq o (← litI 1)) toI32 (← lit 0))

/-- `IntEdgeConv`: float → int and float → int32 on the edge floats. -/
def buildIntEdgeConv (arena : Arena) : Except String (Arena × ProgramIdx) :=
  assemble arena "IntEdgeConv" #[{ name := "out", type? := some (.scalar .float) }] #[] do
    pure { assigns := #[(.port ⟨0⟩, ← intEdgeConvSig)] }

/-- The edge programs, by registered name. -/
def intEdgeFixtures : Array (String × (Arena → Except String (Arena × ProgramIdx))) := #[
  ("IntEdge64", buildIntEdge .int), ("IntEdge32", buildIntEdge .int32),
  ("IntEdgeConv", buildIntEdgeConv)]

abbrev ModesM := BuildM (Array ModalMode)

def modeArray (modes : Array (BuildM ModalMode)) : ModesM := modes.mapM id

def cmodeToModalMode (mode : Tropical.Testing.ArrowOracles.CMode) : BuildM ModalMode := do
  pure {
    sigma := ← litF (-mode.pole.re)
    omega := ← litF mode.pole.im
    cre := ← litF mode.amp.re
    cim := ← litF mode.amp.im
    deg := mode.deg
  }

def buildModalBankArrow (name : String) (modes : ModesM) (anchor : BuildM Sig)
    (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let modes ← modes
    let anchor ← anchor
    let term := modalBankTerm modes anchor (← clockLit)
    emitTerm (normalize term)) arena

def buildModalBankDirect (name : String) (modes : ModesM) (anchor : BuildM Sig)
    (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    modalBankSig (← modes) (← clockLit) (← anchor)) arena

def buildModalBankTable (name : String) (modes : ModesM) (anchor : BuildM Sig)
    (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    modalBankSigTable (← modes) (← clockLit) (← anchor)) arena

def buildBloomComposed (name : String) (pairs : BuildM (Array BloomPair))
    (anchor : BuildM Sig) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    bloomComposedSig (← pairs) (← clockLit) (← anchor)) arena

def buildModalBankPair (nameRe nameIm : String) (modes : ModesM)
    (anchor : BuildM Sig) (arena : Arena) :
    Except String ((Arena × ProgramIdx) × (Arena × ProgramIdx)) := do
  let reBuild : BuildM Sig := do
    let (real, _) ← modalBankSigPairTable (← modes) (← clockLit) (← anchor)
    pure real
  let imBuild : BuildM Sig := do
    let (_, imag) ← modalBankSigPairTable (← modes) (← clockLit) (← anchor)
    pure imag
  pure (← buildExprCarrier nameRe reBuild arena, ← buildExprCarrier nameIm imBuild arena)

def buildHeterodyne (name : String) (modes : ModesM) (wm b : Float)
    (anchor : BuildM Sig) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let anchor ← anchor
    let clock ← clockLit
    let (real, imag) ← modalBankSigPairTable (← modes) clock anchor
    let relative ← relClockQ clock anchor
    let qScale ← clockOneLit
    let secondsQ ← div (← toFloatE relative) qScale
    let seconds ← div secondsQ (← sampleRate)
    let angle ← mul (← litF b) (← sinSig (← mul (← litF wm) seconds))
    let realPart ← mul real (← cosSig angle)
    let imagPart ← mul imag (← sinSig angle)
    sub realPart imagPart) arena

def buildIntegratedPoleReading (name : String) (carrier lfo : ModesM)
    (omegaDepth : Float) (anchor : BuildM Sig) (arena : Arena) :
    Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let anchor ← anchor
    let clock ← clockLit
    let (real, imag) ← modalBankSigPairTable (← carrier) clock anchor
    let integrated ← integrateBank (← lfo)
    let integral ← modalBankSig integrated clock anchor
    let angle ← mul (← litF omegaDepth) integral
    let realPart ← mul real (← cosSig angle)
    let imagPart ← mul imag (← sinSig angle)
    sub realPart imagPart) arena

def buildModalReverb (name : String)
    (voice reverb : Array (Tropical.Testing.ArrowOracles.Cplx ×
      Tropical.Testing.ArrowOracles.Cplx))
    (anchor : BuildM Sig) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildModalBankArrow name (do
    (Tropical.Testing.ArrowOracles.residueCompose voice reverb).mapM cmodeToModalMode)
    anchor arena

def buildModalReverbSym (name : String) (voice reverb : ModesM)
    (anchor : BuildM Sig) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildModalBankArrow name (do residueComposeE (← voice) (← reverb)) anchor arena

def buildModalReverbSymC (name : String) (voice reverb : ModesM)
    (anchor : BuildM Sig) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildModalBankArrow name (do residueComposeEC (← voice) (← reverb)) anchor arena

def buildModalReverbDD (name : String) (voice reverb : ModesM)
    (anchor : BuildM Sig) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let pairs ← residueComposeDD (← voice) (← reverb)
    modalBankSigTableDD pairs (← clockLit) (← anchor)) arena

def buildModalReverbBanked (name : String) (voice reverb : ModesM)
    (anchor : BuildM Sig) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildModalBankArrow name (do residueComposeBanked (← voice) (← reverb)) anchor arena

def buildModalBankWarped (name : String) (modes : ModesM) (anchor : BuildM Sig)
    (warp : Clock → BuildM Clock) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let term : ArrowTerm := .warp warp
      (modalBankTerm (← modes) (← anchor) (← clockLit))
    emitTerm (normalize term)) arena

private def buildDamp (damp? : Option (BuildM Sig × BuildM Sig)) :
    BuildM (Option (Sig × Sig)) :=
  match damp? with
  | none => pure none
  | some (depth, rate) => do pure (some (← depth, ← rate))

def buildModalBankDir (name : String) (modes : ModesM) (anchor dir : BuildM Sig)
    (arena : Arena) (damp? : Option (BuildM Sig × BuildM Sig) := none) :
    Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let term := modalBankTermDir (← modes) (← anchor) (← clockLit) (← dir)
      (← buildDamp damp?)
    emitTerm (normalize term)) arena

def buildModalBankDirWith
    (lower : Array ModalMode → Sig → Sig → Sig → Option Sig → BuildM Sig)
    (name : String) (modes : ModesM) (anchor dir : BuildM Sig) (arena : Arena)
    (damp? : Option (BuildM Sig × BuildM Sig) := none) :
    Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let term := modalBankTermDirWith lower (← modes) (← anchor) (← clockLit)
      (← dir) (← buildDamp damp?)
    emitTerm (normalize term)) arena

def buildModalAddrRamp (name : String) (modes : ModesM) (anchor : BuildM Sig)
    (offsetSeconds : Float) (arena : Arena) : Except String (Arena × ProgramIdx) :=
  buildExprCarrier name (do
    let clock ← clockLit
    let address : ArrowTerm := .arrUn (fun clock => do
      let sample ← rshift clock (← clockFracLit)
      let seconds ← div (← toFloatE sample) (← sampleRate)
      sub seconds (← litF offsetSeconds)) (.clk clock)
    let bank := modalBankTerm (← modes) (← anchor) clock
    emitTerm (normalize (.swarp modalAddrWarp address bank))) arena

end Tropical.EmitArrow
