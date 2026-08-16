import Tropical.EmitArrow
import Tropical.Ir.Strata
import Tropical.Ir.CompileResolved
import Tropical.Testing.PlanWire
import Tropical.Testing.ArrowOracles

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
  let scale ← lit 4294967296
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
  let shift ← lit 32
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
  let scale ← lit 4294967296
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
    let qScale ← lit 4294967296
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
    let qScale ← lit 4294967296
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
  let shift ← lit 32
  let high ← rshift clock shift
  let mask ← lit 4294967295
  let low ← bitAnd clock mask
  let increment ← lit 21426140
  let highProduct ← mul increment high
  let lowProduct ← mul increment low
  let lowShifted ← rshift lowProduct shift
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
  let shifted ← rshift clock (← lit 32)
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
    let qScale ← lit 4294967296
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
      let sample ← rshift clock (← lit 32)
      let seconds ← div (← toFloatE sample) (← sampleRate)
      sub seconds (← litF offsetSeconds)) (.clk clock)
    let bank := modalBankTerm (← modes) (← anchor) clock
    emitTerm (normalize (.swarp modalAddrWarp address bank))) arena

end Tropical.EmitArrow
