import Tropical.EmitArrow
import Tropical.Stdlib
import Tropical.Playground.DecodeMetadata
import Tropical.Ir.Strata
import Tropical.Ir.CompileResolved
import Tropical.Testing.PlanWire

/-!
# Arena-native authoring qualification fixtures

The Phase 5 gates qualify the arena-native authoring path directly.  The
recursive comparison halves have been retired; exact observations captured
while both implementations coexisted now serve as frozen acceptance values.
The fixtures pin node counts and wire sizes, check authored and reachable
arenas, and exercise instances, nested outputs, sharing, reductions, numeric
helpers, arrow effects, and the production modal carrier.
-/

namespace Tropical.Testing.EmitArrow

open Tropical.Ir

private abbrev NativeBody := Tropical.EmitArrow.ProgramBody

structure FixtureCorpus where
  arena : Arena
  leaf : ProgramIdx
  root : ProgramIdx

structure Phase1Evidence where
  programCount : Nat
  authoredUniqueNodes : Nat
  leafReachableNodes : Nat
  rootReachableNodes : Nat
  leafWireBytes : Nat
  rootWireBytes : Nat
deriving Repr

structure Phase2Evidence where
  stdlibPrograms : Nat
  stdlibUniqueNodes : Nat
  numericReachableNodes : Nat
  numericWireBytes : Nat
  carrierInstances : Nat
  carrierReachableNodes : Nat
  carrierWireBytes : Nat
deriving Repr

structure Phase3Evidence where
  authoredUniqueNodes : Nat
  reachableUniqueNodes : Nat
  routedReductions : Nat
  wireBytes : Nat
deriving Repr

structure Phase5Evidence where
  vocabularyKinds : Nat
  reservedParameters : Nat
  vocabularyFingerprint : String
deriving Repr

private def output : Array OutputDecl :=
  #[{ name := "out", type? := some (.scalar .float) }]

/-! The ID-native vertical slice.  `shared` is bound once and reused; the
    second equal `add` must observe an intern hit.  The routed reduction and
    the `99 + 99` expression are deliberately dead so the production exit can
    demonstrate reachable-only copying. -/
private def buildNativeCorpus : Except String FixtureCorpus := do
  let (arena, leaf) ← Tropical.EmitArrow.assemble {}
      "Phase1Leaf" output #[] do
    let seven ← Tropical.EmitArrow.lit 7
    pure ({ assigns := #[(.port ⟨0⟩, seven)] } : NativeBody)

  let (arena, identity) ← Tropical.EmitArrow.assemble arena
      "Phase1Identity" output #[] do
    let zero ← Tropical.EmitArrow.lit 0
    let input ← Tropical.EmitArrow.inputRef ⟨0⟩
    pure ({
      inputs := #[{
        name := "x"
        type? := some (.scalar .float)
        defaultSig := some zero
      }]
      assigns := #[(.port ⟨0⟩, input)]
    } : NativeBody)

  let (arena, root) ← Tropical.EmitArrow.assemble arena
      "Phase1Root" output #[("Phase1Identity", identity)] do
    let two ← Tropical.EmitArrow.lit 2
    let three ← Tropical.EmitArrow.lit 3
    let shared ← Tropical.EmitArrow.add two three
    let equalAgain ← Tropical.EmitArrow.add two three
    unless shared == equalAgain do
      throw "EmitArrow phase-1 fixture: equal construction missed eintern"

    let loop ← Tropical.EmitArrow.loopIdx 7
    let table ← Tropical.EmitArrow.arr #[two, three]
    let indexed ← Tropical.EmitArrow.index table loop
    let contribution ← Tropical.EmitArrow.mul indexed shared
    let bank ← Tropical.EmitArrow.bankSum
      2 #[table] contribution none 7

    -- Constructor/order coverage for the routed sibling.  It is dead at the
    -- root on purpose; the copied reachable graph must omit it.
    let routed ← Tropical.EmitArrow.routedSum
      2 2 #[some 1, none] #[table] #[shared, indexed] none 7
    let builder ← get
    unless builder.exprs.deref routed == some
        (.routedSum 2 2 #[some 1, none] #[table]
          #[shared, indexed] none 7) do
      throw "EmitArrow phase-1 fixture: routedSum changed authored order"

    let ninetyNine ← Tropical.EmitArrow.lit 99
    let _dead ← Tropical.EmitArrow.add ninetyNine ninetyNine

    let instanceIdx ← Tropical.EmitArrow.inst
      "identity" "Phase1Identity" #[{ port := ⟨0⟩, value := shared }]
    let nested ← Tropical.EmitArrow.nestedOut instanceIdx ⟨0⟩
    let result ← Tropical.EmitArrow.add nested bank
    pure ({ assigns := #[(.port ⟨0⟩, result)] } : NativeBody)

  pure { arena, leaf, root }

private def resolvedWire (corpus : FixtureCorpus) (idx : ProgramIdx) :
    Except String (ExprArena × String) := do
  let (exprs, core) ← (Tropical.Ir.Strata.runResolved
    { inlineNested := true } corpus.arena idx).mapError (·.message)
  let plan ← Tropical.Ir.CompileResolved.compileResolved core exprs
  let wire ← plan.toWire
  pure (exprs, wire.compress)

private def failureIsAtomic : Bool :=
  let original : Arena := {}
  let failed := Tropical.EmitArrow.assemble original
    "Phase1Refusal" output #[] do
      let _ ← Tropical.EmitArrow.lit 1
      throw "phase-1 refusal"
  match failed with
  | .error message =>
      message == "phase-1 refusal" &&
        original.programs.isEmpty && original.exprs.nodes.isEmpty
  | .ok _ => false

/-- Pure acceptance evidence for the phase-1 vertical slice.  Exact node and
    wire-size observations make explicit sharing, reachability, and output
    stability observable after retirement of the recursive fixture. -/
def phase1Evidence : Except String Phase1Evidence := do
  let native ← buildNativeCorpus

  unless native.arena.programs.size == 3 do
    throw s!"EmitArrow phase-1 fixture: expected 3 programs, got {native.arena.programs.size}"
  unless native.arena.exprs.nodes.size == 16 do
    throw s!"EmitArrow phase-1 fixture: expected 16 authored unique nodes, got {native.arena.exprs.nodes.size}"
  unless native.arena.exprs.wf do
    throw "EmitArrow phase-1 fixture: authored expression arena is not child-descending"
  unless failureIsAtomic do
    throw "EmitArrow phase-1 fixture: failed build changed publication behavior"

  let (nativeLeafExprs, nativeLeafWire) ← resolvedWire native native.leaf
  unless nativeLeafExprs.nodes.size == 1 do
    throw s!"EmitArrow phase-1 fixture: expected 1 reachable leaf node, got {nativeLeafExprs.nodes.size}"
  unless nativeLeafExprs.wf do
    throw "EmitArrow phase-1 fixture: reachable leaf arena is not child-descending"
  unless nativeLeafWire.length == 341 do
    throw s!"EmitArrow phase-1 fixture: expected 341 leaf wire bytes, got {nativeLeafWire.length}"

  let (nativeRootExprs, nativeRootWire) ← resolvedWire native native.root
  unless nativeRootExprs.nodes.size == 9 do
    throw s!"EmitArrow phase-1 fixture: expected 9 reachable root nodes, got {nativeRootExprs.nodes.size}"
  unless nativeRootExprs.nodes.size < native.arena.exprs.nodes.size do
    throw "EmitArrow phase-1 fixture: dead authored nodes survived reachability GC"
  unless nativeRootExprs.wf do
    throw "EmitArrow phase-1 fixture: reachable root arena is not child-descending"
  unless nativeRootWire.length == 1945 do
    throw s!"EmitArrow phase-1 fixture: expected 1945 root wire bytes, got {nativeRootWire.length}"

  pure {
    programCount := native.arena.programs.size
    authoredUniqueNodes := native.arena.exprs.nodes.size
    leafReachableNodes := nativeLeafExprs.nodes.size
    rootReachableNodes := nativeRootExprs.nodes.size
    leafWireBytes := nativeLeafWire.length
    rootWireBytes := nativeRootWire.length
  }

def phase1VerticalSlicePasses : Bool := phase1Evidence.isOk

def runPhase1Gate : IO Bool := do
  match phase1Evidence with
  | .ok evidence =>
    IO.println s!"  PASS  arena-native-phase1  {repr evidence}"
    pure true
  | .error error =>
    IO.println s!"  FAIL  arena-native-phase1  {error}"
    pure false

private def numericOutputs : Array OutputDecl := #[
  { name := "fixed_sin", type? := some (.scalar .int) },
  { name := "fixed_cos", type? := some (.scalar .int) },
  { name := "fixed_out", type? := some (.scalar .float) },
  { name := "phasor", type? := some (.scalar .float) },
  { name := "sin", type? := some (.scalar .float) },
  { name := "exp", type? := some (.scalar .float) },
  { name := "log", type? := some (.scalar .float) },
  { name := "atan2", type? := some (.scalar .float) },
  { name := "cos", type? := some (.scalar .float) },
  { name := "two_pi", type? := some (.scalar .float) },
  { name := "half_pi", type? := some (.scalar .float) }]

private def buildNativeNumerics : Except String (Arena × ProgramIdx) :=
  Tropical.EmitArrow.assemble {} "Phase2Numerics"
      numericOutputs #[] do
    let phaseQ ← Tropical.EmitArrow.inputRef ⟨0⟩
    let freq ← Tropical.EmitArrow.inputRef ⟨1⟩
    let offset ← Tropical.EmitArrow.inputRef ⟨2⟩
    let clk ← Tropical.EmitArrow.inputRef ⟨3⟩
    let x ← Tropical.EmitArrow.inputRef ⟨4⟩
    let y ← Tropical.EmitArrow.inputRef ⟨5⟩
    let fixedSin ← Tropical.EmitArrow.fixedSinCycSig phaseQ
    let fixedCos ← Tropical.EmitArrow.fixedCosCycSig phaseQ
    let fixedOut ← Tropical.EmitArrow.fixedOutQ 30 fixedSin
    let phasor ← Tropical.EmitArrow.phasorPhaseSig freq offset clk
    let sine ← Tropical.EmitArrow.sinSig x
    let exponential ← Tropical.EmitArrow.expSig x
    let logarithm ← Tropical.EmitArrow.logSig x
    let angle ← Tropical.EmitArrow.atan2E y x
    let cosine ← Tropical.EmitArrow.cosSig x
    let twoPi ← Tropical.EmitArrow.twoPiE
    let halfPi ← Tropical.EmitArrow.halfPiE
    let zero ← Tropical.EmitArrow.lit 0
    let twoTwenty ← Tropical.EmitArrow.lit 220
    let one ← Tropical.EmitArrow.lit 1
    let half ← Tropical.EmitArrow.lit 5 1
    let inputs : Array Tropical.EmitArrow.AInputDecl := #[
      { name := "phase", type? := some (.scalar .int), defaultSig := some zero },
      { name := "freq", type? := some (.scalar .float), defaultSig := some twoTwenty },
      { name := "offset", type? := some (.scalar .float), defaultSig := some zero },
      { name := "clk", type? := some (.scalar .int), defaultSig := some zero },
      { name := "x", type? := some (.scalar .float), defaultSig := some one },
      { name := "y", type? := some (.scalar .float), defaultSig := some half }]
    pure ({ inputs, assigns := #[
      (.port ⟨0⟩, fixedSin), (.port ⟨1⟩, fixedCos),
      (.port ⟨2⟩, fixedOut), (.port ⟨3⟩, phasor),
      (.port ⟨4⟩, sine), (.port ⟨5⟩, exponential),
      (.port ⟨6⟩, logarithm), (.port ⟨7⟩, angle),
      (.port ⟨8⟩, cosine), (.port ⟨9⟩, twoPi),
      (.port ⟨10⟩, halfPi)] } : NativeBody)

private def buildNativeArrowCarrier (arena : Arena)
    (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← Tropical.EmitArrow.buildRegistry arena resolved
    #["FixedSinOsc"]
  Tropical.EmitArrow.assemble arena "Phase2ArrowCarrier" output
      registry do
    let base ← Tropical.EmitArrow.clockLit
    let delta ← Tropical.EmitArrow.litI 17
    -- Independently reconstructing an equal node must hit the same interner,
    -- while the bound IDs below are the values actually reused by the term.
    let reuseA ← Tropical.EmitArrow.add base delta
    let reuseB ← Tropical.EmitArrow.add base delta
    unless reuseA == reuseB do
      throw "EmitArrow phase-2 fixture: repeated scalar construction missed eintern"
    let voice := fun (pitch : Int) => ({
      programName := "FixedSinOsc"
      wire := fun clock => do
        let pitch ← Tropical.EmitArrow.lit pitch
        let zero ← Tropical.EmitArrow.lit 0
        pure #[
          { port := ⟨0⟩, value := pitch },
          { port := ⟨1⟩, value := clock },
          { port := ⟨2⟩, value := zero }]
      phaseAnchor := some (⟨2⟩, fun shift => do
        let shiftFloat ← Tropical.EmitArrow.toFloatE shift
        let scale ← Tropical.EmitArrow.lit 1 9
        Tropical.EmitArrow.mul shiftFloat scale)
    } : Tropical.EmitArrow.Voice)
    let modulator : Tropical.EmitArrow.ArrowTerm :=
      .gen (voice 11) "mod" base
    let carrier : Tropical.EmitArrow.ArrowTerm :=
      .gen (voice 220) "carrier" base
    let signalWarp := fun clock signal => do
      let eight ← Tropical.EmitArrow.lit 8
      let scaled ← Tropical.EmitArrow.mul signal eight
      let delta ← Tropical.EmitArrow.toIntE scaled
      Tropical.EmitArrow.sub clock delta
    let warped : Tropical.EmitArrow.ArrowTerm := .warp
      (fun clock => Tropical.EmitArrow.add clock delta)
      (.swarp signalWarp modulator carrier)
    let sibling : Tropical.EmitArrow.ArrowTerm := .warp
      (fun clock => Tropical.EmitArrow.sub clock delta)
      (.gen (voice 330) "sibling" base)
    let combine := fun (signals : Array Tropical.EmitArrow.Sig) => do
      let two ← Tropical.EmitArrow.lit 2
      let left ← Tropical.EmitArrow.mul two signals[0]!
      let three ← Tropical.EmitArrow.lit 3
      let right ← Tropical.EmitArrow.mul three signals[1]!
      Tropical.EmitArrow.add left right
    let term : Tropical.EmitArrow.ArrowTerm :=
      .arrN combine #[warped, sibling]
    let outputSignal ← Tropical.EmitArrow.emitTerm
      (Tropical.EmitArrow.normalize term)
    let builder ← get
    unless builder.decls.map (·.name) == #["mod0", "carrier1", "sibling2"] do
      throw "EmitArrow phase-2 fixture: instance effects are not left-to-right"
    pure ({ assigns := #[(.port ⟨0⟩, outputSignal)] } : NativeBody)

def phase2Evidence : Except String Phase2Evidence := do
  let (nativeNumerics, nativeNumericProgram) ← buildNativeNumerics
  let nativeNumericCorpus : FixtureCorpus := {
    arena := nativeNumerics, leaf := nativeNumericProgram,
    root := nativeNumericProgram }
  let (nativeNumericExprs, nativeNumericWire) ←
    resolvedWire nativeNumericCorpus nativeNumericProgram
  unless nativeNumericExprs.wf do
    throw "EmitArrow phase-2 fixture: reachable numeric arena is not child-descending"
  unless nativeNumericExprs.nodes.size == 268 do
    throw s!"EmitArrow phase-2 fixture: expected 268 reachable numeric nodes, got {nativeNumericExprs.nodes.size}"
  unless nativeNumericWire.length == 46451 do
    throw s!"EmitArrow phase-2 fixture: expected 46451 numeric wire bytes, got {nativeNumericWire.length}"

  let (nativeStdlib, nativeResolved) ← Tropical.EmitArrow.buildStdlibChain
  unless nativeStdlib.programs.size == 15 do
    throw s!"EmitArrow phase-2 fixture: expected 15 stdlib programs, got {nativeStdlib.programs.size}"
  unless nativeStdlib.exprs.nodes.size == 283 do
    throw s!"EmitArrow phase-2 fixture: expected 283 stdlib nodes, got {nativeStdlib.exprs.nodes.size}"
  unless nativeStdlib.exprs.wf do
    throw "EmitArrow phase-2 fixture: stdlib authored arena is not child-descending"
  let (nativeArena, nativeCarrier) ←
    buildNativeArrowCarrier nativeStdlib nativeResolved
  let some nativeProgram := nativeArena.program? nativeCarrier
    | throw "EmitArrow phase-2 fixture: native carrier missing"
  unless nativeProgram.decls.size == 3 do
    throw s!"EmitArrow phase-2 fixture: expected 3 carrier instances, got {nativeProgram.decls.size}"
  let nativeCorpus : FixtureCorpus :=
    { arena := nativeArena, leaf := nativeCarrier, root := nativeCarrier }
  let (nativeExprs, nativeWire) ← resolvedWire nativeCorpus nativeCarrier
  unless nativeExprs.wf do
    throw "EmitArrow phase-2 fixture: reachable arrow arena is not child-descending"
  unless nativeExprs.nodes.size == 206 do
    throw s!"EmitArrow phase-2 fixture: expected 206 reachable carrier nodes, got {nativeExprs.nodes.size}"
  unless nativeWire.length == 36103 do
    throw s!"EmitArrow phase-2 fixture: expected 36103 carrier wire bytes, got {nativeWire.length}"
  pure {
    stdlibPrograms := nativeStdlib.programs.size
    stdlibUniqueNodes := nativeStdlib.exprs.nodes.size
    numericReachableNodes := nativeNumericExprs.nodes.size
    numericWireBytes := nativeNumericWire.length
    carrierInstances := nativeProgram.decls.size
    carrierReachableNodes := nativeExprs.nodes.size
    carrierWireBytes := nativeWire.length
  }

def runPhase2Gate : IO Bool := do
  match phase2Evidence with
  | .ok evidence =>
    IO.println s!"  PASS  arena-native-phase2  {repr evidence}"
    pure true
  | .error error =>
    IO.println s!"  FAIL  arena-native-phase2  {error}"
    pure false

/-! Phase 3 retains a room → factored all-pass product → room spine and
    crosses the specialized routed terminal used by production Phaser. -/
private def buildNativeModalCarrier : Except String (Arena × ProgramIdx) :=
  Tropical.EmitArrow.assemble {} "Phase3ModalCarrier" output #[] do
    let mode := fun (sigma omega cre cim : Int) => do
      let sigma ← Tropical.EmitArrow.lit sigma 1
      let omega ← Tropical.EmitArrow.lit omega
      let cre ← Tropical.EmitArrow.lit cre 1
      let cim ← Tropical.EmitArrow.lit cim 1
      pure ({ sigma, omega, cre, cim } : Tropical.EmitArrow.ModalMode)
    let source ← #[ (4, 220, 8, 1), (6, -330, 5, -2) ].mapM
      fun (sigma, omega, cre, cim) => mode sigma omega cre cim
    let room1 ← #[ (8, 37, 3, 1), (11, -51, 2, -1) ].mapM
      fun (sigma, omega, cre, cim) => mode sigma omega cre cim
    let room2 ← #[ (13, 73, 2, 1), (17, -91, 1, -1) ].mapM
      fun (sigma, omega, cre, cim) => mode sigma omega cre cim
    let zero ← Tropical.EmitArrow.lit 0
    let center ← Tropical.EmitArrow.lit 700
    let sweep ← Tropical.EmitArrow.lit 15 1
    let rate ← Tropical.EmitArrow.lit 2 1
    let mix ← Tropical.EmitArrow.lit 5 1
    let clock ← Tropical.EmitArrow.clockLit
    let control := Tropical.EmitArrow.ModalControlRef.constant
    let (phaser, topology) := Tropical.EmitArrow.modalPhaserTopology
      "fixture" "room1" (control center) (control sweep) (control rate)
      (control mix) #[1.0]
    let graph : Tropical.EmitArrow.PatchGraph := {
      nodes := #[
        { id := "source", node := .modalSource source zero clock none },
        { id := "room1", node := .modalReverb "source" room1 (some { dir := zero }) }]
        ++ topology ++ #[
        { id := "fixture", node := phaser },
        { id := "room2", node := .modalReverb "fixture" room2 (some { dir := zero }) }]
      output := "room2" }
    let term ← Tropical.EmitArrow.lowerGraph graph
    let signal ← Tropical.EmitArrow.emitTerm
      (Tropical.EmitArrow.normalize term)
    pure ({ assigns := #[(.port ⟨0⟩, signal)] } : NativeBody)

private def routedReductionCount (exprs : ExprArena) : Nat :=
  exprs.nodes.foldl (fun total node =>
    if node matches .routedSum .. then total + 1 else total) 0

def phase3Evidence : Except String Phase3Evidence := do
  let (nativeArena, nativeProgram) ← buildNativeModalCarrier
  unless nativeArena.exprs.wf do
    throw "EmitArrow phase-3 fixture: authored modal arena is not child-descending"
  unless nativeArena.exprs.nodes.size == 2357 do
    throw s!"EmitArrow phase-3 fixture: expected 2357 authored nodes, got {nativeArena.exprs.nodes.size}"
  let nativeCorpus : FixtureCorpus := {
    arena := nativeArena, leaf := nativeProgram, root := nativeProgram }
  let (nativeExprs, nativeWire) ← resolvedWire nativeCorpus nativeProgram
  unless nativeExprs.wf do
    throw "EmitArrow phase-3 fixture: reachable modal arena is not child-descending"
  unless nativeExprs.nodes.size == 2160 do
    throw s!"EmitArrow phase-3 fixture: expected 2160 reachable nodes, got {nativeExprs.nodes.size}"
  let routed := routedReductionCount nativeExprs
  unless routed == 24 do
    throw s!"EmitArrow phase-3 fixture: expected 24 routed reductions, got {routed}"
  unless nativeWire.length == 719467 do
    throw s!"EmitArrow phase-3 fixture: expected 719467 wire bytes, got {nativeWire.length}"
  pure {
    authoredUniqueNodes := nativeArena.exprs.nodes.size
    reachableUniqueNodes := nativeExprs.nodes.size
    routedReductions := routed
    wireBytes := nativeWire.length }

def runPhase3Gate : IO Bool := do
  match phase3Evidence with
  | .ok evidence =>
    IO.println s!"  PASS  arena-native-phase3  {repr evidence}"
    pure true
  | .error error =>
    IO.println s!"  FAIL  arena-native-phase3  {error}"
    pure false

def phase5Evidence : Except String Phase5Evidence := do
  let kinds := Tropical.Playground.Metadata.vocabularyKinds
  unless kinds.size == 20 do
    throw s!"EmitArrow phase-5 fixture: expected 20 served vocabulary kinds, got {kinds.size}"
  let parameters := Tropical.Playground.Metadata.collectParams #[]
  let expectedNames := #[
    Tropical.Playground.Metadata.masterVelocityParam,
    Tropical.Playground.Metadata.masterTauBaseParam,
    Tropical.Playground.Metadata.masterGainParam]
  unless parameters.map (·.1) == expectedNames do
    throw "EmitArrow phase-5 fixture: reserved parameter order changed"
  let fingerprint := Tropical.Playground.Metadata.vocabularyFingerprint
  unless fingerprint == "fnv1a64:e2d9d7b44e8c3bbf" do
    throw s!"EmitArrow phase-5 fixture: vocabulary fingerprint changed to {fingerprint}"
  pure {
    vocabularyKinds := kinds.size
    reservedParameters := parameters.size
    vocabularyFingerprint := fingerprint }

def runPhase5Gate : IO Bool := do
  match phase5Evidence with
  | .ok evidence =>
    IO.println s!"  PASS  arena-native-phase5  {repr evidence}"
    pure true
  | .error error =>
    IO.println s!"  FAIL  arena-native-phase5  {error}"
    pure false

end Tropical.Testing.EmitArrow
