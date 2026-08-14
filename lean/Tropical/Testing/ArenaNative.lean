import Tropical.EmitArrow.Sig
import Tropical.EmitArrow.Term
import Tropical.EmitArrow.Numerics
import Tropical.EmitArrow.ArenaSig
import Tropical.EmitArrow.ArenaTerm
import Tropical.EmitArrow.Patch
import Tropical.EmitArrow.ArenaPatch
import Tropical.Stdlib
import Tropical.Ir.Strata
import Tropical.Ir.CompileResolved
import Tropical.Testing.PlanWire

/-!
# Arena-native authoring phase-1 fixture

This module keeps a deliberately small old/new corpus while both authoring
representations coexist.  It compares the production plan wire exactly (not
just by hash), pins unique-node counts, checks the authored and reachable
arenas, and exercises an instance input, nested output, explicit sharing,
`bankSum`, and `routedSum` ordering.
-/

namespace Tropical.Testing.ArenaNative

open Tropical.Ir

private abbrev NativeBody := Tropical.EmitArrow.ArenaNative.ProgramBody

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

private def output : Array OutputDecl :=
  #[{ name := "out", type? := some (.scalar .float) }]

/-! The frozen recursive-authoring baseline.  Expressions and declaration
    order intentionally mirror the arena-native fixture below. -/
private def buildLegacyCorpus : Except String FixtureCorpus := do
  let (arena, leaf) := Tropical.EmitArrow.assemble {} "Phase1Leaf" #[] output
    #[] #[(.port ⟨0⟩, Tropical.EmitArrow.lit 7)] #[]

  let identityInputs : Array Tropical.EmitArrow.AInputDecl := #[{
    name := "x"
    type? := some (.scalar .float)
    defaultSig := some (Tropical.EmitArrow.lit 0)
  }]
  let (arena, identity) := Tropical.EmitArrow.assemble arena "Phase1Identity"
    identityInputs output #[]
    #[(.port ⟨0⟩, .inputRef ⟨0⟩)] #[]

  let two := Tropical.EmitArrow.lit 2
  let three := Tropical.EmitArrow.lit 3
  let shared := Tropical.EmitArrow.add two three
  let table : Tropical.EmitArrow.Sig := .arr #[two, three]
  let loop : Tropical.EmitArrow.Sig := .loopIdx 7
  let indexed : Tropical.EmitArrow.Sig := .index table loop
  let body := Tropical.EmitArrow.mul indexed shared
  let bank : Tropical.EmitArrow.Sig := .bankSum 2 #[table] body none 7
  let instanceDecl : Tropical.EmitArrow.AInst := {
    name := "identity"
    programName := "Phase1Identity"
    inputs := #[{ port := ⟨0⟩, value := shared }]
  }
  let nested : Tropical.EmitArrow.Sig := .nestedOut ⟨0⟩ ⟨0⟩
  let result := Tropical.EmitArrow.add nested bank
  let (arena, root) := Tropical.EmitArrow.assemble arena "Phase1Root" #[]
    output #[instanceDecl] #[(.port ⟨0⟩, result)]
    #[("Phase1Identity", identity)]
  pure { arena, leaf, root }

/-! The ID-native vertical slice.  `shared` is bound once and reused; the
    second equal `add` must observe an intern hit.  The routed reduction and
    the `99 + 99` expression are deliberately dead so the production exit can
    demonstrate reachable-only copying. -/
private def buildNativeCorpus : Except String FixtureCorpus := do
  let (arena, leaf) ← Tropical.EmitArrow.ArenaNative.assemble {}
      "Phase1Leaf" output #[] do
    let seven ← Tropical.EmitArrow.ArenaNative.lit 7
    pure ({ assigns := #[(.port ⟨0⟩, seven)] } : NativeBody)

  let (arena, identity) ← Tropical.EmitArrow.ArenaNative.assemble arena
      "Phase1Identity" output #[] do
    let zero ← Tropical.EmitArrow.ArenaNative.lit 0
    let input ← Tropical.EmitArrow.ArenaNative.inputRef ⟨0⟩
    pure ({
      inputs := #[{
        name := "x"
        type? := some (.scalar .float)
        defaultSig := some zero
      }]
      assigns := #[(.port ⟨0⟩, input)]
    } : NativeBody)

  let (arena, root) ← Tropical.EmitArrow.ArenaNative.assemble arena
      "Phase1Root" output #[("Phase1Identity", identity)] do
    let two ← Tropical.EmitArrow.ArenaNative.lit 2
    let three ← Tropical.EmitArrow.ArenaNative.lit 3
    let shared ← Tropical.EmitArrow.ArenaNative.add two three
    let equalAgain ← Tropical.EmitArrow.ArenaNative.add two three
    unless shared == equalAgain do
      throw "ArenaNative phase-1 fixture: equal construction missed eintern"

    let loop ← Tropical.EmitArrow.ArenaNative.loopIdx 7
    let table ← Tropical.EmitArrow.ArenaNative.arr #[two, three]
    let indexed ← Tropical.EmitArrow.ArenaNative.index table loop
    let contribution ← Tropical.EmitArrow.ArenaNative.mul indexed shared
    let bank ← Tropical.EmitArrow.ArenaNative.bankSum
      2 #[table] contribution none 7

    -- Constructor/order coverage for the routed sibling.  It is dead at the
    -- root on purpose; the copied reachable graph must omit it.
    let routed ← Tropical.EmitArrow.ArenaNative.routedSum
      2 2 #[some 1, none] #[table] #[shared, indexed] none 7
    let builder ← get
    unless builder.exprs.deref routed == some
        (.routedSum 2 2 #[some 1, none] #[table]
          #[shared, indexed] none 7) do
      throw "ArenaNative phase-1 fixture: routedSum changed authored order"

    let ninetyNine ← Tropical.EmitArrow.ArenaNative.lit 99
    let _dead ← Tropical.EmitArrow.ArenaNative.add ninetyNine ninetyNine

    let instanceIdx ← Tropical.EmitArrow.ArenaNative.inst
      "identity" "Phase1Identity" #[{ port := ⟨0⟩, value := shared }]
    let nested ← Tropical.EmitArrow.ArenaNative.nestedOut instanceIdx ⟨0⟩
    let result ← Tropical.EmitArrow.ArenaNative.add nested bank
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
  let failed := Tropical.EmitArrow.ArenaNative.assemble original
    "Phase1Refusal" output #[] do
      let _ ← Tropical.EmitArrow.ArenaNative.lit 1
      throw "phase-1 refusal"
  match failed with
  | .error message =>
      message == "phase-1 refusal" &&
        original.programs.isEmpty && original.exprs.nodes.isEmpty
  | .ok _ => false

/-- Pure acceptance evidence for the phase-1 vertical slice.  Exact node
    counts make explicit sharing and reachability observable; exact wire
    equality is stronger than a fixture-local golden hash. -/
def phase1Evidence : Except String Phase1Evidence := do
  let legacy ← buildLegacyCorpus
  let native ← buildNativeCorpus

  unless native.arena.programs.size == 3 do
    throw s!"ArenaNative phase-1 fixture: expected 3 programs, got {native.arena.programs.size}"
  unless native.arena.exprs.nodes.size == 16 do
    throw s!"ArenaNative phase-1 fixture: expected 16 authored unique nodes, got {native.arena.exprs.nodes.size}"
  unless native.arena.exprs.wf do
    throw "ArenaNative phase-1 fixture: authored expression arena is not child-descending"
  unless failureIsAtomic do
    throw "ArenaNative phase-1 fixture: failed build changed publication behavior"

  let (legacyLeafExprs, legacyLeafWire) ← resolvedWire legacy legacy.leaf
  let (nativeLeafExprs, nativeLeafWire) ← resolvedWire native native.leaf
  unless nativeLeafExprs.nodes.size == 1 do
    throw s!"ArenaNative phase-1 fixture: expected 1 reachable leaf node, got {nativeLeafExprs.nodes.size}"
  unless nativeLeafExprs.wf do
    throw "ArenaNative phase-1 fixture: reachable leaf arena is not child-descending"
  unless nativeLeafExprs.nodes == legacyLeafExprs.nodes &&
      nativeLeafWire == legacyLeafWire do
    throw "ArenaNative phase-1 fixture: leaf program differs from recursive baseline"

  let (legacyRootExprs, legacyRootWire) ← resolvedWire legacy legacy.root
  let (nativeRootExprs, nativeRootWire) ← resolvedWire native native.root
  unless nativeRootExprs.nodes.size == 9 do
    throw s!"ArenaNative phase-1 fixture: expected 9 reachable root nodes, got {nativeRootExprs.nodes.size}"
  unless nativeRootExprs.nodes.size < native.arena.exprs.nodes.size do
    throw "ArenaNative phase-1 fixture: dead authored nodes survived reachability GC"
  unless nativeRootExprs.wf do
    throw "ArenaNative phase-1 fixture: reachable root arena is not child-descending"
  unless nativeRootExprs.nodes == legacyRootExprs.nodes &&
      nativeRootWire == legacyRootWire do
    throw "ArenaNative phase-1 fixture: root program differs from recursive baseline"

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

private def buildLegacyArrowCarrier (arena : Arena)
    (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← Tropical.EmitArrow.buildRegistry arena resolved #["FixedSinOsc"]
  let base := Tropical.EmitArrow.clockLit
  let delta := Tropical.EmitArrow.litI 17
  let voice := fun (pitch : Int) => ({
    programName := "FixedSinOsc"
    wire := fun clock => #[
      { port := ⟨0⟩, value := Tropical.EmitArrow.lit pitch },
      { port := ⟨1⟩, value := clock },
      { port := ⟨2⟩, value := Tropical.EmitArrow.lit 0 }]
    phaseAnchor := some (⟨2⟩, fun shift =>
      Tropical.EmitArrow.mul (Tropical.EmitArrow.toFloatE shift)
        (Tropical.EmitArrow.lit 1 9))
  } : Tropical.EmitArrow.Voice)
  let modulator : Tropical.EmitArrow.ArrowTerm :=
    .gen (voice 11) "mod" base
  let carrier : Tropical.EmitArrow.ArrowTerm :=
    .gen (voice 220) "carrier" base
  let signalWarp := fun clock signal =>
    Tropical.EmitArrow.sub clock (Tropical.EmitArrow.toIntE
      (Tropical.EmitArrow.mul signal (Tropical.EmitArrow.lit 8)))
  let warped : Tropical.EmitArrow.ArrowTerm := .warp
    (fun clock => Tropical.EmitArrow.add clock delta)
    (.swarp signalWarp modulator carrier)
  let sibling : Tropical.EmitArrow.ArrowTerm := .warp
    (fun clock => Tropical.EmitArrow.sub clock delta)
    (.gen (voice 330) "sibling" base)
  let combine := fun (signals : Array Tropical.EmitArrow.Sig) =>
    Tropical.EmitArrow.add
      (Tropical.EmitArrow.mul (Tropical.EmitArrow.lit 2) signals[0]!)
      (Tropical.EmitArrow.mul (Tropical.EmitArrow.lit 3) signals[1]!)
  let term : Tropical.EmitArrow.ArrowTerm := .arrN combine #[warped, sibling]
  let (outputSignal, builder) :=
    Tropical.EmitArrow.emitTerm (Tropical.EmitArrow.normalize term) {}
  unless builder.decls.map (·.name) == #["mod0", "carrier1", "sibling2"] do
    throw "ArenaNative phase-2 fixture: legacy instance order changed"
  pure (Tropical.EmitArrow.assemble arena "Phase2ArrowCarrier" #[] output
    builder.decls #[(.port ⟨0⟩, outputSignal)] registry)

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

private def buildLegacyNumerics : Except String (Arena × ProgramIdx) :=
  let phaseQ : Tropical.EmitArrow.Sig := .inputRef ⟨0⟩
  let freq : Tropical.EmitArrow.Sig := .inputRef ⟨1⟩
  let offset : Tropical.EmitArrow.Sig := .inputRef ⟨2⟩
  let clk : Tropical.EmitArrow.Sig := .inputRef ⟨3⟩
  let x : Tropical.EmitArrow.Sig := .inputRef ⟨4⟩
  let y : Tropical.EmitArrow.Sig := .inputRef ⟨5⟩
  let fixedSin := Tropical.EmitArrow.fixedSinCycSig phaseQ
  let inputs : Array Tropical.EmitArrow.AInputDecl := #[
    { name := "phase", type? := some (.scalar .int),
      defaultSig := some (Tropical.EmitArrow.lit 0) },
    { name := "freq", type? := some (.scalar .float),
      defaultSig := some (Tropical.EmitArrow.lit 220) },
    { name := "offset", type? := some (.scalar .float),
      defaultSig := some (Tropical.EmitArrow.lit 0) },
    { name := "clk", type? := some (.scalar .int),
      defaultSig := some (Tropical.EmitArrow.lit 0) },
    { name := "x", type? := some (.scalar .float),
      defaultSig := some (Tropical.EmitArrow.lit 1) },
    { name := "y", type? := some (.scalar .float),
      defaultSig := some (Tropical.EmitArrow.lit 5 1) }]
  let assigns : Array (OutputTarget × Tropical.EmitArrow.Sig) := #[
    (.port ⟨0⟩, fixedSin),
    (.port ⟨1⟩, Tropical.EmitArrow.fixedCosCycSig phaseQ),
    (.port ⟨2⟩, Tropical.EmitArrow.fixedOutQ 30 fixedSin),
    (.port ⟨3⟩, Tropical.EmitArrow.phasorPhaseSig freq offset clk),
    (.port ⟨4⟩, Tropical.EmitArrow.sinSig x),
    (.port ⟨5⟩, Tropical.EmitArrow.expSig x),
    (.port ⟨6⟩, Tropical.EmitArrow.logSig x),
    (.port ⟨7⟩, Tropical.EmitArrow.atan2E y x),
    (.port ⟨8⟩, Tropical.EmitArrow.cosSig x),
    (.port ⟨9⟩, Tropical.EmitArrow.twoPiE),
    (.port ⟨10⟩, Tropical.EmitArrow.halfPiE)]
  .ok (Tropical.EmitArrow.assemble {} "Phase2Numerics" inputs
    numericOutputs #[] assigns #[])

private def buildNativeNumerics : Except String (Arena × ProgramIdx) :=
  Tropical.EmitArrow.ArenaNative.assemble {} "Phase2Numerics"
      numericOutputs #[] do
    let phaseQ ← Tropical.EmitArrow.ArenaNative.inputRef ⟨0⟩
    let freq ← Tropical.EmitArrow.ArenaNative.inputRef ⟨1⟩
    let offset ← Tropical.EmitArrow.ArenaNative.inputRef ⟨2⟩
    let clk ← Tropical.EmitArrow.ArenaNative.inputRef ⟨3⟩
    let x ← Tropical.EmitArrow.ArenaNative.inputRef ⟨4⟩
    let y ← Tropical.EmitArrow.ArenaNative.inputRef ⟨5⟩
    let fixedSin ← Tropical.EmitArrow.ArenaNative.fixedSinCycSig phaseQ
    let fixedCos ← Tropical.EmitArrow.ArenaNative.fixedCosCycSig phaseQ
    let fixedOut ← Tropical.EmitArrow.ArenaNative.fixedOutQ 30 fixedSin
    let phasor ← Tropical.EmitArrow.ArenaNative.phasorPhaseSig freq offset clk
    let sine ← Tropical.EmitArrow.ArenaNative.sinSig x
    let exponential ← Tropical.EmitArrow.ArenaNative.expSig x
    let logarithm ← Tropical.EmitArrow.ArenaNative.logSig x
    let angle ← Tropical.EmitArrow.ArenaNative.atan2E y x
    let cosine ← Tropical.EmitArrow.ArenaNative.cosSig x
    let twoPi ← Tropical.EmitArrow.ArenaNative.twoPiE
    let halfPi ← Tropical.EmitArrow.ArenaNative.halfPiE
    let zero ← Tropical.EmitArrow.ArenaNative.lit 0
    let twoTwenty ← Tropical.EmitArrow.ArenaNative.lit 220
    let one ← Tropical.EmitArrow.ArenaNative.lit 1
    let half ← Tropical.EmitArrow.ArenaNative.lit 5 1
    let inputs : Array Tropical.EmitArrow.ArenaNative.AInputDecl := #[
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
  let registry ← Tropical.EmitArrow.ArenaNative.buildRegistry arena resolved
    #["FixedSinOsc"]
  Tropical.EmitArrow.ArenaNative.assemble arena "Phase2ArrowCarrier" output
      registry do
    let base ← Tropical.EmitArrow.ArenaNative.clockLit
    let delta ← Tropical.EmitArrow.ArenaNative.litI 17
    -- Independently reconstructing an equal node must hit the same interner,
    -- while the bound IDs below are the values actually reused by the term.
    let reuseA ← Tropical.EmitArrow.ArenaNative.add base delta
    let reuseB ← Tropical.EmitArrow.ArenaNative.add base delta
    unless reuseA == reuseB do
      throw "ArenaNative phase-2 fixture: repeated scalar construction missed eintern"
    let voice := fun (pitch : Int) => ({
      programName := "FixedSinOsc"
      wire := fun clock => do
        let pitch ← Tropical.EmitArrow.ArenaNative.lit pitch
        let zero ← Tropical.EmitArrow.ArenaNative.lit 0
        pure #[
          { port := ⟨0⟩, value := pitch },
          { port := ⟨1⟩, value := clock },
          { port := ⟨2⟩, value := zero }]
      phaseAnchor := some (⟨2⟩, fun shift => do
        let shiftFloat ← Tropical.EmitArrow.ArenaNative.toFloatE shift
        let scale ← Tropical.EmitArrow.ArenaNative.lit 1 9
        Tropical.EmitArrow.ArenaNative.mul shiftFloat scale)
    } : Tropical.EmitArrow.ArenaNative.Voice)
    let modulator : Tropical.EmitArrow.ArenaNative.ArrowTerm :=
      .gen (voice 11) "mod" base
    let carrier : Tropical.EmitArrow.ArenaNative.ArrowTerm :=
      .gen (voice 220) "carrier" base
    let signalWarp := fun clock signal => do
      let eight ← Tropical.EmitArrow.ArenaNative.lit 8
      let scaled ← Tropical.EmitArrow.ArenaNative.mul signal eight
      let delta ← Tropical.EmitArrow.ArenaNative.toIntE scaled
      Tropical.EmitArrow.ArenaNative.sub clock delta
    let warped : Tropical.EmitArrow.ArenaNative.ArrowTerm := .warp
      (fun clock => Tropical.EmitArrow.ArenaNative.add clock delta)
      (.swarp signalWarp modulator carrier)
    let sibling : Tropical.EmitArrow.ArenaNative.ArrowTerm := .warp
      (fun clock => Tropical.EmitArrow.ArenaNative.sub clock delta)
      (.gen (voice 330) "sibling" base)
    let combine := fun (signals : Array Tropical.EmitArrow.ArenaNative.Sig) => do
      let two ← Tropical.EmitArrow.ArenaNative.lit 2
      let left ← Tropical.EmitArrow.ArenaNative.mul two signals[0]!
      let three ← Tropical.EmitArrow.ArenaNative.lit 3
      let right ← Tropical.EmitArrow.ArenaNative.mul three signals[1]!
      Tropical.EmitArrow.ArenaNative.add left right
    let term : Tropical.EmitArrow.ArenaNative.ArrowTerm :=
      .arrN combine #[warped, sibling]
    let outputSignal ← Tropical.EmitArrow.ArenaNative.emitTerm
      (Tropical.EmitArrow.ArenaNative.normalize term)
    let builder ← get
    unless builder.decls.map (·.name) == #["mod0", "carrier1", "sibling2"] do
      throw "ArenaNative phase-2 fixture: instance effects are not left-to-right"
    pure ({ assigns := #[(.port ⟨0⟩, outputSignal)] } : NativeBody)

def phase2Evidence : Except String Phase2Evidence := do
  let (legacyNumerics, legacyNumericProgram) ← buildLegacyNumerics
  let (nativeNumerics, nativeNumericProgram) ← buildNativeNumerics
  let legacyNumericCorpus : FixtureCorpus := {
    arena := legacyNumerics, leaf := legacyNumericProgram,
    root := legacyNumericProgram }
  let nativeNumericCorpus : FixtureCorpus := {
    arena := nativeNumerics, leaf := nativeNumericProgram,
    root := nativeNumericProgram }
  let (legacyNumericExprs, legacyNumericWire) ←
    resolvedWire legacyNumericCorpus legacyNumericProgram
  let (nativeNumericExprs, nativeNumericWire) ←
    resolvedWire nativeNumericCorpus nativeNumericProgram
  unless nativeNumericExprs.nodes == legacyNumericExprs.nodes &&
      nativeNumericWire == legacyNumericWire do
    let mut firstDifference := "none"
    for index in [0:max nativeNumericExprs.nodes.size legacyNumericExprs.nodes.size] do
      if firstDifference == "none" &&
          nativeNumericExprs.nodes[index]? != legacyNumericExprs.nodes[index]? then
        firstDifference := s!"{index}: native={repr nativeNumericExprs.nodes[index]?} legacy={repr legacyNumericExprs.nodes[index]?}"
    throw s!"ArenaNative phase-2 fixture: numeric helpers differ from recursive baseline (nodes {nativeNumericExprs.nodes.size}/{legacyNumericExprs.nodes.size}, wire {nativeNumericWire.length}/{legacyNumericWire.length}, first {firstDifference})"
  let (legacyStdlib, legacyResolved) ← Tropical.EmitArrow.buildStdlibChain
  let (nativeStdlib, nativeResolved) ← Tropical.EmitArrow.buildStdlibChain
  unless nativeStdlib.programs.size == 15 do
    throw s!"ArenaNative phase-2 fixture: expected 15 stdlib programs, got {nativeStdlib.programs.size}"
  unless nativeStdlib.exprs.wf do
    throw "ArenaNative phase-2 fixture: stdlib authored arena is not child-descending"
  let (legacyArena, legacyCarrier) ←
    buildLegacyArrowCarrier legacyStdlib legacyResolved
  let (nativeArena, nativeCarrier) ←
    buildNativeArrowCarrier nativeStdlib nativeResolved
  let some _legacyProgram := legacyArena.program? legacyCarrier
    | throw "ArenaNative phase-2 fixture: legacy carrier missing"
  let some nativeProgram := nativeArena.program? nativeCarrier
    | throw "ArenaNative phase-2 fixture: native carrier missing"
  unless nativeProgram.decls.size == 3 do
    throw s!"ArenaNative phase-2 fixture: expected 3 carrier instances, got {nativeProgram.decls.size}"
  let legacyCorpus : FixtureCorpus :=
    { arena := legacyArena, leaf := legacyCarrier, root := legacyCarrier }
  let nativeCorpus : FixtureCorpus :=
    { arena := nativeArena, leaf := nativeCarrier, root := nativeCarrier }
  let (legacyExprs, legacyWire) ← resolvedWire legacyCorpus legacyCarrier
  let (nativeExprs, nativeWire) ← resolvedWire nativeCorpus nativeCarrier
  unless nativeExprs.wf do
    throw "ArenaNative phase-2 fixture: reachable arrow arena is not child-descending"
  unless nativeExprs.nodes == legacyExprs.nodes && nativeWire == legacyWire do
    throw "ArenaNative phase-2 fixture: monadic ArrowTerm differs from recursive baseline"
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

/-! Phase 3 retains a room → factored all-pass product → room spine.  This is
    deliberately small enough for an exact recursive/native comparison while
    still crossing the specialized routed terminal used by production Phaser. -/
private def buildLegacyModalCarrier : Except String (Arena × ProgramIdx) := do
  let mode := fun (sigma omega cre cim : Int) =>
    ({ sigma := Tropical.EmitArrow.lit sigma 1
       omega := Tropical.EmitArrow.lit omega
       cre := Tropical.EmitArrow.lit cre 1
       cim := Tropical.EmitArrow.lit cim 1 } : Tropical.EmitArrow.ModalMode)
  let source := #[mode 4 220 8 1, mode 6 (-330) 5 (-2)]
  let room1 := #[mode 8 37 3 1, mode 11 (-51) 2 (-1)]
  let room2 := #[mode 13 73 2 1, mode 17 (-91) 1 (-1)]
  let zero := Tropical.EmitArrow.lit 0
  let control := fun value => Tropical.EmitArrow.ModalControlRef.constant value
  let (phaser, topology) := Tropical.EmitArrow.modalPhaserTopology
    "fixture" "room1" (control (Tropical.EmitArrow.lit 700))
    (control (Tropical.EmitArrow.lit 15 1))
    (control (Tropical.EmitArrow.lit 2 1))
    (control (Tropical.EmitArrow.lit 5 1)) #[1.0]
  let graph : Tropical.EmitArrow.PatchGraph := {
    nodes := #[
      { id := "source", node := .modalSource source zero Tropical.EmitArrow.clockLit none },
      { id := "room1", node := .modalReverb "source" room1 (some { dir := zero }) }]
      ++ topology ++ #[
      { id := "fixture", node := phaser },
      { id := "room2", node := .modalReverb "fixture" room2 (some { dir := zero }) }]
    output := "room2" }
  let term ← Tropical.EmitArrow.lowerGraph graph
  let (signal, builder) := Tropical.EmitArrow.emitTerm
    (Tropical.EmitArrow.normalize term) {}
  pure (Tropical.EmitArrow.assemble {} "Phase3ModalCarrier" #[] output
    builder.decls #[(.port ⟨0⟩, signal)] #[])

private def buildNativeModalCarrier : Except String (Arena × ProgramIdx) :=
  Tropical.EmitArrow.ArenaNative.assemble {} "Phase3ModalCarrier" output #[] do
    let mode := fun (sigma omega cre cim : Int) => do
      let sigma ← Tropical.EmitArrow.ArenaNative.lit sigma 1
      let omega ← Tropical.EmitArrow.ArenaNative.lit omega
      let cre ← Tropical.EmitArrow.ArenaNative.lit cre 1
      let cim ← Tropical.EmitArrow.ArenaNative.lit cim 1
      pure ({ sigma, omega, cre, cim } : Tropical.EmitArrow.ArenaNative.ModalMode)
    let source ← #[ (4, 220, 8, 1), (6, -330, 5, -2) ].mapM
      fun (sigma, omega, cre, cim) => mode sigma omega cre cim
    let room1 ← #[ (8, 37, 3, 1), (11, -51, 2, -1) ].mapM
      fun (sigma, omega, cre, cim) => mode sigma omega cre cim
    let room2 ← #[ (13, 73, 2, 1), (17, -91, 1, -1) ].mapM
      fun (sigma, omega, cre, cim) => mode sigma omega cre cim
    let zero ← Tropical.EmitArrow.ArenaNative.lit 0
    let center ← Tropical.EmitArrow.ArenaNative.lit 700
    let sweep ← Tropical.EmitArrow.ArenaNative.lit 15 1
    let rate ← Tropical.EmitArrow.ArenaNative.lit 2 1
    let mix ← Tropical.EmitArrow.ArenaNative.lit 5 1
    let clock ← Tropical.EmitArrow.ArenaNative.clockLit
    let control := Tropical.EmitArrow.ArenaNative.ModalControlRef.constant
    let (phaser, topology) := Tropical.EmitArrow.ArenaNative.modalPhaserTopology
      "fixture" "room1" (control center) (control sweep) (control rate)
      (control mix) #[1.0]
    let graph : Tropical.EmitArrow.ArenaNative.PatchGraph := {
      nodes := #[
        { id := "source", node := .modalSource source zero clock none },
        { id := "room1", node := .modalReverb "source" room1 (some { dir := zero }) }]
        ++ topology ++ #[
        { id := "fixture", node := phaser },
        { id := "room2", node := .modalReverb "fixture" room2 (some { dir := zero }) }]
      output := "room2" }
    let term ← Tropical.EmitArrow.ArenaNative.lowerGraph graph
    let signal ← Tropical.EmitArrow.ArenaNative.emitTerm
      (Tropical.EmitArrow.ArenaNative.normalize term)
    pure ({ assigns := #[(.port ⟨0⟩, signal)] } : NativeBody)

private def routedReductionCount (exprs : ExprArena) : Nat :=
  exprs.nodes.foldl (fun total node =>
    if node matches .routedSum .. then total + 1 else total) 0

def phase3Evidence : Except String Phase3Evidence := do
  let (legacyArena, legacyProgram) ← buildLegacyModalCarrier
  let (nativeArena, nativeProgram) ← buildNativeModalCarrier
  unless nativeArena.exprs.wf do
    throw "ArenaNative phase-3 fixture: authored modal arena is not child-descending"
  let legacyCorpus : FixtureCorpus := {
    arena := legacyArena, leaf := legacyProgram, root := legacyProgram }
  let nativeCorpus : FixtureCorpus := {
    arena := nativeArena, leaf := nativeProgram, root := nativeProgram }
  let (legacyExprs, legacyWire) ← resolvedWire legacyCorpus legacyProgram
  let (nativeExprs, nativeWire) ← resolvedWire nativeCorpus nativeProgram
  unless nativeExprs.wf do
    throw "ArenaNative phase-3 fixture: reachable modal arena is not child-descending"
  unless nativeExprs.nodes == legacyExprs.nodes && nativeWire == legacyWire do
    let mut firstDifference := "none"
    for index in [0:max nativeExprs.nodes.size legacyExprs.nodes.size] do
      if firstDifference == "none" &&
          nativeExprs.nodes[index]? != legacyExprs.nodes[index]? then
        firstDifference := s!"{index}: native={repr nativeExprs.nodes[index]?} legacy={repr legacyExprs.nodes[index]?}"
    throw s!"ArenaNative phase-3 fixture: factored Phaser modal carrier differs from recursive baseline (nodes {nativeExprs.nodes.size}/{legacyExprs.nodes.size}, wire {nativeWire.length}/{legacyWire.length}, first {firstDifference})"
  let routed := routedReductionCount nativeExprs
  unless routed > 0 do
    throw "ArenaNative phase-3 fixture: specialized modal carrier emitted no routed reduction"
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

end Tropical.Testing.ArenaNative
