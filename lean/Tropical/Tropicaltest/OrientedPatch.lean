import Tropical.EmitArrow.Patch
import Tropical.Tropicaltest.Stress

/-!
# Production PatchGraph gates for room-local direction

These fixtures enter through hand-built `PatchGraph`s and the real
`lowerGraph → normalize → emitTerm` seam.  The numeric references are
independent Float partial fractions for one real mode per source/room.
-/

namespace Tropical.Tropicaltest.OrientedPatch

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)
open Tropical.EmitArrow

private def sampleRate : Float := 44100.0
private def anchorNat : Nat := 512
private def anchorSig : BuildM Sig := lit (Int.ofNat anchorNat)
private def frameCount : Nat := 1024

private def realMode (sigma : Int) : BuildM ModalMode := do
  let sigma ← lit sigma
  let omega ← lit 0
  let cre ← lit 1
  let cim ← lit 0
  pure { sigma, omega, cre, cim }

private def sourceModes : BuildM (Array ModalMode) := do pure #[← realMode 2]
private def roomOne : BuildM (Array ModalMode) := do pure #[← realMode 5]
private def roomTwo : BuildM (Array ModalMode) := do pure #[← realMode 9]

private def direction (past : Bool) : BuildM (Option ModalDir) := do
  let dir ← if past then lit 1 else lit 0
  pure (some { dir })

private def mirrorAtAnchor (clock : Clock) : BuildM Clock := do
  let twiceAnchorQ : Int := Int.ofNat (2 * anchorNat) * 4294967296
  sub (← litI twiceAnchorQ) clock

private def oneRoomLocalReverse : BuildM PatchGraph := do
  let source ← sourceModes
  let anchor ← anchorSig
  let clock ← clockLit
  let room ← roomOne
  let directionValue ← direction true
  pure (PatchGraph.mk #[
      { id := "source", node := .modalSource source anchor clock none },
      { id := "room", node := (.modalReverb "source" room directionValue) }]
    "room")

private def oneRoomOutputReverse : BuildM PatchGraph := do
  let source ← sourceModes
  let anchor ← anchorSig
  let clock ← clockLit
  let room ← roomOne
  let directionValue ← direction false
  pure (PatchGraph.mk #[
      { id := "source", node := .modalSource source anchor clock none },
      { id := "room", node := (.modalReverb "source" room directionValue) },
      { id := "output-reverse", node := (.warpFx "room" mirrorAtAnchor) }]
    "output-reverse")

private def twoRoomGraph (roomOnePast roomTwoPast : Bool) : BuildM PatchGraph := do
  let source ← sourceModes
  let anchor ← anchorSig
  let clock ← clockLit
  let roomOne ← roomOne
  let roomTwo ← roomTwo
  let directionOneValue ← direction roomOnePast
  let directionTwoValue ← direction roomTwoPast
  pure (PatchGraph.mk #[
      { id := "source", node := .modalSource source anchor clock none },
      { id := "room-one",
        node := (.modalReverb "source" roomOne directionOneValue) },
      { id := "room-two",
        node := (.modalReverb "room-one" roomTwo directionTwoValue) }]
    "room-two")

private def constantControl (value : Sig) : ModalControlRef :=
  ModalControlRef.constant value

private def equalRt60Modes (rt60 : Sig) : BuildM (Array ModalMode) := do
  let sigma ← div (← lit 10) rt60
  let omega ← lit 0
  let cre ← lit 1
  let cim ← lit 0
  pure #[{ sigma, omega, cre, cim }]

/-- Two separately authored room nodes whose frozen RT60 controls happen to be
equal.  Each builds the same physical pole `sigma = 10/rt60 = 5`; the second
convolution must take the repeated-pole beta route instead of dividing by zero. -/
private def equalRt60Graph : BuildM PatchGraph := do
  let rt60 ← lit 2
  let zero ← lit 0
  let room := fun input => Node.modalRoom input equalRt60Modes
    (constantControl rt60) (constantControl zero)
  let source ← sourceModes
  let anchor ← anchorSig
  let clock ← clockLit
  pure (PatchGraph.mk #[
      { id := "source", node := .modalSource source anchor clock none },
      { id := "equal-room-one", node := (room "source") },
      { id := "equal-room-two", node := (room "equal-room-one") }]
    "equal-room-two")

private def repeatedRoomCrossingGraph (afterGauge : Bool) : BuildM PatchGraph := do
  let rt60 ← lit 2
  let zero ← lit 0
  let room := fun input => Node.modalRoom input equalRt60Modes
    (constantControl rt60) (constantControl zero)
  let source ← sourceModes
  let anchor ← anchorSig
  let clock ← clockLit
  let nodesPrefix : Array PatchNode := #[
    { id := "source", node := .modalSource source anchor clock none },
    { id := "equal-room-one", node := room "source" },
    { id := "equal-room-two", node := room "equal-room-one" }]
  if afterGauge then
    let gauge ← lit 1
    let nodes := nodesPrefix.push
      ({ id := "gauge", node := .modalGauge "equal-room-two" gauge } : PatchNode)
    pure (PatchGraph.mk nodes "gauge")
  else
    let nodes := nodesPrefix.push
      ({ id := "equal-room-three", node := room "equal-room-two" } : PatchNode)
    pure (PatchGraph.mk nodes "equal-room-three")

private def degreePositiveGraph : BuildM PatchGraph := do
  let source := #[({ (← realMode 2) with deg := 1 } : ModalMode)]
  let anchor ← anchorSig
  let clock ← clockLit
  let room ← roomOne
  let directionValue ← direction false
  pure (PatchGraph.mk #[
      { id := "source", node := .modalSource source anchor clock none },
      { id := "room", node := (.modalReverb "source" room directionValue) }]
    "room")

private def sourceCrossingGraph (near triple : Bool) : BuildM PatchGraph := do
  let sourceSigma ← if near then lit 50001 4 else do add (← lit 2) (← lit 3)
  let source := #[({ (← realMode 5) with sigma := sourceSigma } : ModalMode)]
  let repeated ← realMode 5
  let tripleSigma ← sub (← lit 6) (← lit 1)
  let ordinaryRoom ← roomTwo
  let second := if triple then
      #[({ repeated with sigma := tripleSigma } : ModalMode)]
    else ordinaryRoom
  let anchor ← anchorSig
  let clock ← clockLit
  let directionValue ← direction false
  pure (PatchGraph.mk #[
      { id := "source", node := .modalSource source anchor clock none },
      { id := "crossing-room-one",
        node := (.modalReverb "source" #[repeated] directionValue) },
      { id := "crossing-room-two",
        node := (.modalReverb "crossing-room-one" second directionValue) }]
    "crossing-room-two")

/-- The same exact source/room crossing with the resonant room authored second.
    The terminal convolution is commutative, but this ordering exercises the
    room-2 confluence bookkeeping independently of the room-1 path. -/
private def sourceSecondCrossingGraph : BuildM PatchGraph := do
  let source := #[← realMode 5]
  let anchor ← anchorSig
  let clock ← clockLit
  let room ← roomTwo
  let directionValue ← direction false
  pure (PatchGraph.mk #[
      { id := "source", node := .modalSource source anchor clock none },
      { id := "ordinary-room-one",
        node := (.modalReverb "source" room directionValue) },
      { id := "crossing-room-two",
        node := (.modalReverb "ordinary-room-one" source directionValue) }]
    "crossing-room-two")

/-- One complex repeated-pole DD atom.  The non-real coefficient makes the
    carrier phase observable; installing it on each terminal arm separately
    pins both the causal reduction and its anchor-mirrored past read. -/
private def complexTerminalPair : BuildM PairedMode := do
  let pole := (← neg (← lit 5), ← lit 700)
  let coefficient := (← lit 34 2, ← lit (-13) 2)
  pure { lam := pole, nu := pole, c := coefficient }

private def complexTerminalSig (past : Bool) : BuildM Sig := do
  let zero ← lit 0
  let bank : Oriented.Bank := { atZero := (zero, zero) }
  let pair ← complexTerminalPair
  let terminal : Oriented.TerminalBank :=
    if past then { bank, pastPaired := #[pair] }
    else { bank, futurePaired := #[pair] }
  terminal.realizeSig (← clockLit) (← anchorSig)

private def graphPlan (arena : Arena) (name : String) (graph : BuildM PatchGraph) :
    Except String FlatPlan := do
  buildAndFinish (buildExprCarrier name (do
    emitTerm (normalize (← lowerGraph (← graph)))) arena)

private def renderGraph (arena : Arena) (name : String) (graph : BuildM PatchGraph) :
    IO (Except String (Array Float)) :=
  match graphPlan arena name graph with
  | .error error => pure (.error error)
  | .ok plan => renderPlanSamples plan frameCount

private def renderSignal (arena : Arena) (name : String) (signal : BuildM Sig) :
    IO (Except String (Array Float)) :=
  match buildAndFinish (buildExprCarrier name signal arena) with
  | .error error => pure (.error error)
  | .ok plan => renderPlanSamples plan frameCount

/-- A physical left-half-plane pole plus its one-sided orientation. -/
private structure Factor where
  physicalPole : Float
  past : Bool := false

private def Factor.actualPole (factor : Factor) : Float :=
  if factor.past then -factor.physicalPole else factor.physicalPole

/-- Residue of `(-1)^numberOfPast / product (s-alpha)` at one actual pole. -/
private def residueAt (factors : Array Factor) (index : Nat) (factor : Factor) :
    Float :=
  let sign := if factors.foldl (fun count item =>
      if item.past then count + 1 else count) 0 % 2 == 1 then -1.0 else 1.0
  let alpha := factor.actualPole
  let denominator := factors.zipIdx.foldl (fun product (other, otherIndex) =>
    if index == otherIndex then product
    else product * (alpha - other.actualPole)) 1.0
  sign / denominator

/-- Independent bilateral inverse-Laplace oracle.  LHP actual poles contribute
future atoms with residue `R`; RHP poles contribute past atoms with amplitude
`-R`.  The zero value is their shared mixed-orientation limit. -/
private def analyticOracle (factors : Array Factor) (relativeSeconds : Float) :
    Float :=
  if relativeSeconds > 0.0 then
    factors.zipIdx.foldl (fun total (factor, index) =>
      if factor.past then total
      else total + residueAt factors index factor *
        Float.exp (factor.actualPole * relativeSeconds)) 0.0
  else if relativeSeconds < 0.0 then
    factors.zipIdx.foldl (fun total (factor, index) =>
      if factor.past then total - residueAt factors index factor *
        Float.exp (factor.actualPole * relativeSeconds)
      else total) 0.0
  else
    factors.zipIdx.foldl (fun total (factor, index) =>
      if factor.past then total else total + residueAt factors index factor) 0.0

private def oneRoomLocalOracle (relativeSeconds : Float) : Float :=
  analyticOracle #[
    { physicalPole := -2.0 },
    { physicalPole := -5.0, past := true }] relativeSeconds

private def oneRoomOutputReverseOracle (relativeSeconds : Float) : Float :=
  analyticOracle #[
    { physicalPole := -2.0 },
    { physicalPole := -5.0 }] (-relativeSeconds)

private def twoRoomOracle (roomOnePast roomTwoPast : Bool)
    (relativeSeconds : Float) : Float :=
  analyticOracle #[
    { physicalPole := -2.0 },
    { physicalPole := -5.0, past := roomOnePast },
    { physicalPole := -9.0, past := roomTwoPast }] relativeSeconds

private def equalRt60Oracle (relativeSeconds : Float) : Float :=
  if relativeSeconds > 0.0 then
    (Float.exp (-2.0 * relativeSeconds) - Float.exp (-5.0 * relativeSeconds)) /
        9.0 -
      relativeSeconds / 3.0 * Float.exp (-5.0 * relativeSeconds)
  else 0.0

private def degreePositiveOracle (relativeSeconds : Float) : Float :=
  if relativeSeconds > 0.0 then
    (-1.0 / 9.0 + relativeSeconds / 3.0) *
        Float.exp (-2.0 * relativeSeconds) +
      1.0 / 9.0 * Float.exp (-5.0 * relativeSeconds)
  else 0.0

private def sourceCrossingOracle (relativeSeconds : Float) : Float :=
  if relativeSeconds > 0.0 then
    let e5 := Float.exp (-5.0 * relativeSeconds)
    (-e5 / 16.0 + relativeSeconds * e5 / 4.0 +
      Float.exp (-9.0 * relativeSeconds) / 16.0)
  else 0.0

private def sourceTripleCrossingOracle (relativeSeconds : Float) : Float :=
  if relativeSeconds > 0.0 then
    relativeSeconds * relativeSeconds * Float.exp (-5.0 * relativeSeconds) / 2.0
  else 0.0

private def sourceNearCrossingOracle (relativeSeconds : Float) : Float :=
  analyticOracle #[
    { physicalPole := -5.0001 },
    { physicalPole := -5.0 },
    { physicalPole := -9.0 }] relativeSeconds

private def complexPairOracle (past : Bool) (relativeSeconds : Float) : Float :=
  let active := if past then relativeSeconds < 0.0 else relativeSeconds > 0.0
  if active then
    let age := relativeSeconds.abs
    age * Float.exp (-5.0 * age) *
      (0.34 * Float.cos (700.0 * age) + 0.13 * Float.sin (700.0 * age))
  else 0.0

private def fractionalControlClockOk : Bool :=
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      let delta ← sub (← lit 2147483648) (← lit 0)
      div (← toFloatE delta) (← lit 4294967296) with
  | .error _ => false
  | .ok (arena, result) =>
    (sigConstDFrom? (sigConstTable arena) result).map Tropical.Exact.DyadicI.toFloat == some 0.5

private def maxOracleError (samples : Array Float) (oracle : Float → Float) :
    Float := Id.run do
  let mut maximum := 0.0
  for i in [0:min frameCount samples.size] do
    let relative := (i.toFloat - anchorNat.toFloat) / sampleRate
    let error := (samples[i]! - oracle relative).abs
    if error > maximum then maximum := error
  return maximum

private def maxDifference (left right : Array Float) : Float := Id.run do
  let mut maximum := 0.0
  for i in [0:min left.size right.size] do
    let difference := (left[i]! - right[i]!).abs
    if difference > maximum then maximum := difference
  return maximum

private def maxWindow (samples : Array Float) (lo hi : Nat) : Float := Id.run do
  let mut maximum := 0.0
  for i in [lo:min hi samples.size] do
    if samples[i]!.abs > maximum then maximum := samples[i]!.abs
  return maximum

private structure OneRoomResult where
  localError : Float
  outputReverseError : Float
  localVsOutputReverse : Float
  localPost : Float
  outputReversePost : Float

private structure ComplexPairResult where
  futureError : Float
  pastError : Float
  mirrorDifference : Float
  finite : Bool

private def checkOneRoom (arena : Arena) : IO (Except String OneRoomResult) := do
  match ← renderGraph arena "oriented_local_reverse" oneRoomLocalReverse,
        ← renderGraph arena "oriented_output_reverse" oneRoomOutputReverse with
  | .ok localSamples, .ok outputReverse =>
      pure (.ok {
        localError := maxOracleError localSamples oneRoomLocalOracle
        outputReverseError := maxOracleError outputReverse oneRoomOutputReverseOracle
        localVsOutputReverse := maxDifference localSamples outputReverse
        localPost := maxWindow localSamples (anchorNat + 1) frameCount
        outputReversePost := maxWindow outputReverse (anchorNat + 1) frameCount })
  | .error error, _ | _, .error error => pure (.error error)

private def checkComplexPair (arena : Arena) : IO (Except String ComplexPairResult) := do
  match ← renderSignal arena "oriented_complex_pair_future" (complexTerminalSig false),
        ← renderSignal arena "oriented_complex_pair_past" (complexTerminalSig true) with
  | .ok future, .ok past =>
      let mut mirrorDifference := 0.0
      for offset in [1:min anchorNat (frameCount - anchorNat)] do
        let difference := (future[anchorNat + offset]! - past[anchorNat - offset]!).abs
        if difference > mirrorDifference then mirrorDifference := difference
      pure (.ok {
        futureError := maxOracleError future (complexPairOracle false)
        pastError := maxOracleError past (complexPairOracle true)
        mirrorDifference
        finite := future.all (fun sample => sample.isFinite) &&
          past.all (fun sample => sample.isFinite) })
  | .error error, _ | _, .error error => pure (.error error)

private def controlConstant (constants : Array (Option Tropical.Exact.DyadicI))
    (control : ModalControlRef) : Option Float :=
  match control.fallback with
  | .konst signal =>
    (sigConstDFrom? constants signal).map Tropical.Exact.DyadicI.toFloat
  | _ => none

private def stageSignature (constants : Array (Option Tropical.Exact.DyadicI))
    (stage : ModalStage) : Option (Float × Float) := do
  let .ordinaryRoom room := stage | none
  let .fixed modes := room.kernel | none
  let mode ← modes[0]?
  let sigma ← (sigConstDFrom? constants mode.sigma).map Tropical.Exact.DyadicI.toFloat
  let dir ← controlConstant constants room.direction
  pure (sigma, dir)

private def stagesKeepOrder (roomOnePast roomTwoPast : Bool) : Bool :=
  let rankOf := fun id =>
    if id == "source" then some 0
    else if id == "room-one" then some 1
    else if id == "room-two" then some 2
    else none
  match Tropical.Testing.ArrowFixtures.freezeBuild {} do
      lowerModal (← twoRoomGraph roomOnePast roomTwoPast) rankOf "room-two" 2 with
  | .error _ => false
  | .ok (arena, forest) => match forest[0]? with
    | none => false
    | some branch =>
        let constants := sigConstTable arena
        match branch.stages.toList with
        | [first, second] =>
            stageSignature constants first == some (5.0, if roomOnePast then 1.0 else 0.0) &&
            stageSignature constants second == some (9.0, if roomTwoPast then 1.0 else 0.0)
        | _ => false

private structure TwoRoomResult where
  maximumOracleError : Float
  minimumPairDifference : Float
  authoredOrder : Bool
  equalRt60Error : Float
  equalRt60Finite : Bool
  equalRt60Peak : Float
  degreePositiveError : Float
  degreePositiveFinite : Bool
  sourceCrossingError : Float
  sourceSecondCrossingError : Float
  sourceTripleCrossingError : Float
  sourceNearCrossingError : Float
  sourceCrossingsFinite : Bool
  repeatedRoomRefused : Bool
  roomRoomGaugeRefused : Bool

private def checkTwoRooms (arena : Arena) : IO (Except String TwoRoomResult) := do
  let cases : Array (Bool × Bool × String) := #[
    (false, false, "ff"), (false, true, "fr"),
    (true, false, "rf"), (true, true, "rr")]
  let mut rendered : Array (Array Float) := #[]
  let mut maximumError := 0.0
  let mut orderOk := true
  for (roomOnePast, roomTwoPast, label) in cases do
    match ← renderGraph arena s!"oriented_two_{label}"
        (twoRoomGraph roomOnePast roomTwoPast) with
    | .error error => return .error error
    | .ok samples =>
        let error := maxOracleError samples
          (twoRoomOracle roomOnePast roomTwoPast)
        if error > maximumError then maximumError := error
        orderOk := orderOk && stagesKeepOrder roomOnePast roomTwoPast
        rendered := rendered.push samples
  let mut minimumDifference := 1.0e30
  for i in [0:rendered.size] do
    for j in [i + 1:rendered.size] do
      let difference := maxDifference rendered[i]! rendered[j]!
      if difference < minimumDifference then minimumDifference := difference
  match ← renderGraph arena "oriented_equal_rt60" equalRt60Graph with
  | .error error => pure (.error error)
  | .ok equalRt60 =>
      match ← renderGraph arena "oriented_degree_positive" degreePositiveGraph with
      | .error error => pure (.error error)
      | .ok degreePositive =>
          let crossing ← renderGraph arena "oriented_source_crossing"
            (sourceCrossingGraph false false)
          let tripleCrossing ← renderGraph arena "oriented_source_triple_crossing"
            (sourceCrossingGraph false true)
          let secondCrossing ← renderGraph arena "oriented_source_second_crossing"
            sourceSecondCrossingGraph
          let nearCrossing ← renderGraph arena "oriented_source_near_crossing"
            (sourceCrossingGraph true false)
          let (.ok crossing) := crossing
            | return .error "source crossing render"
          let (.ok tripleCrossing) := tripleCrossing
            | return .error "source triple crossing render"
          let (.ok secondCrossing) := secondCrossing
            | return .error "source second-room crossing render"
          let (.ok nearCrossing) := nearCrossing
            | return .error "source near crossing render"
          let repeatedRoomRefused := match
              Tropical.Testing.ArrowFixtures.runBuild arena do
                lowerGraph (← repeatedRoomCrossingGraph false) with
            | .error error => error == "lower: nonterminal repeated-room crossing at 'equal-room-three' refused (a later room, phaser, or gauge requires the composable divided-difference carrier)"
            | .ok _ => false
          let roomRoomGaugeRefused := match
              Tropical.Testing.ArrowFixtures.runBuild arena do
                lowerGraph (← repeatedRoomCrossingGraph true) with
            | .error error => error == "lower: nonterminal repeated-room crossing at 'gauge' refused (a later room, phaser, or gauge requires the composable divided-difference carrier)"
            | .ok _ => false
          pure (.ok {
            maximumOracleError := maximumError
            minimumPairDifference := minimumDifference
            authoredOrder := orderOk
            equalRt60Error := maxOracleError equalRt60 equalRt60Oracle
            equalRt60Finite := equalRt60.all (fun sample => sample.isFinite)
            equalRt60Peak := maxWindow equalRt60 (anchorNat + 1) frameCount
            degreePositiveError := maxOracleError degreePositive degreePositiveOracle
            degreePositiveFinite := degreePositive.all (fun sample => sample.isFinite)
            sourceCrossingError := maxOracleError crossing sourceCrossingOracle
            sourceSecondCrossingError := maxOracleError secondCrossing sourceCrossingOracle
            sourceTripleCrossingError := maxOracleError tripleCrossing
              sourceTripleCrossingOracle
            sourceNearCrossingError := maxOracleError nearCrossing sourceNearCrossingOracle
            sourceCrossingsFinite := crossing.all (fun sample => sample.isFinite) &&
              secondCrossing.all (fun sample => sample.isFinite) &&
              tripleCrossing.all (fun sample => sample.isFinite) &&
              nearCrossing.all (fun sample => sample.isFinite)
            repeatedRoomRefused
            roomRoomGaugeRefused })

/-- End-to-end production gate for local room direction. -/
def runOrientedPatch (arena : Arena) : IO Bool := do
  match ← checkOneRoom arena, ← checkTwoRooms arena, ← checkComplexPair arena with
  | .ok one, .ok two, .ok complex =>
      IO.println "room-local direction from hand-built PatchGraphs:"
      IO.println s!"        one room  local oracle {one.localError} · output-reverse oracle {one.outputReverseError} · local≠output-reverse {one.localVsOutputReverse} · post {one.localPost}/{one.outputReversePost}"
      IO.println s!"        two rooms FF/FR/RF/RR max oracle error {two.maximumOracleError} · min pair distance {two.minimumPairDifference} · authored stage order {two.authoredOrder}"
      IO.println s!"        equal RT60 independently authored: finite {two.equalRt60Finite} · repeated-pole oracle error {two.equalRt60Error} · peak {two.equalRt60Peak}"
      IO.println s!"        degree-positive terminal: finite {two.degreePositiveFinite} · oracle error {two.degreePositiveError}; guarded crossings: room-room-room {two.repeatedRoomRefused} · room-room-gauge {two.roomRoomGaugeRefused}"
      IO.println s!"        source/room confluence: finite {two.sourceCrossingsFinite} · room-1/room-2/three-pole/near oracle {two.sourceCrossingError}/{two.sourceSecondCrossingError}/{two.sourceTripleCrossingError}/{two.sourceNearCrossingError}"
      IO.println s!"        float-banked complex DD: finite {complex.finite} · future/past oracle {complex.futureError}/{complex.pastError} · mirror diff {complex.mirrorDifference}"
      IO.println s!"        terminal control clock retains a half-sample Q32.32 offset: {fractionalControlClockOk}"
      let pass := one.localError < 2.0e-6 && one.outputReverseError < 2.0e-6 &&
        one.localVsOutputReverse > 1.0e-2 && one.localPost > 1.0e-2 &&
        one.outputReversePost < 1.0e-12 &&
        two.maximumOracleError < 2.0e-6 &&
        two.minimumPairDifference > 1.0e-3 && two.authoredOrder &&
        two.equalRt60Finite && two.equalRt60Error < 2.0e-6 &&
        two.equalRt60Peak > 1.0e-6 &&
        two.degreePositiveFinite && two.degreePositiveError < 2.0e-6 &&
        two.sourceCrossingsFinite && two.sourceCrossingError < 2.0e-6 &&
        two.sourceSecondCrossingError < 2.0e-6 &&
        two.sourceTripleCrossingError < 2.0e-6 &&
        two.sourceNearCrossingError < 2.0e-6 &&
        two.repeatedRoomRefused && two.roomRoomGaugeRefused &&
        complex.finite && complex.futureError < 2.0e-6 &&
        complex.pastError < 2.0e-6 && complex.mirrorDifference == 0.0 &&
        fractionalControlClockOk
      if pass then
        passGate "modal-oriented-patch"
          "room reverse is kernel-local (not complete-output reverse); two rooms retain independent FF/FR/RF/RR controls and authored stage order; float-banked complex DD preserves future/past phase and exact mirroring; equal frozen RT60 takes a finite repeated-pole limit; general-degree terminals preserve polynomial factors; unsupported nonterminal DD crossings refuse; terminal controls retain fractional Q32.32 time"
      else
        failGate "modal-oriented-patch" "numeric or structural contract failed"
  | .error error, _, _ | _, .error error, _ | _, _, .error error =>
      failGate "modal-oriented-patch" s!"build/render: {firstLine error}"

end Tropical.Tropicaltest.OrientedPatch
