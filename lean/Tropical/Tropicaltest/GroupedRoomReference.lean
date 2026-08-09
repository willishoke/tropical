import Tropical.Tropicaltest.Modal
import Tropical.EmitArrow.Modal.GroupedRoomReference

/-!
# Source-generic grouped-room reference gate

The oracle below sums the causal and anti-causal carrier trains directly in
`Float`.  It does not use the production prefix generator, geometric-block
evaluator, or emitted expression, so agreement is a genuine differential.
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

namespace Tropical.Tropicaltest.GroupedRoomReference

open Tropical.EmitArrow

private def jn (m : Int) (e : Nat := 0) : Lean.JsonNumber :=
  { mantissa := m, exponent := e }

private instance : Inhabited RoomCarrierGroup :=
  ⟨{ id := "", period := 1, radius := jn 5 1, carrier := #[jn 0] }⟩

private instance : Inhabited ModalMode :=
  ⟨{ sigma := lit 1, omega := lit 0, cre := lit 0 }⟩

private def tinyProfile : RoomProfile := {
  id := "synthetic-two-cycle-room"
  profileVersion := 2
  evaluatorVersion := groupedRoomReferenceEvaluatorVersion
  sampleRate := 44100
  groups := #[
    { id := "early-3", period := 3, radius := jn 72 2,
      carrier := #[jn 3 1, jn (-12) 2, jn 8 2] },
    { id := "late-5", period := 5, radius := jn 61 2,
      carrier := #[jn 17 2, jn 5 2, jn (-9) 2, jn 4 2, jn 2 2] }]
  admission := {
    poles := {
      minFrequencyHz := jn 10
      maxFrequencyHz := jn 5000
      minSigma := jn 1
      maxSigma := jn 50 }
    maxBranches := 1
    maxSourceRows := 4
    maxCarrierGroups := 2
    maxPeriod := 5
    maxGeneratedScalars := 256 } }

private structure OracleMode where
  sigma : Float
  omega : Float
  cre : Float
  cim : Float
deriving Inhabited

private def OracleMode.emit (m : OracleMode) : ModalMode := {
  sigma := litF m.sigma
  omega := litF m.omega
  cre := litF m.cre
  cim := litF m.cim }

private def sourceAOracle : Array OracleMode := #[
  { sigma := 7.0, omega := 701.0, cre := 0.7, cim := 0.2 },
  { sigma := 13.0, omega := 1337.0, cre := -0.35, cim := 0.15 }]

private def sourceBOracle : Array OracleMode := #[
  { sigma := 9.0, omega := 947.0, cre := 0.42, cim := -0.18 },
  { sigma := 17.0, omega := 1703.0, cre := 0.31, cim := 0.11 }]

private def sourceA : Array ModalMode := sourceAOracle.map OracleMode.emit
private def sourceB : Array ModalMode := sourceBOracle.map OracleMode.emit

private structure OCplx where
  re : Float
  im : Float

namespace OCplx
private def zero : OCplx := ⟨0.0, 0.0⟩
private def add (a b : OCplx) : OCplx := ⟨a.re + b.re, a.im + b.im⟩
private def scale (s : Float) (a : OCplx) : OCplx := ⟨s * a.re, s * a.im⟩
end OCplx

private def poleAtSamples (m : OracleMode) (t : Float) : OCplx :=
  let sec := t / 44100.0
  let env := Float.exp (-m.sigma * sec)
  let ph := m.omega * sec
  ⟨env * Float.cos ph, env * Float.sin ph⟩

private def powNatF (x : Float) (n : Nat) : Float := Id.run do
  let mut y := 1.0
  for _ in [0:n] do y := y * x
  return y

/-- Direct time-domain carrier train.  Causal is a finite history sum; reverse
    is a 512-tap future sum whose omitted tail is below 1e-70 for this profile.
    No prefix or geometric block is shared with production. -/
private def directPair (m : OracleMode) (g : RoomCarrierGroup)
    (position u : Float) : Float := Id.run do
  let radius := g.radius.toFloat
  let mut causal := OCplx.zero
  if u >= 0.0 then
    let last := Float.floor u |>.toUInt64.toNat
    let mut radiusN := 1.0
    for n in [0:last + 1] do
      let c := g.carrier[n % g.period]!.toFloat
      causal := causal.add ((poleAtSamples m (u - n.toFloat)).scale (c * radiusN))
      radiusN := radiusN * radius
  let dFloat := Float.ceil (-u)
  let d := if dFloat > 0.0 then dFloat.toUInt64.toNat else 0
  let mut reverse := OCplx.zero
  let mut radiusN := powNatF radius d
  for k in [0:512] do
    let n := d + k
    let c := g.carrier[n % g.period]!.toFloat
    reverse := reverse.add ((poleAtSamples m (u + n.toFloat)).scale (c * radiusN))
    radiusN := radiusN * radius
  let forwardReal := m.cre * causal.re - m.cim * causal.im
  let reverseReal := m.cre * reverse.re - m.cim * reverse.im
  let p := if position < -1.0 then -1.0 else if position > 1.0 then 1.0 else position
  let forwardGain := Float.sqrt (0.5 * (1.0 + p))
  let reverseGain := Float.sqrt (0.5 * (1.0 - p))
  return forwardGain * forwardReal + reverseGain * reverseReal

private def directRoom (profile : RoomProfile) (modes : Array OracleMode)
    (position u : Float) : Float :=
  modes.foldl (fun total m =>
    profile.groups.foldl (fun acc g => acc + directPair m g position u) total) 0.0

private def sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat

private def directSamples (profile : RoomProfile) (modes : Array OracleMode)
    (position : Float) (count : Nat) (uAt : Nat → Float) : Array Float :=
  (Array.range count).map fun i => sinkGain * directRoom profile modes position (uAt i)

private def relL2 (got ref : Array Float) : Float := Id.run do
  let mut num := 0.0
  let mut den := 0.0
  for i in [0:min got.size ref.size] do
    let d := got[i]! - ref[i]!
    num := num + d * d
    den := den + ref[i]! * ref[i]!
  return Float.sqrt (num / (den + 1e-300))

private def maxAbsDiff (a b : Array Float) : Float := Id.run do
  let mut mx := 0.0
  for i in [0:min a.size b.size] do
    let d := (a[i]! - b[i]!).abs
    if d > mx then mx := d
  return mx

private def buildRoomPlan (arena : Arena) (name : String)
    (s : RoomReferenceSpecialization) (clk anchor position : Sig) :
    Except String Tropical.Plan.FlatPlan :=
  buildAndFinish (.ok (buildExprCarrier name
    (groupedRoomReferenceSig s clk anchor position) arena))

private def renderRoom (arena : Arena) (name : String)
    (s : RoomReferenceSpecialization) (clk anchor position : Sig) (count : Nat) :
    IO (Except String (Array Float)) :=
  match buildRoomPlan arena name s clk anchor position with
  | .error e => pure (.error s!"build {name}: {firstLine e}")
  | .ok p => do
    match ← renderPlanSamples p count with
    | .error e => pure (.error s!"render {name}: {firstLine e}")
    | .ok got => pure (.ok got)

private def renderAgainstOracle (arena : Arena) (name : String)
    (spec : RoomReferenceSpecialization) (modes : Array OracleMode)
    (position : Float) (clk anchor : Sig) (count : Nat) (uAt : Nat → Float) :
    IO (Except String (Array Float × Float)) := do
  match buildRoomPlan arena name spec clk anchor (litF position) with
  | .error e => pure (.error s!"build {name}: {firstLine e}")
  | .ok p =>
    match ← renderPlanSamples p count with
    | .error e => pure (.error s!"render {name}: {firstLine e}")
    | .ok got =>
      let ref := directSamples tinyProfile modes position count uAt
      pure (.ok (got, relL2 got ref))

private def exclusionHas (kind : RoomExclusionKind) (row? group? : Option Nat)
    (r : Except RoomRefusal RoomReferenceSpecialization) : Bool :=
  match r with
  | .ok _ => false
  | .error refusal => refusal.exclusions.any fun x =>
      x.kind == kind && (row?.isNone || x.row? == row?) &&
        (group?.isNone || x.group? == group?)

private def prefixEqual (a b : RoomReferenceSpecialization) : Bool :=
  a.forwardPrefix == b.forwardPrefix && a.reversePrefix == b.reversePrefix

private def specializationOk
    (r : Except RoomRefusal RoomReferenceSpecialization) : Bool :=
  match r with
  | .ok _ => true
  | .error _ => false

set_option maxRecDepth 2048 in
def runGroupedRoomReference (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let profile := tinyProfile
  match specializeGroupedRoomReference profile 44100 sourceA,
        specializeGroupedRoomReference profile 44100 sourceB with
  | .error e, _ => failGate "grouped-room-reference" s!"source A admission: {e.summary}"
  | _, .error e => failGate "grouped-room-reference" s!"source B admission: {e.summary}"
  | .ok specA, .ok specB =>
    let ampOnlyModes : Array ModalMode := #[
      { sourceA[0]! with cre := lit 2 }, sourceA[1]!]
    let frequencyModes : Array ModalMode := #[
      { sourceA[0]! with omega := lit 801 }, sourceA[1]!]
    let dampingModes : Array ModalMode := #[
      { sourceA[0]! with sigma := lit 8 }, sourceA[1]!]
    let phaseModes : Array ModalMode := #[
      { sourceA[0]! with cre := lit (-2) 1, cim := lit 7 1 }, sourceA[1]!]
    let zeroResidueModes := sourceA.map fun m => { m with cre := lit 0, cim := lit 0 }
    let emptyResult := specializeGroupedRoomReference profile 44100 #[]
    let ampResult := specializeGroupedRoomReference profile 44100 ampOnlyModes
    match emptyResult, ampResult with
    | .error e, _ => failGate "grouped-room-reference" s!"empty source admission: {e.summary}"
    | _, .error e => failGate "grouped-room-reference" s!"amplitude-only admission: {e.summary}"
    | .ok emptySpec, .ok ampSpec =>
      let repeated := specializeGroupedRoomReference profile 44100 sourceA
      let repeatedPrefixes := match repeated with
        | .ok s => prefixEqual specA s
        | .error _ => false
      let structural := profile == tinyProfile && profile.groups.size == 2 &&
        specA.profileId == profile.id && specA.generatedScalarCount == 64 &&
        specA.generatedScalarCount ==
          2 * (specA.forwardPrefix.size + specA.reversePrefix.size) &&
        prefixEqual specA ampSpec && !prefixEqual specA specB && repeatedPrefixes

      let frequencyResult := specializeGroupedRoomReference profile 44100 frequencyModes
      let dampingResult := specializeGroupedRoomReference profile 44100 dampingModes
      let phaseResult := specializeGroupedRoomReference profile 44100 phaseModes
      let zeroResidueResult := specializeGroupedRoomReference profile 44100 zeroResidueModes

      let badPeriodGroup := { profile.groups[0]! with period := 4 }
      let badPeriod := { profile with groups := #[badPeriodGroup, profile.groups[1]!] }
      let badRadiusGroup := { profile.groups[0]! with radius := jn 1 }
      let badRadius := { profile with groups := #[badRadiusGroup, profile.groups[1]!] }
      let duplicateIdGroup := { profile.groups[1]! with id := profile.groups[0]!.id }
      let duplicateIds := { profile with groups := #[profile.groups[0]!, duplicateIdGroup] }
      let livePole : Array ModalMode := #[
        { sourceA[0]! with sigma := .paramRef ⟨0⟩ }, sourceA[1]!]
      let nonfinitePole : Array ModalMode := #[
        { sourceA[0]! with sigma := div (lit 1) (lit 0) }, sourceA[1]!]
      let degreeOne : Array ModalMode := #[{ sourceA[0]! with deg := 1 }, sourceA[1]!]
      let outOfDomain : Array ModalMode := #[
        { sourceA[0]! with sigma := lit 100 }, sourceA[1]!]
      let boundaryModes : Array ModalMode := #[
        ModalMode.hz (lit 10) (lit 1) (lit 1),
        ModalMode.hz (lit 5000) (lit 50) (lit 1)]
      let belowFrequency : Array ModalMode := #[ModalMode.hz (lit 9) (lit 1) (lit 1)]
      let aboveFrequency : Array ModalMode := #[ModalMode.hz (lit 5001) (lit 50) (lit 1)]
      let rowCap := { profile with admission := { profile.admission with maxSourceRows := 1 } }
      let scalarCap := { profile with admission := { profile.admission with maxGeneratedScalars := 63 } }
      let identityBad := { profile with id := "", profileVersion := 0 }
      let evaluatorBad := { profile with evaluatorVersion := "grouped-prefix-reference-v0" }
      let zeroRate := { profile with sampleRate := 0 }
      let domainBad := { profile with admission := { profile.admission with poles :=
        { profile.admission.poles with maxFrequencyHz := jn 10 } } }
      let branchCap := { profile with admission := { profile.admission with maxBranches := 0 } }
      let groupCap := { profile with admission := { profile.admission with maxCarrierGroups := 0 } }
      let periodCap := { profile with admission := { profile.admission with maxPeriod := 0 } }
      let explosiveGroup : RoomCarrierGroup := {
        id := "explosive-1", period := 1, radius := jn 9 1,
        carrier := #[jn 1000001] }
      let explosiveProfile : RoomProfile := {
        profile with
        groups := #[explosiveGroup]
        admission := {
          poles := profile.admission.poles
          maxBranches := 1
          maxSourceRows := 1
          maxCarrierGroups := 1
          maxPeriod := 1
          maxGeneratedScalars := 4 } }
      let explosiveMode : Array ModalMode := #[sourceA[0]!]
      let refusals :=
        exclusionHas .invalidProfileIdentity none none
          (specializeGroupedRoomReference identityBad 44100 sourceA) &&
        exclusionHas .unsupportedEvaluator none none
          (specializeGroupedRoomReference evaluatorBad 44100 sourceA) &&
        exclusionHas .invalidSampleRate none none
          (specializeGroupedRoomReference zeroRate 0 sourceA) &&
        exclusionHas .invalidPoleDomain none none
          (specializeGroupedRoomReference domainBad 44100 sourceA) &&
        exclusionHas .branchCapacity none none
          (specializeGroupedRoomReference branchCap 44100 sourceA) &&
        exclusionHas .groupCapacity none none
          (specializeGroupedRoomReference groupCap 44100 sourceA) &&
        exclusionHas .invalidCarrier none (some 0)
          (specializeGroupedRoomReference periodCap 44100 sourceA) &&
        exclusionHas .sampleRateMismatch none none
          (specializeGroupedRoomReference profile 48000 sourceA) &&
        exclusionHas .invalidCarrier none (some 0)
          (specializeGroupedRoomReference badPeriod 44100 sourceA) &&
        exclusionHas .invalidCarrier none (some 0)
          (specializeGroupedRoomReference badRadius 44100 sourceA) &&
        exclusionHas .invalidCarrier none (some 1)
          (specializeGroupedRoomReference duplicateIds 44100 sourceA) &&
        exclusionHas .nonconstantPole (some 0) none
          (specializeGroupedRoomReference profile 44100 livePole) &&
        exclusionHas .nonfinitePole (some 0) none
          (specializeGroupedRoomReference profile 44100 nonfinitePole) &&
        exclusionHas .unsupportedDegree (some 0) none
          (specializeGroupedRoomReference profile 44100 degreeOne) &&
        exclusionHas .poleOutOfDomain (some 0) none
          (specializeGroupedRoomReference profile 44100 outOfDomain) &&
        specializationOk (specializeGroupedRoomReference profile 44100 boundaryModes) &&
        exclusionHas .poleOutOfDomain (some 0) none
          (specializeGroupedRoomReference profile 44100 belowFrequency) &&
        exclusionHas .poleOutOfDomain (some 0) none
          (specializeGroupedRoomReference profile 44100 aboveFrequency) &&
        exclusionHas .sourceCapacity none none
          (specializeGroupedRoomReference rowCap 44100 sourceA) &&
        exclusionHas .generatedScalarCapacity none none
          (specializeGroupedRoomReference scalarCap 44100 sourceA) &&
        exclusionHas .prefixComputation (some 0) (some 0)
          (specializeGroupedRoomReference explosiveProfile 44100 explosiveMode)

      let n := 128
      let anchorF := 23.25
      let anchor := lit 2325 2
      let positions : Array Float := #[-1.0, -0.35, 0.0, 0.45, 1.0]
      let mut oracleWorst := 0.0
      let mut renderFailure : Option String := none
      let mut baseAtZero : Array Float := #[]
      let mut forwardPreZero := true
      let mut reversePreNonzero := false
      let mut planShape := true
      for (position, pi) in positions.zipIdx do
        if renderFailure.isNone then
          let plan := buildRoomPlan arena s!"room_ref_a_{pi}" specA clockLit anchor (litF position)
          match plan with
          | .error e => renderFailure := some s!"build position {position}: {firstLine e}"
          | .ok p =>
            planShape := planShape && planTagCount "ReduceBegin" p == 2
            match ← renderPlanSamples p n with
            | .error e => renderFailure := some s!"render position {position}: {firstLine e}"
            | .ok got =>
              let ref := directSamples profile sourceAOracle position n
                (fun i => i.toFloat - anchorF)
              let err := relL2 got ref
              if err > oracleWorst then oracleWorst := err
              if position == 0.0 then baseAtZero := got
              if position == 1.0 then
                for i in [0:24] do
                  if got[i]! != 0.0 then forwardPreZero := false
              if position == -1.0 then
                reversePreNonzero := (got.take 24).any (· != 0.0)

      let mut sourceBOracleOk := false
      match ← renderAgainstOracle arena "room_ref_b" specB sourceBOracle 0.2
          clockLit anchor n (fun i => i.toFloat - anchorF) with
      | .error e => if renderFailure.isNone then renderFailure := some e
      | .ok (_, err) =>
        sourceBOracleOk := err < 2e-7
        if err > oracleWorst then oracleWorst := err

      let mut amplitudeChanges := false
      match ← renderAgainstOracle arena "room_ref_amp" ampSpec
          #[{ sourceAOracle[0]! with cre := 2.0 }, sourceAOracle[1]!]
          0.0 clockLit anchor n (fun i => i.toFloat - anchorF) with
      | .error e => if renderFailure.isNone then renderFailure := some e
      | .ok (got, err) =>
        amplitudeChanges := maxAbsDiff got baseAtZero > 1e-4
        if err > oracleWorst then oracleWorst := err

      let mut isolatedMutations := false
      let mut zeroResiduesZero := false
      match frequencyResult, dampingResult, phaseResult, zeroResidueResult with
      | .ok frequencySpec, .ok dampingSpec, .ok phaseSpec, .ok zeroSpec =>
        let prefixContract := !prefixEqual specA frequencySpec &&
          !prefixEqual specA dampingSpec && prefixEqual specA phaseSpec &&
          prefixEqual specA zeroSpec
        match ← renderRoom arena "room_ref_frequency" frequencySpec clockLit anchor (lit 0) n,
              ← renderRoom arena "room_ref_damping" dampingSpec clockLit anchor (lit 0) n,
              ← renderRoom arena "room_ref_phase" phaseSpec clockLit anchor (lit 0) n,
              ← renderRoom arena "room_ref_anchor" specA clockLit (lit 2425 2) (lit 0) n,
              ← renderRoom arena "room_ref_zero_residue" zeroSpec clockLit anchor (lit 0) n with
        | .ok frequencyGot, .ok dampingGot, .ok phaseGot, .ok anchorGot, .ok zeroGot =>
          isolatedMutations := prefixContract &&
            maxAbsDiff frequencyGot baseAtZero > 1e-5 &&
            maxAbsDiff dampingGot baseAtZero > 1e-5 &&
            maxAbsDiff phaseGot baseAtZero > 1e-5 &&
            maxAbsDiff anchorGot baseAtZero > 1e-5
          zeroResiduesZero := zeroGot.all (· == 0.0)
        | .error e, _, _, _, _ | _, .error e, _, _, _ |
          _, _, .error e, _, _ | _, _, _, .error e, _ | _, _, _, _, .error e =>
          if renderFailure.isNone then renderFailure := some e
      | _, _, _, _ =>
        if renderFailure.isNone then
          renderFailure := some "isolated mutation specialization unexpectedly refused"

      let mut emptyZero := false
      match buildRoomPlan arena "room_ref_empty" emptySpec clockLit anchor (lit 0) with
      | .error e => if renderFailure.isNone then renderFailure := some (firstLine e)
      | .ok p => match ← renderPlanSamples p n with
        | .error e => if renderFailure.isNone then renderFailure := some (firstLine e)
        | .ok got => emptyZero := got.all (· == 0.0)

      let mut runtimeRateMismatchZero := false
      match buildRoomPlan arena "room_ref_rate_mismatch" specA clockLit anchor (lit 0) with
      | .error e => if renderFailure.isNone then renderFailure := some (firstLine e)
      | .ok p =>
        let mismatched := { p with sampleRate := jn 48000 }
        match ← renderPlanSamples mismatched n with
        | .error e => if renderFailure.isNone then renderFailure := some (firstLine e)
        | .ok got => runtimeRateMismatchZero := got.all (· == 0.0)

      let farSamples : Int := 1000000000
      let farQ : Int := farSamples * 4294967296
      let farAnchor := lit 100000002325 2
      let mut farTranslationExact := false
      match buildRoomPlan arena "room_ref_far" specA
          (add clockLit (litI farQ)) farAnchor (lit 0) with
      | .error e => if renderFailure.isNone then renderFailure := some (firstLine e)
      | .ok p => match ← renderPlanSamples p n with
        | .error e => if renderFailure.isNone then renderFailure := some (firstLine e)
        | .ok got => farTranslationExact := bitDiffCount got baseAtZero == 0

      let holdClock := litI (115 * 1073741824)
      let reverseClock := sub (litI (133 * 1073741824)) clockLit
      let mut holdReverseOk := false
      match ← renderAgainstOracle arena "room_ref_hold" specA sourceAOracle 0.45
          holdClock anchor n (fun _ => 5.5),
          ← renderAgainstOracle arena "room_ref_reverse" specA sourceAOracle (-0.35)
          reverseClock anchor n (fun i => 10.0 - i.toFloat) with
      | .ok (hold, holdErr), .ok (_, reverseErr) =>
        let holdExact := hold.all (· == hold[0]!)
        holdReverseOk := holdExact && holdErr < 2e-7 && reverseErr < 2e-7
        if holdErr > oracleWorst then oracleWorst := holdErr
        if reverseErr > oracleWorst then oracleWorst := reverseErr
      | .error e, _ | _, .error e =>
        if renderFailure.isNone then renderFailure := some e

      match renderFailure with
      | some e => failGate "grouped-room-reference" e
      | none =>
        let ok := structural && refusals && planShape && oracleWorst < 2e-7 &&
          sourceBOracleOk && amplitudeChanges && isolatedMutations && zeroResiduesZero &&
          runtimeRateMismatchZero && emptyZero && forwardPreZero && reversePreNonzero &&
          farTranslationExact && holdReverseOk
        IO.println s!"        room-only profile: groups={profile.groups.size}, generated={specA.generatedScalarCount} scalars, source grids A/B both admitted"
        IO.println s!"        direct carrier-train oracle worst relative L2={oracleWorst}; nested Plan-6 reductions={planShape}"
        IO.println s!"        profile/prefix separation={structural}; indexed refusals={refusals}; rate/zero/mutations={runtimeRateMismatchZero}/{zeroResiduesZero}/{isolatedMutations}"
        IO.println s!"        empty/endpoint/translation/hold/reverse={emptyZero}/{forwardPreZero}/{farTranslationExact}/{holdReverseOk}"
        if ok then
          passGate "grouped-room-reference"
            "one unchanged room profile specializes unrelated modal poles; exact prefixes and Plan-6 evaluator agree with the direct causal/reverse oracle"
        else
          failGate "grouped-room-reference" s!"structural={structural} refusals={refusals} shape={planShape} oracle={oracleWorst} sourceB={sourceBOracleOk} amp={amplitudeChanges} mutations={isolatedMutations} zeroResidues={zeroResiduesZero} rateMismatch={runtimeRateMismatchZero} empty={emptyZero} fwdPre={forwardPreZero} revPre={reversePreNonzero} far={farTranslationExact} holdRev={holdReverseOk}"

end Tropical.Tropicaltest.GroupedRoomReference
