import Tropical.Playground
import Tropical.StagedLoad
import Tropical.Tropicaltest.Stress
import Tropical.Testing.ArrowFixtures

/-!
# Current-universe modal phaser gates

These fixtures keep the Tropical-native semantics explicit: at each response
coordinate the controls choose one static continuous-time all-pass cascade.  A
direct Float rational-product oracle checks the rendered modal response; no
recursive pedal simulator participates in correctness.
-/

namespace Tropical.Tropicaltest.Phaser

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)
open Tropical.Playground
open Tropical.EmitArrow
open Tropical.Testing.ArrowFixtures

private def passGate (label detail : String) : IO Bool := do
  IO.println s!"  PASS  {label}  {detail}"
  pure true

private def failGate (label detail : String) : IO Bool := do
  IO.println s!"  FAIL  {label}  {detail}"
  pure false

private def sampleRate : Float := 44100.0
private def twoPi : Float := 6.283185307179586
private def anchor : Nat := 64
private def frameCount : Nat := 512

private abbrev CplxF := Float × Float

private def cadd (a b : CplxF) : CplxF := (a.1 + b.1, a.2 + b.2)
private def csub (a b : CplxF) : CplxF := (a.1 - b.1, a.2 - b.2)
private def cmul (a b : CplxF) : CplxF :=
  (a.1 * b.1 - a.2 * b.2, a.1 * b.2 + a.2 * b.1)
private def cdiv (a b : CplxF) : CplxF :=
  let norm := b.1 * b.1 + b.2 * b.2
  ((a.1 * b.1 + a.2 * b.2) / norm,
   (a.2 * b.1 - a.1 * b.2) / norm)
private def cscale (scale : Float) (a : CplxF) : CplxF :=
  (scale * a.1, scale * a.2)

private def constantControl (value : BuildM Sig) : BuildM ModalControlRef := do
  pure (ModalControlRef.constant (← value))

private def testPhaser (id input : String) : BuildM (Node × Array PatchNode) := do
  pure <| modalPhaserTopology id input
    (← constantControl (lit 700)) (← constantControl (lit 15 1))
    (← constantControl (lit 2 1)) (← constantControl (lit 5 1))
    Tropical.Playground.Metadata.modalPhaserRatios

private def sourceMode (sigma : Int := 3) : BuildM ModalMode := do
  ModalMode.hz (← lit 220) (← lit sigma) (← lit 1)

private def phaserGraph : BuildM PatchGraph := do
  let (phaser, topology) ← testPhaser "phaser" "source"
  let mode ← sourceMode
  let anchorExpr ← lit (Int.ofNat anchor)
  let clock ← clockLit
  let nodes : Array PatchNode := #[
      { id := "source", node := .modalSource #[mode] anchorExpr clock none },
      { id := "phaser", node := phaser }] ++ topology
  pure { nodes, output := "phaser" }

private def graphPlan (arena : Arena) (name : String) (graph : BuildM PatchGraph) :
    Except String FlatPlan :=
  buildAndFinish <| buildExprCarrier name (do
    let term ← lowerGraph (← graph)
    emitTerm (normalize term)) arena

private def renderGraph (arena : Arena) (name : String) (graph : BuildM PatchGraph) :
    IO (Except String (Array Float)) :=
  match graphPlan arena name graph with
  | .error error => pure (.error error)
  | .ok plan => renderPlanSamples plan frameCount

private def planInstructionCount (plan : FlatPlan) : Nat :=
  plan.instanceFunctions.foldl (fun total fn =>
    total + (Tropical.Ir.Stage0.collectBlocks fn).foldl
      (fun subtotal block => subtotal + block.size) 0) 0

private def planDivisionCount (plan : FlatPlan) : Nat :=
  plan.instanceFunctions.foldl (fun total fn =>
    total + (Tropical.Ir.Stage0.collectBlocks fn).foldl
      (fun subtotal block => subtotal + block.countP (·.tag == "Div")) 0) 0

private def benchmarkPhaserJson (sections : Nat) : Except String Lean.Json := do
  Lean.Json.parse ("{\"nodes\":["
    ++ String.intercalate "," #[
    "{\"id\":\"address\",\"kind\":\"source\",\"params\":{\"freq\":0.63,\"morph\":0},\"sel\":{},\"in\":{\"freq\":[]}}",
    "{\"id\":\"resonator\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":6},\"sel\":{},\"in\":{\"addr\":[\"address\"]}}",
    "{\"id\":\"phaser\",\"kind\":\"phaser\",\"params\":{\"center\":700,\"sweep\":1.5,\"rate\":0.2,\"mix\":0.5,\"_benchmark_stages\":"
      ++ toString sections ++ "},\"sel\":{},\"in\":{\"in\":[\"resonator\"]}}",
    "{\"id\":\"out\",\"kind\":\"out\",\"params\":{},\"sel\":{},\"in\":{\"in\":[\"phaser\"]}}"].toList
    ++ "],\"out\":\"out\",\"taps\":[]}")

private structure StagedShape where
  sections : Nat
  rows : Nat
  exactInstructions : Nat
  audioInstructions : Nat
  exactDivisions : Nat
  audioDivisions : Nat
deriving Inhabited

private def stagedShape (arena : Arena)
    (resolved : Array (String × ProgramIdx)) (sections : Nat) :
    Except String StagedShape := do
  let json ← benchmarkPhaserJson sections
  let compiled ← compilePlanPure arena resolved json
  let plan := compiled.plan
  let mixed := Tropical.Ir.phaserTimeStagingMixedDDEnabled
  let expectedProvenance := if mixed then
    "staged_phaser_admitted:mixed_dd_experiment"
  else "staged_phaser_admitted"
  -- Structural interning may share a field when its endpoint expressions are
  -- identical, hence the lower bounds rather than exact slot counts.
  if plan.tileArraySlots.size < (if mixed then 12 else 7)
      || plan.phaserTimeStaging != some expectedProvenance then
    throw s!"{sections} sections were not admitted (tile slots={plan.tileArraySlots}, provenance={plan.phaserTimeStaging})"
  let split ← Tropical.Ir.TileStage.split plan
  let some _ := split.tile? | throw s!"{sections} sections produced no tile program"
  let simpleRows := if mixed then 6 + (sections + 1) / 2 else 6 + sections
  let pairedRows := if mixed then sections / 2 else 0
  for slot in plan.tileArraySlots do
    let size := plan.arraySlotSizes[slot]?.getD 0
    if size != simpleRows && (!mixed || size != pairedRows) then
      throw s!"{sections} sections: endpoint slot {slot} has wrong row count {size}"
  pure {
    sections
    rows := simpleRows + pairedRows
    exactInstructions := planInstructionCount plan
    audioInstructions := planInstructionCount split.audio
    exactDivisions := planDivisionCount plan
    audioDivisions := planDivisionCount split.audio }

private def stagedTerminalStructureCheck (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  if !Tropical.Ir.phaserTimeStagingEnabled then
    return true
  let shapes := #[6, 12, 18].mapM (stagedShape arena resolved)
  let ineligible := do
    let json ← benchmarkPhaserJson 5
    pure (← compilePlanPure arena resolved json).plan
  match shapes, ineligible with
  | .ok results, .ok fallback =>
      let compact := results.all fun shape =>
        shape.audioInstructions < shape.exactInstructions
          && shape.audioDivisions * 4 < shape.exactDivisions
      let incrementsLinear := decide (
        results[2]!.audioInstructions - results[1]!.audioInstructions
          ≤ 2 * (results[1]!.audioInstructions - results[0]!.audioInstructions))
      let refused := fallback.tileArraySlots.isEmpty
        && fallback.phaserTimeStaging
          == some "staged_phaser_fallback:no_admissible_terminal"
      for shape in results do
        IO.println s!"        staged {shape.sections} sections/{shape.rows} rows: exact/audio instructions={shape.exactInstructions}/{shape.audioInstructions}, divisions={shape.exactDivisions}/{shape.audioDivisions}"
      if compact && incrementsLinear && refused then
        passGate "phaser-time-stage-structure"
          (if Tropical.Ir.phaserTimeStagingMixedDDEnabled then
            "6/12/18 admitted with falsification-only mixed simple/DD endpoint fields; compact audio work grows approximately linearly; five-section topology retains exact fallback provenance"
          else
            "6/12/18 admitted with ordinary absolute endpoint fields; compact audio work grows approximately linearly; five-section topology retains exact fallback provenance")
      else
        failGate "phaser-time-stage-structure"
          s!"compact={compact} linear={incrementsLinear} ineligibleFallback={refused}"
  | .error error, _ =>
      failGate "phaser-time-stage-structure" s!"admission: {firstLine error}"
  | _, .error error =>
      failGate "phaser-time-stage-structure" s!"fallback: {firstLine error}"

private def lfoPhase (sample : Nat) (rate : Float) : Float :=
  let inc := (rate * 4294967296.0 / sampleRate).toUInt64.toNat
  ((inc * sample) % 4294967296).toFloat / 4294967296.0

private def sectionPoles (sample : Nat) : Array Float :=
  let octave := 1.5 * Float.sin (twoPi * lfoPhase sample 0.2)
  let scale := Float.pow 2.0 octave
  Tropical.Playground.Metadata.modalPhaserRatios.map fun ratio =>
    twoPi * 700.0 * ratio * scale

private def allpassAt (s : CplxF) (poles : Array Float) : CplxF :=
  poles.foldl (fun value a =>
    cmul value (cdiv (s.1 - a, s.2) (s.1 + a, s.2))) (1.0, 0.0)

private def evalResidue (pole residue : CplxF) (time : Float) : Float :=
  let carrier := residue.1 * Float.cos (pole.2 * time) -
    residue.2 * Float.sin (pole.2 * time)
  Float.exp (pole.1 * time) * carrier

/-- Direct distinct-pole inverse-Laplace oracle for
    `X(s)·(0.5 + 0.5·product A_k(s))`. -/
private def phaserOracle (sample : Nat) : Float :=
  let time := (sample.toFloat - anchor.toFloat) / sampleRate
  if time ≤ 0.0 then 0.0 else Id.run do
    let poles := sectionPoles sample
    let lam : CplxF := (-3.0, twoPi * 220.0)
    let sourceResidue : CplxF := (1.0, 0.0)
    let sourceWet := cmul sourceResidue (allpassAt lam poles)
    let sourceMixed := cadd (cscale 0.5 sourceResidue) (cscale 0.5 sourceWet)
    let mut total := evalResidue lam sourceMixed time
    for (a, k) in poles.zipIdx do
      let pole : CplxF := (-a, 0.0)
      let mut allpassResidue : CplxF := (-2.0 * a, 0.0)
      for (other, j) in poles.zipIdx do
        if j != k then
          allpassResidue := cmul allpassResidue
            (cdiv (pole.1 - other, pole.2) (pole.1 + other, pole.2))
      let sourceAtPole := cdiv sourceResidue (csub pole lam)
      let residue := cscale 0.5 (cmul allpassResidue sourceAtPole)
      total := total + evalResidue pole residue time
    return total

private def maximumOracleError (samples : Array Float) : Float := Id.run do
  let mut worst := 0.0
  for i in [0:samples.size] do
    let error := (samples[i]! - phaserOracle i).abs
    if error > worst then worst := error
  worst

/-- JIT differential for the fused two-room product schedule against the
    ordinary exact terminal over the independently gated compact decoration.
    The fixture deliberately uses interior directions so future and past room
    arms, the zero seam, and all six cold phaser-pole rows participate. -/
private def fusedProductError (arena : Arena) : IO (Except String Float) := do
  let built := buildExprCarrier "modal_phaser_fused_differential" (do
    let source : Array ModalMode := #[← sourceMode]
    let room1a ← ModalMode.hz (← lit 90) (← lit 9) (← lit 3 1)
    let room1b ← ModalMode.hz (← lit 370) (← lit 13) (← lit (-1) 2)
    let room1 : Array ModalMode := #[room1a, room1b]
    let room2a ← ModalMode.hz (← lit 90) (← lit 7) (← lit 2 1)
    let room2b ← ModalMode.hz (← lit 370) (← lit 15) (← lit 1 2)
    let room2 : Array ModalMode := #[room2a, room2b]
    let tails ← Tropical.Playground.Metadata.modalPhaserRatios.mapM fun ratio => do
      Oriented.allpassTail (← litF (twoPi * 700.0 * ratio))
    let mix ← lit 5 1
    let direction1 ← lit 25 2
    let direction2 ← lit 75 2
    let decorated ← Oriented.decorateDegreeZeroCausalPhaser source tails mix
    let some reference ← Oriented.factoredTwoRoomTerminal? decorated room1 room2
        direction1 direction2
      | throw "reference terminal refused decorated fixture"
    let some specialized ← Oriented.factoredTwoRoomPhaserTerminal? source tails
        room1 room2 mix direction1 direction2
      | throw "fused terminal refused admitted fixture"
    let clock ← clockLit
    let anchorE ← lit (Int.ofNat anchor)
    sub (← specialized.realizeSig clock anchorE)
      (← reference.realizeSig clock anchorE)) arena
  let plan ← match buildAndFinish built with
    | .error error => return .error error
    | .ok plan => pure plan
  match ← renderPlanSamples plan frameCount with
  | .error error => pure (.error error)
  | .ok samples =>
      pure (.ok (samples.foldl (fun worst sample => max worst sample.abs) 0.0))

private def controlValue (control : ModalControlRef) : BuildM (Option Float) := do
  let .konst expression := control.fallback | pure none
  sigConstF? expression

private def lowerModalRoot (graph : PatchGraph) (id : String) : BuildM ModalForest := do
  let ids := graph.nodes.map (·.id)
  let deps := graph.nodes.map fun node => node.node.inputIds.filterMap ids.idxOf?
  let some ranks := Tropical.Ir.topoRanks? deps | throw "fixture cycle"
  let some index := ids.idxOf? id | throw s!"fixture node '{id}' not found"
  let some rank := ranks[index]? | throw s!"fixture node '{id}' has no rank"
  lowerModal graph (fun name => (ids.idxOf? name).bind (ranks[·]?)) id rank

/-- Visible topology and retained compiler structure both grow by one factor
    per authored section; no parallel product is distributed here. -/
private def retainedSizeCheck (ratios : Array Float) : BuildM Bool := do
  let (phaser, topology) := modalPhaserTopology "phaser" "source"
    (← constantControl (lit 700)) (← constantControl (lit 15 1))
    (← constantControl (lit 2 1)) (← constantControl (lit 5 1)) ratios
  let mode ← sourceMode
  let anchorExpr ← lit 0
  let clock ← clockLit
  let graph : PatchGraph := {
    nodes := #[
      { id := "source", node := .modalSource #[mode] anchorExpr clock none },
      { id := "phaser", node := phaser }] ++ topology
    output := "phaser" }
  let tailNodes := topology.countP fun node => node.node matches .modalLinear _ _
  let junctions := topology.countP fun node => node.node matches .modalMix _
  match ← lowerModalRoot graph "phaser" with
  | #[branch] => match branch.stages.toList with
      | [.linear stage] => do
          let values ← branch.controls.filterMapM controlValue
          match (← stage.build (← clockLit) (← values.mapM litF)).dryWetAllpassCascadeShape? with
          | some (tails, _) =>
              pure <| topology.size == 2 * ratios.size && tailNodes == ratios.size &&
                junctions == ratios.size && tails.size == ratios.size &&
                branch.controls.size == 3 * ratios.size + 1
          | none => pure false
      | _ => pure false
  | _ => pure false

private def deferredStructureCheck : BuildM Bool := do
  let graph ← phaserGraph
  let explicitTails := graph.nodes.countP fun node =>
    node.node matches .modalLinear _ _
  let explicitJunctions := graph.nodes.countP fun node =>
    node.node matches .modalMix _
  let ratios := Tropical.Playground.Metadata.modalPhaserRatios
  let small ← retainedSizeCheck (ratios.extract 0 1)
  let medium ← retainedSizeCheck (ratios.extract 0 2)
  let full ← retainedSizeCheck ratios
  let shape ← match ← lowerModalRoot graph "phaser" with
  | #[branch] => do
      let retained ← match branch.stages.toList with
      | [.linear stage] => do
          let values ← branch.controls.filterMapM controlValue
          match (← stage.build (← clockLit) (← values.mapM litF)).dryWetAllpassCascadeShape? with
          | some (tails, mix) => do
              let mixValue ← sigConstF? mix
              let control0 ← match branch.controls[0]? with
                | some control => controlValue control
                | none => pure none
              let control1 ← match branch.controls[1]? with
                | some control => controlValue control
                | none => pure none
              let control2 ← match branch.controls[2]? with
                | some control => controlValue control
                | none => pure none
              pure <| tails.size == 6 && mixValue == some 0.5 &&
                branch.controls.size == 19 &&
                control0 == some 700.0 && control1 == some 1.5 &&
                control2 == some 0.2
          | none => pure false
      | _ => pure false
      pure <| explicitTails == 6 && explicitJunctions == 6 &&
        nodeIsModal graph "phaser" && retained
  | _ => pure false
  pure (small && medium && full && shape)

/-- Filter is the independent second producer proving the retained algebra is
    not merely a renamed phaser stage. -/
private def filterUsesGenericKernelCheck : BuildM Bool := do
  let empty := Lean.Json.mkObj []
  let (node, _) ← Tropical.Playground.Compiler.buildNode
    (fun _ => none) "filter" "filter" empty empty empty
  match node with
  | .modalLinear _ stage =>
      let values ← stage.controls.filterMapM controlValue
      pure (← stage.build (← clockLit) (← values.mapM litF)).orientedShape?.isSome
  | _ => pure false

private def modalMixOrderCheck : BuildM Bool := do
  let (phaser, topology) ← testPhaser "phaser" "mix"
  let modeA ← sourceMode 3
  let modeB ← sourceMode 7
  let anchorA ← lit 0
  let anchorB ← lit 0
  let clockA ← clockLit
  let clockB ← clockLit
  let graph : PatchGraph :=
    { nodes := #[
        { id := "a", node := .modalSource #[modeA] anchorA clockA none },
        { id := "b", node := .modalSource #[modeB] anchorB clockB none },
        { id := "mix", node := .modalMix #["b", "a"] },
        { id := "phaser", node := phaser }] ++ topology
      output := "phaser" }
  match ← lowerModalRoot graph "phaser" with
  | #[first, second] => do
      let sigmaOf := fun branch => match branch.source with
        | .plain modes => match modes[0]? with
          | some mode => sigConstF? mode.sigma
          | none => pure none
        | _ => pure none
      pure <| (← sigmaOf first) == some 7.0 && (← sigmaOf second) == some 3.0 &&
        first.stages.size == 1 && second.stages.size == 1
  | _ => pure false

private def refusalChecks : BuildM Bool := do
  let (phaser, topology) ← testPhaser "phaser" "bloom"
  let mode ← sourceMode
  let anchorExpr ← lit 0
  let clock ← clockLit
  let bloomed : PatchGraph :=
    { nodes := #[
        { id := "bloom", node := .modalSource #[mode] anchorExpr clock none
            none (some (0.1, 1.8)) },
        { id := "phaser", node := phaser }] ++ topology
      output := "phaser" }
  let builder ← get
  let bloomOk := match (lowerGraph bloomed).run builder with
    | .error error => error == "lower: bloomed linear-kernel crossing at 'phaser' refused (the live linear/Gamma crossing is not implemented)"
    | .ok _ => false
  pure bloomOk

private def phaserPatchJson : String :=
  "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"ph\",\"kind\":\"phaser\",\"params\":{\"center\":700,\"sweep\":1.5,\"rate\":0.2,\"mix\":0.5},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"ph\"]}}],\"out\":\"out\"}"

private def hierarchicalPhaserPatchJson : String :=
  "{\"version\":3,\"scene\":{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"ph\",\"kind\":\"module\",\"definition\":\"tropical.modal.phaser\",\"definition_version\":1,\"params\":{\"center\":700,\"sweep\":1.5,\"rate\":0.2,\"mix\":0.5},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"ph\"]}}],\"out\":\"out\"}}"

private def illegalSignalToModalModuleJson : String :=
  "{\"version\":3,\"scene\":{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    "{\"id\":\"ph\",\"kind\":\"module\",\"definition\":\"tropical.modal.phaser\",\"definition_version\":1,\"in\":{\"in\":[\"osc\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"ph\"]}}],\"out\":\"out\"}}"

private def blockDifference (left right : Array Float) : Float := Id.run do
  let mut energy := 0.0
  for i in [0:min left.size right.size] do
    let delta := left[i]! - right[i]!
    energy := energy + delta * delta
  energy

private def oneControlLive (compiled : CompiledPatch) (name : String)
    (newValue : Float) : IO Bool := do
  let left ← Tropical.Ffi.Runtime.new 1024
  Tropical.StagedLoad.loadTyped left compiled.plan compiled.stageBlocks
  let right ← Tropical.Ffi.Runtime.new 1024
  Tropical.StagedLoad.loadTyped right compiled.plan compiled.stageBlocks
  let v0? ← left.slotIndex? s!"param:ph.{name}#v0"
  let v1? ← left.slotIndex? s!"param:ph.{name}#v1"
  left.process
  right.process
  if let some v0 := v0? then left.setSlot v0 newValue
  if let some v1 := v1? then left.setSlot v1 newValue
  left.process
  right.process
  let moved := decodeF64LE (← left.outputBytes)
  let baseline := decodeF64LE (← right.outputBytes)
  pure (v0?.isSome && v1?.isSome && blockDifference moved baseline > 1.0e-14)

private def surfaceAndLiveCheck (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO (Except String Bool) := do
  let json ← match Lean.Json.parse phaserPatchJson with
    | .error error => return .error s!"json: {error}"
    | .ok value => pure value
  let compiled ← match compilePlanPure arena resolved json with
    | .error error => return .error s!"compile: {firstLine error}"
    | .ok value => pure value
  let center ← oneControlLive compiled "center" 1700.0
  let sweep ← oneControlLive compiled "sweep" 0.2
  let rate ← oneControlLive compiled "rate" 3.0
  let mix ← oneControlLive compiled "mix" 0.1
  pure (.ok (center && sweep && rate && mix))

private def hierarchyEquivalenceCheck (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO (Except String Bool) := do
  let prepared : Except String
      (Lean.Json × CompiledPatch × CompiledPatch × CompiledPatch × Bool) := do
    let legacy ← Lean.Json.parse phaserPatchJson
    let hierarchical ← Lean.Json.parse hierarchicalPhaserPatchJson
    let flat ← elaboratePatchHierarchy hierarchical
    let libraryDefinitions ← (hierarchyLibraryJson.getObjVal? "definitions")
      |>.bind (fun value => value.getArr?)
    let some installedPhaser := libraryDefinitions[1]?
      | throw "module library has no Phaser definition"
    let customPhaser := match installedPhaser with
      | .obj fields => .obj (fields.insert "id" (.str "user.ph.definition"))
      | _ => installedPhaser
    let authored := match hierarchical with
      | .obj fields =>
        let scene := match fields["scene"]? with
          | some (.obj sceneFields) => .obj <| sceneFields.insert "nodes" (.arr #[
              Lean.Json.mkObj [
                ("id", .str "res"), ("kind", .str "resonator"),
                ("params", Lean.Json.mkObj [("freq", .num ⟨220, 0⟩), ("decay", .num ⟨4, 0⟩)])],
              Lean.Json.mkObj [
                ("id", .str "ph"), ("kind", .str "module"),
                ("definition", .str "user.ph.definition"),
                ("definition_version", Lean.toJson 1),
                ("params", Lean.Json.mkObj [("center", .num ⟨700, 0⟩),
                  ("sweep", .num ⟨15, 1⟩), ("rate", .num ⟨2, 1⟩),
                  ("mix", .num ⟨5, 1⟩)]),
                ("in", Lean.Json.mkObj [("in", .arr #[.str "res"])])],
              Lean.Json.mkObj [
                ("id", .str "out"), ("kind", .str "out"),
                ("in", Lean.Json.mkObj [("in", .arr #[.str "ph"])])]])
          | other => other.getD (.obj {})
        .obj <| fields.insert "definitions" (.arr #[customPhaser])
          |>.insert "scene" scene
      | _ => hierarchical
    let legacyCompiled ← compilePlanPure arena resolved legacy
    let hierarchicalCompiled ← compilePlanPure arena resolved hierarchical
    let authoredCompiled ← compilePlanPure arena resolved authored
    let illegal ← Lean.Json.parse illegalSignalToModalModuleJson
    let typedBoundaryRefusal := match compilePlanPure arena resolved illegal with
      | .error error => (error.splitOn "signal→modal").length > 1
      | .ok _ => false
    pure (flat, legacyCompiled, hierarchicalCompiled, authoredCompiled,
      typedBoundaryRefusal)
  match prepared with
  | .error error => pure (.error error)
  | .ok (flat, legacyCompiled, hierarchicalCompiled, authoredCompiled,
      typedBoundaryRefusal) =>
    let flatRaws := Tropical.Playground.Metadata.rawsOf flat
    let sourceMapSize := match flat.getObjVal? "source_map" with
      | .ok (.arr entries) => entries.size
      | _ => 0
    match ← renderPlanSamples legacyCompiled.plan frameCount,
        ← renderPlanSamples hierarchicalCompiled.plan frameCount,
        ← renderPlanSamples authoredCompiled.plan frameCount with
    | .error error, _, _ | _, .error error, _ | _, _, .error error => pure (.error error)
    | .ok legacySamples, .ok hierarchicalSamples, .ok authoredSamples =>
      let maxDifference := (Array.range (min legacySamples.size hierarchicalSamples.size)).foldl
        (fun worst index => max worst (legacySamples[index]! - hierarchicalSamples[index]!).abs) 0.0
      let authoredDifference := (Array.range (min legacySamples.size authoredSamples.size)).foldl
        (fun worst index => max worst (legacySamples[index]! - authoredSamples[index]!).abs) 0.0
      let aliases := #["center", "sweep", "rate", "mix"].all fun knob =>
        hierarchicalCompiled.plan.paramDisciplines.any (·.name == s!"ph.{knob}")
      let noLeakedInternals := !hierarchicalCompiled.plan.paramDisciplines.any fun discipline =>
        "__h3_".isPrefixOf discipline.name
      IO.eprintln s!"        hierarchy flat={flatRaws.size}/15 source-map={sourceMapSize}/13 aliases={aliases} leaked-internals={!noLeakedInternals} typed-boundary={typedBoundaryRefusal} shipped-diff={maxDifference} authored-diff={authoredDifference}"
      pure <| Except.ok <| flatRaws.size == 15 && sourceMapSize == 13 &&
        flatRaws.any (fun raw => raw.id == "ph" && raw.kind == "modalblend") &&
        aliases && noLeakedInternals && typedBoundaryRefusal &&
        maxDifference < 2.0e-5 && authoredDifference < 2.0e-5

private def hierarchyValidationCheck : Bool :=
  let selfReference :=
    "{\"version\":3,\"definitions\":[{\"id\":\"user.loop\",\"version\":1,\"input\":\"input\",\"output\":\"output\",\"input_domain\":\"modal\",\"output_domain\":\"modal\",\"parameters\":[],\"nodes\":[{\"id\":\"input\",\"kind\":\"module_input\"},{\"id\":\"again\",\"kind\":\"module\",\"definition\":\"user.loop\",\"definition_version\":1,\"in\":{\"in\":[\"input\"]}},{\"id\":\"output\",\"kind\":\"module_output\",\"in\":{\"in\":[\"again\"]}}]}],\"scene\":{\"nodes\":[],\"out\":\"\"}}"
  let innerCycle :=
    "{\"version\":3,\"definitions\":[{\"id\":\"user.cycle\",\"version\":1,\"input\":\"input\",\"output\":\"output\",\"input_domain\":\"modal\",\"output_domain\":\"modal\",\"parameters\":[],\"nodes\":[{\"id\":\"input\",\"kind\":\"module_input\"},{\"id\":\"a\",\"kind\":\"modalmix\",\"in\":{\"in\":[\"b\"]}},{\"id\":\"b\",\"kind\":\"modalmix\",\"in\":{\"in\":[\"a\"]}},{\"id\":\"output\",\"kind\":\"module_output\",\"in\":{\"in\":[\"a\"]}}]}],\"scene\":{\"nodes\":[],\"out\":\"\"}}"
  let installedOverride :=
    "{\"version\":3,\"definitions\":[{\"id\":\"tropical.modal.allpass1\",\"version\":1,\"input\":\"input\",\"output\":\"output\",\"parameters\":[],\"nodes\":[{\"id\":\"input\",\"kind\":\"module_input\"},{\"id\":\"output\",\"kind\":\"module_output\",\"in\":{\"in\":[\"input\"]}}]}],\"scene\":{\"nodes\":[],\"out\":\"\"}}"
  let errorsAsExpected := fun source needle => match Lean.Json.parse source with
    | .error _ => false
    | .ok json => match elaboratePatchHierarchy json with
      | .error error => (error.splitOn needle).length > 1
      | .ok _ => false
  let stable := match Lean.Json.parse hierarchicalPhaserPatchJson with
    | .error _ => false
    | .ok json => match elaboratePatchHierarchy json, elaboratePatchHierarchy json with
      | .ok left, .ok right => left.compress == right.compress
      | _, _ => false
  errorsAsExpected selfReference "definition-reference cycle" &&
    errorsAsExpected innerCycle "cycle in definition 'user.cycle'" &&
    errorsAsExpected installedOverride "cannot replace installed definition" && stable

/-- Compile the product shape through the exact two-room terminal and report its
    actual Metal scratch publication.  The hard support gate remains 24,576 B. -/
private structure ProductScratch where
  total : Nat
  arrays : Nat
  maxRoutedRecords : Nat
  arraySlots : Nat
  coeffArraySlots : Nat
  nonCoeffSizes : Array Nat

private def scratchForJson (arena : Arena)
    (resolved : Array (String × ProgramIdx)) (source : String) :
    Except String ProductScratch := do
  let json ← Lean.Json.parse source
  let compiled ← compilePlanPure arena resolved json
  let split ← Tropical.Ir.Stage0.hoistTyped compiled.plan compiled.stageBlocks
  let audio := split.audio
  let arrays := (Array.range audio.arraySlotCount).foldl (fun bytes slot =>
    if audio.coeffArraySlots.contains slot then bytes else
      bytes + 4 * max (audio.arraySlotSizes[slot]?.getD 1) 1) 0
  let maxRoutedRecords := audio.instanceFunctions.foldl (fun maximum fn =>
    (Tropical.Ir.Stage0.collectBlocks fn).foldl (fun maximum block =>
      block.foldl (fun maximum instruction =>
        if instruction.tag == "RoutedSumBegin" then
          max maximum instruction.routedRoutes.size else maximum) maximum) maximum) 0
  pure {
    total := audio.metalThreadgroupScratchBytes
    arrays
    maxRoutedRecords
    arraySlots := audio.arraySlotCount
    coeffArraySlots := audio.coeffArraySlots.size
    nonCoeffSizes := (Array.range audio.arraySlotCount).filterMap fun slot =>
      if audio.coeffArraySlots.contains slot then none
      else some ((audio.arraySlotSizes[slot]?).getD 1) }

private def productScratchCheck (arena : Arena)
    (resolved : Array (String × ProgramIdx)) :
    Except String (ProductScratch × ProductScratch) := do
  let jsonHead := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"a\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.6},\"in\":{\"in\":[\"res\"]}},"
  let productTail :=
    "{\"id\":\"b\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.9},\"in\":{\"in\":[\"ph\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"b\"]}}],\"out\":\"out\"}"
  let productSource := jsonHead ++
    "{\"id\":\"ph\",\"kind\":\"phaser\",\"params\":{\"center\":700,\"sweep\":1.5,\"rate\":0.2,\"mix\":0.5},\"in\":{\"in\":[\"a\"]}}," ++ productTail
  let baselineSource := jsonHead ++
    "{\"id\":\"b\",\"kind\":\"reverb\",\"params\":{\"rt60\":0.9},\"in\":{\"in\":[\"a\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"b\"]}}],\"out\":\"out\"}"
  pure (← scratchForJson arena resolved productSource,
    ← scratchForJson arena resolved baselineSource)

def runPhaser (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  IO.eprintln "current-universe modal phaser: structural gates"
  let structuralChecks := runBuild arena do
    pure (← deferredStructureCheck, ← filterUsesGenericKernelCheck,
      ← modalMixOrderCheck, ← refusalChecks)
  let (structural, genericFilter, forestOrder, refusals) ←
    match structuralChecks with
    | .error error =>
        IO.eprintln s!"        arena-native structural fixture: {firstLine error}"
        pure (false, false, false, false)
    | .ok (_, checks) => pure checks
  let hierarchyValidation := hierarchyValidationCheck
  let stagedStructure ← stagedTerminalStructureCheck arena resolved
  IO.eprintln "        rendering independent-oracle fixture"
  let numeric ← renderGraph arena "modal_phaser_oracle" phaserGraph
  let fused ← fusedProductError arena
  IO.eprintln "        compiling and driving four live controls"
  let live ← surfaceAndLiveCheck arena resolved
  IO.eprintln "        expanding nested v3 Phaser/Allpass definitions"
  let hierarchy ← hierarchyEquivalenceCheck arena resolved
  IO.eprintln "        compiling canonical two-room product scratch fixture"
  let product := productScratchCheck arena resolved
  match numeric, fused, live, hierarchy, product with
  | .ok samples, .ok fusedError, .ok controlsLive, .ok hierarchyEquivalent,
      .ok (scratch, baseline) =>
      let oracleError := maximumOracleError samples
      IO.println s!"        topology-derived={structural} generic-filter={genericFilter} forest-order={forestOrder} refusals={refusals} hierarchy-validation={hierarchyValidation} oracle max abs={oracleError}"
      IO.println s!"        fused two-room JIT vs generic max abs={fusedError} ({fusedError * 1.0e9}e-9)"
      IO.println s!"        four served controls live without relower={controlsLive}; canonical 6→14→6-section→14 Metal scratch={scratch.total}/24576 (arrays={scratch.arrays}, max-routes={scratch.maxRoutedRecords}×4, slots={scratch.arraySlots}/{scratch.coeffArraySlots} coeff)"
      IO.println s!"        nested v3 Phaser→Allpass expansion equivalent={hierarchyEquivalent}; public ph.* aliases retained"
      IO.println s!"        two-room baseline scratch={baseline.total} (arrays={baseline.arrays}, max-routes={baseline.maxRoutedRecords}×4, slots={baseline.arraySlots}/{baseline.coeffArraySlots} coeff)"
      IO.println s!"        product non-coeff array floats={repr scratch.nonCoeffSizes}; baseline={repr baseline.nonCoeffSizes}"
      if structural && genericFilter && forestOrder && refusals && hierarchyValidation && stagedStructure && oracleError < 2.0e-5 &&
          -- The two exact schedules sum the same analytic rows in different
          -- routed orders.  This absolute lens is below two accumulated
          -- Q4.28 quanta per participating row and remains meaningful at
          -- dry/wet cancellation zeros where relative error is not.
          fusedError < 2.0e-7 &&
          controlsLive && hierarchyEquivalent && scratch.total ≤ 24576 then
        passGate "modal-phaser"
          "topology-derived generic kernel; Filter is a second producer; independent rational render oracle; authored modalMix order; live controls; named bloom crossing refusal; exact two-room product stays inside Metal scratch policy"
      else
        failGate "modal-phaser" "structural, numerical, live-control, or product scratch contract failed"
  | .error error, _, _, _, _ => failGate "modal-phaser" s!"render: {firstLine error}"
  | _, .error error, _, _, _ => failGate "modal-phaser" s!"fused: {firstLine error}"
  | _, _, .error error, _, _ => failGate "modal-phaser" s!"surface/live: {firstLine error}"
  | _, _, _, .error error, _ => failGate "modal-phaser" s!"hierarchy: {firstLine error}"
  | _, _, _, _, .error error => failGate "modal-phaser" s!"product: {firstLine error}"

end Tropical.Tropicaltest.Phaser
