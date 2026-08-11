import Tropical.Playground
import Tropical.StagedLoad
import Tropical.Tropicaltest.Stress

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
open Tropical.EmitArrow
open Tropical.Playground

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

private def constantControl (value : Sig) : ModalControlRef :=
  ModalControlRef.constant value

private def testPhaser (input : String) : Node :=
  .modalPhaser input
    (constantControl (lit 700)) (constantControl (lit 15 1))
    (constantControl (lit 2 1)) (constantControl (lit 5 1))
    modalPhaserRatios

private def sourceMode (sigma : Int := 3) : ModalMode :=
  ModalMode.hz (lit 220) (lit sigma) (lit 1)

private def phaserGraph : PatchGraph :=
  { nodes := #[
      { id := "source", node := .modalSource #[sourceMode] (lit (Int.ofNat anchor))
          clockLit none },
      { id := "phaser", node := testPhaser "source" }]
    output := "phaser" }

private def graphPlan (arena : Arena) (name : String) (graph : PatchGraph) :
    Except String FlatPlan := do
  let term ← lowerGraph graph
  let (output, _) := emitTerm (normalize term) {}
  buildAndFinish (.ok (buildExprCarrier name output arena))

private def renderGraph (arena : Arena) (name : String) (graph : PatchGraph) :
    IO (Except String (Array Float)) :=
  match graphPlan arena name graph with
  | .error error => pure (.error error)
  | .ok plan => renderPlanSamples plan frameCount

private def lfoPhase (sample : Nat) (rate : Float) : Float :=
  let inc := (rate * 4294967296.0 / sampleRate).toUInt64.toNat
  ((inc * sample) % 4294967296).toFloat / 4294967296.0

private def sectionPoles (sample : Nat) : Array Float :=
  let octave := 1.5 * Float.sin (twoPi * lfoPhase sample 0.2)
  let scale := Float.pow 2.0 octave
  modalPhaserRatios.map fun ratio => twoPi * 700.0 * ratio * scale

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
  let source : Array ModalMode := #[sourceMode]
  let room1 : Array ModalMode := #[
    ModalMode.hz (lit 90) (lit 9) (lit 3 1),
    ModalMode.hz (lit 370) (lit 13) (lit (-1) 2)]
  let room2 : Array ModalMode := #[
    ModalMode.hz (lit 90) (lit 7) (lit 2 1),
    ModalMode.hz (lit 370) (lit 15) (lit 1 2)]
  let tails := modalPhaserRatios.map fun ratio =>
    Oriented.allpassTail (litF (twoPi * 700.0 * ratio))
  let mix := lit 5 1
  let direction1 := lit 25 2
  let direction2 := lit 75 2
  let decorated := Oriented.decorateDegreeZeroCausalPhaser source tails mix
  let reference ← match Oriented.factoredTwoRoomTerminal? decorated room1 room2
      direction1 direction2 with
    | none => return .error "reference terminal refused decorated fixture"
    | some terminal => pure terminal
  let specialized ← match Oriented.factoredTwoRoomPhaserTerminal? source tails
      room1 room2 mix direction1 direction2 with
    | none => return .error "fused terminal refused admitted fixture"
    | some terminal => pure terminal
  let clock := clockLit
  let anchorE := lit (Int.ofNat anchor)
  let difference := sub (specialized.realizeSig clock anchorE)
    (reference.realizeSig clock anchorE)
  let plan ← match buildAndFinish (.ok
      (buildExprCarrier "modal_phaser_fused_differential" difference arena)) with
    | .error error => return .error error
    | .ok plan => pure plan
  match ← renderPlanSamples plan frameCount with
  | .error error => pure (.error error)
  | .ok samples =>
      pure (.ok (samples.foldl (fun worst sample => max worst sample.abs) 0.0))

private def controlValue (control : ModalControlRef) : Option Float := do
  let .konst expression := control.fallback | none
  sigConstF? expression

private def deferredStructureCheck : Bool :=
  let rankOf := fun id => if id == "source" then some 0 else
    if id == "phaser" then some 1 else none
  match lowerModal phaserGraph rankOf "phaser" 1 with
  | .ok #[branch] =>
      nodeIsModal phaserGraph "phaser" && match branch.stages.toList with
      | [.phaser stage] =>
          stage.ratios == modalPhaserRatios &&
            controlValue stage.center == some 700.0 &&
            controlValue stage.sweep == some 1.5 &&
            controlValue stage.rate == some 0.2 &&
            controlValue stage.mix == some 0.5 &&
            branch.controls.size == 4
      | _ => false
  | _ => false

private def modalMixOrderCheck : Bool :=
  let graph : PatchGraph :=
    { nodes := #[
        { id := "a", node := .modalSource #[sourceMode 3] (lit 0) clockLit none },
        { id := "b", node := .modalSource #[sourceMode 7] (lit 0) clockLit none },
        { id := "mix", node := .modalMix #["b", "a"] },
        { id := "phaser", node := testPhaser "mix" }]
      output := "phaser" }
  let rankOf := fun id => if id == "a" then some 0 else if id == "b" then some 1
    else if id == "mix" then some 2 else if id == "phaser" then some 3 else none
  match lowerModal graph rankOf "phaser" 3 with
  | .ok #[first, second] =>
      let sigmaOf := fun branch => match branch.source with
        | .plain modes => modes[0]?.bind (sigConstF? ·.sigma)
        | _ => none
      sigmaOf first == some 7.0 && sigmaOf second == some 3.0 &&
        first.stages.size == 1 && second.stages.size == 1
  | _ => false

private def refusalChecks : Bool :=
  let bloomed : PatchGraph :=
    { nodes := #[
        { id := "bloom", node := .modalSource #[sourceMode] (lit 0) clockLit none
            none (some (0.1, 1.8)) },
        { id := "phaser", node := testPhaser "bloom" }]
      output := "phaser" }
  let badRatios : PatchGraph :=
    { nodes := #[
        { id := "source", node := .modalSource #[sourceMode] (lit 0) clockLit none },
        { id := "phaser", node := .modalPhaser "source"
            (constantControl (lit 700)) (constantControl (lit 1))
            (constantControl (lit 1)) (constantControl (lit 5 1))
            #[1.0, 1.0, 2.0, 3.0, 4.0, 5.0] }]
      output := "phaser" }
  let bloomOk := match lowerGraph bloomed with
    | .error error => error == "lower: bloomed phaser crossing at 'phaser' refused (the live all-pass/Gamma crossing is not implemented)"
    | .ok _ => false
  let ranks := fun id => if id == "source" then some 0 else if id == "phaser" then some 1 else none
  let ratiosOk := match lowerModal badRatios ranks "phaser" 1 with
    | .error error => error == "modal phaser 'phaser': expected six distinct positive structural ratios"
    | .ok _ => false
  bloomOk && ratiosOk

private def phaserPatchJson : String :=
  "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"ph\",\"kind\":\"phaser\",\"params\":{\"center\":700,\"sweep\":1.5,\"rate\":0.2,\"mix\":0.5},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"ph\"]}}],\"out\":\"out\"}"

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
  let structural := deferredStructureCheck
  let forestOrder := modalMixOrderCheck
  let refusals := refusalChecks
  IO.eprintln "        rendering independent-oracle fixture"
  let numeric ← renderGraph arena "modal_phaser_oracle" phaserGraph
  let fused ← fusedProductError arena
  IO.eprintln "        compiling and driving four live controls"
  let live ← surfaceAndLiveCheck arena resolved
  IO.eprintln "        compiling canonical two-room product scratch fixture"
  let product := productScratchCheck arena resolved
  match numeric, fused, live, product with
  | .ok samples, .ok fusedError, .ok controlsLive, .ok (scratch, baseline) =>
      let oracleError := maximumOracleError samples
      IO.println s!"        deferred={structural} forest-order={forestOrder} refusals={refusals} oracle max abs={oracleError}"
      IO.println s!"        fused two-room JIT vs generic max abs={fusedError} ({fusedError * 1.0e9}e-9)"
      IO.println s!"        four served controls live without relower={controlsLive}; canonical 6→32→6-section→32 Metal scratch={scratch.total}/24576 (arrays={scratch.arrays}, max-routes={scratch.maxRoutedRecords}×4, slots={scratch.arraySlots}/{scratch.coeffArraySlots} coeff)"
      IO.println s!"        two-room baseline scratch={baseline.total} (arrays={baseline.arrays}, max-routes={baseline.maxRoutedRecords}×4, slots={baseline.arraySlots}/{baseline.coeffArraySlots} coeff)"
      IO.println s!"        product non-coeff array floats={repr scratch.nonCoeffSizes}; baseline={repr baseline.nonCoeffSizes}"
      if structural && forestOrder && refusals && oracleError < 2.0e-5 &&
          -- The two exact schedules sum the same analytic rows in different
          -- routed orders.  This absolute lens is below two accumulated
          -- Q4.28 quanta per participating row and remains meaningful at
          -- dry/wet cancellation zeros where relative error is not.
          fusedError < 2.0e-7 &&
          controlsLive && scratch.total ≤ 24576 then
        passGate "modal-phaser"
          "deferred current-universe stage; independent rational render oracle; authored modalMix order; live center/sweep/rate/mix; named bloom/collision refusals; exact two-room product stays inside Metal scratch policy"
      else
        failGate "modal-phaser" "structural, numerical, live-control, or product scratch contract failed"
  | .error error, _, _, _ => failGate "modal-phaser" s!"render: {firstLine error}"
  | _, .error error, _, _ => failGate "modal-phaser" s!"fused: {firstLine error}"
  | _, _, .error error, _ => failGate "modal-phaser" s!"surface/live: {firstLine error}"
  | _, _, _, .error error => failGate "modal-phaser" s!"product: {firstLine error}"

end Tropical.Tropicaltest.Phaser
