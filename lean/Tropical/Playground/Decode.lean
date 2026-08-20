import Tropical.Playground.Vocabulary
import Tropical.Playground.DecodeMetadata

/-!
# Playground.Decode

Production patch decoding into the active arena-native builder.  JSON shape,
parameter-table policy, and graph topology are shared with the public
playground vocabulary; only expression-bearing construction lives here.
-/

namespace Tropical.Playground.Compiler

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Playground.Metadata

def masterClock (pidx : String → Option Nat) : BuildM Clock := do
  let one ← lit 1
  let zero ← lit 0
  let velocity ← pref pidx Tropical.Playground.Metadata.masterVelocityParam one
  let tauBase ← pref pidx Tropical.Playground.Metadata.masterTauBaseParam zero
  let twoPow32 ← lit 4294967296
  let velocityFixed ← mul velocity twoPow32
  let velocityQ ← toIntE velocityFixed
  let sr ← sampleRate
  let tauSamples ← mul tauBase sr
  let tauFixed ← mul tauSamples twoPow32
  let tauQ ← toIntE tauFixed
  let coordinate ← sampleIndex
  let advancing ← mul velocityQ coordinate
  add tauQ advancing

private def defaultValue (kind : String) (params : Json) (kname : String) : BuildM Sig := do
  let fallback ← fallbackOf kind kname
  jExpr params kname fallback

private def paramValue (pidx : String → Option Nat) (paramName : String → String)
    (kind : String) (params : Json) (kname : String) : BuildM Sig := do
  let dflt ← defaultValue kind params kname
  if Tropical.Playground.Metadata.isGlided kind kname then
    glideExpr pidx (paramName kname) dflt
  else
    pref pidx (paramName kname) dflt

def buildNodeWithParamNames (pidx : String → Option Nat)
    (paramName : String → String) (id kind : String)
    (_sel params inObj : Json) : BuildM (Node × Array PatchNode) := do
  let implicitFanIn : Bool → String × Array PatchNode := fun modal =>
    let inputs := Tropical.Playground.Metadata.portSources inObj "in"
    match inputs.toList with
    | [] => ("__silence__", #[])
    | [input] => (input, #[])
    | _ =>
      let domain := if modal then "modal" else "signal"
      let helperId := s!"__fanin_{domain}_{id}_in"
      let helperNode := if modal then Node.modalMix inputs else Node.mix inputs
      (helperId, #[{ id := helperId, node := helperNode }])
  let (signalInput, signalFanIn) := implicitFanIn false
  let (modalInput, modalFanIn) := implicitFanIn true
  let clk ← masterClock pidx
  let modalControl (kname : String) : BuildM ModalControlRef := do
    let dflt ← defaultValue kind params kname
    let fallback : ArrowTerm ←
      if Tropical.Playground.Metadata.isGlided kind kname then
        pure ((.arrUn (fun clkQ => glideExprQAt pidx (paramName kname) dflt clkQ)
          (.clk clk)) : ArrowTerm)
      else do
        let value ← pref pidx (paramName kname) dflt
        pure ((.konst value) : ArrowTerm)
    pure ({ fallback := fallback
            signalNode? := (Tropical.Playground.Metadata.portSources inObj kname)[0]? } : ModalControlRef)
  match kind with
  | "knob" =>
    match pidx (paramName "value") with
    | some i => pure (.knob i, #[])
    | none => pure (.mix #[], #[])
  | "source" =>
    let pitchDefault ← defaultValue kind params "freq"
    let pitchE ← match (Tropical.Playground.Metadata.portSources inObj "freq")[0]? with
      | some wire => pref pidx s!"{wire}.value" pitchDefault
      | none => paramValue pidx paramName kind params "freq"
    let morphE ← paramValue pidx paramName kind params "morph"
    let anchor : Option Anchor ← match pidx s!"{paramName "freq"}#phase" with
      | some i => do
        let phase ← paramRef ⟨i⟩
        pure (some (phase, pitchDefault))
      | none => pure none
    match (Tropical.Playground.Metadata.portSources inObj "pm")[0]? with
    | some modId =>
      let pmAnchor ← match anchor with
        | some value => pure value
        | none => do
          let zero ← lit 0
          pure (zero, pitchDefault)
      let depth ← lit 3 1
      pure (.pm modId (voiceOf pitchE morphE (some pmAnchor)) clk depth, #[])
    | none => pure (.source (voiceOf pitchE morphE anchor) clk, #[])
  | "pluck" =>
    let pitchDefault ← defaultValue kind params "freq"
    let pitchE ← match (Tropical.Playground.Metadata.portSources inObj "freq")[0]? with
      | some wire => pref pidx s!"{wire}.value" pitchDefault
      | none => paramValue pidx paramName kind params "freq"
    let anchor : Option Anchor ← match pidx s!"{paramName "freq"}#phase" with
      | some i => do
        let phase ← paramRef ⟨i⟩
        pure (some (phase, pitchDefault))
      | none => pure none
    let morph ← paramValue pidx paramName kind params "morph"
    let eventRate ← paramValue pidx paramName kind params "event_rate"
    pure (.source (pluckedVoiceE pitchE morph eventRate anchor) clk, #[])
  | "comb" =>
    let seconds ← paramValue pidx paramName kind params "delay"
    let delay ← deltaOf seconds
    let gain ← paramValue pidx paramName kind params "decay"
    let tail ← (Array.range 6).mapM fun j => do
      let k := j + 1
      let weight ← gPow gain k
      let kSig ← lit (Int.ofNat k)
      let offset ← mul kSig delay
      pure (weight, fun c => add c offset)
    let one ← lit 1
    pure (.comb signalInput (#[(one, fun c => pure c)] ++ tail), signalFanIn)
  | "flange" =>
    let seconds ← paramValue pidx paramName kind params "depth"
    let delay ← deltaOf seconds
    pure (.flange signalInput (fun c => sub c delay) (fun c => add c delay), signalFanIn)
  | "delay" =>
    let seconds ← paramValue pidx paramName kind params "amount"
    let delay ← deltaOf seconds
    pure (.warpFx signalInput (fun c => sub c delay), signalFanIn)
  | "reverse" => pure (.warpFx signalInput neg, signalFanIn)
  | "fm" =>
    let carrierDefault ← defaultValue kind params "carrier"
    let carrier ← paramValue pidx paramName kind params "carrier"
    let anchor : Option Anchor ← match pidx s!"{paramName "carrier"}#phase" with
      | some i => do
        let phase ← paramRef ⟨i⟩
        pure (some (phase, carrierDefault))
      | none => pure none
    let depth ← paramValue pidx paramName kind params "depth"
    pure (.fm signalInput (sineVoiceE carrier anchor) clk depth, signalFanIn)
  | "sflange" =>
    let depth ← paramValue pidx paramName kind params "depth"
    match (Tropical.Playground.Metadata.portSources inObj "mod")[0]? with
    | some modId => pure (.sflange signalInput modId depth, signalFanIn)
    | none =>
      let rate ← paramValue pidx paramName kind params "rate"
      let lfoId := s!"__lfo_{id}"
      pure (.sflange signalInput lfoId depth,
        signalFanIn ++ #[{ id := lfoId, node := .source (sineVoiceE rate) clk }])
  | "mix" => pure (.mix (Tropical.Playground.Metadata.portSources inObj "in"), #[])
  | "ring" => pure (.ring (Tropical.Playground.Metadata.portSources inObj "in"), #[])
  | "resonator" =>
    let f0 ← paramValue pidx paramName kind params "freq"
    let decay ← paramValue pidx paramName kind params "decay"
    let npart := (Tropical.Playground.Metadata.jInt params "partials" 6).toNat
    let address := (Tropical.Playground.Metadata.portSources inObj "addr")[0]?
    let zero ← lit 0
    match Tropical.Playground.Metadata.jNum? params "partials_max" with
    | none =>
      let modes ← resonatorBank f0 decay npart
      pure (.modalSource modes zero clk address, #[])
    | some _ =>
      let capacity := (Tropical.Playground.Metadata.jInt params "partials_max" 6).toNat
      let countDefault ← lit (Int.ofNat npart)
      let count ← pref pidx (paramName "partials") countDefault
      let modes ← resonatorBank f0 decay capacity
      pure (.modalSource modes zero clk address (some count), #[])
  | "reverb" =>
    let rt60 ← modalControl "rt60"
    let rtRange := Tropical.Playground.Metadata.displayRangeOf "reverb" "rt60"
    let direction ← modalControl "dir"
    let build := fun frozenRt60 => do
      let bounded ← match rtRange with
        | some (lo, hi) => do
          let lo ← litF lo
          let hi ← litF hi
          clampE frozenRt60 lo hi
        | none => pure frozenRt60
      reverbRoom bounded rtRange 14 (60, 0) (6000, 0)
    pure (.modalRoom modalInput build rt60 direction, modalFanIn)
  | "filter" =>
    let zero ← lit 0
    let cutoff ← paramValue pidx paramName kind params "cutoff"
    let resonance ← paramValue pidx paramName kind params "resonance"
    let stage : ModalLinearStage := {
      controls := #[ModalControlRef.constant zero]
      build := fun _ values => do
        let modes ← filterPair cutoff resonance
        let direction ← match values[0]? with
          | some value => pure value
          | none => lit 0
        pure (.proper (.oriented modes direction)) }
    pure (.modalLinear modalInput stage, modalFanIn)
  | "phaser" =>
    let center ← modalControl "center"
    let sweep ← modalControl "sweep"
    let rate ← modalControl "rate"
    let mix ← modalControl "mix"
    let requestedStages :=
      (Tropical.Playground.Metadata.jInt params "_benchmark_stages" 0).toNat
    let ratios := if Tropical.Ir.phaserTimeStagingEnabled
        && requestedStages ≥ 2 && requestedStages ≤ 18 then
      modalPhaserBenchmarkRatios requestedStages
    else modalPhaserRatios
    let (node, topology) := modalPhaserTopology id modalInput center sweep rate mix ratios
    pure (node, modalFanIn ++ topology)
  | "modal_allpass_tail" =>
    let center ← modalControl "center"
    let sweep ← modalControl "sweep"
    let rate ← modalControl "rate"
    pure (.modalLinear modalInput (modalAllpassTailStage center sweep rate
      (Tropical.Playground.Metadata.jFloat params "ratio" 1.0)), modalFanIn)
  | "modalblend" =>
    let dry := ((Tropical.Playground.Metadata.portSources inObj "dry")[0]?).getD "__silence__"
    let wet := ((Tropical.Playground.Metadata.portSources inObj "wet")[0]?).getD "__silence__"
    let mix ← modalControl "mix"
    pure (.modalBlend dry wet mix, #[])
  | "modalmix" => pure (.modalMix (Tropical.Playground.Metadata.portSources inObj "in"), #[])
  | "gauge" =>
    let gauge ← modalControl "g"
    pure (.modalGaugeControl modalInput gauge, modalFanIn)
  | "gong" =>
    let anchorSeconds ← litOfD (Tropical.Playground.Metadata.jExactD params "t" (0, 0))
    let sr ← sampleRate
    let anchor ← mul anchorSeconds sr
    let fullFromJson ← jModes params "modes_full"
    let halfFromJson ← jModes params "modes_half"
    let (full, half) ← if fullFromJson.isEmpty && halfFromJson.isEmpty then
        defaultGongModes (Tropical.Playground.Metadata.jFloat params "freq" 110.0)
      else pure (fullFromJson, halfFromJson)
    let beta ← paramValue pidx paramName kind params "beta"
    let settle ← litOfD (Tropical.Playground.Metadata.jExactD params "g" (18, 1))
    pure (gongStrikeNodes id clk anchor beta settle full half)
  | "bloomgong" =>
    let beta := Tropical.Playground.Metadata.jFloat params "beta" 0.05
    let settle := Tropical.Playground.Metadata.jFloat params "g" 1.8
    let scale := Tropical.Playground.Metadata.jFloat params "scale" 1.0
    let decoded ← jModes params "modes"
    let modes ← if decoded.isEmpty then do
        let (full, _) ← defaultGongModes
          (Tropical.Playground.Metadata.jFloat params "freq" 110.0)
        pure full
      else pure decoded
    let anchorSeconds ← litOfD (Tropical.Playground.Metadata.jExactD params "t" (0, 0))
    let sr ← sampleRate
    let anchor ← mul anchorSeconds sr
    pure (.modalSource modes anchor clk none none (some (beta * scale / settle, settle)), #[])
  | "string" =>
    let decoded ← jModes params "modes"
    let modes ← if decoded.isEmpty then
        defaultStringModes
          (Tropical.Playground.Metadata.jDec params "freq" (196, 0))
          (Tropical.Playground.Metadata.jDec params "decay" (996, 3))
      else pure decoded
    let anchorSeconds ← litOfD (Tropical.Playground.Metadata.jExactD params "t" (0, 0))
    let sr ← sampleRate
    let anchor ← mul anchorSeconds sr
    let address := (Tropical.Playground.Metadata.portSources inObj "addr")[0]?
    pure (.modalSource modes anchor clk address, #[])
  | _ => pure (.mix (Tropical.Playground.Metadata.portSources inObj "in"), #[])

def buildNode (pidx : String → Option Nat) (id kind : String)
    (sel params inObj : Json) : BuildM (Node × Array PatchNode) :=
  buildNodeWithParamNames pidx (fun knob => s!"{id}.{knob}")
    id kind sel params inObj

/-- Decode graph topology and every scalar-bearing node in one active builder. -/
def decodeGraph (j : Json) (params : Array (String × JsonNumber)) : BuildM PatchGraph := do
  let outId := match (j.getObjVal? "out").toOption with
    | some (.str value) => value
    | _ => ""
  let raws := Tropical.Playground.Metadata.rawsOf j
  let pidx : String → Option Nat := fun name => params.findIdx? (·.1 == name)
  let mut nodes : Array PatchNode := #[]
  for raw in raws do
    if raw.kind == "out" then continue
    let (node, extras) ← buildNodeWithParamNames pidx
      (Tropical.Playground.Metadata.paramNameOf raw)
      raw.id raw.kind raw.sel raw.params raw.inObj
    nodes := nodes.push { id := raw.id, node }
    for extra in extras do
      nodes := nodes.push extra
  let outputInputs := match raws.find? (·.id == outId) with
    | some raw => if raw.kind == "out" then
        Tropical.Playground.Metadata.portSources raw.inObj "in" else #[raw.id]
    | none => #[]
  let masterIdx := (pidx Tropical.Playground.Metadata.masterGainParam).getD 0
  nodes := nodes.push { id := "__mixbus__", node := .mix outputInputs }
  nodes := nodes.push { id := "__master__", node := .knob masterIdx }
  nodes := nodes.push { id := "__out__", node := .ring #["__mixbus__", "__master__"] }
  nodes := nodes.push { id := "__silence__", node := .mix #[] }
  pure { nodes, output := "__out__" }

end Tropical.Playground.Compiler
