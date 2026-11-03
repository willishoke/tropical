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
  -- The velocity is a Q·.32 scale, independent of the clock rail's format.
  let twoPow32 ← lit 4294967296
  let velocityFixed ← mul velocity twoPow32
  let velocityQ ← toIntE velocityFixed
  let sr ← sampleRate
  let tauSamples ← mul tauBase sr
  let tauFixed ← mul tauSamples (← clockOneLit)
  let tauQ ← toIntE tauFixed
  -- `clock = tau_base + velocity · τ` on the clock rail, exact at fractional τ.
  let advancing ← scaleClockQ velocityQ (← tau)
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

/-- A generator control: a wire from a knob reads the knob's own slot (a fixed,
    τ-invariant expression, exactly as before); a wire from any other Sig node
    is a signal input lowered as a term; no wire reads this node's own param
    (or its default). `knobSlot` names a knob node's slot, whatever its param
    is called (a knob inside a module carries a `param_aliases` name). -/
private def controlOf (pidx : String → Option Nat) (paramName : String → String)
    (knobSlot : String → Option Nat)
    (kind : String) (params inObj : Json) (kname : String) : BuildM ControlIn := do
  match (Tropical.Playground.Metadata.portSources inObj kname)[0]? with
  | some wire =>
    match knobSlot wire with
    | some i => pure (.fixed (← paramRef ⟨i⟩))
    | none => pure (.wire wire)
  | none => pure (.fixed (← paramValue pidx paramName kind params kname))

/-- `knobSlot` for a node set: the slot of a `knob` node's `value`, under its
    aliased name if it has one. -/
def knobSlotOf (pidx : String → Option Nat)
    (raws : Array Tropical.Playground.Metadata.Raw) (id : String) : Option Nat := do
  let raw ← raws.find? (·.id == id)
  if raw.kind != "knob" then none
  pidx (Tropical.Playground.Metadata.paramNameOf raw "value")

/-- The filter's pole pair from its controls, each clamped to its knob range
    (`displayRangeOf`) in both the knob and the signal form: the pair exists
    only for a positive cutoff and a resonance above ≈ −0.02 (q > ½), and a
    cutoff past Nyquist folds, so the knob range is the filter's domain — as
    reverb's rt60 is clamped to its range. -/
private def boundedFilterPair (cutoff resonance : Sig) : BuildM (Array ModalMode) := do
  let bounded := fun (value : Sig) (name : String) => do
    match Tropical.Playground.Metadata.displayRangeOf "filter" name with
    | some (lo, hi) => clampE value (← litF lo) (← litF hi)
    | none => pure value
  filterPair (← bounded cutoff "cutoff") (← bounded resonance "resonance")

def buildNodeWithParamNames (pidx : String → Option Nat)
    (paramName : String → String) (id kind : String)
    (_sel params inObj : Json)
    (knobSlot : String → Option Nat := fun wire => pidx s!"{wire}.value") :
    BuildM (Node × Array PatchNode) := do
  let controlOf := controlOf pidx paramName knobSlot
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
    let freqC ← controlOf kind params inObj "freq"
    let morphC ← controlOf kind params inObj "morph"
    let anchor : Option Anchor ← match pidx s!"{paramName "freq"}#phase" with
      | some i => do
        let phase ← paramRef ⟨i⟩
        pure (some (phase, pitchDefault))
      | none => pure none
    let pmWire := (Tropical.Playground.Metadata.portSources inObj "pm")[0]?
    -- A signal-driven pitch or morph: the control cones are read under the
    -- generator's own clock (`genC`), a `pm` modulator as one more control
    -- (its phase offset `0.3·m`, as on the knob path). The anchor applies
    -- only when the freq is this node's own knob — a `#phase` companion
    -- compensates a STEP of a knob. An expression pitch is not anchored: its
    -- phase is `inc(τ)·τ`, the current increment times the clock, not the
    -- integral of the pitch, so a pitch that moves over time moves the phase
    -- by `Δinc·τ` as well.
    if freqC.isWire || morphC.isWire then
      let zero ← lit 0
      let depth ← lit 3 1
      let controls := #[freqC, morphC] ++ (pmWire.map (#[ControlIn.wire ·])).getD #[]
      let mk := fun (cs : Array Sig) =>
        let pitch := (cs[0]?).getD pitchDefault
        let morph := (cs[1]?).getD zero
        match cs[2]? with
        | none => voiceOf pitch morph anchor
        | some m =>
          let phase := (anchor.map (·.1)).getD zero
          (voiceOf pitch morph (some (phase, pitchDefault))).withPhaseOffset (mul depth m)
      return (.sourceC mk (fun _ raw => pure raw) clk controls, #[])
    let .fixed pitchE := freqC | throw "source: unreachable (knob form with a wired freq)"
    let .fixed morphE := morphC | throw "source: unreachable (knob form with a wired morph)"
    match pmWire with
    | some modId =>
      let pmAnchor ← match anchor with
        | some value => pure value
        | none => do
          let zero ← lit 0
          pure (zero, pitchDefault)
      let depth ← lit 3 1
      pure (.pm modId (voiceOf pitchE morphE (some pmAnchor)) clk depth, #[])
    | none => pure (.source (voiceOf pitchE morphE anchor) clk, #[])
  | "sine" | "saw" | "pulse" =>
    -- `sine` is the exact `FixedSinOsc`; `saw`/`pulse` are polyBLEP waveforms:
    -- the exact Q32 phasor, then a pure function of (phase, increment[, width]).
    -- Anchored like `source` when the freq is this node's own knob.
    let pitchDefault ← defaultValue kind params "freq"
    let freq ← controlOf kind params inObj "freq"
    let phase? ← match pidx s!"{paramName "freq"}#phase" with
      | some i => pure (some (← paramRef ⟨i⟩))
      | none => pure none
    if kind == "sine" then
      let mk := fun (cs : Array Sig) =>
        sineVoiceE ((cs[0]?).getD pitchDefault) (phase?.map (·, pitchDefault))
      return (.sourceC mk (fun _ raw => pure raw) clk #[freq], #[])
    let mk := fun (cs : Array Sig) => phasorVoiceE ((cs[0]?).getD pitchDefault) phase?
    if kind == "saw" then
      let post := fun (cs : Array Sig) (p : Sig) => do
        blepSaw p (← phaseIncrementOf ((cs[0]?).getD pitchDefault))
      pure (.sourceC mk post clk #[freq], #[])
    else
      let width ← controlOf kind params inObj "width"
      let post := fun (cs : Array Sig) (p : Sig) => do
        let half ← lit 5 1
        blepPulse p (← phaseIncrementOf ((cs[0]?).getD pitchDefault)) ((cs[1]?).getD half)
      pure (.sourceC mk post clk #[freq, width], #[])
  | "pluck" =>
    let pitchDefault ← defaultValue kind params "freq"
    let freqC ← controlOf kind params inObj "freq"
    let anchor : Option Anchor ← match pidx s!"{paramName "freq"}#phase" with
      | some i => do
        let phase ← paramRef ⟨i⟩
        pure (some (phase, pitchDefault))
      | none => pure none
    let morphC ← controlOf kind params inObj "morph"
    let eventC ← controlOf kind params inObj "event_rate"
    if freqC.isWire || morphC.isWire || eventC.isWire then
      let morphDefault ← defaultValue kind params "morph"
      let eventDefault ← defaultValue kind params "event_rate"
      let mk := fun (cs : Array Sig) =>
        pluckedVoiceE ((cs[0]?).getD pitchDefault) ((cs[1]?).getD morphDefault)
          ((cs[2]?).getD eventDefault) anchor
      return (.sourceC mk (fun _ raw => pure raw) clk #[freqC, morphC, eventC], #[])
    let .fixed pitchE := freqC | throw "pluck: unreachable (knob form with a wired freq)"
    let .fixed morphE := morphC | throw "pluck: unreachable (knob form with a wired morph)"
    let .fixed eventE := eventC | throw "pluck: unreachable (knob form with a wired event_rate)"
    pure (.source (pluckedVoiceE pitchE morphE eventE anchor) clk, #[])
  | "comb" =>
    let delayC ← controlOf kind params inObj "delay"
    -- (The knob form's expressions are built in their original order.)
    let delay? ← match delayC with
      | .fixed seconds => some <$> deltaOf seconds
      | .wire _ => pure none
    let decayC ← controlOf kind params inObj "decay"
    if delayC.isWire || decayC.isWire then
      -- Signal controls: tap k reads the input at `c + k·Δ(s)` and weighs it
      -- `gᵏ`, `s` and `g` the control terms at this node's clock.
      let body := fun (ins cs : Array ArrowTerm) => do
        let input := (ins[0]?).getD (.sum #[])
        let delayT := (cs[0]?).getD (.sum #[])
        let decayT := (cs[1]?).getD (.sum #[])
        let one ← lit 1
        let tail ← (Array.range 6).mapM fun j => do
          let k := j + 1
          let kSig ← lit (Int.ofNat k)
          let weight : ArrowTerm ← match decayT with
            | .konst gain => pure (.konst (← gPow gain k))
            | other => pure (.arrUn (fun gain => gPow gain k) other)
          let tap := ArrowTerm.warpedBy (fun c seconds => do
            add c (← mul kSig (← deltaOf seconds))) delayT input
          pure (ArrowTerm.scaledBy weight tap)
        pure (.sum (#[.scale one input] ++ tail))
      return (.effectC #[signalInput] #[delayC, decayC] body, signalFanIn)
    let some delay := delay? | throw "comb: unreachable (knob form with a wired delay)"
    let .fixed gain := decayC | throw "comb: unreachable (knob form with a wired decay)"
    let tail ← (Array.range 6).mapM fun j => do
      let k := j + 1
      let weight ← gPow gain k
      let kSig ← lit (Int.ofNat k)
      let offset ← mul kSig delay
      pure (weight, fun c => add c offset)
    let one ← lit 1
    pure (.comb signalInput (#[(one, fun c => pure c)] ++ tail), signalFanIn)
  | "flange" =>
    let depthC ← controlOf kind params inObj "depth"
    if depthC.isWire then
      -- A signal depth: `½x + ¼x(τ − d) + ¼x(τ + d)`, `d` the depth term in
      -- seconds at this node's clock (either sign).
      let body := fun (ins cs : Array ArrowTerm) => do
        let input := (ins[0]?).getD (.sum #[])
        let depthT := (cs[0]?).getD (.sum #[])
        let half ← lit 5 1
        let quarter ← lit 25 2
        pure (.sum #[
          .scale half input,
          .scale quarter (ArrowTerm.warpedBy delayWarp depthT input),
          .scale quarter (ArrowTerm.warpedBy advanceWarp depthT input)])
      return (.effectC #[signalInput] #[depthC] body, signalFanIn)
    let .fixed seconds := depthC | throw "flange: unreachable (knob form with a wired depth)"
    let delay ← deltaOf seconds
    pure (.flange signalInput (fun c => sub c delay) (fun c => add c delay), signalFanIn)
  | "delay" =>
    -- A signal-driven time: any wire (an LFO, a knob, a sum), either sign.
    if let some wire := (Tropical.Playground.Metadata.portSources inObj "amount")[0]? then
      return (.sdelay signalInput wire, signalFanIn)
    let seconds ← paramValue pidx paramName kind params "amount"
    let delay ← deltaOf seconds
    pure (.warpFx signalInput (fun c => sub c delay), signalFanIn)
  | "reverse" => pure (.warpFx signalInput neg, signalFanIn)
  | "fm" =>
    let carrierDefault ← defaultValue kind params "carrier"
    let carrierC ← controlOf kind params inObj "carrier"
    let anchor : Option Anchor ← match pidx s!"{paramName "carrier"}#phase" with
      | some i => do
        let phase ← paramRef ⟨i⟩
        pure (some (phase, carrierDefault))
      | none => pure none
    let depthC ← controlOf kind params inObj "depth"
    if carrierC.isWire || depthC.isWire then
      -- Signal controls: the carrier is a control-driven sine (`genC`) under
      -- the FM warp, so its freq cone is read at the warped clock with it; the
      -- depth, like the modulator, at this node's clock.
      let body := fun (ins cs : Array ArrowTerm) => do
        let modulator := (ins[0]?).getD (.sum #[])
        let carrierT := (cs[0]?).getD (.konst carrierDefault)
        let depthT := (cs[1]?).getD (.sum #[])
        let carrier : ArrowTerm := .genC
          (fun signals => sineVoiceE ((signals[0]?).getD carrierDefault) anchor)
          (fun _ raw => pure raw) id clk #[carrierT]
        match depthT with
        | .konst depth => pure (.swarp (fmWarp depth) modulator carrier)
        | other => pure (.swarp fmWarpScaled (.prod other modulator) carrier)
      return (.effectC #[signalInput] #[carrierC, depthC] body, signalFanIn)
    let .fixed carrierE := carrierC | throw "fm: unreachable (knob form with a wired carrier)"
    let .fixed depthE := depthC | throw "fm: unreachable (knob form with a wired depth)"
    pure (.fm signalInput (sineVoiceE carrierE anchor) clk depthE, signalFanIn)
  | "sflange" =>
    -- A convenience module over delay/ring/mix (`sflangeTopology`); with no
    -- `mod` wire, its own sine LFO at `rate` sweeps it.
    let depth ← controlOf kind params inObj "depth"
    let dry ← controlOf kind params inObj "dry"
    let wet ← controlOf kind params inObj "wet"
    let (modId, lfo) ← match (Tropical.Playground.Metadata.portSources inObj "mod")[0]? with
      | some modId => pure (modId, #[])
      | none => do
        let rate ← paramValue pidx paramName kind params "rate"
        let lfoId := s!"__lfo_{id}"
        pure (lfoId, #[{ id := lfoId, node := .source (sineVoiceE rate) clk }])
    let (node, topology) := sflangeTopology id signalInput modId depth dry wet
    pure (node, signalFanIn ++ lfo ++ topology)
  | "mix" => pure (.mix (Tropical.Playground.Metadata.portSources inObj "in"), #[])
  | "ring" => pure (.ring (Tropical.Playground.Metadata.portSources inObj "in"), #[])
  | "resonator" =>
    let f0 ← paramValue pidx paramName kind params "freq"
    let decay ← paramValue pidx paramName kind params "decay"
    let npart := (Tropical.Playground.Metadata.jInt params "partials" 6).toNat
    let address := (Tropical.Playground.Metadata.portSources inObj "addr")[0]?
    -- The strike time `t` in seconds, as `string`/`gong` read it; absent is
    -- a strike at 0 (the literal, so graphs without `t` emit unchanged).
    let zero ← match Tropical.Playground.Metadata.jNum? params "t" with
      | none => lit 0
      | some _ => do
        let seconds ← litOfD (Tropical.Playground.Metadata.jExactD params "t" (0, 0))
        mul seconds (← sampleRate)
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
    let cutoffC ← controlOf kind params inObj "cutoff"
    let resonanceC ← controlOf kind params inObj "resonance"
    if cutoffC.isWire || resonanceC.isWire then
      -- Signal controls, frozen at the response coordinate like the phaser's
      -- sweep: the pair is rebuilt from the controls' values there, clamped to
      -- the knob range as the knob form is (`boundedFilterPair`).
      let ref := fun (c : ControlIn) => match c with
        | .fixed value => ModalControlRef.constant value
        | .wire wire => { fallback := .konst zero, signalNode? := some wire }
      let stage : ModalLinearStage := {
        controls := #[ModalControlRef.constant zero, ref cutoffC, ref resonanceC]
        build := fun _ values => do
          let value := fun (i : Nat) => match values[i]? with
            | some value => pure value
            | none => lit 0
          let modes ← boundedFilterPair (← value 1) (← value 2)
          let direction ← match values[0]? with
            | some value => pure value
            | none => lit 0
          pure (.proper (.oriented modes direction)) }
      return (.modalLinear modalInput stage, modalFanIn)
    let .fixed cutoff := cutoffC | throw "filter: unreachable (knob form with a wired cutoff)"
    let .fixed resonance := resonanceC | throw "filter: unreachable (knob form with a wired resonance)"
    let stage : ModalLinearStage := {
      controls := #[ModalControlRef.constant zero]
      build := fun _ values => do
        let modes ← boundedFilterPair cutoff resonance
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
    let (node, topology) :=
      modalPhaserTopology id modalInput center sweep rate mix modalPhaserRatios
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
      raw.id raw.kind raw.sel raw.params raw.inObj (knobSlotOf pidx raws)
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
