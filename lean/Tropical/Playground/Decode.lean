import Tropical.Playground.Vocabulary
import Tropical.Playground.DecodeMetadata
import Tropical.Plan

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

/-- The refusal for a time-varying wire into a uniform inlet (a frequency or
    rate, `Metadata.isUniform`). The generator's phase is `inc·τ`, the
    increment times the clock: exact for a constant increment, and for one that
    moves over time not the integral of the frequency, so the phase drifts by
    `Δinc·τ` — without bound as τ grows. `label` names the node as authored
    (`nodeLabel`). -/
def uniformRefusal (label kind port source : String) : String :=
  let advice := match kind with
    | "phaser" | "modal_allpass_tail" =>
      " To move the sweep with a signal, wire it into `center` or `sweep`."
    | "harmonic" =>
      if port == "partials" then
        " The partial count is the family's trip count, fixed within a call. To move the timbre with a signal, wire it into `tilt` or `even`, or into an envelope's `cutoff`."
      else
        " A moving fundamental is a phase-modulated family, not yet supported (S5). To modulate pitch, pass the harmonic (or its envelope) through a `delay` whose `amount` is the signal (a clock warp)."
    | "pluck" =>
      if port == "event_rate" then "" else
        " To modulate pitch, wire the signal into a `source`'s `pm` inlet (phase modulation) or into an `fm` node's `in` (frequency modulation, a clock warp)."
    | _ =>
      " To modulate pitch, wire the signal into a `source`'s `pm` inlet (phase modulation) or into an `fm` node's `in` (frequency modulation, a clock warp)."
  let why := if kind == "harmonic" && port == "partials" then
      "The family's region trips this many partials per call."
    else s!"The phase is computed as {port}·τ, which is exact only while {port} is constant."
  s!"time-varying {port}: {label} inlet '{port}' is wired from '{source}', which varies over time. This inlet takes only a value constant within a kernel call: a knob, arithmetic on knobs, or a constant. {why}{advice}"

/-- A generator control: a wire from a knob reads the knob's own slot (a fixed,
    τ-invariant expression, exactly as before); a wire from any other Sig node
    is a signal input lowered as a term — into a uniform inlet, one whose value
    must be constant within a kernel call (`ControlIn.uniform`); no wire reads
    this node's own param (or its default). `knobSlot` names a knob node's
    slot, whatever its param is called (a knob inside a module carries a
    `param_aliases` name). -/
private def controlOf (pidx : String → Option Nat) (paramName : String → String)
    (knobSlot : String → Option Nat) (label : String)
    (kind : String) (params inObj : Json) (kname : String) : BuildM ControlIn := do
  match (Tropical.Playground.Metadata.portSources inObj kname)[0]? with
  | some wire =>
    match knobSlot wire with
    | some i => pure (.fixed (← paramRef ⟨i⟩))
    | none =>
      if Tropical.Playground.Metadata.isUniform kind kname then
        pure (.uniform wire (uniformRefusal label kind kname wire))
      else pure (.wire wire)
  | none => pure (.fixed (← paramValue pidx paramName kind params kname))

/-- `knobSlot` for a node set: the slot of a `knob` node's `value`, under its
    aliased name if it has one. -/
def knobSlotOf (pidx : String → Option Nat)
    (raws : Array Tropical.Playground.Metadata.Raw) (id : String) : Option Nat := do
  let raw ← raws.find? (·.id == id)
  if raw.kind != "knob" then none
  pidx (Tropical.Playground.Metadata.paramNameOf raw "value")

/-- The slot of a knob node's `#phase` companion, present when the knob is
    wired straight into an anchored inlet (`Metadata.knobFeedsAnchor`). -/
def knobPhaseSlotOf (pidx : String → Option Nat)
    (raws : Array Tropical.Playground.Metadata.Raw) (id : String) : Option Nat := do
  let raw ← raws.find? (·.id == id)
  if raw.kind != "knob" then none
  pidx s!"{Tropical.Playground.Metadata.paramNameOf raw "value"}#phase"

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

/-- A `harmonic` node's structural fields (`params`, not slots): the capacity
    `partials_max`, the projection `wave`, and an optional literal base table
    `table`. Every malformed field is an error naming the node. -/
structure HarmonicSpec where
  capacity : Nat
  wave : PartialWave
  table : Option (Array JsonNumber)

def harmonicSpec (label : String) (params : Json) : Except String HarmonicSpec := do
  let table ← match params.getObjVal? "table" with
    | .error _ => pure none
    | .ok (.arr values) =>
      let numbers := values.filterMap (·.getNum?.toOption)
      if numbers.size != values.size || numbers.isEmpty then
        throw s!"{label}: `table` must be a non-empty array of numbers (one weight per partial)"
      pure (some numbers)
    | .ok _ => throw s!"{label}: `table` must be a non-empty array of numbers (one weight per partial)"
  let capacity ← match params.getObjVal? "partials_max" with
    | .error _ => pure ((table.map (·.size)).getD 64)
    | .ok (.num n) =>
      if n.exponent != 0 || n.mantissa < 1 then
        throw s!"{label}: `partials_max` must be a positive integer"
      pure n.mantissa.toNat
    | .ok _ => throw s!"{label}: `partials_max` must be a positive integer"
  if capacity > familyCapacity then
    throw s!"{label}: `partials_max` {capacity} exceeds the family's capacity of {familyCapacity} partials"
  if let some t := table then
    if t.size != capacity then
      throw s!"{label}: `table` has {t.size} weights but `partials_max` is {capacity}"
  let wave ← match params.getObjVal? "wave" with
    | .error _ => pure PartialWave.sine
    | .ok (.str "sine") => pure .sine
    | .ok (.str "cosine") => pure .cosine
    | .ok other => throw s!"{label}: unknown wave {other.compress}; a harmonic's wave is \"sine\" or \"cosine\""
  pure { capacity, wave, table }

/-- The `harmonic` family from its control signals `#[freq, tilt, even,
    partials]`: weights `w_k = base_k · k^(−tilt) · (k even ? even : 1)`.
    Without a table `base_k = 1`, and `k^(−tilt)` is computed as
    `(1/k)·exp((1 − tilt)·ln k)` from exact `1/k` and `ln k` columns, so at
    tilt 1 the weight is the saw's `1/k` bit for bit (`exp 0 = 1` exactly). -/
def harmonicFamily (id : String) (spec : HarmonicSpec) (phase? : Option Sig)
    (clk : Clock) (controls : Array Sig) : BuildM PartialFamily := do
  let control := fun (i : Nat) => match controls[i]? with
    | some value => pure value
    | none => throw s!"harmonic '{id}': unreachable (control {i} missing)"
  let freq ← control 0
  let tilt ← control 1
  let even ← control 2
  let partials ← control 3
  let ks := (Array.range spec.capacity).map (· + 1)
  -- `ln k` in exact arithmetic (the bake layer calls no libm; `exact-corpse`),
  -- rounded once to the double the literal carries exactly.
  let lnK ← ks.mapM fun k => litExact (Tropical.Exact.DyadicI.log (Tropical.Exact.DyadicI.ofNat k)).toFloat
  let (base, tableMode) ← match spec.table with
    | some table => pure (← table.mapM fun n => lit n.mantissa n.exponent, true)
    | none => pure (← ks.mapM fun k => litExact (1.0 / k.toFloat), false)
  let weight := fun (cols ins : Array Sig) (n _kf : Sig) => do
    let baseK := cols[0]!
    let lnKk := cols[1]!
    let tiltNow := ins[0]!
    let evenNow := ins[1]!
    let exponent ← if tableMode then neg tiltNow else sub (← lit 1) tiltNow
    let slope ← expSig (← mul exponent lnKk)
    let w ← mul baseK slope
    let parity ← toFloatE (← bitAnd n (← litI 1))
    let isOdd ← gt parity (← lit 5 1)
    selectE isOdd w (← mul w evenNow)
  pure {
    name := id, freq, phase := phase?, clk, count := spec.capacity
    liveCount := some partials
    weights := .closedForm #[base, lnK] #[tilt, even] weight
    wave := spec.wave }

/-- An `envelope` node's shape (`params.shape`, structural). -/
def envelopeShape (label : String) (params : Json) : Except String SvfResponse :=
  match params.getObjVal? "shape" with
  | .error _ => pure .lowpass
  | .ok (.str "lowpass") => pure .lowpass
  | .ok (.str "bandpass") => pure .bandpass
  | .ok (.str "highpass") => pure .highpass
  | .ok other =>
    throw s!"{label}: unknown shape {other.compress}; an envelope's shape is \"lowpass\", \"bandpass\" or \"highpass\""

def buildNodeWithParamNames (pidx : String → Option Nat)
    (paramName : String → String) (id kind : String)
    (_sel params inObj : Json) (rate : Nat := Tropical.Plan.defaultSampleRate)
    (knobSlot : String → Option Nat := fun wire => pidx s!"{wire}.value")
    (knobPhaseSlot : String → Option Nat := fun wire => pidx s!"{wire}.value#phase")
    (label : String := s!"'{id}' ({kind})") :
    BuildM (Node × Array PatchNode) := do
  let controlOf := controlOf pidx paramName knobSlot label
  -- The `#phase` companion anchoring an anchored inlet: the node's own knob's,
  -- or, when a knob is wired straight in, that knob's. A wired expression has
  -- none (and is constant within a call: a uniform inlet).
  let anchorPhase? (kname : String) : BuildM (Option Sig) := do
    let slot := match (Tropical.Playground.Metadata.portSources inObj kname)[0]? with
      | some wire => knobPhaseSlot wire
      | none => pidx s!"{paramName kname}#phase"
    match slot with
    | some i => some <$> paramRef ⟨i⟩
    | none => pure none
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
    let signalNode? := (Tropical.Playground.Metadata.portSources inObj kname)[0]?
    let uniformRefusal? := if Tropical.Playground.Metadata.isUniform kind kname then
        signalNode?.map (uniformRefusal label kind kname ·)
      else none
    pure ({ fallback, signalNode?, uniformRefusal? } : ModalControlRef)
  match kind with
  | "knob" =>
    match pidx (paramName "value") with
    | some i => pure (.knob i, #[])
    | none => pure (.mix #[], #[])
  | "source" =>
    let pitchDefault ← defaultValue kind params "freq"
    let freqC ← controlOf kind params inObj "freq"
    let morphC ← controlOf kind params inObj "morph"
    let anchor : Option Anchor := (← anchorPhase? "freq").map (·, pitchDefault)
    let pmWire := (Tropical.Playground.Metadata.portSources inObj "pm")[0]?
    -- A wired pitch or morph: the control cones are read under the
    -- generator's own clock (`genC`), a `pm` modulator as one more control
    -- (its phase offset `0.3·m`, as on the knob path). The freq inlet is
    -- uniform: a wired pitch is constant within a kernel call (a time-varying
    -- one is refused, `uniformRefusal`), so the phase `inc·τ` is exact. The
    -- anchor applies when the freq is this node's own knob or a knob wired
    -- straight in — a `#phase` companion compensates a STEP of a knob; an
    -- expression of knobs is not anchored, and a step of it moves the phase
    -- by `Δinc·τ`.
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
    -- Anchored like `source`: by this node's own knob, or a knob wired in.
    let pitchDefault ← defaultValue kind params "freq"
    let freq ← controlOf kind params inObj "freq"
    let phase? ← anchorPhase? "freq"
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
  | "harmonic" =>
    -- One family term (`ArrowTerm.familyC`): its controls are lowered as
    -- terms in this node's clock context; `freq` and `partials` are uniform
    -- inlets, `tilt` and `even` take any wire. Anchored like `saw`.
    let spec ← match harmonicSpec label params with
      | .ok spec => pure spec
      | .error e => throw e
    let freqC ← controlOf kind params inObj "freq"
    let tiltC ← controlOf kind params inObj "tilt"
    let evenC ← controlOf kind params inObj "even"
    let partialsC ← controlOf kind params inObj "partials"
    let phase? ← anchorPhase? "freq"
    let body := fun (_ : Array ArrowTerm) (cs : Array ArrowTerm) =>
      pure (ArrowTerm.familyC (harmonicFamily id spec phase? clk) cs)
    pure (.effectC #[] #[freqC, tiltC, evenC, partialsC] body, #[])
  | "envelope" =>
    -- The envelope wraps the TERM on its `in` wire (`ArrowTerm.envelope`);
    -- `normalize` distributes it to the families below, and emission refuses
    -- anything it cannot sit on, naming this node and wire (`site`).
    let shape ← match envelopeShape label params with
      | .ok shape => pure shape
      | .error e => throw e
    let cutoffC ← controlOf kind params inObj "cutoff"
    let qC ← controlOf kind params inObj "q"
    let sources := Tropical.Playground.Metadata.portSources inObj "in"
    let wired := if sources.isEmpty then "unwired"
      else "wired from " ++ ", ".intercalate (sources.toList.map (s!"'{·}'"))
    let law := { svfEnvelopeLaw shape with site := some s!"{label} inlet 'in' ({wired})" }
    let body := fun (ins cs : Array ArrowTerm) =>
      pure (ArrowTerm.envelope law cs ((ins[0]?).getD (.sum #[])))
    pure (.effectC #[signalInput] #[cutoffC, qC] body, signalFanIn)
  | "pluck" =>
    let pitchDefault ← defaultValue kind params "freq"
    let freqC ← controlOf kind params inObj "freq"
    let anchor : Option Anchor := (← anchorPhase? "freq").map (·, pitchDefault)
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
      | _ => pure none
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
    let anchor : Option Anchor := (← anchorPhase? "carrier").map (·, carrierDefault)
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
        | .wire wire | .uniform wire _ => { fallback := .konst zero, signalNode? := some wire }
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
          (Tropical.Playground.Metadata.jDec params "decay" (996, 3)) rate
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

/-- A node as its author wrote it, for messages: `'id' (kind)`, or for an atom
    a module expanded into (`Playground.Hierarchy`, recorded in `source_map`),
    the module instance the author placed and the atom inside it. -/
def nodeLabel (j : Json) (id kind : String) : String :=
  let entries := match j.getObjVal? "source_map" with
    | .ok (.arr entries) => entries
    | _ => #[]
  let entry? := entries.find? fun entry =>
    match entry.getObjValAs? String "expanded" with
    | .ok expanded => expanded == id
    | .error _ => false
  match entry? with
  | some entry =>
    match entry.getObjValAs? (Array String) "path", entry.getObjValAs? String "local" with
    | .ok path, .ok localId =>
      s!"'{path[0]?.getD id}' (a module; its {kind} '{localId}')"
    | _, _ => s!"'{id}' ({kind})"
  | none => s!"'{id}' ({kind})"

/-- Decode graph topology and every scalar-bearing node in one active builder. -/
def decodeGraph (j : Json) (params : Array (String × JsonNumber))
    (rate : Nat := Tropical.Plan.defaultSampleRate) : BuildM PatchGraph := do
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
      raw.id raw.kind raw.sel raw.params raw.inObj rate (knobSlotOf pidx raws)
      (knobPhaseSlotOf pidx raws) (nodeLabel j raw.id raw.kind)
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
