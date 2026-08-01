import Tropical.Engine.ProgramIO

/-!
# Engine.Audio — audio lifecycle and param control

Start/stop/status for the native DAC, the single discipline-dispatched
`set_param` operation, param listing, and the debug render tap.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

-- ── Audio / params (native — the engine owns the runtime and DAC) ───────────

private def validatePositiveInt (v : Option Json) (param : String)
    (default : Nat) : EngineM Nat :=
  match v with
  | none => pure default
  | some j@(.num n) =>
    let f := n.toFloat
    if f == f.floor && f > 0 then pure f.toUInt64.toNat
    else
      throwBare .invalidValue
        s!"{param} must be a positive integer; got {j.compress}"
        (param := some param) (value := some j)
  | some j =>
    throwBare .invalidValue
      s!"{param} must be a positive integer; got {j.compress}"
      (param := some param) (value := some j)

private def containsSub (hay needle : String) : Bool :=
  needle.isEmpty || (hay.splitOn needle).length > 1

private def isRunningJson (dac : Ffi.Dac) : EngineM Json := do
  pure <| Json.mkObj [("is_running", Json.bool (← dac.isRunning))]

def handleStartAudio (env : Env) (args : Json) : EngineM Json := do
  let dac ← match ← env.dac.get with
    | some d => pure d
    | none =>
      let sampleRate ← validatePositiveInt (arg? args "sample_rate") "sample_rate" 44100
      let channels   ← validatePositiveInt (arg? args "channels") "channels" 2
      let d ← Ffi.Dac.fromRuntime env.runtime sampleRate.toUInt32 channels.toUInt32
      env.dac.set (some d)
      pure d
  match argStr? args "device_name" with
  | some deviceName =>
    let devices ← Ffi.listDevices
    let lower := deviceName.toLower
    match devices.find? (fun d => containsSub d.name.toLower lower) with
    | none =>
      throwEnum .unknownDevice "device_name" (Json.str deviceName)
        (devices.map (·.name))
    | some m =>
      if ← dac.isRunning then
        let _ ← dac.switchDevice m.id
      else
        dac.start
        let _ ← dac.switchDevice m.id
      pure <| Json.mkObj [("is_running", Json.bool (← dac.isRunning)),
                          ("device", Json.str m.name)]
  | none =>
    if !(← dac.isRunning) then dac.start
    isRunningJson dac

def handleStopAudio (env : Env) : EngineM Json := do
  match ← env.dac.get with
  | none => throwBare .invalidState "DAC has not been created yet."
  | some dac =>
    dac.stop
    isRunningJson dac

def handleAudioStatus (env : Env) : EngineM Json := do
  match ← env.dac.get with
  | none => pure <| Json.mkObj [("is_running", Json.bool false)]
  | some dac =>
    let stats ← dac.stats
    pure <| Json.mkObj [
      ("is_running", Json.bool (← dac.isRunning)),
      ("is_reconnecting", Json.bool (← dac.isReconnecting)),
      ("stats", Json.mkObj [
        ("callbackCount", toJson stats.callbackCount.toNat),
        ("avgCallbackMs", toJson stats.avgCallbackMs),
        ("maxCallbackMs", toJson stats.maxCallbackMs),
        ("underrunCount", toJson stats.underrunCount.toNat),
        ("overrunCount",  toJson stats.overrunCount.toNat)])]

/-- `set_param`: update the mirror AND drive the live `param:<name>`
    module slot. (The TS engine only wrote the detached Param handle,
    which the session-path kernel never reads — set_param was audibly
    inert until the next recompile.) -/
def handleSetParam (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let valueJ := (getField? args "value").getD jsonNull
  let st ← env.state.get
  if (st.params.find? (·.1 == name)).isNone then
    throwEnum .unknownParam "name" (Json.str name) (st.params.map (·.1))
  let value ← match valueJ with
    | .num n => pure n.toFloat
    | _ => internalError s!"set_param: value must be a number, got {valueJ.compress}"
  env.state.modify (·.setParamValue name valueJ)
  if let some idx := ← env.runtime.slotIndex? s!"param:{name}" then
    env.runtime.setSlot idx value
  pure <| Json.mkObj [("name", Json.str name), ("value", valueJ)]

/-- Apply the glide discipline for `set_param`: a CLOSED-FORM parameter ramp
    (no per-sample state). The
    kernel evaluates `f(τ) = v0 + (v1−v0)·smoothstep(clamp((τ−t0)/dur, 0, 1))` from
    three slots (`param:<name>#v0/#v1/#t0`); this op RE-ANCHORS the ramp: read the
    current sample index `now`, evaluate the ramp at `now` (so the new ramp starts
    exactly where we are — no jump), then set `v0 = current, v1 = target, t0 = now`.
    Stateless and click-free — the "state" is the anchor slots, navigable like τ.
    `dur = 0.02·SR` samples (20 ms) matches the kernel's ramp, at any sample rate. -/
private def applyParamGlide (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let target ← match getField? args "value" with
    | some (.num n) => pure n.toFloat
    | _ => internalError "set_param: glide value must be a number"
  let slotOf (sfx : String) : EngineM (Option UInt32) :=
    env.runtime.slotIndex? s!"param:{name}#{sfx}"
  let some _ := ← slotOf "v0"
    | internalError s!"set_param: no glide slots for '{name}'"
  let read (sfx : String) : EngineM Float := do
    match ← slotOf sfx with | some i => env.runtime.getSlot i | none => pure 0.0
  let write (sfx : String) (v : Float) : EngineM Unit := do
    match ← slotOf sfx with | some i => env.runtime.setSlot i v | none => pure ()
  let now ← env.runtime.currentSampleIndex
  let dur := (← env.runtime.sampleRate) * 0.02   -- 20 ms, matching the kernel's ramp
  let v0 ← read "v0"; let v1 ← read "v1"; let t0 ← read "t0"
  let raw := (now - t0) / dur
  let s := if raw < 0.0 then 0.0 else if raw > 1.0 then 1.0 else raw
  let curr := v0 + (v1 - v0) * (s * s * (3.0 - 2.0 * s))
  write "v0" curr
  write "v1" target
  write "t0" now
  -- the mirror tracks the ramp's TARGET (its settled value), like every
  -- other write path — a glide previously skipped this, so list_params lied.
  env.state.modify (·.setParamValue name (toJson target))
  pure <| Json.mkObj [("name", Json.str name), ("value", toJson target)]

/-- Apply the anchor discipline for `set_param`: a PHASE-ANCHORED frequency
    change. `freq = f·τ + φ_off` — so
    changing `f` alone jumps the phase by `Δf·τ` (a click that grows with τ). If the
    source carries a `#phase` offset slot, this bumps it by the phase the frequency
    change would have jumped — `Δφ = ((inc₀ − inc₁)·T) / 2³²` cycles, `inc =
    ⌊freq·2³²/SR⌋` (the phasor's own quantized increment), `T` = now — so the phase
    stays CONTINUOUS across the change (a soft freq corner, not a hard click). This
    is the stateless anchor: the accumulated phase lives in a param, navigable like
    τ. No `#phase` slot (saw/morph, or knob-driven freq) ⇒ a raw freq write. -/
private def applyParamAnchor (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let target ← match getField? args "value" with
    | some (.num n) => pure n.toFloat
    | _ => internalError "set_param: anchor value must be a number"
  let some freqIdx := ← env.runtime.slotIndex? s!"param:{name}"
    | internalError s!"set_param: no anchor slot '{name}'"
  match ← env.runtime.slotIndex? s!"param:{name}#phase" with
  | some phaseIdx =>
    let now ← env.runtime.currentSampleIndex
    let f0 ← env.runtime.getSlot freqIdx
    let sr ← env.runtime.sampleRate
    let inc0 := Float.floor (f0 * 4294967296.0 / sr)
    let inc1 := Float.floor (target * 4294967296.0 / sr)
    let dcyc := ((inc0 - inc1) * now) / 4294967296.0
    let off0 ← env.runtime.getSlot phaseIdx
    let raw := off0 + dcyc
    env.runtime.setSlot phaseIdx (raw - Float.floor raw)   -- frac → [0, 1)
    env.runtime.setSlot freqIdx target
  | none => env.runtime.setSlot freqIdx target
  env.state.modify (·.setParamValue name (toJson target))
  pure <| Json.mkObj [("name", Json.str name), ("value", toJson target)]

/-- Apply the velocity discipline for `set_param`: the GLOBAL TIME-WARP scrub.
    The master clock is
    `M(n) = tau_base·SR·2³² + velocity·2³²·n`; changing `velocity` alone would jump
    `M` by `Δv·n` (a click growing with n). This re-bases the host-held origin so
    `M` stays value-continuous across the change: read `now`, set
    `tau_base += (v_old − v_new)·now/SR`, then write the new velocity. Exactly the
    stateless `ScrubClock` host-split (and the clock-domain twin of
    anchor discipline): the accumulator lives in a param, navigable like τ, so the
    kernel stays `f(τ)`. `velocity = 1` forward · `0` freeze · `−1` reverse · `>1`
    varispeed. `name` is the velocity slot (`master.velocity`); the origin slot is
    the sibling `master.tau_base`. -/
private def applyParamVelocity (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let target ← match getField? args "value" with
    | some (.num n) => pure n.toFloat
    | _ => internalError "set_param: velocity value must be a number"
  let some velIdx := ← env.runtime.slotIndex? s!"param:{name}"
    | internalError s!"set_param: no velocity slot '{name}'"
  let tauName := name.replace "velocity" "tau_base"
  let some tauIdx := ← env.runtime.slotIndex? s!"param:{tauName}"
    | internalError s!"set_param: no velocity origin slot '{tauName}'"
  let now ← env.runtime.currentSampleIndex
  let sr ← env.runtime.sampleRate
  let v0 ← env.runtime.getSlot velIdx
  let tb0 ← env.runtime.getSlot tauIdx
  let tb1 := tb0 + (v0 - target) * now / sr
  env.runtime.setSlot tauIdx tb1
  env.runtime.setSlot velIdx target
  env.state.modify fun st =>
    (st.setParamValue tauName (toJson tb1)).setParamValue name (toJson target)
  pure <| Json.mkObj [
    ("name", Json.str name),
    ("value", toJson target),
    ("tau_base", toJson tb1),
    ("observed_sample_index", toJson now),
    ("effective_sample_index", toJson now)]

/-- The one public `set_param`, dispatched by the plan's own discipline table
    (the host
    contract, design/host-param-dispatch.md): the caller never chooses a write
    verb. Raw for table-less names (session-model patches, old plans) — with
    the one engine-owned exception that the reserved master-clock velocity
    param always re-bases, table or no table (the engine consulting its own
    reserved name is not a client choosing semantics). -/
def handleSetParamDispatch (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let st ← env.state.get
  let disc := match st.paramDisciplines.find? (·.name == name) with
    | some d => d.discipline
    | none => if name == Tropical.Playground.masterVelocityParam then "velocity" else "raw"
  match disc with
  | "glide" => applyParamGlide env args
  | "anchor" => applyParamAnchor env args
  | "velocity" => applyParamVelocity env args
  | _ => handleSetParam env args

def handleListParams (env : Env) : EngineM Json := do
  let st ← env.state.get
  pure <| Json.arr <| st.params.map fun (n, v) =>
    Json.mkObj [("name", Json.str n), ("value", v)]

/-- Differential-harness probe (rpc-only, not an MCP tool): render N
    buffers through the Lean-owned runtime, return the raw samples as
    hex — lets the recorded scripts assert audio equivalence between
    engines. -/
def handleDebugRender (env : Env) (args : Json) : EngineM Json := do
  let frames := match arg? args "frames" with
    | some (.num n) => n.toFloat.toUInt64.toNat
    | _ => 4
  let hexDigit (n : UInt8) : Char :=
    if n < 10 then Char.ofNat ('0'.toNat + n.toNat)
    else Char.ofNat ('a'.toNat + n.toNat - 10)
  let mut hex := ""
  for _ in [0:frames] do
    env.runtime.process
    let bytes ← env.runtime.outputBytes
    let mut chunk := ""
    for b in bytes.toList do
      chunk := chunk.push (hexDigit (b >>> 4)) |>.push (hexDigit (b &&& 0xf))
    hex := hex ++ chunk
  pure <| Json.mkObj [("frames", toJson frames), ("hex", Json.str hex)]

end Tropical.Engine
