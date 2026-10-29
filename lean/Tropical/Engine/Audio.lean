import Tropical.Engine.ProgramIO

/-!
# Engine.Audio — audio lifecycle and param control

Start/stop/status for the DAC, the single `set_param` (a shell over the
host's discipline-dispatched write), param listing, and the debug render tap.
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

private def isRunningJson (dac : Tropical.Host.OutputPort) : EngineM Json := do
  pure <| Json.mkObj [("is_running", Json.bool (← dac.isRunning))]

def handleStartAudio (env : Env) (args : Json) : EngineM Json := do
  let dac ← match ← env.dac.get with
    | some d => pure d
    | none =>
      let sampleRate ← validatePositiveInt (arg? args "sample_rate") "sample_rate" 44100
      let channels   ← validatePositiveInt (arg? args "channels") "channels" 2
      let d ← env.host.openOutput sampleRate.toUInt32 channels.toUInt32
      env.dac.set (some d)
      pure d
  match argStr? args "device_name" with
  | some deviceName =>
    let devices ← env.host.listDevices
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
  let backend := env.host.backend.wire
  match ← env.dac.get with
  | none => pure <| Json.mkObj [
      ("is_running", Json.bool false),
      ("backend", Json.str backend)]
  | some dac =>
    let stats ← dac.stats
    let sampleRate ← env.host.sampleRate
    let bufferFrames ← dac.bufferFrames
    -- The wire keys keep their historical names: on libtropical the
    -- off-callback renderer is the Metal tile worker, and Reversible reads
    -- these two fields by name.
    let render ← env.host.renderStats
    pure <| Json.mkObj [
      ("is_running", Json.bool (← dac.isRunning)),
      ("is_reconnecting", Json.bool (← dac.isReconnecting)),
      ("backend", Json.str backend),
      ("sampleRate", toJson sampleRate),
      ("bufferFrames", toJson bufferFrames.toNat),
      ("stats", Json.mkObj [
        ("callbackCount", toJson stats.callbackCount.toNat),
        ("avgCallbackMs", toJson stats.avgCallbackMs),
        ("maxCallbackMs", toJson stats.maxCallbackMs),
        ("metalRenderTimeNs", toJson render.renderTimeNs.toNat),
        ("metalRenderedFrames", toJson render.renderedFrames.toNat),
        ("underrunCount", toJson stats.underrunCount.toNat),
        ("overrunCount",  toJson stats.overrunCount.toNat)])]

/-- The one public `set_param`: hand the write to the host and record the
    target in the session's param mirror. The host executes the discipline
    the loaded manifest declares for the name (raw / glide / anchor /
    velocity — `design/host-param-dispatch.md`) with its own `now` and sample
    rate, so no client and no authoring layer ever chooses a write verb. A
    name the loaded manifest has no slot for is the `unknown_param` envelope,
    with the mirror's names as the options; any other host failure is an
    `internal_error` carrying the host's message. -/
def handleSetParam (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let valueJ := (getField? args "value").getD jsonNull
  let value ← match valueJ with
    | .num n => pure n.toFloat
    | _ => internalError s!"set_param: value must be a number, got {valueJ.compress}"
  let write ← match ← (env.host.setParam name value).toBaseIO with
    | .ok (some w) => pure w
    | .ok none =>
      let st ← env.state.get
      throwEnum .unknownParam "name" (Json.str name) (st.params.map (·.1))
    | .error e => internalError (toString e)
  env.state.modify (·.setParamValue name valueJ)
  pure <| Json.mkObj [
    ("name", Json.str name), ("value", valueJ),
    ("observed_sample_index", toJson write.observedSampleIndex.toNat),
    ("effective_sample_index", toJson write.effectiveSampleIndex.toNat)]

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
    env.host.process
    let bytes ← env.host.outputBytes
    let mut chunk := ""
    for b in bytes.toList do
      chunk := chunk.push (hexDigit (b >>> 4)) |>.push (hexDigit (b &&& 0xf))
    hex := hex ++ chunk
  pure <| Json.mkObj [("frames", toJson frames), ("hex", Json.str hex)]

end Tropical.Engine
