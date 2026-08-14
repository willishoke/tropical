/-!
# Native FFI — the engine talks to libtropical directly

Phase 2 of the Lean port: `@[extern]` bindings over `lean/ffi/shim.c`,
which wraps `engine/c_api/tropical_c.h`. This replaces the koffi layer
for everything the engine owns — the runtime handle (plan loading,
slot control), the DAC, and device enumeration.

Handles are Lean external objects; finalizers call the matching
`*_free`. The `Dac` structure holds its `Runtime` as a Lean field so
the runtime object cannot be finalized while an audio callback reads
it.

Params have no FFI here by design: on the session path a param is a
`param:<name>` module slot — the control plane is `Runtime.setSlot`.
(The C `tropical_param_t` smoothing object never reaches the kernel on
this path; the TS engine's `set_param` was audibly inert until the
next recompile, which Phase 2 fixes by writing the slot.)
-/

namespace Tropical.Ffi

private opaque RuntimePointed : NonemptyType
/-- A `tropical_runtime_t` (FlatRuntime): the hot-swappable kernel host. -/
def Runtime : Type := RuntimePointed.type
instance : Nonempty Runtime := RuntimePointed.property

private opaque DacPointed : NonemptyType
/-- A raw `tropical_dac_t` handle. Use `Dac` (below), which keeps the
    runtime alive. -/
def DacHandle : Type := DacPointed.type
instance : Nonempty DacHandle := DacPointed.property

@[extern "shim_last_error"]
opaque lastError : IO String

-- ── Runtime ──────────────────────────────────────────────────────────────────

@[extern "shim_runtime_new"]
opaque Runtime.new (bufferLength : UInt32) : IO Runtime

@[extern "shim_runtime_process"]
opaque Runtime.process (rt : @& Runtime) : IO Unit

@[extern "shim_runtime_process_offline"]
private opaque Runtime.processOfflineRaw (rt : @& Runtime) : IO Bool

/-- Deterministic render tools only. Metal waits off-RT for the worker's next
    exact tile, then executes the ordinary bounded callback path once. -/
def Runtime.processOffline (rt : Runtime) : IO Unit := do
  if !(← rt.processOfflineRaw) then
    throw <| IO.userError s!"runtime processOffline failed: {← lastError}"

/-- Copy of the output buffer: `bufferLength` little-endian float64s. -/
@[extern "shim_runtime_output_bytes"]
opaque Runtime.outputBytes (rt : @& Runtime) : IO ByteArray

@[extern "shim_runtime_get_buffer_length"]
opaque Runtime.bufferLength (rt : @& Runtime) : IO UInt32

@[extern "shim_runtime_slot_index"]
opaque Runtime.slotIndexRaw (rt : @& Runtime) (name : @& String) : IO UInt32

@[extern "shim_runtime_set_slot"]
opaque Runtime.setSlot (rt : @& Runtime) (idx : UInt32) (value : Float) : IO Unit

@[extern "shim_runtime_get_slot"]
opaque Runtime.getSlot (rt : @& Runtime) (idx : UInt32) : IO Float

/-- The runtime's current sample index (τ of the next buffer), as a Float — the
    control plane stamps a glide's start-tick with this so a closed-form param
    ramp anchors to "now". -/
@[extern "shim_runtime_current_sample_index"]
opaque Runtime.currentSampleIndex (rt : @& Runtime) : IO Float

/-- The same source coordinate without the legacy f64 projection. Glide
    dispatch uses this to publish exact timestamp limbs. -/
@[extern "shim_runtime_current_sample_index_u64"]
opaque Runtime.currentSampleIndexU64 (rt : @& Runtime) : IO UInt64

/-- The active kernel's sample rate — the value the kernel reads for `sampleRate()`.
    The control plane uses it to reproduce the phasor's quantized increment when
    phase-anchoring a live frequency change (and for the glide's ms→sample dur). -/
@[extern "shim_runtime_sample_rate"]
opaque Runtime.sampleRate (rt : @& Runtime) : IO Float

/-- Cumulative wall time spent rendering Metal tiles and the number of sample
    frames those attempts produced. Their interval ratio is render utilization;
    the DAC callback itself only copies prepared samples and is intentionally
    not a useful Metal load measurement. -/
@[extern "shim_runtime_metal_render_time_ns"]
opaque Runtime.metalRenderTimeNs (rt : @& Runtime) : IO UInt64

@[extern "shim_runtime_metal_rendered_frame_count"]
opaque Runtime.metalRenderedFrameCount (rt : @& Runtime) : IO UInt64

/-- The C `UINT32_MAX` no-such-slot sentinel, mapped to `none`. -/
def Runtime.slotIndex? (rt : Runtime) (name : String) : IO (Option UInt32) := do
  let idx ← rt.slotIndexRaw name
  pure <| if idx == 0xffffffff then none else some idx

@[extern "shim_runtime_load_ir_staged"]
opaque Runtime.loadIrStagedRaw (rt : @& Runtime) (irText : @& String) (mslSource : @& String)
    (coeffIr : @& String) (manifestJson : @& String) : IO Bool

/-- Exact immutable program/control identity created by one runtime load. -/
structure PublishedGeneration where
  programVersion : UInt64
  controlVersion : UInt64
deriving Repr, BEq

@[extern "shim_runtime_load_ir_staged_generation"]
private opaque Runtime.loadIrStagedGenerationRaw
    (rt : @& Runtime) (irText : @& String) (mslSource : @& String)
    (coeffIr : @& String) (manifestJson : @& String) : IO (UInt64 × UInt64)

/-- Staged load: LLVM IR → JIT (always), MSL → Metal when non-empty, plus an
    optional stage-0 coefficient kernel (`coeffIr`, empty = no split) —
    a second single-function module the engine runs once before publish and
    after every control-plane slot write, landing τ-independent values in
    `coef:<n>` module slots the audio kernel reads. -/
def Runtime.loadIrStaged (rt : Runtime) (irText mslSource coeffIr manifestJson : String) :
    IO Unit := do
  if !(← rt.loadIrStagedRaw irText mslSource coeffIr manifestJson) then
    throw <| IO.userError s!"runtime loadIrStaged failed: {← lastError}"

/-- Generation-returning staged load for an atomic compile handshake. The pair
    comes from the publication operation itself, rather than a later sample. -/
def Runtime.loadIrStagedGeneration (rt : Runtime)
    (irText mslSource coeffIr manifestJson : String) :
    IO PublishedGeneration := do
  let (programVersion, controlVersion) ←
    rt.loadIrStagedGenerationRaw irText mslSource coeffIr manifestJson
  pure { programVersion, controlVersion }

@[extern "shim_runtime_load_ir_staged_with_observation_generation"]
private opaque Runtime.loadIrStagedWithObservationGenerationRaw (rt : @& Runtime)
    (irText : @& String) (mslSource : @& String) (coeffIr : @& String)
    (manifestJson : @& String) (observationIr : @& String)
    (observationCoeffIr : @& String) (observationManifestJson : @& String) :
    IO (UInt64 × UInt64)

/-- Atomically publish an audio/Metal staged artifact and a distinct JIT-only
    observation projection. Matching controls are copied by slot name; program
    storage and mutable workspaces remain disjoint. -/
def Runtime.loadIrStagedWithObservationGeneration (rt : Runtime)
    (irText mslSource coeffIr manifestJson : String)
    (observationIr observationCoeffIr observationManifestJson : String) :
    IO PublishedGeneration := do
  let (programVersion, controlVersion) ←
    rt.loadIrStagedWithObservationGenerationRaw
      irText mslSource coeffIr manifestJson
      observationIr observationCoeffIr observationManifestJson
  pure { programVersion, controlVersion }

/-- Control-plane/test-only: reposition the active kernel's sample clock
    (render verbs' `--start`). -/
@[extern "shim_runtime_set_sample_index"]
opaque Runtime.setSampleIndex (rt : @& Runtime) (idx : UInt64) : IO Unit

/-- Lower textual LLVM IR (Lean's EmitLlvm output) to a complete wasm32 module,
    in-process via the engine's LLVM + lld. Build-time only — requires
    libtropical built with `TROPICAL_WASM_EMIT`. Empty result ⇒ throw. -/
@[extern "shim_compile_ir_to_wasm"]
opaque compileIrToWasmRaw (irText : @& String) : IO ByteArray

def compileIrToWasm (irText : String) : IO ByteArray := do
  let wasm ← compileIrToWasmRaw irText
  if wasm.size == 0 then
    throw <| IO.userError s!"compileIrToWasm failed: {← lastError}"
  pure wasm

-- ── DAC ──────────────────────────────────────────────────────────────────────

@[extern "shim_dac_new_runtime"]
private opaque dacNewRuntime (rt : @& Runtime) (sampleRate channels : UInt32) : IO DacHandle

@[extern "shim_dac_start"]
private opaque dacStart (dac : @& DacHandle) : IO Unit

@[extern "shim_dac_stop"]
private opaque dacStop (dac : @& DacHandle) : IO Unit

@[extern "shim_dac_is_running"]
private opaque dacIsRunning (dac : @& DacHandle) : IO Bool

@[extern "shim_dac_is_reconnecting"]
private opaque dacIsReconnecting (dac : @& DacHandle) : IO Bool

@[extern "shim_dac_switch_device"]
private opaque dacSwitchDevice (dac : @& DacHandle) (deviceId : UInt32) : IO Bool

@[extern "shim_dac_stat_callback_count"]
private opaque dacStatCallbackCount (dac : @& DacHandle) : IO UInt64

@[extern "shim_dac_stat_avg_ms"]
private opaque dacStatAvgMs (dac : @& DacHandle) : IO Float

@[extern "shim_dac_stat_max_ms"]
private opaque dacStatMaxMs (dac : @& DacHandle) : IO Float

@[extern "shim_dac_stat_underruns"]
private opaque dacStatUnderruns (dac : @& DacHandle) : IO UInt64

@[extern "shim_dac_stat_overruns"]
private opaque dacStatOverruns (dac : @& DacHandle) : IO UInt64

@[extern "shim_dac_buffer_frames"]
private opaque dacBufferFrames (dac : @& DacHandle) : IO UInt32

/-- A DAC bound to the runtime it reads — the `runtime` field keeps the
    Lean runtime object (and so the C handle) alive for the DAC's
    lifetime. -/
structure Dac where
  handle  : DacHandle
  runtime : Runtime

structure DacStats where
  callbackCount : UInt64
  avgCallbackMs : Float
  maxCallbackMs : Float
  underrunCount : UInt64
  overrunCount  : UInt64

namespace Dac

def fromRuntime (rt : Runtime) (sampleRate channels : UInt32) : IO Dac := do
  pure { handle := (← dacNewRuntime rt sampleRate channels), runtime := rt }

def start (d : Dac) : IO Unit := dacStart d.handle
def stop (d : Dac) : IO Unit := dacStop d.handle
def isRunning (d : Dac) : IO Bool := dacIsRunning d.handle
def isReconnecting (d : Dac) : IO Bool := dacIsReconnecting d.handle
def switchDevice (d : Dac) (deviceId : UInt32) : IO Bool := dacSwitchDevice d.handle deviceId
def bufferFrames (d : Dac) : IO UInt32 := dacBufferFrames d.handle

def stats (d : Dac) : IO DacStats := do
  pure {
    callbackCount := ← dacStatCallbackCount d.handle
    avgCallbackMs := ← dacStatAvgMs d.handle
    maxCallbackMs := ← dacStatMaxMs d.handle
    underrunCount := ← dacStatUnderruns d.handle
    overrunCount  := ← dacStatOverruns d.handle }

end Dac

-- ── Device enumeration ───────────────────────────────────────────────────────

@[extern "shim_audio_device_count"]
opaque audioDeviceCount : IO UInt32

@[extern "shim_audio_device_ids"]
opaque audioDeviceIds : IO (Array UInt32)

@[extern "shim_audio_device_name"]
opaque audioDeviceName (deviceId : UInt32) : IO String

@[extern "shim_audio_default_output_device"]
opaque audioDefaultOutputDevice : IO UInt32

structure DeviceInfo where
  id   : UInt32
  name : String

def listDevices : IO (Array DeviceInfo) := do
  let ids ← audioDeviceIds
  ids.mapM fun id => do pure { id, name := (← audioDeviceName id) }

-- ── Socket endpoint ──────────────────────────────────────────────────────────

private opaque SocketHandlePointed : NonemptyType
/-- A raw `tropical_socket_t` handle (the C++ `SocketServer`). Use `Socket`
    (below), which keeps the runtime alive. -/
def SocketHandle : Type := SocketHandlePointed.type
instance : Nonempty SocketHandle := SocketHandlePointed.property

@[extern "shim_socket_listen"]
private opaque socketListenRaw (rt : @& Runtime) (addr : @& String) : IO SocketHandle

@[extern "shim_socket_next_control"]
private opaque socketNextControlRaw (s : @& SocketHandle) : IO (Option (UInt64 × String))

@[extern "shim_socket_send_response"]
private opaque socketSendResponseRaw (s : @& SocketHandle) (clientId : UInt64) (bytes : @& String) : IO Unit

/-- A socket bound to the runtime it drives — the `runtime` field keeps the
    Lean runtime object (and its C handle) alive for the socket's lifetime,
    since the socket's threads read the runtime's slots and telemetry. -/
structure Socket where
  handle  : SocketHandle
  runtime : Runtime

namespace Socket

/-- Bind + listen on a Unix-domain socket at `addr`; raise the engine's error
    string on failure. -/
def listen (rt : Runtime) (addr : String) : IO Socket := do
  pure { handle := (← socketListenRaw rt addr), runtime := rt }

/-- Block until a control-plane request arrives (returns `(clientId, line)`)
    or the socket shuts down (`none`). -/
def nextControl (s : Socket) : IO (Option (UInt64 × String)) := socketNextControlRaw s.handle

/-- Send a JSON-RPC response line back to one client. -/
def sendResponse (s : Socket) (clientId : UInt64) (bytes : String) : IO Unit :=
  socketSendResponseRaw s.handle clientId bytes

end Socket

end Tropical.Ffi
