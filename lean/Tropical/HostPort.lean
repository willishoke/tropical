import Tropical.Artifact

/-!
# HostPort — the runtime, as the session engine sees it

The session engine compiles, publishes, and drives a runtime it does not
own: `Engine.Env` holds a `HostPort`, a record of `IO` operations, and the
composition root (`Tropical.NativeBoot`) fills it from the native FFI
(`Tropical.NativeHost`). Nothing on the engine side names `Tropical.Ffi`.

The vocabulary here is exactly what the engine uses today: publish an
artifact (optionally with a JIT-only observation realization), write a param
through the host's discipline dispatch, render for the debug tap, read the
sample rate and Metal telemetry, and open the audio output. There is no raw
slot or clock access: the engine declares, the host executes.
-/

namespace Tropical.Host

open Tropical.Artifact (KernelArtifact)

/-- Exact immutable program/control identity created by one runtime load. -/
structure PublishedGeneration where
  programVersion : UInt64
  controlVersion : UInt64
deriving Repr, BEq

structure DacStats where
  callbackCount : UInt64
  avgCallbackMs : Float
  maxCallbackMs : Float
  underrunCount : UInt64
  overrunCount  : UInt64

structure DeviceInfo where
  id   : UInt32
  name : String

/-- What a host reports back from one param write: the completed buffer
    boundary it observed, and the first audible sample its discipline math
    used as `now`. -/
structure ParamWrite where
  observedSampleIndex : UInt64
  effectiveSampleIndex : UInt64

/-- An open output stream on the host. -/
structure DacPort where
  start : IO Unit
  stop : IO Unit
  isRunning : IO Bool
  isReconnecting : IO Bool
  switchDevice : UInt32 → IO Bool
  bufferFrames : IO UInt32
  stats : IO DacStats

structure HostPort where
  /-- Publish one artifact atomically; returns the generation it created. -/
  publish : KernelArtifact → IO PublishedGeneration
  /-- Publish a realtime artifact together with a distinct JIT-only
      observation artifact. Controls project into the observation by
      matching slot names. -/
  publishWithObservation : (audio observation : KernelArtifact) → IO PublishedGeneration
  /-- Write a param through the discipline the loaded manifest declares for
      it (`design/host-param-dispatch.md`): the host resolves the companions,
      reads its own clock and sample rate, and lands every slot of the write
      in one control generation. `none` when the loaded manifest has no slot
      for the name; throws with the host's message on any other failure. -/
  setParam : String → Float → IO (Option ParamWrite)
  sampleRate : IO Float
  /-- Render one buffer; `outputBytes` reads it back. -/
  process : IO Unit
  outputBytes : IO ByteArray
  metalRenderTimeNs : IO UInt64
  metalRenderedFrameCount : IO UInt64
  openDac : (sampleRate channels : UInt32) → IO DacPort
  listDevices : IO (Array DeviceInfo)

end Tropical.Host
