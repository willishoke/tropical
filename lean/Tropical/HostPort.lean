import Tropical.Artifact

/-!
# HostPort — the runtime, as the session engine sees it

The session engine compiles, publishes, and drives a runtime it does not
own: `Engine.Env` holds a `HostPort`, a record of `IO` operations, and the
composition root (`Tropical.NativeBoot`) fills it from the native FFI
(`Tropical.NativeHost`). Nothing on the engine side names `Tropical.Ffi`.

The vocabulary here is exactly what the engine uses today: publish an
artifact (optionally with a JIT-only observation realization), read and
write named slots, read the clock, render for the debug tap, read Metal
telemetry, and open the audio output.
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
  slotIndex? : String → IO (Option UInt32)
  setSlot : UInt32 → Float → IO Unit
  getSlot : UInt32 → IO Float
  currentSampleIndex : IO Float
  currentSampleIndexU64 : IO UInt64
  sampleRate : IO Float
  /-- Render one buffer; `outputBytes` reads it back. -/
  process : IO Unit
  outputBytes : IO ByteArray
  metalRenderTimeNs : IO UInt64
  metalRenderedFrameCount : IO UInt64
  openDac : (sampleRate channels : UInt32) → IO DacPort
  listDevices : IO (Array DeviceInfo)

end Tropical.Host
