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
sample rate and render telemetry, and open the output stream. There is no
raw slot or clock access: the engine declares, the host executes. The port
names no backend: which realization a host runs is a value it reports
(`backend`), and its telemetry is counters, not a backend's API.
-/

namespace Tropical.Host

open Tropical.Artifact (KernelArtifact)

/-- Exact immutable program/control identity created by one runtime load. -/
structure PublishedGeneration where
  programVersion : UInt64
  controlVersion : UInt64
deriving Repr, BEq

/-- The realization a host runs the published artifact on. The engine emits
    what that needs (an MSL kernel and the CPU pre-pass for `metal`) and
    reports it in `audio_status`. -/
inductive Backend where
  | jit | metal
deriving BEq, DecidableEq, Repr, Inhabited

def Backend.wire : Backend → String
  | .jit => "jit" | .metal => "metal"

/-- Callback statistics of an open output stream. -/
structure OutputStats where
  callbackCount : UInt64
  avgCallbackMs : Float
  maxCallbackMs : Float
  underrunCount : UInt64
  overrunCount  : UInt64

/-- Cumulative off-callback render work: the wall time a host spent
    rendering ahead of its output stream and the frames that produced. A
    host that renders inside the callback reports zeros. -/
structure RenderStats where
  renderTimeNs : UInt64
  renderedFrames : UInt64

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
structure OutputPort where
  start : IO Unit
  stop : IO Unit
  isRunning : IO Bool
  isReconnecting : IO Bool
  switchDevice : UInt32 → IO Bool
  bufferFrames : IO UInt32
  stats : IO OutputStats

structure HostPort where
  /-- The frames per call this host renders: the static shape every
      artifact it loads must be compiled for (`design/typed-kernel.md`). -/
  frames : Nat
  /-- The realization this host runs. -/
  backend : Backend
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
  renderStats : IO RenderStats
  openOutput : (sampleRate channels : UInt32) → IO OutputPort
  listDevices : IO (Array DeviceInfo)

end Tropical.Host
