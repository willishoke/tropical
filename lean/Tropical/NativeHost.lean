import Tropical.Ffi
import Tropical.HostPort

/-!
# NativeHost — the `HostPort` adapter over libtropical

Fills the engine's host port from the native FFI. Each field is a direct
forward to the `Tropical.Ffi` operation of the same name; the closures keep
the runtime object (and so the C handle) alive for the port's lifetime.
-/

namespace Tropical.Ffi

open Tropical.Host (HostPort OutputPort Backend)

def Dac.port (d : Dac) : OutputPort where
  start := d.start
  stop := d.stop
  isRunning := d.isRunning
  isReconnecting := d.isReconnecting
  switchDevice := d.switchDevice
  bufferFrames := d.bufferFrames
  stats := d.stats

def Runtime.hostPort (rt : Runtime) (frames : Nat) (backend : Backend) : HostPort where
  frames := frames
  backend := backend
  publish a :=
    rt.loadGeneration a.ir a.msl a.stage0Ir a.manifest
  publishWithObservation a o :=
    rt.loadWithObservationGeneration
      a.ir a.msl a.stage0Ir a.manifest o.ir o.manifest
  setParam := rt.setParam
  sampleRate := rt.sampleRate
  process := rt.process
  outputBytes := rt.outputBytes
  -- libtropical's off-callback renderer is its Metal tile worker; the JIT
  -- renders in the callback and these read zero.
  renderStats := do
    pure { renderTimeNs := ← rt.metalRenderTimeNs,
           renderedFrames := ← rt.metalRenderedFrameCount }
  openOutput sampleRate channels := do
    pure (← Dac.fromRuntime rt sampleRate channels).port
  listDevices := listDevices

end Tropical.Ffi
