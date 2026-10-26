import Tropical.Ffi
import Tropical.HostPort

/-!
# NativeHost — the `HostPort` adapter over libtropical

Fills the engine's host port from the native FFI. Each field is a direct
forward to the `Tropical.Ffi` operation of the same name; the closures keep
the runtime object (and so the C handle) alive for the port's lifetime.
-/

namespace Tropical.Ffi

open Tropical.Host (HostPort DacPort)

def Dac.port (d : Dac) : DacPort where
  start := d.start
  stop := d.stop
  isRunning := d.isRunning
  isReconnecting := d.isReconnecting
  switchDevice := d.switchDevice
  bufferFrames := d.bufferFrames
  stats := d.stats

def Runtime.hostPort (rt : Runtime) : HostPort where
  publish a :=
    rt.loadIrTimeStagedGeneration a.ir a.msl a.coefficientIr a.tileIr a.manifest
  publishWithObservation a o :=
    rt.loadIrTimeStagedWithObservationGeneration
      a.ir a.msl a.coefficientIr a.tileIr a.manifest
      o.ir o.coefficientIr o.manifest
  slotIndex? := rt.slotIndex?
  setSlot := rt.setSlot
  getSlot := rt.getSlot
  currentSampleIndex := rt.currentSampleIndex
  currentSampleIndexU64 := rt.currentSampleIndexU64
  sampleRate := rt.sampleRate
  process := rt.process
  outputBytes := rt.outputBytes
  metalRenderTimeNs := rt.metalRenderTimeNs
  metalRenderedFrameCount := rt.metalRenderedFrameCount
  openDac sampleRate channels := do
    pure (← Dac.fromRuntime rt sampleRate channels).port
  listDevices := listDevices

end Tropical.Ffi
