import Tropical.Tropicaltest.Synthetic

/-!
# Tropical.Tropicaltest.Reversibility

Float64-render helpers (`decodeF64LE`, `renderSamples`) shared across the audio gates, plus the cf-only enforcement gate (surface `reg`/`next` is unparseable). The standalone reversibility probes were curated out; the moat's reversal is guarded by the `reverse_reverb` / `scrub_reverb` cf goldens (a struck reverse reverb IS the reversal path).
-/

open Tropical
open Tropical.Plan

/-- Decode little-endian float64 bytes (the runtime's mono output) to samples. -/
def decodeF64LE (b : ByteArray) : Array Float := Id.run do
  let n := b.size / 8
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let mut u : UInt64 := 0
    -- little-endian: byte (i*8+k) carries place value 256^k; read MSB→LSB
    for j in [0:8] do
      u := u * 256 + (b.get! (i * 8 + (7 - j))).toUInt64
    out := out.push (Float.ofBits u)
  pure out

/-- Render a plan to exactly `n` mono samples in one process call (buffer = n,
    so `sampleIndex()` runs 0 .. n-1 with no fade — fresh runtimes start with
    fade disabled, `fade_in_remaining_ = 0`). -/
def renderSamples (planJson : String) (n : Nat) : IO (Except String (Array Float)) := do
  match Lean.Json.parse planJson with
  | .error e => pure (.error s!"parse: {e}")
  | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
    | .error e => pure (.error s!"ofWire: {e}")
    | .ok plan => do
      let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
      Tropical.StagedLoad.load rt plan
      rt.process
      pure (.ok (decodeF64LE (← rt.outputBytes)))

