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

-- ── CF-only enforcement: surface `reg`/`next` is unparseable ──────────────────
-- The Phase-1 guarantee, now STRUCTURAL: `reg`/`next` were deleted from the
-- surface grammar and the IR, so a program declaring them does not even parse
-- (the keywords are gone — `reg` lexes as a bare identifier and the statement
-- fails). A closed-form program (`Sin` — fold + temps, no reg) parses,
-- elaborates and strata-processes normally. The `Sin` case is the landmine pin
-- — emit-level SSA temps are not regs and must survive. Both are self-contained
-- (no instance deps), so they process standalone with a no-op external resolver.
def cfOnlyRejectSrc : String :=
  "```tropical\nprogram CfProbe(step: float = 1) -> (acc: float) {\n  reg s = 0\n  acc = s\n  next s = s + step\n}\n```"

def runCfOnly (name md : String) (expectReject : Bool) : IO Bool := do
  match Tropical.Parse.Surface.parseMarkdownProgram md with
  | .error e =>
    if expectReject then
      passGate s!"cf-only/{name}" "rejected per-sample state at parse"
    else
      failGate s!"cf-only/{name}" s!"parse: {firstLine e}"
  | .ok prog =>
    match Tropical.Ir.elaborateInto {} prog (some fun _ => none) with
    | .error e =>
      if expectReject then
        passGate s!"cf-only/{name}" "rejected per-sample state at elaboration"
      else
        failGate s!"cf-only/{name}" s!"unexpected reject: {firstLine e.message}"
    | .ok (arena, root) =>
      match Tropical.Ir.Strata.run { upto := 5 } arena root with
      | .error e =>
        failGate s!"cf-only/{name}" s!"strata error: {firstLine e.message}"
      | .ok _ =>
        if expectReject then
          failGate s!"cf-only/{name}" "compiled but should be rejected"
        else
          passGate s!"cf-only/{name}" "compiles (temps survive)"
