import Tropical.Tropicaltest.Synthetic

/-!
# Tropical.Tropicaltest.Reversibility

Reversibility gates: a closed-form-in-τ patch fed a palindromic τ renders a palindrome (bit-exact), the ClockPhasor≡FixedPhasor identity, the forward+reverse cancellation, and the cf-only enforcement (surface `reg`/`next` is unparseable).
-/

open Tropical
open Tropical.Plan

-- ── Reversibility: a closed-form-in-τ patch fed a palindromic τ is a palindrome ─
-- The architectural claim made testable. `ReversibleProbe` drives a symmetric
-- time coordinate from the sample counter (forward to `half`, then back) and
-- feeds it to a stateless closed-form patch (comb over modal voice). Equal τ ⟹
-- equal output, so the render must be a bit-exact palindrome about `half`. A
-- single mismatched pair means a register leaked in — statefulness broke purity.

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

/-- Compile a probe patch, render `2*half` samples, assert the output is a
    bit-exact palindrome about index `half` (and non-silent). One witness,
    two probes: `reversibility` (comb over modal voice) and `flanger`
    (`ThroughZeroFlanger` — the LFO that sweeps `delta` is itself a function
    of `tau`, so unfreezing the comb adds no state; a latched oscillator
    would diverge between the forward and reverse halves). -/
private def runPalindrome (label patchPath : String) : IO Bool := do
  let half : Nat := 2048
  let n : Nat := 2 * half
  match ← compilePatch patchPath .fused with
  | .error e => failGate label s!"compile: {firstLine e}"
  | .ok planJson =>
    match ← renderSamples planJson n with
    | .error e => failGate label s!"render: {firstLine e}"
    | .ok samples =>
      if samples.size < n then
        failGate label s!"got {samples.size} samples (want {n})"
      else do
        let mut mism := 0
        let mut firstBad := 0
        for k in [1:half] do
          if (samples[half + k]!).toBits != (samples[half - k]!).toBits then
            if mism == 0 then firstBad := k
            mism := mism + 1
        let mut energy := 0.0
        let mut maxAbs := 0.0
        for k in [0:n] do
          let v := samples[k]!
          energy := energy + v * v
          if v.abs > maxAbs then maxAbs := v.abs
        if mism != 0 then
          failGate label s!"{mism} mismatched pairs (first k={firstBad})"
        else if energy <= 1e-6 then
          failGate label s!"signal is silent (energy {energy})"
        else
          passGate label s!"bit-exact palindrome over {half-1} pairs (peak |x|={maxAbs}, energy={energy})"

def runReversibility : IO Bool :=
  runPalindrome "reversibility" "patches/reversible_probe.json"

def runFlangerReversibility : IO Bool :=
  runPalindrome "flanger" "patches/flanger_probe.json"

/-- Fixed-point clock substrate witness: `ClockPhasor(clk: clock())` must be
    bit-for-bit identical to `FixedPhasor` (the root clock `θ = sampleIndex <<
    32` has zero fraction, so the split-multiply collapses to `inc·n + off`).
    The probe outputs `FixedPhasor.phase − ClockPhasor.phase`; assert it is
    exactly zero at every sample. -/
def runClockPhasorEquiv : IO Bool := do
  let n : Nat := 4096
  match ← compilePatch "patches/clock_phasor_probe.json" .fused with
  | .error e => failGate "clock-phasor" s!"compile: {firstLine e}"
  | .ok planJson =>
    match ← renderSamples planJson n with
    | .error e => failGate "clock-phasor" s!"render: {firstLine e}"
    | .ok samples =>
      if samples.size < n then
        failGate "clock-phasor" s!"got {samples.size} samples (want {n})"
      else do
        let mut maxAbs := 0.0
        let mut firstBad := 0
        let mut bad := 0
        for k in [0:n] do
          let v := samples[k]!
          if v.toBits != (0.0 : Float).toBits then
            if bad == 0 then firstBad := k
            bad := bad + 1
          if v.abs > maxAbs then maxAbs := v.abs
        if bad != 0 then
          failGate "clock-phasor" s!"{bad} nonzero samples (first k={firstBad}, max|Δ|={maxAbs})"
        else
          passGate "clock-phasor" "ClockPhasor(clock()) ≡ FixedPhasor bit-for-bit"

/-- Per-oscillator reverse witness: `FixedSinOsc(clk: -θ)` is the negated
    forward sine, so `forward + reverse` cancels. Reports the residual (≈ Sin
    polynomial range-reduction asymmetry); asserts it is at most a small
    epsilon. -/
def runClockReverseProbe : IO Bool := do
  let n : Nat := 4096
  match ← compilePatch "patches/clock_reverse_probe.json" .fused with
  | .error e => failGate "clock-reverse" s!"compile: {firstLine e}"
  | .ok planJson =>
    match ← renderSamples planJson n with
    | .error e => failGate "clock-reverse" s!"render: {firstLine e}"
    | .ok samples =>
      let mut maxAbs := 0.0
      for k in [0:samples.size] do
        if samples[k]!.abs > maxAbs then maxAbs := samples[k]!.abs
      if maxAbs < 1e-6 then
        passGate "clock-reverse" s!"forward+reverse cancels (max|Δ|={maxAbs})"
      else
        failGate "clock-reverse" s!"residual too large (max|Δ|={maxAbs})"

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
