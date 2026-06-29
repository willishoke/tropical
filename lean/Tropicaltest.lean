import Tropical.Ffi
import Tropical.Engine
import Tropical.Plan
import Tropical.Ir.EmitLlvm
import Tropical.PlanDecode
import Tropical.Parse.Surface.Markdown
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Compile
import Tropical.ArrowWarp
import Lean.Data.Json

/-!
# tropicaltest — the post-TS golden + native-equiv runner (Phase 8)

Replaces `scripts/validate_stdlib.ts` and the native equiv suite. Anchored by
the **frozen audio goldens** (the correctness floor) plus the **native
mode-equiv** cross-check (fused vs microkernel — both ship), all driven off
`Tropical.Ffi.Runtime` (the same `libtropical` the JIT serves). No TS, no koffi.

Goldens reproduce because the Lean engine emits the same plan (`diff-plan`)
and the Lean FFI renders the same dylib byte-for-byte (`diff-render`).
-/

open Tropical

private def FRAMES : Nat := 16
private def BUFFER : Nat := 256

/-- Render a plan JSON to its raw little-endian f64 PCM bytes (16×256 = 4096
    samples). Lean owns codegen: parse the plan, emit IR (EmitLlvm), load via
    load_ir. The goldens reproduce because EmitLlvm is byte-identical to the
    retired C++ plan compiler (proven across the corpus in Phase 1b). -/
def renderPlanBytes (planJson : String) : IO ByteArray := do
  let plan ← match Lean.Json.parse planJson with
    | .error e => throw (IO.userError s!"renderPlanBytes: parse: {e}")
    | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
      | .error e => throw (IO.userError s!"renderPlanBytes: ofWire: {e}")
      | .ok p => pure p
  let ir ← match Tropical.Ir.EmitLlvm.emitKernel plan with
    | .error e => throw (IO.userError s!"renderPlanBytes: emitKernel: {e}")
    | .ok s => pure s
  let rt ← Tropical.Ffi.Runtime.new BUFFER.toUInt32
  rt.loadIr ir planJson
  let mut acc := ByteArray.empty
  for _ in [0:FRAMES] do
    rt.process
    acc := acc ++ (← rt.outputBytes)
  pure acc

/-- SHA256 hex via `shasum -a 256` — the same digest node crypto and
    `diff_render.sh` produce over the same bytes. -/
def sha256Hex (bytes : ByteArray) : IO String := do
  let tmp := "/tmp/tropicaltest-render.bin"
  IO.FS.writeBinFile tmp bytes
  -- `shasum -a 256` on macOS, `sha256sum` on Linux; both print "<hex>  <file>".
  let out ← (try IO.Process.run { cmd := "shasum", args := #["-a", "256", tmp] }
             catch _ => IO.Process.run { cmd := "sha256sum", args := #[tmp] })
  pure (out.splitOn " " |>.headD "")

/-- Compile a tropical_program_2 patch (by path) to a plan in `mode`. Boots a
    fresh engine per compile (the proven `diffcli compile` pattern — a reused
    booted env does not survive sequential loads). -/
def compilePatch (path : String) (mode : Tropical.Plan.CompilationMode) :
    IO (Except String String) := do
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM String := do
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str path)])
    Tropical.Engine.compileMirrorPlan env mode
  match ← act.run with
  | .ok planJson => pure (.ok planJson)
  | .error f => pure (.error f.toJson.compress)

/-- Render a FlatPlan via the Lean-emitted-IR path (EmitLlvm → load_ir). -/
def renderIrBytes (plan : Tropical.Plan.FlatPlan) : IO (Except String ByteArray) := do
  match plan.toWire, Tropical.Ir.EmitLlvm.emitKernel plan with
  | .ok manifest, .ok ir =>
    let rt ← Tropical.Ffi.Runtime.new BUFFER.toUInt32
    rt.loadIr ir manifest.compress
    let mut acc := ByteArray.empty
    for _ in [0:FRAMES] do
      rt.process
      acc := acc ++ (← rt.outputBytes)
    pure (.ok acc)
  | .error e, _ => pure (.error s!"toWire: {e}")
  | _, .error e => pure (.error s!"emitKernel: {e}")

private def firstLine (s : String) : String := (s.splitOn "\n").headD ""

-- ── Synthetic op-coverage plan ───────────────────────────────────────────────
-- Exercises ops the patch corpus doesn't reach (GreaterEq, NotEqual, Or,
-- BitOr, BitNot, FloorDiv, Sqrt, Floor, Ceil, Abs, ToInt/ToBool, Not), so a
-- typo in a predicate/intrinsic string in EmitLlvm is caught before the
-- one-way C++-codegen deletion. Built directly as a FlatPlan; compared
-- load_plan vs load_ir like the rest of section (d).
section OpCoverage
open Tropical.Plan

private def jn (m : Int) (e : Nat := 0) : Lean.JsonNumber := { mantissa := m, exponent := e }
private def cF (m : Int) (e : Nat := 0) : NOperand := .const (jn m e) .float
private def cI (m : Int) : NOperand := .const (jn m) .int
private def rgF (slot : Nat) : NOperand := .reg slot .float
private def rgI (slot : Nat) : NOperand := .reg slot .int
private def rgB (slot : Nat) : NOperand := .reg slot .bool

/-- Each op computes into a temp; results funnel through ToFloat and an
    Add chain to a single slot the sink reads. -/
def opCoverageInstrs : Array NInstr := #[
  instrScalar "GreaterEq" 0 #[cF 5, cF 3] .bool,          -- true
  instrScalar "ToFloat"   1 #[rgB 0] .float,              -- 1
  instrScalar "NotEqual"  2 #[cF 5, cF 3] .bool,          -- true
  instrScalar "ToFloat"   3 #[rgB 2] .float,              -- 1
  instrScalar "Sqrt"      4 #[cF 16] .float,              -- 4
  instrScalar "Floor"     5 #[cF 37 1] .float,            -- 3.7 → 3
  instrScalar "Ceil"      6 #[cF 32 1] .float,            -- 3.2 → 4
  instrScalar "Abs"       7 #[cF (-25) 1] .float,         -- -2.5 → 2.5
  instrScalar "BitOr"     8 #[cI 1, cI 2] .int,           -- 3
  instrScalar "ToFloat"   9 #[rgI 8] .float,
  instrScalar "BitNot"   10 #[cI 0] .int,                 -- -1
  instrScalar "ToFloat"  11 #[rgI 10] .float,
  instrScalar "Not"      12 #[cF 0] .bool,                -- true
  instrScalar "ToFloat"  13 #[rgB 12] .float,             -- 1
  instrScalar "Or"       14 #[cF 0, cF 1] .bool,          -- true
  instrScalar "ToFloat"  15 #[rgB 14] .float,             -- 1
  instrScalar "FloorDiv" 16 #[cF 7, cF 2] .float,         -- 3
  instrScalar "ToInt"    17 #[cF 37 1] .int,              -- 3
  instrScalar "ToFloat"  18 #[rgI 17] .float,             -- 3
  instrScalar "ToBool"   19 #[cF 9] .bool,                -- true
  instrScalar "ToFloat"  20 #[rgB 19] .float,             -- 1
  instrScalar "Add"      21 #[rgF 1, rgF 3] .float,
  instrScalar "Add"      22 #[rgF 21, rgF 4] .float,
  instrScalar "Add"      23 #[rgF 22, rgF 5] .float,
  instrScalar "Add"      24 #[rgF 23, rgF 6] .float,
  instrScalar "Add"      25 #[rgF 24, rgF 7] .float,
  instrScalar "Add"      26 #[rgF 25, rgF 9] .float,
  instrScalar "Add"      27 #[rgF 26, rgF 11] .float,
  instrScalar "Add"      28 #[rgF 27, rgF 13] .float,
  instrScalar "Add"      29 #[rgF 28, rgF 15] .float,
  instrScalar "Add"      30 #[rgF 29, rgF 16] .float,
  instrScalar "Add"      31 #[rgF 30, rgF 18] .float,
  instrScalar "Add"      32 #[rgF 31, rgF 20] .float,
  instrWriteSlot 0 (rgF 32)]

def opCoveragePlan : FlatPlan :=
  let inst := InstanceFunction.mk "root" "root" #[] opCoverageInstrs #[] 0 0 33 #[]
  { sampleRate := jn 44100, compilationMode := .fused,
    arraySlotNames := #[], registerCount := 33, arraySlotCount := 0,
    arraySlotSizes := #[], instanceFunctions := #[inst],
    sinks := #[{ inputs := #[0], gain := jn 1, target := 0 }],
    sources := defaultSources, slotCount := 1, slotNames := #["out"],
    slotDefaults := #[Lean.Json.num (jn 0)] }

end OpCoverage

private def sortedNames (dir : String) (suffix : String) : IO (Array String) := do
  let entries ← (System.FilePath.mk dir).readDir
  let names := entries.filterMap fun e =>
    if e.fileName.endsWith suffix then some (e.fileName.dropRight suffix.length) else none
  pure (names.qsort fun a b => decide (a < b))

/-- Compile a patch (fused) and hash its rendered 16×256 output. -/
private def hashOf (patchPath : String) : IO (Except String String) := do
  match ← compilePatch patchPath .fused with
  | .error e => pure (.error e)
  | .ok planJson => pure (.ok (← sha256Hex (← renderPlanBytes planJson)))

/-- A file-backed patch golden: compare to `goldenPath`, or rewrite it under
    `--write` (the `validate_stdlib --write` re-baseline, now Lean-owned). -/
private def runGolden (writeMode : Bool) (name patchPath goldenPath : String) : IO Bool := do
  match ← hashOf patchPath with
  | .error e => IO.println s!"  FAIL  {name}  compile: {firstLine e}"; pure false
  | .ok got =>
    if writeMode then
      IO.FS.writeFile goldenPath (got ++ "\n")
      IO.println s!"  WROTE {name}  {got.take 16}"; pure true
    else
      let expected := firstLine (← IO.FS.readFile goldenPath)
      if got == expected then IO.println s!"  PASS  {name}  {got.take 16}"; pure true
      else IO.println s!"  FAIL  {name}  expected {expected.take 16} got {got.take 16}"; pure false

/-- A golden whose expected hash is supplied inline (migration fixtures, whose
    hash lives inside a JSON record — read-only). -/
private def checkGoldenHash (name patchPath expected : String) : IO Bool := do
  match ← hashOf patchPath with
  | .error e => IO.println s!"  FAIL  {name}  compile: {firstLine e}"; pure false
  | .ok got =>
    if got == expected then IO.println s!"  PASS  {name}  {got.take 16}"; pure true
    else IO.println s!"  FAIL  {name}  expected {expected.take 16} got {got.take 16}"; pure false

/-- Regression for `let`-binding serialization order. `Sin`'s `let` is
    order-dependent (`poly` uses `r2`, `sign` uses `odd_n`) and its binding
    names do not sort in declaration order ("poly" < "r2"). Surface-parse →
    encode (`toJson`) → re-parse → decode (`decodeProgram`) → elaborate must
    succeed; if `let` bind were serialized as a key-reordering object, the
    round-trip would scramble the bindings and elaboration would fail with
    "unknown name". -/
private def runLetRoundtrip : IO Bool := do
  let md ← IO.FS.readFile "stdlib/Sin.md"
  match Tropical.Parse.Surface.parseMarkdownProgram md with
  | .error e => IO.println s!"  FAIL  let-roundtrip  parse: {firstLine e}"; pure false
  | .ok prog =>
    match Tropical.Parse.JsonV.parse prog.toJson.compress with
    | .error e => IO.println s!"  FAIL  let-roundtrip  reparse: {firstLine e}"; pure false
    | .ok jv =>
      match Tropical.Parse.decodeProgram jv with
      | .error e => IO.println s!"  FAIL  let-roundtrip  decode: {firstLine e}"; pure false
      | .ok prog2 =>
        match Tropical.Ir.elaborateInto {} prog2 (some fun _ => none) with
        | .error e => IO.println s!"  FAIL  let-roundtrip  elaborate: {firstLine e.message}"; pure false
        | .ok _ => IO.println "  PASS  let-roundtrip  Sin survives encode→decode→elaborate"; pure true

-- ── Reversibility: a closed-form-in-τ patch fed a palindromic τ is a palindrome ─
-- The architectural claim made testable. `ReversibleProbe` drives a symmetric
-- time coordinate from the sample counter (forward to `half`, then back) and
-- feeds it to a stateless closed-form patch (comb over modal voice). Equal τ ⟹
-- equal output, so the render must be a bit-exact palindrome about `half`. A
-- single mismatched pair means a register leaked in — statefulness broke purity.

/-- Decode little-endian float64 bytes (the runtime's mono output) to samples. -/
private def decodeF64LE (b : ByteArray) : Array Float := Id.run do
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
private def renderSamples (planJson : String) (n : Nat) : IO (Except String (Array Float)) := do
  match Lean.Json.parse planJson with
  | .error e => pure (.error s!"parse: {e}")
  | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
    | .error e => pure (.error s!"ofWire: {e}")
    | .ok plan => match Tropical.Ir.EmitLlvm.emitKernel plan with
      | .error e => pure (.error s!"emitKernel: {e}")
      | .ok ir => do
        let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
        rt.loadIr ir planJson
        rt.process
        pure (.ok (decodeF64LE (← rt.outputBytes)))

/-- Compile the reversible probe patch, render `2*half` samples, assert the
    output is a bit-exact palindrome about index `half` (and non-silent). -/
private def runReversibility : IO Bool := do
  let half : Nat := 2048
  let n : Nat := 2 * half
  match ← compilePatch "patches/reversible_probe.json" .fused with
  | .error e => IO.println s!"  FAIL  reversibility  compile: {firstLine e}"; pure false
  | .ok planJson =>
    match ← renderSamples planJson n with
    | .error e => IO.println s!"  FAIL  reversibility  render: {firstLine e}"; pure false
    | .ok samples =>
      if samples.size < n then
        IO.println s!"  FAIL  reversibility  got {samples.size} samples (want {n})"
        pure false
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
          IO.println s!"  FAIL  reversibility  {mism} mismatched pairs (first k={firstBad})"
          pure false
        else if energy <= 1e-6 then
          IO.println s!"  FAIL  reversibility  signal is silent (energy {energy})"
          pure false
        else
          IO.println s!"  PASS  reversibility  bit-exact palindrome over {half-1} pairs (peak |x|={maxAbs}, energy={energy})"
          pure true

/-- Same palindrome witness pointed at `ThroughZeroFlanger`: the LFO that
    sweeps `delta` is itself a function of `tau`, so unfreezing the comb adds
    no state. A latched oscillator would diverge between the forward and
    reverse halves; this stays bit-exact, sweep and all. -/
private def runFlangerReversibility : IO Bool := do
  let half : Nat := 2048
  let n : Nat := 2 * half
  match ← compilePatch "patches/flanger_probe.json" .fused with
  | .error e => IO.println s!"  FAIL  flanger  compile: {firstLine e}"; pure false
  | .ok planJson =>
    match ← renderSamples planJson n with
    | .error e => IO.println s!"  FAIL  flanger  render: {firstLine e}"; pure false
    | .ok samples =>
      if samples.size < n then
        IO.println s!"  FAIL  flanger  got {samples.size} samples (want {n})"
        pure false
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
          IO.println s!"  FAIL  flanger  {mism} mismatched pairs (first k={firstBad})"
          pure false
        else if energy <= 1e-6 then
          IO.println s!"  FAIL  flanger  signal is silent (energy {energy})"
          pure false
        else
          IO.println s!"  PASS  flanger  bit-exact palindrome over {half-1} pairs (peak |x|={maxAbs}, energy={energy})"
          pure true

/-- Fixed-point clock substrate witness: `ClockPhasor(clk: clock())` must be
    bit-for-bit identical to `FixedPhasor` (the root clock `θ = sampleIndex <<
    32` has zero fraction, so the split-multiply collapses to `inc·n + off`).
    The probe outputs `FixedPhasor.phase − ClockPhasor.phase`; assert it is
    exactly zero at every sample. -/
private def runClockPhasorEquiv : IO Bool := do
  let n : Nat := 4096
  match ← compilePatch "patches/clock_phasor_probe.json" .fused with
  | .error e => IO.println s!"  FAIL  clock-phasor  compile: {firstLine e}"; pure false
  | .ok planJson =>
    match ← renderSamples planJson n with
    | .error e => IO.println s!"  FAIL  clock-phasor  render: {firstLine e}"; pure false
    | .ok samples =>
      if samples.size < n then
        IO.println s!"  FAIL  clock-phasor  got {samples.size} samples (want {n})"
        pure false
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
          IO.println s!"  FAIL  clock-phasor  {bad} nonzero samples (first k={firstBad}, max|Δ|={maxAbs})"
          pure false
        else
          IO.println "  PASS  clock-phasor  ClockPhasor(clock()) ≡ FixedPhasor bit-for-bit"
          pure true

/-- Per-oscillator reverse witness: `FixedSinOsc(clk: -θ)` is the negated
    forward sine, so `forward + reverse` cancels. Reports the residual (≈ Sin
    polynomial range-reduction asymmetry); asserts it is at most a small
    epsilon. -/
private def runClockReverseProbe : IO Bool := do
  let n : Nat := 4096
  match ← compilePatch "patches/clock_reverse_probe.json" .fused with
  | .error e => IO.println s!"  FAIL  clock-reverse  compile: {firstLine e}"; pure false
  | .ok planJson =>
    match ← renderSamples planJson n with
    | .error e => IO.println s!"  FAIL  clock-reverse  render: {firstLine e}"; pure false
    | .ok samples =>
      let mut maxAbs := 0.0
      for k in [0:samples.size] do
        if samples[k]!.abs > maxAbs then maxAbs := samples[k]!.abs
      if maxAbs < 1e-6 then
        IO.println s!"  PASS  clock-reverse  forward+reverse cancels (max|Δ|={maxAbs})"
        pure true
      else
        IO.println s!"  FAIL  clock-reverse  residual too large (max|Δ|={maxAbs})"
        pure false

-- ── CF-only enforcement: surface `reg`/`next` is unparseable ──────────────────
-- The Phase-1 guarantee, now STRUCTURAL: `reg`/`next` were deleted from the
-- surface grammar and the IR, so a program declaring them does not even parse
-- (the keywords are gone — `reg` lexes as a bare identifier and the statement
-- fails). A closed-form program (`Sin` — fold + temps, no reg) parses,
-- elaborates and strata-processes normally. The `Sin` case is the landmine pin
-- — emit-level SSA temps are not regs and must survive. Both are self-contained
-- (no instance deps), so they process standalone with a no-op external resolver.
private def cfOnlyRejectSrc : String :=
  "```tropical\nprogram CfProbe(step: float = 1) -> (acc: float) {\n  reg s = 0\n  acc = s\n  next s = s + step\n}\n```"

private def runCfOnly (name md : String) (expectReject : Bool) : IO Bool := do
  match Tropical.Parse.Surface.parseMarkdownProgram md with
  | .error e =>
    if expectReject then
      IO.println s!"  PASS  cf-only/{name}  rejected per-sample state at parse"; pure true
    else
      IO.println s!"  FAIL  cf-only/{name}  parse: {firstLine e}"; pure false
  | .ok prog =>
    match Tropical.Ir.elaborateInto {} prog (some fun _ => none) with
    | .error e =>
      if expectReject then
        IO.println s!"  PASS  cf-only/{name}  rejected per-sample state at elaboration"; pure true
      else
        IO.println s!"  FAIL  cf-only/{name}  unexpected reject: {firstLine e.message}"; pure false
    | .ok (arena, root) =>
      match Tropical.Ir.Strata.run { upto := 5 } arena root with
      | .error e =>
        IO.println s!"  FAIL  cf-only/{name}  strata error: {firstLine e.message}"; pure false
      | .ok _ =>
        if expectReject then
          IO.println s!"  FAIL  cf-only/{name}  compiled but should be rejected"; pure false
        else
          IO.println s!"  PASS  cf-only/{name}  compiles (temps survive)"; pure true

-- ── (h) ArrowWarp arrow laws: algebraically-equal warps ⇒ bit-identical audio ─
-- Slice 3. The warp arrow laws are certified as AUDIO goldens: build the law's
-- LHS and RHS as two ArrowWarp carrier programs (a single FixedSinOsc clocked at
-- two algebraically-equal clock expressions), render both, and assert the
-- rendered audio is byte-identical (SHA256). The laws hold byte-exactly because
-- warps are integer add/sub on the Q32.32 fixed-point clock — exact and
-- associative — so the two sides feed the oscillator a bit-identical int64 clock
-- and render bit-identical audio, EVEN THOUGH the emitted plans differ (no
-- algebraic tree normalization). The render bridge reuses the production session
-- path: buildClockCarrier (ArrowWarp) → Strata.run → Core.check → compileSession
-- (the carrier as a one-instance root wired to the dac) → FlatPlan → renderIrBytes.

open Tropical.Ir (Arena ProgramIdx)

/-- Read + strictly decode a serialized ParsedProgram (mirrors Diffcli.readParsed). -/
private def arrowReadParsed (path : String) : IO (Except String Tropical.Parse.Program) := do
  let text ← IO.FS.readFile path
  pure <| do
    let jv ← Tropical.Parse.JsonV.parse text |>.mapError (s!"JSON parse error: {·}")
    Tropical.Parse.decodeProgram jv

/-- Elaborate the whole stdlib bridge chain in manifest order (mirrors
    Diffcli.elabChain), so `FixedSinOsc` (and its transitive voice deps) are in
    the arena for `buildClockCarrier` to link against. Done once and reused. -/
private def arrowElabStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) := do
  let manifestText ← IO.FS.readFile "stdlib/parsed/manifest.json"
  let names : Except String (Array String) := do
    let jv ← Tropical.Parse.JsonV.parse manifestText |>.mapError (s!"manifest parse error: {·}")
    let some (Tropical.Parse.JsonV.arr items) := jv.getField? "programs"
      | .error "manifest missing 'programs' array"
    items.mapM fun | .str s => .ok s | _ => .error "manifest 'programs' entries must be strings"
  match names with
  | .error e => pure (.error e)
  | .ok names => do
    let mut arena : Arena := {}
    let mut resolved : Array (String × ProgramIdx) := #[]
    for name in names do
      match ← arrowReadParsed s!"stdlib/parsed/{name}.json" with
      | .error e => return .error s!"{name}.json: {e}"
      | .ok prog =>
        let r := resolved
        match Tropical.Ir.elaborateInto arena prog (some fun n => (r.find? (·.1 == n)).map (·.2)) with
        | .error e => return .error s!"{name}: {e.message}"
        | .ok (arena', idx) => arena := arena'; resolved := resolved.push (name, idx)
    pure (.ok (arena, resolved))

/-- Build one ArrowWarp clock carrier (named `name`, clocked at `clkE`) into a
    runnable `FlatPlan` via the production session path. -/
private def compileArrowCarrier (arena : Arena) (resolved : Array (String × ProgramIdx))
    (name : String) (clkE : Tropical.ArrowWarp.Clock) :
    Except String Tropical.Plan.FlatPlan := do
  let (arena', idx) ← Tropical.ArrowWarp.buildClockCarrier name clkE arena resolved
  let (arena'', root') ← (Tropical.Ir.Strata.run { upto := 5 } arena' idx).mapError (·.message)
  let core ← Tropical.Ir.Core.check arena'' root'
  -- The carrier is the synthetic session root, wired straight to the dac at its
  -- `out` port (`__root__.out`). No session wires, no params, no inputs.
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    mode := .fused }
  Tropical.Compile.compileSession input

/-- Certify one warp law: build LHS and RHS carriers, render both, assert the
    rendered audio is byte-identical (SHA256). Also reports whether the emitted
    plans are byte-identical (EXPECTED NO — the algebra is exact in the clock,
    not normalized in the tree). PASS iff the audio hashes match. -/
private def runArrowLaw (name : String)
    (lhsClk rhsClk : Tropical.ArrowWarp.Clock)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match compileArrowCarrier arena resolved s!"{name}_lhs" lhsClk,
        compileArrowCarrier arena resolved s!"{name}_rhs" rhsClk with
  | .error e, _ => IO.println s!"  FAIL  arrow-law/{name}  build lhs: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  arrow-law/{name}  build rhs: {firstLine e}"; pure false
  | .ok lhsPlan, .ok rhsPlan =>
    match lhsPlan.toWire, rhsPlan.toWire with
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  arrow-law/{name}  toWire: {firstLine e}"; pure false
    | .ok lhsWire, .ok rhsWire =>
      let plansIdentical := lhsWire.compress == rhsWire.compress
      match ← renderIrBytes lhsPlan, ← renderIrBytes rhsPlan with
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  arrow-law/{name}  render: {firstLine e}"; pure false
      | .ok lhsBytes, .ok rhsBytes =>
        let lhsHash ← sha256Hex lhsBytes
        let rhsHash ← sha256Hex rhsBytes
        let planNote := if plansIdentical then "plans identical" else "plans differ (expected)"
        if lhsHash == rhsHash then
          IO.println s!"  PASS  arrow-law/{name}  audio ≡ {lhsHash.take 16} ({planNote})"
          pure true
        else
          IO.println s!"  FAIL  arrow-law/{name}  audio differs: lhs {lhsHash.take 16} rhs {rhsHash.take 16} ({planNote})"
          pure false

-- ── (h) slice 4 — the cartesian diagonal / fan-out law + its COST story ───────
-- The diagonal `Δ = id &&& id`: one source fanned into two differently-warped
-- flangers ≡ two independent (source+flanger) pairs. AUDIO: byte-identical (the
-- duplicated source is the same closed form fed the same clock, so sharing is
-- invisible to the output). COST (the real content): whether within-program
-- strata CSE collapses the duplicated osc(clk) so both forms reach the SAME
-- minimal DAG — i.e. "fan-out is a pure let". We report eval/register/instruction
-- counts for both forms and whether the emitted plans are byte-identical (the
-- program name does NOT flow into the plan — the root instance is `__root__` —
-- so plan byte-identity is a pure structural test of the CSE collapse).

/-- Count instructions across an InstanceFunction tree (body + nested children). -/
private partial def countFnInstrs (f : Tropical.Plan.InstanceFunction) : Nat :=
  f.instructions.size + f.children.foldl (fun acc c => acc + countFnInstrs c) 0

/-- Total instruction count of a FlatPlan (all instance functions, recursive). -/
private def planInstrCount (p : Tropical.Plan.FlatPlan) : Nat :=
  p.instanceFunctions.foldl (fun acc f => acc + countFnInstrs f) 0

/-- Strata-process an already-built carrier program, then compile it via the
    production session path. Returns (post-strata binderCount, post-strata
    declCount, FlatPlan) — the post-strata program is the CSE'd DAG, so its
    binderCount is the count of distinct shared subexpressions. -/
private def diagStrataCompile (arena : Arena) (idx : ProgramIdx) :
    Except String (Nat × Nat × Tropical.Plan.FlatPlan) := do
  let (arena'', root') ← (Tropical.Ir.Strata.run { upto := 5 } arena idx).mapError (·.message)
  let some prog := arena''.program? root'
    | .error "diagonal: post-strata root program index out of range"
  let core ← Tropical.Ir.Core.check arena'' root'
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    mode := .fused }
  let plan ← Tropical.Compile.compileSession input
  pure (prog.binderCount, prog.decls.size, plan)

/-- Certify the diagonal law: build SHARED (one fanned source) and INDEPENDENT
    (two sources) carriers, assert their rendered audio is byte-identical (the
    law), and report the cost story — per-form eval/register/instruction counts
    and whether the emitted plans are byte-identical (CSE collapse). PASS iff the
    audio hashes match. -/
private def runDiagonalLaw (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.ArrowWarp.buildSharedDiagonal arena resolved,
        Tropical.ArrowWarp.buildIndependentDiagonal arena resolved with
  | .error e, _ => IO.println s!"  FAIL  arrow-law/diagonal  build shared: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  arrow-law/diagonal  build independent: {firstLine e}"; pure false
  | .ok (arenaS, idxS), .ok (arenaI, idxI) =>
    -- Pre-strata instance counts: the visible structural difference (5 vs 6).
    let preShared := (arenaS.program? idxS).map (·.decls.size) |>.getD 0
    let preIndep := (arenaI.program? idxI).map (·.decls.size) |>.getD 0
    match diagStrataCompile arenaS idxS, diagStrataCompile arenaI idxI with
    | .error e, _ => IO.println s!"  FAIL  arrow-law/diagonal  compile shared: {firstLine e}"; pure false
    | _, .error e => IO.println s!"  FAIL  arrow-law/diagonal  compile independent: {firstLine e}"; pure false
    | .ok (bcS, dcS, planS), .ok (bcI, dcI, planI) =>
      let instrS := planInstrCount planS
      let instrI := planInstrCount planI
      match planS.toWire, planI.toWire with
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  arrow-law/diagonal  toWire: {firstLine e}"; pure false
      | .ok wireS, .ok wireI =>
        let plansIdentical := wireS.compress == wireI.compress
        match ← renderIrBytes planS, ← renderIrBytes planI with
        | .error e, _ | _, .error e =>
          IO.println s!"  FAIL  arrow-law/diagonal  render: {firstLine e}"; pure false
        | .ok bytesS, .ok bytesI =>
          let hashS ← sha256Hex bytesS
          let hashI ← sha256Hex bytesI
          -- The plans being byte-identical IS the definitive collapse (the
          -- carrier program name never reaches the plan — the root instance is
          -- `__root__`). NB the post-strata binder DAGs DIFFER (independent
          -- carries the extra inlined osc(clk) body's binders); the duplicate
          -- source is deduped at EMIT (compileResolved value-numbering), not in
          -- the strata binder DAG — yet both reach the same minimal kernel.
          IO.println s!"        cost  shared:      pre-strata insts={preShared} post-strata binders={bcS} decls={dcS} plan-instrs={instrS} regs={planS.registerCount}"
          IO.println s!"        cost  independent: pre-strata insts={preIndep} post-strata binders={bcI} decls={dcI} plan-instrs={instrI} regs={planI.registerCount}"
          IO.println s!"        cost  plans byte-identical: {plansIdentical}  ·  CSE collapsed both to same {instrS}-instr/{planS.registerCount}-reg DAG: {plansIdentical && instrS == instrI}  (post-strata binders {bcS} vs {bcI} differ — dedup is at emit)"
          if hashS == hashI then
            IO.println s!"  PASS  arrow-law/diagonal  audio ≡ {hashS.take 16} (shared ≡ independent)"
            pure true
          else
            IO.println s!"  FAIL  arrow-law/diagonal  audio differs: shared {hashS.take 16} independent {hashI.take 16}"
            pure false

def main (args : List String) : IO UInt32 := do
  let writeMode := args.contains "--write"
  let mut failed := 0
  let mut total := 0

  -- ── (a) Patch audio goldens (tests/golden/*.hash) ──────────────────────────
  IO.println "patch goldens:"
  for name in ← sortedNames "tests/golden" ".hash" do
    let patchPath := s!"patches/{name}.json"
    if ← System.FilePath.pathExists patchPath then
      total := total + 1
      if !(← runGolden writeMode name patchPath s!"tests/golden/{name}.hash") then failed := failed + 1

  -- ── (b) Migration audio goldens (tests/golden/migration/*.json) ────────────
  IO.println "migration goldens:"
  for fixture in ← sortedNames "tests/golden/migration" ".json" do
    let goldenText ← IO.FS.readFile s!"tests/golden/migration/{fixture}.json"
    let expected? : Option String := do
      let g ← (Lean.Json.parse goldenText).toOption
      let h ← (g.getObjVal? "hash").toOption
      h.getStr?.toOption
    let fixText ← IO.FS.readFile s!"tests/fixtures/flat_plan/{fixture}.json"
    let input? : Option Lean.Json := do
      let f ← (Lean.Json.parse fixText).toOption
      (f.getObjVal? "input").toOption
    match expected?, input? with
    | some expected, some input =>
      total := total + 1
      let tmpPatch := "/tmp/tropicaltest-fixture.json"
      IO.FS.writeFile tmpPatch input.compress
      if !(← checkGoldenHash fixture tmpPatch expected) then failed := failed + 1
    | _, _ => IO.println s!"  SKIP  {fixture}  (missing hash/input)"

  -- ── (c) Synthetic op-coverage: EmitLlvm over the rare ops, frozen hash ─────
  -- The patch corpus exercises 24 of 29 ops; this funnels the rest
  -- (GreaterEq, NotEqual, Or, BitOr, BitNot, FloorDiv, Sqrt, Floor, Ceil,
  -- Abs, ToInt, ToBool, Not) through one sink. The expected hash was frozen
  -- from the C++ plan compiler before it was retired (Phase 2), so this
  -- catches any EmitLlvm regression on those ops now that the differential
  -- oracle is gone.
  IO.println "op coverage (EmitLlvm, golden hash):"
  total := total + 1
  match ← renderIrBytes opCoveragePlan with
  | .error e => IO.println s!"  FAIL  op-coverage  {firstLine e}"; failed := failed + 1
  | .ok bytes =>
    let got ← sha256Hex bytes
    let expected := "9d47595cec2e690076b395ca072c03fc20cb8ba838a7b8ac60c16a91da0ea1b8"
    if got == expected then IO.println "  PASS  op-coverage"
    else
      IO.println s!"  FAIL  op-coverage  expected {expected.take 16} got {got.take 16}"
      failed := failed + 1

  -- ── (d) let-binding serialization order (ordered-array round-trip) ─────────
  IO.println "let serialization order:"
  total := total + 1
  if !(← runLetRoundtrip) then failed := failed + 1

  -- ── (e) Reversibility: closed-form-in-τ ⇒ palindromic render ───────────────
  IO.println "reversibility (closed-form-in-tau palindrome):"
  total := total + 1
  if !(← runReversibility) then failed := failed + 1
  total := total + 1
  if !(← runFlangerReversibility) then failed := failed + 1
  total := total + 1
  if !(← runClockPhasorEquiv) then failed := failed + 1
  total := total + 1
  if !(← runClockReverseProbe) then failed := failed + 1

  -- ── (f) CF goldens (tests/golden/cf/*.hash) — the closed-form corpus ───────
  -- The corpus that must stay green through every phase of the CF-only
  -- migration (it is rendered via the same path as the legacy goldens but only
  -- ever holds register-free, closed-form-in-τ patches). Same shape as (a):
  -- scan the dir, compile patches/<name>.json fused, freeze the render hash.
  IO.println "cf goldens:"
  for name in ← sortedNames "tests/golden/cf" ".hash" do
    let patchPath := s!"patches/{name}.json"
    if ← System.FilePath.pathExists patchPath then
      total := total + 1
      if !(← runGolden writeMode name patchPath s!"tests/golden/cf/{name}.hash") then failed := failed + 1

  -- ── (g) CF-only enforcement: cfOnly strata mode rejects per-sample state ───
  IO.println "cf-only enforcement (reg/next unrepresentable):"
  total := total + 1
  if !(← runCfOnly "CfProbe" cfOnlyRejectSrc (expectReject := true)) then failed := failed + 1
  total := total + 1
  if !(← runCfOnly "Sin" (← IO.FS.readFile "stdlib/Sin.md") (expectReject := false)) then
    failed := failed + 1

  -- ── (h) ArrowWarp arrow laws (slice 3): warp algebra ≡ in rendered audio ────
  IO.println "arrow laws (warp algebra ≡ byte-identical audio):"
  match ← arrowElabStdlib with
  | .error e =>
    IO.println s!"  FAIL  arrow-laws  elaborate stdlib: {firstLine e}"
    total := total + 3; failed := failed + 3
  | .ok (arena, resolved) =>
    -- Law 1 — inverse/cancellation:  warp(back δ) ⋙ warp(fwd δ) = id
    total := total + 1
    if !(← runArrowLaw "inverse"
          Tropical.ArrowWarp.invLawLhsClock Tropical.ArrowWarp.invLawRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 2 — additive delay/functoriality:
    --   warp(back δ₁) ⋙ warp(back δ₂) = warp(back (δ₁+δ₂))
    total := total + 1
    if !(← runArrowLaw "additive"
          Tropical.ArrowWarp.addLawLhsClock Tropical.ArrowWarp.addLawRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 3 — the cartesian diagonal / fan-out: one source ⋙ (flanger δ₁ &&&
    --   flanger δ₂) ≡ two independent (source+flanger) pairs (+ the COST story).
    total := total + 1
    if !(← runDiagonalLaw arena resolved) then
      failed := failed + 1

  IO.println ""
  IO.println s!"{total - failed}/{total} passed"
  return if failed == 0 then 0 else 1
