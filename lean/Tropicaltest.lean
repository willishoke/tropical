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
import Tropical.Ir.CompileResolved
import Tropical.Compile
import Tropical.EmitArrow
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

/-- Compile a patch via the C4 DIRECT session-root path (`sessionToResolvedRoot`,
    no `sessionToParsed → elaborate`). For the round-trip-deletion equivalence
    gate: this plan must equal `compilePatch`'s (the elaborate path). -/
def compilePatchArrow (path : String) (mode : Tropical.Plan.CompilationMode) :
    IO (Except String String) := do
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM String := do
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str path)])
    Tropical.Engine.compileMirrorPlanViaArrow env mode
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

-- ── (h) EmitArrow arrow laws: algebraically-equal warps ⇒ bit-identical audio ─
-- Slice 3. The warp arrow laws are certified as AUDIO goldens: build the law's
-- LHS and RHS as two EmitArrow carrier programs (a single FixedSinOsc clocked at
-- two algebraically-equal clock expressions), render both, and assert the
-- rendered audio is byte-identical (SHA256). The laws hold byte-exactly because
-- warps are integer add/sub on the Q32.32 fixed-point clock — exact and
-- associative — so the two sides feed the oscillator a bit-identical int64 clock
-- and render bit-identical audio, EVEN THOUGH the emitted plans differ (no
-- algebraic tree normalization). The render bridge reuses the production session
-- path: buildClockCarrier (EmitArrow) → Strata.run → Core.check → compileSession
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

/-- Build one EmitArrow clock carrier (named `name`, clocked at `clkE`) into a
    runnable `FlatPlan` via the production session path. -/
private def compileArrowCarrier (arena : Arena) (resolved : Array (String × ProgramIdx))
    (name : String) (clkE : Tropical.EmitArrow.Clock) :
    Except String Tropical.Plan.FlatPlan := do
  let (arena', idx) ← Tropical.EmitArrow.buildClockCarrier name clkE arena resolved
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

-- ── (h′) EmitArrow corpus gate: EmitArrow's emit ≡ strata's emit, per program ─
-- The cutover (phase C1) reproduces the corpus one program at a time. This is
-- the reusable instrument: given a resolved program (built by an EmitArrow
-- constructor) and a stdlib program NAME, emit BOTH through the production
-- per-program recipe — strata (full) → Core.check → compileResolved → wire —
-- and compare byte-for-byte. The stdlib side is exactly what `diffcli
-- emit-stdlib <Name>` produces (the target strata produces TODAY); the
-- EmitArrow side is what the cutover would emit instead. Byte-identity proves
-- EmitArrow covers strata's job for that program — the slices-1/2 byte-gate,
-- generalized to any program, run as a tropicaltest assertion rather than an
-- external `diff` of two diffcli verbs.

/-- The production per-program emit recipe (the `diffcli emit-*` body): strata
    (all ported passes, inline) → Core.check → compileResolved → wire JSON. The
    canonical `tropical_plan_5`-per-instance bytes a program emits today. -/
private def emitResolvedWire (arena : Arena) (idx : ProgramIdx) : Except String String := do
  let (arena', root') ← (Tropical.Ir.Strata.run
      { upto := Tropical.Ir.Strata.portedPasses, inlineNested := true } arena idx).mapError (·.message)
  let core ← Tropical.Ir.Core.check arena' root'
  let plan ← Tropical.Ir.CompileResolved.compileResolved core
  let wire ← plan.toWire
  pure wire.compress

/-- THE CORPUS GATE (reusable). Build a program with `builder` (an EmitArrow
    constructor over the elaborated stdlib arena), then assert its emit is
    byte-identical to stdlib program `stdName`'s emit — the form strata produces
    today (`diffcli emit-stdlib {stdName}`). `arena`/`resolved` come from
    `arrowElabStdlib`; `stdName` is emitted from the ORIGINAL arena (the builder
    only appends), so the comparison is EmitArrow-emit vs strata-emit of the same
    target. This covers the corpus one program at a time. -/
private def runEmitCorpusGate (label stdName : String)
    (arena : Arena) (resolved : Array (String × ProgramIdx))
    (builder : Arena → Array (String × ProgramIdx) → Except String (Arena × ProgramIdx)) :
    IO Bool := do
  let some (_, stdIdx) := resolved.find? (·.1 == stdName)
    | IO.println s!"  FAIL  corpus-gate/{label}  stdlib '{stdName}' not in elaborated chain"
      pure false
  match builder arena resolved with
  | .error e => IO.println s!"  FAIL  corpus-gate/{label}  build: {firstLine e}"; pure false
  | .ok (arena', idx) =>
    match emitResolvedWire arena' idx, emitResolvedWire arena stdIdx with
    | .error e, _ => IO.println s!"  FAIL  corpus-gate/{label}  emit EmitArrow: {firstLine e}"; pure false
    | _, .error e => IO.println s!"  FAIL  corpus-gate/{label}  emit stdlib {stdName}: {firstLine e}"; pure false
    | .ok got, .ok want =>
      if got == want then
        IO.println s!"  PASS  corpus-gate/{label}  EmitArrow ≡ emit-stdlib {stdName} ({got.length}B)"
        pure true
      else
        IO.println s!"  FAIL  corpus-gate/{label}  EmitArrow ≠ emit-stdlib {stdName} (EmitArrow {got.length}B, stdlib {want.length}B)"
        pure false

/-- Certify one warp law: build LHS and RHS carriers, render both, assert the
    rendered audio is byte-identical (SHA256). Also reports whether the emitted
    plans are byte-identical (EXPECTED NO — the algebra is exact in the clock,
    not normalized in the tree). PASS iff the audio hashes match. -/
private def runArrowLaw (name : String)
    (lhsClk rhsClk : Tropical.EmitArrow.Clock)
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
  match Tropical.EmitArrow.buildSharedDiagonal arena resolved,
        Tropical.EmitArrow.buildIndependentDiagonal arena resolved with
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

-- ── (h) slice 5 — REVERSE (the moat) as warp(neg): involution, reverse-swaps- ──
-- delay, and reverse-equivariance of the symmetric flanger. Laws 1-2 reuse
-- `runArrowLaw` (byte-identical: the clock side is exact int64). Law 3 cannot be
-- byte-identical in float — under reverse the ±δ taps swap tree slot, so the
-- value weighted-sum reassociates `(A+B)+C` vs `(A+C)+B`, and float add is not
-- associative — so we assert the DENOTATIONAL form (max|Δ| < ε) and REPORT the
-- byte-identity + max|Δ| as the finding. It would be byte-exact with a fixed-
-- point value carrier (the clock side, laws 1-2, is already exact).

/-- Certify reverse-equivariance of the symmetric flanger:
    `warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)`. Builds both flanger carriers
    (neg OUTER vs INNER, swapping the ±δ tap slots), renders both, asserts the
    DENOTATIONAL law (max|Δ| < ε), and reports byte-identity + max|Δ|. PASS iff
    max|Δ| < ε and the signal is non-silent. -/
private def runReverseFlangerCommute
    (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let eps := 1e-6
  match Tropical.EmitArrow.buildReverseThenFlanger arena resolved,
        Tropical.EmitArrow.buildFlangerThenReverse arena resolved with
  | .error e, _ => IO.println s!"  FAIL  arrow-law/reverse-flanger-commute  build lhs: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  arrow-law/reverse-flanger-commute  build rhs: {firstLine e}"; pure false
  | .ok (arenaL, idxL), .ok (arenaR, idxR) =>
    match diagStrataCompile arenaL idxL, diagStrataCompile arenaR idxR with
    | .error e, _ => IO.println s!"  FAIL  arrow-law/reverse-flanger-commute  compile lhs: {firstLine e}"; pure false
    | _, .error e => IO.println s!"  FAIL  arrow-law/reverse-flanger-commute  compile rhs: {firstLine e}"; pure false
    | .ok (_, _, planL), .ok (_, _, planR) =>
      match ← renderIrBytes planL, ← renderIrBytes planR with
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  arrow-law/reverse-flanger-commute  render: {firstLine e}"; pure false
      | .ok bytesL, .ok bytesR =>
        let hashL ← sha256Hex bytesL
        let hashR ← sha256Hex bytesR
        let byteIdentical := hashL == hashR
        let samplesL := decodeF64LE bytesL
        let samplesR := decodeF64LE bytesR
        let n := min samplesL.size samplesR.size
        let mut maxAbs := 0.0
        let mut energy := 0.0
        let mut bitDiff := 0
        for k in [0:n] do
          let d := (samplesL[k]! - samplesR[k]!).abs
          if d > maxAbs then maxAbs := d
          if samplesL[k]!.toBits != samplesR[k]!.toBits then bitDiff := bitDiff + 1
          energy := energy + samplesL[k]! * samplesL[k]!
        -- max|Δ| is sub-ULP-scale; Float.toString rounds it to 0.000000. Show the
        -- scaled value (×10¹⁵, i.e. femto-units) so the ULP magnitude is legible.
        IO.println s!"        finding  byte-identical: {byteIdentical}  ·  bit-differing samples: {bitDiff}/{n}  ·  max|Δ|={maxAbs} (={maxAbs * 1e15}e-15, ε={eps})  ·  lhs energy={energy}"
        IO.println s!"        finding  ±δ taps swap slot ⇒ float value-sum reassociates ((A+B)+C vs (A+C)+B); exact on the (fixed-point) clock (laws 1-2), float-tolerance on the (float) value sum — byte-exact with a fixed-point value carrier"
        if energy <= 1e-6 then
          IO.println s!"  FAIL  arrow-law/reverse-flanger-commute  signal silent (energy={energy})"
          pure false
        else if maxAbs < eps then
          IO.println s!"  PASS  arrow-law/reverse-flanger-commute  denotational ≡ (max|Δ|={maxAbs} < ε; byte-identical={byteIdentical})"
          pure true
        else
          IO.println s!"  FAIL  arrow-law/reverse-flanger-commute  max|Δ|={maxAbs} ≥ ε={eps}"
          pure false

-- ── (h) slice 6 — a FIXED-POINT VALUE carrier: slice-5's reassociation ────────
-- failure becomes BYTE-IDENTICAL. The SAME warp combinators, but instantiated at
-- an integer Q0.32 saw source (`fixedPhase`) mixed with INTEGER right-shifts
-- (`fixedFlangerSum`) instead of a float oscillator + float weighted-sum. Integer
-- add is associative AND commutative, so the past/ahead slot swap that reverse
-- induces (the float reassociation `(A+B)+C ≠ (A+C)+B`, slice 5's 1271/4096) is
-- invisible — the law is now byte-exact. No type-system / engine-float change;
-- the carrier is integer `Expr` with one `toFloat` scale at the DAC boundary.

/-- Certify one single-source fixed-point warp law: build LHS and RHS fixed-point
    source carriers (`fixedOut(fixedPhase(clkE))`) at two algebraically-equal
    clocks, render both, assert the audio is BYTE-IDENTICAL (SHA256). The clock
    side is exact int64, so these hold byte-exactly — the fixed-point analog of
    `runArrowLaw`'s float laws (which also pass) over the integer source. -/
private def runFixedSourceLaw (name : String)
    (lhsClk rhsClk : Tropical.EmitArrow.Clock) (arena : Arena) : IO Bool := do
  let (arenaL, idxL) := Tropical.EmitArrow.buildFixedSourceCarrier s!"{name}_lhs" lhsClk arena
  let (arenaR, idxR) := Tropical.EmitArrow.buildFixedSourceCarrier s!"{name}_rhs" rhsClk arena
  match diagStrataCompile arenaL idxL, diagStrataCompile arenaR idxR with
  | .error e, _ => IO.println s!"  FAIL  arrow-law/{name}  compile lhs: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  arrow-law/{name}  compile rhs: {firstLine e}"; pure false
  | .ok (_, _, planL), .ok (_, _, planR) =>
    match ← renderIrBytes planL, ← renderIrBytes planR with
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  arrow-law/{name}  render: {firstLine e}"; pure false
    | .ok bytesL, .ok bytesR =>
      let hashL ← sha256Hex bytesL
      let hashR ← sha256Hex bytesR
      if hashL == hashR then
        IO.println s!"  PASS  arrow-law/{name}  audio ≡ {hashL.take 16} (byte-identical)"
        pure true
      else
        IO.println s!"  FAIL  arrow-law/{name}  audio differs: lhs {hashL.take 16} rhs {hashR.take 16}"
        pure false

/-- THE GATE: certify reverse-equivariance of the FIXED-POINT flanger
    `warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)` — the exact law slice 5 found
    byte-DIFFERENT in float (1271/4096). Builds both carriers (neg OUTER vs INNER,
    swapping the ±δ tap slots) over the INTEGER source + integer shift-add mix,
    renders both, and asserts the audio is BYTE-IDENTICAL (SHA256). Reports the
    differing-sample count (must be 0/N vs slice 5's 1271/4096). PASS iff
    byte-identical and non-silent. -/
private def runReverseFlangerCommuteFixedpoint (arena : Arena) : IO Bool := do
  let (arenaL, idxL) := Tropical.EmitArrow.buildReverseThenFixedFlanger arena
  let (arenaR, idxR) := Tropical.EmitArrow.buildFixedFlangerThenReverse arena
  match diagStrataCompile arenaL idxL, diagStrataCompile arenaR idxR with
  | .error e, _ => IO.println s!"  FAIL  arrow-law/reverse-flanger-commute-fixedpoint  compile lhs: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  arrow-law/reverse-flanger-commute-fixedpoint  compile rhs: {firstLine e}"; pure false
  | .ok (_, _, planL), .ok (_, _, planR) =>
    match ← renderIrBytes planL, ← renderIrBytes planR with
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  arrow-law/reverse-flanger-commute-fixedpoint  render: {firstLine e}"; pure false
    | .ok bytesL, .ok bytesR =>
      let hashL ← sha256Hex bytesL
      let hashR ← sha256Hex bytesR
      let byteIdentical := hashL == hashR
      let samplesL := decodeF64LE bytesL
      let samplesR := decodeF64LE bytesR
      let n := min samplesL.size samplesR.size
      let mut bitDiff := 0
      let mut energy := 0.0
      for k in [0:n] do
        if samplesL[k]!.toBits != samplesR[k]!.toBits then bitDiff := bitDiff + 1
        energy := energy + samplesL[k]! * samplesL[k]!
      IO.println s!"        gate  byte-identical: {byteIdentical}  ·  bit-differing samples: {bitDiff}/{n} (slice-5 float was 1271/4096)  ·  lhs energy={energy}"
      IO.println s!"        gate  integer add is associative — ±δ tap slot swap leaves the Q0.32 mix bit-identical; one toFloat·/2³² scale at the boundary"
      if energy <= 1e-6 then
        IO.println s!"  FAIL  arrow-law/reverse-flanger-commute-fixedpoint  signal silent (energy={energy})"
        pure false
      else if byteIdentical then
        IO.println s!"  PASS  arrow-law/reverse-flanger-commute-fixedpoint  audio ≡ {hashL.take 16} ({bitDiff}/{n} differing — byte-exact)"
        pure true
      else
        IO.println s!"  FAIL  arrow-law/reverse-flanger-commute-fixedpoint  audio differs ({bitDiff}/{n}): lhs {hashL.take 16} rhs {hashR.take 16}"
        pure false

-- ── (h″) The convolution stress test — the bubble, EXECUTED, with a NON-FACADE
-- oracle. An FIR filter is fan-out + clock-warps + scale + sum (a convolution IS
-- the flanger with more taps). We compute the convolution TWO independent ways
-- and demand they agree:
--   tropical: each tap warps the CLOCK by j samples (j·2³² in Q32.32); the
--     oscillator is evaluated at the warped clock, weighted, summed — the bubble
--     doing the work inside the kernel.
--   oracle:   render the BARE oscillator once, then shift the resulting Float
--     array by j, scale by kⱼ, sum — ordinary Lean arithmetic that NEVER touches
--     the warp lowering.
-- Agreement proves "warp the clock by j samples" realizes "delay the output by j
-- samples" IN THE ACTUAL COMPILER, checked by an oracle independent of the
-- lowering (this is what defeats correct-by-facade — eval-walking the same term
-- could not). The filter-effect figure confirms the FIR is non-degenerate.

/-- A j-sample clock delay: subtract `j·2³²` (Q32.32) from the clock. `j = 0` is
    identity (`sub c 0 = c`). -/
private def firShift (j : Nat) : Tropical.EmitArrow.Clock → Tropical.EmitArrow.Clock :=
  fun c => Tropical.EmitArrow.sub c
    (Tropical.EmitArrow.toIntE (Tropical.EmitArrow.lit (Int.ofNat j * 4294967296)))

/-- 3-tap FIR `[0.25, 0.5, 0.25]` at integer-sample delays `[0,1,2]`, as a bank
    of CLOCK warps over the closed-form 12 kHz voice (pitch high enough that the
    lowpass visibly attenuates). -/
private def firTaps : Array Tropical.EmitArrow.Tap := #[
  { name := "k0", warp := fun c => c, weight := Tropical.EmitArrow.lit 25 2 },
  { name := "k1", warp := firShift 1, weight := Tropical.EmitArrow.lit 5 1 },
  { name := "k2", warp := firShift 2, weight := Tropical.EmitArrow.lit 25 2 } ]

/-- The bare voice: a single identity tap, weight 1 — the source samples the
    oracle convolves by hand. -/
private def bareTaps : Array Tropical.EmitArrow.Tap := #[
  { name := "x", warp := fun c => c, weight := Tropical.EmitArrow.lit 1 } ]

/-- Compile a closed-form tap-bank carrier (the 12 kHz voice) to a runnable
    `FlatPlan` via the production session path — same recipe as
    `compileArrowCarrier`. -/
private def compileTapCarrier (arena : Arena) (resolved : Array (String × ProgramIdx))
    (name : String) (taps : Array Tropical.EmitArrow.Tap) :
    Except String Tropical.Plan.FlatPlan := do
  let (arena', idx) ← Tropical.EmitArrow.buildTapCarrier name
    Tropical.EmitArrow.litPitch12kVoice taps arena resolved
  let (arena'', root') ← (Tropical.Ir.Strata.run { upto := 5 } arena' idx).mapError (·.message)
  let core ← Tropical.Ir.Core.check arena'' root'
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    mode := .fused }
  Tropical.Compile.compileSession input

/-- Render a `FlatPlan` to exactly `n` contiguous mono samples (buffer = n, no
    fade), like `renderSamples` but from an in-hand plan. -/
private def renderPlanSamples (plan : Tropical.Plan.FlatPlan) (n : Nat) :
    IO (Except String (Array Float)) := do
  match plan.toWire, Tropical.Ir.EmitLlvm.emitKernel plan with
  | .ok manifest, .ok ir =>
    let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
    rt.loadIr ir manifest.compress
    rt.process
    pure (.ok (decodeF64LE (← rt.outputBytes)))
  | .error e, _ => pure (.error s!"toWire: {e}")
  | _, .error e => pure (.error s!"emitKernel: {e}")

/-- THE NON-FACADE GATE: tropical's clock-warped FIR ≡ an array-shift convolution
    of the independently-rendered bare oscillator. -/
private def runConvolutionOracle (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let kernel : Array Float := #[0.25, 0.5, 0.25]   -- delays 0,1,2
  let maxDelay := kernel.size - 1
  match compileTapCarrier arena resolved "Fir3" firTaps,
        compileTapCarrier arena resolved "Bare" bareTaps with
  | .error e, _ => IO.println s!"  FAIL  convolution-oracle  build fir: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  convolution-oracle  build bare: {firstLine e}"; pure false
  | .ok firPlan, .ok barePlan =>
    match ← renderPlanSamples firPlan n, ← renderPlanSamples barePlan n with
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  convolution-oracle  render: {firstLine e}"; pure false
    | .ok got, .ok x =>
      let mut maxAbs : Float := 0.0
      let mut filterEffect : Float := 0.0
      let mut energy : Float := 0.0
      for t in [maxDelay:n] do
        let mut acc : Float := 0.0
        for j in [0:kernel.size] do
          acc := acc + kernel[j]! * x[t - j]!
        let g := got[t]!
        let d := (g - acc).abs
        if d > maxAbs then maxAbs := d
        let fe := (g - x[t]!).abs
        if fe > filterEffect then filterEffect := fe
        energy := energy + g * g
      let eps : Float := 1e-9
      if energy <= 1e-6 then
        IO.println s!"  FAIL  convolution-oracle  signal silent (energy={energy})"
        pure false
      else if maxAbs < eps then
        IO.println s!"  PASS  convolution-oracle  clock-warp FIR ≡ array-shift conv  (max|Δ|={maxAbs}, filter-effect={filterEffect}, samples={n - maxDelay})"
        pure true
      else
        IO.println s!"  FAIL  convolution-oracle  max|Δ|={maxAbs} (≥ {eps}); filter-effect={filterEffect}"
        pure false

-- ── (h‴) The MODULATED-CLOCK stress test — a fractional, NONLINEAR warp, to see
-- whether the bubble is a side-effect of affineness (it should not be). The warp
-- φ(τ) = clk − ⌊depth·mod(τ)·2³²⌋ is sub-sample and nonlinear (mod is a sine);
-- it evaluates the carrier at clock values BETWEEN integer samples, which the
-- array-shift oracle cannot reach. So the oracle is an INDEPENDENT closed-form
-- reference (Lean `Float.sin` on the modulated phase), calibrated against the
-- bare oscillator first: tropical's `Sin` is a polynomial, so this is a
-- TOLERANCE check, not bit-exact — but the tolerance is the bare osc's own
-- poly/quantization floor, so the test isolates the WARP's contribution. A warp
-- that secretly needed affineness would diverge by O(1), far above that floor.

private def finishCarrier (arena : Arena) (idx : ProgramIdx) :
    Except String Tropical.Plan.FlatPlan := do
  let (arena'', root') ← (Tropical.Ir.Strata.run { upto := 5 } arena idx).mapError (·.message)
  let core ← Tropical.Ir.Core.check arena'' root'
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    mode := .fused }
  Tropical.Compile.compileSession input

private def buildAndFinish (built : Except String (Arena × ProgramIdx)) :
    Except String Tropical.Plan.FlatPlan := do
  let (a, i) ← built
  finishCarrier a i

/-- tropical's `Sin`, transcribed exactly from stdlib/Sin.md: reduce by π
    (n = round(x/π), r = x − n·π), parity sign, degree-11 Taylor Horner in r².
    The SAME polynomial the engine evaluates — so the oracle is a straight-line
    "standard representation," not a true-sine benchmark. -/
private def sinH (x : Float) : Float :=
  let nF := (x * 0.3183098861837907).round
  let r := x - nF * 3.141592653589793
  let oddF := nF - 2.0 * (nF / 2.0).floor          -- n & 1, as 0.0 / 1.0
  let sign := 1.0 - 2.0 * oddF
  let r2 := r * r
  let poly := (((((-2.505210838544172e-8) * r2 + 0.0000027557319223985893) * r2
      + (-0.0001984126984126984)) * r2 + 0.008333333333333333) * r2
      + (-0.16666666666666666)) * r2 + 1.0
  sign * (r * poly)

/-- `ClockPhasor.phase` at a Q32.32 clock value, transcribed exactly (integer
    math, offset = 0). inc = ⌊freqHz·2³²/SR⌋. Uses `fdiv`/`fmod` so it matches the
    engine's arithmetic shift (`>>`, floor) and two's-complement mask (`&`) for
    NEGATIVE clocks too — i.e. negative time / backward extrapolation. For clk ≥ 0
    these agree with plain /,%. -/
private def phasorPhase (clk : Int) (freqHz : Int) : Float :=
  let inc : Int := (freqHz * 4294967296) / 44100
  let thi := Int.fdiv clk 4294967296
  let tlo := Int.fmod clk 4294967296
  let acc := inc * thi + Int.fdiv (inc * tlo) 4294967296
  Float.ofInt (Int.fmod acc 4294967296) / 4294967296.0

/-- Float → Int truncation toward zero (matches the engine's `toInt`). -/
private def truncToInt (v : Float) : Int :=
  if v ≥ 0.0 then Int.ofNat v.toUInt64.toNat
  else -(Int.ofNat (-v).toUInt64.toNat)

/-- THE STANDARD-REP GATE: tropical's arrow-emitted, sine-modulated, SUB-SAMPLE
    clock warp vs a straight-line reimplementation using THE SAME Horner `Sin`
    and the SAME integer phasor. The polynomial cancels (it is identical on both
    sides — no true sine in the loop), so the residual is purely the warp/emit
    path and float op-ordering. The warp is genuinely nonlinear and fractional:
    φ(τ) = clk − toInt(depth · Sin(mod phase) · 2³²). -/
private def runModulatedClock (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let lo : Nat := 8
  let depth : Float := 3.0
  let twoPi : Float := 6.283185307179586
  let two32 : Float := 4294967296.0
  let sinkGain : Float := 0.05   -- defaultSinkGain (Plan.lean): scales OUTPUTS, not the mid-graph warp
  match buildAndFinish (Tropical.EmitArrow.buildTapCarrier "BareFc"
          (Tropical.EmitArrow.litPitchVoice 2000) bareTaps arena resolved),
        buildAndFinish (Tropical.EmitArrow.buildFmCarrier "FmOsc" 2000 200 3 arena resolved) with
  | .error e, _ => IO.println s!"  FAIL  modulated-clock  build bare: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  modulated-clock  build fm: {firstLine e}"; pure false
  | .ok barePlan, .ok fmPlan =>
    match ← renderPlanSamples barePlan n, ← renderPlanSamples fmPlan n with
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  modulated-clock  render: {firstLine e}"; pure false
    | .ok bare, .ok got =>
      let mut e0 : Float := 0.0           -- calibration: engine bare vs standard rep
      let mut efm : Float := 0.0          -- engine fm vs standard rep (the warp test)
      let mut warpEffect : Float := 0.0   -- |fm − bare|
      let mut maxBare : Float := 0.0
      let mut calBitDiff : Nat := 0       -- bit-differing samples (engine bare vs std)
      let mut fmBitDiff : Nat := 0        -- bit-differing samples (engine fm vs std)
      for t in [lo:n] do
        let clk : Int := Int.ofNat t * 4294967296
        -- calibration: engine's bare carrier vs the standard-rep carrier
        let refBare := sinkGain * sinH (twoPi * phasorPhase clk 2000)
        if (bare[t]! - refBare).abs > e0 then e0 := (bare[t]! - refBare).abs
        if bare[t]!.toBits != refBare.toBits then calBitDiff := calBitDiff + 1
        if bare[t]!.abs > maxBare then maxBare := bare[t]!.abs
        -- the warp: mid-graph (unit-scale) modulator = Sin at the modulator phase;
        -- offset = toInt(depth·mod·2³²); φ = clk − offset (sub-sample, nonlinear)
        let rawMod := sinH (twoPi * phasorPhase clk 200)
        let phi : Int := clk - truncToInt (depth * rawMod * two32)
        let refFm := sinkGain * sinH (twoPi * phasorPhase phi 2000)
        if (got[t]! - refFm).abs > efm then efm := (got[t]! - refFm).abs
        if got[t]!.toBits != refFm.toBits then fmBitDiff := fmBitDiff + 1
        if (got[t]! - bare[t]!).abs > warpEffect then warpEffect := (got[t]! - bare[t]!).abs
      let samples := n - lo
      IO.println s!"        standard rep = same Horner Sin + same integer phasor (no true sine):"
      IO.println s!"        calibrate  engine bare vs standard rep:  max|Δ|={e0}  ·  bit-differing {calBitDiff}/{samples}"
      IO.println s!"        result     engine fm   vs standard rep:  max|Δ|={efm}  ·  bit-differing {fmBitDiff}/{samples}  ·  warp effect |fm−bare| max={warpEffect}"
      if maxBare < 1e-3 then
        IO.println s!"  FAIL  modulated-clock  carrier silent (maxBare={maxBare})"; pure false
      else if e0 > 1e-6 then
        IO.println s!"  FAIL  modulated-clock  calibration off (e0={e0}) — Sin/phasor transcription wrong, test invalid"; pure false
      else if warpEffect < 0.2 * maxBare then
        IO.println s!"  FAIL  modulated-clock  modulation negligible (warp {warpEffect} vs amp {maxBare})"; pure false
      else if efm < 10.0 * e0 + 1e-9 then
        IO.println s!"  PASS  modulated-clock  fractional nonlinear warp ≡ standard rep (fm err {efm} ≈ floor {e0}; warp effect {warpEffect})"; pure true
      else
        IO.println s!"  FAIL  modulated-clock  fm err {efm} ≫ floor {e0} — warp diverges from the standard rep"; pure false

/-- PM-of-PM: the modulator is ITSELF a warped oscillator (mod2 warps mod's
    clock, mod warps the carrier's clock). Bit-exact against a THREE-level nested
    standard rep (same Horner Sin + integer phasor at each level) ⇒ the warp /
    substitution composes through nesting. Also asserts the second level is
    non-trivial: PM(PM) differs from single-level PM. -/
private def runPmPm (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let lo : Nat := 8
  let d1 : Float := 3.0
  let d2 : Float := 3.0
  let twoPi : Float := 6.283185307179586
  let two32 : Float := 4294967296.0
  let sinkGain : Float := 0.05
  match buildAndFinish (Tropical.EmitArrow.buildPmPmCarrier "PmPm" 2000 200 700 3 3 arena resolved),
        buildAndFinish (Tropical.EmitArrow.buildFmCarrier "Fm1" 2000 200 3 arena resolved) with
  | .error e, _ => IO.println s!"  FAIL  pm-of-pm  build pmpm: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  pm-of-pm  build fm1: {firstLine e}"; pure false
  | .ok pmpmPlan, .ok fmPlan =>
    match ← renderPlanSamples pmpmPlan n, ← renderPlanSamples fmPlan n with
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  pm-of-pm  render: {firstLine e}"; pure false
    | .ok got, .ok fm1 =>
      let mut e0 : Float := 0.0           -- engine pm(pm) vs nested standard rep
      let mut bitDiff : Nat := 0
      let mut maxOut : Float := 0.0
      let mut nestEffect : Float := 0.0   -- |pm(pm) − single-level pm| (does level 2 matter)
      for t in [lo:n] do
        let clk : Int := Int.ofNat t * 4294967296
        let mod2 := sinH (twoPi * phasorPhase clk 700)
        let modClk : Int := clk - truncToInt (d2 * mod2 * two32)
        let mod := sinH (twoPi * phasorPhase modClk 200)
        let carClk : Int := clk - truncToInt (d1 * mod * two32)
        let ref := sinkGain * sinH (twoPi * phasorPhase carClk 2000)
        if (got[t]! - ref).abs > e0 then e0 := (got[t]! - ref).abs
        if got[t]!.toBits != ref.toBits then bitDiff := bitDiff + 1
        if got[t]!.abs > maxOut then maxOut := got[t]!.abs
        if (got[t]! - fm1[t]!).abs > nestEffect then nestEffect := (got[t]! - fm1[t]!).abs
      let samples := n - lo
      IO.println s!"        nested standard rep (mod2→mod→carrier, same Horner Sin + integer phasor):"
      IO.println s!"        result   engine pm(pm) vs nested rep: max|Δ|={e0}  ·  bit-differing {bitDiff}/{samples}"
      IO.println s!"        nesting  |pm(pm) − single-level pm| max={nestEffect}  (level-2 must be non-trivial)"
      if maxOut < 1e-3 then
        IO.println s!"  FAIL  pm-of-pm  carrier silent (maxOut={maxOut})"; pure false
      else if nestEffect < 1e-3 then
        IO.println s!"  FAIL  pm-of-pm  level-2 negligible (nesting effect {nestEffect}) — not stressing the nest"; pure false
      else if bitDiff == 0 then
        IO.println s!"  PASS  pm-of-pm  nested warp ≡ nested standard rep bit-for-bit ({bitDiff}/{samples}; nesting effect {nestEffect})"; pure true
      else
        IO.println s!"  FAIL  pm-of-pm  {bitDiff}/{samples} bit-differing (max|Δ|={e0}) — nested substitution diverges"; pure false

/-- The NEGATIVE-TIME boundary — the moat. A 20-sample delay warp pulls the clock
    negative for t < 20, so the carrier is evaluated BEFORE sample 0. Closed-form
    random access gives the exact backward-extrapolated sine; a streaming delay
    line could only emit zeros (no past). Asserts: (1) bit-exact vs a random-access
    standard rep at ALL t including negative time, and (2) the output at negative
    time is non-zero — the engine does NOT zero-pad, which is the random-access
    capability a stream cannot have. -/
private def runNegativeClock (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let delta : Nat := 20
  let twoPi : Float := 6.283185307179586
  let sinkGain : Float := 0.05
  let delayTap : Tropical.EmitArrow.Tap :=
    { name := "d"
      warp := fun c => Tropical.EmitArrow.sub c
        (Tropical.EmitArrow.toIntE (Tropical.EmitArrow.lit (Int.ofNat delta * 4294967296)))
      weight := Tropical.EmitArrow.lit 1 }
  match buildAndFinish (Tropical.EmitArrow.buildTapCarrier "DelayFc"
          (Tropical.EmitArrow.litPitchVoice 2000) #[delayTap] arena resolved) with
  | .error e => IO.println s!"  FAIL  negative-clock  build: {firstLine e}"; pure false
  | .ok plan =>
    match ← renderPlanSamples plan n with
    | .error e => IO.println s!"  FAIL  negative-clock  render: {firstLine e}"; pure false
    | .ok got =>
      let mut bitDiff : Nat := 0
      let mut maxOut : Float := 0.0
      let mut negCount : Nat := 0
      let mut negMag : Float := 0.0      -- |output| at negative-time samples (streaming ⇒ 0)
      let mut negBitDiff : Nat := 0
      for t in [0:n] do
        let clk : Int := Int.ofNat t * 4294967296
        let phi : Int := clk - Int.ofNat delta * 4294967296   -- (t − 20)·2³²
        let ref := sinkGain * sinH (twoPi * phasorPhase phi 2000)
        if got[t]!.toBits != ref.toBits then bitDiff := bitDiff + 1
        if got[t]!.abs > maxOut then maxOut := got[t]!.abs
        if phi < 0 then
          negCount := negCount + 1
          if got[t]!.abs > negMag then negMag := got[t]!.abs
          if got[t]!.toBits != ref.toBits then negBitDiff := negBitDiff + 1
      IO.println s!"        random-access rep: osc(φ), φ=(t−{delta})·2³², INCLUDING negative time:"
      IO.println s!"        result   engine vs random-access rep: bit-differing {bitDiff}/{n}  (neg-time samples {negCount}, differing {negBitDiff})"
      IO.println s!"        moat     |output| at negative time max={negMag}  (a streaming delay line would emit 0 here)"
      if maxOut < 1e-3 then
        IO.println s!"  FAIL  negative-clock  silent (maxOut={maxOut})"; pure false
      else if negCount == 0 then
        IO.println s!"  FAIL  negative-clock  delay didn't pull the clock negative ({negCount} neg samples)"; pure false
      else if negMag < 1e-3 then
        IO.println s!"  FAIL  negative-clock  engine zero-pads at negative time (negMag={negMag}) — not random-access"; pure false
      else if bitDiff == 0 then
        IO.println s!"  PASS  negative-clock  random-access exact at negative time ({bitDiff}/{n}; {negCount} neg-time samples, |out|≤{negMag} where a stream emits 0)"; pure true
      else
        IO.println s!"  FAIL  negative-clock  {bitDiff}/{n} bit-differing — negative-time phasor diverges"; pure false

-- ── (h⁶) PRODUCTS / MIMO standard-rep differential (the DATA axis) ────────────
-- `MorphOsc` built from the cartesian combinators (ClockPhasor ⋙ (saw &&& Sin)
-- ⋙ crossfade) vs a straight-line reimplementation reusing the SAME integer
-- phasor and Horner `Sin`. No warp, no sub-sample clock — so this is BIT-EXACT
-- (like the convolution oracle, not the modulated-clock tolerance check). Three
-- morph settings prove the diagonal feeds two GENUINELY DIFFERENT consumers and
-- the crossfade blends them: morph=0 ≡ pure saw, morph=1 ≡ pure sine, morph=0.5
-- ≡ the blend — each bit-exact, and saw ≢ sine (non-degenerate MIMO).
private def runMorphOscDifferential (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 1024
  let lo : Nat := 4
  let freqHz : Int := 2000
  let twoPi : Float := 6.283185307179586
  let sinkGain : Float := 0.05
  -- the standard rep: the SAME crossfade arithmetic the engine emits, on the
  -- SAME integer phasor + Horner Sin (`(1−m)·(2·phase−1) + m·Sin(2π·phase)`).
  let refOut := fun (morphF : Float) (clk : Int) =>
    let phase := phasorPhase clk freqHz
    sinkGain * ((1.0 - morphF) * (2.0 * phase - 1.0) + morphF * sinH (twoPi * phase))
  let render := fun (nm : String) (m : Tropical.EmitArrow.Sig) =>
    match buildAndFinish (Tropical.EmitArrow.buildMorphOscLit nm freqHz m arena resolved) with
    | .error e => (pure (.error e) : IO (Except String (Array Float)))
    | .ok plan => renderPlanSamples plan n
  match ← render "MorphSaw" (Tropical.EmitArrow.lit 0),
        ← render "MorphSin" (Tropical.EmitArrow.lit 1),
        ← render "MorphBlend" (Tropical.EmitArrow.lit 5 1) with
  | .error e, _, _ | _, .error e, _ | _, _, .error e =>
    IO.println s!"  FAIL  morphosc-mimo  build/render: {firstLine e}"; pure false
  | .ok saw, .ok sinv, .ok blend =>
    let mut sawDiff : Nat := 0
    let mut sinDiff : Nat := 0
    let mut blendDiff : Nat := 0
    let mut maxBlend : Float := 0.0
    let mut sawVsSin : Float := 0.0        -- the diagonal feeds two distinct shapes
    for t in [lo:n] do
      let clk : Int := Int.ofNat t * 4294967296
      if saw[t]!.toBits   != (refOut 0.0 clk).toBits then sawDiff   := sawDiff   + 1
      if sinv[t]!.toBits  != (refOut 1.0 clk).toBits then sinDiff   := sinDiff   + 1
      if blend[t]!.toBits != (refOut 0.5 clk).toBits then blendDiff := blendDiff + 1
      if blend[t]!.abs > maxBlend then maxBlend := blend[t]!.abs
      if (saw[t]! - sinv[t]!).abs > sawVsSin then sawVsSin := (saw[t]! - sinv[t]!).abs
    let samples := n - lo
    IO.println s!"        standard rep = same integer phasor + Horner Sin, same crossfade arithmetic:"
    IO.println s!"        result   engine MorphOsc vs std rep:  bit-differing  saw {sawDiff}/{samples} · sine {sinDiff}/{samples} · blend {blendDiff}/{samples}"
    IO.println s!"        mimo     diagonal feeds distinct consumers: max|saw−sine|={sawVsSin}"
    if maxBlend < 1e-3 then
      IO.println s!"  FAIL  morphosc-mimo  carrier silent (maxBlend={maxBlend})"; pure false
    else if sawVsSin < 1e-2 then
      IO.println s!"  FAIL  morphosc-mimo  saw ≈ sine (max|Δ|={sawVsSin}) — diagonal degenerate"; pure false
    else if sawDiff == 0 && sinDiff == 0 && blendDiff == 0 then
      IO.println s!"  PASS  morphosc-mimo  ClockPhasor ⋙ (saw &&& Sin) ⋙ crossfade ≡ standard rep, bit-exact (saw/sine/blend 0/{samples}; max|saw−sine|={sawVsSin})"; pure true
    else
      IO.println s!"  FAIL  morphosc-mimo  bit-differing (saw {sawDiff} · sine {sinDiff} · blend {blendDiff}) — MIMO build diverges from the standard rep"; pure false

-- ── (h⁷) THE SLIDE (WARP-PUSH): downstream insert → upstream warp, by the compiler ─
-- The reified arrow term + `normalize` push warps up to the generators. Three
-- gates: (1) byte-identity vs stdlib FlangeSin lives in the corpus section
-- (slide(osc ⋙ flange) ≡ hand-written upstream FlangeSin); (2) slide-past-arr —
-- a pointwise shaper between osc and flange, so the warp must COMMUTE PAST it
-- (R1); (3) cascade — osc ⋙ flange ⋙ flange yields the 9-tap convolved
-- multiplicity automatically.

/-- Test 2: the warp must slide PAST a pointwise shaper to reach the generator.
    `slide(osc ⋙ shaper ⋙ flange)` must byte-equal the hand-written upstream form
    (shaper applied to osc at each warped clock). Byte-equal ⇒ R1 fired. -/
private def runSlidePastArr (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildSlideShaperDownstream arena resolved,
        Tropical.EmitArrow.buildSlideShaperUpstream arena resolved with
  | .ok (aD, iD), .ok (aU, iU) =>
    match emitResolvedWire aD iD, emitResolvedWire aU iU with
    | .ok bytesD, .ok bytesU =>
      if bytesD == bytesU then
        IO.println s!"  PASS  slide-past-arr  warp commuted past the shaper: slide(osc ⋙ shaper ⋙ flange) ≡ upstream ({bytesD.length}B)"; pure true
      else
        IO.println s!"  FAIL  slide-past-arr  slide(downstream) ≠ upstream (down {bytesD.length}B, up {bytesU.length}B) — R1 (commute past arr) wrong"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  slide-past-arr  emit: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  slide-past-arr  build: {firstLine e}"; pure false

/-- Test 4: the product slide law. `slide(warp φ (x ⊗ y))` must byte-equal the
    hand-written upstream form (φ on each factor). Byte-equal ⇒ the warp
    distributed over `×` — both factors of the VCA reclock. This is what makes
    `prod` (signal×signal, the amplitude/VCA multiply that `scale` can't express)
    lawful under the slide, so an envelope factored as its own term rides every
    downstream delay tap. -/
private def runSlideProd (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildSlideProdDownstream arena resolved,
        Tropical.EmitArrow.buildSlideProdUpstream arena resolved with
  | .ok (aD, iD), .ok (aU, iU) =>
    match emitResolvedWire aD iD, emitResolvedWire aU iU with
    | .ok bytesD, .ok bytesU =>
      if bytesD == bytesU then
        IO.println s!"  PASS  slide-past-prod  warp distributed over ×: slide(warp(x ⊗ y)) ≡ (warp x) ⊗ (warp y) ({bytesD.length}B)"; pure true
      else
        IO.println s!"  FAIL  slide-past-prod  slide(downstream) ≠ upstream (down {bytesD.length}B, up {bytesU.length}B) — warp did NOT distribute over the product"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  slide-past-prod  emit: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  slide-past-prod  build: {firstLine e}"; pure false

/-- THE BOOTSTRAP gate. A `FixedSinOsc` built as a TERM over `{clk, +, ×, round}`
    (`fixedSinOscTerm` = `Sin(2π·phasor)`, no `gen`, no `.trop` instance) must
    render bit-for-bit identical to the `.trop` `FixedSinOsc` at the same pitch and
    clock. Bit-exact ⇒ the generator IS the term — the arrow layer no longer needs
    `.trop` for its atoms; the phasor and the sine are `{clk, +, ×}` all the way
    down. -/
private def runBootstrapSin (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let refPlan := buildAndFinish (Tropical.EmitArrow.buildClockCarrier "boot_ref" Tropical.EmitArrow.clockLit arena resolved)
  let termPlan := buildAndFinish (.ok (Tropical.EmitArrow.buildBootstrapSinOsc "boot_term" arena))
  match refPlan, termPlan with
  | .ok rp, .ok tp =>
    match ← renderPlanSamples rp 2048, ← renderPlanSamples tp 2048 with
    | .ok refS, .ok termS =>
      let n := min refS.size termS.size
      let mut bitDiff := 0
      let mut maxAbs : Float := 0.0
      let mut energy : Float := 0.0
      for i in [0:n] do
        energy := energy + refS[i]! * refS[i]!
        if refS[i]! != termS[i]! then bitDiff := bitDiff + 1
        let d := (refS[i]! - termS[i]!).abs
        if d > maxAbs then maxAbs := d
      IO.println s!"        term = Sin(2π·phasor) over the clock leaf, no gen; ref = .trop FixedSinOsc @220:"
      IO.println s!"        result   term vs .trop:  bit-differing {bitDiff}/{n}  ·  max|Δ|={maxAbs}  ·  energy={energy}"
      if bitDiff == 0 && energy > 1e-6 then
        IO.println s!"  PASS  bootstrap-sin  phasor+sine as terms ≡ .trop FixedSinOsc, bit-exact ({n} samples, energy={energy})"; pure true
      else
        IO.println s!"  FAIL  bootstrap-sin  bit-differing {bitDiff}/{n} (max|Δ|={maxAbs}) — the term diverges from the .trop generator"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  bootstrap-sin  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  bootstrap-sin  build: {firstLine e}"; pure false

/-- THE BOOTSTRAP-EXP gate. `expSig` (the modal envelope primitive, transcribed
    from stdlib/Exp) evaluated by the engine over a ramp `x∈[−10,10]` must match
    libm `exp` to its minimax tolerance. An independent oracle (true exp, not a
    second copy of the same polynomial), so a transcribed-coefficient typo shows
    up as error ≫ 1e-5. This is the envelope's `bootstrap-sin`. -/
private def runBootstrapExp (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match buildAndFinish (.ok (Tropical.EmitArrow.buildExpProbe "exp_probe" arena)) with
  | .ok p =>
    match ← renderPlanSamples p 2048 with
    | .ok s =>
      let n := min s.size 2048
      let sinkGain : Float := 0.05   -- defaultSinkGain: the carrier's output sink
      let mut maxRel : Float := 0.0
      let mut worstX : Float := 0.0
      for i in [0:n] do
        let x := i.toFloat * 0.009765625 - 10.0
        let ref := sinkGain * Float.exp x
        let rel := (s[i]! - ref).abs / ref
        if rel > maxRel then
          maxRel := rel
          worstX := x
      IO.println s!"        expSig(x) vs libm exp, x∈[−10,10] across 2048 samples:"
      IO.println s!"        result   max relative error = {maxRel}  (at x={worstX})"
      if maxRel < 1e-5 then
        IO.println s!"  PASS  bootstrap-exp  emitted polynomial exp ≡ true exp to {maxRel} (minimax) — transcription correct"; pure true
      else
        IO.println s!"  FAIL  bootstrap-exp  max rel err {maxRel} (want <1e-5) at x={worstX}"; pure false
    | .error e => IO.println s!"  FAIL  bootstrap-exp  render: {firstLine e}"; pure false
  | .error e => IO.println s!"  FAIL  bootstrap-exp  build: {firstLine e}"; pure false

/-- THE MODAL ISLAND gate. A decaying-resonator bank (`Σ amp·e^{−σd}·cos(ωd)`,
    gated causal at a strike time) built through the ARROW path (`arrUn`/`clk`,
    then `emitTerm`) must render bit-for-bit identical to the same bank built
    straight-line — the standard-rep differential for the pole/modal island's
    emit path. We also assert the two properties that make it a MODAL signal and
    not noise: causality (exactly silent before the strike — a streaming reverb
    could not gate a future-anchored tail) and decay (the tail loses energy).
    Bit-exact ⇒ the arrow layer realises the bank without corruption; silent+
    decaying ⇒ it is a real closed-form resonator bank, random-access by clk. -/
private def runModalBank (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    { freq := Tropical.EmitArrow.lit 220, sigma := Tropical.EmitArrow.lit 30 1,
      amp := Tropical.EmitArrow.lit 6 1 },
    { freq := Tropical.EmitArrow.lit 330, sigma := Tropical.EmitArrow.lit 40 1,
      amp := Tropical.EmitArrow.lit 4 1 },
    { freq := Tropical.EmitArrow.lit 440, sigma := Tropical.EmitArrow.lit 55 1,
      amp := Tropical.EmitArrow.lit 3 1 }]
  let anchor := Tropical.EmitArrow.lit 200
  let arrowPlan := buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankArrow "modal_arrow" modes anchor arena))
  let directPlan := buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankDirect "modal_direct" modes anchor arena))
  match arrowPlan, directPlan with
  | .ok ap, .ok dp =>
    match ← renderPlanSamples ap 2048, ← renderPlanSamples dp 2048 with
    | .ok aS, .ok dS =>
      let n := min aS.size dS.size
      let mut bitDiff := 0
      for i in [0:n] do
        if aS[i]! != dS[i]! then bitDiff := bitDiff + 1
      let mut preMax : Float := 0.0
      for i in [0:200] do
        let a := aS[i]!.abs
        if a > preMax then preMax := a
      let mut eEarly : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [200:600] do eEarly := eEarly + aS[i]! * aS[i]!
      for i in [1648:2048] do eLate := eLate + aS[i]! * aS[i]!
      IO.println s!"        bank = Σ amp·e^(−σd)·cos(2πf·d) @ 220/330/440, struck @ sample 200 (d=clk/2³²/SR−anchor):"
      IO.println s!"        result   arrow vs straight-line:  bit-differing {bitDiff}/{n}  ·  pre-strike |max|={preMax}  ·  E[early]={eEarly}  E[late]={eLate}"
      if bitDiff == 0 && preMax == 0.0 && eEarly > 1e-6 && eLate < eEarly then
        IO.println s!"  PASS  modal-bank  gated decaying-sinusoid bank: arrow ≡ straight-line bit-exact, causal (silent pre-strike), decaying ({n} samples)"; pure true
      else
        IO.println s!"  FAIL  modal-bank  bitDiff={bitDiff} preMax={preMax} (want 0) eEarly={eEarly} (>1e-6) eLate={eLate} (<eEarly)"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  modal-bank  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  modal-bank  build: {firstLine e}"; pure false

/-- THE MODAL DEGREE gate. A degree-1 mode `amp·d·e^{−σd}` (a repeated pole — the
    resonance "swell") rendered by the engine must match `sinkGain·d·e^{−σd}` to
    minimax tolerance (an absolute oracle, validating the new `d^deg` factor), and
    must RISE to a peak at d≈1/σ before decaying — the τ·e signature a simple pole
    (monotone decay) cannot produce. -/
private def runModalDegree (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    { freq := Tropical.EmitArrow.lit 0, sigma := Tropical.EmitArrow.lit 25,
      amp := Tropical.EmitArrow.lit 1, deg := 1 }]
  let anchor := Tropical.EmitArrow.lit 200
  match buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankArrow "modal_deg" modes anchor arena)) with
  | .ok p =>
    match ← renderPlanSamples p 8192 with
    | .ok s =>
      let sinkGain : Float := 0.05
      let n := min s.size 8192
      let mut preMax : Float := 0.0
      for i in [0:201] do
        let a := s[i]!.abs
        if a > preMax then preMax := a
      let mut maxRel : Float := 0.0
      let mut peakVal : Float := 0.0
      let mut peakI : Nat := 0
      for i in [201:n] do
        let d := (i.toFloat - 200.0) / 44100.0
        let ref := sinkGain * d * Float.exp (-25.0 * d)
        if ref.abs > 1e-5 then
          let rel := (s[i]! - ref).abs / ref.abs
          if rel > maxRel then maxRel := rel
        let a := s[i]!.abs
        if a > peakVal then
          peakVal := a
          peakI := i
      let peakD := (peakI.toFloat - 200.0) / 44100.0
      IO.println s!"        deg-1 τ·e mode (σ=25, f=0) vs sinkGain·d·e^(−25d):"
      IO.println s!"        result   preMax={preMax} · max rel err={maxRel} · peak @ sample {peakI} (d={peakD}s, expect 1/σ=0.04)"
      if preMax == 0.0 && maxRel < 1e-4 && peakI > 1500 && peakI < 2400 then
        IO.println s!"  PASS  modal-degree  τ·e swell ≡ d·e^(−σd) to {maxRel}; rises to peak at d≈1/σ then decays"; pure true
      else
        IO.println s!"  FAIL  modal-degree  preMax={preMax} maxRel={maxRel} peakI={peakI}"; pure false
    | .error e => IO.println s!"  FAIL  modal-degree  render: {firstLine e}"; pure false
  | .error e => IO.println s!"  FAIL  modal-degree  build: {firstLine e}"; pure false

section ResidueGates
open Tropical.EmitArrow

/-- THE RESIDUE CALCULUS gate (exact, build-time). `voice ⋙ reverb` composed by
    `residueCompose` must reproduce the convolution's Taylor jet at t=0: moment
    `Σ Aᵢμᵢᵏ` equals `y⁽ᵏ⁾(0)` for k=0..6, and the 0th moment `Σ A = 0` (a wrong
    sign, denominator, or a missing ringing term breaks one). `Σ A = 0` also means
    the composed tail starts continuously — the reverb has no onset click for free.
    Pure complex ±×÷; the emit path is checked separately by `modal-reverb`. -/
private def runResidueMoments (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voice : Array (Cplx × Cplx) := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)]
  let reverb : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)]
  let modes := residueCompose voice reverb
  let err := residueMomentError voice reverb 6
  let sumA := modes.foldl (fun s m => s.add m.amp) (⟨0.0, 0.0⟩ : Cplx)
  let sumAbsA := modes.foldl (fun s m => s + m.amp.abs) 0.0
  let onset := sumA.abs / (sumAbsA + 1e-300)
  IO.println s!"        voice(2 poles) ⋙ reverb(4 poles) → {modes.size} residue modes; jet-match k=0..6:"
  IO.println s!"        result   max relative moment error = {err}  ·  onset ΣA/Σ|A| = {onset}"
  if err < 1e-9 && onset < 1e-9 then
    IO.println s!"  PASS  residue-moments  composed modes reproduce the convolution jet to k=6 (err={err}); ΣA=0 ⇒ click-free onset"; pure true
  else
    IO.println s!"  FAIL  residue-moments  err={err} (want <1e-9) onset={onset} (want <1e-9)"; pure false

/-- THE RESIDUE REVERB gate (emit). `buildModalReverb` runs the residue calculus
    and emits the composed bank; it must render a real, causal, DECAYING signal
    that starts CONTINUOUSLY at the strike — the `Σ A = 0` property means the first
    post-strike sample is ≈0 and grows (no onset click), unlike an authored bank
    whose partials all start at full amplitude. -/
private def runModalReverb (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voice : Array (Cplx × Cplx) := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)]
  let reverb : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)]
  let anchor := lit 200
  -- render ~370 ms: the composed tail ramps up (click-free onset) over the first
  -- tens of ms, then decays over its RT — so compare energy AFTER the onset peak.
  match buildAndFinish (.ok (buildModalReverb "modal_reverb" voice reverb anchor arena)) with
  | .ok p =>
    match ← renderPlanSamples p 16384 with
    | .ok s =>
      let n := min s.size 16384
      let mut preMax : Float := 0.0
      for i in [0:201] do
        let a := s[i]!.abs
        if a > preMax then preMax := a
      let firstPost := s[201]!.abs
      let mut peak : Float := 0.0
      for i in [201:n] do
        let a := s[i]!.abs
        if a > peak then peak := a
      let mut eMid : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [2048:6144] do eMid := eMid + s[i]! * s[i]!
      for i in [12288:16384] do eLate := eLate + s[i]! * s[i]!
      IO.println s!"        buildModalReverb rendered (voice ⋙ reverb, struck @ sample 200):"
      IO.println s!"        result   pre-strike |max|={preMax} · first-post |s|={firstPost} · peak={peak} · E[mid]={eMid} E[late]={eLate}"
      if preMax == 0.0 && peak > 1e-4 && firstPost < 0.02 * peak && eLate < eMid then
        IO.println s!"  PASS  modal-reverb  residue-composed bank renders: causal, click-free onset (|first|≪peak), decaying tail ({n} samples)"; pure true
      else
        IO.println s!"  FAIL  modal-reverb  preMax={preMax} peak={peak} firstPost={firstPost} eMid={eMid} eLate={eLate}"; pure false
    | .error e => IO.println s!"  FAIL  modal-reverb  render: {firstLine e}"; pure false
  | .error e => IO.println s!"  FAIL  modal-reverb  build: {firstLine e}"; pure false

/-- THE DEGENERATE RESIDUE gate. A voice pole placed EXACTLY on a reverb pole
    (sympathetic resonance) must compose to a `τ·e^{μd}` DOUBLE POLE, not blow up.
    residueCompose must emit exactly one deg-1 mode, and — crucially — the
    degree-aware moments must STILL reproduce the convolution jet (the double pole
    contributes `A·k·μ^{k−1}`), so the exact-coincidence limit is handled, not
    dodged. -/
private def runResidueDegenerate (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voice : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 260.0⟩, ⟨1.0, 0.0⟩)]        -- λ sits exactly on reverb pole #2
  let reverb : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-3.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),        -- ν = λ (coincident)
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩)]
  let modes := residueCompose voice reverb
  let nDeg1 := modes.foldl (fun c m => if m.deg == 1 then c + 1 else c) 0
  let err := residueMomentError voice reverb 6
  IO.println s!"        voice pole = reverb pole #2 (sympathetic): {modes.size} modes, {nDeg1} of degree 1:"
  IO.println s!"        result   deg-1 modes = {nDeg1}  ·  degree-aware moment error k=0..6 = {err}"
  if nDeg1 == 1 && err < 1e-9 then
    IO.println s!"  PASS  residue-degenerate  coincident pole → one τ·e double pole; jet still exact (err={err}) — no blow-up"; pure true
  else
    IO.println s!"  FAIL  residue-degenerate  nDeg1={nDeg1} (want 1) err={err} (want <1e-9)"; pure false

end ResidueGates

/-- Test 3: `osc ⋙ flange ⋙ flange` — the slide pushes the outer warps through
    the inner flanger's sum and fuses them, producing the oscillator read at the
    nine convolved offsets automatically (the proper multiplicity, derived). We
    assert the generator count (9 vs 3) and that the cascade is a real, non-silent
    filter distinct from a single flanger. -/
private def runSlideCascade (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildSlideDoubleFlanger arena resolved,
        Tropical.EmitArrow.buildSlideSingleFlanger arena resolved with
  | .ok (aD, iD), .ok (aS, iS) =>
    let ninstD := ((aD.program? iD).map (·.decls.size)).getD 0
    let ninstS := ((aS.program? iS).map (·.decls.size)).getD 0
    match buildAndFinish (.ok (aD, iD)), buildAndFinish (.ok (aS, iS)) with
    | .ok planD, .ok planS =>
      match ← renderPlanSamples planD 512, ← renderPlanSamples planS 512 with
      | .ok dbl, .ok sgl =>
        let mut energy : Float := 0.0
        let mut diff : Float := 0.0
        for t in [8:512] do
          energy := energy + dbl[t]! * dbl[t]!
          if (dbl[t]! - sgl[t]!).abs > diff then diff := (dbl[t]! - sgl[t]!).abs
        IO.println s!"        cascade osc ⋙ flange ⋙ flange: {ninstD} generator instances (single flange: {ninstS}); the slide convolved the kernels — 9 = 3⊛3 taps, no coincident-offset merge"
        if ninstD == 9 && ninstS == 3 && energy > 1e-6 && diff > 1e-4 then
          IO.println s!"  PASS  slide-cascade  9-tap multiplicity derived by the slide (energy={energy}, |double−single|max={diff})"; pure true
        else
          IO.println s!"  FAIL  slide-cascade  ninstD={ninstD} (want 9) ninstS={ninstS} (want 3) energy={energy} diff={diff}"; pure false
      | .error e, _ | _, .error e => IO.println s!"  FAIL  slide-cascade  render: {firstLine e}"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  slide-cascade  finish: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  slide-cascade  build: {firstLine e}"; pure false

-- ── (h⁸) THE PATCHER LOWERING: a downstream-only patch graph → arrow term ──────
-- The MVP front end. A wire is the effect applied to the upstream term (⋙), a
-- fan-out is the shared upstream term (Δ), a mixer is the sum. L1 (byte-identity
-- vs FlangeSin from a GRAPH) is in the corpus section; here: L2 (a chain graph ≡
-- the hand-built term) and L3 (a fan-out graph renders, with the diagonal).

/-- L2: lowering the chain graph `osc → flange → flange` must byte-equal the
    hand-written nested term (`buildSlideDoubleFlanger`). Graph-lowering ≡
    hand-term ⇒ the front end composes effects exactly as `⋙`. -/
private def runLoweringChain (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildDoubleFlangeFromGraph arena resolved,
        Tropical.EmitArrow.buildSlideDoubleFlanger arena resolved with
  | .ok (aG, iG), .ok (aH, iH) =>
    match emitResolvedWire aG iG, emitResolvedWire aH iH with
    | .ok bytesG, .ok bytesH =>
      if bytesG == bytesH then
        IO.println s!"  PASS  lowering-chain  lower(osc→flange→flange) ≡ hand-built nested term ({bytesG.length}B)"; pure true
      else
        IO.println s!"  FAIL  lowering-chain  graph {bytesG.length}B ≠ hand-term {bytesH.length}B"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  lowering-chain  emit: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  lowering-chain  build: {firstLine e}"; pure false

/-- L3: a fan-out patch — `osc` fanned into two flangers, mixed (the diagonal +
    the product collapse through the lowering). Asserts six generator instances
    (3 per flanger; the source re-derived per tap) and a real, non-silent mix. -/
private def runLoweringFanOut (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildFanOutFromGraph arena resolved with
  | .ok (aF, iF) =>
    let ninst := ((aF.program? iF).map (·.decls.size)).getD 0
    match buildAndFinish (.ok (aF, iF)) with
    | .ok plan =>
      match ← renderPlanSamples plan 512 with
      | .ok got =>
        let mut energy : Float := 0.0
        for t in [8:512] do energy := energy + got[t]! * got[t]!
        IO.println s!"        fan-out osc → (flange δ₁ &&& flange δ₂) → mix: {ninst} generator instances (the diagonal re-sources the osc per tap)"
        if ninst == 6 && energy > 1e-6 then
          IO.println s!"  PASS  lowering-fanout  diagonal + mix through the lowering ({ninst} instances, energy={energy})"; pure true
        else
          IO.println s!"  FAIL  lowering-fanout  ninst={ninst} (want 6) energy={energy}"; pure false
      | .error e => IO.println s!"  FAIL  lowering-fanout  render: {firstLine e}"; pure false
    | .error e => IO.println s!"  FAIL  lowering-fanout  finish: {firstLine e}"; pure false
  | .error e => IO.println s!"  FAIL  lowering-fanout  build: {firstLine e}"; pure false

/-- Modulated-effect node: a `.fm` node routes one node's signal into a carrier's
    clock (FM/PM). Gated byte-identical against the hand-built carriers the
    bit-exact modulated-clock / PM-of-PM differentials already render: M1 a single
    FM node ≡ `buildFmCarrier`; M2 nested `.fm` nodes ≡ `buildPmPmCarrier`. So the
    `osc → flange → osc.fm` edge lowers to the proven modulated warp. -/
private def runModulatedNode (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let cmp := fun (label : String)
      (g h : Except String (Arena × ProgramIdx)) =>
    match g, h with
    | .ok (aG, iG), .ok (aH, iH) =>
      match emitResolvedWire aG iG, emitResolvedWire aH iH with
      | .ok bg, .ok bh =>
        if bg == bh then (true, s!"  PASS  modulated-node/{label}  graph fm node ≡ hand-built carrier ({bg.length}B)")
        else (false, s!"  FAIL  modulated-node/{label}  graph {bg.length}B ≠ carrier {bh.length}B")
      | .error e, _ | _, .error e => (false, s!"  FAIL  modulated-node/{label}  emit: {firstLine e}")
    | .error e, _ | _, .error e => (false, s!"  FAIL  modulated-node/{label}  build: {firstLine e}")
  let (ok1, msg1) := cmp "fm"
    (Tropical.EmitArrow.buildFmFromGraph 2000 200 3 arena resolved)
    (Tropical.EmitArrow.buildFmCarrier "FmRef" 2000 200 3 arena resolved)
  IO.println msg1
  let (ok2, msg2) := cmp "pm-of-pm"
    (Tropical.EmitArrow.buildPmPmFromGraph 2000 200 50 3 2 arena resolved)
    (Tropical.EmitArrow.buildPmPmCarrier "PmRef" 2000 200 50 3 2 arena resolved)
  IO.println msg2
  pure (ok1 && ok2)

-- ── (h⁹) C4: session → resolved root DIRECTLY ≡ the elaborate round-trip ───────
-- Every patch compiles to a session; this gate compiles each BOTH ways — the
-- production `sessionToParsed → elaborate` path and the direct
-- `sessionToResolvedRoot` path — and asserts the plans are byte-identical, so
-- the round-trip deletion is provably faithful before it becomes the default.
private def runSessionViaArrowEquiv : IO Bool := do
  let entries ← (System.FilePath.mk "patches").readDir
  let names := (entries.filterMap fun e =>
    if e.fileName.endsWith ".json" then some e.fileName else none).qsort (· < ·)
  let mut ok := true
  let mut matched := 0
  let mut skipped := 0
  for fn in names do
    let path := s!"patches/{fn}"
    match ← compilePatch path .fused with
    | .error _ => skipped := skipped + 1   -- not session-compilable on the baseline either
    | .ok elabPlan =>
      match ← compilePatchArrow path .fused with
      | .error e =>
        IO.println s!"  FAIL  session-via-arrow/{fn}  direct compile: {firstLine e}"; ok := false
      | .ok arrowPlan =>
        if elabPlan == arrowPlan then matched := matched + 1
        else
          IO.println s!"  FAIL  session-via-arrow/{fn}  plan differs (elab {elabPlan.length}B, direct {arrowPlan.length}B)"
          ok := false
  if ok then
    IO.println s!"  PASS  session-via-arrow  direct root ≡ elaborated root, plan-identical ({matched} patches{if skipped > 0 then s!"; {skipped} non-session skipped" else ""})"
  pure ok

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

  -- ── (c′) C4: session → resolved root directly ≡ the elaborate round-trip ───
  IO.println "session via direct root (sessionToResolvedRoot ≡ sessionToParsed→elaborate):"
  total := total + 1
  if !(← runSessionViaArrowEquiv) then failed := failed + 1

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

  -- ── (h) EmitArrow arrow laws (slice 3): warp algebra ≡ in rendered audio ────
  IO.println "arrow laws (warp algebra ≡ byte-identical audio):"
  match ← arrowElabStdlib with
  | .error e =>
    IO.println s!"  FAIL  arrow-laws  elaborate stdlib: {firstLine e}"
    total := total + 13; failed := failed + 13
  | .ok (arena, resolved) =>
    -- ── (h′) EmitArrow corpus gate: EmitArrow's emit ≡ strata's emit byte-wise ─
    -- The cutover instrument (phase C1). FlangeSin/ReversibleComb generalize the
    -- slices-1/2 byte-gate (formerly an external `diff` of two diffcli verbs);
    -- FixedSinOsc builds the foundational voice DIRECTLY (phasor + Sin poly, no
    -- sourced instance) and is byte-identical to `emit-stdlib FixedSinOsc`.
    IO.println "emitarrow corpus gate (EmitArrow emit ≡ strata emit, byte-identical):"
    total := total + 1
    if !(← runEmitCorpusGate "FlangeSin" "FlangeSin" arena resolved
          Tropical.EmitArrow.buildFlanger) then
      failed := failed + 1
    total := total + 1
    if !(← runEmitCorpusGate "ReversibleComb" "ReversibleComb" arena resolved
          Tropical.EmitArrow.buildReversibleComb) then
      failed := failed + 1
    total := total + 1
    if !(← runEmitCorpusGate "FixedSinOsc" "FixedSinOsc" arena resolved
          Tropical.EmitArrow.buildFixedSinOsc) then
      failed := failed + 1
    -- products/MIMO: MorphOsc — a real multi-port body (ClockPhasor ⋙ (saw &&&
    -- Sin) ⋙ crossfade) built from the cartesian combinators, byte-identical.
    total := total + 1
    if !(← runEmitCorpusGate "MorphOsc" "MorphOsc" arena resolved
          Tropical.EmitArrow.buildMorphOsc) then
      failed := failed + 1
    -- THE SLIDE (WARP-PUSH), Test 1: build FlangeSin from the DOWNSTREAM-insert
    -- form (osc ⋙ flange, warps unreduced), run the slide, emit — byte-identical
    -- to stdlib FlangeSin. The compiler turns "flanger dropped downstream" into
    -- "oscillator read at warped clocks." First compiler-driven downstream→upstream.
    total := total + 1
    if !(← runEmitCorpusGate "FlangeSinSlide" "FlangeSin" arena resolved
          Tropical.EmitArrow.buildFlangerViaSlide) then
      failed := failed + 1
    -- THE PATCHER LOWERING, L1: lower the GRAPH `osc → flange` (instances + a
    -- downstream wire), slide, emit — byte-identical to stdlib FlangeSin. The
    -- user's patch graph, lowered end to end, reaches the exact hand-written
    -- program. This is the MVP front end hitting the frozen artifact.
    total := total + 1
    if !(← runEmitCorpusGate "FlangeFromGraph" "FlangeSin" arena resolved
          Tropical.EmitArrow.buildFlangeFromGraph) then
      failed := failed + 1
    IO.println "arrow laws (warp algebra ≡ byte-identical audio):"
    -- Law 1 — inverse/cancellation:  warp(back δ) ⋙ warp(fwd δ) = id
    total := total + 1
    if !(← runArrowLaw "inverse"
          Tropical.EmitArrow.invLawLhsClock Tropical.EmitArrow.invLawRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 2 — additive delay/functoriality:
    --   warp(back δ₁) ⋙ warp(back δ₂) = warp(back (δ₁+δ₂))
    total := total + 1
    if !(← runArrowLaw "additive"
          Tropical.EmitArrow.addLawLhsClock Tropical.EmitArrow.addLawRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 3 — the cartesian diagonal / fan-out: one source ⋙ (flanger δ₁ &&&
    --   flanger δ₂) ≡ two independent (source+flanger) pairs (+ the COST story).
    total := total + 1
    if !(← runDiagonalLaw arena resolved) then
      failed := failed + 1
    -- ── slice 5 — REVERSE (the moat) as warp(neg) ───────────────────────────
    -- Law 4 — involution:  warp(neg) ⋙ warp(neg) = id  (byte-identical: −(−x)=x)
    total := total + 1
    if !(← runArrowLaw "reverse-involution"
          Tropical.EmitArrow.revInvolutionLhsClock Tropical.EmitArrow.revInvolutionRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 5 — reverse-swaps-delay:
    --   warp(neg) ⋙ warp(back δ) = warp(fwd δ) ⋙ warp(neg)  (byte-identical: −clk+δ)
    total := total + 1
    if !(← runArrowLaw "reverse-swaps-delay"
          Tropical.EmitArrow.revSwapLhsClock Tropical.EmitArrow.revSwapRhsClock
          arena resolved) then
      failed := failed + 1
    -- Law 6 — reverse commutes with the symmetric flanger:
    --   warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)  (denotational ≡; ±δ tap-sum
    --   reassociates in float — the finding; byte-exact only with fixed-point values)
    total := total + 1
    if !(← runReverseFlangerCommute arena resolved) then
      failed := failed + 1
    -- ── slice 6 — a FIXED-POINT VALUE carrier: slice-5's law 6 byte-exact ────
    -- THE GATE — Law 7: reverse commutes with the FIXED-POINT flanger,
    --   warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg) over the integer source +
    --   integer shift-add mix. Slice 5's float version was 1271/4096 differing;
    --   integer add is associative so this is now BYTE-IDENTICAL (0/N).
    total := total + 1
    if !(← runReverseFlangerCommuteFixedpoint arena) then
      failed := failed + 1
    -- Laws 8-10: the single-source warp laws over the FIXED-POINT source —
    --   involution / reverse-swaps-delay / additive — all byte-exact (the clock
    --   side is exact int64), mirroring the float `runArrowLaw` cases.
    total := total + 1
    if !(← runFixedSourceLaw "reverse-involution-fixedpoint"
          Tropical.EmitArrow.revInvolutionLhsClock Tropical.EmitArrow.revInvolutionRhsClock
          arena) then
      failed := failed + 1
    total := total + 1
    if !(← runFixedSourceLaw "reverse-swaps-delay-fixedpoint"
          Tropical.EmitArrow.revSwapLhsClock Tropical.EmitArrow.revSwapRhsClock
          arena) then
      failed := failed + 1
    total := total + 1
    if !(← runFixedSourceLaw "additive-fixedpoint"
          Tropical.EmitArrow.addLawLhsClock Tropical.EmitArrow.addLawRhsClock
          arena) then
      failed := failed + 1
    -- ── (h″) convolution stress test: the bubble executed; oracle independent of
    --   the lowering (array-shift of the bare osc), so it can catch a wrong-but-
    --   self-consistent lowering — correct-by-facade goes red here, not green.
    IO.println "convolution stress test (clock-warp FIR ≡ independent array-shift conv):"
    total := total + 1
    if !(← runConvolutionOracle arena resolved) then
      failed := failed + 1
    -- ── (h‴) modulated clock: a fractional NONLINEAR warp vs an independent
    --   closed-form reference, calibrated against the bare osc — tests that the
    --   bubble is not a side-effect of affineness.
    IO.println "modulated-clock stress test (fractional nonlinear warp ≡ closed form):"
    total := total + 1
    if !(← runModulatedClock arena resolved) then
      failed := failed + 1
    -- ── (h⁗) PM-of-PM: the modulator is itself a warped oscillator; bit-exact
    --   against a 3-level nested standard rep ⇒ the substitution composes.
    IO.println "pm-of-pm stress test (nested warp ≡ nested standard rep):"
    total := total + 1
    if !(← runPmPm arena resolved) then
      failed := failed + 1
    -- ── (h⁵) negative-time boundary (the moat): a delay pulls the clock < 0;
    --   random access gives the exact backward sine where a stream would zero-pad.
    IO.println "negative-time boundary (random access ≠ streaming zero-pad):"
    total := total + 1
    if !(← runNegativeClock arena resolved) then
      failed := failed + 1
    -- ── (h⁶) products / MIMO: a real multi-port body from the cartesian
    --   combinators vs a straight-line standard rep — the DATA axis (the warp
    --   gates above are all the CLOCK axis).
    IO.println "products/MIMO standard-rep differential (multi-port body ≡ closed form):"
    total := total + 1
    if !(← runMorphOscDifferential arena resolved) then
      failed := failed + 1
    -- ── (h⁷) THE SLIDE (WARP-PUSH): the compiler pushes a downstream effect's
    --   warps up to the generators. R1 (commute past arr) + the cascade
    --   multiplicity; Test 1 (byte-identity vs FlangeSin) is in the corpus block.
    IO.println "warp-push slide (downstream insert → upstream warp, by the compiler):"
    total := total + 1
    if !(← runSlidePastArr arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runSlideCascade arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runSlideProd arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBootstrapSin arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBootstrapExp arena resolved) then
      failed := failed + 1
    IO.println "modal island (decaying-resonator bank as a term over the clock):"
    total := total + 1
    if !(← runModalBank arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalDegree arena resolved) then
      failed := failed + 1
    IO.println "residue calculus (voice ⋙ reverb composed at build time):"
    total := total + 1
    if !(← runResidueMoments arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalReverb arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runResidueDegenerate arena resolved) then
      failed := failed + 1
    -- ── (h⁸) THE PATCHER LOWERING: downstream-only patch graph → arrow term →
    --   slide → emit. L1 (byte-identity vs FlangeSin from a graph) is in the
    --   corpus block; here L2 (chain graph ≡ hand-term) and L3 (fan-out + mix).
    IO.println "patcher lowering (downstream-only patch graph → arrow term):"
    total := total + 1
    if !(← runLoweringChain arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runLoweringFanOut arena resolved) then
      failed := failed + 1
    -- modulated-effect node: signal-into-clock (FM/PM), the osc → flange →
    -- osc.fm edge; ≡ the bit-exact-proven FM / PM-of-PM carriers.
    total := total + 1
    if !(← runModulatedNode arena resolved) then
      failed := failed + 1

  IO.println ""
  IO.println s!"{total - failed}/{total} passed"
  return if failed == 0 then 0 else 1
