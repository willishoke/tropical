import Tropical.Ffi
import Tropical.Engine
import Tropical.Plan
import Tropical.Ir.EmitLlvm
import Tropical.PlanDecode
import Tropical.Parse.Surface.Markdown
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
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

  IO.println ""
  IO.println s!"{total - failed}/{total} passed"
  return if failed == 0 then 0 else 1
