import Tropical.Ffi
import Tropical.Engine
import Tropical.StagedLoad
import Tropical.Playground
import Tropical.Plan
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
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
    samples). Lean owns codegen: parse the plan, stage-0 split + emit IR
    (StagedLoad → EmitLlvm), load via load_ir_staged. The goldens byte-gate
    the split: hoisting must not move a single output bit. -/
def renderPlanBytes (planJson : String) : IO ByteArray := do
  let plan ← match Lean.Json.parse planJson with
    | .error e => throw (IO.userError s!"renderPlanBytes: parse: {e}")
    | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
      | .error e => throw (IO.userError s!"renderPlanBytes: ofWire: {e}")
      | .ok p => pure p
  let rt ← Tropical.Ffi.Runtime.new BUFFER.toUInt32
  Tropical.StagedLoad.load rt plan
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

/-- Render a FlatPlan via the Lean-emitted-IR path (stage-0 split →
    EmitLlvm → load_ir_staged). -/
def renderIrBytes (plan : Tropical.Plan.FlatPlan) : IO (Except String ByteArray) := do
  try
    let rt ← Tropical.Ffi.Runtime.new BUFFER.toUInt32
    Tropical.StagedLoad.load rt plan
    let mut acc := ByteArray.empty
    for _ in [0:FRAMES] do
      rt.process
      acc := acc ++ (← rt.outputBytes)
    pure (.ok acc)
  catch e => pure (.error e.toString)

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

-- ── Synthetic reduce-region plan (banks-as-data slice 3a) ────────────────────
-- The indexed-reduction primitive, gated by construction: the SAME
-- computation — Σₖ toInt(table[k]·2²⁸)·(k+1), an i64 modular sum over a
-- packed table, scaled back to float and mixed with τ so the output is
-- audible and per-sample — built twice, as a ReduceBegin/ReduceEnd loop
-- and fully unrolled. The i64 sum is associative, so the two must render
-- BYTE-IDENTICAL bits; a frozen hash pins both against regression.
section ReduceCoverage
open Tropical.Plan

-- (jn/cF/cI/rgF/rgI are the OpCoverage section's private helpers.)

/-- The table: 4 floats packed into array slot 0. -/
private def tableInstrs : Array NInstr :=
  #[instrPack 0 #[cF 15 1, cF (-225) 2, cF 3, cF 5 1]]

/-- One mode's contribution given the index operand `k` (temps `t`..): -/
private def modeBody (t : Nat) (k : NOperand) : Array NInstr := #[
  instrIndex t #[.arrayReg 0, k] .float,                        -- v = table[k]
  instrScalar "Mul" (t+1) #[rgF t, cF 268435456] .float,        -- v·2²⁸
  instrScalar "ToInt" (t+2) #[rgF (t+1)] .int,
  instrScalar "Add" (t+3) #[k, cI 1] .int,                      -- k+1
  instrScalar "Mul" (t+4) #[rgI (t+2), rgI (t+3)] .int]         -- w·(k+1)

/-- Shared tail: scale the i64 accumulator (temp `acc`) to float, mix
    with τ (so the render is per-sample), write the output slot. -/
private def tailInstrs (acc t : Nat) : Array NInstr := #[
  instrScalar "ToFloat" t #[rgI acc] .float,
  instrScalar "Div" (t+1) #[rgF t, cF 268435456] .float,
  instrScalar "ToFloat" (t+2) #[Tropical.Plan.opTick] .float,
  instrScalar "Mod" (t+3) #[rgF (t+2), cF 64] .float,
  instrScalar "Mul" (t+4) #[rgF (t+1), rgF (t+3)] .float,
  instrWriteSlot 0 (rgF (t+4))]

private def reducePlanOf (body : Array NInstr) (regCount : Nat) : FlatPlan :=
  let inst := InstanceFunction.mk "root" "root" #[] body #[] 0 0 regCount #[]
  { sampleRate := jn 44100, compilationMode := .fused,
    arraySlotNames := #["table"], registerCount := regCount, arraySlotCount := 1,
    arraySlotSizes := #[4], instanceFunctions := #[inst],
    sinks := #[{ inputs := #[0], gain := jn 1, target := 0 }],
    sources := defaultSources, slotCount := 1, slotNames := #["out"],
    slotDefaults := #[Lean.Json.num (jn 0)] }

/-- The loop form: acc = temp 1; body temps 2..6; tail temps 7.. -/
private def reduceLoopPlan : FlatPlan :=
  let body := tableInstrs
    ++ #[instrReduceBegin 1 (cI 0) 4 .int]
    ++ modeBody 2 .loopIdx
    ++ #[instrScalar "Add" 1 #[rgI 1, rgI 6] .int,
         instrReduceEnd 1 .int]
    ++ tailInstrs 1 7
  reducePlanOf body 12

/-- The unrolled twin: same ops, literal indices, explicit adds. -/
private def reduceUnrolledPlan : FlatPlan := Id.run do
  let mut body := tableInstrs
  let mut t := 2
  let mut accs : Array Nat := #[]
  for k in [0:4] do
    body := body ++ modeBody t (cI (Int.ofNat k))
    accs := accs.push (t+4)
    t := t + 5
  -- acc = ((0 + w0) + w1) + w2 + w3 — the loop's fold order.
  body := body.push (instrScalar "Add" t #[cI 0, rgI accs[0]!] .int)
  for i in [1:4] do
    body := body.push (instrScalar "Add" (t+i) #[rgI (t+i-1), rgI accs[i]!] .int)
  body := body ++ tailInstrs (t+3) (t+4)
  reducePlanOf body (t+9)

private def runReduceCoverage : IO Bool := do
  match ← renderIrBytes reduceLoopPlan, ← renderIrBytes reduceUnrolledPlan with
  | .ok looped, .ok unrolled =>
    if looped != unrolled then
      IO.println "  FAIL  reduce-coverage  loop and unrolled renders differ"
      return false
    let got ← sha256Hex looped
    let expected := "7e5cea9663274157d79741c0c29cd36f3ce76bf6c4d3ae5bcd736050db68a0db"
    if got != expected then
      IO.println s!"  FAIL  reduce-coverage  expected {expected.take 16} got {got.take 16}"
      return false
    -- MSL smoke: the region must emit as a real for-loop (full SNR
    -- gating arrives with the modal banked plans in slice 3b).
    match Tropical.Ir.EmitMsl.emitKernel reduceLoopPlan with
    | .error e =>
      IO.println s!"  FAIL  reduce-coverage  EmitMsl: {firstLine e}"; pure false
    | .ok msl =>
      if (msl.splitOn "for (long rd").length >= 2 then
        IO.println s!"  PASS  reduce-coverage  loop ≡ unrolled, hash {got.take 16}, MSL loop emitted"
        pure true
      else
        IO.println "  FAIL  reduce-coverage  MSL kernel has no reduce loop"
        pure false
  | .error e, _ => IO.println s!"  FAIL  reduce-coverage  loop: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  reduce-coverage  unrolled: {firstLine e}"; pure false

end ReduceCoverage

private def sortedNames (dir : String) (suffix : String) : IO (Array String) := do
  let entries ← (System.FilePath.mk dir).readDir
  let names := entries.filterMap fun e =>
    if e.fileName.endsWith suffix then some (e.fileName.dropRight suffix.length) else none
  pure (names.qsort fun a b => decide (a < b))

/-- Compile a patch through the session mirror, returning the plan plus
    the typed per-instruction stage blocks (the split classification). -/
def compilePatchStaged (path : String) :
    IO (Except String (Tropical.Plan.FlatPlan
      × Array (Array (Option Tropical.Ir.Stage)))) := do
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM _ := do
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str path)])
    Tropical.Engine.compileMirrorStaged env .fused
  match ← act.run with
  | .ok (_, plan, blocks) => pure (.ok (plan, blocks))
  | .error f => pure (.error f.toJson.compress)

/-- Render a FlatPlan via the TYPED split (stage attribute →
    hoistTyped → EmitLlvm → load_ir_staged). -/
def renderTypedBytes (plan : Tropical.Plan.FlatPlan)
    (blocks : Array (Array (Option Tropical.Ir.Stage))) : IO ByteArray := do
  let rt ← Tropical.Ffi.Runtime.new BUFFER.toUInt32
  Tropical.StagedLoad.loadTyped rt plan blocks
  let mut acc := ByteArray.empty
  for _ in [0:FRAMES] do
    rt.process
    acc := acc ++ (← rt.outputBytes)
  pure acc

/-- Compile a patch (fused) and hash its rendered 16×256 output. The
    render goes through the TYPED split — the goldens byte-gate it. -/
private def hashOf (patchPath : String) : IO (Except String String) := do
  match ← compilePatchStaged patchPath with
  | .error e => pure (.error e)
  | .ok (plan, blocks) => pure (.ok (← sha256Hex (← renderTypedBytes plan blocks)))

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

/-- Compile a patch and emit its Metal kernel source (the EmitMsl path). -/
private def emitMslOf (patchPath : String) : IO (Except String String) := do
  match ← compilePatch patchPath .fused with
  | .error e => pure (.error e)
  | .ok planJson =>
    match Lean.Json.parse planJson with
    | .error e => pure (.error s!"parse: {e}")
    | .ok j =>
      match Tropical.Plan.FlatPlan.ofWire j with
      | .error e => pure (.error s!"ofWire: {e}")
      | .ok plan => pure (Tropical.Ir.EmitMsl.emitKernel plan)

/-- MSL GOLDEN: the emitted Metal kernel text for a patch, frozen under
    `tests/golden/msl/<name>.metal`. Text-frozen (like the IR philosophy:
    emitter changes must be deliberate re-freezes, never drift), and it
    carries the folded i64 landing constants — the byte-exactness claim
    for the integer datapath on the GPU lives in this text. -/
private def runMslGolden (writeMode : Bool) (name patchPath : String) : IO Bool := do
  let goldenPath := s!"tests/golden/msl/{name}.metal"
  match ← emitMslOf patchPath with
  | .error e => IO.println s!"  FAIL  msl-golden/{name}  {firstLine e}"; pure false
  | .ok msl =>
    if writeMode then
      IO.FS.writeFile goldenPath msl
      IO.println s!"  WROTE msl-golden/{name}  ({msl.length}B)"; pure true
    else
      match ← (try (pure (some (← IO.FS.readFile goldenPath))) catch _ => pure none) with
      | none => IO.println s!"  FAIL  msl-golden/{name}  missing {goldenPath} (run --write)"; pure false
      | some expected =>
        if msl == expected then
          IO.println s!"  PASS  msl-golden/{name}  ({msl.length}B, text-frozen)"; pure true
        else
          IO.println s!"  FAIL  msl-golden/{name}  emitted MSL differs from frozen ({msl.length}B vs {expected.length}B)"; pure false

/-- THE FOLD gate: EmitMsl's emit-time f64 constant folding must land the
    LITERAL-frequency phase increment as the exact i64 the CPU computes —
    `toInt(440·2³²/44100)` evaluated here in f64, asserted present in the
    emitted text as a `long` literal. This is the byte-exact-phase claim
    for literal patches on the f32 GPU (design/fixed-carrier.md). -/
private def runMslFold : IO Bool := do
  let expected : Int := Int.ofNat ((440.0 * 4294967296.0 / 44100.0).toUInt64.toNat)
  match ← emitMslOf "web/patches/pure-sine-440.json" with
  | .error e => IO.println s!"  FAIL  msl-fold  {firstLine e}"; pure false
  | .ok msl =>
    if (msl.splitOn s!"{expected}L").length > 1 then
      IO.println s!"  PASS  msl-fold  literal landing folded in f64: increment {expected} present as i64 in the kernel"; pure true
    else
      IO.println s!"  FAIL  msl-fold  expected folded increment {expected}L not found in emitted MSL"; pure false

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
    | .ok plan => do
      let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
      Tropical.StagedLoad.load rt plan
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
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved { upto := 5 } arena' idx).mapError (·.message)
  -- The carrier is the synthetic session root, wired straight to the dac at its
  -- `out` port (`__root__.out`). No session wires, no params, no inputs.
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    arena := coreArena
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
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved
      { upto := Tropical.Ir.Strata.portedPasses, inlineNested := true } arena idx).mapError (·.message)
  let plan ← Tropical.Ir.CompileResolved.compileResolved core coreArena
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

/-- Count instructions with a given tag across an InstanceFunction tree. -/
private partial def countFnTagged (t : String) (f : Tropical.Plan.InstanceFunction) : Nat :=
  (f.instructions.filter (·.tag == t)).size
    + f.children.foldl (fun acc c => acc + countFnTagged t c) 0

/-- Count instructions with a given tag across a FlatPlan (e.g. "ReduceBegin"
    to count reduce regions, "Pack" to count materialized columns). -/
private def planTagCount (t : String) (p : Tropical.Plan.FlatPlan) : Nat :=
  p.instanceFunctions.foldl (fun acc f => acc + countFnTagged t f) 0

/-- Strata-process an already-built carrier program, then compile it via the
    production session path. Returns (post-strata binderCount, post-strata
    declCount, FlatPlan) — the post-strata program is the CSE'd DAG, so its
    binderCount is the count of distinct shared subexpressions. -/
private def diagStrataCompile (arena : Arena) (idx : ProgramIdx) :
    Except String (Nat × Nat × Tropical.Plan.FlatPlan) := do
  let (arena'', root') ← (Tropical.Ir.Strata.run { upto := 5 } arena idx).mapError (·.message)
  let some prog := arena''.program? root'
    | .error "diagonal: post-strata root program index out of range"
  let (coreArena, core) ← Tropical.Ir.checkResolvedArena arena'' root'
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    arena := coreArena
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
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved { upto := 5 } arena' idx).mapError (·.message)
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    arena := coreArena
    mode := .fused }
  Tropical.Compile.compileSession input

/-- Render a `FlatPlan` to exactly `n` contiguous mono samples (buffer = n, no
    fade), like `renderSamples` but from an in-hand plan. -/
private def renderPlanSamples (plan : Tropical.Plan.FlatPlan) (n : Nat) :
    IO (Except String (Array Float)) := do
  try
    let rt ← Tropical.Ffi.Runtime.new (UInt32.ofNat n)
    Tropical.StagedLoad.load rt plan
    rt.process
    pure (.ok (decodeF64LE (← rt.outputBytes)))
  catch e => pure (.error e.toString)

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
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved { upto := 5 } arena idx).mapError (·.message)
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := #[]
    alloc := {}
    root := core
    arena := coreArena
    mode := .fused }
  Tropical.Compile.compileSession input

private def buildAndFinish (built : Except String (Arena × ProgramIdx)) :
    Except String Tropical.Plan.FlatPlan := do
  let (a, i) ← built
  finishCarrier a i

/-- `stdlib/FixedSin.md` transcribed exactly in Lean Int arithmetic: the Q2.30
    datapath sine at a MASKED Q0.32 cycles phase. `Int.fdiv` is floor division
    = the engine's `ashr`; every Horner operand is non-negative by construction
    (all-positive-with-subtractions), so floor = truncate there; the final
    `(r·acc₀) >> 30` is the one signed floor-shift, matched exactly. -/
private def fixedSinQ (p : Int) : Int :=
  let n := Int.fdiv (p + 1073741824) 2147483648
  let r := p - n * 2147483648
  let sign := 1 - 2 * (Int.fmod n 2)
  let z := Int.fdiv (r * r) 1073741824
  let acc6 := 61 - Int.fdiv z 1073741824
  let acc5 := 3864 - Int.fdiv (acc6 * z) 1073741824
  let acc4 := 172272 - Int.fdiv (acc5 * z) 1073741824
  let acc3 := 5026995 - Int.fdiv (acc4 * z) 1073741824
  let acc2 := 85569306 - Int.fdiv (acc3 * z) 1073741824
  let acc1 := 693598668 - Int.fdiv (acc2 * z) 1073741824
  let acc0 := 1686629713 - Int.fdiv (acc1 * z) 1073741824
  sign * Int.fdiv (r * acc0) 1073741824

/-- The voice sine as the engine now computes it: re-land the float phase as
    its exact Q0.32 integer (lossless — P < 2³² ≪ 2⁵³), run `fixedSinQ`, scale
    Q2.30 → float. The standard-rep twin of `FixedSin(toInt(phase·2³²))/2³⁰`. -/
private def voiceSin (phase : Float) : Float :=
  Float.ofInt (fixedSinQ ((phase * 4294967296.0).toUInt64.toNat)) / 1073741824.0

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
        let refBare := sinkGain * voiceSin (phasorPhase clk 2000)
        if (bare[t]! - refBare).abs > e0 then e0 := (bare[t]! - refBare).abs
        if bare[t]!.toBits != refBare.toBits then calBitDiff := calBitDiff + 1
        if bare[t]!.abs > maxBare then maxBare := bare[t]!.abs
        -- the warp: mid-graph (unit-scale) modulator = Sin at the modulator phase;
        -- offset = toInt(depth·mod·2³²); φ = clk − offset (sub-sample, nonlinear)
        let rawMod := voiceSin (phasorPhase clk 200)
        let phi : Int := clk - truncToInt (depth * rawMod * two32)
        let refFm := sinkGain * voiceSin (phasorPhase phi 2000)
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
        let mod2 := voiceSin (phasorPhase clk 700)
        let modClk : Int := clk - truncToInt (d2 * mod2 * two32)
        let mod := voiceSin (phasorPhase modClk 200)
        let carClk : Int := clk - truncToInt (d1 * mod * two32)
        let ref := sinkGain * voiceSin (phasorPhase carClk 2000)
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
        let ref := sinkGain * voiceSin (phasorPhase phi 2000)
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
    sinkGain * ((1.0 - morphF) * (2.0 * phase - 1.0) + morphF * voiceSin phase)
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

/-- THE FIXED-SINE ACCURACY gate. `fixedSinCycSig` (the Q2.30 integer-datapath
    sine) over the integer phasor, rendered by the engine, vs the TRUE sine at
    the exactly-known phase: the phasor model `P(i) = (21426140·i) mod 2³²` is
    replicated in Lean Int arithmetic, so the oracle `sin(2π·P/2³²)` is
    independent of every polynomial under test (a transcribed-coefficient typo
    or a mis-shifted Horner step shows up directly). Budget: coefficient
    rounding + 9 floor-shifts ≈ 1e-8 abs on the sin scale (−160 dB). -/
private def runFixedSinAccuracy (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier "fixedsin_acc"
      (Tropical.EmitArrow.fixedOutQ 30
        (Tropical.EmitArrow.fixedSinCycSig
          (Tropical.EmitArrow.fixedPhase Tropical.EmitArrow.clockLit))) arena)) with
  | .ok p =>
    match ← renderPlanSamples p 4096 with
    | .ok s =>
      let n := min s.size 4096
      let sinkGain : Float := 0.05
      let twoPi : Float := 6.283185307179586
      let mut maxAbs : Float := 0.0
      let mut worstI : Nat := 0
      for i in [0:n] do
        let pQ : Int := (21426140 * (Int.ofNat i)) % 4294967296
        let ref := Float.sin (twoPi * (Float.ofInt pQ) / 4294967296.0)
        let d := (s[i]! / sinkGain - ref).abs
        if d > maxAbs then
          maxAbs := d
          worstI := i
      IO.println s!"        fixedSin(Q0.32 phasor @220) vs true sin at the exact integer phase, 4096 samples:"
      IO.println s!"        result   max abs error (sin scale) = {maxAbs * 1e9}e-9  (at sample {worstI})"
      if maxAbs < 2e-8 then
        IO.println s!"  PASS  fixedsin-accuracy  Q2.30 datapath sine ≡ true sine to {maxAbs * 1e9}e-9 (≈ −160 dB floor)"; pure true
      else
        IO.println s!"  FAIL  fixedsin-accuracy  max abs err {maxAbs * 1e9}e-9 (want <2e-8) at sample {worstI}"; pure false
    | .error e => IO.println s!"  FAIL  fixedsin-accuracy  render: {firstLine e}"; pure false
  | .error e => IO.println s!"  FAIL  fixedsin-accuracy  build: {firstLine e}"; pure false

/-- THE FIXED-SINE LONG-τ gate. The fixed oscillator read 2³⁰+12345 samples
    into the future must equal the origin oscillator phase-shifted by the
    EXACTLY-computable Q0.32 offset `(inc·K) mod 2³²` — modular arithmetic on
    the circle, byte-for-byte, at any τ. (The float carrier had no such
    identity: its phase argument grew without bound.) K deliberately has low
    bits set so nothing is accidentally exact. -/
private def runFixedSinLongTau (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let K : Int := 1073741824 + 12345
  let Kq : Int := K * 4294967296
  let offset : Int := (21426140 * K) % 4294967296
  let farOsc := Tropical.EmitArrow.fixedOutQ 30
    (Tropical.EmitArrow.fixedSinCycSig
      (Tropical.EmitArrow.fixedPhase
        (Tropical.EmitArrow.add Tropical.EmitArrow.clockLit (Tropical.EmitArrow.litI Kq))))
  let shiftedOsc := Tropical.EmitArrow.fixedOutQ 30
    (Tropical.EmitArrow.fixedSinCycSig
      (Tropical.EmitArrow.bitAnd
        (Tropical.EmitArrow.add
          (Tropical.EmitArrow.fixedPhase Tropical.EmitArrow.clockLit)
          (Tropical.EmitArrow.litI offset))
        (Tropical.EmitArrow.lit 4294967295)))
  match buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier "fixedsin_lt_far" farOsc arena)),
        buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier "fixedsin_lt_shift" shiftedOsc arena)) with
  | .ok fp, .ok sp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples sp 2048 with
    | .ok far, .ok shifted =>
      let n := min far.size shifted.size
      let mut bitDiff := 0
      for i in [0:n] do
        if far[i]! != shifted[i]! then bitDiff := bitDiff + 1
      let mut energy : Float := 0.0
      for i in [0:n] do energy := energy + far[i]! * far[i]!
      IO.println s!"        fixed osc @ clk+(2³⁰+12345) samples vs origin osc phase-shifted (inc·K mod 2³²):"
      IO.println s!"        result   bit-differing {bitDiff}/{n}  ·  energy={energy}"
      if bitDiff == 0 && energy > 1e-6 then
        IO.println s!"  PASS  fixedsin-longtau  modular phase identity byte-exact at τ+2³⁰ samples"; pure true
      else
        IO.println s!"  FAIL  fixedsin-longtau  bitDiff={bitDiff} (want 0) energy={energy} (>1e-6)"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  fixedsin-longtau  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  fixedsin-longtau  build: {firstLine e}"; pure false

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
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 220) (Tropical.EmitArrow.lit 30 1) (Tropical.EmitArrow.lit 6 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 330) (Tropical.EmitArrow.lit 40 1) (Tropical.EmitArrow.lit 4 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 440) (Tropical.EmitArrow.lit 55 1) (Tropical.EmitArrow.lit 3 1)]
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

/-- Warn-only cost ratchet (gates with rank): the FLATNESS assertions in the
    banks gates are HARD — an asymptotic regression means banking is broken,
    not slow, and that check never false-positives on an honest change. The
    CONSTANTS are honest-change territory — a legitimate emit change may move
    them — so drift prints a WARN line (never fails). Refreeze deliberately by
    updating the constant at the call site. Warnings rot when nobody reads
    them; this one is a single greppable token: `WARN`. -/
private def warnBenchConst (gate what : String) (frozen got : Nat) : IO Unit :=
  if got != frozen then
    IO.println s!"  WARN  {gate}  {what}: {got} (frozen {frozen}) — cost constant drifted; refreeze deliberately"
  else pure ()

open Tropical.EmitArrow in
/-- THE BANKS-AS-DATA gate (slice 3b). A decaying-resonator bank lowered through
    the INDEXED REDUCTION (`modalBankSigTable` → `Sig.bankSum` → a `ReduceBegin`
    region) must render BIT-FOR-BIT identical to the same bank unrolled
    (`modalBankSigDirect`) — the i64-modular mode sum is associative, so the loop
    and the fold agree to the bit. This exercises the whole new path end to end:
    `Sig.arr`/`index`/`loopIdx`/`bankSum` through every strata pass, the
    `ENode→CNode` downcast, and the emit-time reduce-region lowering. We also
    assert the PAYOFF: banking shrinks the plan, and the per-mode MARGINAL
    instruction cost drops (the DSP body no longer unrolls — only the coefficient
    fills still scale, and those are destined for the s0 kernel next). -/
private def runBanksAsData (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- A K-mode decaying bank, all deg-0 (the uniform datapath). Frequencies spread
  -- so the bank is non-trivial; small amps so the i64 sum stays in headroom.
  let mkModes (k : Nat) : Array ModalMode :=
    (Array.range k).map fun i =>
      ModalMode.hz (lit (Int.ofNat (220 + 40 * i))) (lit 30 1) (lit 2 1)
  let anchor := lit 200
  let modes := mkModes 12
  let directPlan := buildAndFinish (.ok (buildModalBankDirect "bank_unrolled" modes anchor arena))
  let tablePlan  := buildAndFinish (.ok (buildModalBankTable  "bank_looped"   modes anchor arena))
  match directPlan, tablePlan with
  | .ok dp, .ok tp =>
    match ← renderPlanSamples dp 2048, ← renderPlanSamples tp 2048 with
    | .ok dS, .ok tS =>
      let n := min dS.size tS.size
      let mut bitDiff := 0
      for i in [0:n] do
        if dS[i]! != tS[i]! then bitDiff := bitDiff + 1
      let mut preMax : Float := 0.0
      for i in [0:200] do
        let a := tS[i]!.abs
        if a > preMax then preMax := a
      let mut eEarly : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [200:600] do eEarly := eEarly + tS[i]! * tS[i]!
      for i in [1648:2048] do eLate := eLate + tS[i]! * tS[i]!
      -- Compile-scaling: the same bank at two mode counts, both lowerings.
      let dSmall := buildAndFinish (.ok (buildModalBankDirect "d6"  (mkModes 6)  anchor arena))
      let dBig   := buildAndFinish (.ok (buildModalBankDirect "d24" (mkModes 24) anchor arena))
      let tSmall := buildAndFinish (.ok (buildModalBankTable  "t6"  (mkModes 6)  anchor arena))
      let tBig   := buildAndFinish (.ok (buildModalBankTable  "t24" (mkModes 24) anchor arena))
      match dSmall, dBig, tSmall, tBig with
      | .ok ds, .ok db, .ok ts, .ok tb =>
        let dMarginal := planInstrCount db - planInstrCount ds   -- unrolled per-mode marginal (×18)
        let tMarginal := planInstrCount tb - planInstrCount ts   -- banked marginal (fills only)
        let shrinks := decide (planInstrCount tp < planInstrCount dp)
        IO.println s!"        bank = Σ amp·e^(−σd)·cos(2πf·d), 12 modes, struck @ 200 — unrolled vs looped:"
        IO.println s!"        result   bit-differing {bitDiff}/{n}  ·  pre-strike |max|={preMax}  ·  E[early]={eEarly}  E[late]={eLate}"
        IO.println s!"        payoff   plan-instrs 12-mode: unrolled={planInstrCount dp} looped={planInstrCount tp} (shrinks={shrinks})"
        IO.println s!"        payoff   per-mode marginal (6→24 modes): unrolled +{dMarginal}  ·  banked +{tMarginal} (body no longer unrolls)"
        warnBenchConst "banks-as-data" "12-mode looped plan-instrs" 184 (planInstrCount tp)
        warnBenchConst "banks-as-data" "banked per-mode marginal (6→24)" 72 tMarginal
        if bitDiff == 0 && preMax == 0.0 && eEarly > 1e-6 && eLate < eEarly
           && shrinks && tMarginal < dMarginal then
          IO.println s!"  PASS  banks-as-data  looped ≡ unrolled bit-exact ({n} samples), causal, decaying; plan shrinks, marginal +{tMarginal}<+{dMarginal}"; pure true
        else
          IO.println s!"  FAIL  banks-as-data  bitDiff={bitDiff} preMax={preMax} shrinks={shrinks} tMarg={tMarginal} dMarg={dMarginal}"; pure false
      | _, _, _, _ =>
        IO.println s!"  FAIL  banks-as-data  scaling build failed"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-as-data  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-as-data  build: {firstLine e}"; pure false

open Tropical.EmitArrow in
/-- THE BANKS-AS-DATA DIRECTION gate. The direction bank lowered through TWO
    indexed reductions over one set of coefficient columns
    (`modalBankSigDirTable` — the forward and reverse accumulators as two
    `bankFold`s) must render BIT-FOR-BIT identical to the unrolled pair-fold
    (`modalBankSigDir`), across the crossfade (dir = 0.5), the pure reverse
    (dir = 1 — the ANTI-CAUSAL region must actually carry energy, so the mirrored
    phase `modePhaseQFromIncr(incr, −clkRel)` is genuinely exercised), and the
    sway path (`dampScale?` threading the columns). Also asserts the payoff:
    the banked plan shrinks, and both regions loop (per-mode marginal collapses).
    This is the gate that retires the "hand-bank every effect" objection: no
    direction table twin exists — both sides route through the SAME generic
    `bankFold`, and this gate pins that the generic path carries a richer body
    (two accumulators, mirrored phase, sway) without a transcription step. -/
private def runBanksAsDataDir (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mkModes (k : Nat) : Array ModalMode :=
    (Array.range k).map fun i =>
      ModalMode.hz (lit (Int.ofNat (220 + 40 * i))) (lit 30 1) (lit 2 1)
  let anchor := lit 200
  let modes := mkModes 12
  let sway : Option (Sig × Sig) := some (lit 5 1, lit 20 1)
  -- explicit lambdas: Lean eta-expands optParam references by inserting the
  -- default, which would drop the `dampScale?` slot from the function type
  let unrolled : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig :=
    fun ms clk a d s? => modalBankSigDir ms clk a d s?
  let looped : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig :=
    fun ms clk a d s? => modalBankSigDirTable ms clk a d s?
  -- Three configs: crossfade, pure reverse, crossfade+sway — each unrolled vs banked.
  let cfgs : Array (String × Sig × Option (Sig × Sig)) :=
    #[("mid", litF 0.5, none), ("rev", lit 1, none), ("sway", litF 0.5, sway)]
  let mut ok := true
  let mut planPair : Option (Nat × Nat) := none
  for (tag, dir, damp?) in cfgs do
    let uPlan := buildAndFinish (.ok (buildModalBankDirWith unrolled
      s!"dir_{tag}_unrolled" modes anchor dir arena damp?))
    let tPlan := buildAndFinish (.ok (buildModalBankDirWith looped
      s!"dir_{tag}_looped" modes anchor dir arena damp?))
    match uPlan, tPlan with
    | .ok up, .ok tp =>
      match ← renderPlanSamples up 2048, ← renderPlanSamples tp 2048 with
      | .ok uS, .ok tS =>
        let n := min uS.size tS.size
        let mut bitDiff := 0
        for i in [0:n] do
          if uS[i]! != tS[i]! then bitDiff := bitDiff + 1
        -- the reverse config must carry PRE-STRIKE energy (the anti-causal loop lives)
        let mut preE : Float := 0.0
        for i in [0:200] do preE := preE + tS[i]! * tS[i]!
        let preOk := tag != "rev" || preE > 1e-6
        if tag == "mid" then planPair := some (planInstrCount up, planInstrCount tp)
        if bitDiff != 0 || !preOk then
          IO.println s!"        dir[{tag}]  bitDiff={bitDiff}/{n} preE={preE} — MISMATCH"
          ok := false
        else
          IO.println s!"        dir[{tag}]  bit-identical ({n} samples){if tag == "rev" then s!", pre-strike E={preE}" else ""}"
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  banks-as-data-dir  render[{tag}]: {firstLine e}"; ok := false
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  banks-as-data-dir  build[{tag}]: {firstLine e}"; ok := false
  -- Payoff: banked direction plan shrinks at 12 modes; per-mode marginal collapses
  -- (BOTH regions loop — the marginal is the column fills alone).
  let uSmall := buildAndFinish (.ok (buildModalBankDirWith unrolled "du6"  (mkModes 6)  anchor (litF 0.5) arena))
  let uBig   := buildAndFinish (.ok (buildModalBankDirWith unrolled "du24" (mkModes 24) anchor (litF 0.5) arena))
  let tSmall := buildAndFinish (.ok (buildModalBankDirWith looped "dt6"  (mkModes 6)  anchor (litF 0.5) arena))
  let tBig   := buildAndFinish (.ok (buildModalBankDirWith looped "dt24" (mkModes 24) anchor (litF 0.5) arena))
  match planPair, uSmall, uBig, tSmall, tBig with
  | some (uc, tc), .ok us, .ok ub, .ok ts, .ok tb =>
    let uMarginal := planInstrCount ub - planInstrCount us
    let tMarginal := planInstrCount tb - planInstrCount ts
    let shrinks := decide (tc < uc)
    IO.println s!"        payoff   plan-instrs 12-mode: unrolled={uc} looped={tc} (shrinks={shrinks})"
    IO.println s!"        payoff   per-mode marginal (6→24 modes): unrolled +{uMarginal}  ·  banked +{tMarginal}"
    warnBenchConst "banks-as-data-dir" "12-mode looped plan-instrs" 317 tc
    warnBenchConst "banks-as-data-dir" "banked per-mode marginal (6→24)" 72 tMarginal
    if ok && shrinks && tMarginal < uMarginal then
      IO.println s!"  PASS  banks-as-data-dir  looped ≡ unrolled bit-exact (mid/rev/sway), reverse audible; plan shrinks, marginal +{tMarginal}<+{uMarginal}"; pure true
    else
      IO.println s!"  FAIL  banks-as-data-dir  ok={ok} shrinks={shrinks} tMarg={tMarginal} uMarg={uMarginal}"; pure false
  | _, _, _, _, _ =>
    IO.println s!"  FAIL  banks-as-data-dir  scaling build failed"; pure false

open Tropical.EmitArrow in
/-- THE FLOAT-BANK gate (typed accumulator). A FLOAT fold lowered through
    `Sig.bankSum` must render bit-identical to the same fold unrolled. This is
    the claim that banking needs NO algebraic precondition: the loop visits
    elements in the order the unroll nests its adds, so order preservation —
    not associativity — carries bit-exactness, floats included. (The i64
    restriction in the original `compileBankSum` was scaffolding; the
    accumulator now follows the body's type.) -/
private def runBanksFloat (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let k := 16
  -- t = τ seconds (the dSec recipe, anchor 0) — varies per sample so the sum
  -- is a live float datapath, not a constant the optimizer could fold away.
  let t := div (div (toFloatE clockLit) (lit 4294967296)) .sampleRate
  let amps := (Array.range k).map fun i => litF (0.31 + 0.07 * i.toFloat)
  let col := Sig.arr amps
  let unrolled := amps.foldl (fun acc a => add acc (mul a t)) (lit 0)
  let looped := Sig.bankSum k #[col] (mul (Sig.index col Sig.loopIdx) t) none
  let uPlan := buildAndFinish (.ok (buildExprCarrier "fbank_unrolled" unrolled arena))
  let tPlan := buildAndFinish (.ok (buildExprCarrier "fbank_looped" looped arena))
  match uPlan, tPlan with
  | .ok up, .ok tp =>
    match ← renderPlanSamples up 2048, ← renderPlanSamples tp 2048 with
    | .ok uS, .ok tS =>
      let n := min uS.size tS.size
      let mut bitDiff := 0
      for i in [0:n] do
        if uS[i]! != tS[i]! then bitDiff := bitDiff + 1
      let mut energy : Float := 0.0
      for i in [0:n] do energy := energy + tS[i]! * tS[i]!
      let shrinks := decide (planInstrCount tp < planInstrCount up)
      IO.println s!"        float bank Σₖ ampₖ·t, {k} elements — unrolled vs looped (f64 accumulator):"
      IO.println s!"        result   bit-differing {bitDiff}/{n} · energy={energy} · plan-instrs unrolled={planInstrCount up} looped={planInstrCount tp}"
      warnBenchConst "banks-float" "looped plan-instrs" 12 (planInstrCount tp)
      if bitDiff == 0 && energy > 1e-6 && shrinks then
        IO.println s!"  PASS  banks-float  looped ≡ unrolled bit-exact for a FLOAT fold (order preservation, no associativity); plan shrinks"; pure true
      else
        IO.println s!"  FAIL  banks-float  bitDiff={bitDiff} energy={energy} shrinks={shrinks}"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-float  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-float  build: {firstLine e}"; pure false

/-- THE TRUNK-FOLD gate (loop-everything). A surface-language SUMMING fold —
    through the FULL front door (raise → elaborate → strata → emit) — lowers to
    an indexed reduction, renders byte-identical to its hand-unrolled add chain,
    and the plan is FLAT in element count (the Pack carries the column; the loop
    body is O(1)). Horner folds (`acc·x + c`) are shape-ineligible and keep
    unrolling, so the transcendental stdlib is untouched — checked implicitly by
    every other gate in this suite. -/
private def runBanksFoldTrunk (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let jn (m : Nat) (e : Nat) : Lean.Json := Lean.Json.num ⟨Int.ofNat m, e⟩
  let amp (i : Nat) : Lean.Json := jn (31 + 7 * i) 2          -- 0.31 + 0.07·i, exact decimals
  let addJ (a b : Lean.Json) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "add"), ("args", Lean.Json.arr #[a, b])]
  let mulHalf (a : Lean.Json) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "mul"), ("args", Lean.Json.arr #[a, jn 5 1])]
  let binding (n : String) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "binding"), ("name", Lean.Json.str n)]
  let mkPatch (k : Nat) (unrolled : Bool) : Lean.Json :=
    let amps := (Array.range k).map amp
    let expr :=
      if unrolled then
        -- ((0 + c₀·½) + c₁·½) + … — the fold's own unroll order
        amps.foldl (fun acc a => addJ acc (mulHalf a)) (jn 0 0)
      else
        Lean.Json.mkObj [("op", Lean.Json.str "fold"), ("over", Lean.Json.arr amps),
          ("init", jn 0 0), ("acc_var", Lean.Json.str "acc"), ("elem_var", Lean.Json.str "e"),
          ("body", addJ (binding "acc") (mulHalf (binding "e")))]
    let inner := Lean.Json.mkObj [
      ("name", Lean.Json.str "FoldProbe"),
      ("ports", Lean.Json.mkObj [("inputs", Lean.Json.arr #[]),
        ("outputs", Lean.Json.arr #[Lean.Json.str "out"])]),
      ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
        ("decls", Lean.Json.arr #[]),
        ("assigns", Lean.Json.arr #[Lean.Json.mkObj [
          ("op", Lean.Json.str "outputAssign"), ("name", Lean.Json.str "out"),
          ("expr", expr)]])])]
    Lean.Json.mkObj [
      ("schema", Lean.Json.str "tropical_program_2"),
      ("name", Lean.Json.str "fold_trunk_probe"),
      ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
        ("decls", Lean.Json.arr #[
          Lean.Json.mkObj [("op", Lean.Json.str "programDecl"),
            ("name", Lean.Json.str "FoldProbe"), ("program", inner)],
          Lean.Json.mkObj [("op", Lean.Json.str "instanceDecl"),
            ("name", Lean.Json.str "p"), ("program", Lean.Json.str "FoldProbe"),
            ("inputs", Lean.Json.mkObj [])]]),
        ("assigns", Lean.Json.arr #[])]),
      ("audio_outputs", Lean.Json.arr #[Lean.Json.mkObj [
        ("instance", Lean.Json.str "p"), ("output", Lean.Json.str "out")]])]
  let compileAt (k : Nat) (unrolled : Bool) (tag : String) :
      IO (Except String Tropical.Plan.FlatPlan) := do
    let tmp := s!"/tmp/tropicaltest-fold-{tag}.json"
    IO.FS.writeFile tmp (mkPatch k unrolled).compress
    match ← compilePatch tmp .fused with
    | .error e => pure (.error e)
    | .ok planJson =>
      match Lean.Json.parse planJson with
      | .error e => pure (.error s!"parse: {e}")
      | .ok j => pure ((Tropical.Plan.FlatPlan.ofWire j).mapError (s!"ofWire: {·}"))
  match ← compileAt 16 false "f16", ← compileAt 16 true "u16" with
  | .ok fp, .ok up =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
    | .ok fS, .ok uS =>
      let n := min fS.size uS.size
      let mut bitDiff := 0
      for i in [0:n] do
        if fS[i]! != uS[i]! then bitDiff := bitDiff + 1
      let nonzero := fS.any (· != 0.0)
      match ← compileAt 8 false "f8", ← compileAt 64 false "f64" with
      | .ok f8, .ok f64 =>
        let d := planInstrCount f64 - planInstrCount f8
        let shrinks := decide (planInstrCount fp < planInstrCount up)
        let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
        IO.println s!"        surface fold Σₖ ampₖ·½ through raise→elab→strata→emit, 16 elements (loop-everything={looping}):"
        IO.println s!"        result   bit-differing {bitDiff}/{n} vs hand-unrolled · nonzero={nonzero}"
        IO.println s!"        payoff   plan-instrs: fold(16)={planInstrCount fp} unrolled(16)={planInstrCount up} · fold(8)={planInstrCount f8} fold(64)={planInstrCount f64} (Δ={d})"
        if looping then
          warnBenchConst "banks-fold-trunk" "fold plan-instrs (any K)" 8 (planInstrCount fp)
          if bitDiff == 0 && nonzero && shrinks && d ≤ 2 then
            IO.println s!"  PASS  banks-fold-trunk  surface fold banks: byte-equal to unroll, plan FLAT in K (Δ={d} ≤ 2, 8→64)"; pure true
          else
            IO.println s!"  FAIL  banks-fold-trunk  bitDiff={bitDiff} nonzero={nonzero} shrinks={shrinks} Δ={d}"; pure false
        else
          -- escape hatch: the fold must genuinely revert to unrolling
          if bitDiff == 0 && nonzero && !shrinks && d > 2 then
            IO.println s!"  PASS  banks-fold-trunk  escape hatch reverts: fold unrolls (Δ={d} grows), byte-equal"; pure true
          else
            IO.println s!"  FAIL  banks-fold-trunk  (unroll mode) bitDiff={bitDiff} nonzero={nonzero} shrinks={shrinks} Δ={d}"; pure false
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  banks-fold-trunk  scaling compile: {firstLine e}"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-fold-trunk  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-fold-trunk  compile: {firstLine e}"; pure false

/-- Wrap a single output expression in the minimal one-instance
    `tropical_program_2` patch the fold gates probe with (`p.out = expr`).
    `typeDefs` (optional) rides the inner program's ports — the tag-fold
    bail case needs a payload-less sum in scope. -/
private def foldProbePatchJson (expr : Lean.Json)
    (typeDefs : Array Lean.Json := #[]) : Lean.Json :=
  let ports :=
    if typeDefs.isEmpty then
      Lean.Json.mkObj [("inputs", Lean.Json.arr #[]),
        ("outputs", Lean.Json.arr #[Lean.Json.str "out"])]
    else
      Lean.Json.mkObj [("inputs", Lean.Json.arr #[]),
        ("outputs", Lean.Json.arr #[Lean.Json.str "out"]),
        ("type_defs", Lean.Json.arr typeDefs)]
  let inner := Lean.Json.mkObj [
    ("name", Lean.Json.str "FoldProbe"),
    ("ports", ports),
    ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
      ("decls", Lean.Json.arr #[]),
      ("assigns", Lean.Json.arr #[Lean.Json.mkObj [
        ("op", Lean.Json.str "outputAssign"), ("name", Lean.Json.str "out"),
        ("expr", expr)]])])]
  Lean.Json.mkObj [
    ("schema", Lean.Json.str "tropical_program_2"),
    ("name", Lean.Json.str "fold_probe"),
    ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
      ("decls", Lean.Json.arr #[
        Lean.Json.mkObj [("op", Lean.Json.str "programDecl"),
          ("name", Lean.Json.str "FoldProbe"), ("program", inner)],
        Lean.Json.mkObj [("op", Lean.Json.str "instanceDecl"),
          ("name", Lean.Json.str "p"), ("program", Lean.Json.str "FoldProbe"),
          ("inputs", Lean.Json.mkObj [])]]),
      ("assigns", Lean.Json.arr #[])]),
    ("audio_outputs", Lean.Json.arr #[Lean.Json.mkObj [
      ("instance", Lean.Json.str "p"), ("output", Lean.Json.str "out")]])]

/-- Compile a fold-probe expression through the FULL front door
    (raise → elaborate → strata → emit) and parse the resulting plan. -/
private def compileFoldProbe (expr : Lean.Json) (tag : String)
    (typeDefs : Array Lean.Json := #[]) : IO (Except String Tropical.Plan.FlatPlan) := do
  let tmp := s!"/tmp/tropicaltest-columnize-{tag}.json"
  IO.FS.writeFile tmp (foldProbePatchJson expr typeDefs).compress
  match ← compilePatch tmp .fused with
  | .error e => pure (.error e)
  | .ok planJson =>
    match Lean.Json.parse planJson with
    | .error e => pure (.error s!"parse: {e}")
    | .ok j => pure ((Tropical.Plan.FlatPlan.ofWire j).mapError (s!"ofWire: {·}"))

-- Shared JSON expression builders for the columnize gates.
private def cgJn (m : Nat) (e : Nat) : Lean.Json := Lean.Json.num ⟨Int.ofNat m, e⟩
private def cgA (i : Nat) : Lean.Json := cgJn (31 + 7 * i) 2   -- aᵢ = 0.31 + 0.07·i
private def cgB (i : Nat) : Lean.Json := cgJn (11 + 5 * i) 2   -- bᵢ = 0.11 + 0.05·i
private def cgAdd (a b : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "add"), ("args", Lean.Json.arr #[a, b])]
private def cgMulHalf (a : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "mul"), ("args", Lean.Json.arr #[a, cgJn 5 1])]
private def cgIndex (a b : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "index"), ("args", Lean.Json.arr #[a, b])]
private def cgBinding (n : String) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "binding"), ("name", Lean.Json.str n)]
/-- One tuple contribution: `a·½ + b`. -/
private def cgTerm (a b : Lean.Json) : Lean.Json := cgAdd (cgMulHalf a) b
private def cgFold (over : Lean.Json) (body : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("op", Lean.Json.str "fold"), ("over", over),
    ("init", cgJn 0 0), ("acc_var", Lean.Json.str "acc"), ("elem_var", Lean.Json.str "e"),
    ("body", body)]

/-- THE COLUMNIZE gate (columnize-over-shapes). A surface-language summing fold
    over TUPLE elements — Σ (aᵢ·½ + bᵢ) over [[a₀,b₀],…] through the FULL front
    door — de-structures into per-position coefficient columns (the AoS→SoA iso
    `Array (A×B) ≅ Array A × Array B`, done generically by `tryBankFoldE`) and
    banks as ONE multi-table reduction: byte-equal to the hand-unrolled add
    chain, exactly one `ReduceBegin` region with n=2 column `Pack`s, and the
    plan FLAT in element count (HARD — a growth regression means banking is
    broken). Under `TROPICAL_BANKS_UNROLL` the fold genuinely unrolls
    (0 regions) and still matches bit-exact. -/
private def runBanksColumnize (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let foldExpr (k : Nat) : Lean.Json :=
    let pairs := (Array.range k).map fun i => Lean.Json.arr #[cgA i, cgB i]
    cgFold (Lean.Json.arr pairs)
      (cgAdd (cgBinding "acc")
        (cgTerm (cgIndex (cgBinding "e") (cgJn 0 0)) (cgIndex (cgBinding "e") (cgJn 1 0))))
  let unrollExpr (k : Nat) : Lean.Json :=
    (Array.range k).foldl (fun acc i => cgAdd acc (cgTerm (cgA i) (cgB i))) (cgJn 0 0)
  match ← compileFoldProbe (foldExpr 8) "f8", ← compileFoldProbe (unrollExpr 8) "u8" with
  | .ok fp, .ok up =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
    | .ok fS, .ok uS =>
      let n := min fS.size uS.size
      let mut bitDiff := 0
      for i in [0:n] do
        if fS[i]! != uS[i]! then bitDiff := bitDiff + 1
      let nonzero := fS.any (· != 0.0)
      match ← compileFoldProbe (foldExpr 64) "f64" with
      | .ok f64 =>
        let d := planInstrCount f64 - planInstrCount fp
        let regions := planTagCount "ReduceBegin" fp
        let packs := planTagCount "Pack" fp
        let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
        IO.println s!"        surface fold Σₖ (aₖ·½ + bₖ) over [[a₀,b₀],…] (K=8), full front door (loop-everything={looping}):"
        IO.println s!"        result   bit-differing {bitDiff}/{n} vs hand-unrolled · nonzero={nonzero} · regions={regions} · column-Packs={packs}"
        IO.println s!"        payoff   plan-instrs: fold(8)={planInstrCount fp} unrolled(8)={planInstrCount up} fold(64)={planInstrCount f64} (Δ={d})"
        if looping then
          warnBenchConst "banks-columnize" "tuple-fold plan-instrs (any K)" 11 (planInstrCount fp)
          if bitDiff == 0 && nonzero && regions == 1 && packs == 2 && d ≤ 2 then
            IO.println s!"  PASS  banks-columnize  tuple fold banks as SoA: 1 region × 2 columns, byte-equal to unroll, plan FLAT in K (Δ={d} ≤ 2, 8→64)"; pure true
          else
            IO.println s!"  FAIL  banks-columnize  bitDiff={bitDiff} nonzero={nonzero} regions={regions} packs={packs} Δ={d}"; pure false
        else
          if bitDiff == 0 && nonzero && regions == 0 && d > 2 then
            IO.println s!"  PASS  banks-columnize  escape hatch reverts: tuple fold unrolls (0 regions, Δ={d} grows), byte-equal"; pure true
          else
            IO.println s!"  FAIL  banks-columnize  (unroll mode) bitDiff={bitDiff} nonzero={nonzero} regions={regions} Δ={d}"; pure false
      | .error e =>
        IO.println s!"  FAIL  banks-columnize  scaling compile: {firstLine e}"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-columnize  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  banks-columnize  compile: {firstLine e}"; pure false

/-- THE COLUMNIZE BAIL-OUT gate. The shapes `tryBankFoldE` refuses must still
    compile CORRECTLY via unrolling — never crash, never mis-bank:
    - RAGGED arities (a 2-tuple next to a 3-tuple) → unroll, byte-equal to the
      hand-written chain, 0 regions in BOTH flag states;
    - a NON-LITERAL index into the tuple element (`e[sampleIndex mod 2]`) → the
      symbolic tuple survives lowering, the residual guard bails, unroll — the
      alternating output pins that the dynamic index is genuinely live;
    - a fold over PAYLOAD-LESS TAGS: sum elements cannot reach ArrayLower as
      tags at all — SumLower rewrites them to scalar variant literals first —
      so the fold BANKS AS SCALARS (1 region when looping), today's behavior,
      asserted here so the `.tag` claim stays pinned. -/
private def runBanksColumnizeBail (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let looping := Tropical.Ir.Strata.ArrayLower.banksLoopEnabled
  let check (name : String) (foldE unrollE : Lean.Json)
      (typeDefs : Array Lean.Json) (wantRegions : Nat) : IO (Option String) := do
    match ← compileFoldProbe foldE s!"bail-{name}-f" typeDefs,
          ← compileFoldProbe unrollE s!"bail-{name}-u" typeDefs with
    | .ok fp, .ok up =>
      match ← renderPlanSamples fp 2048, ← renderPlanSamples up 2048 with
      | .ok fS, .ok uS =>
        let n := min fS.size uS.size
        let mut bitDiff := 0
        for i in [0:n] do
          if fS[i]! != uS[i]! then bitDiff := bitDiff + 1
        let nonzero := fS.any (· != 0.0)
        let regions := planTagCount "ReduceBegin" fp
        IO.println s!"        bail[{name}]  bit-differing {bitDiff}/{n} · nonzero={nonzero} · regions={regions} (want {wantRegions})"
        if bitDiff == 0 && nonzero && regions == wantRegions then pure none
        else pure (some s!"{name}: bitDiff={bitDiff} nonzero={nonzero} regions={regions} want={wantRegions}")
      | .error e, _ | _, .error e => pure (some s!"{name} render: {firstLine e}")
    | .error e, _ | _, .error e => pure (some s!"{name} compile: {firstLine e}")
  -- (a) ragged arities: [[a₀,b₀],[a₁,b₁,0.99]] — mixed 2-/3-tuples never bank.
  let raggedFold := cgFold
    (Lean.Json.arr #[Lean.Json.arr #[cgA 0, cgB 0],
                     Lean.Json.arr #[cgA 1, cgB 1, cgJn 99 2]])
    (cgAdd (cgBinding "acc")
      (cgTerm (cgIndex (cgBinding "e") (cgJn 0 0)) (cgIndex (cgBinding "e") (cgJn 1 0))))
  let raggedUnroll :=
    cgAdd (cgAdd (cgJn 0 0) (cgTerm (cgA 0) (cgB 0))) (cgTerm (cgA 1) (cgB 1))
  -- (b) non-literal index: e[sampleIndex mod 2] — the projection cannot fold,
  --     the symbolic tuple survives, the residual guard unrolls.
  let sampIdx := Lean.Json.mkObj [("op", Lean.Json.str "sampleIndex")]
  let dynIdx := Lean.Json.mkObj [("op", Lean.Json.str "mod"),
    ("args", Lean.Json.arr #[sampIdx, cgJn 2 0])]
  let dynFold := cgFold
    (Lean.Json.arr #[Lean.Json.arr #[cgA 0, cgB 0], Lean.Json.arr #[cgA 1, cgB 1]])
    (cgAdd (cgBinding "acc") (cgIndex (cgBinding "e") dynIdx))
  let dynUnroll :=
    cgAdd (cgAdd (cgJn 0 0) (cgIndex (Lean.Json.arr #[cgA 0, cgB 0]) dynIdx))
      (cgIndex (Lean.Json.arr #[cgA 1, cgB 1]) dynIdx)
  -- (c) payload-less tags: SumLower rewrites them to variant literals BEFORE
  --     ArrayLower, so the fold banks as scalars — `.tag` never reaches the
  --     shape check as an element.
  let flagDefs : Array Lean.Json := #[Lean.Json.mkObj [
    ("kind", Lean.Json.str "sum"), ("name", Lean.Json.str "Flag"),
    ("variants", Lean.Json.arr #[
      Lean.Json.mkObj [("name", Lean.Json.str "A"), ("payload", Lean.Json.arr #[])],
      Lean.Json.mkObj [("name", Lean.Json.str "B"), ("payload", Lean.Json.arr #[])],
      Lean.Json.mkObj [("name", Lean.Json.str "C"), ("payload", Lean.Json.arr #[])]])]]
  let tagJ (v : String) : Lean.Json :=
    Lean.Json.mkObj [("op", Lean.Json.str "tag"), ("variant", Lean.Json.str v)]
  let tagFold := cgFold
    (Lean.Json.arr #[tagJ "A", tagJ "C", tagJ "B"])
    (cgAdd (cgBinding "acc") (cgMulHalf (cgBinding "e")))
  let tagUnroll :=   -- variant indices 0, 2, 1 in fold order
    cgAdd (cgAdd (cgAdd (cgJn 0 0) (cgMulHalf (cgJn 0 0)))
      (cgMulHalf (cgJn 2 0))) (cgMulHalf (cgJn 1 0))
  let mut fails : Array String := #[]
  if let some f ← check "ragged" raggedFold raggedUnroll #[] 0 then fails := fails.push f
  if let some f ← check "dyn-index" dynFold dynUnroll #[] 0 then fails := fails.push f
  if let some f ← check "tags" tagFold tagUnroll flagDefs (if looping then 1 else 0) then
    fails := fails.push f
  if fails.isEmpty then
    let tagWord := if looping then "banks as SCALARS (1 region)" else "unrolls with the flag off (0 regions)"
    IO.println s!"  PASS  banks-columnize-bail  ragged + dynamic-index unroll byte-equal (0 regions); tag fold reaches ArrayLower as variant literals post-SumLower and {tagWord}"
    pure true
  else
    IO.println s!"  FAIL  banks-columnize-bail  {String.intercalate " · " fails.toList}"
    pure false

/-- THE MODAL DEGREE gate. A degree-1 mode `amp·d·e^{−σd}` (a repeated pole — the
    resonance "swell") rendered by the engine must match `sinkGain·d·e^{−σd}` to
    minimax tolerance (an absolute oracle, validating the new `d^deg` factor), and
    must RISE to a peak at d≈1/σ before decaying — the τ·e signature a simple pole
    (monotone decay) cannot produce. -/
private def runModalDegree (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    { sigma := Tropical.EmitArrow.lit 25, omega := Tropical.EmitArrow.lit 0,
      cre := Tropical.EmitArrow.lit 1, deg := 1 }]
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

/-- THE LONG-τ gate. Time-translation exactness at astronomical clock offsets:
    the SAME bank struck K samples later, read K samples later, must be BYTE-
    IDENTICAL to the bank at the origin (K = 2³⁰ samples ≈ 6.8 hours at 44.1k).
    Both clocks are FRACTIONAL-sample — the production scrub form
    `M(n) = toInt(velocity·2³²)·n` and a sub-sample offset — because that is
    where the old float path actually rounded: at whole samples
    `toFloat((2³⁰+s)·2³²)` has ~31 significant bits and was accidentally exact,
    but a fractional clock plus the 2³⁰-sample shift needs >53 bits, so
    `toFloat` rounds and the unreduced `ω·dSec` walks off the phase — precision
    decayed with τ exactly when the clock was warped. On the integer relative
    clock (`relClockQ` + `modePhaseW`) `clkRel` is the same i64 on both sides
    at ANY low bits, so every downstream op sees identical bytes. Energy floors
    keep silent agreement from passing. -/
private def runLongTauModal (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 220) (Tropical.EmitArrow.lit 30 1) (Tropical.EmitArrow.lit 6 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 330) (Tropical.EmitArrow.lit 40 1) (Tropical.EmitArrow.lit 4 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 440) (Tropical.EmitArrow.lit 55 1) (Tropical.EmitArrow.lit 3 1)]
  let K : Int := 1073741824                    -- 2³⁰ samples
  let Kq : Int := K * 4294967296               -- the same shift as a Q32.32 clock offset
  let mkPair (tag : String) (clk : Tropical.EmitArrow.Clock) :
      Except String Tropical.Plan.FlatPlan × Except String Tropical.Plan.FlatPlan :=
    (buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier s!"modal_lt_{tag}_base"
        (Tropical.EmitArrow.modalBankSig modes clk (Tropical.EmitArrow.lit 200)) arena)),
     buildAndFinish (.ok (Tropical.EmitArrow.buildExprCarrier s!"modal_lt_{tag}_far"
        (Tropical.EmitArrow.modalBankSig modes
          (Tropical.EmitArrow.add clk (Tropical.EmitArrow.litI Kq))
          (Tropical.EmitArrow.lit (K + 200)) ) arena)))
  let check (tag : String) (pair : Except String Tropical.Plan.FlatPlan × Except String Tropical.Plan.FlatPlan) :
      IO (Option (Nat × Float)) := do
    match pair with
    | (.ok bp, .ok fp) =>
      match ← renderPlanSamples bp 1024, ← renderPlanSamples fp 1024 with
      | .ok base, .ok far =>
        let n := min base.size far.size
        let mut bitDiff := 0
        for i in [0:n] do
          if base[i]! != far[i]! then bitDiff := bitDiff + 1
        let mut energy : Float := 0.0
        for i in [200:n] do energy := energy + base[i]! * base[i]!
        pure (some (bitDiff, energy))
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  modal-longtau  render ({tag}): {firstLine e}"; pure none
    | (.error e, _) | (_, .error e) =>
      IO.println s!"  FAIL  modal-longtau  build ({tag}): {firstLine e}"; pure none
  -- the master-clock scrub form: M(n) = toInt(velocity·2³²)·n at velocity 1.001
  -- (toInt(1.001·2³²) = 4299262263) — every sample has populated low bits. The
  -- literals ride `litI`: a bare `lit` would float-promote the clock arithmetic
  -- and round the very bits this gate exists to protect.
  let velClk := Tropical.EmitArrow.mul
    (Tropical.EmitArrow.rshift Tropical.EmitArrow.clockLit (Tropical.EmitArrow.lit 32))
    (Tropical.EmitArrow.litI 4299262263)
  -- a bare sub-sample offset: one 2⁻³² unit off the whole-sample grid.
  let subClk := Tropical.EmitArrow.add Tropical.EmitArrow.clockLit (Tropical.EmitArrow.litI 1)
  match ← check "vel" (mkPair "vel" velClk), ← check "sub" (mkPair "sub" subClk) with
  | some (d1, e1), some (d2, e2) =>
    IO.println s!"        bank @ origin vs struck+read 2³⁰ samples later (≈6.8h), fractional clocks, 1024 samples:"
    IO.println s!"        result   velocity-1.001 clock: bit-differing {d1}/1024 (E={e1})  ·  sub-sample offset: {d2}/1024 (E={e2})"
    if d1 == 0 && d2 == 0 && e1 > 1e-6 && e2 > 1e-6 then
      IO.println s!"  PASS  modal-longtau  time-translation byte-exact at τ+2³⁰ samples on fractional (scrub-form) clocks"; pure true
    else
      IO.println s!"  FAIL  modal-longtau  bitDiff vel={d1} sub={d2} (want 0) energy vel={e1} sub={e2} (>1e-6)"; pure false
  | _, _ => pure false

/-- THE REVERSE-REVERB gate (the moat). The modal bank read through a reversing
    warp φ(c) = 2·C·2³² − c (reflect scene time around sample C=1024) must equal
    the FORWARD bank time-mirrored: rev[i] ≡ fwd[2C−i], bit-for-bit. This is
    zero-latency reverse reverb — a stateless closed form addressed at negative
    velocity, impossible on a streaming delay line. The warp threads through the
    modal `arrUn … (.clk c)` via the same `.warp` a master-clock scrub uses. -/
private def runModalReverse (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array Tropical.EmitArrow.ModalMode := #[
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 220) (Tropical.EmitArrow.lit 30 1) (Tropical.EmitArrow.lit 6 1),
    Tropical.EmitArrow.ModalMode.hz (Tropical.EmitArrow.lit 330) (Tropical.EmitArrow.lit 40 1) (Tropical.EmitArrow.lit 4 1)]
  let anchor := Tropical.EmitArrow.lit 200
  let twoC : Int := 2048 * 4294967296          -- reflect around sample C = 1024
  let revφ : Tropical.EmitArrow.Clock → Tropical.EmitArrow.Clock :=
    fun c => Tropical.EmitArrow.sub (Tropical.EmitArrow.lit twoC) c
  match buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankArrow "modal_fwd" modes anchor arena)),
        buildAndFinish (.ok (Tropical.EmitArrow.buildModalBankWarped "modal_rev" modes anchor revφ arena)) with
  | .ok fp, .ok rp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples rp 2048 with
    | .ok fwd, .ok rev =>
      let n := min fwd.size rev.size
      let mut bitDiff := 0
      let mut differsFwd := 0        -- rev ≠ fwd somewhere (warp is non-trivial)
      let mut revEnergy : Float := 0.0
      for i in [1:n] do
        if rev[i]! != fwd[2048 - i]! then bitDiff := bitDiff + 1
        if rev[i]! != fwd[i]! then differsFwd := differsFwd + 1
        revEnergy := revEnergy + rev[i]! * rev[i]!
      IO.println s!"        modal bank forward vs reversed (φ reflects scene time around sample 1024):"
      IO.println s!"        result   rev[i] vs fwd[2048−i]: bit-differing {bitDiff}/{n}  ·  rev≠fwd at {differsFwd} samples  ·  rev energy={revEnergy}"
      if bitDiff == 0 && differsFwd > 0 && revEnergy > 1e-6 then
        IO.println s!"  PASS  modal-reverse  reversed reading ≡ forward time-mirrored, bit-exact — zero-latency reverse reverb ({n} samples)"; pure true
      else
        IO.println s!"  FAIL  modal-reverse  bitDiff={bitDiff} differsFwd={differsFwd} revEnergy={revEnergy}"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  modal-reverse  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  modal-reverse  build: {firstLine e}"; pure false

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

/-- THE DIRECTION gate. `dir` crossfades the tail's time-direction and must, above
    all, STAY AUDIBLE across its range (the pole-rotation version silently collapsed
    the interior because ω≫σ threw the frequency into the damping). (A) `dir=0`
    reduces bit-for-bit to the forward bank; (B) `dir=1` is that bank TIME-MIRRORED
    (rev[i] ≡ fwd[2C−i]) — genuine reverse reverb, no warp; (C) `dir=0.5` stays finite
    AND carries real energy (a substantial fraction of the forward bank's), i.e. it is
    audible, not a collapsed transient — the property the rotation version lacked. -/
private def runModalDirection (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 330) (lit 40 1) (lit 4 1)]
  let anchor := lit 1024                        -- mid of 2048 ⇒ 2C = 2048
  let fwdB := buildModalBankArrow "dir_fwd" modes anchor arena
  let idB  := buildModalBankDir "dir_id"  modes anchor (lit 0) arena        -- forward
  let revB := buildModalBankDir "dir_rev" modes anchor (lit 1) arena        -- reverse
  let midB := buildModalBankDir "dir_mid" modes anchor (litF 0.5) arena     -- crossfade
  match buildAndFinish (.ok fwdB), buildAndFinish (.ok idB),
        buildAndFinish (.ok revB), buildAndFinish (.ok midB) with
  | .ok fp, .ok ip, .ok rp, .ok mp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples ip 2048,
          ← renderPlanSamples rp 2048, ← renderPlanSamples mp 2048 with
    | .ok fwd, .ok idv, .ok rev, .ok mid =>
      let n := 2048
      let mut idDiff := 0
      let mut fwdE : Float := 0.0
      for i in [0:n] do
        if idv[i]! != fwd[i]! then idDiff := idDiff + 1
        fwdE := fwdE + fwd[i]! * fwd[i]!
      let mut revDiff := 0
      let mut revDiffersFwd := 0
      for i in [1:n] do
        if rev[i]! != fwd[2048 - i]! then revDiff := revDiff + 1
        if rev[i]! != fwd[i]! then revDiffersFwd := revDiffersFwd + 1
      let mut midE : Float := 0.0
      let mut midFinite := true
      for i in [0:n] do
        let a := mid[i]!.abs
        if !a.isFinite then midFinite := false
        midE := midE + a * a
      IO.println s!"        direction crossfade (forward↔reverse, σ/ω fixed):"
      IO.println s!"        (A) dir=0 vs fwd bitDiff={idDiff}  ·  (B) dir=1 mirror bitDiff={revDiff} (differs-fwd @{revDiffersFwd})"
      IO.println s!"        (C) dir=0.5 finite={midFinite} · E={midE} vs forward E={fwdE} (AUDIBLE ⇒ E ≫ 0)"
      let aOk := idDiff == 0
      let bOk := revDiff == 0 && revDiffersFwd > 0
      let cOk := midFinite && fwdE > 1e-6 && midE > 0.1 * fwdE
      if aOk && bOk && cOk then
        IO.println s!"  PASS  modal-direction  dir=0 forward (bit-exact) · dir=1 reverse (mirror bit-exact) · dir=0.5 AUDIBLE (E={midE}, {midE/fwdE} of fwd)"; pure true
      else
        IO.println s!"  FAIL  modal-direction  A={aOk} B={bOk} C={cOk} (idDiff={idDiff} revDiff={revDiff} midE={midE} fwdE={fwdE})"; pure false
    | _, _, _, _ => IO.println s!"  FAIL  modal-direction  render error"; pure false
  | _, _, _, _ => IO.println s!"  FAIL  modal-direction  build error"; pure false

/-- THE SWAY gate. Decay sway modulates each mode's damping by `1 + depth·sin(2π·
    rate·t)` on the ENVELOPE clock only. (S1) at depth 0 it is bit-for-bit the
    un-swayed bank (the LFO term folds to ×1); (S2) at depth>0 the tail differs (its
    decay breathes) yet stays causal (silent pre-strike) and bounded. Pitch is
    untouched by construction (the oscillator reads the plain `dSec`); the LFO rides
    the same clock leaf as the bank, so a master scrub reverses it coherently. -/
private def runModalSway (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 330) (lit 40 1) (lit 4 1)]
  let anchor := lit 200
  let noSway := buildModalBankDir "sway_no" modes anchor (lit 0) arena
  let sway0  := buildModalBankDir "sway_0"  modes anchor (lit 0) arena (some (lit 0, lit 3 1))
  let swayD  := buildModalBankDir "sway_d"  modes anchor (lit 0) arena (some (lit 5 1, lit 20 1))
  match buildAndFinish (.ok noSway), buildAndFinish (.ok sway0), buildAndFinish (.ok swayD) with
  | .ok np, .ok zp, .ok dp =>
    match ← renderPlanSamples np 2048, ← renderPlanSamples zp 2048, ← renderPlanSamples dp 2048 with
    | .ok nos, .ok zos, .ok dos =>
      let n := 2048
      let mut z0Diff := 0
      for i in [0:n] do if zos[i]! != nos[i]! then z0Diff := z0Diff + 1
      let mut modDiff := 0
      let mut preMax : Float := 0.0
      let mut dFinite := true
      let mut dPeak : Float := 0.0
      for i in [0:n] do
        if dos[i]! != nos[i]! then modDiff := modDiff + 1
        let a := dos[i]!.abs
        if !a.isFinite then dFinite := false
        if a > dPeak then dPeak := a
      for i in [0:201] do
        let a := dos[i]!.abs
        if a > preMax then preMax := a
      IO.println s!"        decay sway (σ·(1+depth·sin 2πrt) on the envelope clock only):"
      IO.println s!"        (S1) depth 0 vs no-sway bitDiff={z0Diff}  ·  (S2) depth>0 differs @{modDiff}/{n}, pre-strike |max|={preMax}, peak={dPeak}, finite={dFinite}"
      let s1 := z0Diff == 0
      let s2 := modDiff > 100 && preMax == 0.0 && dFinite && dPeak > 1e-4 && dPeak < 1e3
      if s1 && s2 then
        IO.println s!"  PASS  modal-sway  depth 0 ≡ un-swayed (bit-exact) · depth>0 breathes the decay, causal & bounded"; pure true
      else
        IO.println s!"  FAIL  modal-sway  S1={s1} (bitDiff {z0Diff}) S2={s2} (modDiff {modDiff} preMax {preMax} peak {dPeak} finite {dFinite})"; pure false
    | _, _, _ => IO.println s!"  FAIL  modal-sway  render error"; pure false
  | _, _, _ => IO.println s!"  FAIL  modal-sway  build error"; pure false

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

/-- THE SYMBOLIC RESIDUE gate. The residue calculus emitted as `Expr` couplings
    (`residueComposeE`, so poles/coeffs can be live slots) must, on LITERAL poles,
    fold to the same bank as the validated Float `residueCompose`. Same voice ⋙
    reverb built both ways renders equal (differing only by litF input-vs-output
    rounding). This is what makes modal params live without changing the math. -/
private def runResidueSymbolic (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let voiceF : Array (Cplx × Cplx) := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩)]
  let reverbF : Array (Cplx × Cplx) := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)]
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let anchor := lit 200
  match buildAndFinish (.ok (buildModalReverb "rv_baked" voiceF reverbF anchor arena)),
        buildAndFinish (.ok (buildModalReverbSym "rv_sym" (voiceF.map toMode) (reverbF.map toMode) anchor arena)) with
  | .ok bp, .ok sp =>
    match ← renderPlanSamples bp 4096, ← renderPlanSamples sp 4096 with
    | .ok bs, .ok ss =>
      let n := min bs.size ss.size
      let mut maxAbs : Float := 0.0
      let mut energy : Float := 0.0
      for i in [0:n] do
        let d := (bs[i]! - ss[i]!).abs
        if d > maxAbs then maxAbs := d
        energy := energy + bs[i]! * bs[i]!
      -- Two builds of the same 10-mode bank whose weights may differ by an ulp
      -- pre-landing (litF round-trip), so each mode may jump one Q4.28 quantum
      -- (design/fixed-carrier.md) × the 0.05 sink gain, 2× slack.
      let bound := 10.0 * 3.7252903e-9 * 0.05 * 2.0
      IO.println s!"        Expr-residue (literal poles) vs Float-baked residue, voice(2)⋙reverb(4):"
      IO.println s!"        result   max|Δ|={maxAbs * 1e9}e-9  ·  quantum bound={bound * 1e9}e-9"
      if maxAbs < bound && energy > 1e-9 then
        IO.println s!"  PASS  symbolic-residue  Expr couplings fold to the validated Float residue within the Q4.28 landing quantum (max|Δ| {maxAbs * 1e9}e-9) — live-capable, same math"; pure true
      else
        IO.println s!"  FAIL  symbolic-residue  max|Δ|={maxAbs * 1e9}e-9 (bound {bound * 1e9}e-9) energy={energy}"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  symbolic-residue  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  symbolic-residue  build: {firstLine e}"; pure false

/-- THE COLLECTED RESIDUE gate. `residueComposeEC` (m+n modes: pole union with
    cross-weighted residues) must render pointwise-equal to the uncollected
    `residueComposeE` (m+m·n modes) — they are the same partial-fraction expansion
    with the per-pair ringing amps summed per reverb pole, so equality is algebraic
    and the tolerance only absorbs the DOCUMENTED datapath quantization: each mode
    lands its envelope×weight once in Q4.28 (design/fixed-carrier.md), so the two
    structures truncate independently and may differ by up to (m+n + m+m·n)
    quanta·sinkGain absolutely — the bound is quantum-tied, not relative. Also
    asserts the collection is structural: m+n modes out, not m+m·n. This is what
    makes `voice ⋙ reverb` affordable as the DEFAULT lowering — a factor m fewer
    transcendentals — which is in turn what lets a reverb keep its source's
    spectrum (and live pitch knob) instead of discarding them. -/
private def runResidueCollected (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let tp := 6.283185307179586
  let toMode := fun (pa : Cplx × Cplx) =>
    ({ sigma := litF (-pa.1.re), omega := litF pa.1.im,
       cre := litF pa.2.re, cim := litF pa.2.im } : ModalMode)
  let voice : Array ModalMode := #[
    (⟨-2.0, tp * 220.0⟩, ⟨1.0, 0.0⟩),
    (⟨-2.5, tp * 330.0⟩, ⟨0.6, 0.0⟩),
    (⟨-3.5, tp * 440.0⟩, (⟨0.4, -0.2⟩ : Cplx))].map toMode
  let reverb : Array ModalMode := #[
    (⟨-3.0, tp * 180.0⟩, ⟨0.7, 0.2⟩),
    (⟨-4.0, tp * 260.0⟩, ⟨-0.5, 0.4⟩),
    (⟨-5.0, tp * 350.0⟩, ⟨0.3, -0.6⟩),
    (⟨-6.0, tp * 500.0⟩, ⟨0.4, 0.1⟩)].map toMode
  let nU := (residueComposeE voice reverb).size
  let nC := (residueComposeEC voice reverb).size
  let anchor := lit 200
  match buildAndFinish (.ok (buildModalReverbSym "rv_unc" voice reverb anchor arena)),
        buildAndFinish (.ok (buildModalReverbSymC "rv_col" voice reverb anchor arena)) with
  | .ok up, .ok cp =>
    match ← renderPlanSamples up 4096, ← renderPlanSamples cp 4096 with
    | .ok us, .ok cs =>
      let n := min us.size cs.size
      let mut maxAbs : Float := 0.0
      let mut energy : Float := 0.0
      for i in [0:n] do
        let d := (us[i]! - cs[i]!).abs
        if d > maxAbs then maxAbs := d
        energy := energy + us[i]! * us[i]!
      -- (nU + nC) independent Q4.28 weight landings × the 0.05 sink gain, with
      -- 2× slack for the poly/final-shift ulps riding along.
      let bound := (nU + nC).toFloat * 3.7252903e-9 * 0.05 * 2.0
      IO.println s!"        collected (m+n={nC}) vs uncollected (m+m·n={nU}), voice(3)⋙reverb(4):"
      IO.println s!"        result   max|Δ|={maxAbs * 1e9}e-9  ·  quantum bound={bound * 1e9}e-9"
      if maxAbs < bound && energy > 1e-9 && nC == 7 && nU == 15 then
        IO.println s!"  PASS  residue-collected  pole-union bank ≡ per-pair bank within the Q4.28 landing quantum (max|Δ| {maxAbs * 1e9}e-9 < {bound * 1e9}e-9); {nU}→{nC} modes — fusion affordable as the default"; pure true
      else
        IO.println s!"  FAIL  residue-collected  max|Δ|={maxAbs * 1e9}e-9 (bound {bound * 1e9}e-9) energy={energy} nC={nC} (want 7) nU={nU} (want 15)"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  residue-collected  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  residue-collected  build: {firstLine e}"; pure false

end ResidueGates

open Tropical.EmitArrow in
/-- THE MODAL PATCH gate (the session surface). A modal-island `PatchGraph`
    (`resonator → reverb → out`) lowered through `lowerModal` (residue in pole
    space) and realized at its boundary must render a real, causal, decaying
    signal — and, read through a reversing master clock, play the tail backward
    bit-for-bit. This is the whole seam end to end: a patch graph, not a builder. -/
private def runModalPatch (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let res : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 440) (lit 45 1) (lit 3 1),
    ModalMode.hz (lit 660) (lit 60 1) (lit 2 1)]
  let room : Array ModalMode := #[
    { sigma := lit 3, omega := mul twoPiE (lit 180), cre := lit 7 1, cim := lit 2 1 },
    { sigma := lit 4, omega := mul twoPiE (lit 300), cre := lit (-5) 1, cim := lit 4 1 },
    { sigma := lit 5, omega := mul twoPiE (lit 520), cre := lit 3 1, cim := lit (-6) 1 }]
  let anchor := lit 200
  let twoC : Int := 2048 * 4294967296
  let mkGraph := fun (clk : Clock) => ({
    nodes := #[
      { id := "res", node := .modalSource res anchor clk none },
      { id := "rev", node := .modalReverb "res" room none }],
    output := "rev" } : PatchGraph)
  let carrier := fun (name : String) (clk : Clock) => (do
    let term ← lowerGraph (mkGraph clk)
    let (out, _) := emitTerm (normalize term) {}
    .ok (buildExprCarrier name out arena) : Except String (Arena × ProgramIdx))
  let revClk : Clock := sub (lit twoC) clockLit
  match buildAndFinish (carrier "mp_fwd" clockLit),
        buildAndFinish (carrier "mp_rev" revClk) with
  | .ok fp, .ok rp =>
    match ← renderPlanSamples fp 2048, ← renderPlanSamples rp 2048 with
    | .ok fwd, .ok rev =>
      let n := min fwd.size rev.size
      let mut preMax : Float := 0.0
      for i in [0:201] do
        let a := fwd[i]!.abs
        if a > preMax then preMax := a
      let mut peak : Float := 0.0
      for i in [201:n] do
        let a := fwd[i]!.abs
        if a > peak then peak := a
      let mut eEarly : Float := 0.0
      let mut eLate : Float := 0.0
      for i in [201:900] do eEarly := eEarly + fwd[i]! * fwd[i]!
      for i in [1349:2048] do eLate := eLate + fwd[i]! * fwd[i]!
      let mut bitDiff := 0
      for i in [1:n] do
        if rev[i]! != fwd[2048 - i]! then bitDiff := bitDiff + 1
      IO.println s!"        patch: resonator(3) → reverb(3) → out, lowered from a PatchGraph:"
      IO.println s!"        result   pre-strike |max|={preMax} · peak={peak} · E[early]={eEarly} E[late]={eLate} · rev≡fwd-mirror bitDiff {bitDiff}/{n}"
      if preMax == 0.0 && peak > 1e-6 && eLate < eEarly && bitDiff == 0 then
        IO.println s!"  PASS  modal-patch  resonator→reverb→out compiles from a graph: causal, decaying, reverse-scrubs bit-exact"; pure true
      else
        IO.println s!"  FAIL  modal-patch  preMax={preMax} peak={peak} eEarly={eEarly} eLate={eLate} bitDiff={bitDiff}"; pure false
    | .error e, _ | _, .error e => IO.println s!"  FAIL  modal-patch  render: {firstLine e}"; pure false
  | .error e, _ | _, .error e => IO.println s!"  FAIL  modal-patch  build: {firstLine e}"; pure false

/-- THE MODAL LIVE gate (the payoff). A JSON patch `resonator(freq) → reverb → out`
    compiled through the real `compilePlanPure` — decode → lowerModal → symbolic
    residue → realize → strata → session compile → a JIT-loadable kernel — and its
    pole frequency/decay and the room rt60 resolve to LIVE module slots
    (`param:<id>.<knob>`), settable via `setSlot` with no relower. That the residue
    calculus is symbolic is exactly what keeps the poles live; `symbolic-residue`
    proves the couplings are the right functions of those slots. (This harness
    can't drive a session plan's DAC to audio — that's the Engine/bun path — so the
    audible sweep is left to those; here we prove it compiles and is live.) -/
private def runModalLive (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => IO.println s!"  FAIL  modal-live  json parse: {e}"; pure false
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => IO.println s!"  FAIL  modal-live  compile: {firstLine e}"; pure false
  | .ok (plan, _, stageBlocks) =>
    match plan.toWire, Tropical.Ir.EmitLlvm.emitKernel plan with
    | .ok _, .ok _ =>
      -- A slot that EXISTS but is never READ is a dead knob — exactly the
      -- reverb-discards-the-voice regression (the pitch knob accepted writes into
      -- a slot no instruction referenced). So presence is only half the gate: run
      -- two identical runtimes a block, move `res.freq` on ONE, and require the
      -- next blocks to diverge THROUGH the reverb. Identical second blocks =
      -- dead knob = FAIL. Under the stage-0 split this also gates the
      -- coefficient re-run: the amps live in the coefficient kernel, so the
      -- divergence only happens if set_slot re-runs it.
      let rt ← Tropical.Ffi.Runtime.new 2048
      Tropical.StagedLoad.loadTyped rt plan stageBlocks
      let rt2 ← Tropical.Ffi.Runtime.new 2048
      Tropical.StagedLoad.loadTyped rt2 plan stageBlocks
      let fIdx? ← rt.slotIndex? "param:res.freq"
      let dPresent := (← rt.slotIndex? "param:res.decay").isSome
      let rtPresent := (← rt.slotIndex? "param:rev.rt60").isSome
      rt.process
      rt2.process
      let b1a := decodeF64LE (← rt.outputBytes)
      let b1b := decodeF64LE (← rt2.outputBytes)
      if let some fIdx := fIdx? then rt.setSlot fIdx 440.0
      rt.process
      rt2.process
      let b2a := decodeF64LE (← rt.outputBytes)
      let b2b := decodeF64LE (← rt2.outputBytes)
      let mut sameB1 := true
      for i in [0:min b1a.size b1b.size] do
        if b1a[i]! != b1b[i]! then sameB1 := false
      let mut dE : Float := 0.0
      let mut e0 : Float := 0.0
      for i in [0:min b2a.size b2b.size] do
        let d := b2a[i]! - b2b[i]!
        dE := dE + d * d
        e0 := e0 + b2b[i]! * b2b[i]!
      let knobRead := dE > 1e-12 && e0 > 1e-12
      IO.println s!"        JSON resonator(freq,decay) → reverb(rt60) → out compiled via compilePlanPure:"
      IO.println s!"        result   JIT-loadable · slots: freq={fIdx?.isSome} decay={dPresent} rt60={rtPresent} · pre-move blocks identical={sameB1} · post-move ΔE/E={dE / (e0 + 1e-300)}"
      if fIdx?.isSome && dPresent && rtPresent && sameB1 && knobRead then
        IO.println s!"  PASS  modal-live  modal params are live slots AND the kernel reads them: moving res.freq moves the signal THROUGH the reverb (setSlot, no relower)"; pure true
      else
        IO.println s!"  FAIL  modal-live  freq={fIdx?.isSome} decay={dPresent} rt60={rtPresent} sameB1={sameB1} knobRead={knobRead} (ΔE={dE}) — a present-but-unread slot is a dead knob"; pure false
    | .error e, _ => IO.println s!"  FAIL  modal-live  toWire: {firstLine e}"; pure false
    | _, .error e => IO.println s!"  FAIL  modal-live  emitKernel: {firstLine e}"; pure false

/-- Count instructions matching `pred` across a plan's instance-function tree. -/
private partial def countInstrsFn (pred : Tropical.Plan.NInstr → Bool) :
    Tropical.Plan.InstanceFunction → Nat
  | f => (f.instructions.filter pred).size
         + f.children.foldl (fun acc c => acc + countInstrsFn pred c) 0

private def countInstrs (pred : Tropical.Plan.NInstr → Bool) (p : Tropical.Plan.FlatPlan) : Nat :=
  p.instanceFunctions.foldl (fun acc f => acc + countInstrsFn pred f) 0

/-- Array-dst fills (`Pack`/`SetElement` — coefficient columns). `sessionArray`
    I/O is excluded (still s1). -/
private def planArrayFills (p : Tropical.Plan.FlatPlan) : Nat :=
  countInstrs (fun i => match i.dst with | .array _ => true | _ => false) p

/-- Reduce regions (banked mode loops). -/
private def planReduces (p : Tropical.Plan.FlatPlan) : Nat :=
  countInstrs (fun i => i.tag == "ReduceBegin") p

-- ── Region-aware Stage0 (WS3a): an all-s0 reduce region hoists AS A UNIT ─────
section RegionHoist
open Tropical.Plan

/-- A reduce region whose WHOLE computation is coefficient-shaped: the table
    Packs from live param slots (never written in-plan → s0-external), the body
    indexes it at `loopIdx` and weights by (k+1) — no τ anywhere inside the
    region. An s1 consumer mixes the accumulator with τ so the render is
    per-sample. `dyn` adds the trip-count-as-data operand (`ReduceBegin`
    args[1] ← the `param:n` slot, default 2 < capacity 4, so the dynamic
    render differs from the static one — a real clamp path, not a no-op).
    Temps: acc=1, body 2..5, tail 6..8. Built synthetically (FlatPlan + hand
    stage blocks through `hoistTyped`, the reduce-coverage pattern): no
    surface graph reaches an s0 region yet — the modal banks all read τ. -/
private def regionS0PlanOf (dyn : Bool) : FlatPlan :=
  let countOp? : Option NOperand := if dyn then some (.slot 3 .float) else none
  let body : Array NInstr := #[
    instrPack 0 #[.slot 1 .float, .slot 2 .float, .slot 1 .float, .slot 2 .float],
    instrReduceBegin 1 (cF 0) 4 .float countOp?,
    instrIndex 2 #[.arrayReg 0, .loopIdx] .float,               -- v = table[k]
    instrScalar "Add" 3 #[.loopIdx, cI 1] .int,                 -- k+1
    instrScalar "ToFloat" 4 #[rgI 3] .float,
    instrScalar "Mul" 5 #[rgF 2, rgF 4] .float,                 -- v·(k+1)
    instrScalar "Add" 1 #[rgF 1, rgF 5] .float,
    instrReduceEnd 1 .float,
    instrScalar "ToFloat" 6 #[Tropical.Plan.opTick] .float,     -- the s1 consumer
    instrScalar "Mod" 7 #[rgF 6, cF 64] .float,
    instrScalar "Mul" 8 #[rgF 1, rgF 7] .float,
    instrWriteSlot 0 (rgF 8)]
  let inst := InstanceFunction.mk "root" "root" #[] body #[] 0 0 9 #[]
  { sampleRate := jn 44100, compilationMode := .fused,
    arraySlotNames := #["table"], registerCount := 9, arraySlotCount := 1,
    arraySlotSizes := #[4], instanceFunctions := #[inst],
    sinks := #[{ inputs := #[0], gain := jn 1, target := 0 }],
    sources := defaultSources, slotCount := 4,
    slotNames := #["out", "param:a", "param:b", "param:n"],
    slotDefaults := #[Lean.Json.num (jn 0), Lean.Json.num (jn 5 1),
      Lean.Json.num (jn 25 2), Lean.Json.num (jn 2)] }

/-- Hand stage blocks in the partitioner's shape (`collectBlocks`: preamble,
    body): everything in and around the region is s0 (exactly what the
    intern-time attribute derives with `loopIdx` stage-neutral), the τ tail s1. -/
private def regionS0Stages : Array (Array (Option Tropical.Ir.Stage)) :=
  let s0 : Option Tropical.Ir.Stage := some .s0
  let s1 : Option Tropical.Ir.Stage := some .s1
  #[#[], #[s0, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1, s1]]

/-- THE REGION-HOIST gate (region-aware Stage0, WS3a). A reduce region whose
    body is entirely s0 (param-slot-derived, no clock/tick dependence) plus an
    s1 consumer of its result: the typed split moves the WHOLE region
    (delimiters + body, with its table Pack) into the coefficient stream — the
    audio kernel contains ZERO regions and reads the sum via a `coef:` slot —
    and the render is BYTE-EXACT against the flow split (which never hoists
    regions: effectively the unsplit reference). Checked for the static region
    AND the dynamic-count region (args[1] ← a param slot). This also verifies
    the runtime claim end to end: the coefficient kernel containing a
    `ReduceBegin` region is emitted through the same `EmitLlvm`, JIT-compiled,
    and executed by `run_coeff` before buffer 0. -/
private def runBanksRegionHoist : IO Bool := do
  let check := fun (label : String) (dyn : Bool) => do
    let plan := regionS0PlanOf dyn
    match Tropical.Ir.Stage0.hoistTyped plan regionS0Stages with
    | .error e => IO.println s!"  FAIL  banks-region-hoist  {label} split: {firstLine e}"; pure false
    | .ok split =>
      let audioReduces := planReduces split.audio
      let coeffReduces := match split.coeff? with | some c => planReduces c | none => 0
      let coeffFills := match split.coeff? with | some c => planArrayFills c | none => 0
      let hasCoefSlot := split.audio.slotNames.any (· == "coef:0")
      let cols := split.audio.coeffArraySlots
      let typed ← renderTypedBytes plan regionS0Stages
      match ← renderIrBytes plan with
      | .error e => IO.println s!"  FAIL  banks-region-hoist  {label} flow render: {firstLine e}"; pure false
      | .ok flow =>
        let n := min typed.size flow.size
        let mut bitDiff := 0
        for i in [0:n] do
          if typed[i]! != flow[i]! then bitDiff := bitDiff + 1
        let mut energy : Float := 0.0
        for s in decodeF64LE typed do energy := energy + s * s
        IO.println (s!"        {label}: regions audio={audioReduces} coeff={coeffReduces} · "
          ++ s!"coeff fills={coeffFills} · coef:0 slot={hasCoefSlot} · columns={cols} · "
          ++ s!"typed≡flow bitDiff={bitDiff}/{n} · E={energy}")
        pure (audioReduces == 0 && coeffReduces == 1 && coeffFills == 1
          && hasCoefSlot && cols == #[0] && typed.size == flow.size
          && bitDiff == 0 && energy > 1e-6)
  let okS ← check "static count" false
  let okD ← check "dynamic count (slot, 2 < capacity 4)" true
  if okS && okD then
    IO.println ("  PASS  banks-region-hoist  all-s0 region moves AS A UNIT "
      ++ "(delimiters + body + table Pack) to the coeff kernel; audio is region-free, "
      ++ "reads the sum via coef:0; typed ≡ flow byte-exact, static AND dynamic count")
    pure true
  else
    IO.println s!"  FAIL  banks-region-hoist  static={okS} dynamic={okD}"
    pure false

end RegionHoist

/-- THE PER-ARRAY STAGING gate (banks-as-data blocker 3). `modal-live` proves the
    banked lowering still renders correctly under live knobs; this proves the
    PAYOFF structurally — with the banked lowering on, a live-param bank's
    coefficient columns (`Pack` fills) move OUT of the audio kernel and INTO the
    s0 coefficient kernel, and the audio kernel is left array-fill-free (its
    in-loop `Index` reads the shared, coeff-filled storage). Adapts to the flag:
    flag on ⇒ columns hoist; flag off ⇒ unrolled, no columns (no spurious hoist). -/
private def runBanksStaging (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- A bare resonator → out: a UNIFORM (deg-0) forward bank with live freq/decay,
  -- so with the flag on it banks (a reverb would compose to a possibly-ragged
  -- bank via residueComposeEC's deg-1 coincident poles — not the payoff we gate).
  let src := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => IO.println s!"  FAIL  banks-staging  json: {e}"; pure false
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => IO.println s!"  FAIL  banks-staging  compile: {firstLine e}"; pure false
  | .ok (plan, _, stageBlocks) =>
    match Tropical.Ir.Stage0.hoistTyped plan stageBlocks with
    | .error e => IO.println s!"  FAIL  banks-staging  split: {firstLine e}"; pure false
    | .ok split =>
      let flagOn := Tropical.EmitArrow.banksTableEnabled
      let coeffFills := match split.coeff? with | some c => planArrayFills c | none => 0
      let audioFills := planArrayFills split.audio
      let reduces := planReduces split.audio
      -- banks-region-stay: the resonator's region is CLOCK-DEPENDENT (the
      -- body reads τ), so the whole-region move (banks-region-hoist's
      -- positive) must NOT fire here — the region stays in the audio
      -- kernel, only its live columns hoist.
      let coeffReduces := match split.coeff? with | some c => planReduces c | none => 0
      -- Correctness: render the SAME banked plan via the TYPED split (live columns
      -- hoisted to the coeff kernel, audio reads shared array_ptrs) and via the
      -- FLOW split (arrays stay s1, everything in audio). Byte-identical ⇒
      -- per-array staging preserves the render (the coeff kernel fills the shared
      -- coefficient storage the audio kernel reads — run_coeff before buffer 0).
      let typedBytes ← renderTypedBytes plan stageBlocks
      match ← renderIrBytes plan with
      | .error e => IO.println s!"  FAIL  banks-staging  flow render: {firstLine e}"; pure false
      | .ok flowBytes =>
        let n := min typedBytes.size flowBytes.size
        let mut bitDiff := 0
        for i in [0:n] do
          if typedBytes[i]! != flowBytes[i]! then bitDiff := bitDiff + 1
        let mut energy : Float := 0.0
        for s in decodeF64LE typedBytes do energy := energy + s * s
        IO.println s!"        resonator→out (live freq/decay), banks-table={flagOn}:"
        IO.println s!"        result   reduce regions audio={reduces} coeff={coeffReduces} · array fills coeff={coeffFills} audio={audioFills} · typed≡flow bitDiff={bitDiff}/{n} · E={energy}"
        let renderOk := bitDiff == 0 && energy > 1e-6
        if flagOn then
          -- The bank loops and its LIVE columns (incr←freq, sigma←decay) hoist to
          -- the s0 kernel. CONST columns (cre=1/k^1.1, cim=0) stay in audio as fold
          -- Packs (one instruction each, LICM'd out of the sample loop). A Pack is
          -- O(1) instructions regardless of K, so the audio kernel is flat in mode
          -- count; per-array staging moves the LIVE coefficient mass off the audio
          -- thread. Byte-identity to the flow split proves the shared-array crossing.
          -- banks-region-stay: exactly ONE region, in the AUDIO kernel — the
          -- τ-reading body keeps the loop per-sample; the coeff kernel is
          -- region-free (the whole-region move must not fire on a clock-
          -- dependent bank).
          if reduces == 1 && coeffReduces == 0 && coeffFills > 0 && renderOk then
            IO.println s!"  PASS  banks-staging  bank looped ({reduces} region, in audio; 0 in coeff — clock-dependent region stays); {coeffFills} live column(s) → s0 kernel via shared array_ptrs; {audioFills} const baked; typed split ≡ flow byte-exact"; pure true
          else
            IO.println s!"  FAIL  banks-staging  flag on: reduces={reduces} coeffReduces={coeffReduces} coeff={coeffFills} renderOk={renderOk}"; pure false
        else
          if reduces == 0 && coeffReduces == 0 && coeffFills == 0 && renderOk then
            IO.println s!"  PASS  banks-staging  flag off: unrolled bank, no loop/columns, typed ≡ flow byte-exact"; pure true
          else
            IO.println s!"  FAIL  banks-staging  flag off: reduces={reduces} coeffReduces={coeffReduces} coeff={coeffFills} renderOk={renderOk}"; pure false

/-- The Metal column tripwire. The MSL ABI has no array binding — every
    plan array is a thread-private local — so a typed-split audio plan
    that advertises hoisted coefficient columns (`coeff_array_slots`)
    would read UNINITIALIZED memory on the GPU while playing correctly
    on the JIT. Silent wrongness; no render gate can see it (the
    metal_vs_jit corpus has zero arrays). The guard is structural:
    `EmitMsl.emitKernel` must REFUSE the split audio plan (loud before
    load — the previous kernel keeps playing) and must accept the
    UNSPLIT plan (fills in-kernel), which is what the session load
    falls back to on the metal backend. Banked default ⇒ columns hoist
    ⇒ refusal expected; under `TROPICAL_BANKS_UNROLL` nothing hoists ⇒
    both emissions must succeed (no false positive). -/
private def runMslColumnGuard (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => IO.println s!"  FAIL  msl-column-guard  json: {e}"; pure false
  | .ok j =>
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => IO.println s!"  FAIL  msl-column-guard  compile: {firstLine e}"; pure false
  | .ok (plan, _, stageBlocks) =>
    match Tropical.Ir.Stage0.hoistTyped plan stageBlocks with
    | .error e => IO.println s!"  FAIL  msl-column-guard  split: {firstLine e}"; pure false
    | .ok split =>
      let banked := Tropical.EmitArrow.banksTableEnabled
      let cols := split.audio.coeffArraySlots.size
      let splitMsl := Tropical.Ir.EmitMsl.emitKernel split.audio
      let unsplitMsl := Tropical.Ir.EmitMsl.emitKernel plan
      IO.println s!"        banked={banked} · hoisted columns={cols} · split-msl={if splitMsl.isOk then "ok" else "refused"} · unsplit-msl={if unsplitMsl.isOk then "ok" else "refused"}"
      if banked then
        match splitMsl, unsplitMsl with
        | .error _, .ok _ =>
          if cols > 0 then
            IO.println s!"  PASS  msl-column-guard  {cols} hoisted column(s): split plan refused, unsplit plan emits (the metal fallback)"; pure true
          else
            IO.println s!"  FAIL  msl-column-guard  banked: refused with no columns advertised"; pure false
        | _, _ =>
          IO.println s!"  FAIL  msl-column-guard  banked: cols={cols} splitOk={splitMsl.isOk} unsplitOk={unsplitMsl.isOk} (want refuse/ok)"; pure false
      else
        if cols == 0 && splitMsl.isOk && unsplitMsl.isOk then
          IO.println s!"  PASS  msl-column-guard  unrolled: no columns hoisted, both emissions clean (no false positive)"; pure true
        else
          IO.println s!"  FAIL  msl-column-guard  unrolled: cols={cols} splitOk={splitMsl.isOk} unsplitOk={unsplitMsl.isOk}"; pure false

/-- THE COMPILE-FLATNESS BENCHMARK (banks-as-data payoff). Where `banks-staging`
    proves the payoff STRUCTURALLY at one mode count (columns hoist, audio is
    fill-free), this MEASURES it across scale: compile the SAME room at K=6 and
    K=512 modes and show the AUDIO kernel's plan instruction count is flat in K
    (within a small constant) while only the coefficient kernel grows.

    Why the audio kernel is flat: the resonator's freq/decay are LIVE (session
    slots → stage-0), so their coefficient columns (incr←freq, sigma←decay) hoist
    to the s0 coefficient kernel — those fills scale with K there, at knob rate,
    O0. The amps (cre=1/k^1.1, cim=0) are compile-TIME constants, so their columns
    bake into the audio kernel as fold `Pack`s — but a `Pack` is ONE instruction
    regardless of K, so the audio kernel stays flat. The banked audio body is then
    a fixed O(1) reduce region reading the shared, coeff-filled storage.

    NOTE the nuance: this flatness rides on freq/decay being live. A fully-STATIC
    bank would NOT be flat — with no live columns to hoist, its column arithmetic
    stays as fold in the audio kernel and grows with K. True static flatness needs
    blocker 4's fill-as-reduce. Flag off ⇒ the bank is unrolled and the audio
    kernel grows ~linearly in K; we only REPORT that (the documented contrast). -/
private def runBanksBench (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let flagOn := Tropical.EmitArrow.banksTableEnabled
  -- A bare resonator → out with live freq/decay, at a graph-configurable mode
  -- count K (the `"partials"` param threaded through Playground.buildNode).
  let mkSrc := fun (k : Nat) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":"
      ++ toString k ++ "}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  -- Compile at K, split via the typed stage-0 hoist, and report
  -- (audio-kernel instrs, coeff-kernel instrs).
  let compileAt : Nat → Except String (Nat × Nat) := fun k => do
    let j ← Lean.Json.parse (mkSrc k)
    let (plan, _, stageBlocks) ← Tropical.Playground.compilePlanPure arena resolved j
    let split ← Tropical.Ir.Stage0.hoistTyped plan stageBlocks
    let audioN := planInstrCount split.audio
    let coeffN := match split.coeff? with | some c => planInstrCount c | none => 0
    pure (audioN, coeffN)
  match compileAt 6, compileAt 512 with
  | .error e, _ => IO.println s!"  FAIL  banks-bench  K=6 compile: {firstLine e}"; pure false
  | _, .error e => IO.println s!"  FAIL  banks-bench  K=512 compile: {firstLine e}"; pure false
  | .ok (a6, c6), .ok (a512, c512) =>
    let dAudio := if a512 ≥ a6 then a512 - a6 else a6 - a512
    IO.println s!"        bare resonator→out (live freq/decay), banks-table={flagOn}:"
    IO.println s!"        audio-kernel instrs: K=6 → {a6}   K=512 → {a512}   Δ={dAudio}"
    IO.println s!"        coeff-kernel instrs: K=6 → {c6}   K=512 → {c512}"
    if flagOn then
      -- Banked: the audio kernel is a fixed reduce body + O(1) const Packs, flat
      -- in K; the LIVE columns' fills live in the coeff kernel and scale there.
      let flat := dAudio ≤ 8
      let coeffGrows := decide (c512 > c6)
      if flat then
        IO.println s!"  PASS  banks-bench  flag on: audio kernel FLAT in K (Δ={dAudio} ≤ 8, K=6→512); coeff kernel scales with K ({c6}→{c512}, grows={coeffGrows}) at knob rate"; pure true
      else
        IO.println s!"  FAIL  banks-bench  flag on: audio kernel NOT flat (Δ={dAudio} > 8) — a K-dependent audio instruction leaked past the coeff hoist"; pure false
    else
      -- Unrolled: no loop, no columns; the whole bank's arithmetic is in the
      -- audio kernel and grows with K. Not a failure — the documented contrast.
      IO.println s!"  PASS  banks-bench  flag off: unrolled bank, audio kernel GROWS with K ({a6}→{a512}, Δ={dAudio}) — the contrast the banked path removes"; pure true

/-- THE TRIP-COUNT gate (trip-count-as-data v1: the room-size knob). A resonator
    with the optional STATIC `partials_max` capacity carries a LIVE `partials`
    slot whose in-kernel read is the bank's effective trip count, clamped to
    capacity — mode count stops being topology. (a) at the default knob
    (= capacity) it renders BIT-EXACT to the fully-static graph at the same
    count; (b) knob at 4 ≡ static partials=4 (the clamped loop visits the same
    mode prefix in unroll order — same ops, same bits; the dynamic plan's
    capacity-sized columns beyond index 4 are never read); (c) knob above
    capacity clamps (≡ the capacity render); (d) knob at 0 sums no modes —
    silence from the bank, the patch's only source. A dynamic-count bank always
    BANKS (a runtime count cannot unroll), so this holds in both flag states of
    TROPICAL_BANKS_UNROLL. The static graph must NOT grow the slot (opt-in:
    `partials_max` absent ⇒ no `param:res.partials`). -/
private def runBanksCount (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let staticSrc := fun (k : Nat) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":"
      ++ toString k ++ "}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let dynSrc := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":16,\"partials_max\":16}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let compile := fun (src : String) => do
    let j ← Lean.Json.parse src
    Tropical.Playground.compilePlanPure arena resolved j
  -- Load a compiled graph, optionally preset a slot, render one 2048-sample
  -- block (the modal-live pattern: a live slot driven through a render).
  let render := fun (plan : Tropical.Plan.FlatPlan)
      (blocks : Array (Array (Option Tropical.Ir.Stage)))
      (preset? : Option Float) => do
    let rt ← Tropical.Ffi.Runtime.new 2048
    Tropical.StagedLoad.loadTyped rt plan blocks
    let mut slotOk := true
    if let some v := preset? then
      match ← rt.slotIndex? "param:res.partials" with
      | some idx => rt.setSlot idx v
      | none => slotOk := false
    rt.process
    pure (slotOk, decodeF64LE (← rt.outputBytes))
  let bitDiffOf := fun (a b : Array Float) => Id.run do
    let mut d := 0
    for i in [0:min a.size b.size] do
      if a[i]! != b[i]! then d := d + 1
    return d
  let energyOf := fun (a : Array Float) => a.foldl (fun acc s => acc + s * s) 0.0
  match compile (staticSrc 16), compile (staticSrc 4), compile dynSrc with
  | .error e, _, _ => IO.println s!"  FAIL  banks-count  static-16 compile: {firstLine e}"; pure false
  | _, .error e, _ => IO.println s!"  FAIL  banks-count  static-4 compile: {firstLine e}"; pure false
  | _, _, .error e => IO.println s!"  FAIL  banks-count  dynamic compile: {firstLine e}"; pure false
  | .ok (p16, _, b16), .ok (p4, _, b4), .ok (pd, _, bd) =>
    -- opt-in: the static graph must NOT have grown a partials slot.
    let rtS ← Tropical.Ffi.Runtime.new 2048
    Tropical.StagedLoad.loadTyped rtS p16 b16
    let staticHasSlot := (← rtS.slotIndex? "param:res.partials").isSome
    let (_, s16) ← render p16 b16 none
    let (_, s4)  ← render p4 b4 none
    let (_, dDef)   ← render pd bd none          -- knob at its default (16)
    let (ok4, d4)   ← render pd bd (some 4.0)    -- knob at 4
    let (okC, dC)   ← render pd bd (some 100.0)  -- above capacity → clamps to 16
    let (okZ, dZ)   ← render pd bd (some 0.0)    -- zero modes → silence
    let slotLive := ok4 && okC && okZ
    let e16 := energyOf s16
    let dA := bitDiffOf dDef s16
    let dB := bitDiffOf d4 s4
    let dCn := bitDiffOf dC s16
    let eZ := energyOf dZ
    IO.println s!"        resonator partials_max=16 (LIVE partials slot) vs fully-static graphs:"
    IO.println s!"        result   default(16)≡static16 bitDiff={dA}/{s16.size} · knob4≡static4 bitDiff={dB}/{s4.size} · knob100≡static16 bitDiff={dCn}/{s16.size}"
    IO.println s!"        result   E[static16]={e16} · E[knob0]={eZ} · slot live={slotLive} · static graph has slot={staticHasSlot} (want false)"
    if dA == 0 && dB == 0 && dCn == 0 && eZ ≤ 1e-24 && e16 > 1e-6 && slotLive && !staticHasSlot then
      IO.println s!"  PASS  banks-count  live trip count ≡ static at 16/4, clamps at 100, silent at 0 — mode count is data, not topology"; pure true
    else
      IO.println s!"  FAIL  banks-count  dA={dA} dB={dB} dC={dCn} eZ={eZ} e16={e16} slotLive={slotLive} staticHasSlot={staticHasSlot}"; pure false

/-- THE CACHE-INVARIANCE gate (the trip-count payoff). The kernel cache is keyed
    by md5(ir_text) (`OrcJitEngine`), so a knob that changed the IR text would
    force a full recompile. Two compiles of the SAME graph differing only in the
    `partials` DEFAULT (4 vs 12, same `partials_max`) must emit IDENTICAL LLVM
    IR — the count is a slot read; its default lives in plan metadata, never in
    the kernel text. A `partials_max` change must CHANGE the text: capacity IS
    topology (column sizes, the loop's static bound). Asserted on both the
    unsplit kernel and the typed-split audio kernel (the artifact the staged
    load actually caches). -/
private def runBanksCountCache (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := fun (dflt cap : Nat) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4,\"partials\":"
      ++ toString dflt ++ ",\"partials_max\":" ++ toString cap ++ "}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let irOf : Nat → Nat → Except String (String × String) := fun dflt cap => do
    let j ← (Lean.Json.parse (src dflt cap)).mapError (s!"json: {·}")
    let (plan, _, blocks) ← Tropical.Playground.compilePlanPure arena resolved j
    let split ← Tropical.Ir.Stage0.hoistTyped plan blocks
    pure (← Tropical.Ir.EmitLlvm.emitKernel plan, ← Tropical.Ir.EmitLlvm.emitKernel split.audio)
  match irOf 4 16, irOf 12 16, irOf 4 24 with
  | .error e, _, _ | _, .error e, _ | _, _, .error e =>
    IO.println s!"  FAIL  banks-count-cache  compile/emit: {firstLine e}"; pure false
  | .ok (u4, a4), .ok (u12, a12), .ok (u24, a24) =>
    let knobInvariant := u4 == u12 && a4 == a12
    let capMoves := u4 != u24 && a4 != a24
    IO.println s!"        same graph, partials default 4 vs 12 (cap 16) vs cap 24:"
    IO.println s!"        result   knob-invariant IR: unsplit={u4 == u12} audio={a4 == a12} ({u4.length}B) · capacity moves IR: unsplit={u4 != u24} audio={a4 != a24}"
    if knobInvariant && capMoves then
      IO.println s!"  PASS  banks-count-cache  IR text is knob-invariant (md5 cache hit across counts); partials_max changes it (capacity is topology)"; pure true
    else
      IO.println s!"  FAIL  banks-count-cache  knobInvariant={knobInvariant} capMoves={capMoves}"; pure false

/-- Build the resonator → filter → out patch graph as Json. -/
private def filterPatchJson (fc : Int) (resM : Int) (resE : Nat) (srcF srcDecay : Int) : Lean.Json :=
  let node := fun (id kind : String) (params : List (String × Lean.Json))
                  (ins : List (String × Lean.Json)) =>
    Lean.Json.mkObj <|
      [("id", Lean.Json.str id), ("kind", Lean.Json.str kind),
       ("params", Lean.Json.mkObj params)] ++
      (if ins.isEmpty then [] else [("in", Lean.Json.mkObj ins)])
  Lean.Json.mkObj [
    ("nodes", Lean.Json.arr #[
      node "res" "resonator" [("freq", Lean.Json.num (jn srcF)), ("decay", Lean.Json.num (jn srcDecay))] [],
      node "flt" "filter" [("cutoff", Lean.Json.num (jn fc)), ("resonance", Lean.Json.num (jn resM resE))]
        [("in", Lean.Json.arr #[Lean.Json.str "res"])],
      node "out" "out" [] [("in", Lean.Json.arr #[Lean.Json.str "flt"])]]),
    ("out", Lean.Json.str "out")]

private def renderFilterPatch (arena : Arena) (resolved : Array (String × ProgramIdx))
    (j : Lean.Json) (n : Nat) : IO (Except String (Array Float)) := do
  match Tropical.Playground.compilePlanPure arena resolved j with
  | .error e => pure (.error s!"compile: {firstLine e}")
  | .ok (plan, _, _) =>
    match ← renderPlanSamples plan n with
    | .error e => pure (.error s!"render: {firstLine e}")
    | .ok samples => pure (.ok samples)

private def tailEnergy (xs : Array Float) (lo : Nat) : Float := Id.run do
  let mut acc : Float := 0.0
  for i in [lo:xs.size] do
    acc := acc + xs[i]! * xs[i]!
  pure acc

/-- THE MODAL FILTER gate (the VCFQ). A `filter` node is a `modalReverb`
    whose room is one EXACT conjugate pole pair (`filterPair`), so three
    behaviors must hold, all through the ordinary graph surface:
    (A) LOWPASS: the same struck resonator through cutoff=4000 vs cutoff=60
        loses most of its energy (the composition's forced modes carry
        `a·H(λ)`, and |H| is small far above fc).
    (B) THE PING: with a fast-dying excitation and resonance at the top of
        the knob (Q ≈ 44), the tail is the FILTER ringing at ω_d ≈ 2π·fc —
        zero-crossing rate within 3% of fc. This is the Serge character the
        node exists for: high resonance IS a struck resonator.
    (C) LIVE: cutoff is a glided live knob — writing its glide endpoints
        (#v0/#v1) diverges the output vs an untouched twin, THROUGH the
        composition (no relower, no dead knob). -/
private def runModalFilter (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- (A) lowpass attenuation
  match ← renderFilterPatch arena resolved (filterPatchJson 4000 3 1 220 4) 4096,
        ← renderFilterPatch arena resolved (filterPatchJson 60 3 1 220 4) 4096 with
  | .error e, _ | _, .error e => IO.println s!"  FAIL  modal-filter  (A) {e}"; pure false
  | .ok openS, .ok closedS =>
    let eOpen := tailEnergy openS 200
    let eClosed := tailEnergy closedS 200
    -- (B) the ping: fast-dying strike at 1800 Hz, filter fc=500 res=1 (Q≈44);
    -- by half the window the source is gone and the tail is the filter's ring.
    match ← renderFilterPatch arena resolved (filterPatchJson 500 1 0 1800 60) 8192 with
    | .error e => IO.println s!"  FAIL  modal-filter  (B) {e}"; pure false
    | .ok ping =>
      let mut crossings := 0
      for i in [4097:8192] do
        if (ping[i-1]! < 0.0 && ping[i]! >= 0.0) || (ping[i-1]! >= 0.0 && ping[i]! < 0.0) then
          crossings := crossings + 1
      let tailSec := (8192.0 - 4097.0) / 44100.0
      let ringHz := crossings.toFloat / 2.0 / tailSec
      let eTail := tailEnergy ping 4097
      -- (C) live cutoff on one of two twin runtimes
      match Tropical.Playground.compilePlanPure arena resolved (filterPatchJson 800 5 1 220 4) with
      | .error e => IO.println s!"  FAIL  modal-filter  (C) compile: {firstLine e}"; pure false
      | .ok (plan, _, stageBlocks) =>
      match plan.toWire, Tropical.Ir.EmitLlvm.emitKernel plan with
      | .ok _, .ok _ =>
        let rt ← Tropical.Ffi.Runtime.new 2048
        Tropical.StagedLoad.loadTyped rt plan stageBlocks
        let rt2 ← Tropical.Ffi.Runtime.new 2048
        Tropical.StagedLoad.loadTyped rt2 plan stageBlocks
        let v0? ← rt.slotIndex? "param:flt.cutoff#v0"
        let v1? ← rt.slotIndex? "param:flt.cutoff#v1"
        rt.process; rt2.process
        if let some v0 := v0? then rt.setSlot v0 60.0
        if let some v1 := v1? then rt.setSlot v1 60.0
        rt.process; rt2.process
        let a := decodeF64LE (← rt.outputBytes)
        let b := decodeF64LE (← rt2.outputBytes)
        let mut dE : Float := 0.0
        let mut e0 : Float := 0.0
        for i in [0:min a.size b.size] do
          let d := a[i]! - b[i]!
          dE := dE + d * d
          e0 := e0 + b[i]! * b[i]!
        let slotsPresent := v0?.isSome && v1?.isSome
        let knobsLive := slotsPresent && dE > 1e-9 * e0 && e0 > 1e-9
        IO.println s!"        filter = residue-composed conjugate pole pair (H exact); resonator → filter → out:"
        IO.println s!"        (A) E[cutoff 4kHz]={eOpen} vs E[60Hz]={eClosed} (ratio {eOpen/(eClosed+1e-300)})"
        IO.println s!"        (B) Q≈44 ping: tail rings at {ringHz} Hz (fc=500, want ±3%), E[tail]={eTail}"
        IO.println s!"        (C) cutoff glide slots present={slotsPresent} · ΔE/E after move={dE/(e0+1e-300)}"
        if eOpen > 20.0 * eClosed && eClosed > 0.0 &&
           ringHz > 485.0 && ringHz < 515.0 && eTail > 1e-8 && knobsLive then
          IO.println s!"  PASS  modal-filter  lowpass attenuates ({eOpen/(eClosed+1e-300)}x), Q≈44 pings at {ringHz} Hz, cutoff live through the composition"; pure true
        else
          IO.println s!"  FAIL  modal-filter  eOpen={eOpen} eClosed={eClosed} ringHz={ringHz} (want 485-515) eTail={eTail} live={knobsLive}"; pure false
      | .error e, _ | _, .error e =>
        IO.println s!"  FAIL  modal-filter  (C) emit: {firstLine e}"; pure false

open Tropical.EmitArrow in
/-- THE MODAL ADDRESS gate. A resonator's `addr` inlet: a patched CF signal BECOMES
    the bank's absolute time-coordinate (`modalAddrWarp`), so the causal gate — the
    strike — tracks the SIGNAL, not the master clock. Three things, end to end:
    (1) SCALING — address = time (offset 0) reproduces the un-addressed bank (only a
        float round-trip apart), proving `s`-seconds maps to exactly the right clock;
    (2) TRIGGER — address = time − 0.01 s stays silent until sample 0.01·SR, then
        rings, proving the signal RELOCATES the strike (scrub-to-trigger, not a
        re-strike); (3) DECODE — a JSON `osc → res.addr, res → out` compiles through
        the real `compilePlanPure` to codegen (the graph surface wires the address). -/
private def runModalAddr (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let modes : Array ModalMode := #[
    ModalMode.hz (lit 220) (lit 30 1) (lit 6 1),
    ModalMode.hz (lit 440) (lit 45 1) (lit 3 1)]
  let n := 2048
  let onsetSec : Float := 0.01
  let onset := (onsetSec * 44100.0).toUInt64.toNat
  -- (3) decode: a real patched-signal address (an LFO osc) through the JSON surface.
  let src := "{\"nodes\":[" ++
    "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":40,\"morph\":0}}," ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4},\"in\":{\"addr\":[\"lfo\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"res\"]}}],\"out\":\"out\"}"
  let decodeOk : Bool := match Lean.Json.parse src with
    | .error _ => false
    | .ok j => match Tropical.Playground.compilePlanPure arena resolved j with
      | .error _ => false
      | .ok (plan, _, _) => (Tropical.Ir.EmitLlvm.emitKernel plan).toOption.isSome
  match buildAndFinish (.ok (buildModalBankArrow "ma_ref" modes (lit 0) arena)),
        buildAndFinish (.ok (buildModalAddrRamp "ma_id" modes (lit 0) 0.0 arena)),
        buildAndFinish (.ok (buildModalAddrRamp "ma_off" modes (lit 0) onsetSec arena)) with
  | .ok refp, .ok idp, .ok offp =>
    match ← renderPlanSamples refp n, ← renderPlanSamples idp n, ← renderPlanSamples offp n with
    | .ok ref, .ok ida, .ok off =>
      let mut maxErr : Float := 0.0
      for i in [0:n] do
        let e := (ida[i]! - ref[i]!).abs
        if e > maxErr then maxErr := e
      let mut preMax : Float := 0.0
      for i in [0:onset] do
        let a := off[i]!.abs
        if a > preMax then preMax := a
      let mut postPeak : Float := 0.0
      for i in [onset+50:n] do
        let a := off[i]!.abs
        if a > postPeak then postPeak := a
      IO.println s!"        addressed resonator (a patched CF signal AS the time coordinate):"
      IO.println s!"        result   identity-addr max|Δ| vs un-addressed={maxErr} · offset-addr pre-onset|max|={preMax} post-onset peak={postPeak} · graph decode ok={decodeOk}"
      if maxErr < 1e-4 && preMax == 0.0 && postPeak > 1e-6 && decodeOk then
        IO.println s!"  PASS  modal-addr  a patched signal drives the bank's time: address=time ≡ un-addressed; offset relocates the strike; graph decode compiles"; pure true
      else
        IO.println s!"  FAIL  modal-addr  maxErr={maxErr} preMax={preMax} postPeak={postPeak} decodeOk={decodeOk}"; pure false
    | .error e, _, _ | _, .error e, _ | _, _, .error e => IO.println s!"  FAIL  modal-addr  render: {firstLine e}"; pure false
  | _, _, _ => IO.println s!"  FAIL  modal-addr  build"; pure false

-- ── THE DEAD-SLOT LINT ─────────────────────────────────────────────────────
/-- Module-slot indices READ by an instance function: every `.slot` operand of
    every instruction (preamble, body, pre-input), recursively through children.
    A `WriteSlot` dst is a write, not a read — a slot only written is still dead. -/
private partial def slotReadsOf (f : Tropical.Plan.InstanceFunction) : Array Nat :=
  let ofInstrs := fun (instrs : Array Tropical.Plan.NInstr) =>
    instrs.flatMap fun ins => ins.args.filterMap fun
      | .slot i _ => some i
      | _ => none
  ofInstrs f.preambleInstructions ++ ofInstrs f.instructions
    ++ ofInstrs f.preInputInstructions ++ f.children.flatMap slotReadsOf

/-- `param:*` entries of `slotNames` referenced by NO instruction operand in the
    plan. Sink `inputs` are slot reads too (they consume the `__root__.out`-style
    output slots), but a param slot whose only consumer is a sink would itself be
    a wiring bug, so the lint demands an instruction read for `param:*`. A hit is
    a dead knob: `setSlot` succeeds, no instruction listens — the class the
    reverb-discards-the-voice's-poles regression shipped in. -/
def unreadParamSlots (plan : Tropical.Plan.FlatPlan) : Array String := Id.run do
  let reads := plan.instanceFunctions.flatMap slotReadsOf
  let mut dead : Array String := #[]
  for i in [0:plan.slotNames.size] do
    let name := plan.slotNames[i]!
    if "param:".isPrefixOf name && !reads.contains i then
      dead := dead.push name
  return dead

open Tropical.Playground in
/-- THE VOCABULARY-DRIVENNESS gate (successor to the table-coherence gate,
    which retired with the static `nodeSchema` it was pinning). `get_vocabulary`
    is only honest if the served table can DRIVE a client: for every kind,
    generate a minimal patch FROM the vocabulary alone (wire each `in` inlet
    from a domain-matching helper; leave optional normals intact), compile it
    through the real `compilePlanPure`, and assert every knob the table
    declares registers a live slot and nothing registered goes unread. A kind
    added to the table is covered here with no test edit. -/
private def runVocabDriven (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mut ok := true
  let mut covered := 0
  let mut issues : Array String := #[]
  for kind in vocabularyKinds do
    if kind == "out" then continue
    let mut nodes : Array Lean.Json := #[]
    let mut inFields : List (String × Lean.Json) := []
    for p in portSpecs kind do
      if !p.accepts.isEmpty && p.name == "in" then
        if p.accepts == #[PortDomain.modal] then
          nodes := nodes.push (Lean.Json.mkObj
            [("id", .str "helper_m"), ("kind", .str "resonator"), ("params", Lean.Json.mkObj [])])
          inFields := inFields ++ [("in", Lean.Json.arr #[.str "helper_m"])]
        else
          nodes := nodes.push (Lean.Json.mkObj
            [("id", .str "helper_s"), ("kind", .str "source"), ("params", Lean.Json.mkObj [])])
          inFields := inFields ++ [("in", Lean.Json.arr #[.str "helper_s"])]
    nodes := nodes.push (Lean.Json.mkObj
      [("id", .str "dut"), ("kind", .str kind), ("params", Lean.Json.mkObj []),
       ("in", Lean.Json.mkObj inFields)])
    -- a control-outlet kind (a bare Knob) drives nothing by itself — its
    -- natural minimal patch consumes it through a source's control inlet, so
    -- the patch has a generator (and the master-clock slots have a reader).
    if outletOf kind == some .control then
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "consumer"), ("kind", .str "source"), ("params", Lean.Json.mkObj []),
         ("in", Lean.Json.mkObj [("freq", Lean.Json.arr #[.str "dut"])])])
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "outn"), ("kind", .str "out"),
         ("in", Lean.Json.mkObj [("in", Lean.Json.arr #[.str "consumer"])])])
    else
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "outn"), ("kind", .str "out"),
         ("in", Lean.Json.mkObj [("in", Lean.Json.arr #[.str "dut"])])])
    let patch := Lean.Json.mkObj [("nodes", Lean.Json.arr nodes), ("out", .str "outn")]
    match Tropical.Playground.compilePlanPure arena resolved patch with
    | .error e => ok := false; issues := issues.push s!"{kind}: compile: {firstLine e}"
    | .ok (plan, _, _) =>
      covered := covered + 1
      -- every table knob must land as a slot: raw/anchor knobs under the bare
      -- name, glided knobs as their #v0 anchor triple.
      for p in portSpecs kind do
        if p.knob.isSome then
          let base := s!"param:dut.{p.name}"
          if !(plan.slotNames.any (fun s => s == base || s == s!"{base}#v0")) then
            ok := false; issues := issues.push s!"{kind}: {base} not registered"
      let dead := unreadParamSlots plan
      if !dead.isEmpty then
        ok := false; issues := issues.push s!"{kind}: unread {dead}"
  IO.println s!"        {covered} kinds, each compiled from a vocabulary-generated minimal patch:"
  IO.println s!"        result   {if issues.isEmpty then "every declared knob registers and is read" else toString issues}"
  if ok then
    IO.println "  PASS  vocab-driven  the served vocabulary drives a compiling patch per kind — declared knobs live, nothing unread"; pure true
  else
    IO.println s!"  FAIL  vocab-driven  {issues}"; pure false

open Tropical.Playground in
/-- THE REALIZED-STATE REPORT gate. The `load_patch_graph` reply must state
    FACTS a surface can render — wired vs normalled inputs, live params with
    disciplines, excluded nodes — and never a warning (house contract: legal-
    but-incomplete compiles silently; the report tells, it does not scold).
    This is the protocol-level net for the silence-with-`{ok:true}` class. -/
private def runRealizedReport : IO Bool := do
  let mkPatch := fun (withMod : Bool) (dangler : Bool) =>
    let base := "{\"nodes\":[" ++
      "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
      (if withMod then "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":0.4}}," else "") ++
      (if dangler then "{\"id\":\"orphan\",\"kind\":\"source\",\"params\":{\"freq\":99}}," else "") ++
      "{\"id\":\"sfw\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"osc\"]" ++
      (if withMod then ",\"mod\":[\"lfo\"]" else "") ++ "}}," ++
      "{\"id\":\"outn\",\"kind\":\"out\",\"in\":{\"in\":[\"sfw\"]}}],\"out\":\"outn\"}"
    base
  let inputState := fun (rep : Lean.Json) (node port : String) =>
    match rep.getObjVal? "inputs" with
    | .ok (.arr a) => (a.find? fun ij =>
        ((ij.getObjVal? "node").toOption.bind (·.getStr?.toOption)) == some node &&
        ((ij.getObjVal? "port").toOption.bind (·.getStr?.toOption)) == some port).bind
        fun ij => (ij.getObjVal? "state").toOption.bind (·.getStr?.toOption)
    | _ => none
  let paramNames := fun (rep : Lean.Json) =>
    match rep.getObjVal? "params" with
    | .ok (.arr a) => a.filterMap fun (pj : Lean.Json) =>
        (pj.getObjVal? "name").toOption.bind (·.getStr?.toOption)
    | _ => #[]
  let nodeStatus := fun (rep : Lean.Json) (id : String) =>
    match rep.getObjVal? "nodes" with
    | .ok (.arr a) => (a.find? fun nj =>
        ((nj.getObjVal? "id").toOption.bind (·.getStr?.toOption)) == some id).bind
        fun nj => (nj.getObjVal? "status").toOption.bind (·.getStr?.toOption)
    | _ => none
  match Lean.Json.parse (mkPatch false false), Lean.Json.parse (mkPatch true false),
        Lean.Json.parse (mkPatch false true) with
  | .ok jUnwired, .ok jWired, .ok jDangler =>
    let repU := realizedReport jUnwired #[]
    let repW := realizedReport jWired #[]
    let repD := realizedReport jDangler #[]
    let unwiredOk := inputState repU "sfw" "mod" == some "normalled"
      && (paramNames repU).contains "sfw.rate"
    let wiredOk := inputState repW "sfw" "mod" == some "wired"
      && !(paramNames repW).contains "sfw.rate"
    let danglerOk := nodeStatus repD "orphan" == some "excluded"
      && nodeStatus repD "osc" == some "active"
    -- the no-warnings contract, checked on the wire form itself
    let noWarnOk := ((repU.compress ++ repW.compress ++ repD.compress).toLower.splitOn "warn").length == 1
    IO.println s!"        unwired: mod={inputState repU "sfw" "mod"} rate-param={((paramNames repU).contains "sfw.rate")} · wired: mod={inputState repW "sfw" "mod"} rate-param={((paramNames repW).contains "sfw.rate")} · orphan={nodeStatus repD "orphan"}"
    if unwiredOk && wiredOk && danglerOk && noWarnOk then
      IO.println "  PASS  realized-report  facts, not warnings: normalled/wired inputs, owned knobs absent when superseded, excluded nodes named"; pure true
    else
      IO.println s!"  FAIL  realized-report  unwired={unwiredOk} wired={wiredOk} dangler={danglerOk} noWarn={noWarnOk}"; pure false
  | _, _, _ => IO.println "  FAIL  realized-report  patch json parse"; pure false

open Tropical.Playground in
/-- THE MANIFEST-DISCIPLINE gate. `param_disciplines` is host-contract data:
    every entry must be consistent with the slots the same plan carries (glide
    companions exist and the base slot doesn't; anchor/raw base slots exist;
    the velocity entry names its tau_base companion), and a knob superseded by
    a wired normal must be absent from the table exactly as it is from the
    slots. A host dispatching from this table can then trust it blind. -/
private def runManifestDisciplines (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let patch := fun (withMod : Bool) => "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    (if withMod then "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":0.4}}," else "") ++
    "{\"id\":\"sfw\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"osc\"]" ++
    (if withMod then ",\"mod\":[\"lfo\"]" else "") ++ "}}," ++
    "{\"id\":\"outn\",\"kind\":\"out\",\"in\":{\"in\":[\"sfw\"]}}],\"out\":\"outn\"}"
  let check := fun (label : String) (plan : Tropical.Plan.FlatPlan) => Id.run do
    let mut issues : Array String := #[]
    let names := plan.slotNames
    for d in plan.paramDisciplines do
      let base := s!"param:{d.name}"
      for c in d.companions do
        if !names.contains s!"param:{c}" then
          issues := issues.push s!"{label}: {d.name} companion {c} has no slot"
      match d.discipline with
      | "glide" =>
        if names.contains base then
          issues := issues.push s!"{label}: glided {d.name} has a base slot (companions are the value)"
        if d.glideDurSec.isNone then
          issues := issues.push s!"{label}: glided {d.name} missing glide_dur_sec"
      | "raw" | "anchor" | "velocity" =>
        if !names.contains base then
          issues := issues.push s!"{label}: {d.discipline} {d.name} has no base slot"
      | other => issues := issues.push s!"{label}: unknown discipline {other}"
    return issues
  match Lean.Json.parse (patch false), Lean.Json.parse (patch true) with
  | .ok ju, .ok jw =>
    match Tropical.Playground.compilePlanPure arena resolved ju,
          Tropical.Playground.compilePlanPure arena resolved jw with
    | .ok (pu, _, _), .ok (pw, _, _) =>
      let mut issues := check "unwired" pu ++ check "wired" pw
      if !(pu.paramDisciplines.any (·.name == "sfw.rate")) then
        issues := issues.push "unwired: sfw.rate missing from disciplines"
      if pw.paramDisciplines.any (·.name == "sfw.rate") then
        issues := issues.push "wired: superseded sfw.rate present in disciplines"
      if !(pu.paramDisciplines.any fun d => d.name == "master.velocity"
            && d.discipline == "velocity" && d.companions.contains "master.tau_base") then
        issues := issues.push "master.velocity entry wrong"
      IO.println s!"        {pu.paramDisciplines.size}+{pw.paramDisciplines.size} manifest entries checked against their plans' slots:"
      IO.println s!"        result   {if issues.isEmpty then "consistent" else toString issues}"
      if issues.isEmpty then
        IO.println "  PASS  manifest-disciplines  param_disciplines ≡ the plan's slots — a host can dispatch from it blind"; pure true
      else
        IO.println s!"  FAIL  manifest-disciplines  {issues}"; pure false
    | .error e, _ | _, .error e =>
      IO.println s!"  FAIL  manifest-disciplines  compile: {firstLine e}"; pure false
  | _, _ => IO.println "  FAIL  manifest-disciplines  json parse"; pure false

/-- THE DEAD-SLOT LINT gate (the systemic net for the dead-knob class). Canonical
    patches covering every playground node kind compile through the real
    `compilePlanPure`, and every `param:*` slot each plan registers must be READ
    by some instruction operand. Presence-only checks pass a dead knob — the slot
    exists, `setSlot` succeeds, nothing listens — which is exactly how the
    reverb-drops-the-source's-poles regression stayed invisible; unreadness in the
    PLAN is the property that catches the whole class, whatever node grows it next. -/
private def runDeadSlotLint (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- Allowlist for slots proven legitimately unread, one justified entry at a
  -- time (`<patch-label>:<slot>`); a blanket param exclusion would re-open the
  -- hole the gate exists to close. Currently empty: every registered knob in
  -- the canonical patches must be live.
  let allow : Array String := #[]
  -- Every buildNode kind: knob, source, pluck, comb, flange, sflange, fm,
  -- delay, reverse, mix, ring, resonator, reverb, modalmix, out. The top-level
  -- "out" field is mandatory — without it the patch gracefully compiles to
  -- silence and EVERY knob goes dead (which this gate would report).
  let signalChain := "{\"nodes\":[" ++
    "{\"id\":\"k\",\"kind\":\"knob\",\"params\":{\"value\":110}}," ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"morph\":0.2},\"in\":{\"freq\":[\"k\"]}}," ++
    "{\"id\":\"fl\",\"kind\":\"flange\",\"params\":{\"depth\":0.0007},\"in\":{\"in\":[\"osc\"]}}," ++
    "{\"id\":\"sf\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"fl\"]}}," ++
    "{\"id\":\"fmn\",\"kind\":\"fm\",\"params\":{\"carrier\":330,\"depth\":8},\"in\":{\"in\":[\"sf\"]}}," ++
    "{\"id\":\"dl\",\"kind\":\"delay\",\"params\":{\"amount\":0.004},\"in\":{\"in\":[\"fmn\"]}}," ++
    "{\"id\":\"rv\",\"kind\":\"reverse\",\"in\":{\"in\":[\"dl\"]}}," ++
    "{\"id\":\"osc2\",\"kind\":\"source\",\"params\":{\"freq\":330}}," ++
    "{\"id\":\"rg\",\"kind\":\"ring\",\"in\":{\"in\":[\"rv\",\"osc2\"]}}," ++
    "{\"id\":\"pl\",\"kind\":\"pluck\",\"params\":{\"freq\":110}}," ++
    "{\"id\":\"cb\",\"kind\":\"comb\",\"params\":{\"delay\":0.012,\"decay\":0.7},\"in\":{\"in\":[\"pl\"]}}," ++
    "{\"id\":\"mx\",\"kind\":\"mix\",\"in\":{\"in\":[\"rg\",\"cb\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"mx\"]}}],\"out\":\"out\"}"
  -- The regression's exact shape: the reverb must keep its source's poles live.
  let modalChain := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"rvb\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rvb\"]}}],\"out\":\"out\"}"
  let modalMix := "{\"nodes\":[" ++
    "{\"id\":\"res1\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"res2\",\"kind\":\"resonator\",\"params\":{\"freq\":330,\"decay\":3}}," ++
    "{\"id\":\"mm\",\"kind\":\"modalmix\",\"in\":{\"in\":[\"res1\",\"res2\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"mm\"]}}],\"out\":\"out\"}"
  -- The once-KNOWN-HOLE, now in the canonical set: `sflange` with a `mod` cord
  -- patched. `rate` parameterizes `mod`'s normalled LFO (ownerPort in the
  -- port-spec table), so wiring `mod` removes the LFO, the knob, and the slot
  -- together — the lint additionally asserts the slot's ABSENCE below, so the
  -- old dead-knob encoding (registered-but-unread) can't quietly return.
  let sflangeWired := "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":0.4,\"morph\":1}}," ++
    "{\"id\":\"sfw\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"osc\"],\"mod\":[\"lfo\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"sfw\"]}}],\"out\":\"out\"}"
  let patches : Array (String × String) := #[
    ("signal-chain", signalChain), ("modal-chain", modalChain), ("modal-mix", modalMix),
    ("sflange-wired", sflangeWired)]
  let mut ok := true
  let mut checked := 0
  let mut deadAll : Array String := #[]
  for (label, src) in patches do
    match Lean.Json.parse src with
    | .error e => IO.println s!"  FAIL  dead-slot-lint  {label}: json parse: {e}"; ok := false
    | .ok j =>
      match Tropical.Playground.compilePlanPure arena resolved j with
      | .error e => IO.println s!"  FAIL  dead-slot-lint  {label}: compile: {firstLine e}"; ok := false
      | .ok (plan, _, _) =>
        -- Byte-identity harness for refactor phases: TROPICAL_DUMP_PLANS=<dir>
        -- writes each canonical plan's wire form for before/after comparison
        -- (a refactor that promises plan identity proves it with `cmp`).
        if let some dir := (← IO.getEnv "TROPICAL_DUMP_PLANS") then
          if let .ok m := plan.toWire then
            IO.FS.writeFile s!"{dir}/{label}.json" m.compress
        let nParams := (plan.slotNames.filter ("param:".isPrefixOf ·)).size
        -- zero registered params means the patch didn't decode as intended —
        -- vacuous passes are the graceful-exclusion failure mode.
        if nParams == 0 then
          IO.println s!"  FAIL  dead-slot-lint  {label}: no param:* slots registered — the patch decoded to nothing"; ok := false
        checked := checked + nParams
        let dead := (unreadParamSlots plan).filter (fun s => !allow.contains s!"{label}:{s}")
        if !dead.isEmpty then
          deadAll := deadAll ++ dead.map (s!"{label}: {·}")
          ok := false
        -- the sflange fix, asserted structurally: a knob owned by a wired
        -- normal must not even REGISTER (absence, not just unreadness).
        if label == "sflange-wired" && plan.slotNames.contains "param:sfw.rate" then
          IO.println s!"  FAIL  dead-slot-lint  {label}: param:sfw.rate registered despite wired mod — the owned-knob rule regressed"
          ok := false
  IO.println s!"        {patches.size} canonical patches over the full node vocabulary, {checked} param:* slots:"
  IO.println s!"        result   unread param slots: {if deadAll.isEmpty then "none" else toString deadAll}"
  if ok then
    IO.println "  PASS  dead-slot-lint  every registered param:* slot is read by an instruction — no dead knobs behind graceful exclusion"; pure true
  else
    IO.println s!"  FAIL  dead-slot-lint  dead knobs (setSlot lands, no instruction reads): {deadAll}"; pure false

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

-- ── (h¹⁰) Stage differential: intern-time attribute vs the flow pass ──────────
-- Phase 1 of the typed stage-0 refactor. The typed side (StageSig at
-- `intern`, resolved along the partition recursion) must never classify an
-- instruction LATER than the plan-level flow pass (`Stage0.classify`) — the
-- flow pass is the trusted reference, and typed ⊑ flow means the attribute
-- is at least as precise everywhere and wrong nowhere the flow pass can
-- see. Strictly-earlier divergences are expected in exactly one category —
-- fold-valued wire crossings the flow pass can't see through (its slot
-- availability rule stops at the surviving writer) — and are reported, not
-- failed; Phase 2's byte-identical-audio differential is the semantic gate
-- on hoisting them.
/-- The v1 array/param/input conservatism overlay: the flow pass pins
    these to s1 at the *instruction* level, so the typed side must be
    compared under the same placement rule. -/
private def overlayS1 (i : Tropical.Plan.NInstr) : Bool :=
  i.tag == "ReduceBegin" || i.tag == "ReduceEnd"
  || (match i.dst with
    | .array _ => true | .sessionArray _ => true | _ => false)
  || i.args.any fun a => match a with
    | .arrayReg _ | .sessionArrayReg _ | .param _ _ | .input _ _ | .loopIdx => true
    | _ => false

private def runStageDifferential : IO Bool := do
  let entries ← (System.FilePath.mk "patches").readDir
  let names := (entries.filterMap fun e =>
    if e.fileName.endsWith ".json" then some e.fileName else none).qsort (· < ·)
  let mut ok := true
  let mut compared := 0
  let mut skipped := 0
  let mut unmapped := 0
  let mut divergent := 0
  for fn in names do
    match ← compilePatchStaged s!"patches/{fn}" with
    | .error _ => skipped := skipped + 1
    | .ok (plan, typedBlocks) =>
      let mut flowBlocks : Array (Array Tropical.Plan.NInstr) := #[]
      for f in plan.instanceFunctions do
        flowBlocks := flowBlocks ++ Tropical.Ir.Stage0.collectBlocks f
      if typedBlocks.size != flowBlocks.size then
        IO.println s!"  FAIL  stage-diff/{fn}  block count: typed {typedBlocks.size}, flow {flowBlocks.size}"
        ok := false
        continue
      let (flowStages, _) := Tropical.Ir.Stage0.classify plan
      let linear := flowBlocks.flatten
      let typedLinear := typedBlocks.flatten
      if typedLinear.size != linear.size || flowStages.size != linear.size then
        IO.println s!"  FAIL  stage-diff/{fn}  length: typed {typedLinear.size}, instrs {linear.size}, flow {flowStages.size}"
        ok := false
        continue
      for idx in [0:linear.size] do
        match typedLinear[idx]! with
        | none => unmapped := unmapped + 1
        | some t0 =>
          let t := if overlayS1 linear[idx]! then Tropical.Ir.Stage.s1 else t0
          let f := flowStages[idx]!
          if !(t.le f) then
            IO.println s!"  FAIL  stage-diff/{fn}  instr {idx}: typed {repr t} > flow {repr f} ({linear[idx]!.tag})"
            ok := false
          else if t != f then
            divergent := divergent + 1
          compared := compared + 1
  if ok then
    IO.println (s!"  PASS  stage-diff  typed ⊑ flow over {compared} instructions "
      ++ s!"({divergent} strictly earlier, {unmapped} unmapped"
      ++ s!"{if skipped > 0 then s!"; {skipped} non-session skipped" else ""})")
  pure ok

-- ── (h¹¹) Split equivalence: typed split ≡ flow split, in rendered bytes ──────
-- Phase 2's semantic gate. The typed split hoists strictly more (the
-- fold-crossing category), and every extra hoist must move NO output bit:
-- render each corpus patch through both splits and require byte equality.
private def runSplitEquiv : IO Bool := do
  let entries ← (System.FilePath.mk "patches").readDir
  let names := (entries.filterMap fun e =>
    if e.fileName.endsWith ".json" then some e.fileName else none).qsort (· < ·)
  let mut ok := true
  let mut matched := 0
  let mut skipped := 0
  for fn in names do
    match ← compilePatchStaged s!"patches/{fn}" with
    | .error _ => skipped := skipped + 1
    | .ok (plan, blocks) =>
      let typed ← renderTypedBytes plan blocks
      match ← renderIrBytes plan with
      | .error e =>
        IO.println s!"  FAIL  split-equiv/{fn}  flow render: {firstLine e}"; ok := false
      | .ok flow =>
        if typed == flow then matched := matched + 1
        else
          IO.println s!"  FAIL  split-equiv/{fn}  typed and flow renders differ"
          ok := false
  if ok then
    IO.println (s!"  PASS  split-equiv  typed split ≡ flow split byte-for-byte "
      ++ s!"({matched} patches{if skipped > 0 then s!"; {skipped} non-session skipped" else ""})")
  pure ok

-- The gate ledger below is one long do-block; its elaboration depth tracks the
-- gate count, and the default 512 is now too small.
set_option maxRecDepth 1024 in
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

  -- ── (c⁗) Reduce region: loop ≡ unrolled, frozen hash (banks slice 3a) ──────
  IO.println "reduce coverage (ReduceBegin/End ≡ unrolled, EmitLlvm):"
  total := total + 1
  if !(← runReduceCoverage) then failed := failed + 1

  -- ── (c⁗′) Region-aware Stage0: an all-s0 region hoists as a unit (WS3a) ────
  IO.println "banks region hoist (all-s0 reduce region → coefficient kernel):"
  total := total + 1
  if !(← runBanksRegionHoist) then failed := failed + 1

  -- ── (c′) C4: session → resolved root directly ≡ the elaborate round-trip ───
  IO.println "session via direct root (sessionToResolvedRoot ≡ sessionToParsed→elaborate):"
  total := total + 1
  if !(← runSessionViaArrowEquiv) then failed := failed + 1

  -- ── (c″) Stage differential: intern-time attribute ⊑ the flow pass ─────────
  IO.println "stage differential (typed StageSig vs Stage0 flow classification):"
  total := total + 1
  if !(← runStageDifferential) then failed := failed + 1

  -- ── (c‴) Split equivalence: typed split ≡ flow split, rendered bytes ───────
  IO.println "split equivalence (typed hoist ≡ flow hoist, byte-for-byte):"
  total := total + 1
  if !(← runSplitEquiv) then failed := failed + 1

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

  -- ── (f′) MSL emitter goldens: the Metal backend's codegen, text-frozen ─────
  IO.println "msl emitter (EmitMsl text-frozen + the f64 fold):"
  for (name, patchPath) in [
      ("pure-sine-440", "web/patches/pure-sine-440.json"),
      ("tz-flanger", "web/patches/tz-flanger.json"),
      ("reverse_reverb", "patches/reverse_reverb.json")] do
    total := total + 1
    if !(← runMslGolden writeMode name patchPath) then failed := failed + 1
  total := total + 1
  if !(← runMslFold) then failed := failed + 1

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
    -- The fixed-point DATAPATH sine (scope A): the Sig builder and the literate
    -- .md describe one algorithm, byte-identical through the same emit recipe.
    total := total + 1
    if !(← runEmitCorpusGate "FixedSin" "FixedSin" arena resolved
          Tropical.EmitArrow.buildFixedSin) then
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
    -- ── PER-LAW CARRIER TABLE (load-bearing — see design/fixed-carrier.md) ──
    -- Which value carrier each law rides is a CHOICE, not an accident: laws
    -- whose two sides evaluate the same ops on bit-equal clocks are exact on
    -- EITHER carrier (1-5 ride float to also witness the float path); law 6
    -- deliberately rides FLOAT as the documented reassociation exhibit (the
    -- ±δ tap swap moves bytes in float — denotational-only equality); law 7
    -- is the SAME law on the FIXED carrier where integer-add associativity
    -- makes it byte-exact; laws 8-10 mirror 1/4/5 on the fixed source.
    --   1 inverse ················ float   (clock-exact on either)
    --   2 additive ··············· float   (clock-exact on either)
    --   3 diagonal/fan-out ······· float   (clock-exact on either)
    --   4 reverse-involution ····· float   (clock-exact on either)
    --   5 reverse-swaps-delay ···· float   (clock-exact on either)
    --   6 reverse⋄flanger ········ FLOAT   (the reassociation exhibit: NOT byte)
    --   7 reverse⋄flanger ········ FIXED   (byte-exact — the scope-A property)
    --   8-10 single-source laws ·· FIXED   (byte-exact)
    -- The modal island (modal-bank/-reverse/-direction/-longtau gates above)
    -- rides the fixed datapath end to end as of scope-A B3.
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
    IO.println "fixed-point datapath sine (scope A — the sample values in i64):"
    total := total + 1
    if !(← runFixedSinAccuracy arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runFixedSinLongTau arena resolved) then
      failed := failed + 1
    IO.println "modal island (decaying-resonator bank as a term over the clock):"
    total := total + 1
    if !(← runModalBank arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksAsData arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksAsDataDir arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksFloat arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksFoldTrunk arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksColumnize arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksColumnizeBail arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalDegree arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalReverse arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runLongTauModal arena resolved) then
      failed := failed + 1
    IO.println "residue calculus (voice ⋙ reverb composed at build time):"
    total := total + 1
    if !(← runResidueMoments arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalReverb arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalDirection arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalSway arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runResidueDegenerate arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runResidueCollected arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runResidueSymbolic arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalPatch arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalLive arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksStaging arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runMslColumnGuard arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksBench arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksCount arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runBanksCountCache arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalFilter arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runModalAddr arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runRealizedReport) then
      failed := failed + 1
    total := total + 1
    if !(← runManifestDisciplines arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runVocabDriven arena resolved) then
      failed := failed + 1
    total := total + 1
    if !(← runDeadSlotLint arena resolved) then
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
