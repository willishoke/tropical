import Tropical.Tropicaltest.Harness

/-!
# Tropical.Tropicaltest.Synthetic

Synthetic-plan gates: the op-coverage kernel (the rare ops one sink funnels), the reduce-region loop-vs-unroll coverage, the golden-hash runners, and the let-serialization round-trip.
-/

open Tropical

-- ── Synthetic op-coverage plan ───────────────────────────────────────────────
-- Exercises ops the patch corpus doesn't reach (GreaterEq, NotEqual, Or,
-- BitOr, BitNot, FloorDiv, Sqrt, Floor, Ceil, Abs, ToInt/ToBool, Not), so a
-- typo in a predicate/intrinsic string in EmitLlvm is caught before the
-- one-way C++-codegen deletion. Built directly as a FlatPlan; compared
-- load_plan vs load_ir like the rest of section (d).
section OpCoverage
open Tropical.Plan

def jn (m : Int) (e : Nat := 0) : Lean.JsonNumber := { mantissa := m, exponent := e }
def cF (m : Int) (e : Nat := 0) : NOperand := .const (jn m e) .float
def cI (m : Int) : NOperand := .const (jn m) .int
def rgF (slot : Nat) : NOperand := .reg slot .float
def rgI (slot : Nat) : NOperand := .reg slot .int
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

def runReduceCoverage : IO Bool := do
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
      failGate "reduce-coverage" s!"EmitMsl: {firstLine e}"
    | .ok msl =>
      if (msl.splitOn "for (long rd").length >= 2 then
        passGate "reduce-coverage" s!"loop ≡ unrolled, hash {got.take 16}, MSL loop emitted"
      else
        failGate "reduce-coverage" "MSL kernel has no reduce loop"
  | .error e, _ => failGate "reduce-coverage" s!"loop: {firstLine e}"
  | _, .error e => failGate "reduce-coverage" s!"unrolled: {firstLine e}"

end ReduceCoverage

def sortedNames (dir : String) (suffix : String) : IO (Array String) := do
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
def runGolden (writeMode : Bool) (name patchPath goldenPath : String) : IO Bool := do
  match ← hashOf patchPath with
  | .error e => failGate s!"{name}" s!"compile: {firstLine e}"
  | .ok got =>
    if writeMode then
      IO.FS.writeFile goldenPath (got ++ "\n")
      IO.println s!"  WROTE {name}  {got.take 16}"; pure true
    else
      let expected := firstLine (← IO.FS.readFile goldenPath)
      if got == expected then passGate s!"{name}" s!"{got.take 16}"
      else failGate s!"{name}" s!"expected {expected.take 16} got {got.take 16}"

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
def runMslGolden (writeMode : Bool) (name patchPath : String) : IO Bool := do
  let goldenPath := s!"tests/golden/msl/{name}.metal"
  match ← emitMslOf patchPath with
  | .error e => failGate s!"msl-golden/{name}" s!"{firstLine e}"
  | .ok msl =>
    if writeMode then
      IO.FS.writeFile goldenPath msl
      IO.println s!"  WROTE msl-golden/{name}  ({msl.length}B)"; pure true
    else
      match ← (try (pure (some (← IO.FS.readFile goldenPath))) catch _ => pure none) with
      | none => failGate s!"msl-golden/{name}" s!"missing {goldenPath} (run --write)"
      | some expected =>
        if msl == expected then
          passGate s!"msl-golden/{name}" s!"({msl.length}B, text-frozen)"
        else
          failGate s!"msl-golden/{name}" s!"emitted MSL differs from frozen ({msl.length}B vs {expected.length}B)"

/-- THE FOLD gate: EmitMsl's emit-time f64 constant folding must land the
    LITERAL-frequency phase increment as the exact i64 the CPU computes —
    `toInt(440·2³²/44100)` evaluated here in f64, asserted present in the
    emitted text as a `long` literal. This is the byte-exact-phase claim
    for literal patches on the f32 GPU (design/fixed-carrier.md). -/
def runMslFold : IO Bool := do
  let expected : Int := Int.ofNat ((440.0 * 4294967296.0 / 44100.0).toUInt64.toNat)
  match ← emitMslOf "web/patches/pure-sine-440.json" with
  | .error e => failGate "msl-fold" s!"{firstLine e}"
  | .ok msl =>
    if (msl.splitOn s!"{expected}L").length > 1 then
      passGate "msl-fold" s!"literal landing folded in f64: increment {expected} present as i64 in the kernel"
    else
      failGate "msl-fold" s!"expected folded increment {expected}L not found in emitted MSL"

/-- A golden whose expected hash is supplied inline (migration fixtures, whose
    hash lives inside a JSON record — read-only). -/
def checkGoldenHash (name patchPath expected : String) : IO Bool := do
  match ← hashOf patchPath with
  | .error e => failGate s!"{name}" s!"compile: {firstLine e}"
  | .ok got =>
    if got == expected then passGate s!"{name}" s!"{got.take 16}"
    else failGate s!"{name}" s!"expected {expected.take 16} got {got.take 16}"

/-- Regression for `let`-binding serialization order. `Sin`'s `let` is
    order-dependent (`poly` uses `r2`, `sign` uses `odd_n`) and its binding
    names do not sort in declaration order ("poly" < "r2"). Surface-parse →
    encode (`toJson`) → re-parse → decode (`decodeProgram`) → elaborate must
    succeed; if `let` bind were serialized as a key-reordering object, the
    round-trip would scramble the bindings and elaboration would fail with
    "unknown name". -/
def runLetRoundtrip : IO Bool := do
  let md ← IO.FS.readFile "stdlib/Sin.md"
  match Tropical.Parse.Surface.parseMarkdownProgram md with
  | .error e => failGate "let-roundtrip" s!"parse: {firstLine e}"
  | .ok prog =>
    match Tropical.Parse.JsonV.parse prog.toJson.compress with
    | .error e => failGate "let-roundtrip" s!"reparse: {firstLine e}"
    | .ok jv =>
      match Tropical.Parse.decodeProgram jv with
      | .error e => failGate "let-roundtrip" s!"decode: {firstLine e}"
      | .ok prog2 =>
        match Tropical.Ir.elaborateInto {} prog2 (some fun _ => none) with
        | .error e => failGate "let-roundtrip" s!"elaborate: {firstLine e.message}"
        | .ok _ => passGate "let-roundtrip" "Sin survives encode→decode→elaborate"
