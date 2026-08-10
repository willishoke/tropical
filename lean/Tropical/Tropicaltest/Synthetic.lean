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

-- ── Static routed reduction (Plan 6) ───────────────────────────────────────
section RoutedSumCoverage
open Tropical.Plan

private def routedRoutes4 : Array (Option Nat) := #[
  some 0, none,   some 1,
  some 2, some 0, some 1,
  none,   some 2, some 0,
  some 1, some 2, none]

/-- Three mapped values for one authored item.  Keeping the arithmetic in the
    plan (instead of pre-folding literals here) makes the differential exercise
    the routed body's `loopIdx` binder and temporary lifetime. -/
private def routedMapped (k : NOperand) (t : Nat) : Array NInstr := #[
  instrScalar "ToFloat" t #[k] .float,
  instrScalar "Add" (t+1) #[rgF t, cF 1] .float,
  instrScalar "Mul" (t+2) #[rgF (t+1), cF 1 1] .float,
  instrScalar "Mul" (t+3) #[rgF (t+1), cF (-25) 1] .float]

private def routedPlan (capacity : Nat := 4)
    (count? : Option NOperand := none) : FlatPlan :=
  let routes := if capacity == 4 then routedRoutes4
    else Array.replicate (capacity * 3) (some 0)
  let body := #[instrRoutedSumBegin 0 capacity 3 routes count? 17]
    ++ routedMapped (.loopIdx 17) 0
    ++ #[instrRoutedSumYield 0 #[rgF 1, rgF 2, rgF 3],
         instrRoutedSumEnd 0,
         instrIndex 4 #[.arrayReg 0, cI 0] .float,
         instrIndex 5 #[.arrayReg 0, cI 1] .float,
         instrIndex 6 #[.arrayReg 0, cI 2] .float,
         instrWriteSlot 0 (rgF 4), instrWriteSlot 1 (rgF 5),
         instrWriteSlot 2 (rgF 6)]
  let inst := InstanceFunction.mk "root" "root" #[] body #[] 0 0 7 #[]
  { sampleRate := jn 44100, compilationMode := .fused,
    arraySlotNames := #["routed-output"], registerCount := 7,
    arraySlotCount := 1, arraySlotSizes := #[3], instanceFunctions := #[inst],
    sinks := #[{ inputs := #[0], gain := jn 1, target := 0 },
      { inputs := #[1], gain := jn 1, target := 1 },
      { inputs := #[2], gain := jn 1, target := 2 }],
    sources := defaultSources, slotCount := 4,
    slotNames := #["out:0", "out:1", "out:2", "param:count"],
    slotDefaults := #[Lean.Json.num (jn 0), Lean.Json.num (jn 0),
      Lean.Json.num (jn 0), Lean.Json.num (jn 3)] }

/-- Literal-index twin with the exact routed contract's fold order:
    item-major, then authored emit order within each item. -/
private def routedUnrolledPlan (trips : Nat) : FlatPlan := Id.run do
  let mut body : Array NInstr := #[]
  let mut acc : Array (Option Nat) := Array.replicate 3 none
  let mut next := 0
  for item in [0:trips] do
    body := body ++ routedMapped (cI (Int.ofNat item)) next
    let values := #[next+1, next+2, next+3]
    next := next + 4
    for emit in [0:3] do
      if let some output := routedRoutes4[item * 3 + emit]! then
        let lhs := match acc[output]! with
          | some t => rgF t
          | none => cF 0
        body := body.push (instrScalar "Add" next #[lhs, rgF values[emit]!] .float)
        acc := acc.set! output (some next)
        next := next + 1
  for output in [0:3] do
    let value := match acc[output]! with | some t => rgF t | none => cF 0
    body := body.push (instrWriteSlot output value)
  let inst := InstanceFunction.mk "root" "root" #[] body #[] 0 0 next #[]
  return ({ sampleRate := jn 44100, compilationMode := .fused,
            arraySlotNames := #[], registerCount := next, arraySlotCount := 0,
            arraySlotSizes := #[], instanceFunctions := #[inst],
            sinks := #[{ inputs := #[0], gain := jn 1, target := 0 },
              { inputs := #[1], gain := jn 1, target := 1 },
              { inputs := #[2], gain := jn 1, target := 2 }],
            sources := defaultSources, slotCount := 3,
            slotNames := #["out:0", "out:1", "out:2"],
            slotDefaults := #[Lean.Json.num (jn 0), Lean.Json.num (jn 0),
              Lean.Json.num (jn 0)] } : FlatPlan)

private def routedTags (plan : FlatPlan) : Array String :=
  plan.instanceFunctions.foldl (fun out f =>
    out ++ (f.instructions.filterMap fun i =>
      if i.tag.startsWith "RoutedSum" then some i.tag else none)) #[]

private def renderRoutedMsl (plan : FlatPlan) : IO (Except String ByteArray) := do
  try
    let rt ← Tropical.Ffi.Runtime.new BUFFER.toUInt32
    Tropical.StagedLoad.loadMsl rt plan
    let mut acc := ByteArray.empty
    -- One prepared worker tile is enough to validate the cooperative kernel.
    -- Draining many blocks in a tight offline loop would intentionally outrun
    -- the asynchronous refill owner and measure starvation silence instead.
    for _ in [0:1] do
      rt.process
      acc := acc ++ (← rt.outputBytes)
    pure (.ok acc)
  catch e => pure (.error e.toString)

private def routedDecodeF64 (bytes : ByteArray) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for sample in [0:bytes.size / 8] do
    let mut bits : UInt64 := 0
    for j in [0:8] do
      bits := bits * 256 + (bytes.get! (sample * 8 + (7 - j))).toUInt64
    out := out.push (Float.ofBits bits)
  return out

private def routedMetalUnavailable (e : String) : Bool :=
  e.endsWith
    "FlatRuntime: MSL source supplied but the engine was built without TROPICAL_METAL"

def runRoutedSumCoverage : IO Bool := do
  let static := routedPlan
  let dynamic := routedPlan 4 (some (.slot 3 .float))
  let large := routedPlan 64
  let staticBody := static.instanceFunctions[0]!.instructions
  let largeBody := large.instanceFunctions[0]!.instructions
  let compact := staticBody.size == largeBody.size && routedTags static ==
    #["RoutedSumBegin", "RoutedSumYield", "RoutedSumEnd"]
  let roundTrip := match static.toWire with
    | .error _ => false
    | .ok wire => match FlatPlan.ofWire wire with
      | .error _ => false
      | .ok decoded =>
        match decoded.toWire with
        | .error _ => false
        | .ok decodedWire =>
          decodedWire == wire && routedTags decoded == routedTags static
  let allS0 : Array (Option Tropical.Ir.Stage) :=
    Array.replicate staticBody.size (some .s0)
  let atomic := match Tropical.Ir.Stage0.hoistTyped static #[#[], allS0] with
    | .error _ => false
    | .ok split =>
      routedTags split.audio == routedTags static &&
        (match split.coeff? with | none => true | some p => routedTags p |>.isEmpty)
  let sourceSmoke := match Tropical.Ir.EmitLlvm.emitKernel static,
      Tropical.Ir.EmitMsl.emitKernel static with
    | .ok llvm, .ok msl =>
      (llvm.splitOn "@routed_routes_").length > 1 &&
        (msl.splitOn "threadgroup_position_in_grid").length > 1 &&
        (msl.splitOn "threadgroup_barrier").length >= 4 &&
        (msl.splitOn "routed_outputs").length == 1 &&
        (msl.splitOn "routed_record_0 / 3u < routed_trips").length == 1 &&
        (msl.splitOn "constant uint routed_csr_0").length > 1 &&
        (msl.splitOn "switch (rs").length == 1 &&
        (msl.splitOn s!"tropical.threadgroup_scratch_bytes={static.metalThreadgroupScratchBytes}").length > 1
    | _, _ => false
  match ← renderIrBytes static, ← renderIrBytes (routedUnrolledPlan 4),
      ← renderIrBytes dynamic, ← renderIrBytes (routedUnrolledPlan 3) with
  | .ok routedStatic, .ok unrolledStatic, .ok routedDynamic, .ok unrolledDynamic =>
    let coreOk := compact && roundTrip && atomic && sourceSmoke &&
      routedStatic == unrolledStatic && routedDynamic == unrolledDynamic
    if !coreOk then
      failGate "routed-sum-coverage" s!"compact={compact} roundTrip={roundTrip} atomic={atomic} source={sourceSmoke} staticEq={routedStatic == unrolledStatic} dynamicEq={routedDynamic == unrolledDynamic}"
    else
      match ← renderRoutedMsl static with
      | .ok metalStatic =>
        let cpu := routedDecodeF64 unrolledStatic
        let gpu := routedDecodeF64 metalStatic
        let metalError := (Array.range (min cpu.size gpu.size)).foldl
          (fun worst i => max worst (Float.abs (cpu[i]! - gpu[i]!))) 0.0
        if !gpu.isEmpty && gpu.size <= cpu.size && metalError < 1e-5 then
          passGate "routed-sum-coverage"
            s!"static+dynamic routed folds equal authored-order unrolling; cooperative Metal max error={metalError}; compact Plan 6 round-trips; Stage0 atomic"
        else
          failGate "routed-sum-coverage"
            s!"cooperative Metal returned samples={gpu.size}/{cpu.size} maxError={metalError}"
      | .error e =>
        if routedMetalUnavailable e then
          passGate "routed-sum-coverage"
            "static+dynamic routed folds equal authored-order unrolling; compact Plan 6 round-trips; Stage0 atomic; MSL source contract checked (Metal execution unavailable in this build)"
        else
          failGate "routed-sum-coverage" s!"cooperative Metal: {firstLine e}"
  | .error e, _, _, _ => failGate "routed-sum-coverage" s!"static routed: {firstLine e}"
  | _, .error e, _, _ => failGate "routed-sum-coverage" s!"static unrolled: {firstLine e}"
  | _, _, .error e, _ => failGate "routed-sum-coverage" s!"dynamic routed: {firstLine e}"
  | _, _, _, .error e => failGate "routed-sum-coverage" s!"dynamic unrolled: {firstLine e}"

end RoutedSumCoverage

-- ── The patch-bay refusal gate (elaborator retirement, phase 5) ──────────────
-- The wire is a patch bay: a tropical_program_2 file carrying a programDecl is
-- refused at ingest with the retirement message. (The transitional
-- opzoo-cutover byte-gate that compared the two registration paths died with
-- the programDecl ingest it compared against — its builder side lives on as
-- the `--fixtures` path the wasm≡JIT suite drives by name.)
section PatchBayRefusal

/-- A minimal programDecl-bearing patch file. -/
private def programDeclPatchJson : Lean.Json :=
  let assign := Lean.Json.mkObj [
    ("op", Lean.Json.str "outputAssign"), ("name", Lean.Json.str "out"),
    ("expr", Lean.Json.num 1)]
  let innerBody := Lean.Json.mkObj [
    ("op", Lean.Json.str "block"),
    ("decls", Lean.Json.arr #[]),
    ("assigns", Lean.Json.arr #[assign])]
  let inner := Lean.Json.mkObj [
    ("op", Lean.Json.str "program"), ("name", Lean.Json.str "P"),
    ("ports", Lean.Json.mkObj [("inputs", Lean.Json.arr #[]),
      ("outputs", Lean.Json.arr #[Lean.Json.str "out"])]),
    ("body", innerBody)]
  let progDecl := Lean.Json.mkObj [
    ("op", Lean.Json.str "programDecl"), ("name", Lean.Json.str "P"),
    ("program", inner)]
  Lean.Json.mkObj [
    ("schema", Lean.Json.str "tropical_program_2"),
    ("name", Lean.Json.str "decl_probe"),
    ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
      ("decls", Lean.Json.arr #[progDecl])])]

def runPatchBayRefusal : IO Bool := do
  let path := "/tmp/tropicaltest-programdecl-probe.json"
  IO.FS.writeFile path programDeclPatchJson.compress
  match ← compilePatch path .fused with
  | .ok _ =>
    failGate "patch-bay-refusal"
      "programDecl-bearing file LOADED — program definitions over the wire must be refused"
  | .error e =>
    if (e.splitOn "program definitions over the wire are retired").length > 1 then
      passGate "patch-bay-refusal" s!"programDecl refused at ingest: {firstLine e}"
    else
      failGate "patch-bay-refusal" s!"programDecl failed with the WRONG error (want the retirement message): {firstLine e}"

end PatchBayRefusal

-- ── The CF-only cycle tripwire gate ──────────────────────────────────────────
-- Wire cycles are unbuildable through the IR, but the session mirror can spell
-- one in a patch file; the compile must refuse it at `assertSessionAcyclic`
-- (the shared total detector, `Ir.findCycle`) with a nameable loop.
section CycleRefusal

private def instDecl (name : String) (inputs : Array (String × Lean.Json)) : Lean.Json :=
  Lean.Json.mkObj [
    ("op", Lean.Json.str "instanceDecl"), ("name", Lean.Json.str name),
    ("program", Lean.Json.str "SoftClip"),
    ("inputs", Lean.Json.mkObj inputs.toList)]

private def refTo (inst : String) : Lean.Json :=
  Lean.Json.mkObj [
    ("op", Lean.Json.str "ref"), ("instance", Lean.Json.str inst),
    ("output", Lean.Json.str "out")]

private def cyclePatchJson (decls : Array Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [
    ("schema", Lean.Json.str "tropical_program_2"),
    ("name", Lean.Json.str "cycle_probe"),
    ("body", Lean.Json.mkObj [("op", Lean.Json.str "block"),
      ("decls", Lean.Json.arr decls),
      ("assigns", Lean.Json.arr #[Lean.Json.mkObj [
        ("op", Lean.Json.str "outputAssign"), ("name", Lean.Json.str "dac.out"),
        ("expr", refTo "a")]])])]

private def expectCycleRefusal (label : String) (path : String)
    (decls : Array Lean.Json) : IO Bool := do
  IO.FS.writeFile path (cyclePatchJson decls).compress
  match ← compilePatch path .fused with
  | .ok _ =>
    failGate s!"cycle-refusal/{label}" "cyclic session COMPILED — CF-only tripwire is dead"
  | .error e =>
    if (e.splitOn "CF-only — inter-instance cycles are not allowed").length > 1 then
      passGate s!"cycle-refusal/{label}" (firstLine e)
    else
      failGate s!"cycle-refusal/{label}"
        s!"cyclic session refused with the WRONG error (want the CF-only message): {firstLine e}"

def runCycleRefusal : IO Bool := do
  let selfLoop ← expectCycleRefusal "self" "/tmp/tropicaltest-cycle-self.json"
    #[instDecl "a" #[("input", refTo "a")]]
  let pair ← expectCycleRefusal "pair" "/tmp/tropicaltest-cycle-pair.json"
    #[instDecl "a" #[("input", refTo "b")],
      instDecl "b" #[("input", refTo "a")]]
  pure (selfLoop && pair)

end CycleRefusal

-- ── Production legacy-state non-emission ───────────────────────────────────
-- The current source/IR types have no persistent state constructor. NInstr's
-- tag is intentionally open for backend structural operations, though, so this
-- gate also checks representative production artifacts in hand. Wire text is
-- consulted only for the schema tag; legacy-state absence is checked on typed
-- plans before serialization.
section ProductionNonEmission

open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

private def legacyStateTags : Array String := #[
  "Register", "StateReg", "StateLoad", "StateStore", "Update", "NextUpdate",
  "Delay", "DelayInit", "StateInit", "Writeback", "SmoothParam"]

private def checkInstrBlock (path : String) (instrs : Array NInstr) :
    Except String Unit := do
  for i in instrs do
    if legacyStateTags.contains i.tag then
      throw s!"{path}: production plan emitted legacy state instruction '{i.tag}'"

private def checkInstanceFunction (path : String)
    (f : InstanceFunction) : Except String Unit := do
  checkInstrBlock s!"{path}/preamble" f.preambleInstructions
  checkInstrBlock s!"{path}/pre-input" f.preInputInstructions
  checkInstrBlock s!"{path}/body" f.instructions
  let _ ← f.children.attach.mapM fun c =>
    checkInstanceFunction s!"{path}/{c.1.instanceName}" c.1
  pure ()
termination_by sizeOf f
decreasing_by
  exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children c.2

private def checkFlatPlan (path : String) (plan : FlatPlan) :
    Except String Unit := do
  for f in plan.instanceFunctions do
    checkInstanceFunction s!"{path}/{f.instanceName}" f
  let wire ← plan.toWire
  match wire.getObjVal? "schema" with
  | .ok (.str "tropical_plan_6") => pure ()
  | .ok got =>
    throw s!"{path}: production plan schema is {got.compress}, expected tropical_plan_6"
  | .error e => throw s!"{path}: production plan has no schema: {e}"

private def checkPerInstancePlan (path : String) (plan : PerInstancePlan) :
    Except String Unit := do
  checkInstrBlock s!"{path}/body" plan.instructions
  for (block, i) in plan.perChildPreInput.zipIdx do
    checkInstrBlock s!"{path}/child-{i}" block

private def playgroundProbe : Lean.Json :=
  let source := Lean.Json.mkObj [
    ("id", .str "source"), ("kind", .str "source"),
    ("params", Lean.Json.mkObj [("freq", .num (jn 440))])]
  let output := Lean.Json.mkObj [
    ("id", .str "out"), ("kind", .str "out"),
    ("in", Lean.Json.mkObj [("in", .arr #[.str "source"])])]
  Lean.Json.mkObj [
    ("nodes", .arr #[source, output]),
    ("out", .str "out")]

private def compileSessionProbe (path : String) :
    IO (Except String FlatPlan) := do
  let env ← Tropical.Engine.boot
  let action : Tropical.EngineM FlatPlan := do
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", .str path)])
    Tropical.Engine.compileMirrorFlatPlan env .fused
  match ← action.run with
  | .ok plan => pure (.ok plan)
  | .error failure => pure (.error failure.toJson.compress)

private def compileExportProbe : IO (Except String FlatPlan) := do
  let env ← Tropical.Engine.boot
  let action : Tropical.EngineM FlatPlan := do
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [
      ("path", .str "web/patches/pure-sine-440.json")])
    let _ ← Tropical.Engine.handleExportProgram env (Lean.Json.mkObj [
      ("name", .str "NonEmissionExport"),
      ("outputs", Lean.Json.mkObj [
        ("out", Lean.Json.mkObj [
          ("instance", .str "osc"), ("output", .str "sine")])])])
    let _ ← Tropical.Engine.handleAddInstance env (Lean.Json.mkObj [
      ("program", .str "NonEmissionExport"),
      ("instance_name", .str "export_probe")])
    Tropical.Engine.compileMirrorFlatPlan env .fused
  match ← action.run with
  | .ok plan => pure (.ok plan)
  | .error failure => pure (.error failure.toJson.compress)

/-- Representative production paths must remain unable to emit legacy state:
    a stdlib builder's per-instance plan, a playground graph, a program_2/MCP
    session, and an exported composite re-instantiated into that session. -/
def runProductionNonEmission (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mut failures : Array String := #[]

  match resolved.find? (·.1 == "FixedSinOsc") with
  | none => failures := failures.push "builder: FixedSinOsc missing from stdlib"
  | some (_, idx) =>
    match (Tropical.Ir.Strata.runResolved
        { inlineNested := true } arena idx).mapError (·.message) with
    | .error e => failures := failures.push s!"builder: lower: {firstLine e}"
    | .ok (coreArena, core) =>
      match Tropical.Ir.CompileResolved.compileResolved core coreArena with
      | .error e => failures := failures.push s!"builder: compile: {firstLine e}"
      | .ok plan =>
        if let .error e := checkPerInstancePlan "builder/FixedSinOsc" plan then
          failures := failures.push e

  match Tropical.Playground.compilePlanPure arena resolved playgroundProbe with
  | .error e => failures := failures.push s!"playground: {firstLine e}"
  | .ok compiled =>
    if let .error e := checkFlatPlan "playground" compiled.plan then
      failures := failures.push e

  match ← compileSessionProbe "web/patches/pure-sine-440.json" with
  | .error e => failures := failures.push s!"session/program_2: {firstLine e}"
  | .ok plan =>
    if let .error e := checkFlatPlan "session/program_2" plan then
      failures := failures.push e

  match ← compileExportProbe with
  | .error e => failures := failures.push s!"export: {firstLine e}"
  | .ok plan =>
    if let .error e := checkFlatPlan "export" plan then
      failures := failures.push e

  if failures.isEmpty then
    passGate "production-non-emission"
      "builder + playground + program_2/session + export emit typed state-free plan_6 only"
  else
    failGate "production-non-emission" (String.intercalate "; " failures.toList)

end ProductionNonEmission

-- ── Serialized-plan schema boundary ────────────────────────────────────────

/-- The Lean-owned serialized-plan entry point is deliberately Plan-6-only.
    This is a separate boundary gate from production non-emission: even a
    hand-authored document cannot revive an older schema or retired carrier. -/
def runPlanSchemaRejection : IO Bool := do
  let rejectsWith (j : Lean.Json) (needle : String) : Bool :=
    match Tropical.Plan.FlatPlan.ofWire j with
    | .ok _ => false
    | .error e => (e.splitOn needle).length > 1
  let plan4 := Lean.Json.mkObj [("schema", .str "tropical_plan_4")]
  let plan5 := Lean.Json.mkObj [("schema", .str "tropical_plan_5")]
  let unknown := Lean.Json.mkObj [("schema", .str "tropical_plan_99")]
  let missing := Lean.Json.mkObj []
  let retired := Lean.Json.mkObj [
    ("schema", .str "tropical_plan_6"),
    ("state_init", .arr #[])]
  if rejectsWith plan4
        "unsupported schema 'tropical_plan_4'; expected 'tropical_plan_6'"
      && rejectsWith plan5
        "unsupported schema 'tropical_plan_5'; expected 'tropical_plan_6'"
      && rejectsWith unknown
        "unsupported schema 'tropical_plan_99'; expected 'tropical_plan_6'"
      && rejectsWith missing "missing string field 'schema'"
      && rejectsWith retired
        "retired field 'state_init' is not valid in tropical_plan_6" then
    passGate "plan6-schema-rejection"
      "PlanDecode rejects plan_4, plan_5, unknown/missing schemas, and retired carriers"
  else
    failGate "plan6-schema-rejection"
      "PlanDecode accepted an unsupported schema/carrier or returned the wrong boundary error"

def sortedNames (dir : String) (suffix : String) : IO (Array String) := do
  let entries ← (System.FilePath.mk dir).readDir
  let names := entries.filterMap fun e =>
    if e.fileName.endsWith suffix then some (e.fileName.dropEnd suffix.length).toString else none
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
  | .ok compiled => pure (.ok (compiled.plan, compiled.stageBlocks))
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

-- ── The device-boundary early-warning gate (the safety class) ────────────────

/-- Peak |sample| of a little-endian f64 render, and whether every sample is
    finite. Folds the bytes in place — no intermediate `Array Float`. -/
private def peakAbsLE (b : ByteArray) : Float × Bool := Id.run do
  let n := b.size / 8
  let mut peak : Float := 0.0
  let mut finite := true
  for i in [0:n] do
    let mut u : UInt64 := 0
    for j in [0:8] do
      u := u * 256 + (b.get! (i * 8 + (7 - j))).toUInt64
    let x := (Float.ofBits u).abs
    if !x.isFinite then finite := false
    else if x > peak then peak := x
  pure (peak, finite)

/-- THE STANDING BOUNDED-OUTPUT ASSERTION — the early-warning half of the
    safety-class gate landed 2026-07-20.

    Every audio patch in the corpus renders with `max |out| ≤ C` and every
    sample finite, where `C` is the device-boundary bound
    (`kDeviceOutputBound`, engine/dac/TropicalDAC.hpp).

    It is trivially green today (the corpus peaks at 1.807 against C = 256), and
    that is the point: it exists to catch the NEXT rail before a speaker does.
    Note what the headroom number does NOT mean: C is set by the modal
    vocabulary's legitimate reach (≈ Q·master_gain ≈ 163 at the top of the
    resonance knob), not by this corpus, so a large ratio here is a statement
    about the corpus being quiet, not about the margin being safe.
    It matters MORE than it would have under the rejected design. The clamp
    deliberately does not live in the emitted kernel — the kernel's output is
    the value of `f(τ)`, read as a number by `render-bytes`, the goldens, the
    wasm≡JIT differential and the numeric coverage gates, and bounding it there
    would make the compiler lie (`bootstrap-exp` legitimately renders exp(10) ≈
    22026 through the sink). So nothing upstream of the DAC is bounded by
    construction, and THIS gate is what notices. Had it existed before the i64
    modal-datapath rail incident (peaks of 63.3,
    `design/modal-datapath-rail.local.md`) it would have caught it in CI.

    Deliberately scoped to AUDIO patches — the synthetic numeric fixtures
    (`op-coverage`, `reduce-coverage`, `bootstrap-exp`) use the output buffer as
    a general readout and legitimately exceed C. Bounding *them* is what the
    rejected design got wrong. -/
def runDeviceBound : IO Bool := do
  let bound : Float := 256.0   -- must equal kDeviceOutputBound (TropicalDAC.hpp)
  let corpus := [
    "patches/reverse_reverb.json", "patches/scrub_reverb.json",
    "web/patches/pure-sine-440.json", "web/patches/ring-mod.json",
    "web/patches/tz-flanger.json"]
  let mut worst : Float := 0.0
  let mut worstName := ""
  let mut bad : Array String := #[]
  for path in corpus do
    if ← System.FilePath.pathExists path then
      match ← compilePatchStaged path with
      | .error e => bad := bad.push s!"{path}: compile {firstLine e}"
      | .ok (plan, blocks) =>
        let (peak, finite) := peakAbsLE (← renderTypedBytes plan blocks)
        if !finite then bad := bad.push s!"{path}: NON-FINITE sample"
        else if peak > bound then bad := bad.push s!"{path}: peak {peak} > C={bound}"
        if peak > worst then worst := peak; worstName := path
  IO.println s!"        device bound C = {bound}; worst corpus peak {worst} ({worstName})"
  if bad.isEmpty then
    passGate "device-bound" s!"every audio patch renders bounded and finite (worst {worst} ≤ C = {bound}, {(bound / worst)}× headroom) — the early warning for the next rail"
  else
    failGate "device-bound" s!"unbounded render(s): {String.intercalate "; " bad.toList}"

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

/-- A migration audio golden (`tests/golden/migration/<fixture>.json`): render
    the fixture's `flat_plan` `input` and hash the output. Under `--write`,
    rewrite the JSON with the fresh hash and a 32-sample peek; else compare the
    stored hash. Only `hash` is load-bearing — `first_samples` is a human peek.
    Regenerable like every other golden, so a deliberate level change (e.g. the
    unity-sink re-freeze) re-baselines here too instead of needing a hand-edit. -/
def runMigrationGolden (writeMode : Bool) (fixture : String) : IO Bool := do
  let goldenPath := s!"tests/golden/migration/{fixture}.json"
  let fixText ← IO.FS.readFile s!"tests/fixtures/flat_plan/{fixture}.json"
  let input? : Option Lean.Json := do
    let f ← (Lean.Json.parse fixText).toOption
    (f.getObjVal? "input").toOption
  match input? with
  | none => IO.println s!"  SKIP  {fixture}  (missing input)"; pure true
  | some input =>
    let tmpPatch := "/tmp/tropicaltest-fixture.json"
    IO.FS.writeFile tmpPatch input.compress
    match ← compilePatchStaged tmpPatch with
    | .error e => failGate s!"{fixture}" s!"compile: {firstLine e}"
    | .ok (plan, blocks) =>
      let bytes ← renderTypedBytes plan blocks
      let got ← sha256Hex bytes
      if writeMode then
        -- first 32 f64 samples (little-endian) — a documentary peek, not checked
        let nPeek := min 32 (bytes.size / 8)
        let mut peek : Array String := #[]
        for i in [0:nPeek] do
          let mut u : UInt64 := 0
          for j in [0:8] do
            u := u * 256 + (bytes.get! (i * 8 + (7 - j))).toUInt64
          peek := peek.push (Float.ofBits u).toString
        let samples := String.intercalate ",\n    " peek.toList
        let text := "{\n"
          ++ s!"  \"fixture\": \"{fixture}.json\",\n"
          ++ s!"  \"sample_count\": {bytes.size / 8},\n"
          ++ s!"  \"hash\": \"{got}\",\n"
          ++ s!"  \"first_samples\": [\n    {samples}\n  ]\n"
          ++ "}\n"
        IO.FS.writeFile goldenPath text
        IO.println s!"  WROTE {fixture}  {got.take 16}"; pure true
      else
        let goldenText ← IO.FS.readFile goldenPath
        let expected? : Option String := do
          let g ← (Lean.Json.parse goldenText).toOption
          (← (g.getObjVal? "hash").toOption).getStr?.toOption
        match expected? with
        | none => failGate s!"{fixture}" "golden missing hash field"
        | some expected =>
          if got == expected then passGate s!"{fixture}" s!"{got.take 16}"
          else failGate s!"{fixture}" s!"expected {expected.take 16} got {got.take 16}"
