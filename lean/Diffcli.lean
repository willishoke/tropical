import Tropical.Ffi
import Tropical.Ir.Codec
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.CompileResolved
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.StagedLoad
import Tropical.PlanDecode
import Tropical.Engine
import Tropical.EmitArrow
import Tropical.Stdlib
import Tropical.Testing.ArrowFixtures
import Tropical.Testing.EngineMirror
import Tropical.Testing.PlanWire

/-!
The `diffcli` executable — build, inspection, render, and backend-qualification
verbs over the Lean-owned compiler and native runtime.

    diffcli render-bytes <plan.json> [--frames N] [--buffer N]

Loads a tropical_plan_6 JSON file into a fresh runtime, renders
`frames × buffer` samples, and writes the raw little-endian float64
stream to stdout. `… | shasum -a 256` reproduces the golden hashes in
tests/golden/ with the standard frame and buffer counts.

The compile/render verbs — `compile`, `compile-wasm`, `render-bytes`,
`render-metal`, `render-graph`, `emit-ir`, `emit-msl` — boot the engine
(the stdlib is the `Tropical.Stdlib` arrow builders) and print the plan,
LLVM/MSL IR, or rendered bytes for a patch. The former surface/bridge
verbs (parse-md, parse-all, elab-stdlib, elab-file, strata-stdlib,
strata-file, emit-stdlib, emit-file, emitarrow-*, parsed-roundtrip,
voice-desugar, raise) were retired with the literate `.md` language,
the parse bridge, and finally the elaborator (the `raise` verb printed
the ParsedProgram, which no longer exists).
-/

def parseNatFlag (args : List String) (flag : String) (default : Nat) : Nat :=
  match args.idxOf? flag with
  | some i => match args[i+1]? with
    | some v => v.toNat?.getD default
    | none => default
  | none => default

def renderBytes (args : List String) : IO UInt32 := do
  let some planPath := args.find? (fun a => !a.startsWith "--" && a.endsWith ".json")
    | IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N] [--start S]"
      return 1
  let frames := parseNatFlag args "--frames" 16
  let buffer := parseNatFlag args "--buffer" 256
  let start := parseNatFlag args "--start" 0
  let planJson ← IO.FS.readFile planPath
  -- Lean owns codegen: parse the plan, stage-0 split + emit IR, load via
  -- load_ir_staged. There is no C++ plan compiler.
  let plan ← match Lean.Json.parse planJson with
    | .error e => IO.eprintln s!"render-bytes: parse: {e}"; return 1
    | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
      | .error e => IO.eprintln s!"render-bytes: ofWire: {e}"; return 1
      | .ok p => pure p
  let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
  Tropical.StagedLoad.load rt plan
  if start != 0 then rt.setSampleIndex start.toUInt64
  let stdout ← IO.getStdout
  for _ in [0:frames] do
    rt.process
    stdout.write (← rt.outputBytes)
  stdout.flush
  return 0

/-- `diffcli render-metal <plan.json>` — render through the METAL backend
    (EmitMsl → load_ir_msl → off-RT worker tiles), same byte protocol as
    `render-bytes`. The f64 JIT is dual-loaded but the output comes from the
    f32 GPU kernel — this is the device-under-test side of `metal_vs_jit`.
    Requires libtropical built with TROPICAL_METAL. The render tool may wait
    outside the callback so an offline tight loop cannot outrun the worker. -/
def renderMetal (args : List String) : IO UInt32 := do
  let some planPath := args.find? (fun a => !a.startsWith "--" && a.endsWith ".json")
    | IO.eprintln "usage: diffcli render-metal <plan.json> [--frames N] [--buffer N] [--start S]"
      return 1
  let frames := parseNatFlag args "--frames" 16
  let buffer := parseNatFlag args "--buffer" 256
  let start := parseNatFlag args "--start" 0
  let planJson ← IO.FS.readFile planPath
  let plan ← match Lean.Json.parse planJson with
    | .error e => IO.eprintln s!"render-metal: parse: {e}"; return 1
    | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
      | .error e => IO.eprintln s!"render-metal: ofWire: {e}"; return 1
      | .ok p => pure p
  let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
  Tropical.StagedLoad.loadMsl rt plan
  if start != 0 then rt.setSampleIndex start.toUInt64
  let stdout ← IO.getStdout
  for _ in [0:frames] do
    rt.processOffline
    stdout.write (← rt.outputBytes)
  stdout.flush
  return 0

/-- `diffcli render-graph <graph.json> [--metal] [--frames N] [--buffer N]
    [--start S]` — compile a playground PatchGraph (`{"nodes":[…],"out":…}`)
    through the TYPED session path (`Playground.compilePlan` → typed
    stage-0 split, so banked coefficient columns hoist to the coefficient
    kernel) and render, JIT (default) or Metal (`--metal`). Same byte
    protocol as `render-bytes`. This is the device side of the banked
    `metal_vs_jit` gate: unlike `render-metal` (flow split — arrays pinned
    per-sample, no columns), the typed split here exercises the
    `coeff_columns` GPU crossing exactly as a live session on
    `TROPICAL_BACKEND=metal` does. Prints `hoisted columns=N` on stderr so
    the harness can assert the crossing is actually exercised. -/
def renderGraph (args : List String) : IO UInt32 := do
  let some graphPath := args.find? (fun a => !a.startsWith "--" && a.endsWith ".json")
    | IO.eprintln "usage: diffcli render-graph <graph.json> [--metal] [--frames N] [--buffer N] [--start S]"
      return 1
  let frames := parseNatFlag args "--frames" 16
  let buffer := parseNatFlag args "--buffer" 256
  let start := parseNatFlag args "--start" 0
  let metal := args.contains "--metal"
  let text ← IO.FS.readFile graphPath
  let j ← match Lean.Json.parse text with
    | .error e => IO.eprintln s!"render-graph: parse: {e}"; return 1
    | .ok j => pure j
  match ← Tropical.Playground.compilePlan j with
  | .error e => IO.eprintln s!"render-graph: compile: {e}"; return 1
  | .ok compiled =>
    let split ← Tropical.StagedLoad.splitTyped compiled.plan compiled.stageBlocks
    IO.eprintln s!"render-graph: hoisted columns={split.audio.coeffArraySlots.size}"
    let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
    if metal then Tropical.StagedLoad.loadMslTyped rt compiled.plan compiled.stageBlocks
    else Tropical.StagedLoad.loadTyped rt compiled.plan compiled.stageBlocks
    if start != 0 then rt.setSampleIndex start.toUInt64
    let stdout ← IO.getStdout
    for _ in [0:frames] do
      if metal then rt.processOffline else rt.process
      stdout.write (← rt.outputBytes)
    stdout.flush
    return 0

private def parseStrFlag (args : List String) (flag : String) : Option String :=
  args.findSome? fun a =>
    if a.startsWith (flag ++ "=") then some (a.drop (flag.length + 1)).toString else none

private def parseFloatFlag (args : List String) (flag : String)
    (default : Float) : Float :=
  match parseStrFlag args flag with
  | some v => match Lean.Json.parse v with
    | .ok j => (j.getNum?.toOption.map (·.toFloat)).getD default
    | .error _ => default
  | none => default

/-- `diffcli render-sweep <graph.json> --param=<knob> [--center=800]
    [--octaves=2] [--rate-hz=2] [--frames N] [--buffer N]` — render a
    playground graph while the CONTROL PLANE sweeps one knob:
    `value(t) = center · 2^(octaves · sin(2π · rate · t))`, written to the
    knob's glide slots (`#v0`/`#v1`, or the bare `param:` slot) once per
    process call, at the block-start time. Per-sample automation is
    `--buffer 1` — the update granularity IS the buffer, which is the point:
    the same verb at two buffer sizes prices coefficient-update quantization
    (the settle tradeoff) with no other variable moving. Byte protocol as
    `render-bytes`. Rate is the engine default 44100. -/
def renderSweep (args : List String) : IO UInt32 := do
  let some graphPath := args.find? (fun a => !a.startsWith "--" && a.endsWith ".json")
    | IO.eprintln "usage: diffcli render-sweep <graph.json> --param=<knob> [--center=C] [--octaves=O] [--rate-hz=R] [--frames N] [--buffer N]"
      return 1
  let some knob := parseStrFlag args "--param"
    | IO.eprintln "render-sweep: --param=<knob> is required (e.g. --param=flt.cutoff)"
      return 1
  let frames := parseNatFlag args "--frames" 16
  let buffer := parseNatFlag args "--buffer" 256
  let center := parseFloatFlag args "--center" 800.0
  let octaves := parseFloatFlag args "--octaves" 2.0
  let rateHz := parseFloatFlag args "--rate-hz" 2.0
  let text ← IO.FS.readFile graphPath
  let j ← match Lean.Json.parse text with
    | .error e => IO.eprintln s!"render-sweep: parse: {e}"; return 1
    | .ok j => pure j
  match ← Tropical.Playground.compilePlan j with
  | .error e => IO.eprintln s!"render-sweep: compile: {e}"; return 1
  | .ok compiled =>
    let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
    Tropical.StagedLoad.loadTyped rt compiled.plan compiled.stageBlocks
    let v0? ← rt.slotIndex? s!"param:{knob}#v0"
    let v1? ← rt.slotIndex? s!"param:{knob}#v1"
    let bare? ← rt.slotIndex? s!"param:{knob}"
    let slots := #[v0?, v1?, bare?].filterMap id
    if slots.isEmpty then
      IO.eprintln s!"render-sweep: no slot found for knob '{knob}'"
      return 1
    let sampleRate := 44100.0
    let twoPi := 6.283185307179586
    let stdout ← IO.getStdout
    let mut framesDone : Nat := 0
    for _ in [0:frames] do
      let t := framesDone.toFloat / sampleRate
      let value := center * Float.exp2 (octaves * Float.sin (twoPi * rateHz * t))
      for slot in slots do
        rt.setSlot slot value
      rt.process
      stdout.write (← rt.outputBytes)
      framesDone := framesDone + buffer
    stdout.flush
    return 0

-- ── compile (Phase 6 stage 6d — the compile_patch.ts contract) ──────────────

/-- `diffcli compile <patch.json> [--mode=<m>]` → plan JSON on stdout.
    Boots the engine (stdlib from the pre-parsed bridge), loads the
    patch through the engine's own ingest, and rebuilds the plan from
    the mirror at the requested mode. The side-B command of
    diff_plan.ts / diff_audio.ts. -/
def compileVerb (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--")
    | IO.eprintln "usage: diffcli compile <patch.json> [--mode=fused|microkernel|microkernel-deep] [--fixtures]"
      return 1
  let modeStr := (parseStrFlag args "--mode").getD "fused"
  let some mode := Tropical.Plan.CompilationMode.ofWire? modeStr
    | IO.eprintln s!"unknown compilation mode: {modeStr}"
      return 1
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM String := do
    if args.contains "--fixtures" then Tropical.Engine.registerTestFixtures env
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str patch)])
    Tropical.Engine.compileMirrorPlan env mode
  match ← act.run with
  | .ok planJson =>
    IO.println planJson
    return 0
  | .error f =>
    IO.eprintln f.toJson.compress
    return 1

/-- Compile a patch to an in-memory FlatPlan (the shape both the plan path
    and the IR path consume). `fixtures` additionally registers the
    test-fixture programs (`OpZoo`) so equivalence patches can instantiate
    them by name. -/
private def compileToFlatPlan (patch : String) (fixtures : Bool := false) :
    IO (Except String Tropical.Plan.FlatPlan) := do
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM Tropical.Plan.FlatPlan := do
    if fixtures then Tropical.Engine.registerTestFixtures env
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str patch)])
    Tropical.Engine.compileMirrorFlatPlan env .fused
  match ← act.run with
  | .ok p => pure (.ok p)
  | .error f => pure (.error f.toJson.compress)

/-- `diffcli emit-ir <patch.json>` → the Lean-emitted LLVM IR on stdout. -/
def emitIrVerb (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--")
    | IO.eprintln "usage: diffcli emit-ir <patch.json>"; return 1
  match ← compileToFlatPlan patch with
  | .error e => IO.eprintln e; return 1
  | .ok plan =>
    match Tropical.Ir.EmitLlvm.emitKernel plan with
    | .error e => IO.eprintln s!"emitKernel: {e}"; return 1
    | .ok ir => IO.println ir; return 0

/-- `diffcli emit-msl <patch.json>` → the Lean-emitted Metal Shading
    Language kernel on stdout (the Metal backend's codegen; sibling of
    `emit-ir`). Sanity compile-check without any engine:
    `diffcli emit-msl p.json | xcrun -sdk macosx metal -x metal -c -o /dev/null -`. -/
def emitMslVerb (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--")
    | IO.eprintln "usage: diffcli emit-msl <patch.json>"; return 1
  match ← compileToFlatPlan patch with
  | .error e => IO.eprintln e; return 1
  | .ok plan =>
    match Tropical.Ir.EmitMsl.emitKernel plan with
    | .error e => IO.eprintln s!"emitKernel (msl): {e}"; return 1
    | .ok msl => IO.println msl; return 0

/-- `diffcli compile-wasm <patch.json> --out <out.wasm>` → a complete wasm32
    module, emitted in-process (Lean IR → engine LLVM+lld, no subprocess). The
    plan_6 JSON from `diffcli compile` serves as the browser-side manifest. -/
def compileWasmVerb (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--" && a.endsWith ".json")
    | IO.eprintln "usage: diffcli compile-wasm <patch.json> --out <out.wasm> [--fixtures]"; return 1
  let some outPath := parseStrFlag args "--out"
    | IO.eprintln "compile-wasm: --out <path> required"; return 1
  match ← compileToFlatPlan patch (fixtures := args.contains "--fixtures") with
  | .error e => IO.eprintln e; return 1
  | .ok plan =>
    match Tropical.Ir.EmitLlvm.emitKernel plan with
    | .error e => IO.eprintln s!"emitKernel: {e}"; return 1
    | .ok ir =>
      let wasm ← Tropical.Ffi.compileIrToWasm ir
      IO.FS.writeBinFile (System.FilePath.mk outPath) wasm
      IO.eprintln s!"compile-wasm: wrote {wasm.size} bytes → {outPath}"
      return 0

def main (args : List String) : IO UInt32 := do
  match args with
  | "render-bytes" :: rest => renderBytes rest
  | "render-metal" :: rest => renderMetal rest
  | "render-graph" :: rest => renderGraph rest
  | "render-sweep" :: rest => renderSweep rest
  | "emit-ir" :: rest => emitIrVerb rest
  | "emit-msl" :: rest => emitMslVerb rest
  | "compile-wasm" :: rest => compileWasmVerb rest
  | "compile" :: rest => compileVerb rest
  | _ =>
    IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N]\n       diffcli render-metal <plan.json>\n       diffcli render-graph <graph.json>\n       diffcli emit-ir <patch.json>\n       diffcli emit-msl <patch.json>\n       diffcli compile <patch.json> [--mode=M]\n       diffcli compile-wasm <patch.json> --out <out.wasm>"
    return 1
