import Tropical.Ffi
import Tropical.Toolchain.Wasm
import Tropical.Ir.Codec
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.CompileResolved
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.StagedLoad
import Tropical.PlanDecode
import Tropical.Engine
import Tropical.NativeBoot
import Tropical.EmitArrow
import Tropical.Stdlib
import Tropical.Testing.ArrowFixtures
import Tropical.Testing.EngineMirror
import Tropical.Testing.PlanWire
import Tropical.CliArgs

/-!
The `diffcli` executable — build, inspection, render, and backend-qualification
verbs over the Lean-owned compiler and native runtime.

    diffcli render-bytes <plan.json> [--frames N] [--buffer N]

Loads a tropical_plan_7 JSON file into a fresh runtime, renders
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

Every verb reads its flags with `Tropical.CliArgs`, as `tropicalc` does: a
value flag is `--name=V` or `--name V`, and an unknown flag or a malformed
value is an error. `--rate R` (a positive integer) is the rate a kernel is
compiled for: the manifest's `rate` and the kernel's `rate` source; the
default is `Plan.defaultSampleRate`.
-/

open Tropical.CliArgs

/-- Run a verb: read its flags and its one input path, decode the flag values
    it needs, and hand both to `body`. Any error goes to stderr with the
    verb's usage, and the verb exits 1. -/
private def runVerb {σ : Type} (verb usage : String) (flags : List Flag)
    (args : List String) (decode : Args → Except String σ)
    (body : Args → σ → String → IO UInt32) : IO UInt32 := do
  match (do
      let a ← parse flags args
      let path ← a.single
      let opts ← decode a
      pure (a, opts, path)) with
  | .error e =>
    IO.eprintln s!"diffcli {verb}: {e}\nusage: diffcli {verb} {usage}"
    return 1
  | .ok (a, opts, path) => body a opts path

def renderBytes (args : List String) : IO UInt32 :=
  runVerb "render-bytes" "<plan.json> [--frames N] [--buffer N] [--start S] [--all-channels]"
    [valued "frames", valued "buffer", valued "start", switch "all-channels"] args
    (fun a => return (← a.natD "frames" 16, ← a.natD "buffer" 256, ← a.natD "start" 0))
    fun a (frames, buffer, start) planPath => do
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
  -- `--all-channels`: every output channel, frame-major, instead of the
  -- channel-0 view (the golden hashes are channel 0).
  let allChannels := a.has "all-channels"
  let stdout ← IO.getStdout
  for _ in [0:frames] do
    rt.process
    stdout.write (← if allChannels then rt.outputInterleavedBytes else rt.outputBytes)
  stdout.flush
  return 0

/-- `diffcli render-metal <plan.json>` — render through the METAL backend
    (EmitMsl → load_ir_msl → off-RT worker tiles), same byte protocol as
    `render-bytes`. The f64 JIT is dual-loaded but the output comes from the
    f32 GPU kernel — this is the device-under-test side of `metal_vs_jit`.
    Requires libtropical built with TROPICAL_METAL. The render tool may wait
    outside the callback so an offline tight loop cannot outrun the worker. -/
def renderMetal (args : List String) : IO UInt32 :=
  runVerb "render-metal" "<plan.json> [--frames N] [--buffer N] [--start S]"
    [valued "frames", valued "buffer", valued "start"] args
    (fun a => return (← a.natD "frames" 16, ← a.natD "buffer" 256, ← a.natD "start" 0))
    fun _ (frames, buffer, start) planPath => do
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
    [--start S] [--rate R] [--all-channels] [--stage-census]
    [--dump-plan=<path>]` — compile a playground PatchGraph (`{"nodes":[…],"out":…}`)
    through the TYPED session path (`Playground.compilePlan` → typed
    stage-0 split, so banked coefficient columns hoist to the coefficient
    kernel) and render, JIT (default) or Metal (`--metal`). Same byte
    protocol as `render-bytes`. This is the device side of the banked
    `metal_vs_jit` gate: unlike `render-metal` (flow split — arrays pinned
    per-sample, no columns), the typed split here exercises the
    `coeff_columns` GPU crossing exactly as a live session on
    `TROPICAL_BACKEND=metal` does. Prints `hoisted columns=N` on stderr so
    the harness can assert the crossing is actually exercised. -/
def renderGraph (args : List String) : IO UInt32 :=
  runVerb "render-graph"
    "<graph.json> [--metal] [--frames N] [--buffer N] [--start S] [--rate R] [--all-channels] [--stage-census] [--dump-plan=<path>]"
    [valued "frames", valued "buffer", valued "start", valued "rate", valued "dump-plan",
      switch "metal", switch "all-channels", switch "stage-census"] args
    (fun a => return (← a.natD "frames" 16, ← a.natD "buffer" 256, ← a.natD "start" 0,
      (← a.rate?).getD Tropical.Plan.defaultSampleRate))
    fun a (frames, buffer, start, rate) graphPath => do
  let metal := a.has "metal"
  let text ← IO.FS.readFile graphPath
  let j ← match Lean.Json.parse text with
    | .error e => IO.eprintln s!"render-graph: parse: {e}"; return 1
    | .ok j => pure j
  match Tropical.Playground.compilePlan j rate with
  | .error e => IO.eprintln s!"render-graph: compile: {e}"; return 1
  | .ok compiled =>
    -- `--dump-plan=<path>`: write the compiled plan's wire JSON (so `emit-msl`
    -- / `emit-ir` can be pointed at a graph's plan) and stop
    if let some path := a.value? "dump-plan" then
      match compiled.plan.toWire with
      | .error e => IO.eprintln s!"render-graph: toWire: {e}"; return 1
      | .ok wire =>
        IO.FS.writeFile path wire.compress
        IO.eprintln s!"render-graph: wrote plan to {path}"
        return 0
    let opts ← Tropical.StagedLoad.optionsFromEnv
    let split ← match Tropical.Artifact.splitTyped opts compiled.plan compiled.stageBlocks with
      | .error e => IO.eprintln s!"render-graph: {e}"; return 1
      | .ok s => pure s
    IO.eprintln s!"render-graph: hoisted columns={split.audio.coeffArraySlots.size}"
    -- `--stage-census`: how the typed stages and the split distribute the plan
    if a.has "stage-census" then
      let mut fold := 0; let mut s0 := 0; let mut s1 := 0; let mut untyped := 0
      for block in compiled.stageBlocks do
        for st in block do
          match st with
          | some .fold => fold := fold + 1
          | some .s0 => s0 := s0 + 1
          | some .s1 => s1 := s1 + 1
          | none => untyped := untyped + 1
      IO.eprintln s!"census: typed stages fold={fold} s0={s0} s1={s1} untyped={untyped}"
      let census := fun (plan : Tropical.Plan.FlatPlan) => Id.run do
        let mut instrs := 0; let mut fills := 0; let mut fillSlots : Array Nat := #[]
        let mut reduces := 0; let mut routed := 0
        let mut stack := plan.instanceFunctions.toList
        let mut fuel := 100000
        while fuel > 0 do
          fuel := fuel - 1
          match stack with
          | [] => break
          | f :: rest =>
            stack := rest ++ f.children.toList
            for i in f.preambleInstructions ++ f.instructions ++ f.preInputInstructions do
              instrs := instrs + 1
              if i.tag == "ReduceBegin" then reduces := reduces + 1
              if i.tag == "RoutedSumBegin" then routed := routed + 1
              if let .array sl := i.dst then
                fills := fills + 1
                if !fillSlots.contains sl then fillSlots := fillSlots.push sl
        return (instrs, fills, fillSlots.size, reduces, routed)
      let (ai, af, asl, ar, art) := census split.audio
      IO.eprintln s!"census: audio instrs={ai} array-fills={af} over {asl} array slots · regions reduce={ar} routed={art}"
      match split.coeff? with
      | some c =>
        let (ci, cf, csl, cr, crt) := census c
        IO.eprintln s!"census: coeff instrs={ci} array-fills={cf} over {csl} array slots · regions reduce={cr} routed={crt}"
      | none => IO.eprintln "census: no coefficient kernel"
      return 0
    let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
    if metal then Tropical.StagedLoad.loadMslTyped rt compiled.plan compiled.stageBlocks
    else Tropical.StagedLoad.loadTyped rt compiled.plan compiled.stageBlocks
    if start != 0 then rt.setSampleIndex start.toUInt64
    -- `--all-channels`: every output channel, frame-major (`out` then each
    -- tap), instead of the channel-0 view — so a graph's taps can be gated.
    let allChannels := a.has "all-channels"
    let stdout ← IO.getStdout
    for _ in [0:frames] do
      if metal then rt.processOffline else rt.process
      stdout.write (← if allChannels then rt.outputInterleavedBytes else rt.outputBytes)
    stdout.flush
    return 0

/-- `diffcli render-sweep <graph.json> --param=<knob> [--center=800]
    [--octaves=2] [--rate-hz=2] [--frames N] [--buffer N] [--rate R]` — render a
    playground graph while the CONTROL PLANE sweeps one knob:
    `value(t) = center · 2^(octaves · sin(2π · rate · t))`, written to the
    knob's glide slots (`#v0`/`#v1`, or the bare `param:` slot) once per
    process call, at the block-start time. Per-sample automation is
    `--buffer 1` — the update granularity IS the buffer, which is the point:
    the same verb at two buffer sizes prices coefficient-update quantization
    (the settle tradeoff) with no other variable moving. Byte protocol as
    `render-bytes`. The graph compiles for `--rate` (default
    `Plan.defaultSampleRate`), and the sweep's time base is the same rate. -/
def renderSweep (args : List String) : IO UInt32 :=
  runVerb "render-sweep"
    "<graph.json> --param=<knob> [--center=C] [--octaves=O] [--rate-hz=R] [--frames N] [--buffer N] [--rate R]"
    [valued "param", valued "center", valued "octaves", valued "rate-hz",
      valued "frames", valued "buffer", valued "rate"] args
    (fun a => do
      let some knob := a.value? "param"
        | throw "--param=<knob> is required (e.g. --param=flt.cutoff)"
      return (knob, ← a.natD "frames" 16, ← a.natD "buffer" 256,
        ← a.floatD "center" 800.0, ← a.floatD "octaves" 2.0, ← a.floatD "rate-hz" 2.0,
        (← a.rate?).getD Tropical.Plan.defaultSampleRate))
    fun _ (knob, frames, buffer, center, octaves, rateHz, rate) graphPath => do
  let text ← IO.FS.readFile graphPath
  let j ← match Lean.Json.parse text with
    | .error e => IO.eprintln s!"render-sweep: parse: {e}"; return 1
    | .ok j => pure j
  match Tropical.Playground.compilePlan j rate with
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
    let sampleRate := rate.toFloat
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

/-- Is this source a playground graph (door 1: a flat PatchGraph or a v3
    hierarchy document, `Playground.isGraphDocument`) rather than a
    `tropical_program_2` patch (door 2)? `--graph` forces the graph door. -/
private def isPatchGraph (a : Args) (j : Lean.Json) : Bool :=
  a.has "graph" || Tropical.Playground.isGraphDocument j

/-- A patch's plan at `--rate`, when given: nothing above the plan bakes a
    rate (the IR reads `sampleRate`, a plan source), so the plan takes it after
    compile. -/
private def atRate (rate? : Option Nat) (plan : Tropical.Plan.FlatPlan) :
    Tropical.Plan.FlatPlan :=
  match rate? with
  | some r => { plan with sampleRate := (r : Lean.JsonNumber) }
  | none => plan

/-- `diffcli compile <patch-or-graph.json> [--mode=<m>] [--rate R]
    [--fixtures] [--graph]` → plan JSON on stdout. Boots the engine (stdlib
    from the pre-parsed bridge), loads the patch through the engine's own
    ingest, and rebuilds the plan from the mirror at the requested mode. The
    side-B command of diff_plan.ts / diff_audio.ts, and the engine side of
    `tools/check_artifact_parity.py` against `tropicalc --emit=plan`. -/
def compileVerb (args : List String) : IO UInt32 :=
  runVerb "compile"
    "<patch-or-graph.json> [--mode=fused|microkernel|microkernel-deep] [--rate R] [--fixtures] [--graph]"
    [valued "mode", valued "rate", switch "fixtures", switch "graph"] args
    (fun a => do
      let modeStr := (a.value? "mode").getD "fused"
      let some mode := Tropical.Plan.CompilationMode.ofWire? modeStr
        | throw s!"unknown compilation mode: {modeStr}"
      return (modeStr, mode, ← a.rate?))
    fun a (modeStr, mode, rate?) patch => do
  let env ← Tropical.NativeBoot.boot
  -- A PatchGraph compiles as the engine's `load_patch_graph` does: against
  -- the boot-time stdlib arena (fused; the graph door has no other mode).
  if let .ok j := Lean.Json.parse (← IO.FS.readFile patch) then
    if isPatchGraph a j then
      if mode != .fused then
        IO.eprintln s!"compile: a PatchGraph compiles fused only (got --mode={modeStr})"
        return 1
      match Tropical.Playground.compilePlanPure env.stdlib.1 env.stdlib.2 j
          (rate?.getD Tropical.Plan.defaultSampleRate) with
      | .error e => IO.eprintln s!"compile: graph: {e}"; return 1
      | .ok compiled =>
        match compiled.plan.toWire with
        | .error e => IO.eprintln s!"compile: toWire: {e}"; return 1
        | .ok wire => IO.println wire.compress; return 0
  let act : Tropical.EngineM Tropical.Plan.FlatPlan := do
    if a.has "fixtures" then Tropical.Engine.registerTestFixtures env
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str patch)])
    Tropical.Engine.compileMirrorFlatPlan env mode
  match ← act.run with
  | .ok plan =>
    match (atRate rate? plan).toWire with
    | .error e => IO.eprintln s!"compile: toWire: {e}"; return 1
    | .ok wire => IO.println wire.compress; return 0
  | .error f =>
    IO.eprintln f.toJson.compress
    return 1

/-- Compile a patch to an in-memory FlatPlan (the shape both the plan path
    and the IR path consume). `fixtures` additionally registers the
    test-fixture programs (`OpZoo`) so equivalence patches can instantiate
    them by name. -/
private def compileToFlatPlan (patch : String) (fixtures : Bool := false) :
    IO (Except String Tropical.Plan.FlatPlan) := do
  let env ← Tropical.NativeBoot.boot
  let act : Tropical.EngineM Tropical.Plan.FlatPlan := do
    if fixtures then Tropical.Engine.registerTestFixtures env
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str patch)])
    Tropical.Engine.compileMirrorFlatPlan env .fused
  match ← act.run with
  | .ok p => pure (.ok p)
  | .error f => pure (.error f.toJson.compress)

/-- The kernel split the wasm/manifest verbs emit for a source, through
    whichever front door its shape names.

    * A PatchGraph compiles through `Playground.compilePlan` and the TYPED
      stage-0 split — exactly what `render-graph` and the live engine's
      `load_patch_graph` load — so the τ-independent coefficient plan (bank
      columns, coefficient tables) becomes the kernel's once-per-call
      prologue, and the wasm matches `render-graph` on the JIT.
    * A `tropical_program_2` patch compiles through the engine mirror. It
      carries no stage blocks, so it is emitted unsplit (`coeff? := none`):
      the whole plan in the sample loop. That is the `TROPICAL_STAGE0=0`
      kill-switch shape — the same values, recomputed per frame instead of
      per call — so it is correct, only slower for coefficient-heavy plans. -/
private def compileToSplit (a : Args) (rate? : Option Nat) (path : String) :
    IO (Except String Tropical.Ir.Stage0.Split) := do
  let text ← IO.FS.readFile path
  let patchSplit := do
    return (← compileToFlatPlan path (fixtures := a.has "fixtures")).map
      fun plan => { audio := atRate rate? plan, coeff? := none }
  match Lean.Json.parse text with
  | .ok j =>
    if isPatchGraph a j then
      -- A graph takes it at decode: the default string's period is a sample count.
      match Tropical.Playground.compilePlan j (rate?.getD Tropical.Plan.defaultSampleRate) with
      | .error e => return .error s!"graph compile: {e}"
      | .ok compiled =>
        let opts ← Tropical.StagedLoad.optionsFromEnv
        return Tropical.Artifact.splitTyped opts compiled.plan compiled.stageBlocks
    else patchSplit
  | .error _ => patchSplit

/-- `diffcli emit-ir <patch.json> [--frames N]` → the Lean-emitted LLVM IR
    on stdout: the port kernel for `N` frames per call (default 512), with
    the stage-0 prologue inside. -/
def emitIrVerb (args : List String) : IO UInt32 :=
  runVerb "emit-ir" "<patch.json> [--frames N]" [valued "frames"] args
    (fun a => return (← a.positive? "frames").getD Tropical.Artifact.defaultFrames)
    fun _ frames patch => do
  match ← compileToFlatPlan patch with
  | .error e => IO.eprintln e; return 1
  | .ok plan =>
    let opts ← Tropical.StagedLoad.optionsFromEnv
    match Tropical.Ir.EmitLlvm.emitKernel (Tropical.Artifact.split opts plan) frames with
    | .error e => IO.eprintln s!"emitKernel: {e}"; return 1
    | .ok ir => IO.println ir; return 0

/-- `diffcli emit-msl <patch.json>` → the Lean-emitted Metal Shading
    Language kernel on stdout (the Metal backend's codegen; sibling of
    `emit-ir`). Sanity compile-check without any engine:
    `diffcli emit-msl p.json | xcrun -sdk macosx metal -x metal -c -o /dev/null -`. -/
def emitMslVerb (args : List String) : IO UInt32 :=
  runVerb "emit-msl" "<patch.json> [--frames N]" [valued "frames"] args
    (fun a => return (← a.positive? "frames").getD Tropical.Artifact.defaultFrames)
    fun _ frames patch => do
  match ← compileToFlatPlan patch with
  | .error e => IO.eprintln e; return 1
  | .ok plan =>
    match Tropical.Ir.EmitMsl.emitKernel plan frames with
    | .error e => IO.eprintln s!"emitKernel (msl): {e}"; return 1
    | .ok msl => IO.println msl; return 0

/-- `diffcli compile-wasm <patch.json> --out <out.wasm> [--frames N] [--rate R]` → a
    complete wasm32 module of the port kernel for `N` frames per call
    (default 128, the worklet quantum), emitted in-process (Lean IR →
    libtropical_wasm's LLVM+lld, no subprocess). `compile-kernel` prints the
    matching `tropical_kernel_1` manifest. The source is a
    `tropical_program_2` patch or a playground graph (detected by shape,
    or forced with `--graph`; see `compileToSplit`). -/
def compileWasmVerb (args : List String) : IO UInt32 :=
  runVerb "compile-wasm"
    "<patch-or-graph.json> --out <out.wasm> [--frames N] [--rate R] [--fixtures] [--graph]"
    [valued "out", valued "frames", valued "rate", switch "fixtures", switch "graph"] args
    (fun a => do
      let some outPath := a.value? "out" | throw "--out <path> is required"
      return (outPath, (← a.positive? "frames").getD 128, ← a.rate?))
    fun a (outPath, frames, rate?) patch => do
  match ← compileToSplit a rate? patch with
  | .error e => IO.eprintln e; return 1
  | .ok split =>
    match Tropical.Ir.EmitLlvm.emitKernel split frames with
    | .error e => IO.eprintln s!"emitKernel: {e}"; return 1
    | .ok ir =>
      let wasm ← Tropical.Toolchain.Wasm.compileIr ir
      IO.FS.writeBinFile (System.FilePath.mk outPath) wasm
      IO.eprintln s!"compile-wasm: wrote {wasm.size} bytes → {outPath}"
      return 0

/-- `diffcli compile-kernel <patch.json> [--frames N] [--rate R] [--fixtures]
    [--graph]` → the
    `tropical_kernel_1` manifest of the patch for `N` frames per call
    (default 128), on stdout. The browser-side manifest. -/
def compileKernelVerb (args : List String) : IO UInt32 :=
  runVerb "compile-kernel" "<patch-or-graph.json> [--frames N] [--rate R] [--fixtures] [--graph]"
    [valued "frames", valued "rate", switch "fixtures", switch "graph"] args
    (fun a => return ((← a.positive? "frames").getD 128, ← a.rate?))
    fun a (frames, rate?) patch => do
  match ← compileToSplit a rate? patch with
  | .error e => IO.eprintln e; return 1
  | .ok split =>
    match Tropical.Kernel.manifest split.audio frames with
    | .error e => IO.eprintln s!"compile-kernel: {e}"; return 1
    | .ok manifest => IO.println manifest; return 0

/-- The numeric contract an object carries, stated in its manifest: what the
    emitted IR and the toolchain's TargetOptions guarantee (`EmitLlvm` writes
    no `fast`/`contract`/`afn` flags and no `llvm.fmuladd`; `compileObject`
    keeps `AllowFPOpFusion = Standard`), so every op rounds as written. -/
def objectNumerics : Lean.Json := Lean.Json.mkObj [
  ("float", "ieee754-binary64"),
  ("int", "i64-twos-complement-wrapping"),
  ("fp_contract", "off"),
  ("fast_math", false),
  ("rounding", "nearest-even"),
  ("denormals", "ieee")]

/-- `diffcli compile-object <patch-or-graph.json> --out <out.o> [--triple T]
    [--cpu C] [--features F] [--frames N] [--rate R] [--graph]` → one
    relocatable object of the port kernel for the target triple (default:
    the host), and its manifest at `<out>.manifest.json`: the
    `tropical_kernel_1` manifest plus an `object` block naming the triple, the
    symbol and the numeric contract. The same emitted LLVM as `compile-wasm`
    and the JIT; no link. `--triple thumbv6m-none-eabi` gives a Cortex-M0+
    object, `--triple wasm32-unknown-unknown` a wasm object (not a module). -/
def compileObjectVerb (args : List String) : IO UInt32 :=
  runVerb "compile-object"
    "<patch-or-graph.json> --out <out.o> [--triple T] [--cpu C] [--features F] [--frames N] [--rate R] [--out-elem f64|i32] [--out-format Q1.15] [--emit-ir] [--fixtures] [--graph]"
    [valued "out", valued "triple", valued "cpu", valued "features", valued "frames",
     valued "rate", valued "out-elem", valued "out-format", switch "emit-ir",
     switch "fixtures", switch "graph"] args
    (fun a => do
      let some outPath := a.value? "out" | throw "--out <path> is required"
      return (outPath, (← a.positive? "frames").getD 128, ← a.rate?))
    fun a (outPath, frames, rate?) patch => do
  match ← compileToSplit a rate? patch with
  | .error e => IO.eprintln e; return 1
  | .ok split =>
    let outElem ← match (a.value? "out-elem").getD "f64" with
      | "f64" => pure Tropical.Kernel.Elem.f64
      | "i32" => pure .i32
      | other => IO.eprintln s!"compile-object: --out-elem {other} (f64 | i32)"; return 1
    let outFormat := (a.value? "out-format").getD (if outElem == .i32 then "int" else "ieee754-binary64")
    match Tropical.Ir.EmitLlvm.emitKernel split frames outElem,
          Tropical.Kernel.ofPlan split.audio frames outElem with
    | .error e, _ => IO.eprintln s!"emitKernel: {e}"; return 1
    | _, .error e => IO.eprintln s!"compile-object: {e}"; return 1
    | .ok ir, .ok ktype =>
      let triple := (a.value? "triple").getD ""
      let cpu := (a.value? "cpu").getD ""
      let features := (a.value? "features").getD ""
      let obj ← Tropical.Toolchain.Wasm.compileObject ir triple cpu features
      IO.FS.writeBinFile (System.FilePath.mk outPath) obj
      if a.has "emit-ir" then IO.FS.writeFile (System.FilePath.mk (outPath ++ ".ll")) ir
      match ktype.toWire with
      | .error e => IO.eprintln s!"compile-object: {e}"; return 1
      | .ok kj =>
        let objectBlock := Lean.Json.mkObj [
          ("triple", if triple.isEmpty then "host" else triple),
          ("cpu", if cpu.isEmpty then "generic" else cpu),
          ("features", features),
          ("format", "relocatable-object"),
          ("symbol", "tropical_kernel"),
          ("frames", frames),
          ("out_format", outFormat),
          ("numerics", objectNumerics)]
        let manifest := kj.setObjVal! "object" objectBlock
        IO.FS.writeFile (System.FilePath.mk (outPath ++ ".manifest.json")) manifest.pretty
        IO.eprintln s!"compile-object: wrote {obj.size} bytes → {outPath}"
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
  | "compile-kernel" :: rest => compileKernelVerb rest
  | "compile-object" :: rest => compileObjectVerb rest
  | "compile" :: rest => compileVerb rest
  | _ =>
    IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N] [--all-channels]\n       diffcli render-metal <plan.json>\n       diffcli render-graph <graph.json> [--all-channels] [--rate R]\n       diffcli render-sweep <graph.json> --param=<knob> [--rate R]\n       diffcli emit-ir <patch.json>\n       diffcli emit-msl <patch.json>\n       diffcli compile <patch-or-graph.json> [--mode=M] [--rate R] [--graph]\n       diffcli compile-wasm <patch-or-graph.json> --out <out.wasm> [--frames N] [--rate R] [--graph]\n       diffcli compile-kernel <patch-or-graph.json> [--frames N] [--rate R] [--graph]\n       diffcli compile-object <patch-or-graph.json> --out <out.o> [--triple T] [--cpu C] [--frames N] [--rate R] [--graph]"
    return 1
