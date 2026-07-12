import Tropical.Ffi
import Tropical.Parse.Raise
import Tropical.Parse.VoiceDesugar
import Tropical.Ir.Elaborator
import Tropical.Ir.Codec
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.CompileResolved
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.StagedLoad
import Tropical.StdlibChain
import Tropical.PlanDecode
import Tropical.Engine
import Tropical.Parse.Surface.Markdown
import Tropical.EmitArrow
import Tropical.Testing.ArrowFixtures
import Tropical.Testing.EngineMirror
import Tropical.Testing.PlanWire

/-!
The `diffcli` executable — differential-harness verbs that exercise the
Lean-owned native runtime and (Phase 4) the parsed-layer port.

    diffcli render-bytes <plan.json> [--frames N] [--buffer N]

Loads a tropical_plan_5 JSON file into a fresh runtime, renders
`frames × buffer` samples, and writes the raw little-endian float64
stream to stdout. `… | shasum -a 256` reproduces the golden hashes in
tests/golden/ when fed the same plans validate_stdlib.ts renders
(frames=16, buffer=256) — the Phase 2 gate that the Lean FFI path is
sample-for-sample the TS koffi path.

    diffcli raise <file.json>

Runs a `tropical_program_2` file through the Lean port of
normalizeProgramFile + raiseProgram and prints
`{program, params?, audio_outputs?}` — or `{error}` with the TS error
string — on stdout. The other side of scripts/diff/diff_raise.ts.

    diffcli parsed-roundtrip <file.json>

Reads serialized ParsedProgram JSON (the stdlib bridge corpus under
stdlib/parsed/), decodes it into the typed AST, re-encodes, prints.
A decode failure prints `{error}` — which the differ reports, since
the TS side echoes the file verbatim. Gates codec fidelity.

    diffcli elab-stdlib <Name>
    diffcli elab-file <parsed.json>

Phase 4 stage 3 (the elaborator gate, scripts/diff/diff_elab.ts).
`elab-stdlib` elaborates the pre-parsed stdlib bridge in manifest order
up to and including <Name> — threading an external resolver over the
raw elaborated results, mirroring scripts/diff/elab_cmd.ts — and prints
the canonical `tropical_resolved_1` encoding of <Name>. `elab-file`
elaborates a self-contained ParsedProgram JSON against the full stdlib
chain as resolver. An `ElaborationError` / `CycleViolation` prints
`{error: <byte-exact TS message>}` and exits 0 (errors are comparable
outputs); a ParsedProgram decode failure is a harness error (exit 1).

    diffcli strata-stdlib <Name> [--upto=K] [--mode=inline|nested] [--type-args=J]
    diffcli strata-file <parsed.json> [same flags]

Phase 5 (the strata gate, scripts/diff/diff_strata.ts). Elaborates
exactly like the elab verbs, then runs strata passes `1..K` and prints
the canonical `tropical_resolved_1` encoding of the result — the
hybrid prefix the TS suffix completes. `--upto` beyond
`Strata.portedPasses` is a harness error (exit 1); strata errors print
`{error: <byte-exact TS message>}` and exit 0.
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
    (EmitMsl → load_ir_msl → GPU dispatch per block), same byte protocol as
    `render-bytes`. The f64 JIT is dual-loaded but the output comes from the
    f32 GPU kernel — this is the device-under-test side of `metal_vs_jit`.
    Requires libtropical built with TROPICAL_METAL. Always sync dispatch. -/
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
    rt.process
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
  | .ok (plan, _, stageBlocks) =>
    let split ← Tropical.StagedLoad.splitTyped plan stageBlocks
    IO.eprintln s!"render-graph: hoisted columns={split.audio.coeffArraySlots.size}"
    let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
    if metal then Tropical.StagedLoad.loadMslTyped rt plan stageBlocks
    else Tropical.StagedLoad.loadTyped rt plan stageBlocks
    if start != 0 then rt.setSampleIndex start.toUInt64
    let stdout ← IO.getStdout
    for _ in [0:frames] do
      rt.process
      stdout.write (← rt.outputBytes)
    stdout.flush
    return 0

private def errorJson (msg : String) : Lean.Json :=
  Lean.Json.mkObj [("error", Lean.Json.str msg)]

def raiseVerb (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: diffcli raise <file.json>"
      return 1
  let text ← IO.FS.readFile path
  let out : Lean.Json :=
    match Tropical.Parse.JsonV.parse text with
    | .error e => errorJson s!"JSON parse error: {e}"
    | .ok jv =>
      match Tropical.Parse.Raise.raiseFile jv with
      | .error msg => errorJson msg
      | .ok (prog, top) => Tropical.Parse.Raise.raisedFileJson prog top
  IO.println out.compress
  return 0

def parsedRoundtripVerb (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: diffcli parsed-roundtrip <file.json>"
      return 1
  let text ← IO.FS.readFile path
  let out : Lean.Json :=
    match Tropical.Parse.JsonV.parse text with
    | .error e => errorJson s!"JSON parse error: {e}"
    | .ok jv =>
      match Tropical.Parse.decodeProgram jv with
      | .error msg => errorJson msg
      | .ok prog => prog.toJson
  IO.println out.compress
  return 0

/-- Voice-ports render proof (Phase 1): desugar a voice-bearing parsed program
    against a hardcoded `FixedSinOsc` voice binding and print the desugared
    ParsedProgram JSON. The chain `parse-md flanger.md | voice-desugar |
    emit-file` lowers the result; rendered, it must match FlangeSin. -/
def voiceDesugarVerb (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: diffcli voice-desugar <parsed.json>"
      return 1
  let text ← IO.FS.readFile path
  let out : Lean.Json :=
    match Tropical.Parse.JsonV.parse text with
    | .error e => errorJson s!"JSON parse error: {e}"
    | .ok jv =>
      match Tropical.Parse.decodeProgram jv with
      | .error msg => errorJson msg
      | .ok prog =>
        let vb : Tropical.Parse.VoiceDesugar.VoiceBinding :=
          { voiceName := "v", programName := "FixedSinOsc",
            clkInput := "clk", output := "sine",
            params := #[("freq", .nameRef "freq")] }
        match Tropical.Parse.VoiceDesugar.desugarProgram #[vb] prog with
        | .error msg => errorJson msg
        | .ok desugared => desugared.toJson
  IO.println out.compress
  return 0

/-- Surface-parse a literate `.md` file and print the ParsedProgram JSON
    (Phase 7). Side B of the `diff-parse` gate; side A is the committed
    `stdlib/parsed/<Name>.json`. -/
def parseMdVerb (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: diffcli parse-md <file.md>"
      return 1
  let text ← IO.FS.readFile path
  let out : Lean.Json :=
    match Tropical.Parse.Surface.parseMarkdownProgram text with
    | .error e => errorJson e
    | .ok prog => prog.toJson
  IO.println out.compress
  return 0

/-- Regenerate the committed stdlib bridge (`stdlib/parsed/<Name>.json` +
    `manifest.json`) from `stdlib/*.md` (Phase 8 — replaces the bun
    `scripts/build_parsed_stdlib.ts`). Sorted readdir, parse each via the
    Phase-7 surface parser, derive registration order by the same
    fixed-point elaborate loop, write Lean-serialized ParsedProgram JSON.
    Output is Lean-canonical (sorted keys) — semantically identical to the
    prior TS-serialized bridge, and idempotent under re-runs. -/
def parseAllVerb (_args : List String) : IO UInt32 := do
  -- Scan stdlib/ (top level) then stdlib/reversible/ — the subdir holds the
  -- reversible-synthesis programs; placing it after the base stdlib keeps each
  -- dir internally sorted, and the fixed-point loop below resolves the actual
  -- dependency order regardless (reversible programs depend on base ones).
  let topEntries ← (System.FilePath.mk "stdlib").readDir
  let topPaths := (topEntries.filterMap fun e =>
      let n := e.fileName
      if n.endsWith ".md" && n != "README.md" then some s!"stdlib/{n}" else none)
    |>.qsort (fun a b => decide (a < b))
  let revDir := "stdlib/reversible"
  let revPaths ← (if ← System.FilePath.pathExists revDir then do
      let revEntries ← (System.FilePath.mk revDir).readDir
      pure ((revEntries.filterMap fun e =>
          let n := e.fileName
          if n.endsWith ".md" && n != "README.md" then some s!"{revDir}/{n}" else none)
        |>.qsort (fun a b => decide (a < b)))
    else pure #[])
  let paths := topPaths ++ revPaths
  let mut parsed : Array (String × Tropical.Parse.Program) := #[]
  for path in paths do
    let text ← IO.FS.readFile path
    match Tropical.Parse.Surface.parseMarkdownProgram text with
    | .error e => IO.eprintln s!"{path}: {e}"; return 1
    | .ok prog => parsed := parsed.push (prog.name, prog)
  -- Fixed-point elaborate loop → registration (manifest) order: each pass
  -- elaborates the programs whose sibling refs already resolved.
  let mut arena : Tropical.Ir.Arena := {}
  let mut resolvedIdx : Array (String × Tropical.Ir.ProgramIdx) := #[]
  let mut order : Array String := #[]
  let mut remaining := parsed
  let mut progress := true
  while progress && !remaining.isEmpty do
    progress := false
    let mut still : Array (String × Tropical.Parse.Program) := #[]
    for (name, prog) in remaining do
      let r := resolvedIdx
      match Tropical.Ir.elaborateInto arena prog (some fun n => (r.find? (·.1 == n)).map (·.2)) with
      | .ok (arena', idx) =>
        arena := arena'
        resolvedIdx := resolvedIdx.push (name, idx)
        order := order.push name
        progress := true
      | .error _ => still := still.push (name, prog)
    remaining := still
  if h : remaining.size > 0 then
    let (name, prog) := remaining[0]
    let r := resolvedIdx
    match Tropical.Ir.elaborateInto arena prog (some fun n => (r.find? (·.1 == n)).map (·.2)) with
    | .error e => IO.eprintln s!"parse-all: failed to elaborate '{name}': {e.message}"; return 1
    | .ok _ => IO.eprintln s!"parse-all: '{name}' unexpectedly elaborated"; return 1
  for (name, prog) in parsed do
    IO.FS.writeFile s!"stdlib/parsed/{name}.json" (prog.toJson.pretty ++ "\n")
  let manifest := Lean.Json.mkObj [("programs", Lean.Json.arr (order.map Lean.Json.str))]
  IO.FS.writeFile "stdlib/parsed/manifest.json" (manifest.pretty ++ "\n")
  IO.println s!"wrote {parsed.size} parsed programs + manifest to stdlib/parsed/"
  return 0

-- ── elab-stdlib / elab-file (Phase 4 stage 3) ────────────────────────────────

private def parsedDir : String := Tropical.StdlibChain.parsedDir

/-- Read + strictly decode a serialized ParsedProgram (the shared bridge
    reader — verbs also point it at arbitrary paths). -/
private def readParsed (path : String) : IO (Except String Tropical.Parse.Program) :=
  Tropical.StdlibChain.readParsed path

private def manifestNames : IO (Except String (Array String)) :=
  Tropical.StdlibChain.manifestNames

/-- The raw-elaborated stdlib chain (elaborate-only — no strata) over an
    explicit name prefix, keeping the structured `ElabError` the verbs
    print as comparable JSON (the failing name is dropped — the caller
    named the chain). -/
private def elabChain (names : Array String) :
    IO (Except String (Except Tropical.Ir.ElabError
      (Tropical.Ir.Arena × Array (String × Tropical.Ir.ProgramIdx)))) := do
  match ← Tropical.StdlibChain.elabChain names with
  | .error e => pure (.error e)
  | .ok (.error (_, e)) => pure (.ok (.error e))
  | .ok (.ok r) => pure (.ok (.ok r))

private def printEncoded (arena : Tropical.Ir.Arena) (root : Tropical.Ir.ProgramIdx) :
    IO UInt32 := do
  match Tropical.Ir.Codec.encodeResolved arena root with
  | .error e =>
    IO.eprintln e
    return 1
  | .ok j =>
    IO.println j.compress
    return 0

private def printElabError (e : Tropical.Ir.ElabError) : IO UInt32 := do
  IO.println (errorJson e.message).compress
  return 0

def elabStdlibVerb (args : List String) : IO UInt32 := do
  let some target := args.head?
    | IO.eprintln "usage: diffcli elab-stdlib <Name>"
      return 1
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    let some i := names.findIdx? (· == target)
      | IO.eprintln s!"elab-stdlib: '{target}' is not in {parsedDir}/manifest.json"
        return 1
    match ← elabChain (names.extract 0 (i + 1)) with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      let some (_, idx) := resolved.find? (·.1 == target)
        | IO.eprintln s!"elab-stdlib: internal: '{target}' missing from chain"
          return 1
      printEncoded arena idx

def elabFileVerb (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: diffcli elab-file <parsed.json>"
      return 1
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    match ← elabChain names with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      match ← readParsed path with
      | .error e => IO.eprintln s!"{path}: {e}"; return 1
      | .ok prog =>
        match Tropical.Ir.elaborateInto arena prog
            (some fun n => (resolved.find? (·.1 == n)).map (·.2)) with
        | .error e => printElabError e
        | .ok (arena', idx) => printEncoded arena' idx

-- ── strata-stdlib / strata-file (Phase 5) ────────────────────────────────────

private def parseStrFlag (args : List String) (flag : String) : Option String :=
  args.findSome? fun a =>
    if a.startsWith (flag ++ "=") then some (a.drop (flag.length + 1)).toString else none

/-- Parse `--type-args={"N":8}` into by-name (name, number) pairs.
    Malformed JSON / non-number values are harness errors. -/
private def parseTypeArgs (args : List String) :
    Except String (Array (String × Lean.JsonNumber)) := do
  let some raw := parseStrFlag args "--type-args" | return #[]
  let jv ← Tropical.Parse.JsonV.parse raw |>.mapError (s!"--type-args parse error: {·}")
  let Tropical.Parse.JsonV.obj fields := jv
    | .error "--type-args must be a JSON object"
  fields.mapM fun (name, v) => match v with
    | .num n => .ok (name, n)
    | _ => .error s!"--type-args['{name}'] must be a number"

/-- Parse the strata flags shared by both verbs. `--upto` beyond
    `Strata.portedPasses` is a harness error — the ratchet that keeps
    un-ported passes out of the comparable-output path. -/
private def parseStrataOptions (args : List String) :
    Except String Tropical.Ir.Strata.Options := do
  -- `--upto=K` (the `=` form; parseNatFlag's space form would silently
  -- fall back to the default and un-gate the ratchet).
  let upto ← match parseStrFlag args "--upto" with
    | none => .ok Tropical.Ir.Strata.portedPasses
    | some s => match s.toNat? with
      | some n => .ok n
      | none => .error s!"--upto={s} is not a number"
  if upto > Tropical.Ir.Strata.portedPasses then
    .error s!"strata: --upto={upto} exceeds ported passes ({Tropical.Ir.Strata.portedPasses})"
  let inlineNested ← match parseStrFlag args "--mode" with
    | none | some "inline" => .ok true
    | some "nested" => .ok false
    | some m => .error s!"unknown --mode={m}"
  let typeArgs ← parseTypeArgs args
  return { upto, inlineNested, typeArgs }

private def printStrata (opts : Tropical.Ir.Strata.Options)
    (arena : Tropical.Ir.Arena) (root : Tropical.Ir.ProgramIdx) : IO UInt32 := do
  match Tropical.Ir.Strata.run opts arena root with
  | .error e =>
    IO.println (errorJson e.message).compress
    return 0
  | .ok (arena', root') =>
    -- Post-strata invariant assertion (inline path only — fractal
    -- harness inputs carry raw-elaborated instance targets; the
    -- production seam asserts those per registered program). A check
    -- failure is a Lean-side invariant bug: harness error, loud.
    if opts.upto == 5 && opts.inlineNested then
      if let .error e := Tropical.Ir.checkResolvedArena arena' root' then
        IO.eprintln e
        return 1
    printEncoded arena' root'

def strataStdlibVerb (args : List String) : IO UInt32 := do
  let some target := args.head?
    | IO.eprintln "usage: diffcli strata-stdlib <Name> [--upto=K] [--mode=M] [--type-args=J] [--cf-only]"
      return 1
  let opts ← match parseStrataOptions args.tail with
    | .error e => IO.eprintln e; return 1
    | .ok o => pure o
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    let some i := names.findIdx? (· == target)
      | IO.eprintln s!"strata-stdlib: '{target}' is not in {parsedDir}/manifest.json"
        return 1
    match ← elabChain (names.extract 0 (i + 1)) with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      let some (_, idx) := resolved.find? (·.1 == target)
        | IO.eprintln s!"strata-stdlib: internal: '{target}' missing from chain"
          return 1
      printStrata opts arena idx

def strataFileVerb (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: diffcli strata-file <parsed.json> [--upto=K] [--mode=M] [--type-args=J] [--cf-only]"
      return 1
  let opts ← match parseStrataOptions args.tail with
    | .error e => IO.eprintln e; return 1
    | .ok o => pure o
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    match ← elabChain names with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      match ← readParsed path with
      | .error e => IO.eprintln s!"{path}: {e}"; return 1
      | .ok prog =>
        match Tropical.Ir.elaborateInto arena prog
            (some fun n => (resolved.find? (·.1 == n)).map (·.2)) with
        | .error e => printElabError e
        | .ok (arena', idx) => printStrata opts arena' idx

-- ── emit-stdlib / emit-file (Phase 6 stage 6b) ──────────────────────────────

/-- Full strata (inline) → Core downcast → compileResolved → wire
    PerInstancePlan JSON. Strata/emit errors are comparable `{error}`
    outputs; a Core-check failure is a Lean-side port bug (exit 1). -/
private def printEmit (typeArgs : Array (String × Lean.JsonNumber))
    (arena : Tropical.Ir.Arena) (root : Tropical.Ir.ProgramIdx) : IO UInt32 := do
  let opts : Tropical.Ir.Strata.Options :=
    { upto := Tropical.Ir.Strata.portedPasses, inlineNested := true, typeArgs }
  match Tropical.Ir.Strata.runResolved opts arena root with
  | .error e =>
    IO.println (errorJson e.message).compress
    return 0
  | .ok (coreArena, core) =>
    match Tropical.Ir.CompileResolved.compileResolved core coreArena with
    | .error msg =>
      IO.println (errorJson msg).compress
      return 0
    | .ok plan =>
      match plan.toWire with
      | .error e => IO.eprintln e; return 1
      | .ok j => IO.println j.compress; return 0

def emitStdlibVerb (args : List String) : IO UInt32 := do
  let some target := args.head?
    | IO.eprintln "usage: diffcli emit-stdlib <Name> [--type-args=J] [--cf-only]"
      return 1
  let typeArgs ← match parseTypeArgs args.tail with
    | .error e => IO.eprintln e; return 1
    | .ok t => pure t
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    let some i := names.findIdx? (· == target)
      | IO.eprintln s!"emit-stdlib: '{target}' is not in {parsedDir}/manifest.json"
        return 1
    match ← elabChain (names.extract 0 (i + 1)) with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      let some (_, idx) := resolved.find? (·.1 == target)
        | IO.eprintln s!"emit-stdlib: internal: '{target}' missing from chain"
          return 1
      printEmit typeArgs arena idx

def emitFileVerb (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: diffcli emit-file <parsed.json> [--type-args=J] [--cf-only]"
      return 1
  let typeArgs ← match parseTypeArgs args.tail with
    | .error e => IO.eprintln e; return 1
    | .ok t => pure t
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    match ← elabChain names with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      match ← readParsed path with
      | .error e => IO.eprintln s!"{path}: {e}"; return 1
      | .ok prog =>
        match Tropical.Ir.elaborateInto arena prog
            (some fun n => (resolved.find? (·.1 == n)).map (·.2)) with
        | .error e => printElabError e
        | .ok (arena', idx) => printEmit typeArgs arena' idx

-- ── emitarrow-flanger (EmitArrow slice 1 — the gated flanger) ───────────────

/-- Build the flanger with the EmitArrow combinator library
    (`Tropical/EmitArrow.lean`) instead of elaborating `FlangeSin.json`, then
    run the SAME emit recipe `emit-file` uses. The gate: this verb's plan JSON
    must be byte-identical to `diffcli emit-file stdlib/parsed/FlangeSin.json`.
    `elabChain` fills the arena with the elaborated stdlib (so `FixedSinOsc` is
    available to link against); `buildFlanger` pushes the EmitArrow-built
    `Program` and returns its `ProgramIdx`. -/
def emitarrowFlangerVerb (_args : List String) : IO UInt32 := do
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    match ← elabChain names with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      match Tropical.EmitArrow.buildFlanger arena resolved with
      | .error e => IO.eprintln s!"emitarrow-flanger: {e}"; return 1
      | .ok (arena', idx) => printEmit #[] arena' idx

/-- The slice-2 gate (pointfree genericity): the SAME `warpBank` combinator
    (`Tropical/EmitArrow.lean`), instantiated at the `ModalVoice` voice instead
    of `FixedSinOsc`, then run through the SAME emit recipe. Its plan JSON must
    be byte-identical to `diffcli emit-file stdlib/parsed/ReversibleComb.json`. -/
def emitarrowRevcombVerb (_args : List String) : IO UInt32 := do
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    match ← elabChain names with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      match Tropical.EmitArrow.buildReversibleComb arena resolved with
      | .error e => IO.eprintln s!"emitarrow-revcomb: {e}"; return 1
      | .ok (arena', idx) => printEmit #[] arena' idx

/-- The C1 "build-the-primitive" gate: `FixedSinOsc` built DIRECTLY by EmitArrow
    (`Tropical/EmitArrow.lean` `buildFixedSinOsc` — the fixed-point phasor +
    the `Sin` polynomial, inlined from scratch, NOT sourced as an instance), then
    run through the SAME emit recipe. Its plan JSON must be byte-identical to
    `diffcli emit-stdlib FixedSinOsc`. -/
def emitarrowFixedSinOscVerb (_args : List String) : IO UInt32 := do
  match ← manifestNames with
  | .error e => IO.eprintln e; return 1
  | .ok names =>
    match ← elabChain names with
    | .error e => IO.eprintln e; return 1
    | .ok (.error e) => printElabError e
    | .ok (.ok (arena, resolved)) =>
      match Tropical.EmitArrow.buildFixedSinOsc arena resolved with
      | .error e => IO.eprintln s!"emitarrow-fixedsinosc: {e}"; return 1
      | .ok (arena', idx) => printEmit #[] arena' idx

-- ── compile (Phase 6 stage 6d — the compile_patch.ts contract) ──────────────

/-- `diffcli compile <patch.json> [--mode=<m>]` → plan JSON on stdout.
    Boots the engine (stdlib from the pre-parsed bridge), loads the
    patch through the engine's own ingest, and rebuilds the plan from
    the mirror at the requested mode. The side-B command of
    diff_plan.ts / diff_audio.ts. -/
def compileVerb (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--")
    | IO.eprintln "usage: diffcli compile <patch.json> [--mode=fused|microkernel|microkernel-deep]"
      return 1
  let modeStr := (parseStrFlag args "--mode").getD "fused"
  let some mode := Tropical.Plan.CompilationMode.ofWire? modeStr
    | IO.eprintln s!"unknown compilation mode: {modeStr}"
      return 1
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM String := do
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
    and the IR path consume). -/
private def compileToFlatPlan (patch : String) :
    IO (Except String Tropical.Plan.FlatPlan) := do
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM Tropical.Plan.FlatPlan := do
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
    plan_5 JSON from `diffcli compile` serves as the browser-side manifest. -/
def compileWasmVerb (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--" && a.endsWith ".json")
    | IO.eprintln "usage: diffcli compile-wasm <patch.json> --out <out.wasm>"; return 1
  let some outPath := parseStrFlag args "--out"
    | IO.eprintln "compile-wasm: --out <path> required"; return 1
  match ← compileToFlatPlan patch with
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
  | "emit-ir" :: rest => emitIrVerb rest
  | "emit-msl" :: rest => emitMslVerb rest
  | "compile-wasm" :: rest => compileWasmVerb rest
  | "raise" :: rest => raiseVerb rest
  | "parsed-roundtrip" :: rest => parsedRoundtripVerb rest
  | "parse-md" :: rest => parseMdVerb rest
  | "voice-desugar" :: rest => voiceDesugarVerb rest
  | "parse-all" :: rest => parseAllVerb rest
  | "elab-stdlib" :: rest => elabStdlibVerb rest
  | "elab-file" :: rest => elabFileVerb rest
  | "strata-stdlib" :: rest => strataStdlibVerb rest
  | "strata-file" :: rest => strataFileVerb rest
  | "emit-stdlib" :: rest => emitStdlibVerb rest
  | "emit-file" :: rest => emitFileVerb rest
  | "emitarrow-flanger" :: rest => emitarrowFlangerVerb rest
  | "emitarrow-revcomb" :: rest => emitarrowRevcombVerb rest
  | "emitarrow-fixedsinosc" :: rest => emitarrowFixedSinOscVerb rest
  | "compile" :: rest => compileVerb rest
  | _ =>
    IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N]\n       diffcli raise <file.json>\n       diffcli parsed-roundtrip <file.json>\n       diffcli elab-stdlib <Name>\n       diffcli elab-file <parsed.json>\n       diffcli strata-stdlib <Name> [--upto=K] [--mode=M] [--type-args=J]\n       diffcli strata-file <parsed.json> [--upto=K] [--mode=M] [--type-args=J]\n       diffcli emit-stdlib <Name> [--type-args=J]\n       diffcli emit-file <parsed.json> [--type-args=J] [--cf-only]"
    return 1
