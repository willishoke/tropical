import Tropical.Ffi
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.Ir.Codec
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.CompileResolved
import Tropical.Ir.EmitLlvm
import Tropical.PlanDecode
import Tropical.Engine
import Tropical.Parse.Surface.Markdown

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
    | IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N]"
      return 1
  let frames := parseNatFlag args "--frames" 16
  let buffer := parseNatFlag args "--buffer" 256
  let planJson ← IO.FS.readFile planPath
  -- Lean owns codegen: parse the plan, emit IR, load via load_ir (planJson
  -- doubles as the metadata manifest). There is no C++ plan compiler.
  let plan ← match Lean.Json.parse planJson with
    | .error e => IO.eprintln s!"render-bytes: parse: {e}"; return 1
    | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
      | .error e => IO.eprintln s!"render-bytes: ofWire: {e}"; return 1
      | .ok p => pure p
  let ir ← match Tropical.Ir.EmitLlvm.emitKernel plan with
    | .ok s => pure s
    | .error e => IO.eprintln s!"render-bytes: emitKernel: {e}"; return 1
  let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
  rt.loadIr ir planJson
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
  let entries ← (System.FilePath.mk "stdlib").readDir
  let mdNames := entries.filterMap fun e =>
    let n := e.fileName
    if n.endsWith ".md" && n != "README.md" then some n else none
  let sorted := mdNames.qsort fun a b => decide (a < b)
  let mut parsed : Array (String × Tropical.Parse.Program) := #[]
  for fname in sorted do
    let text ← IO.FS.readFile s!"stdlib/{fname}"
    match Tropical.Parse.Surface.parseMarkdownProgram text with
    | .error e => IO.eprintln s!"{fname}: {e}"; return 1
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

private def parsedDir : String := "stdlib/parsed"

/-- Read + strictly decode a serialized ParsedProgram. Decode failures
    are harness errors (the TS side has no decoder to disagree with). -/
private def readParsed (path : String) : IO (Except String Tropical.Parse.Program) := do
  let text ← IO.FS.readFile path
  pure <| do
    let jv ← Tropical.Parse.JsonV.parse text |>.mapError (s!"JSON parse error: {·}")
    Tropical.Parse.decodeProgram jv

private def manifestNames : IO (Except String (Array String)) := do
  let text ← IO.FS.readFile s!"{parsedDir}/manifest.json"
  pure <| do
    let jv ← Tropical.Parse.JsonV.parse text |>.mapError (s!"manifest parse error: {·}")
    let some (Tropical.Parse.JsonV.arr items) := jv.getField? "programs"
      | .error "manifest missing 'programs' array"
    items.mapM fun
      | .str s => .ok s
      | _ => .error "manifest 'programs' entries must be strings"

/-- The raw-elaborated stdlib chain (elaborate-only — no strata), in
    manifest order, threading an external resolver over earlier results.
    Mirrors `elabChain` in scripts/diff/elab_cmd.ts. -/
private def elabChain (names : Array String) :
    IO (Except String (Except Tropical.Ir.ElabError
      (Tropical.Ir.Arena × Array (String × Tropical.Ir.ProgramIdx)))) := do
  let mut arena : Tropical.Ir.Arena := {}
  let mut resolved : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for name in names do
    match ← readParsed s!"{parsedDir}/{name}.json" with
    | .error e => return .error s!"{name}.json: {e}"
    | .ok prog =>
      let r := resolved
      match Tropical.Ir.elaborateInto arena prog
          (some fun n => (r.find? (·.1 == n)).map (·.2)) with
      | .error e => return .ok (.error e)
      | .ok (arena', idx) =>
        arena := arena'
        resolved := resolved.push (name, idx)
  return .ok (.ok (arena, resolved))

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
      if let .error e := Tropical.Ir.Core.check arena' root' then
        IO.eprintln e
        return 1
    printEncoded arena' root'

def strataStdlibVerb (args : List String) : IO UInt32 := do
  let some target := args.head?
    | IO.eprintln "usage: diffcli strata-stdlib <Name> [--upto=K] [--mode=M] [--type-args=J]"
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
    | IO.eprintln "usage: diffcli strata-file <parsed.json> [--upto=K] [--mode=M] [--type-args=J]"
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
  match Tropical.Ir.Strata.run opts arena root with
  | .error e =>
    IO.println (errorJson e.message).compress
    return 0
  | .ok (arena', root') =>
    match Tropical.Ir.Core.check arena' root' with
    | .error e =>
      IO.eprintln e
      return 1
    | .ok core =>
      match Tropical.Ir.CompileResolved.compileResolved core with
      | .error msg =>
        IO.println (errorJson msg).compress
        return 0
      | .ok plan =>
        match plan.toWire with
        | .error e => IO.eprintln e; return 1
        | .ok j => IO.println j.compress; return 0

def emitStdlibVerb (args : List String) : IO UInt32 := do
  let some target := args.head?
    | IO.eprintln "usage: diffcli emit-stdlib <Name> [--type-args=J]"
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
    | IO.eprintln "usage: diffcli emit-file <parsed.json> [--type-args=J]"
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

/-- `diffcli diff-ir <patch.json> [--frames N] [--buffer N]` — render the
    same compiled plan through both the plan path (load_plan) and the
    Lean-emitted-IR path (load_ir) and assert byte-identical audio. The
    Phase 1b gate. -/
def diffIrVerb (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--")
    | IO.eprintln "usage: diffcli diff-ir <patch.json> [--frames N] [--buffer N]"; return 1
  let frames := parseNatFlag args "--frames" 16
  let buffer := parseNatFlag args "--buffer" 256
  match ← compileToFlatPlan patch with
  | .error e => IO.eprintln e; return 1
  | .ok plan =>
    let manifest ← match plan.toWire with
      | .ok j => pure j.compress
      | .error e => IO.eprintln s!"toWire: {e}"; return 1
    let ir ← match Tropical.Ir.EmitLlvm.emitKernel plan with
      | .ok s => pure s
      | .error e => IO.eprintln s!"emitKernel: {e}"; return 1
    let render (load : Tropical.Ffi.Runtime → IO Unit) : IO ByteArray := do
      let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
      load rt
      let mut acc := ByteArray.empty
      for _ in [0:frames] do
        rt.process
        acc := acc ++ (← rt.outputBytes)
      pure acc
    let bytesA ← render (fun rt => rt.loadPlan manifest)
    let bytesB ← render (fun rt => rt.loadIr ir manifest)
    if bytesA == bytesB then
      IO.println s!"PASS  diff-ir  {patch}  ({bytesA.size} bytes)"
      return 0
    else
      let firstDiff := (List.range (min bytesA.size bytesB.size)).find?
        (fun i => bytesA.get! i != bytesB.get! i)
      IO.println s!"FAIL  diff-ir  {patch}  sizeA={bytesA.size} sizeB={bytesB.size} firstDiffByte={firstDiff}"
      return 1

/-- `diffcli diff-ir-plan <plan.json> [--frames N] [--buffer N]` — validate
    the plan_5 parser + IR emitter: render a serialized plan via load_plan
    and via ofWire→EmitLlvm→load_ir, assert byte-identical. The gate that
    proves Lean can take over plan→kernel before the C++ codegen is cut. -/
def diffIrPlanVerb (args : List String) : IO UInt32 := do
  let some planPath := args.find? (fun a => !a.startsWith "--" && a.endsWith ".json")
    | IO.eprintln "usage: diffcli diff-ir-plan <plan.json> [--frames N] [--buffer N]"; return 1
  let frames := parseNatFlag args "--frames" 16
  let buffer := parseNatFlag args "--buffer" 256
  let planJson ← IO.FS.readFile planPath
  let plan ← match Lean.Json.parse planJson with
    | .error e => IO.eprintln s!"parse: {e}"; return 1
    | .ok j => match Tropical.Plan.FlatPlan.ofWire j with
      | .error e => IO.eprintln s!"ofWire: {e}"; return 1
      | .ok p => pure p
  let ir ← match Tropical.Ir.EmitLlvm.emitKernel plan with
    | .ok s => pure s
    | .error e => IO.eprintln s!"emitKernel: {e}"; return 1
  let render (load : Tropical.Ffi.Runtime → IO Unit) : IO ByteArray := do
    let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
    load rt
    let mut acc := ByteArray.empty
    for _ in [0:frames] do rt.process; acc := acc ++ (← rt.outputBytes)
    pure acc
  let bytesA ← render (fun rt => rt.loadPlan planJson)
  let bytesB ← render (fun rt => rt.loadIr ir planJson)
  if bytesA == bytesB then
    IO.println s!"PASS  diff-ir-plan  {planPath}  ({bytesA.size} bytes)"; return 0
  else
    IO.println s!"FAIL  diff-ir-plan  {planPath}  sizeA={bytesA.size} sizeB={bytesB.size}"; return 1

def main (args : List String) : IO UInt32 := do
  match args with
  | "render-bytes" :: rest => renderBytes rest
  | "emit-ir" :: rest => emitIrVerb rest
  | "diff-ir" :: rest => diffIrVerb rest
  | "diff-ir-plan" :: rest => diffIrPlanVerb rest
  | "raise" :: rest => raiseVerb rest
  | "parsed-roundtrip" :: rest => parsedRoundtripVerb rest
  | "parse-md" :: rest => parseMdVerb rest
  | "parse-all" :: rest => parseAllVerb rest
  | "elab-stdlib" :: rest => elabStdlibVerb rest
  | "elab-file" :: rest => elabFileVerb rest
  | "strata-stdlib" :: rest => strataStdlibVerb rest
  | "strata-file" :: rest => strataFileVerb rest
  | "emit-stdlib" :: rest => emitStdlibVerb rest
  | "emit-file" :: rest => emitFileVerb rest
  | "compile" :: rest => compileVerb rest
  | _ =>
    IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N]\n       diffcli raise <file.json>\n       diffcli parsed-roundtrip <file.json>\n       diffcli elab-stdlib <Name>\n       diffcli elab-file <parsed.json>\n       diffcli strata-stdlib <Name> [--upto=K] [--mode=M] [--type-args=J]\n       diffcli strata-file <parsed.json> [--upto=K] [--mode=M] [--type-args=J]\n       diffcli emit-stdlib <Name> [--type-args=J]\n       diffcli emit-file <parsed.json> [--type-args=J]"
    return 1
