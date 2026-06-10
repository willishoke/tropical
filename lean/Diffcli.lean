import Tropical.Ffi
import Tropical.Parse.Raise

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
  let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
  rt.loadPlan planJson
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

def main (args : List String) : IO UInt32 := do
  match args with
  | "render-bytes" :: rest => renderBytes rest
  | "raise" :: rest => raiseVerb rest
  | "parsed-roundtrip" :: rest => parsedRoundtripVerb rest
  | _ =>
    IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N]\n       diffcli raise <file.json>\n       diffcli parsed-roundtrip <file.json>"
    return 1
