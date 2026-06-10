import Tropical.Ffi

/-!
The `diffcli` executable — differential-harness verbs that exercise the
Lean-owned native runtime.

    diffcli render-bytes <plan.json> [--frames N] [--buffer N]

Loads a tropical_plan_5 JSON file into a fresh runtime, renders
`frames × buffer` samples, and writes the raw little-endian float64
stream to stdout. `… | shasum -a 256` reproduces the golden hashes in
tests/golden/ when fed the same plans validate_stdlib.ts renders
(frames=16, buffer=256) — the Phase 2 gate that the Lean FFI path is
sample-for-sample the TS koffi path.
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

def main (args : List String) : IO UInt32 := do
  match args with
  | "render-bytes" :: rest => renderBytes rest
  | _ =>
    IO.eprintln "usage: diffcli render-bytes <plan.json> [--frames N] [--buffer N]"
    return 1
