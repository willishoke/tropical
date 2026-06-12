import Tropical.Ffi
import Tropical.Engine
import Tropical.Plan
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
    samples), matching `validate_stdlib`'s buffer. -/
def renderPlanBytes (planJson : String) : IO ByteArray := do
  let rt ← Tropical.Ffi.Runtime.new BUFFER.toUInt32
  rt.loadPlan planJson
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

private def firstLine (s : String) : String := (s.splitOn "\n").headD ""

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

  -- ── (c) Native mode-equiv: fused ≡ microkernel ≡ microkernel-deep ──────────
  IO.println "native mode-equiv (fused vs microkernel vs deep):"
  for fixture in ← sortedNames "tests/fixtures/flat_plan" ".json" do
    let fixText ← IO.FS.readFile s!"tests/fixtures/flat_plan/{fixture}.json"
    let some input := ((Lean.Json.parse fixText).toOption.bind (·.getObjVal? "input" |>.toOption))
      | pure ()
    let tmpPatch := "/tmp/tropicaltest-equiv.json"
    IO.FS.writeFile tmpPatch input.compress
    total := total + 1
    let modes := #[Tropical.Plan.CompilationMode.fused, .microkernel, .microkernelDeep]
    let mut bytesPerMode : Array ByteArray := #[]
    let mut ok := true
    for mode in modes do
      match ← compilePatch tmpPatch mode with
      | .error e => IO.println s!"  FAIL  {fixture}  compile {mode.wire}: {firstLine e}"; ok := false
      | .ok planJson => bytesPerMode := bytesPerMode.push (← renderPlanBytes planJson)
    if ok then
      let allEqual := bytesPerMode.all (· == bytesPerMode[0]!)
      if allEqual then IO.println s!"  PASS  {fixture}  ({bytesPerMode.size} modes agree)"
      else IO.println s!"  FAIL  {fixture}  modes disagree"; ok := false
    if !ok then failed := failed + 1

  IO.println ""
  IO.println s!"{total - failed}/{total} passed"
  return if failed == 0 then 0 else 1
