import Tropical.Ffi
import Tropical.Engine
import Tropical.StagedLoad
import Tropical.Playground
import Tropical.Plan
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.PlanDecode
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.CompileResolved
import Tropical.Compile
import Tropical.EmitArrow
import Tropical.Testing.ArrowFixtures
import Tropical.Testing.EngineMirror
import Tropical.Testing.PlanWire
import Lean.Data.Json

/-!
# Tropical.Tropicaltest.Harness

Shared test substrate: PCM render + sha256, the `compilePatch*` drivers, and the gate combinators (`passGate`/`failGate`/`hashGate`, `bitDiffCount`/`energyOf`). Every suite draws on these.
-/

open Tropical

def FRAMES : Nat := 16
def BUFFER : Nat := 256

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

/-- `compilePatch` with the test-fixture programs (`OpZoo`) registered on top
    of the boot chain — the `diffcli … --fixtures` path, for patches that
    instantiate a fixture by name. -/
def compilePatchFixtures (path : String) (mode : Tropical.Plan.CompilationMode) :
    IO (Except String String) := do
  let env ← Tropical.Engine.boot
  let act : Tropical.EngineM String := do
    Tropical.Engine.registerTestFixtures env
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", Lean.Json.str path)])
    Tropical.Engine.compileMirrorPlan env mode
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

def firstLine (s : String) : String := (s.splitOn "\n").headD ""

-- ── Gate combinators ─────────────────────────────────────────────────────────
-- Every gate's verdict funnels through these two, so the protocol line the
-- runner (and the label-diff check) depends on — `  PASS  <label>  <detail>` /
-- `  FAIL  <label>  <detail>` — is typed in exactly one place.

/-- Print the gate's PASS line and return `true`. -/
def passGate (label msg : String) : IO Bool := do
  IO.println s!"  PASS  {label}  {msg}"
  pure true

/-- Print the gate's FAIL line and return `false`. -/
def failGate (label msg : String) : IO Bool := do
  IO.println s!"  FAIL  {label}  {msg}"
  pure false

/-- Bit-differing sample count over the two arrays' common prefix
    (IEEE `!=` — the pointwise-equality reading the byte-gates use). -/
def bitDiffCount (a b : Array Float) : Nat := Id.run do
  let mut d := 0
  for i in [0:min a.size b.size] do
    if a[i]! != b[i]! then d := d + 1
  return d

/-- Σx² over the whole array. -/
def energyOf (xs : Array Float) : Float :=
  xs.foldl (fun acc s => acc + s * s) 0.0

/-- The hash-equivalence epilogue: sha256 both renders and PASS iff they
    agree, with the gate's own verdict texts built from the hash(es). -/
def hashGate (label : String) (lhs rhs : ByteArray)
    (passMsg : String → String) (failMsg : String → String → String) : IO Bool := do
  let lhsHash ← sha256Hex lhs
  let rhsHash ← sha256Hex rhs
  if lhsHash == rhsHash then passGate label (passMsg lhsHash)
  else failGate label (failMsg lhsHash rhsHash)
