import Tropical.Ffi
import Tropical.StagedLoad
import Tropical.Session.Compile
import Tropical.Playground
import Tropical.Plan
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitMsl
import Tropical.PlanDecode
import Tropical.Parse.Raise
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.CompileResolved
import Tropical.Compile
import Tropical.EmitArrow
import Tropical.Testing.ArrowFixtures
import Tropical.Testing.PlanWire
import Tropical.Tropicaltest.Gates

/-!
# Tropical.Tropicaltest.Harness

The host side of the shared test substrate: PCM render through the native
runtime, and the `compilePatch*` drivers over the pure session compile
(`Tropical.Session.compilePatch`, the same function the engine and
`tropicalc` run). This module knows the compiler and a host; it does not
know the session engine. The verdict combinators
(`passGate`/`failGate`/`hashGate`, `bitDiffCount`/`energyOf`, `sha256Hex`)
live in `Gates`, imported here, so suites that never render take only that.
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

/-- Compile a tropical_program_2 patch (by path) to its plan and typed stage
    blocks, over a fresh stdlib registry each time (a value, so there is
    nothing to reuse across loads). -/
def compilePatchStaged (path : String) :
    IO (Except String (Tropical.Plan.FlatPlan
      × Array (Array (Option Tropical.Ir.Stage)))) := do
  let text ← IO.FS.readFile path
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => throw (IO.userError s!"stdlib boot failed: {f.toJson.compress}")
    | .ok st => pure st
  match (Tropical.Session.compilePatch text).runOn registry with
  | (.ok r, _) => pure (.ok r)
  | (.error f, _) => pure (.error f.toJson.compress)

/-- Compile a tropical_program_2 patch (by path) to a plan in `mode`. -/
def compilePatch (path : String) (mode : Tropical.Plan.CompilationMode) :
    IO (Except String String) := do
  let text ← IO.FS.readFile path
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => throw (IO.userError s!"stdlib boot failed: {f.toJson.compress}")
    | .ok st => pure st
  match (Tropical.Session.compilePatch text mode).runOn registry with
  | (.error f, _) => pure (.error f.toJson.compress)
  | (.ok (plan, _), _) =>
    match plan.toWire with
    | .error msg => pure (.error msg)
    | .ok j => pure (.ok j.compress)

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
