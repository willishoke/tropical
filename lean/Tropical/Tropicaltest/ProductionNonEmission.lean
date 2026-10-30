import Tropical.Tropicaltest.Synthetic
import Tropical.NativeBoot
import Tropical.Testing.EngineMirror

/-!
# Tropical.Tropicaltest.ProductionNonEmission

The production non-emission gate: representative production paths — a
stdlib builder's per-instance plan, a playground graph, a program_2 session
and an exported composite re-instantiated into that session — must stay
unable to emit legacy state. Two of its probes drive the session engine
(load, export_program, add_instance), which is why this gate lives apart
from `Synthetic`: it is the one suite module that needs the engine, and the
suite chain behind `Synthetic` does not.
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

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
  | .ok (.str "tropical_plan_7") => pure ()
  | .ok got =>
    throw s!"{path}: production plan schema is {got.compress}, expected tropical_plan_7"
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
  let env ← Tropical.NativeBoot.boot
  let action : Tropical.EngineM FlatPlan := do
    let _ ← Tropical.Engine.handleLoad env (Lean.Json.mkObj [("path", .str path)])
    Tropical.Engine.compileMirrorFlatPlan env .fused
  match ← action.run with
  | .ok plan => pure (.ok plan)
  | .error failure => pure (.error failure.toJson.compress)

private def compileExportProbe : IO (Except String FlatPlan) := do
  let env ← Tropical.NativeBoot.boot
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
      "builder + playground + program_2/session + export emit typed state-free plan_7 only"
  else
    failGate "production-non-emission" (String.intercalate "; " failures.toList)

end ProductionNonEmission
