import Lean.Data.Json
import Tropical.Plan
import Tropical.Parse.OrderedJson

/-!
# PlanDecode — `tropical_plan_6 JSON → FlatPlan` (the inverse of `toWire`)

Phase 2 deletes the C++ codegen, so the plan-text → kernel capability moves
to Lean: `render-bytes <plan.json>` (and any consumer holding a serialized
plan rather than an in-memory `FlatPlan`) parses here, then emits IR via
`EmitLlvm` and loads it with `load_ir`. Mirrors `Plan.lean`'s `*.toWire`
field-for-field. Plan 6 is the only accepted runtime schema.

Decodes over `JsonV` (the array-backed twin with the `sizeOf` lemmas), so
the instance-tree recursion is total by descent on the document — the
codec-decoder pattern. The public entry keeps its `Lean.Json` signature and
reparses through `JsonV` (semantically the identity).
-/

namespace Tropical.Plan

open Lean (Json JsonNumber)
open Tropical.Parse (JsonV)

private def optArr (j : JsonV) (k : String) : Array JsonV :=
  match j.getField? k with
  | some (.arr a) => a
  | _ => #[]

private def reqStr (j : JsonV) (k : String) : Except String String :=
  match j.getField? k with
  | some (.str s) => pure s
  | _ => .error s!"PlanDecode: missing string field '{k}'"

private def reqNat (j : JsonV) (k : String) : Except String Nat :=
  match j.getField? k with
  | some (.num n) => pure n.toFloat.toUInt64.toNat
  | _ => .error s!"PlanDecode: missing numeric field '{k}'"

private def optNat (j : JsonV) (k : String) (dflt : Nat) : Nat :=
  match j.getField? k with
  | some (.num n) => n.toFloat.toUInt64.toNat
  | _ => dflt

private def rejectRetiredFields (j : JsonV) : Except String Unit := do
  for field in #[
      "state_init", "register_names", "register_types", "register_targets",
      "state_reg_offset", "output_targets", "outputs", "instructions",
      "scheduler_function"] do
    if (j.getField? field).isSome then
      throw s!"PlanDecode: retired field '{field}' is not valid in tropical_plan_6"

private def optNum (j : JsonV) (k : String) (dflt : JsonNumber) : JsonNumber :=
  match j.getField? k with
  | some (.num n) => n
  | _ => dflt

private def scalarOfWire (j : JsonV) (k : String := "scalar_type") : Except String ScalarType := do
  let s ← reqStr j k
  match Tropical.Parse.ScalarKind.ofWire? s with
  | some t => pure t
  | none => .error s!"PlanDecode: bad scalar type '{s}'"

private def operandOfWire (j : JsonV) : Except String NOperand := do
  match ← reqStr j "kind" with
  | "const" =>
    let v ← match j.getField? "val" with
      | some (.num n) => pure n
      | _ => .error "PlanDecode: missing numeric field 'val'"
    pure (.const v (← scalarOfWire j))
  | "input" => pure (.input (← reqNat j "slot") (← scalarOfWire j))
  | "reg" => pure (.reg (← reqNat j "slot") (← scalarOfWire j))
  | "array_reg" => pure (.arrayReg (← reqNat j "slot"))
  | "session_array_reg" => pure (.sessionArrayReg (← reqNat j "slot"))
  | "param" => pure (.param (← reqStr j "ptr") (← scalarOfWire j))
  | "source" => pure (.source (← reqNat j "index") (← scalarOfWire j))
  | "slot" => pure (.slot (← reqNat j "index") (← scalarOfWire j))
  -- Canonical single-region plans omit the zero-valued binder id.
  | "loop_idx" => pure (.loopIdx (optNat j "id" 0))
  | k => .error s!"PlanDecode: bad operand kind '{k}'"

private def dstOfWire (j : JsonV) : Except String DstSlot := do
  let idx ← reqNat j "dst"
  match ← reqStr j "dst_kind" with
  | "temp" => pure (.temp idx)
  | "array" => pure (.array idx)
  | "moduleSlot" => pure (.moduleSlot idx)
  | k => .error s!"PlanDecode: bad dst_kind '{k}'"

private def instrOfWire (j : JsonV) : Except String NInstr := do
  let tag ← reqStr j "tag"
  let dst ← dstOfWire j
  let args ← (optArr j "args").mapM operandOfWire
  let loopCount := optNat j "loop_count" 1
  -- Canonical single-region plans omit the zero-valued binder id.
  let loopId := optNat j "loop_id" 0
  let strides := (optArr j "strides").map fun x =>
    match x with | .num n => n.toFloat.toUInt64.toNat | _ => 0
  let resultType ← scalarOfWire j "result_type"
  let routedOutputCount := optNat j "output_count" 0
  let routedRoutes : Array (Option Nat) ← if tag == "RoutedSumBegin" then
      (optArr j "routes").mapM fun route => match route with
        | .null => pure none
        | .num n => pure (some n.toFloat.toUInt64.toNat)
        | _ => throw "PlanDecode: routed route must be a natural number or null"
    else pure #[]
  pure ({ tag := tag, dst := dst, args := args, loopCount := loopCount,
          strides := strides, resultType := resultType, loopId := loopId,
          routedOutputCount := routedOutputCount,
          routedRoutes := routedRoutes } : NInstr)

private def instanceOfWire (j : JsonV) : Except String InstanceFunction := do
  let name ← reqStr j "name"
  let inm ← reqStr j "instance_name"
  let preamble ← (optArr j "preamble_instructions").mapM instrOfWire
  let instrs ← (optArr j "instructions").mapM instrOfWire
  let preInput ← (optArr j "pre_input_instructions").mapM instrOfWire
  let regOff ← reqNat j "register_offset"
  let arrOff ← reqNat j "array_slot_offset"
  let regCount ← reqNat j "register_count"
  let children ← match _hf : j.getField? "children" with
    | some (.arr items) =>
      items.attach.mapM fun ⟨c, _⟩ => instanceOfWire c
    | _ => pure #[]
  pure (.mk name inm preamble instrs preInput regOff arrOff regCount children)
termination_by sizeOf j
decreasing_by
  have := Tropical.Parse.JsonV.sizeOf_lt_of_getField _hf
  have := Array.sizeOf_lt_of_mem ‹_ ∈ items›
  simp_all <;> omega

private def sinkOfWire (j : JsonV) : Except String SinkSpec := do
  let inputs := (optArr j "inputs").map fun x =>
    match x with | .num n => n.toFloat.toUInt64.toNat | _ => 0
  let gain := optNum j "gain" defaultSinkGain
  let target := optNat j "target" 0
  pure { inputs, gain, target }

private def sourceOfWire (j : JsonV) : Except String SourceKind := do
  match j with
  | .str "tick" => pure .tick
  | .str "rate" => pure .rate
  | .str "tile_phase" => pure .tilePhase
  | .str "tile_tick" => pure .tileTick
  | .str s => .error s!"PlanDecode: bad source kind '{s}'"
  | _ => .error "PlanDecode: source kind must be a string"

private def strArr (j : JsonV) (k : String) : Array String :=
  (optArr j k).map fun x => match x with | .str s => s | _ => ""

private def natArr (j : JsonV) (k : String) : Array Nat :=
  (optArr j k).map fun x => match x with | .num n => n.toFloat.toUInt64.toNat | _ => 0

private def ofWireV (j : JsonV) : Except String FlatPlan := do
  let schema ← reqStr j "schema"
  if schema != "tropical_plan_6" then
    throw s!"PlanDecode: unsupported schema '{schema}'; expected 'tropical_plan_6'"
  rejectRetiredFields j
  let sampleRate := match (j.getField? "config").bind (·.getField? "sampleRate") with
    | some (.num n) => n
    | _ => (44100 : JsonNumber)
  let mode := (match j.getField? "compilation_mode" with
    | some (.str s) => CompilationMode.ofWire? s
    | _ => none).getD .fused
  let instFns ← (optArr j "instance_functions").mapM instanceOfWire
  let sinks ← (optArr j "sinks").mapM sinkOfWire
  let outputChannelCount := optNat j "output_channel_count" 1
  let sources ← if (j.getField? "sources").isSome
    then (optArr j "sources").mapM sourceOfWire
    else pure defaultSources
  let registerCount := optNat j "register_count" 0
  let arraySlotCount := optNat j "array_slot_count" 0
  let slotCount := optNat j "slot_count" 0
  let slotDefaults := (optArr j "slot_defaults").map (·.toJson)
  let phaserTimeStaging := match j.getField? "phaser_time_staging" with
    | some (.str reason) => some reason
    | _ => none
  let plan : FlatPlan := {
    sampleRate, compilationMode := mode,
    arraySlotNames := strArr j "array_slot_names",
    registerCount, arraySlotCount,
    arraySlotSizes := natArr j "array_slot_sizes",
    instanceFunctions := instFns,
    sinks, outputChannelCount, sources,
    slotCount,
    slotNames := strArr j "slot_names",
    slotDefaults,
    coeffArraySlots := natArr j "coeff_array_slots",
    tileArraySlots := natArr j "tile_array_slots",
    tileIntervalFrames := optNat j "tile_interval_frames" 0,
    phaserTimeStaging }
  if !plan.outputLayoutWellFormed then
    throw "PlanDecode: output channel count must be positive and sink targets must be unique and in range"
  pure plan

/-- Parse a tropical_plan_6 JSON object into a `FlatPlan`. -/
def FlatPlan.ofWire (j : Json) : Except String FlatPlan := do
  match Tropical.Parse.JsonV.parse j.compress with
  | .error e => .error s!"PlanDecode: internal reparse failure: {e}"
  | .ok jv => ofWireV jv

end Tropical.Plan
