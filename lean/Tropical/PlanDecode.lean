import Lean.Data.Json
import Tropical.Plan

/-!
# PlanDecode — `tropical_plan_5 JSON → FlatPlan` (the inverse of `toWire`)

Phase 2 deletes the C++ codegen, so the plan-text → kernel capability moves
to Lean: `render-bytes <plan.json>` (and any consumer holding a serialized
plan rather than an in-memory `FlatPlan`) parses here, then emits IR via
`EmitLlvm` and loads it with `load_ir`. Mirrors `Plan.lean`'s `*.toWire`
field-for-field. plan_5 only (the session output shape); legacy plan_4 hand
fixtures are not parsed here.
-/

namespace Tropical.Plan

open Lean (Json JsonNumber)

private def optArr (j : Json) (k : String) : Array Json :=
  (((j.getObjVal? k).bind (·.getArr?)).toOption).getD #[]

private def scalarOfWire (j : Json) (k : String := "scalar_type") : Except String ScalarType := do
  let s ← (← j.getObjVal? k).getStr?
  match Tropical.Parse.ScalarKind.ofWire? s with
  | some t => pure t
  | none => .error s!"PlanDecode: bad scalar type '{s}'"

private def operandOfWire (j : Json) : Except String NOperand := do
  match ← (← j.getObjVal? "kind").getStr? with
  | "const" => pure (.const (← (← j.getObjVal? "val").getNum?) (← scalarOfWire j))
  | "input" => pure (.input (← (← j.getObjVal? "slot").getNat?) (← scalarOfWire j))
  | "reg" => pure (.reg (← (← j.getObjVal? "slot").getNat?) (← scalarOfWire j))
  | "array_reg" => pure (.arrayReg (← (← j.getObjVal? "slot").getNat?))
  | "session_array_reg" => pure (.sessionArrayReg (← (← j.getObjVal? "slot").getNat?))
  | "param" => pure (.param (← (← j.getObjVal? "ptr").getStr?) (← scalarOfWire j))
  | "source" => pure (.source (← (← j.getObjVal? "index").getNat?) (← scalarOfWire j))
  | "slot" => pure (.slot (← (← j.getObjVal? "index").getNat?) (← scalarOfWire j))
  | "loop_idx" => pure .loopIdx
  | k => .error s!"PlanDecode: bad operand kind '{k}'"

private def dstOfWire (j : Json) : Except String DstSlot := do
  let idx ← (← j.getObjVal? "dst").getNat?
  match ← (← j.getObjVal? "dst_kind").getStr? with
  | "temp" => pure (.temp idx)
  | "array" => pure (.array idx)
  | "moduleSlot" => pure (.moduleSlot idx)
  | k => .error s!"PlanDecode: bad dst_kind '{k}'"

private def instrOfWire (j : Json) : Except String NInstr := do
  let tag ← (← j.getObjVal? "tag").getStr?
  let dst ← dstOfWire j
  let args ← (optArr j "args").mapM operandOfWire
  let loopCount := ((j.getObjVal? "loop_count").bind (·.getNat?)).toOption.getD 1
  let strides := (optArr j "strides").map (fun x => (x.getNat?).toOption.getD 0)
  let resultType ← scalarOfWire j "result_type"
  pure { tag, dst, args, loopCount, strides, resultType }

private partial def instanceOfWire (j : Json) : Except String InstanceFunction := do
  let name ← (← j.getObjVal? "name").getStr?
  let inm ← (← j.getObjVal? "instance_name").getStr?
  let preamble ← (optArr j "preamble_instructions").mapM instrOfWire
  let instrs ← (optArr j "instructions").mapM instrOfWire
  let preInput ← (optArr j "pre_input_instructions").mapM instrOfWire
  let regOff ← (← j.getObjVal? "register_offset").getNat?
  let arrOff ← (← j.getObjVal? "array_slot_offset").getNat?
  let regCount ← (← j.getObjVal? "register_count").getNat?
  let children ← (optArr j "children").mapM instanceOfWire
  pure (.mk name inm preamble instrs preInput regOff arrOff regCount children)

private def sinkOfWire (j : Json) : Except String SinkSpec := do
  let inputs := (optArr j "inputs").map (fun x => (x.getNat?).toOption.getD 0)
  let gain := ((j.getObjVal? "gain").bind (·.getNum?)).toOption.getD defaultSinkGain
  let target := ((j.getObjVal? "target").bind (·.getNat?)).toOption.getD 0
  pure { inputs, gain, target }

private def sourceOfWire (j : Json) : Except String SourceKind := do
  match ← j.getStr? with
  | "tick" => pure .tick
  | "rate" => pure .rate
  | s => .error s!"PlanDecode: bad source kind '{s}'"

private def strArr (j : Json) (k : String) : Array String :=
  (optArr j k).map (fun x => (x.getStr?).toOption.getD "")

private def natArr (j : Json) (k : String) : Array Nat :=
  (optArr j k).map (fun x => (x.getNat?).toOption.getD 0)

/-- Parse a tropical_plan_5 JSON object into a `FlatPlan`. -/
def FlatPlan.ofWire (j : Json) : Except String FlatPlan := do
  let sampleRate := (((j.getObjVal? "config").bind (·.getObjVal? "sampleRate")).bind
    (·.getNum?)).toOption.getD (44100 : JsonNumber)
  let mode := (((j.getObjVal? "compilation_mode").bind (·.getStr?)).toOption.bind
    CompilationMode.ofWire?).getD .fused
  let instFns ← (optArr j "instance_functions").mapM instanceOfWire
  let sinks ← (optArr j "sinks").mapM sinkOfWire
  let sources ← if (j.getObjVal? "sources").toOption.isSome
    then (optArr j "sources").mapM sourceOfWire
    else pure defaultSources
  let registerCount := ((j.getObjVal? "register_count").bind (·.getNat?)).toOption.getD 0
  let arraySlotCount := ((j.getObjVal? "array_slot_count").bind (·.getNat?)).toOption.getD 0
  let slotCount := ((j.getObjVal? "slot_count").bind (·.getNat?)).toOption.getD 0
  let slotDefaults := optArr j "slot_defaults"
  pure {
    sampleRate, compilationMode := mode,
    arraySlotNames := strArr j "array_slot_names",
    registerCount, arraySlotCount,
    arraySlotSizes := natArr j "array_slot_sizes",
    instanceFunctions := instFns,
    sinks, sources,
    slotCount,
    slotNames := strArr j "slot_names",
    slotDefaults }

end Tropical.Plan
