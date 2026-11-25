import Tropical.PlanDecode
import Tropical.Semantics.Scalar
import Tropical.Semantics.Plan

/-! The concrete reference (`Semantics.observeFlatPlan` over
`Semantics.Scalar.algebra`, the sink image) on input vectors read from a file:
one line per vector, `tau p1 p2 ...` with the `param:` slots in slot order.
Prints one line per vector with the sink values. Used to test the symbolic
carrier differentially (a bounded check, not a proof). -/

open Tropical

def main (args : List String) : IO UInt32 := do
  let [planPath, vecPath] := args | IO.eprintln "usage: ref-of-plan <plan.json> <vectors>"; return 1
  let some j := (Lean.Json.parse (← IO.FS.readFile planPath)).toOption | IO.eprintln "bad json"; return 1
  let .ok plan := Plan.FlatPlan.ofWire j | IO.eprintln "bad plan"; return 1
  let paramSlots := (List.range plan.slotCount).filter fun i =>
    (plan.slotNames[i]?.getD "").startsWith "param:"
  for line in (← IO.FS.readFile vecPath).splitOn "\n" do
    let ws := (line.splitOn " ").filter (· ≠ "") |>.map String.toInt!
    if ws.isEmpty then continue
    let tau := ws[0]!
    let ps := ws.drop 1
    let slots : Array (Semantics.Value Ir.CVal) := (List.range plan.slotCount).toArray.map fun i =>
      match paramSlots.idxOf? i with
      | some k =>
        if ((plan.slotType? i).map (·.kind)) == some .int then .scalar (.i (Ir.wrap64 (ps[k]!)))
        else .scalar (.i32 (Ir.wrap32 (ps[k]!)))
      | none => match plan.slotDefaults[i]? with
        | some (.num v) => .scalar (.f v.toFloat)
        | _ => .scalar (.f 0.0)
    let sources : Array (Semantics.Value Ir.CVal) := plan.sources.map fun
      | .tau => .scalar (.i (Ir.wrap64 tau))
      | .rate => .scalar (.f plan.sampleRate.toFloat)
    let arrays : Array (Array (Semantics.Value Ir.CVal)) :=
      plan.arraySlotSizes.map fun n => Array.replicate n (.scalar (.f 0.0))
    let inputs : Semantics.PlanInputs Ir.CVal :=
      { sources := sources, initialSlots := slots, initialArrays := arrays }
    match Semantics.observeFlatPlan Semantics.Scalar.algebra inputs plan with
    | .error e => IO.println s!"ERR {e.operation}: {e.detail}"
    | .ok obs =>
      let vals := obs.sinks.toList.map fun
        | .scalar (.f x) => toString x | .scalar (.i n) => toString n
        | .scalar (.i32 n) => toString n | .scalar (.b b) => toString b | _ => "?"
      IO.println (" ".intercalate vals)
  return 0
