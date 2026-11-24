import Tropical.PlanDecode
import Tropical.Semantics.Symbolic
open Tropical
def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: spec-of-plan <plan.json>"; return 1
  let some j := (Lean.Json.parse (← IO.FS.readFile path)).toOption | IO.eprintln "bad json"; return 1
  match Plan.FlatPlan.ofWire j with
  | .error e => IO.eprintln e; return 1
  | .ok plan =>
    match Semantics.Symbolic.specSmt plan s!"spec-of-plan {path}" with
    | .error e => IO.eprintln e; return 1
    | .ok t => IO.print t; return 0
