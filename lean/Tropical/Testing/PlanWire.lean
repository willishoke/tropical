import Tropical.Plan

/-!
# PlanWire — the per-program plan's wire encoding (test support, not production)

`PerInstancePlan` has no production wire format: the session compiler packs
per-instance plans into a `FlatPlan` and only THAT is serialized. This
encoding exists so a single program's emit is printable and byte-comparable
on its own — the `diffcli emit-stdlib`/`emit-file`/`emitarrow-*` gate verbs
print it, and byte-gates diff two such encodings. Nothing on the production
compile path imports this module.

Stays in `namespace Tropical.Plan` so call sites read `plan.toWire` exactly
like the production `FlatPlan.toWire`.
-/

namespace Tropical.Plan

open Lean (Json toJson)

/-- The per-program plan as plain JSON — the byte-comparable form the
    emit gates print and diff. -/
def PerInstancePlan.toWire (p : PerInstancePlan) : Except String Json := do
  return Json.mkObj [
    ("register_count", toJson p.registerCount),
    ("array_slot_count", toJson p.arraySlotCount),
    ("array_slot_sizes", toJson p.arraySlotSizes),
    ("instructions", ← instrsToWire p.instructions),
    ("per_child_pre_input", Json.arr (← p.perChildPreInput.mapM instrsToWire)),
    ("output_targets", toJson p.outputTargets),
    ("array_slot_names", toJson p.arraySlotNames)]

end Tropical.Plan
