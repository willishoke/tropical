import Tropical.Ir.TileStage

/-!
# Structural laws for TileStage

Tile residualization is opt-in through `tileArraySlots`.  The empty case is a
proved structural and semantic identity; it is important because exact/JIT
plans and unadmitted shapes use that path as their reference behavior.

No statement in this module equates polynomial interpolation over a complete
tile with exact expression evaluation.  That comparison remains a measured
tolerance gate.
-/

namespace Tropical.Ir.TileStage

open Tropical.Plan

theorem split_identity_of_no_tile_arrays (plan : FlatPlan)
    (h : plan.tileArraySlots.isEmpty = true) :
    split plan = .ok { audio := plan, tile? := none } := by
  simp [split, h]
  rfl

theorem split_identity_audio_of_no_tile_arrays (plan : FlatPlan)
    (h : plan.tileArraySlots.isEmpty = true) :
    (split plan).map Split.audio = .ok plan := by
  rw [split_identity_of_no_tile_arrays plan h]
  rfl

end Tropical.Ir.TileStage
