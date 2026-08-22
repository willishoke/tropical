import Std.Data.HashMap
import Tropical.Ir.Stage0

/-!
# TileStage — absolute-time endpoint residualization

This is deliberately separate from Stage0.  Stage0 first removes only
τ-independent work and retains its sample-zero/control-write contract.  This
pass then roots a second dependency slice at `tile_array_slots`, classifies only
that slice as materializer-time, and delegates the mechanical residualization
(regions, scalar boundaries, array crossings, block rebuilding) to the typed
Stage0 residualization machinery.  Its proved surface currently covers the
alignment/refusal and empty-selection boundaries; deeper publication
refinement is stated separately in `TileStageLaws`.
-/

namespace Tropical.Ir.TileStage

open Tropical.Plan
open Std (HashMap)

structure Split where
  audio : FlatPlan
  tile? : Option FlatPlan
deriving Inhabited

private def dependencyStages (plan : FlatPlan)
    (blocks : Array (Array NInstr)) : Array (Array (Option Tropical.Ir.Stage)) :=
  Id.run do
    let flat := blocks.flatten
    let mut tempDef : HashMap Nat Nat := {}
    let mut slotDef : HashMap Nat Nat := {}
    let mut arrayDefs : HashMap Nat (Array Nat) := {}
    let mut deps : Array (Array Nat) := #[]
    let mut roots : Array Nat := #[]
    for i in [0:flat.size] do
      let instr := flat[i]!
      let mut ds : Array Nat := #[]
      for arg in instr.args do
        match arg with
        | .reg t _ => if let some d := tempDef.get? t then ds := ds.push d
        | .slot s _ => if let some d := slotDef.get? s then ds := ds.push d
        | .arrayReg s => ds := ds ++ arrayDefs.getD s #[]
        | _ => pure ()
      deps := deps.push ds
      match instr.dst with
      | .temp t => tempDef := tempDef.insert t i
      | .moduleSlot s => slotDef := slotDef.insert s i
      | .array s =>
        arrayDefs := arrayDefs.insert s ((arrayDefs.getD s #[]).push i)
        if plan.tileArraySlots.contains s then roots := roots.push i
      | _ => pure ()
    let mut selected := Array.replicate flat.size false
    let mut work := roots
    while !work.isEmpty do
      let i := work.back!
      work := work.pop
      if !selected[i]! then
        selected := selected.set! i true
        work := work ++ deps[i]!
    -- A hash-consed scalar can feed both the tile image and surviving audio
    -- work.  Moving that shared definition outright would create a scalar
    -- `coef:` crossing whose value changes at tile time; the Metal worker only
    -- publishes tile arrays.  Retain and duplicate the complete scalar support
    -- slice instead.  Stage0's fold closure is exactly the mechanical form we
    -- need here: one copy remains in audio, while one copy follows the image
    -- writer into the materializer.  Array writers are the intentional
    -- crossing and therefore stay s0.
    let isScalarDef (i : Nat) : Bool :=
      match flat[i]!.dst with
      | .temp _ | .moduleSlot _ => true
      | _ => false
    let mut retained := Array.replicate flat.size false
    let mut retainedWork : Array Nat := #[]
    for user in [0:flat.size] do
      if !selected[user]! then
        for dependency in deps[user]! do
          if selected[dependency]! && isScalarDef dependency then
            retainedWork := retainedWork.push dependency
    while !retainedWork.isEmpty do
      let i := retainedWork.back!
      retainedWork := retainedWork.pop
      if !retained[i]! then
        retained := retained.set! i true
        for dependency in deps[i]! do
          if selected[dependency]! && isScalarDef dependency then
            retainedWork := retainedWork.push dependency
    let mut cursor := 0
    let mut result : Array (Array (Option Tropical.Ir.Stage)) := #[]
    for block in blocks do
      let mut stages : Array (Option Tropical.Ir.Stage) := #[]
      for _ in block do
        stages := stages.push (some (if retained[cursor]! then .fold
          else if selected[cursor]! then .s0 else .s1))
        cursor := cursor + 1
      result := result.push stages
    return result

/-- Residualize the marked endpoint images.  Identity for ordinary plans. -/
def split (plan : FlatPlan) : Except String Split := do
  if plan.tileArraySlots.isEmpty then return { audio := plan, tile? := none }
  let mut blocks : Array (Array NInstr) := #[]
  for fn in plan.instanceFunctions do
    blocks := blocks ++ Tropical.Ir.Stage0.collectBlocks fn
  let staged ← Tropical.Ir.Stage0.hoistTyped plan (dependencyStages plan blocks)
  let some tile := staged.coeff?
    | throw "TileStage: endpoint roots produced no materializer instructions"
  -- Stage0's generic rebuild labels every residualized array as a coefficient
  -- column. Restore the first-stage set here: tile columns have their own
  -- lifetime and are appended per dispatch by the worker.
  let audio := { staged.audio with
    coeffArraySlots := plan.coeffArraySlots }
  return { audio, tile? := some tile }

end Tropical.Ir.TileStage
