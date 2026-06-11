import Tropical.Ir.Nodes

/-!
# Strata shared types — Options + Error

Split from `Tropical/Ir/Strata.lean` (the orchestrator) so individual
pass modules under `Tropical/Ir/Strata/` can reference them without an
import cycle.
-/

namespace Tropical.Ir.Strata

open Lean (JsonNumber)

structure Options where
  /-- Run passes `1..upto` (0 = elaborate-only prefix). -/
  upto : Nat := 0
  /-- `false` = the fractal session path: inlineInstances is skipped. -/
  inlineNested : Bool := true
  /-- Type args by NAME (raw numbers; validation — including unknown
      names and non-integers — is specialize's job, with byte-exact
      TS error messages). -/
  typeArgs : Array (String × JsonNumber) := #[]
deriving Repr, Inhabited

/-- A strata error is a comparable output: the TS error message,
    byte-exact. -/
structure Error where
  message : String
deriving Repr, Inhabited

open Tropical.Ir in
/-- Port of decl_tables.ts `getInstanceType` (shared by
    inlineInstances and arrayLower). -/
def getInstanceType (arena : Arena) (enclosing : Program)
    (instName typeKey : String) : Except Error (ProgramIdx × Program) := do
  match enclosing.registryGet? typeKey with
  | some pIdx =>
    let some p := arena.program? pIdx
      | throw ⟨s!"getInstanceType: instance '{instName}' typeKey '{typeKey}' program pool index {pIdx.idx} out of range (internal)"⟩
    return (pIdx, p)
  | none =>
    let keys := ", ".intercalate (enclosing.registry.toList.map (·.1))
    throw ⟨s!"getInstanceType: instance '{instName}' typeKey '{typeKey}' " ++
      s!"not found in enclosing program '{enclosing.name}' registry " ++
      s!"(keys: {keys}). This is a registry-build bug; check buildProgramRegistry call sites."⟩

end Tropical.Ir.Strata
