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

end Tropical.Ir.Strata
