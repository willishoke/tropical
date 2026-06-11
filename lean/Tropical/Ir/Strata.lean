import Tropical.Ir.Nodes

/-!
# Strata pipeline — port of compiler/ir/strata.ts (Phase 5)

The six-pass drop sequence over the resolved IR:

  assertAcyclic → specialize → sumLower → inlineInstances
                → arrayLower → identityElim

Passes land one stage at a time behind the `diff-strata` hybrid gate
(scripts/diff/diff_strata.ts): Lean runs passes `1..upto`, the result
ships through the `tropical_resolved_1` codec, and the TS suffix
completes the pipeline — only final post-strata output is compared, so
a divergence at stage K localizes to pass K. `portedPasses` is the
ratchet: the harness refuses an `upto` beyond it (a harness error, not
a comparable `{error}` output).

Pass numbering (K = passes completed):
  1 specialize (incl. the entry acyclicity assertion)
  2 sumLower
  3 inlineInstances (skipped when `inlineNested := false` — the
    fractal session path; InstanceDecls survive as kernel boundaries)
  4 arrayLower
  5 identityElim

Passes are `Arena → ProgramIdx → …` maps that may push fresh programs
into the pool and abandon old ones — the codec encoder pools ids on
first reference from the root, so unreachable entries are never
emitted.

`Error` carries the byte-exact TS error message (specialize's
validation errors, AcyclicityViolation, …); the diff harness compares
error strings as outputs.
-/

namespace Tropical.Ir.Strata

open Lean (JsonNumber)

/-- Number of passes ported so far. Bumped per Phase 5 stage; the
    diffcli verbs reject `--upto` beyond this. -/
def portedPasses : Nat := 0

structure Options where
  /-- Run passes `1..upto` (0 = elaborate-only prefix). -/
  upto : Nat := portedPasses
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

/-- Run passes `1..opts.upto`. Precondition (enforced by callers):
    `opts.upto ≤ portedPasses`. -/
def run (opts : Options) (arena : Arena) (root : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let _ := opts
  -- Stage 0: no passes ported; the prefix is elaborate-only.
  return (arena, root)

end Tropical.Ir.Strata
