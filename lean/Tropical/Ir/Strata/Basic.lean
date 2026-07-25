import Tropical.Ir.Nodes

/-!
# Strata shared types — Options + Error

Split from `Tropical/Ir/Strata.lean` (the orchestrator) so individual
pass modules under `Tropical/Ir/Strata/` can reference them without an
import cycle.
-/

namespace Tropical.Ir.Strata

structure Options where
  /-- `false` = the fractal session path: inlineInstances is skipped. -/
  inlineNested : Bool := true
deriving Repr, Inhabited

/-- A strata error is a comparable output: the TS error message,
    byte-exact. -/
structure Error where
  message : String
deriving Repr, Inhabited

open Tropical.Ir in
/-- Port of decl_tables.ts `relinkProgramRegistry` (Phase 5 stage 6b):
    swap registry entries to the canonical post-strata programs by
    registry key. Identity-preserving on no-op — the TS `changed` flag's
    object-identity test becomes pool-idx equality here, so the define
    path (whose elaboration already linked canonical indices through
    `templateByName`) returns the input index untouched, while the boot
    path (raw-linked, mirroring TS `localResolved`) gets a fresh
    program whose registry points at the post-strata canon. -/
def relinkProgramRegistry (arena : Arena) (rootIdx : ProgramIdx)
    (byName : String → Option ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"relinkProgramRegistry: program pool index {rootIdx.idx} out of range"⟩
  let mut changed := false
  let mut newReg : Array (String × ProgramIdx) := #[]
  for (key, value) in prog.registry do
    match byName key with
    | some canonical =>
      if canonical != value then
        newReg := newReg.push (key, canonical)
        changed := true
      else
        newReg := newReg.push (key, value)
    | none => newReg := newReg.push (key, value)
  if !changed then return (arena, rootIdx)
  return ({ arena with programs := arena.programs.push { prog with registry := newReg } },
          ⟨arena.programs.size⟩)

end Tropical.Ir.Strata
