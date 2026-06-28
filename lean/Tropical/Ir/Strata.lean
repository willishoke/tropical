import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena
import Tropical.Ir.Strata.Specialize
import Tropical.Ir.Strata.SumLower
import Tropical.Ir.Strata.InlineInstances
import Tropical.Ir.Strata.ArrayLower
import Tropical.Ir.Strata.IdentityElim
import Tropical.Ir.Elaborator

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

Passes are maps over `(Arena, ProgramIdx)` that may push fresh
programs into the pool and abandon old ones — the codec encoder pools
ids on first reference from the root, so unreachable entries are never
emitted.
-/

namespace Tropical.Ir.Strata

/-- Number of passes ported so far. Bumped per Phase 5 stage; the
    diffcli verbs reject `--upto` beyond this. -/
def portedPasses : Nat := 5

/-- Port of acyclic.ts `assertAcyclic` — the strataPipeline-entry
    tripwire. Cycle-breaking is the realization layer's job upstream;
    any cycle here is a caller bug. -/
private def assertAcyclic (arena : Arena) (root : ProgramIdx) :
    Except Error Unit := do
  let some prog := arena.program? root
    | throw ⟨s!"strataPipeline: program pool index {root.idx} out of range"⟩
  let sccs := findInstanceCycles prog
  unless sccs.isEmpty do
    let names := "; ".intercalate (sccs.toList.map fun scc => " → ".intercalate scc.toList)
    throw ⟨s!"strataPipeline: input contains an unbroken inter-instance cycle: {names}"⟩

/-- Run passes `1..opts.upto`. Precondition (enforced by callers):
    `opts.upto ≤ portedPasses`.

    CF-only is now structural: `BodyDecl.reg` no longer exists in the IR, so
    no program can declare per-sample state — the elaborator rejects surface
    `reg`/`next` outright. There is no longer a runtime `assertNoReg` gate. -/
def run (opts : Options) (arena : Arena) (root : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  if opts.upto < 1 then return (arena, root)
  assertAcyclic arena root
  -- Native-DAG (#190): convert to id form, run the passes over the shared
  -- expression DAG (the inlining bloat never materializes), then materialize
  -- the post-strata root back to a tree. (Phase B threads the DAG straight to
  -- emit and drops this final materialization.)
  let ea := EArena.ofArena arena
  let passes : PassM ProgramIdx := do
    let root ← Specialize.runE root opts.typeArgs
    if opts.upto < 2 then return root
    let root ← SumLower.runE root
    if opts.upto < 3 then return root
    let root ← if opts.inlineNested then InlineInstances.runE root else pure root
    if opts.upto < 4 then return root
    let root ← ArrayLower.runE root
    if opts.upto < 5 then return root
    IdentityElim.runE root
  let (postRoot, ea) ← passes.run ea
  ea.materialize postRoot

end Tropical.Ir.Strata
