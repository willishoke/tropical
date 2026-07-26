import Tropical.Ir.Strata.Basic
import Tropical.Ir.Strata.EArena
import Tropical.Ir.Strata.InlineInstances
import Tropical.Ir.Strata.IdentityElim
import Tropical.Ir.Cycles

/-!
# Strata — the direct lowering

`Sig` — the one surviving authoring surface — already IS the trunk IR:
its fourteen constructors are exactly the post-strata `ENode` subset
(no combinators, no sum types, no generics; `bankSum` is the one
bounded indexed reduction, kept as data). So there is no drop-pipeline
here any more — the resolved DAG lowers DIRECTLY, through two named
rewrites and a type boundary:

  assertAcyclic    — the entry tripwire (cycle-breaking is upstream's job;
                     any cycle here is a caller bug)
  inlineInstances  — OPTIONAL (`opts.inlineNested`): inner instance bodies
                     lifted in place. The fractal session path skips it and
                     keeps instances as kernel boundaries.
  identityElim     — the categorical identity-law peephole
  toResolved       — the type boundary (`EArena.toResolved`, called by the
                     `runResolved` exit and by `checkResolvedArena` on the
                     session paths): reify the reachable graph into the
                     emit's `ExprArena`, REJECTING every retired
                     constructor. This is the front-door contract — a
                     JSON-loaded `tropical_program_2` can still spell
                     `fold`/`tag`/… syntactically, and dies there with the
                     retirement message.

The five-pass drop sequence (specialize → sumLower → inlineInstances →
arrayLower → identityElim) was retired 2026-07-25: four of the five
passes had no live producer for the structure they existed to retire —
the literate surface parser and generics that produced it are gone, and
`Sig` cannot spell it (a type-level fact: fourteen constructors, none of
them a combinator). Measured before removal: across the full test
corpus and every patch in the repo, the retired passes rewrote nothing
outside their own unit-test fold probes. A future indexed-family
construct that must survive to a backend as data should arrive the way
`bankSum` did — as a `Sig` constructor with its own emit
interpretation — not as a resurrected erasure pass.
-/

namespace Tropical.Ir.Strata

/-- Port of acyclic.ts `assertAcyclic` — the lowering-entry tripwire.
    Cycle-breaking is the realization layer's job upstream; any cycle
    here is a caller bug. -/
private def assertAcyclic (arena : Arena) (root : ProgramIdx) :
    Except Error Unit := do
  let some prog := arena.program? root
    | throw ⟨s!"strataPipeline: program pool index {root.idx} out of range"⟩
  if let some cyc := findInstanceCycle? arena.exprs prog then
    throw ⟨s!"strataPipeline: input contains an unbroken inter-instance cycle: {renderLoop cyc}"⟩

/-- The direct lowering over the shared expression DAG (the inlining
    bloat never materializes), returning the `EArena` and root index.
    The two exits — `run` (tree, for the codec/registration path) and
    `runResolved` (the emit's `ExprArena`) — share this body. -/
def runToEArena (opts : Options) (arena : Arena) (root : ProgramIdx) :
    Except Error (EArena × ProgramIdx) := do
  assertAcyclic arena root
  let ea := EArena.ofArena arena
  let passes : PassM ProgramIdx := do
    let root ← if opts.inlineNested then InlineInstances.runE root else pure root
    IdentityElim.runE root
  let (postRoot, ea) ← passes.run ea
  return (ea, postRoot)

/-- The tree exit: the lowered root as a tree `Program`. Kept for the
    registration/codec path, which round-trips the lowered instance
    type through `tropical_resolved_1`. -/
def run (opts : Options) (arena : Arena) (root : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let (ea, postRoot) ← runToEArena opts arena root
  ea.materialize postRoot

/-- The compile exit: reify the lowered DAG straight into the emit's
    `(ExprArena × CoreProgram)`, no intermediate tree. This is where the
    retired-constructor rejection (`toResolved`) fires on the
    compile-feeding paths. -/
def runResolved (opts : Options) (arena : Arena) (root : ProgramIdx) :
    Except Error (Tropical.Ir.ExprArena × Tropical.Ir.Core.CoreProgram) := do
  let (ea, postRoot) ← runToEArena opts arena root
  ea.toResolved postRoot

end Tropical.Ir.Strata
