import Tropical.Stdlib

/-!
# StdlibChain — the stdlib arena the arena consumers link against

Retargeted off the parse bridge: `elabStdlib` now returns the arrow-builder
chain (`Tropical.EmitArrow.buildStdlibChain`) rather than elaborating
`stdlib/parsed/*.json`. The name survives so the playground, the arrow gates,
and the diffcli verbs need no call-site change — the bridge readers
(`readParsed`/`manifestNames`/`elabChain`) are gone with the bridge.
-/

namespace Tropical.StdlibChain

/-- The whole stdlib chain the arena consumers (playground boot, the arrow
    gates) link stdlib programs against — the arrow-builder fold. Pure, but kept
    in `IO` for its callers' signatures. -/
def elabStdlib :
    IO (Except String (Tropical.Ir.Arena × Array (String × Tropical.Ir.ProgramIdx))) :=
  pure Tropical.EmitArrow.buildStdlibChain

end Tropical.StdlibChain
