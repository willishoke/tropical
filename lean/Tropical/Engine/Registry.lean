import Tropical.Engine.Compile

/-!
# Engine.Registry — program registration (strata → entry → adopt)

`registerResolved` lands an already-resolved (arrow-builder or
directly-constructed) program in the typed store: engine-side strata,
entry rendering, adoption. `strataConcrete` is the concrete-program
strata driver it shares with nothing else — the former raise → elaborate
path (`registerOne` and its registration batches) died with the
elaborator (2026-07-26): programs are authored in Lean (arrow builders)
or crystallized from a session by `export_program`'s direct
construction; nothing registers from parsed JSON any more.
-/

namespace Tropical.Engine

open Lean (Json toJson)

/-- Run the lowering on a resolved concrete program: relink sub-program
    registry entries to the canonical post-strata registrations
    (`templateByName` restricted to arena-valid entries — load-bearing on
    the boot path, where builders linked against the raw chain), then run
    the full pipeline. -/
def strataConcrete (st : SessionSt) (arena : Tropical.Ir.Arena)
    (rootIdx : Tropical.Ir.ProgramIdx) :
    EngineM (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
  let byName : String → Option Tropical.Ir.ProgramIdx := fun n =>
    (st.templateByName.get? n).filter fun idx =>
      (arena.program? idx).isSome
  let (arena, rootIdx) ←
    match Tropical.Ir.Strata.relinkProgramRegistry arena rootIdx byName with
    | .error e => internalError e.message
    | .ok r => pure r
  match runStrataChecked arena rootIdx with
  | .error msg => internalError msg
  | .ok r => pure r

/-- Register an already-resolved program — `arena'`/`rawIdx` are a builder's
    output (`assemble` appended the RAW program to `arena'` at `rawIdx`,
    linking sub-programs by name against the chain built so far, exactly as
    boot registration links against the raw stdlib map) or `export_program`'s
    direct construction. `strataConcrete` relinks that raw registry to the
    post-strata canon (`templateByName`) and runs strata; the store adopts
    the entry round-trip, one copy. The boot chain
    (`Tropical.EmitArrow.stdlibBuilders`) registers each program this way. -/
def registerResolved (env : Env) (name : String)
    (arena' : Tropical.Ir.Arena) (rawIdx : Tropical.Ir.ProgramIdx) :
    EngineM (Json × Tropical.Ir.ProgramIdx) := do
  let st ← env.state.get
  let (arenaShip, shipIdx) ← strataConcrete st arena' rawIdx
  env.state.modify fun s => { s with arena := arenaShip }
  let entry ← match Tropical.Entries.concreteEntry arenaShip name shipIdx with
    | .error e => internalError e
    | .ok j => pure j
  env.state.modify (·.addProgram (ProgMeta.fromEntry entry))
  let idx? ← adoptResolved env entry
  env.state.modify fun s => { s with
    templateByName := s.templateByName.insert name (idx?.getD rawIdx) }
  pure (entry, rawIdx)

end Tropical.Engine
