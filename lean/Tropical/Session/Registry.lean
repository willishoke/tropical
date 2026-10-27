import Tropical.Session.Monad
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Entries
import Tropical.Stdlib

/-!
# Session.Registry — program registration (strata → entry → adopt)

`registerResolved` lands an already-resolved (arrow-builder or
directly-constructed) program in the typed store: strata, entry rendering,
adoption. `registerStdlib` is the boot chain, registering the
`Tropical.EmitArrow.stdlibBuilders` in manifest order. The former raise →
elaborate path died with the elaborator (2026-07-26): programs are authored
in Lean (arrow builders) or crystallized from a session by
`export_program`'s direct construction; nothing registers from parsed JSON
any more.
-/

namespace Tropical.Session

open Lean (Json)

/-- The session's `inlineNested` (makeSession defaults it to `true`; nothing
    on the MCP path overrides it). Threaded into every strata run so no two
    paths can disagree about which realization the registered catalog
    holds. -/
private def sessionInlineNested : Bool := true

/-- The direct lowering + the Core downcast (inline path), as a pure
    `Except` so call sites choose the envelope: registration failures
    map to `internal_error`. (Retired constructs are refused at the JSON
    front doors — by the time IR exists here it is trunk-only.) -/
def runStrataChecked (arena : Tropical.Ir.Arena)
    (rootIdx : Tropical.Ir.ProgramIdx) :
    Except String (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
  let (arena, rootIdx) ←
    (Tropical.Ir.Strata.run
      { inlineNested := sessionInlineNested } arena rootIdx).mapError (·.message)
  if sessionInlineNested then
    if let .error e := Tropical.Ir.checkResolvedArena arena rootIdx then
      throw s!"core check failed: {e}"
  pure (arena, rootIdx)

/-- Adopt a catalog entry's `resolved` IR into the typed store (the
    `SessionM` shell over `SessionSt.adoptResolved`). Returns the store
    index, or `none` for entries without resolved IR (generic templates). A
    decode failure is an engine bug — the service encoded it — so it maps
    to `internal_error`. -/
def adoptResolved (entry : Json) : SessionM (Option Tropical.Ir.ProgramIdx) := do
  let st ← get
  match st.adoptResolved entry with
  | .error msg => internalError msg
  | .ok (st', idx?) =>
    set st'
    pure idx?

/-- Run the lowering on a resolved concrete program: relink sub-program
    registry entries to the canonical post-strata registrations
    (`templateByName` restricted to arena-valid entries — load-bearing on
    the boot path, where builders linked against the raw chain), then run
    the full pipeline. -/
def strataConcrete (st : SessionSt) (arena : Tropical.Ir.Arena)
    (rootIdx : Tropical.Ir.ProgramIdx) :
    SessionM (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
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
    the entry round-trip, one copy. -/
def registerResolved (name : String)
    (arena' : Tropical.Ir.Arena) (rawIdx : Tropical.Ir.ProgramIdx) :
    SessionM (Json × Tropical.Ir.ProgramIdx) := do
  let st ← get
  let (arenaShip, shipIdx) ← strataConcrete st arena' rawIdx
  modify fun s => { s with arena := arenaShip }
  let entry ← match Tropical.Entries.concreteEntry arenaShip name shipIdx with
    | .error e => internalError e
    | .ok j => pure j
  modify (·.addProgram (ProgMeta.fromEntry entry))
  let idx? ← adoptResolved entry
  modify fun s => { s with
    templateByName := s.templateByName.insert name (idx?.getD rawIdx) }
  pure (entry, rawIdx)

/-- The boot chain: register the stdlib, the arrow-builder chain
    (`Tropical.EmitArrow.stdlibBuilders`), the 15 live programs authored
    directly as `Sig`/`assemble` builders — no `.md` source, no parsed
    bridge, no elaboration. Each builder appends its raw program to the
    arena, linking sub-programs by name against the chain built so far (deps
    precede dependents in manifest order); `registerResolved` relinks that
    raw registry to the post-strata canon and adopts the catalog entry.
    Manifest order == the chain order, so the catalog (and `list_programs`)
    ordering is preserved. -/
def registerStdlib : SessionM Unit := do
  let mut chain : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for (name, build) in Tropical.EmitArrow.stdlibBuilders do
    let st ← get
    let (arena', rawIdx) ← match build st.arena chain with
      | .error e => internalError s!"stdlib builder '{name}': {e}"
      | .ok r => pure r
    let _ ← registerResolved name arena' rawIdx
    chain := chain.push (name, rawIdx)

/-- A fresh session holding the registered stdlib and nothing else: the
    state every compile starts from. -/
def bootRegistry : Except Failure SessionSt :=
  match registerStdlib.runOn {} with
  | (.ok (), st) => .ok st
  | (.error f, _) => .error f

end Tropical.Session
