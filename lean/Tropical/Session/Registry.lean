import Tropical.Session.Monad
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Entries
import Tropical.Stdlib

/-!
# Session.Registry — program registration (strata → adopt → entry)

`registerResolved` lands an already-resolved (arrow-builder or
directly-constructed) program in the typed store: strata, adoption, entry
rendering. `registerStdlib` is the boot chain, registering the
`Tropical.EmitArrow.stdlibBuilders` in manifest order, then records the
stdlib's type families; `registerFamilyMember` registers a family member on
first use. The former raise →
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

/-- Adopt a post-strata program into the typed store (the `SessionM` shell
    over `SessionSt.adoptResolved`): the arena strata returned becomes the
    store's, and the program is recorded under its own name. -/
def adoptResolved (arena : Tropical.Ir.Arena) (idx : Tropical.Ir.ProgramIdx) :
    SessionM Unit := do
  match (← get).adoptResolved arena idx with
  | .error msg => internalError msg
  | .ok st' => set st'

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
    the post-strata program where strata left it, and the catalog entry
    carries its port metadata. -/
def registerResolved (name : String)
    (arena' : Tropical.Ir.Arena) (rawIdx : Tropical.Ir.ProgramIdx) :
    SessionM (Json × Tropical.Ir.ProgramIdx) := do
  let st ← get
  let (arenaShip, shipIdx) ← strataConcrete st arena' rawIdx
  adoptResolved arenaShip shipIdx
  let entry ← match Tropical.Entries.concreteEntry arenaShip name shipIdx with
    | .error e => internalError e
    | .ok j => pure j
  modify fun s => { s.addProgram (ProgMeta.fromEntry entry) with
    templateByName := s.templateByName.insert name shipIdx }
  pure (entry, rawIdx)

/-- The boot chain: register the stdlib, the arrow-builder chain
    (`Tropical.EmitArrow.stdlibBuilders`), the 20 live programs authored
    directly as `Sig`/`assemble` builders — no `.md` source, no parsed
    bridge, no elaboration. Each builder appends its raw program to the
    arena, linking sub-programs by name against the chain built so far (deps
    precede dependents in manifest order); `registerResolved` relinks that
    raw registry to the post-strata canon and adopts the lowered program.
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
  let families := Tropical.EmitArrow.stdlibFamilies
  let conflicts := familyConflicts families (← get).catalogOrder
  if !conflicts.isEmpty then
    internalError s!"stdlib type families: {String.intercalate "; " conflicts.toList}"
  modify fun s => { s with families }

/-- Assemble a type family's member at `size` and register it under its own
    name (`TypeFamily.memberName`), exactly as a boot program is registered.
    The member may instantiate any registered type, linked by name. After this
    the member is an ordinary concrete type: nothing downstream knows it came
    from a family. -/
def registerFamilyMember (f : TypeFamily) (size : Nat) : SessionM Unit := do
  let st ← get
  let name := f.memberName size
  let chain := st.catalogOrder.filterMap fun n =>
    (st.templateByName.get? n).map (n, ·)
  let (arena', rawIdx) ← match f.build name size st.arena chain with
    | .error e => internalError s!"type family '{f.name}' at size {size}: {e}"
    | .ok r => pure r
  -- The family's declared port count is what the budget was checked
  -- against at boot; a member that disagrees is refused before it registers.
  let ports := match arena'.program? rawIdx with
    | some p => some (p.inputs.size + p.outputs.size)
    | none => none
  if ports != some (f.portsAt size) then
    internalError s!"type family '{f.name}' at size {size}: the member has {ports.getD 0} ports, the family declares {f.portsAt size}"
  let _ ← registerResolved name arena' rawIdx

/-- A fresh session holding the registered stdlib and nothing else: the
    state every compile starts from. -/
def bootRegistry : Except Failure SessionSt :=
  match registerStdlib.runOn {} with
  | (.ok (), st) => .ok st
  | (.error f, _) => .error f

end Tropical.Session
