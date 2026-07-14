import Tropical.Engine.Compile

/-!
# Engine.Registry — program registration (raise → elaborate → strata → entry)

`registerOne` runs a `tropical_program_2` through the full registration
pipeline — rename, elaborate, engine-side strata, entry rendering — and lands
it in the typed store. `strataConcrete` is the concrete-program strata driver
shared with the ingest path.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf? validateExpr exprDependencies prettyExpr)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

-- ── Program registration (raise → elaborate → strata → entry) ─────────────────
-- The engine runs the stage-2 raise, the elaboration, AND (Phase 5
-- stage 6b) the strata pipeline itself; the service's
-- `register_program` shrinks to typeDef registration + decode +
-- `makeCompiled`, one call per batch item.

/-- Run the strata pipeline on an elaborated concrete program — the
    engine-side image of the service residue stage 6b retired:
    relink sub-program registry entries to the canonical post-strata
    registrations (`concreteByName` over TS `session.programs` =
    `templateByName` restricted to concrete entries; load-bearing on
    the boot path, where elaboration resolved against the raw stdlib
    map — a structural no-op on the define path), then run the full
    pipeline. -/
def strataConcrete (st : SessionSt) (arena : Tropical.Ir.Arena)
    (rootIdx : Tropical.Ir.ProgramIdx) :
    EngineM (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
  let byName : String → Option Tropical.Ir.ProgramIdx := fun n =>
    (st.templateByName.get? n).filter fun idx =>
      match arena.program? idx with
      | some prog => prog.typeParams.isEmpty
      | none => false
  let (arena, rootIdx) ←
    match Tropical.Ir.Strata.relinkProgramRegistry arena rootIdx byName with
    | .error e => internalError e.message
    | .ok r => pure r
  match runStrataChecked #[] arena rootIdx with
  | .error msg => internalError msg
  | .ok r => pure r

def renameProgram (p : Tropical.Parse.Program) (name : String) :
    Tropical.Parse.Program :=
  .mk name p.typeParams p.ports p.body p.breaksCycles

def portNames (ps : Option (Array Tropical.Parse.ProgramPort)) : Array String :=
  (ps.getD #[]).map fun
    | .bare n => n
    | .spec s => s.name

/-- The registration batch for a def: nested programDecls depth-first
    in post-order (children before parents, source order), each renamed
    to its decl name; the root last. This is exactly the order
    `loadProgramAsType` registered them (it recursed into
    `{...sub.program, name: sub.name}` BEFORE registering the parent).
    Items keep their nested programDecls inline — the elaborator
    re-elaborates them in scope, as TS does. -/
partial def registrationBatch (name : String) (p : Tropical.Parse.Program) :
    Array (String × Tropical.Parse.Program) := Id.run do
  let mut out : Array (String × Tropical.Parse.Program) := #[]
  for d in p.body.decls do
    if let .prog subName inner := d then
      out := out ++ registrationBatch subName inner
  return out.push (name, renameProgram p name)

/-- Register one program: elaborate it over the typed store, run the
    strata pipeline on it (concrete programs only — generics ship the
    raw template, which the service stores unstrata'd and never
    relinks), ship `{name, parsed, resolved}` to the service (typeDef
    registration + decode + `makeCompiled` + registry insert), and
    adopt the returned entry.

    Resolver: `templateByName` — the mirror of TS `session.programs`
    (post-strata for concrete, raw template for generics) — unless the
    caller supplies one (boot passes the raw stdlib map, mirroring
    `loadStdlibFromResolved`'s `localResolved`).

    Store discipline: for concrete programs the store adopts ONLY the
    service's post-strata round trip (the engine's raw elaboration is
    transient arena growth); for generics the engine's raw template IS
    the stored form, since the service ships `resolved: null` for
    generic entries. Returns the entry and the raw elaborated index. -/
def registerOne (env : Env) (name : String) (p : Tropical.Parse.Program)
    (resolver : Option Tropical.Ir.Resolver := none) :
    EngineM (Json × Tropical.Ir.ProgramIdx) := do
  let st ← env.state.get
  let res : Tropical.Ir.Resolver := match resolver with
    | some r => r
    | none => fun n => st.templateByName.get? n
  -- Elaboration failure → internal_error with the verbatim message
  -- (ElaborationError / CycleViolation are plain Errors in TS; the
  -- service's toEnvelope made them internal_error). Items registered
  -- before a mid-batch failure STAY registered — oracle behavior.
  let (arena', rawIdx) ← match Tropical.Ir.elaborateInto st.arena p (some res) with
    | .error e => internalError e.message
    | .ok r => pure r
  -- Concrete only (generics retired): strata the elaborated program and adopt
  -- its post-strata entry — the same tail as `registerResolved`.
  let (arenaShip, shipIdx) ← strataConcrete st arena' rawIdx
  env.state.modify fun st => { st with arena := arenaShip }
  let entry ← match Tropical.Entries.concreteEntry arenaShip name shipIdx with
    | .error e => internalError e
    | .ok j => pure j
  env.state.modify (·.addProgram (ProgMeta.fromEntry entry))
  let idx? ← adoptResolved env entry
  env.state.modify fun st => { st with
    templateByName := st.templateByName.insert name (idx?.getD rawIdx) }
  pure (entry, rawIdx)

/-- Register an already-resolved (arrow-builder) program — the concrete tail of
    `registerOne` with the elaboration skipped. `arena'`/`rawIdx` are a builder's
    output (`assemble` appended the RAW program to `arena'` at `rawIdx`, linking
    sub-programs by name against the chain built so far, exactly as boot
    elaboration linked against the raw stdlib map). `strataConcrete` relinks that
    raw registry to the post-strata canon (`templateByName`) and runs strata;
    the store adopts the service round-trip, one copy. The boot chain
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
