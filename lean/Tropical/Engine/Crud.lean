import Tropical.Engine.Registry

/-!
# Engine.Crud — instance/program lifecycle tool handlers

The per-tool handlers for defining programs and creating, replicating,
removing, listing, and describing instances. `resolveInstanceMeta` resolves a
program name to its registered metadata (shared with the program-I/O path).
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

structure InstanceResolution where
  typeArgs : Option Json
  programMeta : ProgMeta
  resolvedIdx? : Option Tropical.Ir.ProgramIdx

-- ── Per-tool handlers ────────────────────────────────────────────────────────

private def instanceSummary (st : SessionSt) (name : String) : Json :=
  match st.findInstance? name with
  | none => jsonNull
  | some info => Json.mkObj [
      ("name", Json.str name),
      ("type_name", Json.str info.baseTypeName),
      ("type_args", info.typeArgs.getD jsonNull),
      ("inputs", toJson info.progMeta.inputNames),
      ("outputs", toJson info.progMeta.outputNames)]

/-- Resolve a program name (+ optional type args) to instance metadata
    plus the typed-store snapshot for the resolved program, with the TS
    failure shapes. Generic programs specialize through the service's
    `resolve_type`, whose entry is adopted into the store; concrete
    programs take the store's current mapping for the name.

    `toolEnvelopes := false` selects the LOAD/MERGE ingest path's
    failure shapes — `resolveProgramType`'s plain TS Errors
    (`Unknown program type '…'. Known: …` etc.), which the service
    relay surfaced as `internal_error` with the verbatim message. -/
def resolveInstanceMeta (env : Env) (programName : String)
    (typeArgs : Option Json) (programParam : String)
    (toolEnvelopes : Bool := true) :
    EngineM InstanceResolution := do
  let st ← env.state.get
  match st.programs.get? programName with
  | none =>
    -- TS options: [...typeRegistry.keys(), ...programs.keys()] — concrete
    -- names first, then every program name (concrete ones repeat).
    let concrete := st.catalogOrder.filter fun n =>
      (st.programs.get? n).isSome
    if !toolEnvelopes then
      let known := String.intercalate ", " (concrete ++ st.catalogOrder).toList
      internalError s!"Unknown program type '{programName}'. Known: {if known.isEmpty then "(none)" else known}"
    throwEnum .unknownProgram programParam (Json.str programName)
      (concrete ++ st.catalogOrder)
  | some pm =>
    -- No generics: no program declares `type_params`, so `type_args` is always
    -- rejected. Return the concrete metadata + its resolved snapshot.
    match typeArgs with
    | some ta =>
      let keys := match ta with
        | .obj m => String.intercalate ", " (m.toList.map Prod.fst)
        | _ => ""
      if keys.isEmpty then pure {
        typeArgs := none
        programMeta := pm
        resolvedIdx? := st.resolvedByName.get? programName }
      else if !toolEnvelopes then
        internalError s!"Program '{programName}' does not declare type_params; got type_args: {keys}"
      else
        throwBare .invalidTypeArgs
          (s!"Program '{programName}' does not declare type_params; got type_args: {keys}")
          (param := some "type_args") (value := some ta)
    | none => pure {
        typeArgs := none
        programMeta := pm
        resolvedIdx? := st.resolvedByName.get? programName }

-- `define_program` is retired: new DSP types are authored as `Tropical.Stdlib`
-- arrow builders, not defined over the wire. `load`/`merge` still ingest
-- instances + wiring of already-registered programs (`ProgramIO`); the JSON
-- program body is no longer a front door.

def handleAddInstance (env : Env) (args : Json) : EngineM Json := do
  let programName := (argStr? args "program").getD ""
  let instanceName := (argStr? args "instance_name").getD ""
  if instanceName == dacName || instanceName == scopeName then
    throwBare .invalidValue
      s!"'{instanceName}' is a reserved instance name ({dacName} = audio output, {scopeName} = inspection taps). Choose a different name."
      (param := some "instance_name") (value := some (Json.str instanceName))
  let st ← env.state.get
  if (st.findInstance? instanceName).isSome then
    throwBare .instanceExists s!"Instance '{instanceName}' already exists."
      (param := some "instance_name") (value := some (Json.str instanceName))
  let resolved ← resolveInstanceMeta env programName (arg? args "type_args") "program"
  env.state.modify (·.addInstance instanceName
    { baseTypeName := programName
      typeArgs := resolved.typeArgs
      progMeta := resolved.programMeta
      resolvedIdx := resolved.resolvedIdx? })
  pure (instanceSummary (← env.state.get) instanceName)

def handleReplicate (env : Env) (args : Json) : EngineM Json := do
  let programName := (argStr? args "program").getD ""
  let countJ := (arg? args "count").getD jsonNull
  let count? : Option Nat := match countJ with
    | .num n => if n.toFloat == n.toFloat.floor && n.toFloat ≥ 1 then some n.toFloat.toUInt64.toNat else none
    | _ => none
  let some count := count?
    | throwRecord .invalidValue "count" countJ
        [("count", { type := "int", required := true, min := some 1.0 })]
        (some s!"count must be a positive integer, got {tsInterp countJ}")
  let namePrefix := argStr? args "name_prefix"
  let prefix' := namePrefix.getD programName.toLower
  if prefix' == dacName || prefix' == scopeName then
    throwBare .invalidValue
      s!"'{prefix'}' is a reserved instance name ({dacName} = audio output, {scopeName} = inspection taps). Choose a different name_prefix."
      (param := some "name_prefix") (value := some (Json.str prefix'))
  let mut created : Array Json := #[]
  for _ in [0:count] do
    let st ← env.state.get
    let (st', name) := st.nextName prefix'
    env.state.set st'
    if (st'.findInstance? name).isSome then
      throwBare .instanceExists
        s!"Instance '{name}' already exists — pick a different name_prefix"
        (param := some "name_prefix") (value := namePrefix.map Json.str)
    let resolved ← resolveInstanceMeta env programName (arg? args "type_args") "program"
    env.state.modify (·.addInstance name {
      baseTypeName := programName
      typeArgs := resolved.typeArgs
      progMeta := resolved.programMeta
      resolvedIdx := resolved.resolvedIdx? })
    created := created.push (instanceSummary (← env.state.get) name)
  pure <| Json.mkObj [("created", Json.arr created)]

def handleRemoveInstance (env : Env) (args : Json) : EngineM Json := do
  let instanceName := (argStr? args "instance_name").getD ""
  let st ← env.state.get
  let _ ← requireInstance st instanceName "instance_name"
  env.state.modify fun st =>
    let st := st.removeInstance instanceName
    { st with
      wires := st.wires.filter fun w =>
        !(w.instName == instanceName || w.expr.deps.contains instanceName)
      graphOutputs := st.graphOutputs.filter (·.1 != instanceName)
      scopeTaps := st.scopeTaps.filter (·.2.1 != instanceName) }
  syncCompile env
  pure <| Json.mkObj [("removed", Json.str instanceName)]

def handleListPrograms (env : Env) : EngineM Json := do
  let st ← env.state.get
  let portJson (withDefault : Bool) (p : PortInfo) : Json :=
    Json.mkObj <|
      [("name", Json.str p.name),
       ("type", match p.typeStr with | some s => Json.str s | none => jsonNull)]
      ++ (if withDefault then [("default", (p.default.map (·.toJson)).getD jsonNull)] else [])
  let render (m : ProgMeta) : Json :=
    Json.mkObj [
      ("program_name", Json.str m.programName),
      ("inputs", Json.arr (m.inputs.map (portJson true))),
      ("outputs", Json.arr (m.outputs.map (portJson false))),
      ("registers", Json.arr (m.registers.map (portJson false))),
      ("type_params", jsonNull)]
  let metas := st.catalogOrder.filterMap st.programs.get?
  pure <| Json.arr (metas.map render)

def handleListInstances (env : Env) : EngineM Json := do
  let st ← env.state.get
  pure <| Json.arr (st.instances.map fun (n, _) => instanceSummary st n)

def handleGetInfo (env : Env) (args : Json) : EngineM Json := do
  let instanceName := (argStr? args "instance_name").getD ""
  let st ← env.state.get
  let info ← requireInstance st instanceName "instance_name"
  let lookupOutputs := fun n => (st.findInstance? n).map (·.progMeta.outputNames)
  let inputs := info.progMeta.inputs.mapIdx fun i p =>
    let wire := st.findWire? instanceName p.name
    Json.mkObj [
      ("name", Json.str p.name), ("index", toJson i),
      ("type", p.typeObj.getD jsonNull),
      ("expr", match wire with | some w => w.expr.toJson | none => jsonNull),
      ("pretty", match wire with
        | some w => Json.str (w.expr.pretty lookupOutputs)
        | none => jsonNull)]
  let outputs := info.progMeta.outputs.mapIdx fun i p =>
    Json.mkObj [("name", Json.str p.name), ("index", toJson i),
                ("type", p.typeObj.getD jsonNull)]
  let registers := info.progMeta.registers.mapIdx fun i p =>
    Json.mkObj [("name", Json.str p.name), ("index", toJson i),
                ("type", p.typeObj.getD jsonNull)]
  pure <| Json.mkObj [
    ("name", Json.str instanceName),
    ("program", Json.str info.baseTypeName),
    ("type_args", info.typeArgs.getD jsonNull),
    ("inputs", Json.arr inputs),
    ("outputs", Json.arr outputs),
    ("registers", Json.arr registers)]

end Tropical.Engine
