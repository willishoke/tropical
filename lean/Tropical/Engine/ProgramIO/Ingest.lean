import Tropical.Engine.ProgramIO.Export

/-!
# Engine.ProgramIO.Ingest — v2 load and merge

Parse a `tropical_program_2` document through the ordered `JsonV` reader and
fold it into the live session.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

-- ── v2 ingest (load / merge) ─────────────────────────────────────────────────
-- The normalized v2 node is walked directly with key order preserved by
-- `JsonV`. Instances resolve only through registered concrete types, and
-- wires decode into the closed `WireExpr` grammar. The document is a PATCH
-- BAY: `programDecl` is refused at ingest because programs are authored in
-- Lean (arrow builders). Boundary failures retain their stable
-- `internal_error` messages.

open Tropical.Parse (JsonV) in
private def jvBodyEntries (node : JsonV) (k : String) : Array JsonV :=
  match (node.getField? "body").bind (·.getField? k) with
  | some (.arr items) => items
  | _ => #[]

open Tropical.Parse (JsonV) in
private def jvStr? (j : JsonV) (k : String) : Option String :=
  match j.getField? k with
  | some (.str s) => some s
  | _ => none

open Tropical.Parse (JsonV) in
private def jvOp? (j : JsonV) : Option String := jvStr? j "op"

open Tropical.Parse (JsonV) in
private def jvChannel (j : JsonV) (context : String) : EngineM Nat :=
  match j.getField? "channel" with
  | none | some .null => pure 0
  | some (.num n) =>
    let value := n.toFloat
    if value >= 0 && value == value.floor then
      pure value.toUInt64.toNat
    else
      internalError s!"{context}: dac.out channel must be a non-negative integer"
  | some _ =>
    internalError s!"{context}: dac.out channel must be a non-negative integer"

/-- Resolve a body dac-wire expression (port of
    `resolveDacWireExprToGraphOutput`; exact messages). -/
private def resolveDacWire (st : SessionSt) (expr : Tropical.Parse.JsonV)
    (context : String) : EngineM (String × String) := do
  match expr with
  | .obj _ =>
    if jvOp? expr != some "ref" then
      internalError s!"{context}: dac.out wire requires expr.op === 'ref'; got '{(jvOp? expr).getD "undefined"}'."
    let some instName := jvStr? expr "instance"
      | internalError s!"{context}: dac.out wire ref.instance must be a string"
    let some info := st.findInstance? instName
      | internalError s!"{context}: dac.out wire references unknown instance '{instName}'."
    let outNames := info.progMeta.outputNames
    match expr.getField? "output" with
    | some (.num n) =>
      let i := n.toFloat.toUInt64.toNat
      if n.toFloat < 0 || i ≥ outNames.size then
        internalError s!"{context}: dac.out wire output index {n} out of range for '{instName}' ({outNames.size} outputs)."
      pure (instName, outNames[i]!)
    | some (.str s) =>
      if !outNames.contains s then
        internalError s!"{context}: dac.out wire references unknown output '{s}' on '{instName}'. Valid: {String.intercalate ", " outNames.toList}"
      pure (instName, s)
    | _ => internalError s!"{context}: dac.out wire ref.output must be a number or string"
  | _ =>
    internalError s!"{context}: dac.out wire requires a ref-shaped expression (use \{op:'ref',instance,output}); got literal/array."

/-- Walk a normalized v2 node into the engine session (additive; the caller
    cleared state for load). Canonical order is body parameter declarations,
    instances (+ wires), defaults, then body graph outputs. -/
private def ingestProgram (env : Env) (node : Tropical.Parse.JsonV)
    (merge : Bool) : EngineM Unit := do
  let context := if merge then "mergeProgramIntoSession" else "loadProgramAsSession"

  -- Parameter declarations live in the canonical body declaration list.
  let paramSpecs := (jvBodyEntries node "decls").filterMap fun d =>
    if jvOp? d == some "paramDecl" then do
      let name ← jvStr? d "name"
      let value : Json := match d.getField? "value" with
        | some (.num n) => Json.num n
        | _ => Lean.toJson (0 : Nat)
      pure (name, value)
    else none
  -- Merge collision checks (fail fast, TS order: instances then params).
  if merge then
    let st ← env.state.get
    for d in jvBodyEntries node "decls" do
      if jvOp? d == some "instanceDecl" then
        if let some name := jvStr? d "name" then
          if (st.findInstance? name).isSome then
            internalError s!"merge collision: instance '{name}' already exists."
    for (name, _) in paramSpecs do
      if st.params.any (·.1 == name) then
        internalError s!"merge collision: param '{name}' already exists."

  -- Program definitions over the wire are RETIRED — the same
  -- refusal-at-ingest pattern as raise's retiredOp. (The former ingest's
  -- own justification — the FoldProbe fold corpus — self-cancelled: since
  -- the raise refusal landed, a fold-bearing programDecl died at raise and
  -- never reached the elaborator.)
  for d in jvBodyEntries node "decls" do
    if jvOp? d == some "programDecl" then
      let subName := (jvStr? d "name").getD "?"
      internalError <|
        s!"{context}: programDecl '{subName}': program definitions over the wire are retired — " ++
        "programs are authored in Lean (arrow builders); load ingests instances + wiring + params of registered types."

  -- Params before instances (instances may reference them). Idempotent
  -- per name.
  for (name, value) in paramSpecs do
    let st ← env.state.get
    if !st.params.any (·.1 == name) then
      env.state.modify (·.setParamValue name value)

  -- Instances + their wires (raw — no auto-delay wrap on this path).
  for d in jvBodyEntries node "decls" do
    if jvOp? d != some "instanceDecl" then
      continue
    let some instName := jvStr? d "name"
      | internalError s!"{context}: instanceDecl missing name"
    let some programName := jvStr? d "program"
      | internalError s!"{context}: instanceDecl '{instName}' missing program"
    let typeArgs : Option Json := match d.getField? "type_args" with
      | some ta => some ta.toJson
      | none => none
    let resolved ←
      resolveInstanceMeta env programName typeArgs "program" (toolEnvelopes := false)
    env.state.modify (·.addInstance instName (resolved.toInstanceInfo programName))
    -- Wires in declared input-port order (the canonical order; JS's
    -- stable sort leaves unknown ports trailing in JSON-key order,
    -- which JsonV preserves).
    if let some (.obj inputFields) := d.getField? "inputs" then
      let declared := resolved.programMeta.inputNames
      let orderOf := fun (k : String) =>
        match declared.idxOf? k with
        | some i => i
        | none => declared.size
      let sorted := inputFields.zipIdx.qsort fun p q =>
        let oa := orderOf p.1.1
        let ob := orderOf q.1.1
        if oa == ob then Nat.blt p.2 q.2 else Nat.blt oa ob
      for ((input, exprV), _) in sorted do
        -- Decode straight off the ordered JSON — the typed store's
        -- ingest refusal site (no Lean.Json hop).
        let path := s!"{instName}.{input}"
        match WireExpr.ofJsonV exprV path with
        | .error msg => internalError msg
        | .ok expr =>
          -- Same "decodes ≠ compiles" gap the tool boundary closes: a file
          -- may not carry a form only the engine builds, or the session it
          -- loads into is dead on arrival. (The ingest keeps `internal_error`
          -- — the failure is a property of the document, not of a named tool
          -- argument.)
          match expr.uncompilableOp? with
          | some op => internalError (WireExpr.uncompilableMessage path op)
          | none => env.state.modify (·.setWireRaw instName input expr)

  -- Input defaults — every instance in registry order (TS loops the
  -- whole registry, pre-existing instances included on merge).
  let st ← env.state.get
  for (name, info) in st.instances do
    for port in info.progMeta.inputs do
      if let some defaultExpr := port.default then
        let st ← env.state.get
        if (st.findWire? name port.name).isNone then
          env.state.modify (·.setWireRaw name port.name defaultExpr)

  -- Graph outputs are canonical body `dac.out` wires.
  let st ← env.state.get
  let mut outs : Array GraphOutput := #[]
  for a in jvBodyEntries node "assigns" do
    if jvOp? a == some "outputAssign" && jvStr? a "name" == some "dac.out" then
      let some expr := a.getField? "expr"
        | internalError s!"{context}: dac.out wire requires a ref-shaped expression (use \{op:'ref',instance,output}); got literal/array."
      let (sourceInstance, sourceOutput) ← resolveDacWire st expr context
      outs := outs.push {
        channel := ← jvChannel a context
        sourceInstance
        sourceOutput }
  env.state.modify fun st => { st with graphOutputs := st.graphOutputs ++ outs }

def handleLoad (env : Env) (args : Json) : EngineM Json := do
  let path := arg? args "path"
  let program := arg? args "program"
  if path.isNone && program.isNone then
    throwBare .missingArgument "Provide either path (file) or program (inline JSON)."
  -- Stop audio before replacing the session (TS handleLoad semantics).
  if let some dac := ← env.dac.get then
    if ← dac.isRunning then dac.stop
  let rawText ← match path with
    | some (.str p) =>
      match ← (IO.FS.readFile p).toBaseIO with
      | .ok text => pure text
      | .error e => internalError (toString e)
    | _ => pure (program.getD jsonNull).compress
  let t0 ← IO.monoMsNow
  let jv ← match Tropical.Parse.JsonV.parse rawText with
    | .error e => internalError s!"JSON Parse error: {e}"
    | .ok v => pure v
  let node ← match Tropical.Parse.Raise.normalizeProgramFile jv with
    | .error msg => internalError msg
    | .ok r => pure r
  -- Clear the session (the registered programs survive — they hold the stdlib).
  env.state.modify fun st =>
    { st with instances := #[], wires := #[], graphOutputs := #[],
              params := #[], nameCounters := {} }
  ingestProgram env node (merge := false)
  syncCompile env
  let t1 ← IO.monoMsNow
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1))),
    ("timing", Json.mkObj [("wall_ms", toJson (t1 - t0))])]

def handleMerge (env : Env) (args : Json) : EngineM Json := do
  let raw := match arg? args "program" with
    | some p => some p
    | none => arg? args "patch"
  let some program := raw
    | throwBare .missingArgument "Provide a program or patch object."
  let jv ← match Tropical.Parse.JsonV.parse program.compress with
    | .error e => internalError s!"JSON Parse error: {e}"
    | .ok v => pure v
  let node ← match Tropical.Parse.Raise.normalizeProgramFile jv with
    | .error msg => internalError msg
    | .ok r => pure r
  ingestProgram env node (merge := true)
  syncCompile env
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1)))]

end Tropical.Engine
