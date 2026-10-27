import Tropical.Session.Monad
import Tropical.Parse.Raise

/-!
# Session.Ingest — v2 load and merge, over the session state

Parse a `tropical_program_2` document through the ordered `JsonV` reader and
fold it into the session. The normalized v2 node is walked directly with key
order preserved by `JsonV`. Instances resolve only through registered
concrete types, and wires decode into the closed `WireExpr` grammar. The
document is a PATCH BAY: `programDecl` is refused at ingest because programs
are authored in Lean (arrow builders). Boundary failures retain their stable
`internal_error` messages.

`resolveInstanceMeta` is shared with the engine's instance tools, which is
why it carries both failure shapes.
-/

namespace Tropical.Session

open Lean (Json toJson)
open Tropical.Parse (JsonV)

structure InstanceResolution where
  typeArgs : Option Json
  programMeta : ProgMeta
  resolvedIdx? : Option Tropical.Ir.ProgramIdx

def InstanceResolution.toInstanceInfo
    (resolved : InstanceResolution) (baseTypeName : String) : InstanceInfo := {
  baseTypeName
  typeArgs := resolved.typeArgs
  progMeta := resolved.programMeta
  resolvedIdx := resolved.resolvedIdx? }

/-- Resolve a program name (+ optional type args) to instance metadata
    plus the typed-store snapshot for the resolved program, with the TS
    failure shapes. Generic programs specialize through the service's
    `resolve_type`, whose entry is adopted into the store; concrete
    programs take the store's current mapping for the name.

    `toolEnvelopes := false` selects the LOAD/MERGE ingest path's
    failure shapes — `resolveProgramType`'s plain TS Errors
    (`Unknown program type '…'. Known: …` etc.), which the service
    relay surfaced as `internal_error` with the verbatim message. -/
def resolveInstanceMeta (programName : String)
    (typeArgs : Option Json) (programParam : String)
    (toolEnvelopes : Bool := true) :
    SessionM InstanceResolution := do
  let st ← get
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

-- ── v2 ingest (load / merge) ─────────────────────────────────────────────────

private def jvBodyEntries (node : JsonV) (k : String) : Array JsonV :=
  match (node.getField? "body").bind (·.getField? k) with
  | some (.arr items) => items
  | _ => #[]

private def jvStr? (j : JsonV) (k : String) : Option String :=
  match j.getField? k with
  | some (.str s) => some s
  | _ => none

private def jvOp? (j : JsonV) : Option String := jvStr? j "op"

private def jvChannel (j : JsonV) (context : String) : SessionM Nat :=
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
private def resolveDacWire (st : SessionSt) (expr : JsonV)
    (context : String) : SessionM (String × String) := do
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

/-- Walk a normalized v2 node into the session (additive; `loadPatch`
    clears first). Canonical order is body parameter declarations,
    instances (+ wires), defaults, then body graph outputs. -/
def ingestProgram (node : JsonV) (merge : Bool) : SessionM Unit := do
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
    let st ← get
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
    let st ← get
    if !st.params.any (·.1 == name) then
      modify (·.setParamValue name value)

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
      resolveInstanceMeta programName typeArgs "program" (toolEnvelopes := false)
    modify (·.addInstance instName (resolved.toInstanceInfo programName))
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
          | none => modify (·.setWireRaw instName input expr)

  -- Input defaults — every instance in registry order (TS loops the
  -- whole registry, pre-existing instances included on merge).
  let st ← get
  for (name, info) in st.instances do
    for port in info.progMeta.inputs do
      if let some defaultExpr := port.default then
        let st ← get
        if (st.findWire? name port.name).isNone then
          modify (·.setWireRaw name port.name defaultExpr)

  -- Graph outputs are canonical body `dac.out` wires.
  let st ← get
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
  modify fun st => { st with graphOutputs := st.graphOutputs ++ outs }

/-- Parse a document's text and normalize it to the v2 node, with the
    front door's `internal_error` shapes. -/
def normalizePatch (text : String) : SessionM JsonV := do
  let jv ← match JsonV.parse text with
    | .error e => internalError s!"JSON Parse error: {e}"
    | .ok v => pure v
  match Tropical.Parse.Raise.normalizeProgramFile jv with
  | .error msg => internalError msg
  | .ok r => pure r

/-- `load`: replace the session's graph with the document's. The registered
    programs survive — they hold the stdlib. -/
def loadPatch (node : JsonV) : SessionM Unit := do
  modify fun st =>
    { st with instances := #[], wires := #[], graphOutputs := #[],
              params := #[], nameCounters := {} }
  ingestProgram node (merge := false)

/-- `merge`: fold the document's graph into the session's. -/
def mergePatch (node : JsonV) : SessionM Unit :=
  ingestProgram node (merge := true)

end Tropical.Session
