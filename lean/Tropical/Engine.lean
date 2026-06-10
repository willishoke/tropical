import Std.Data.HashMap
import Tropical.Errors
import Tropical.Expr
import Tropical.Wiring
import Tropical.Session
import Tropical.Client

/-!
# The tropical IR engine — tool semantics, in Lean

Port of `mcp/engine.ts`. The session (graph topology, program mirror,
wiring) is owned here; the compiler service supplies program
registration, compilation, runtime/audio FFI, and save/export/load/merge
until those layers are ported in turn.

Every graph mutation ends in `syncCompile`: the session snapshot goes to
the service, which rebuilds its TS session, compiles, and hot-swaps the
kernel. State mutations precede the compile (matching TS: a failed
compile leaves the mutated graph in place; the previous kernel keeps
playing and the error envelope is `retryable`).
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf? validateExpr exprDependencies unwrapDelay prettyExpr)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

structure Env where
  state   : IO.Ref SessionSt
  service : Service

-- Reserved audio-output boundary leaf.
private def dacName : String := "dac"
private def dacOut : String := "out"

-- ── Json access helpers ──────────────────────────────────────────────────────

/-- Field access with `??` semantics: absent and `null` are both `none`. -/
private def arg? (args : Json) (k : String) : Option Json :=
  match getField? args k with
  | some .null => none
  | v => v

private def argStr? (args : Json) (k : String) : Option String :=
  match arg? args k with
  | some (.str s) => some s
  | _ => none

private def argArr (args : Json) (k : String) : Array Json :=
  match arg? args k with
  | some (.arr a) => a
  | _ => #[]

private def jsonNull : Json := Json.null

/-- Render a Json value the way a TS template literal renders it
    (`${x}`): bare strings unquoted, everything else as JSON. -/
private def tsInterp : Json → String
  | .str s => s
  | j => j.compress

-- ── Lookup helpers (ported failure shapes) ───────────────────────────────────

private def requireInstance (st : SessionSt) (name : String) (param : String) : EngineM InstanceInfo :=
  match st.findInstance? name with
  | some info => pure info
  | none => throwEnum .unknownInstance param (Json.str name) st.instanceNames

/-- Output name-or-index → index. Numbers (and digit strings) pass
    through unchecked, mirroring TS. -/
private def resolveOutputIdx (pm : ProgMeta) (nameOrIdx : Json) : EngineM Nat :=
  match nameOrIdx with
  | .num n => pure n.toFloat.toUInt64.toNat
  | .str s =>
    if s.toList.all Char.isDigit && !s.isEmpty then
      pure s.toNat!
    else
      match pm.outputNames.idxOf? s with
      | some i => pure i
      | none => throwEnum .unknownOutput "output" (Json.str s) pm.outputNames
  | j => throwEnum .unknownOutput "output" j pm.outputNames

private def resolveInputIdx (pm : ProgMeta) (nameOrIdx : Json) : EngineM Nat :=
  match nameOrIdx with
  | .num n => pure n.toFloat.toUInt64.toNat
  | .str s =>
    if s.toList.all Char.isDigit && !s.isEmpty then
      pure s.toNat!
    else
      match pm.inputNames.idxOf? s with
      | some i => pure i
      | none => throwEnum .unknownInput "input" (Json.str s) pm.inputNames
  | j => throwEnum .unknownInput "input" j pm.inputNames

private def resolveOutputName (pm : ProgMeta) (nameOrIdx : Json) : EngineM String := do
  let idx ← resolveOutputIdx pm nameOrIdx
  pure <| (pm.outputNames[idx]?).getD (toString idx)

private def resolveInputName (pm : ProgMeta) (nameOrIdx : Json) : EngineM String := do
  let idx ← resolveInputIdx pm nameOrIdx
  pure <| (pm.inputNames[idx]?).getD (toString idx)

private def inputTypeObj (pm : ProgMeta) (idx : Nat) : Option Json :=
  (pm.inputs[idx]?).bind (·.typeObj)

-- ── adaptInputExpr (type-check + auto-broadcast) ─────────────────────────────

private def scalarTypeJson (k : String) : Json :=
  Json.mkObj [("kind", Json.str "scalar"), ("scalar", Json.str k)]

/-- Infer a source port type. Returns both the parsed form (for the
    connection check) and the structured PortType Json (echoed verbatim
    as the `got` field of `type_mismatch` envelopes — TS passes the
    PortType object itself). -/
private def srcTypeOf (st : SessionSt) (node : Json) : Option (PortType × Json) :=
  match node with
  | .num _  => some (.scalar .float, scalarTypeJson "float")
  | .bool _ => some (.scalar .bool, scalarTypeJson "bool")
  | .arr items =>
    some (.array { display := "float", kind := some .float } #[items.size],
          Json.mkObj [("kind", Json.str "array"), ("element", Json.str "float"),
                      ("shape", Json.arr #[Lean.toJson items.size])])
  | .obj _ =>
    if opOf? node == some "ref" then do
      let instName ← getStrField? node "instance"
      let info ← st.findInstance? instName
      let outIdx ← match getField? node "output" with
        | some (.num n) => some n.toFloat.toUInt64.toNat
        | some (.str s) => info.progMeta.outputNames.idxOf? s
        | _ => none
      let port ← info.progMeta.outputs[outIdx]?
      let typeObj ← port.typeObj
      let parsed ← parsePortType? typeObj
      pure (parsed, typeObj)
    else none
  | _ => none

private def adaptInputExpr (st : SessionSt) (node : Json) (dstTypeObj : Option Json)
    (instanceName inputName : String) : EngineM Json := do
  match srcTypeOf st node with
  | none => pure node
  | some (srcType, srcTypeJson) =>
    let dstType := dstTypeObj.bind parsePortType?
    let check := checkArrayConnection (some srcType) dstType node
    if !check.compatible then
      throwPredicate .typeMismatch "expr" node "type_compatible"
        dstTypeObj (some srcTypeJson)
        (some s!"Type mismatch on '{instanceName}'.{inputName}: {check.error.getD ""}")
    pure (check.broadcastExpr.getD node)

-- ── Snapshot compile (`wire()` in TS) ────────────────────────────────────────

/-- Serialize the session and have the service rebuild + compile + load. -/
def syncCompile (env : Env) : EngineM Unit := do
  let st ← env.state.get
  let payload := Json.mkObj [
    ("instances", Json.arr <| st.instances.map fun (n, i) =>
      Json.mkObj [("name", Json.str n), ("program", Json.str i.baseTypeName),
                  ("type_args", i.typeArgs.getD jsonNull)]),
    ("wires", Json.arr <| st.wires.map fun w =>
      Json.mkObj [("key", Json.str w.key), ("expr", w.expr)]),
    ("graph_outputs", Json.arr <| st.graphOutputs.map fun (i, o) =>
      Json.mkObj [("instance", Json.str i), ("output", Json.str o)]),
    ("params", Json.arr <| st.params.map fun (n, v) =>
      Json.mkObj [("name", Json.str n), ("value", v)])]
  let resp ← env.service.call "sync" payload
  -- Adopt the service's param registry (compile can discover params).
  match getField? resp "params" with
  | some (.arr ps) =>
    let params := ps.filterMap fun p => do
      let name ← getStrField? p "name"
      let value ← getField? p "value"
      pure (name, value)
    env.state.modify fun st => { st with params }
  | _ => pure ()

-- ── Catalog adoption ─────────────────────────────────────────────────────────

def adoptEntries (env : Env) (entries : Json) : EngineM Unit := do
  match entries with
  | .arr es =>
    for e in es do
      env.state.modify (·.addProgram (ProgMeta.fromEntry e))
  | _ => pure ()

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

/-- Resolve a program name (+ optional type args) to instance metadata,
    with the TS failure shapes. -/
private def resolveInstanceMeta (env : Env) (programName : String)
    (typeArgs : Option Json) (programParam : String) : EngineM (Option Json × ProgMeta) := do
  let st ← env.state.get
  match st.programs.get? programName with
  | none =>
    -- TS options: [...typeRegistry.keys(), ...programs.keys()] — concrete
    -- names first, then every program name (concrete ones repeat).
    let concrete := st.catalogOrder.filter fun n =>
      match st.programs.get? n with | some m => !m.generic | none => false
    throwEnum .unknownProgram programParam (Json.str programName)
      (concrete ++ st.catalogOrder)
  | some pm =>
    if pm.generic then
      let payload := Json.mkObj <|
        [("program", Json.str programName)]
        ++ (match typeArgs with | some ta => [("type_args", ta)] | none => [])
      let attempt : Except Failure Json ← (env.service.call "resolve_type" payload).run
      match attempt with
      | .ok resp =>
        let entry ← Service.field resp "entry"
        let resolved := (getField? resp "type_args").getD jsonNull
        let resolvedOpt := if resolved.compress == "null" then none else some resolved
        pure (resolvedOpt, ProgMeta.fromEntry entry)
      | .error f =>
        -- Specialization failure → invalid_type_args with the raw message.
        let msg := match f with
          | .raw j => (getStrField? j "message").getD j.compress
          | .env e => e.message
        throwBare .invalidTypeArgs msg (param := some "type_args")
          (value := typeArgs)
    else
      match typeArgs with
      | some ta =>
        let keys := match ta with
          | .obj m => String.intercalate ", " (m.toList.map Prod.fst)
          | _ => ""
        if keys.isEmpty then pure (none, pm)
        else
          throwBare .invalidTypeArgs
            (s!"Program '{programName}' does not declare type_params; got type_args: {keys}")
            (param := some "type_args") (value := some ta)
      | none => pure (none, pm)

def handleDefineProgram (env : Env) (args : Json) : EngineM Json := do
  let def_ := (arg? args "def").getD jsonNull
  let resp ← env.service.call "register_program" (Json.mkObj [("def", def_)])
  adoptEntries env ((getField? resp "entries").getD (.arr #[]))
  Service.field resp "result"

def handleAddInstance (env : Env) (args : Json) : EngineM Json := do
  let programName := (argStr? args "program").getD ""
  let instanceName := (argStr? args "instance_name").getD ""
  if instanceName == dacName then
    throwBare .invalidValue
      s!"'{dacName}' is a reserved instance name (audio output boundary). Choose a different name."
      (param := some "instance_name") (value := some (Json.str instanceName))
  let st ← env.state.get
  if (st.findInstance? instanceName).isSome then
    throwBare .instanceExists s!"Instance '{instanceName}' already exists."
      (param := some "instance_name") (value := some (Json.str instanceName))
  let (typeArgs, pm) ← resolveInstanceMeta env programName (arg? args "type_args") "program"
  env.state.modify (·.addInstance instanceName
    { baseTypeName := programName, typeArgs, progMeta := pm })
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
  if prefix' == dacName then
    throwBare .invalidValue
      s!"'{dacName}' is a reserved instance name (audio output boundary). Choose a different name_prefix."
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
    let (typeArgs, pm) ← resolveInstanceMeta env programName (arg? args "type_args") "program"
    env.state.modify (·.addInstance name { baseTypeName := programName, typeArgs, progMeta := pm })
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
        !(w.instName == instanceName || (exprDependencies w.expr).contains instanceName)
      graphOutputs := st.graphOutputs.filter (·.1 != instanceName) }
  syncCompile env
  pure <| Json.mkObj [("removed", Json.str instanceName)]

def handleListPrograms (env : Env) : EngineM Json := do
  let st ← env.state.get
  let portJson (withDefault : Bool) (p : PortInfo) : Json :=
    Json.mkObj <|
      [("name", Json.str p.name),
       ("type", match p.typeStr with | some s => Json.str s | none => jsonNull)]
      ++ (if withDefault then [("default", p.default.getD jsonNull)] else [])
  let render (m : ProgMeta) : Json :=
    Json.mkObj [
      ("program_name", Json.str m.programName),
      ("inputs", Json.arr (m.inputs.map (portJson true))),
      ("outputs", Json.arr (m.outputs.map (portJson false))),
      ("registers", Json.arr (m.registers.map (portJson false))),
      ("type_params", if m.generic then m.typeParams.getD jsonNull else jsonNull)]
  let metas := st.catalogOrder.filterMap st.programs.get?
  let concrete := (metas.filter (!·.generic)).map render
  let generic := (metas.filter (·.generic)).map render
  pure <| Json.arr (concrete ++ generic)

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
      ("expr", match wire with
        | some w => Tropical.Expr.stripDelayIds w.expr
        | none => jsonNull),
      ("pretty", match wire with
        | some w => Json.str (prettyExpr w.expr lookupOutputs)
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

-- ── wire (the unified mutation tool) ─────────────────────────────────────────

private def resolveDacSource (st : SessionSt) (expr : Json) : EngineM (String × String) := do
  let isRefObj := match expr with
    | .obj _ => true
    | _ => false
  if !isRefObj then
    throwBare .invalidValue
      s!"dac.{dacOut} requires a ref-shaped expression (use refExpr or \{op:'ref',instance,output}). Got literal/array."
      (param := some "expr") (value := some expr)
  if opOf? expr != some "ref" then
    throwBare .invalidValue
      s!"dac.{dacOut} requires expr.op === 'ref'. Got op='{(opOf? expr).getD "undefined"}'."
      (param := some "expr") (value := some expr)
  let instJ := getField? expr "instance"
  let some (Json.str instName) := instJ
    | throwBare .invalidValue s!"dac.{dacOut}: ref.instance must be a string"
        (param := some "instance") (value := instJ)
  let info ← requireInstance st instName "instance"
  let outNames := info.progMeta.outputNames
  match getField? expr "output" with
  | some (.num n) =>
    let idx := n.toFloat.toUInt64.toNat
    if n.toFloat < 0 || idx ≥ outNames.size then
      throwEnum .unknownOutput "output" (Json.num n) outNames
    pure (instName, outNames[idx]!)
  | some (.str s) =>
    if !outNames.contains s then
      throwEnum .unknownOutput "output" (Json.str s) outNames
    pure (instName, s)
  | other =>
    throwBare .invalidValue s!"dac.{dacOut}: ref.output must be a number or string"
      (param := some "output") (value := other)

private def validateOrInternal (expr : Json) (path : String) : EngineM Unit :=
  match validateExpr expr path with
  | .ok () => pure ()
  | .error msg => internalError msg

def handleWire (env : Env) (args : Json) : EngineM Json := do
  let setOps := argArr args "set"
  let removeOps := argArr args "remove"

  -- Removes first
  let mut dacRemoved := 0
  for r in removeOps do
    let rInst := (argStr? r "instance").getD ""
    let rInput := (getField? r "input").getD jsonNull
    if rInst == dacName then
      if rInput != Json.str dacOut then
        throwBare .unknownOutput
          s!"dac has only one output port: '{dacOut}'. Got '{tsInterp rInput}'."
          (param := some "remove[].input") (value := some rInput)
      let st ← env.state.get
      dacRemoved := dacRemoved + st.graphOutputs.size
      env.state.modify fun st => { st with graphOutputs := #[] }
    else
      let st ← env.state.get
      let info ← requireInstance st rInst "remove[].instance"
      let inputId ← resolveInputIdx info.progMeta rInput
      let resolvedName := (info.progMeta.inputNames[inputId]?).getD (toString inputId)
      env.state.modify (·.removeWire rInst resolvedName)

  -- Sets
  let mut results : Array Json := #[]
  let mut dacWires : Array Json := #[]
  for s in setOps do
    let sInst := (argStr? s "instance").getD ""
    let sInput := (getField? s "input").getD jsonNull
    let sExpr := (getField? s "expr").getD jsonNull
    if sInst == dacName then
      if sInput != Json.str dacOut then
        throwBare .unknownOutput
          s!"dac has only one output port: '{dacOut}'. Got '{tsInterp sInput}'."
          (param := some "set[].input") (value := some sInput)
      validateOrInternal sExpr s!"{dacName}.{dacOut}"
      let st ← env.state.get
      let (srcInst, srcOut) ← resolveDacSource st sExpr
      env.state.modify fun st =>
        { st with graphOutputs := st.graphOutputs.push (srcInst, srcOut) }
      dacWires := dacWires.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", sInput), ("expr", sExpr)]
    else
      let st ← env.state.get
      let info ← requireInstance st sInst "set[].instance"
      let inputId ← resolveInputIdx info.progMeta sInput
      let resolvedName := (info.progMeta.inputNames[inputId]?).getD (toString inputId)
      validateOrInternal sExpr s!"{sInst}.{resolvedName}"
      let adapted ← adaptInputExpr st sExpr (inputTypeObj info.progMeta inputId) sInst resolvedName
      let existing := st.findWire? sInst resolvedName
      let toStore := match existing, argStr? s "combine" with
        | some w, some combine =>
          Json.mkObj [("op", Json.str combine),
                      ("args", Json.arr #[unwrapDelay w.expr, adapted])]
        | _, _ => adapted
      env.state.modify (·.setWire sInst resolvedName toStore)
      results := results.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", Json.str resolvedName), ("expr", toStore)]

  syncCompile env
  pure <| Json.mkObj <|
    [("set", Json.arr results)]
    ++ (if dacWires.isEmpty then [] else [("dac", Json.arr dacWires)])
    ++ [("removed", toJson removeOps.size)]
    ++ (if dacRemoved > 0 then [("dacRemoved", toJson dacRemoved)] else [])

-- ── Wiring conveniences ──────────────────────────────────────────────────────

def handleWireChain (env : Env) (args : Json) : EngineM Json := do
  let instanceNames := (argArr args "instances").filterMap fun j =>
    match j with | .str s => some s | _ => none
  let outputPort := (getField? args "output").getD jsonNull
  let inputPort := (getField? args "input").getD jsonNull
  let initialExpr := arg? args "initial_expr"

  if instanceNames.size < 2 && initialExpr.isNone then
    throwBare .arityError
      "wire_chain needs at least 2 instances, or 1 instance with initial_expr"
      (param := some "instances") (value := some (toJson instanceNames))

  let st ← env.state.get
  let mut insts : Array InstanceInfo := #[]
  for n in instanceNames do
    insts := insts.push (← requireInstance st n "instances")

  if let some initial := initialExpr then
    let firstName := instanceNames[0]!
    let firstInst := insts[0]!
    let inputName ← resolveInputName firstInst.progMeta inputPort
    let idx := (firstInst.progMeta.inputNames.idxOf? inputName).getD 0
    let expr ← adaptInputExpr st initial (inputTypeObj firstInst.progMeta idx) firstName inputName
    env.state.modify (·.setWire firstName inputName expr)

  let mut linked : Array Json := #[]
  for i in [0:instanceNames.size - 1] do
    let srcInst := insts[i]!
    let dstInst := insts[i+1]!
    let srcName := instanceNames[i]!
    let dstName := instanceNames[i+1]!
    let outName ← resolveOutputName srcInst.progMeta outputPort
    let inName ← resolveInputName dstInst.progMeta inputPort
    let refExpr := Json.mkObj [("op", Json.str "ref"),
      ("instance", Json.str srcName), ("output", Json.str outName)]
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let st' ← env.state.get
    let expr ← adaptInputExpr st' refExpr (inputTypeObj dstInst.progMeta idx) dstName inName
    env.state.modify (·.setWire dstName inName expr)
    linked := linked.push (Json.str s!"{srcName}.{outName} → {dstName}.{inName}")

  syncCompile env
  pure <| Json.mkObj [("linked", Json.arr linked)]

def handleWireZip (env : Env) (args : Json) : EngineM Json := do
  let sources := argArr args "sources"
  let targets := argArr args "targets"
  if sources.size != targets.size then
    throwBare .lengthMismatch
      s!"sources and targets must be the same length (got {sources.size} vs {targets.size})"
      (param := some "sources")
      (value := some (Json.mkObj [("sources", toJson sources.size), ("targets", toJson targets.size)]))
  let mut linked : Array Json := #[]
  for i in [0:sources.size] do
    let src := sources[i]!
    let dst := targets[i]!
    let srcName := (argStr? src "instance").getD ""
    let dstName := (argStr? dst "instance").getD ""
    let st ← env.state.get
    let srcInst ← requireInstance st srcName "sources[].instance"
    let dstInst ← requireInstance st dstName "targets[].instance"
    let outName ← resolveOutputName srcInst.progMeta ((getField? src "output").getD jsonNull)
    let inName ← resolveInputName dstInst.progMeta ((getField? dst "input").getD jsonNull)
    let refExpr := Json.mkObj [("op", Json.str "ref"),
      ("instance", Json.str srcName), ("output", Json.str outName)]
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let expr ← adaptInputExpr st refExpr (inputTypeObj dstInst.progMeta idx) dstName inName
    env.state.modify (·.setWire dstName inName expr)
    linked := linked.push (Json.str s!"{srcName}.{outName} → {dstName}.{inName}")
  syncCompile env
  pure <| Json.mkObj [("linked", Json.arr linked)]

def handleFanOut (env : Env) (args : Json) : EngineM Json := do
  let rawSource := (getField? args "source").getD jsonNull
  let targets := argArr args "targets"
  let st ← env.state.get

  let isPortRef := match rawSource with
    | .obj _ => (getStrField? rawSource "instance").isSome
                && (getField? rawSource "output").isSome
                && (opOf? rawSource).isNone
    | _ => false
  -- TS checks `typeof rawSource.instance === 'string' && rawSource.output !== undefined`
  -- regardless of an `op` field; a `ref` ExprNode matches the port-ref arm too.
  let isPortRefTS := match rawSource with
    | .obj _ => (getStrField? rawSource "instance").isSome && (getField? rawSource "output").isSome
    | _ => false

  let (sourceExpr, sourceLabel) ←
    if isPortRefTS then do
      let sName := (getStrField? rawSource "instance").getD ""
      let srcInst ← requireInstance st sName "source.instance"
      let outName ← resolveOutputName srcInst.progMeta ((getField? rawSource "output").getD jsonNull)
      pure (Json.mkObj [("op", Json.str "ref"),
              ("instance", Json.str sName), ("output", Json.str outName)],
            s!"{sName}.{outName}")
    else
      pure (rawSource, rawSource.compress)
  let _ := isPortRef

  let mut linked : Array Json := #[]
  for dst in targets do
    let dstName := (argStr? dst "instance").getD ""
    let st' ← env.state.get
    let dstInst ← requireInstance st' dstName "targets[].instance"
    let inName ← resolveInputName dstInst.progMeta ((getField? dst "input").getD jsonNull)
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let expr ← adaptInputExpr st' sourceExpr (inputTypeObj dstInst.progMeta idx) dstName inName
    env.state.modify (·.setWire dstName inName expr)
    linked := linked.push (Json.str s!"{sourceLabel} → {dstName}.{inName}")
  syncCompile env
  pure <| Json.mkObj [("linked", Json.arr linked)]

def handleFanIn (env : Env) (args : Json) : EngineM Json := do
  let sources := argArr args "sources"
  let target := (getField? args "target").getD jsonNull
  if sources.isEmpty then
    throwBare .arityError "sources must be non-empty"
      (param := some "sources") (value := some (Json.arr sources))
  let st ← env.state.get
  let targetName := (argStr? target "instance").getD ""
  let dstInst ← requireInstance st targetName "target.instance"

  let mut terms : Array Json := #[]
  for src in sources do
    let srcName := (argStr? src "instance").getD ""
    let srcInst ← requireInstance st srcName "sources[].instance"
    let outName ← resolveOutputName srcInst.progMeta ((getField? src "output").getD jsonNull)
    let ref := Json.mkObj [("op", Json.str "ref"),
      ("instance", Json.str srcName), ("output", Json.str outName)]
    terms := terms.push <| match getField? src "gain" with
      | some g@(.num _) => Json.mkObj [("op", Json.str "mul"), ("args", Json.arr #[ref, g])]
      | _ => ref

  let sumExpr := terms[1:].foldl
    (fun acc t => Json.mkObj [("op", Json.str "add"), ("args", Json.arr #[acc, t])])
    terms[0]!

  let inName ← resolveInputName dstInst.progMeta ((getField? target "input").getD jsonNull)
  let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
  let expr ← adaptInputExpr st sumExpr (inputTypeObj dstInst.progMeta idx) targetName inName
  env.state.modify (·.setWire targetName inName expr)
  syncCompile env
  pure <| Json.mkObj [("mixed", toJson sources.size),
                      ("target", Json.str s!"{targetName}.{inName}")]

def handleFeedback (env : Env) (args : Json) : EngineM Json := do
  let from_ := (getField? args "from").getD jsonNull
  let to := (getField? args "to").getD jsonNull
  let init := (arg? args "init").getD (toJson (0 : Nat))
  let delayId := argStr? args "delay_id"

  let st ← env.state.get
  let fromName := (argStr? from_ "instance").getD ""
  let toName := (argStr? to "instance").getD ""
  let srcInst ← requireInstance st fromName "from.instance"
  let dstInst ← requireInstance st toName "to.instance"

  let outName ← resolveOutputName srcInst.progMeta ((getField? from_ "output").getD jsonNull)
  let inName ← resolveInputName dstInst.progMeta ((getField? to "input").getD jsonNull)

  let refExpr := Json.mkObj [("op", Json.str "ref"),
    ("instance", Json.str fromName), ("output", Json.str outName)]
  validateOrInternal refExpr s!"{toName}.{inName}"
  let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
  let expr ← adaptInputExpr st refExpr (inputTypeObj dstInst.progMeta idx) toName inName
  env.state.modify (·.setWire toName inName expr (init := init) (id := delayId))

  syncCompile env
  pure <| Json.mkObj [("feedback",
    Json.str s!"{fromName}.{outName} →[delay init={tsInterp init}]→ {toName}.{inName}")]

def handleListWiring (env : Env) (args : Json) : EngineM Json := do
  let filter := argStr? args "instance"
  let st ← env.state.get
  let lookupOutputs := fun n => (st.findInstance? n).map (·.progMeta.outputNames)
  let results := st.wires.filterMap fun w =>
    if let some f := filter then
      if w.instName != f then none else
      some (Json.mkObj [("instance", Json.str w.instName), ("input", Json.str w.portName),
                        ("expr", Json.str (prettyExpr w.expr lookupOutputs))])
    else
      some (Json.mkObj [("instance", Json.str w.instName), ("input", Json.str w.portName),
                        ("expr", Json.str (prettyExpr w.expr lookupOutputs))])
  pure (Json.arr results)

-- ── Program I/O (relayed; the compiler owns these until Phases 3–4) ─────────

def handleExportProgram (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  if name.isEmpty then
    throwBare .missingArgument "name is required" (param := some "name")
  let outputs := arg? args "outputs"
  let outputsEmpty := match outputs with
    | some (.obj m) => m.toList.isEmpty
    | _ => true
  if outputsEmpty then
    throwBare .missingArgument "outputs is required (at least one)" (param := some "outputs")

  let payload := Json.mkObj [
    ("name", Json.str name),
    ("inputs", (arg? args "inputs").getD (Json.mkObj [])),
    ("outputs", outputs.getD jsonNull)]
  let resp ← env.service.call "export_program" payload
  adoptEntries env ((getField? resp "entries").getD (.arr #[]))

  let exportedJ := (getField? resp "exported_instances").getD (.arr #[])
  let exported : Array String := match exportedJ with
    | .arr es => es.filterMap fun j => match j with | .str s => some s | _ => none
    | _ => #[]

  let removeExported := match arg? args "remove_exported" with
    | some (.bool b) => b
    | _ => false
  if removeExported then
    env.state.modify fun st =>
      { st with
        instances := st.instances.filter fun (n, _) => !exported.contains n
        wires := st.wires.filter fun w =>
          !(exported.contains w.instName
            || (exprDependencies w.expr).any exported.contains)
        graphOutputs := st.graphOutputs.filter fun (i, _) => !exported.contains i }
    let st ← env.state.get
    if !st.instances.isEmpty || !st.graphOutputs.isEmpty then
      syncCompile env

  pure <| Json.mkObj [
    ("program_name", Json.str name),
    ("inputs", (getField? resp "inputs").getD (.arr #[])),
    ("outputs", (getField? resp "outputs").getD (.arr #[])),
    ("instances_included", exportedJ),
    ("program", (getField? resp "program").getD jsonNull)]

/-- Adopt a service state dump (after load / merge). -/
private def adoptState (env : Env) (state : Json) (resetCounters : Bool) : EngineM Unit := do
  adoptEntries env ((getField? state "entries").getD (.arr #[]))
  let instances : Array (String × InstanceInfo) :=
    (match getField? state "instances" with
     | some (.arr is) => is
     | _ => #[]).filterMap fun i => do
      let name ← getStrField? i "name"
      let program ← getStrField? i "program"
      let typeArgs := match getField? i "type_args" with
        | some .null | none => none
        | some ta => some ta
      let pm := ProgMeta.fromEntry ((getField? i "entry").getD jsonNull)
      pure (name, { baseTypeName := program, typeArgs, progMeta := pm : InstanceInfo })
  let wires : Array Wire :=
    (match getField? state "wires" with
     | some (.arr ws) => ws
     | _ => #[]).filterMap fun w => do
      let key ← getStrField? w "key"
      let expr ← getField? w "expr"
      match key.splitOn ":" with
      | instName :: rest => pure { instName, portName := String.intercalate ":" rest, expr : Wire }
      | [] => none
  let graphOutputs : Array (String × String) :=
    (match getField? state "graph_outputs" with
     | some (.arr os) => os
     | _ => #[]).filterMap fun o => do
      pure ((← getStrField? o "instance"), (← getStrField? o "output"))
  let params : Array (String × Json) :=
    (match getField? state "params" with
     | some (.arr ps) => ps
     | _ => #[]).filterMap fun p => do
      pure ((← getStrField? p "name"), (← getField? p "value"))
  env.state.modify fun st =>
    { st with instances, wires, graphOutputs, params,
              nameCounters := if resetCounters then {} else st.nameCounters }

def handleLoad (env : Env) (args : Json) : EngineM Json := do
  let path := arg? args "path"
  let program := arg? args "program"
  if path.isNone && program.isNone then
    throwBare .missingArgument "Provide either path (file) or program (inline JSON)."
  let payload := Json.mkObj <|
    (match path with | some p => [("path", p)] | none => [])
    ++ (match program with | some p => [("program", p)] | none => [])
  let resp ← env.service.call "load" payload
  adoptState env ((getField? resp "state").getD jsonNull) (resetCounters := true)
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1))),
    ("timing", (getField? resp "timing").getD jsonNull)]

def handleSave (env : Env) : EngineM Json := do
  let resp ← env.service.call "save" (Json.mkObj [])
  pure <| Json.mkObj [("program", (getField? resp "program").getD jsonNull)]

def handleMerge (env : Env) (args : Json) : EngineM Json := do
  let raw := match arg? args "program" with
    | some p => some p
    | none => arg? args "patch"
  let some program := raw
    | throwBare .missingArgument "Provide a program or patch object."
  let resp ← env.service.call "merge" (Json.mkObj [("program", program)])
  adoptState env ((getField? resp "state").getD jsonNull) (resetCounters := false)
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1)))]

-- ── Audio / params (relayed; FFI moves to Lean in Phase 2) ───────────────────

private def relayTool (env : Env) (method : String) (args : Json) : EngineM Json :=
  env.service.call method args

def handleSetParam (env : Env) (args : Json) : EngineM Json := do
  let resp ← relayTool env "set_param" args
  -- Keep the param mirror current for future snapshots.
  if let (some (.str name), some value) := (getField? resp "name", getField? resp "value") then
    env.state.modify (·.setParamValue name value)
  pure resp

-- ── Dispatcher ───────────────────────────────────────────────────────────────

def handleTool (env : Env) (name : String) (args : Json) : IO Json :=
  wrap <| match name with
  | "define_program"  => handleDefineProgram env args
  | "add_instance"    => handleAddInstance env args
  | "remove_instance" => handleRemoveInstance env args
  | "replicate"       => handleReplicate env args
  | "wire_chain"      => handleWireChain env args
  | "wire_zip"        => handleWireZip env args
  | "fan_out"         => handleFanOut env args
  | "fan_in"          => handleFanIn env args
  | "feedback"        => handleFeedback env args
  | "export_program"  => handleExportProgram env args
  | "list_programs"   => handleListPrograms env
  | "list_instances"  => handleListInstances env
  | "get_info"        => handleGetInfo env args
  | "wire"            => handleWire env args
  | "list_wiring"     => handleListWiring env args
  | "load"            => handleLoad env args
  | "save"            => handleSave env
  | "merge"           => handleMerge env args
  | "start_audio"     => relayTool env "start_audio" args
  | "stop_audio"      => relayTool env "stop_audio" args
  | "audio_status"    => relayTool env "audio_status" args
  | "set_param"       => handleSetParam env args
  | "list_params"     => relayTool env "list_params" args
  | _ => internalError s!"Unknown tool: '{name}'"

-- ── Boot ─────────────────────────────────────────────────────────────────────

/-- Spawn the compiler service, fetch the stdlib catalog, build the Env. -/
def boot : IO (Env × IO.Process.Child ⟨.piped, .piped, .inherit⟩) := do
  let child ← IO.Process.spawn {
    cmd := "bun", args := #["run", "mcp/compiler_service.ts"],
    stdin := .piped, stdout := .piped, stderr := .inherit
  }
  let service : Service := { relay := { stdin := child.stdin, stdout := child.stdout } }
  let state ← IO.mkRef ({} : SessionSt)
  let env : Env := { state, service }
  let catalog ← service.relay.call "boot" (Json.mkObj [])
  match catalog.getObjVal? "catalog" with
  | .ok (.arr entries) =>
    for e in entries do
      state.modify (·.addProgram (ProgMeta.fromEntry e))
  | _ => throw <| IO.userError s!"compiler service boot failed: {catalog.compress.take 200}"
  pure (env, child)

end Tropical.Engine
