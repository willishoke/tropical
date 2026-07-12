import Tropical.Engine.Crud

/-!
# Engine.Wire — the wiring tools

`handleWire` (the unified set/remove mutation) and the conveniences built on it:
chain, zip, fan-out, fan-in, and list-wiring. Each mutation ends in a single
`syncCompile`.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf? validateExpr exprDependencies prettyExpr)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

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
  let mut scopeWires : Array Json := #[]
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
    else if sInst == scopeName then
      let tapName := (argStr? s "input").getD ""
      if tapName.isEmpty then
        throwBare .invalidValue
          s!"scope tap requires a string input name ({scopeName}.<name>)."
          (param := some "set[].input") (value := some sInput)
      validateOrInternal sExpr s!"{scopeName}.{tapName}"
      let st ← env.state.get
      let (srcInst, srcOut) ← resolveDacSource st sExpr
      env.state.modify fun st =>
        { st with scopeTaps := (st.scopeTaps.filter (·.1 != tapName)).push (tapName, srcInst, srcOut) }
      scopeWires := scopeWires.push <| Json.mkObj
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
                      ("args", Json.arr #[w.expr, adapted])]
        | _, _ => adapted
      env.state.modify (·.setWireRaw sInst resolvedName toStore)
      results := results.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", Json.str resolvedName), ("expr", toStore)]

  syncCompile env
  pure <| Json.mkObj <|
    [("set", Json.arr results)]
    ++ (if dacWires.isEmpty then [] else [("dac", Json.arr dacWires)])
    ++ (if scopeWires.isEmpty then [] else [("scope", Json.arr scopeWires)])
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
    env.state.modify (·.setWireRaw firstName inputName expr)

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
    env.state.modify (·.setWireRaw dstName inName expr)
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
    env.state.modify (·.setWireRaw dstName inName expr)
    linked := linked.push (Json.str s!"{srcName}.{outName} → {dstName}.{inName}")
  syncCompile env
  pure <| Json.mkObj [("linked", Json.arr linked)]

def handleFanOut (env : Env) (args : Json) : EngineM Json := do
  let rawSource := (getField? args "source").getD jsonNull
  let targets := argArr args "targets"
  let st ← env.state.get

  -- The port-ref test matches TS: `instance` a string and `output` present,
  -- regardless of an `op` field — a `ref` ExprNode matches the port-ref arm too.
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

  let mut linked : Array Json := #[]
  for dst in targets do
    let dstName := (argStr? dst "instance").getD ""
    let st' ← env.state.get
    let dstInst ← requireInstance st' dstName "targets[].instance"
    let inName ← resolveInputName dstInst.progMeta ((getField? dst "input").getD jsonNull)
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let expr ← adaptInputExpr st' sourceExpr (inputTypeObj dstInst.progMeta idx) dstName inName
    env.state.modify (·.setWireRaw dstName inName expr)
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
  env.state.modify (·.setWireRaw targetName inName expr)
  syncCompile env
  pure <| Json.mkObj [("mixed", toJson sources.size),
                      ("target", Json.str s!"{targetName}.{inName}")]

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

end Tropical.Engine
