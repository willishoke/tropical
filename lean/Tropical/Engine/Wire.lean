import Tropical.Engine.Crud

/-!
# Engine.Wire — the wiring tools

`handleWire` (the unified set/remove mutation) and the conveniences built on it:
chain, zip, fan-out, fan-in, and list-wiring. Each mutation ends in a single
`syncCompile`.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

-- ── wire (the unified mutation tool) ─────────────────────────────────────────

/-- `raw` is the caller's own JSON: `value` must echo what the AGENT sent, and
    the decoder canonicalizes aliases (`paramExpr` → `param`, `{op:'array',…}` →
    a bare array, `sample_clock` → `clock`), so echoing `expr.toJson` would hand
    back a spelling the caller never wrote. `label` is the port being wired, so
    the message names the real target rather than always saying `dac.out`. -/
private def resolveDacSource (st : SessionSt) (expr : WireExpr) (raw : Json)
    (label : String) : EngineM (String × String) := do
  let (instName, output) ← match expr with
    | .ref inst output => pure (inst, output)
    | .num _ | .bool _ | .arr _ =>
      throwBare .invalidValue
        s!"{label} requires a ref-shaped expression (use refExpr or \{op:'ref',instance,output}). Got literal/array."
        (param := some "set[].expr") (value := some raw)
    | other =>
      throwBare .invalidValue
        s!"{label} requires expr.op === 'ref'. Got op='{other.opName}'."
        (param := some "set[].expr") (value := some raw)
  let info ← requireInstance st instName "instance"
  let outNames := info.progMeta.outputNames
  match output with
  | .index n =>
    let idx := n.toFloat.toUInt64.toNat
    if n.toFloat < 0 || idx ≥ outNames.size then
      throwEnum .unknownOutput "output" (Json.num n) outNames
    pure (instName, outNames[idx]!)
  | .name s =>
    if !outNames.contains s then
      throwEnum .unknownOutput "output" (Json.str s) outNames
    pure (instName, s)

/-- The tool-boundary decode: raw expression Json → `WireExpr`. The
    decoder is the refusal site (state ops, retired ops, arity).

    A failure here is a bad ARGUMENT, not an engine fault, so it rides
    `invalid_value` with the offending expression echoed in `value` —
    `ERRORS.md` reserves `internal_error` for unclassified throws, and
    an agent branching on the code must be able to tell "fix your call"
    from "the engine broke". (The old `validateExpr` routed these to
    `internal_error`; the one input where that accidentally differed —
    a `ref` whose `output` is present but neither string nor number —
    used to reach `resolveDacSource`'s classified arm, which `RefOut`
    has since made unreachable by typing.)

    The INGEST path (`load`/`merge`) keeps `internal_error`: there the
    decode failure is a property of a whole file, not of a named tool
    argument. -/
private def decodeWire (expr : Json) (path : String) (param : String) :
    EngineM WireExpr :=
  match WireExpr.ofJson expr path with
  | .error msg =>
    throwBare .invalidValue msg (param := some param) (value := some expr)
  | .ok e =>
    -- Decoding is necessary but not sufficient: five constructors exist for
    -- the engine's own use and no lowering compiles them. Refuse here, or
    -- they reach the store and detonate at the next compile.
    match e.uncompilableOp? with
    | some op =>
      throwBare .invalidValue (WireExpr.uncompilableMessage path op)
        (param := some param) (value := some expr)
    | none => pure e

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
      let decoded ← decodeWire sExpr s!"{dacName}.{dacOut}" "set[].expr"
      let st ← env.state.get
      let (srcInst, srcOut) ← resolveDacSource st decoded sExpr s!"{dacName}.{dacOut}"
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
      let decoded ← decodeWire sExpr s!"{scopeName}.{tapName}" "set[].expr"
      let st ← env.state.get
      let (srcInst, srcOut) ← resolveDacSource st decoded sExpr s!"{scopeName}.{tapName}"
      env.state.modify fun st =>
        { st with scopeTaps := (st.scopeTaps.filter (·.name != tapName)).push {
            name := tapName
            sourceInstance := srcInst
            sourceOutput := srcOut } }
      scopeWires := scopeWires.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", sInput), ("expr", sExpr)]
    else
      let st ← env.state.get
      let info ← requireInstance st sInst "set[].instance"
      let inputId ← resolveInputIdx info.progMeta sInput
      let resolvedName := (info.progMeta.inputNames[inputId]?).getD (toString inputId)
      let decoded ← decodeWire sExpr s!"{sInst}.{resolvedName}" "set[].expr"
      let adapted ← adaptInputExpr st decoded (inputTypeObj info.progMeta inputId)
        sInst resolvedName (raw? := some sExpr) (param := "set[].expr")
      let existing := st.findWire? sInst resolvedName
      let toStore ← match existing, argStr? s "combine" with
        | some w, some combine =>
          match Tropical.Ir.BinaryOpTag.ofWire? combine with
          | some tag => pure (WireExpr.binary tag w.expr adapted)
          | none =>
            throwBare .invalidValue
              s!"combine must be a binary wire op (add, mul, …), got '{combine}'"
              (param := some "set[].combine") (value := some (Json.str combine))
        | _, _ => pure adapted
      env.state.modify (·.setWireRaw sInst resolvedName toStore)
      results := results.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", Json.str resolvedName), ("expr", toStore.toJson)]

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
    let decoded ← decodeWire initial s!"{firstName}.{inputName}" "initial_expr"
    let expr ← adaptInputExpr st decoded (inputTypeObj firstInst.progMeta idx)
      firstName inputName (raw? := some initial) (param := "initial_expr")
    env.state.modify (·.setWireRaw firstName inputName expr)

  let mut linked : Array Json := #[]
  for i in [0:instanceNames.size - 1] do
    let srcInst := insts[i]!
    let dstInst := insts[i+1]!
    let srcName := instanceNames[i]!
    let dstName := instanceNames[i+1]!
    let outName ← resolveOutputName srcInst.progMeta outputPort
    let inName ← resolveInputName dstInst.progMeta inputPort
    let refExpr : WireExpr := .ref srcName (.name outName)
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let st' ← env.state.get
    let expr ← adaptInputExpr st' refExpr (inputTypeObj dstInst.progMeta idx) dstName inName
      (param := "input")
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
    let refExpr : WireExpr := .ref srcName (.name outName)
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let expr ← adaptInputExpr st refExpr (inputTypeObj dstInst.progMeta idx) dstName inName
      (param := "targets[].input")
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
      pure ((WireExpr.ref sName (.name outName)), s!"{sName}.{outName}")
    else do
      pure (← decodeWire rawSource "source" "source", rawSource.compress)

  let mut linked : Array Json := #[]
  for dst in targets do
    let dstName := (argStr? dst "instance").getD ""
    let st' ← env.state.get
    let dstInst ← requireInstance st' dstName "targets[].instance"
    let inName ← resolveInputName dstInst.progMeta ((getField? dst "input").getD jsonNull)
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let expr ← adaptInputExpr st' sourceExpr (inputTypeObj dstInst.progMeta idx) dstName inName
      (raw? := if isPortRefTS then none else some rawSource)
      (param := if isPortRefTS then "targets[].input" else "source")
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

  let mut terms : Array WireExpr := #[]
  for src in sources do
    let srcName := (argStr? src "instance").getD ""
    let srcInst ← requireInstance st srcName "sources[].instance"
    let outName ← resolveOutputName srcInst.progMeta ((getField? src "output").getD jsonNull)
    let ref : WireExpr := .ref srcName (.name outName)
    terms := terms.push <| match getField? src "gain" with
      | some (.num g) => .binary .mul ref (.num g)
      | _ => ref

  let sumExpr := terms[1:].foldl
    (fun acc t => WireExpr.binary .add acc t)
    terms[0]!

  let inName ← resolveInputName dstInst.progMeta ((getField? target "input").getD jsonNull)
  let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
  let expr ← adaptInputExpr st sumExpr (inputTypeObj dstInst.progMeta idx) targetName inName
    (param := "target.input")
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
                        ("expr", Json.str (w.expr.pretty lookupOutputs))])
    else
      some (Json.mkObj [("instance", Json.str w.instName), ("input", Json.str w.portName),
                        ("expr", Json.str (w.expr.pretty lookupOutputs))])
  pure (Json.arr results)

end Tropical.Engine
