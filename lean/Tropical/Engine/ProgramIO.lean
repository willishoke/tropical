import Tropical.Engine.Wire

/-!
# Engine.ProgramIO — save, export, and v2 ingest (load / merge)

`handleSave` serializes the live session to a `tropical_program_2` object;
`handleExportProgram` crystallizes selected instances into a reusable program
type. The ingest half (`handleLoad`/`handleMerge`) parses a `tropical_program_2`
document through the `JsonV` ordered-JSON reader and folds it into the session.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

private structure ExposedInput where
  name : String
  instanceName : String
  portName : String
deriving Inhabited

private def ExposedInput.key (input : ExposedInput) : String :=
  s!"{input.instanceName}:{input.portName}"

-- ── Program I/O (save / export) ──────────────────────────────────────────────

def handleSave (env : Env) : EngineM Json := do
  let st ← env.state.get
  -- CF-only: wires are stored raw (no session-level unit delay wrap), so
  -- save echoes the canonical wires directly.
  let wiresPost := st.wires
  let mut decls : Array Json := #[]
  -- paramDecls first, so they're declared before referencing instances.
  for (name, value) in st.params do
    decls := decls.push <| Json.mkObj
      [("op", Json.str "paramDecl"), ("name", Json.str name), ("value", value)]
  for (name, info) in st.instances do
    let mut inputs : Array (String × Json) := #[]
    for portName in info.progMeta.inputNames do
      if let some w := wiresPost.find? fun w => w.instName == name && w.portName == portName then
        inputs := inputs.push (portName, w.expr.toJson)
    decls := decls.push <| Json.mkObj <|
      [("op", Json.str "instanceDecl"), ("name", Json.str name),
       ("program", Json.str info.baseTypeName)]
      ++ (match info.typeArgs with | some ta => [("type_args", ta)] | none => [])
      ++ (if inputs.isEmpty then [] else [("inputs", Json.mkObj inputs.toList)])
  let mut assigns : Array Json := #[]
  for (inst, output) in st.graphOutputs do
    if let some info := st.findInstance? inst then
      if let some idx := info.progMeta.outputNames.idxOf? output then
        assigns := assigns.push <| Json.mkObj
          [("op", Json.str "outputAssign"), ("name", Json.str "dac.out"),
           ("expr", Json.mkObj [("op", Json.str "ref"), ("instance", Json.str inst),
             ("output", toJson idx)])]
  let body := Json.mkObj <|
    [("op", Json.str "block"), ("decls", Json.arr decls)]
    ++ (if assigns.isEmpty then [] else [("assigns", Json.arr assigns)])
  pure <| Json.mkObj [("program", Json.mkObj
    [("schema", Json.str "tropical_program_2"), ("name", Json.str "patch"),
     ("body", body)])]

/-- Port of `rewriteRefs`: session refs to internal instances become
    `nestedOut`. Typed and exhaustive — refs inside array literals are
    rewritten too (the old Json walker followed `args` only and
    silently skipped `items`). -/
private def rewriteRefs (reachable : Array String) : WireExpr → WireExpr
  | .ref inst output =>
    if reachable.contains inst then .nestedOut inst output else .ref inst output
  | .arr items => .arr (items.attach.map fun ⟨x, _⟩ => rewriteRefs reachable x)
  | .binary tag l r => .binary tag (rewriteRefs reachable l) (rewriteRefs reachable r)
  | .unary tag a => .unary tag (rewriteRefs reachable a)
  | .clamp a b c =>
    .clamp (rewriteRefs reachable a) (rewriteRefs reachable b) (rewriteRefs reachable c)
  | .select a b c =>
    .select (rewriteRefs reachable a) (rewriteRefs reachable b) (rewriteRefs reachable c)
  | .arraySet a b c =>
    .arraySet (rewriteRefs reachable a) (rewriteRefs reachable b) (rewriteRefs reachable c)
  | .index a b => .index (rewriteRefs reachable a) (rewriteRefs reachable b)
  | .broadcastTo a shape => .broadcastTo (rewriteRefs reachable a) shape
  | e => e
termination_by e => sizeOf e
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

/-- Port of `reachableInstances` (stack-pop walk; discovery order is the
    TS Set's insertion order). -/
private def reachableFrom (rootExprs : Array WireExpr) (wires : Array Wire)
    (allInstances : Array String) : Array String := Id.run do
  let mut reachable : Array String := #[]
  let mut queue : Array String := #[]
  for expr in rootExprs do
    for dep in expr.deps do
      if allInstances.contains dep && !reachable.contains dep then
        reachable := reachable.push dep
        queue := queue.push dep
  while !queue.isEmpty do
    let name := queue.back!
    queue := queue.pop
    for w in wires do
      if w.instName == name then
        for dep in w.expr.deps do
          if allInstances.contains dep && !reachable.contains dep then
            reachable := reachable.push dep
            queue := queue.push dep
  return reachable

/-- The TS `portTypeToDecl` over an entry's structured `type_obj`. -/
private def portTypeToDecl (t : Json) : EngineM Json := do
  match getStrField? t "kind" with
  | some "scalar" => pure ((getField? t "scalar").getD jsonNull)
  | some "alias" =>
    pure <| Json.str (((getField? t "alias").bind (getStrField? · "name")).getD "")
  | some "array" =>
    let element : Json := match getField? t "element" with
      | some (.str s) => Json.str s
      | some el => Json.str ((getStrField? el "name").getD "")
      | none => jsonNull
    let shape ← match getField? t "shape" with
      | some (.arr dims) => dims.mapM fun d => do
        match d with
        | .num n => pure (Json.num n)
        | _ => internalError s!"export: array shape carries unresolved type-param '{(getStrField? d "name").getD ""}'"
      | _ => pure #[]
    pure <| Json.mkObj [("kind", Json.str "array"), ("element", element),
      ("shape", Json.arr shape)]
  | _ => pure jsonNull

private def isDefaultPortType (t? : Option Json) : Bool :=
  match t? with
  | none => true
  | some t =>
    getStrField? t "kind" == some "scalar" && getStrField? t "scalar" == some "float"

def handleExportProgram (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  if name.isEmpty then
    throwBare .missingArgument "name is required" (param := some "name")
  let outputsArg := arg? args "outputs"
  let outputPairs : Array (String × Json) := match outputsArg with
    | some (.obj m) => m.toArray
    | _ => #[]
  if outputPairs.isEmpty then
    throwBare .missingArgument "outputs is required (at least one)" (param := some "outputs")

  let st ← env.state.get
  -- CF-only: wires are stored raw; export reads the canonical wires.
  let wiresPost := st.wires
  let allInstances := st.instanceNames

  -- Validate output mappings; build the root ref expressions.
  let mut rootExprs : Array WireExpr := #[]
  for (outName, ref) in outputPairs do
    let instName := (getStrField? ref "instance").getD ""
    let some info := st.findInstance? instName
      | internalError s!"export: output '{outName}' references unknown instance '{instName}'."
    let portName := (getStrField? ref "output").getD ""
    if !info.progMeta.outputNames.contains portName then
      internalError s!"export: instance '{instName}' has no output '{portName}'. Available: {String.intercalate ", " info.progMeta.outputNames.toList}"
    rootExprs := rootExprs.push (.ref instName (.name portName))

  -- Validate input mappings ("instance:port" targets).
  let inputPairs : Array (String × Json) := match arg? args "inputs" with
    | some (.obj m) => m.toArray
    | _ => #[]
  let mut exposed : Array ExposedInput := #[]
  for (inputName, targetJ) in inputPairs do
    let parsed? : Option (String × String) := match targetJ with
      | .str target =>
        match target.splitOn ":" with
        | instPart :: rest =>
          let portPart := String.intercalate ":" rest
          if rest.isEmpty || instPart.isEmpty || portPart.isEmpty then none
          else some (instPart, portPart)
        | [] => none
      | _ => none
    let some (instName, portName) := parsed?
      | internalError s!"export: input '{inputName}' target must be \"instance:port\", got '{tsInterp targetJ}'."
    let some info := st.findInstance? instName
      | internalError s!"export: input '{inputName}' references unknown instance '{instName}'."
    if !info.progMeta.inputNames.contains portName then
      internalError s!"export: instance '{instName}' has no input '{portName}'. Available: {String.intercalate ", " info.progMeta.inputNames.toList}"
    exposed := exposed.push { name := inputName, instanceName := instName, portName }
  let exposedKeys := exposed.map (·.key)
  let findWire := fun (inst port : String) =>
    (wiresPost.find? fun w => w.instName == inst && w.portName == port).map (·.expr)

  -- Reverse reachability from the outputs, extended by exposed-input
  -- wiring defaults.
  let mut reachable := reachableFrom rootExprs wiresPost allInstances
  for input in exposed do
    if let some currentExpr := findWire input.instanceName input.portName then
      for dep in currentExpr.deps do
        if allInstances.contains dep && !reachable.contains dep then
          let extra := reachableFrom #[currentExpr] wiresPost allInstances
          for e in extra do
            if !reachable.contains e then reachable := reachable.push e

  -- Dangling-reference check.
  for instName in reachable do
    for w in wiresPost do
      if w.instName == instName && !exposedKeys.contains w.key then
        for dep in w.expr.deps do
          if !reachable.contains dep && allInstances.contains dep then
            internalError (s!"export: instance '{instName}' wiring '{w.key}' references '{dep}' which is outside the exported subgraph. "
              ++ s!"Either expose it as an input or include '{dep}' in the output dependency chain.")

  -- Port entries (type metadata + folded wiring defaults).
  let portInfoOf := fun (inst port : String) (isInput : Bool) =>
    (st.findInstance? inst).bind fun info =>
      (if isInput then info.progMeta.inputs else info.progMeta.outputs).find? (·.name == port)
  let mut inputEntries : Array Json := #[]
  for input in exposed do
    let typeObj? := (portInfoOf input.instanceName input.portName true).bind (·.typeObj)
    let default? := (findWire input.instanceName input.portName).map (rewriteRefs reachable)
    let typeDecl? ← if isDefaultPortType typeObj? then pure (none : Option Json)
      else some <$> portTypeToDecl (typeObj?.getD jsonNull)
    inputEntries := inputEntries.push <|
      match typeDecl?, default? with
      | none, none => Json.str input.name
      | t?, d? => Json.mkObj <|
        [("name", Json.str input.name)]
        ++ (match t? with | some t => [("type", t)] | none => [])
        ++ (match d? with | some d => [("default", d.toJson)] | none => [])
  let mut outputEntries : Array Json := #[]
  for (outName, ref) in outputPairs do
    let instName := (getStrField? ref "instance").getD ""
    let portName := (getStrField? ref "output").getD ""
    let typeObj? := (portInfoOf instName portName false).bind (·.typeObj)
    if isDefaultPortType typeObj? then
      outputEntries := outputEntries.push (Json.str outName)
    else
      let t ← portTypeToDecl (typeObj?.getD jsonNull)
      outputEntries := outputEntries.push <| Json.mkObj
        [("name", Json.str outName), ("type", t)]

  -- Topological order over the exported subgraph (Kahn, sorted ready
  -- queues; cycle members append in reachable order).
  let topo := Tropical.Lowering.computeInstanceTopoOrder
    (reachable.filterMap fun n => (st.findInstance? n).map (n, ·))
    (wiresPost.filter fun w => reachable.contains w.instName)
  let order := Id.run do
    let mut out := topo
    for n in reachable do
      if !out.contains n then out := out.push n
    return out

  -- Instance decls: exposed ports become {op:'input'}, sibling refs
  -- become nestedOut.
  let mut decls : Array Json := #[]
  for instName in order do
    let some info := st.findInstance? instName
      | internalError s!"export: internal: '{instName}' missing from registry"
    let mut instInputs : Array (String × Json) := #[]
    for portName in info.progMeta.inputNames do
      let key := s!"{instName}:{portName}"
      match exposed.find? (·.key == key) with
      | some input =>
        instInputs := instInputs.push (portName,
          Json.mkObj [("op", Json.str "input"), ("name", Json.str input.name)])
      | none =>
        if let some expr := findWire instName portName then
          instInputs := instInputs.push (portName, (rewriteRefs reachable expr).toJson)
    decls := decls.push <| Json.mkObj <|
      [("op", Json.str "instanceDecl"), ("name", Json.str instName),
       ("program", Json.str info.baseTypeName)]
      ++ (match info.typeArgs with | some ta => [("type_args", ta)] | none => [])
      ++ (if instInputs.isEmpty then [] else [("inputs", Json.mkObj instInputs.toList)])
  let mut assigns : Array Json := #[]
  for (outName, ref) in outputPairs do
    assigns := assigns.push <| Json.mkObj
      [("op", Json.str "outputAssign"), ("name", Json.str outName),
       ("expr", Json.mkObj [("op", Json.str "nestedOut"),
         ("ref", (getField? ref "instance").getD jsonNull),
         ("output", (getField? ref "output").getD jsonNull)])]

  let node := Json.mkObj [
    ("op", Json.str "program"), ("name", Json.str name),
    ("ports", Json.mkObj [("inputs", Json.arr inputEntries),
      ("outputs", Json.arr outputEntries)]),
    ("body", Json.mkObj [("op", Json.str "block"), ("decls", Json.arr decls),
      ("assigns", Json.arr assigns)])]

  -- Register the exported program DIRECTLY: build the resolved `Program`
  -- off the session mirror (the `sessionToResolvedRoot` recipe — export
  -- differs in ports: exposed inputs and exported outputs instead of dac
  -- sinks) and run the `registerResolved` tail (strata + entry + adopt).
  -- The JSON node above is OUTPUT serialization only — there is no
  -- reparse, no raise, no elaborator on this path.
  -- NOTE: this resolves the instance's type by NAME, and `export_program` is
  -- the one route that rebinds a name at runtime. An instance added before a
  -- rebind keeps rendering its own snapshot (`resolvedIdx`) in the session, but
  -- exports through the CURRENT binding of its name — so re-exporting a name
  -- that already has live instances crystallizes a different body than the
  -- session plays. That divergence is pre-existing and NOT fixed here:
  -- resolving through `info.resolvedIdx` does not repair it, because the
  -- emitted decl and the program registry are both keyed by name (verified —
  -- the export still takes the rebound body). Fixing it properly means giving
  -- distinct snapshots distinct registry keys, which is a design change, not a
  -- patch. See the PR discussion.
  let resolveType : String → Except String (Tropical.Ir.ProgramIdx × Tropical.Ir.Program) :=
    fun instName => do
      let some info := st.findInstance? instName
        | .error s!"export: internal: '{instName}' missing from registry"
      let some ti := st.templateByName.get? info.baseTypeName
        | .error s!"export: instance '{instName}': program type '{info.baseTypeName}' is not registered"
      let some tgt := st.arena.program? ti
        | .error s!"export: instance '{instName}': program index for '{info.baseTypeName}' out of range"
      pure (ti, tgt)
  -- Sibling refs resolve to `nestedOut ⟨position in order⟩`; names (params
  -- included — the resolution category order is params, then inputs, and the
  -- exported body declares no params) resolve against the exposed inputs.
  let bodyCtx : WireCtx := {
    instOut := fun rn on => do
      let some idx := order.idxOf? rn
        | .error s!"instance '{rn}' is not declared in this scope"
      let (_, tgt) ← resolveType rn
      match tgt.outputs.findIdx? (·.name == on) with
      | some o => pure (.nestedOut ⟨idx⟩ ⟨o⟩)
      | none =>
        let portList := String.intercalate ", " (tgt.outputs.map (·.name)).toList
        .error s!"instance '{rn}': program '{tgt.name}' has no output '{on}' (have: {portList})"
    paramIdx := fun _ => none
    inputIdx := fun nm => exposed.findIdx? (·.name == nm) }
  -- An input default resolves before any instance is in scope, and sees only
  -- the inputs declared before it (the incremental-scope rule).
  let defaultCtx : Nat → WireCtx := fun visible => {
    instOut := fun rn _ => .error s!"instance '{rn}' is not declared in this scope"
    paramIdx := fun _ => none
    inputIdx := fun nm => (exposed.extract 0 visible).findIdx? (·.name == nm) }
  -- The port surface: exposed inputs (type from the target's resolved decl —
  -- scalar float is the unspelled default — plus the folded wiring default),
  -- then the exported outputs.
  let mut exprs := st.arena.exprs
  let mut inputDecls : Array Tropical.Ir.InputDecl := #[]
  for k in [0:exposed.size] do
    let input := exposed[k]!
    let (_, tgt) ← match resolveType input.instanceName with
      | .error e => internalError e
      | .ok r => pure r
    let some pos := tgt.inputs.findIdx? (·.name == input.portName)
      | internalError s!"export: internal: '{input.instanceName}' has no resolved input '{input.portName}'"
    let type? : Option Tropical.Ir.PortType := match tgt.inputs[pos]!.type? with
      | some (.scalar .float) | none => none
      | some t => some t
    let mut default? : Option Tropical.Ir.ExprId := none
    if let some expr := findWire input.instanceName input.portName then
      match (wireExprToResolved (defaultCtx k) expr).run exprs with
      | .error e => internalError e
      | .ok (eid, exprs') =>
        exprs := exprs'
        default? := some eid
    inputDecls := inputDecls.push { name := input.name, type?, default? }
  let mut outputDecls : Array Tropical.Ir.OutputDecl := #[]
  let mut assignsR : Array Tropical.Ir.OutputAssign := #[]
  for k in [0:outputPairs.size] do
    let (outName, ref) := outputPairs[k]!
    let instName := (getStrField? ref "instance").getD ""
    let portName := (getStrField? ref "output").getD ""
    let (_, tgt) ← match resolveType instName with
      | .error e => internalError e
      | .ok r => pure r
    let some idx := order.idxOf? instName
      | internalError s!"export: internal: output instance '{instName}' not in export order"
    let some o := tgt.outputs.findIdx? (·.name == portName)
      | internalError s!"export: internal: '{instName}' has no resolved output '{portName}'"
    let type? : Option Tropical.Ir.PortType := match tgt.outputs[o]!.type? with
      | some (.scalar .float) | none => none
      | some t => some t
    outputDecls := outputDecls.push { name := outName, type? }
    let (eid, exprs') := (Tropical.Ir.eintern (.nestedOut ⟨idx⟩ ⟨o⟩)).run exprs
    exprs := exprs'
    assignsR := assignsR.push { target := .port ⟨k⟩, expr := eid }
  -- Instance decls in export order: exposed ports become `inputRef`, sibling
  -- refs become `nestedOut`, ports in declared input order.
  let mut declsR : Array Tropical.Ir.BodyDecl := #[]
  for instName in order do
    let (_, tgt) ← match resolveType instName with
      | .error e => internalError e
      | .ok r => pure r
    let some info := st.findInstance? instName
      | internalError s!"export: internal: '{instName}' missing from registry"
    let mut instInputs : Array Tropical.Ir.InstanceInput := #[]
    for portName in info.progMeta.inputNames do
      let key := s!"{instName}:{portName}"
      -- Resolve the port POSITION only for ports that actually land in the
      -- exported decl. `tgt` comes from a late name lookup, so an instance
      -- whose type name was re-registered since it was added (export_program
      -- onto a live name — iterating your own crystallized program) can carry
      -- snapshot ports the current target no longer declares. Resolving those
      -- eagerly aborted the whole export; the elaborator this replaced looped
      -- the SERIALIZED inputs, and an unwired, unexposed port serializes to
      -- nothing. A port that IS exposed or wired still errors here — it has to,
      -- there is no position to bind it to.
      let pos? := tgt.inputs.findIdx? (·.name == portName)
      let unknownPort {α} : EngineM α :=
        internalError s!"export: internal: '{instName}' input '{portName}' is not a declared port of '{tgt.name}'"
      match exposed.find? (·.key == key) with
      | some input =>
        let some pos := pos? | unknownPort
        let some ii := exposed.findIdx? (·.name == input.name)
          | internalError "export: internal: exposed input vanished"
        let (eid, exprs') := (Tropical.Ir.eintern (.inputRef ⟨ii⟩)).run exprs
        exprs := exprs'
        instInputs := instInputs.push { port := ⟨pos⟩, value := eid }
      | none =>
        if let some expr := findWire instName portName then
          let some pos := pos? | unknownPort
          match (wireExprToResolved bodyCtx expr).run exprs with
          | .error e => internalError e
          | .ok (eid, exprs') =>
            exprs := exprs'
            instInputs := instInputs.push { port := ⟨pos⟩, value := eid }
    declsR := declsR.push (.inst instName tgt.name instInputs)
  -- Registry: per-instance target (export order, last write wins per key)
  -- plus the transitive merge — the shape the lowering's relink expects.
  let mut registry : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for instName in order do
    let (ti, tgt) ← match resolveType instName with
      | .error e => internalError e
      | .ok r => pure r
    registry := match registry.findIdx? (·.1 == tgt.name) with
      | some i => registry.set! i (tgt.name, ti)
      | none => registry.push (tgt.name, ti)
    for (kk, vv) in tgt.registry do
      if !registry.any (·.1 == kk) then registry := registry.push (kk, vv)
  let prog : Tropical.Ir.Program := {
    name, inputs := inputDecls, outputs := outputDecls
    decls := declsR, assigns := assignsR, registry }
  -- The acyclic-source contract, enforced where the program is constructed
  -- (the session mirror can hold a cyclic graph after a refused compile).
  if let some cyc := Tropical.Ir.findInstanceCycle? exprs prog then
    internalError (Tropical.Ir.cycleViolationMessage name cyc)
  let arena' := { st.arena with programs := st.arena.programs.push prog, exprs }
  let rawIdx : Tropical.Ir.ProgramIdx := ⟨st.arena.programs.size⟩
  let (rootEntry, _) ← registerResolved env name arena' rawIdx
  let entryNames := fun (k : String) => Json.arr <|
    (match getField? rootEntry k with
     | some (.arr ps) => ps
     | _ => #[]).filterMap fun pj => (getStrField? pj "name").map Json.str

  let exported := order
  let exportedJ := Json.arr (order.map Json.str)

  let removeExported := match arg? args "remove_exported" with
    | some (.bool b) => b
    | _ => false
  if removeExported then
    env.state.modify fun st =>
      { st with
        instances := st.instances.filter fun (n, _) => !exported.contains n
        wires := st.wires.filter fun w =>
          !(exported.contains w.instName
            || w.expr.deps.any exported.contains)
        graphOutputs := st.graphOutputs.filter fun (i, _) => !exported.contains i }
    let st ← env.state.get
    if !st.instances.isEmpty || !st.graphOutputs.isEmpty then
      syncCompile env

  pure <| Json.mkObj [
    ("program_name", Json.str name),
    ("inputs", entryNames "inputs"),
    ("outputs", entryNames "outputs"),
    ("instances_included", exportedJ),
    ("program", Json.mkObj
      [("schema", Json.str "tropical_program_2"), ("name", Json.str name),
       ("ports", (getField? node "ports").getD jsonNull),
       ("body", (getField? node "body").getD jsonNull)])]

-- ── v2 ingest (load / merge) ─────────────────────────────────────────────────
-- Port of `loadProgramAsSession` / `mergeProgramIntoSession` over the
-- engine's mirror: the normalized v2 node (Zod-stripped, key order
-- preserved by JsonV) is walked directly; instances resolve through the
-- engine specialization path with the load-path failure shapes; wires
-- store RAW (loadProgramAsSession sets inputExprNodes directly — no
-- auto-delay wrap). The wire is a PATCH BAY: program definitions
-- (programDecl) are refused at ingest — programs are authored in Lean
-- (arrow builders). All failures are plain TS Errors on the oracle →
-- `internal_error` with the verbatim message.

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

/-- Walk a normalized v2 node + topLevel into the engine session
    (additive; the caller cleared state for load). Mirrors the TS
    ingest order: type_defs → inline programDecls → params → instances
    (+ wires) → defaults → graph outputs. -/
private def ingestProgram (env : Env) (node : Tropical.Parse.JsonV)
    (top : Tropical.Parse.Raise.TopLevel) (merge : Bool) : EngineM Unit := do
  let context := if merge then "mergeProgramIntoSession" else "loadProgramAsSession"

  -- Merged param specs: body paramDecls canonical, topLevel fallback
  -- (dedup by name, body wins).
  let bodyParams := (jvBodyEntries node "decls").filterMap fun d =>
    if jvOp? d == some "paramDecl" then do
      let name ← jvStr? d "name"
      let value : Json := match d.getField? "value" with
        | some (.num n) => Json.num n
        | _ => Lean.toJson (0 : Nat)
      pure (name, value)
    else none
  let mut paramSpecs := bodyParams
  for p in top.params.getD #[] do
    if !paramSpecs.any (·.1 == p.name) then
      paramSpecs := paramSpecs.push (p.name,
        match p.value with | some v => Json.num v | none => Lean.toJson (0 : Nat))

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

  -- Graph outputs: body dac.out wires canonical, file-root
  -- audio_outputs deprecated fallback (appended after).
  let st ← env.state.get
  let mut outs : Array (String × String) := #[]
  for a in jvBodyEntries node "assigns" do
    if jvOp? a == some "outputAssign" && jvStr? a "name" == some "dac.out" then
      let some expr := a.getField? "expr"
        | internalError s!"{context}: dac.out wire requires a ref-shaped expression (use \{op:'ref',instance,output}); got literal/array."
      outs := outs.push (← resolveDacWire st expr context)
  for o in top.audioOutputs.getD #[] do
    match o with
    | .expr _ =>
      internalError s!"{context}: file-root audio_outputs[].expr form not supported. Use \{instance, output} or migrate to body dac.out wires."
    | .ref instName output =>
      if (st.findInstance? instName).isNone then
        internalError s!"{context}: audio_outputs references unknown instance '{instName}'."
      let outName := match output with
        | .str s => s
        | .num n => toString n
        | other => other.toJson.compress
      outs := outs.push (instName, outName)
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
  let (node, top) ← match Tropical.Parse.Raise.normalizeProgramFile jv with
    | .error msg => internalError msg
    | .ok r => pure r
  -- Clear the session (the registered programs survive — they hold the stdlib).
  env.state.modify fun st =>
    { st with instances := #[], wires := #[], graphOutputs := #[],
              params := #[], nameCounters := {} }
  ingestProgram env node top (merge := false)
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
  let (node, top) ← match Tropical.Parse.Raise.normalizeProgramFile jv with
    | .error msg => internalError msg
    | .ok r => pure r
  ingestProgram env node top (merge := true)
  syncCompile env
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1)))]

end Tropical.Engine
