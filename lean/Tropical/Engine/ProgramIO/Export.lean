import Tropical.Engine.Wire

/-!
# Engine.ProgramIO.Export — save and reusable-program export

`handleSave` serializes the live session to a `tropical_program_2` object.
`handleExportProgram` crystallizes selected instances into a reusable program
type.
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

end Tropical.Engine

