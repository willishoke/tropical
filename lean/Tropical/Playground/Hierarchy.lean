import Lean.Data.Json

/-!
# Version-3 patch hierarchy

The persisted authoring document may contain reusable, nested definitions, but
the production patcher continues to consume one flat downstream-only graph.
This module is the single hygienic elaboration boundary between those worlds.

V3 uses ordinary graph nodes plus three authoring-only kinds:

* `module` references a definition by stable id and version;
* `module_input` is the one primary input boundary of a definition; and
* `module_output` is its one primary output boundary.

Definitions carry typed port/parameter metadata for clients.  The elaborator
needs only their ids, versions, ordered nodes, boundary ids, parameter defaults,
and explicit parameter bindings.  Every leaf after expansion is an existing
registered patch kind or one compiler-private hierarchy atom.  No hierarchy
case reaches EmitArrow or a backend.
-/

namespace Tropical.Playground

open Lean (Json JsonNumber)

private def obj? : Json → Option (Std.TreeMap.Raw String Json compare)
  | .obj fields => some fields
  | _ => none

private def arr? : Json → Option (Array Json)
  | .arr values => some values
  | _ => none

private def str? : Json → Option String
  | .str value => some value
  | _ => none

private def nat? : Json → Option Nat
  | .num value =>
      if value.exponent == 0 && value.mantissa ≥ 0 then some value.mantissa.toNat else none
  | _ => none

private def field? (json : Json) (name : String) : Option Json :=
  (obj? json).bind (·[name]?)

private def strField? (json : Json) (name : String) : Option String :=
  (field? json name).bind str?

private def arrField (json : Json) (name : String) : Array Json :=
  (field? json name).bind arr? |>.getD #[]

private def objField (json : Json) (name : String) : Std.TreeMap.Raw String Json compare :=
  (field? json name).bind obj? |>.getD {}

private def jsonSet (json : Json) (name : String) (value : Json) : Json :=
  match json with
  | .obj fields => .obj (fields.insert name value)
  | _ => Json.mkObj [(name, value)]

private def jsonErase (json : Json) (name : String) : Json :=
  match json with
  | .obj fields => .obj (fields.erase name)
  | other => other

private def inputSources (node : Json) (port : String) : Array String :=
  match (objField node "in")[port]? with
  | some (.arr values) => values.filterMap str?
  | _ => #[]

private def setInputs (node : Json) (ports : Array (String × Array String)) : Json :=
  let fields := ports.foldl (fun out (name, sources) =>
    out.insert name (.arr (sources.map Json.str))) {}
  jsonSet node "in" (.obj fields)

private structure HierarchyParameter where
  name : String
  defaultValue : Json

private structure HierarchyDefinition where
  id : String
  version : Nat
  inputNode : String
  outputNode : String
  inputDomain : String
  outputDomain : String
  parameters : Array HierarchyParameter
  nodes : Array Json

private def decodeParameter (owner : String) (json : Json) : Except String HierarchyParameter := do
  let some name := strField? json "name"
    | throw s!"hierarchy: definition '{owner}' has a parameter without a string name"
  if name.isEmpty then
    throw s!"hierarchy: definition '{owner}' has an empty parameter name"
  let some defaultValue := field? json "default"
    | throw s!"hierarchy: definition '{owner}' parameter '{name}' has no default"
  pure { name, defaultValue }

private def decodeDefinition (json : Json) : Except String HierarchyDefinition := do
  let some id := strField? json "id"
    | throw "hierarchy: definition is missing string field 'id'"
  if id.isEmpty then throw "hierarchy: definition id cannot be empty"
  let some version := (field? json "version").bind nat?
    | throw s!"hierarchy: definition '{id}' is missing a nonnegative integer version"
  let some inputNode := strField? json "input"
    | throw s!"hierarchy: definition '{id}' is missing string field 'input'"
  let some outputNode := strField? json "output"
    | throw s!"hierarchy: definition '{id}' is missing string field 'output'"
  let nodes := arrField json "nodes"
  if nodes.isEmpty then throw s!"hierarchy: definition '{id}' has no nodes"
  let inputDomain := strField? json "input_domain" |>.getD "modal"
  let outputDomain := strField? json "output_domain" |>.getD "modal"
  unless inputDomain == "modal" && outputDomain == "modal" do
    throw s!"hierarchy: definition '{id}' v{version} must be Modal → Modal in schema v3"
  let parameters ← (arrField json "parameters").mapM (decodeParameter id)
  pure {
    id, version, inputNode, outputNode, inputDomain, outputDomain
    parameters
    nodes }

private structure BoundParameter where
  localName : String
  globalName : String
  value : Json

private structure ExpandedGraph where
  nodes : Array Json := #[]
  outputs : Array (String × Array String) := #[]
  sourceMap : Array Json := #[]

private def parameterJson (name : String) (value : Json) : Json :=
  Json.mkObj [("name", .str name), ("default", value)]

private def boundaryNode (id kind : String) (inputs : Array String := #[]) : Json :=
  Json.mkObj <| [("id", .str id), ("kind", .str kind)] ++
    if inputs.isEmpty then [] else
      [("in", Json.mkObj [("in", .arr (inputs.map Json.str))])]

private def allpassDefinitionJson : Json :=
  Json.mkObj [
    ("id", .str "tropical.modal.allpass1"), ("version", Lean.toJson 1),
    ("input", .str "input"), ("output", .str "output"),
    ("input_domain", .str "modal"), ("output_domain", .str "modal"),
    ("parameters", .arr #[
      parameterJson "center" (.num ⟨700, 0⟩),
      parameterJson "sweep" (.num ⟨15, 1⟩),
      parameterJson "rate" (.num ⟨2, 1⟩),
      parameterJson "ratio" (.num ⟨1, 0⟩)]),
    ("nodes", .arr #[
      boundaryNode "input" "module_input",
      Json.mkObj [
        ("id", .str "tail"), ("kind", .str "modal_allpass_tail"),
        ("params", Json.mkObj []),
        ("bindings", Json.mkObj [
          ("center", .str "center"), ("sweep", .str "sweep"),
          ("rate", .str "rate"), ("ratio", .str "ratio")]),
        ("in", Json.mkObj [("in", .arr #[.str "input"])])],
      Json.mkObj [
        ("id", .str "section"), ("kind", .str "modalmix"),
        ("in", Json.mkObj [("in", .arr #[.str "input", .str "tail"])])],
      boundaryNode "output" "module_output" #["section"]])]

private def phaserRatioJson : Array Json := #[
  .num ⟨42044820762685725, 17⟩,
  .num ⟨5946035575013605, 16⟩,
  .num ⟨8408964152537145, 16⟩,
  .num ⟨1189207115002721, 15⟩,
  .num ⟨1681792830507429, 15⟩,
  .num ⟨2378414230005442, 15⟩]

private def phaserStageNode (index : Nat) (ratio : Json) : Json :=
  let input := if index == 0 then "input" else s!"stage_{index - 1}"
  Json.mkObj [
    ("id", .str s!"stage_{index}"), ("kind", .str "module"),
    ("definition", .str "tropical.modal.allpass1"),
    ("definition_version", Lean.toJson 1),
    ("params", Json.mkObj [("ratio", ratio)]),
    ("bindings", Json.mkObj [
      ("center", .str "center"), ("sweep", .str "sweep"),
      ("rate", .str "rate")]),
    ("in", Json.mkObj [("in", .arr #[.str input])])]

private def phaserDefinitionJson : Json :=
  Json.mkObj [
    ("id", .str "tropical.modal.phaser"), ("version", Lean.toJson 1),
    ("input", .str "input"), ("output", .str "output"),
    ("input_domain", .str "modal"), ("output_domain", .str "modal"),
    ("parameters", .arr #[
      parameterJson "center" (.num ⟨700, 0⟩),
      parameterJson "sweep" (.num ⟨15, 1⟩),
      parameterJson "rate" (.num ⟨2, 1⟩),
      parameterJson "mix" (.num ⟨5, 1⟩)]),
    ("nodes", .arr <|
      #[boundaryNode "input" "module_input"]
      ++ phaserRatioJson.mapIdx phaserStageNode
      ++ #[Json.mkObj [
          ("id", .str "blend"), ("kind", .str "modalblend"),
          ("bindings", Json.mkObj [("mix", .str "mix")]),
          ("in", Json.mkObj [
            ("dry", .arr #[.str "input"]),
            ("wet", .arr #[.str "stage_5"])])],
        boundaryNode "output" "module_output" #["blend"]])]

/-- The immutable standard definitions a client can instantiate or clone. -/
def hierarchyLibraryJson : Json := Json.mkObj [
  ("schema", .str "tropical_module_library"),
  ("schema_version", Lean.toJson 1),
  ("definitions", .arr #[allpassDefinitionJson, phaserDefinitionJson])]

private def shippedDefinitionJsons : Array Json :=
  #[allpassDefinitionJson, phaserDefinitionJson]

private def orderedNodes (owner : String) (nodes : Array Json) : Except String (Array Json) := do
  let ids ← nodes.mapM fun node => match strField? node "id" with
    | some id => pure id
    | none => throw s!"hierarchy: {owner} contains a node without id"
  unless ids.all (!·.isEmpty) do
    throw s!"hierarchy: {owner} contains an empty node id"
  unless ids.zipIdx.all (fun (id, index) => !(ids.extract 0 index).contains id) do
    throw s!"hierarchy: {owner} contains duplicate node ids"
  for node in nodes do
    let id := strField? node "id" |>.getD "?"
    for (_, value) in (objField node "in").toList do
      let .arr sources := value
        | throw s!"hierarchy: {owner} node '{id}' has a non-array inlet"
      for source in sources do
        let .str sourceId := source
          | throw s!"hierarchy: {owner} node '{id}' has a non-string source"
        unless ids.contains sourceId do
          throw s!"hierarchy: {owner} node '{id}' names missing source '{sourceId}'"
  let mut pending := nodes
  let mut emittedIds : Array String := #[]
  let mut ordered : Array Json := #[]
  for _ in [0:nodes.size] do
    let mut next : Array Json := #[]
    for node in pending do
      let dependencies := (objField node "in").toList.flatMap fun (_, value) =>
        match value with
        | .arr sources => (sources.filterMap str?).toList
        | _ => []
      if dependencies.all emittedIds.contains then
        ordered := ordered.push node
        emittedIds := emittedIds.push (strField? node "id" |>.getD "?")
      else
        next := next.push node
    pending := next
  unless pending.isEmpty do
    throw s!"hierarchy: cycle in {owner}"
  pure ordered

private def validateDefinitionReferences
    (definitions : Array HierarchyDefinition) : Except String Unit := do
  let keys := definitions.map fun definition => (definition.id, definition.version)
  let dependencies ← definitions.mapM fun definition => do
    let mut out := #[]
    for node in definition.nodes do
      if strField? node "kind" == some "module" then
        let localId := strField? node "id" |>.getD "?"
        let some id := strField? node "definition"
          | throw s!"hierarchy: module '{definition.id}.{localId}' is missing definition id"
        let some version := (field? node "definition_version").bind nat?
          | throw s!"hierarchy: module '{definition.id}.{localId}' is missing definition version"
        out := out.push (id, version)
    pure out
  for (definition, deps) in definitions.zip dependencies do
    for dep in deps do
      unless keys.contains dep do
        throw s!"hierarchy: definition '{definition.id}' references unavailable definition '{dep.1}' v{dep.2}"
  let mut resolved : Array (String × Nat) := #[]
  for _ in [0:definitions.size] do
    for (key, deps) in keys.zip dependencies do
      if !resolved.contains key && deps.all resolved.contains then
        resolved := resolved.push key
  unless resolved.size == definitions.size do
    throw "hierarchy: definition-reference cycle"

private def lookupOutput (outputs : Array (String × Array String))
    (id : String) : Option (Array String) :=
  (outputs.find? (·.1 == id)).map (·.2)

private def lookupBound (context : Array BoundParameter)
    (name : String) : Option BoundParameter :=
  context.find? (·.localName == name)

/-- Length-delimited path segments make generated ids injective without
    imposing a forbidden-character rule on authored ids. -/
private def hygienicId (path : Array String) (localId : String) : String :=
  (path.push localId).foldl (fun out segment =>
    out ++ s!"{segment.length}_{segment}") "__h3_"

private def definitionOf (definitions : Array HierarchyDefinition)
    (id : String) (version : Nat) : Option HierarchyDefinition :=
  definitions.find? fun definition =>
    definition.id == id && definition.version == version

private def outputSourceOf (definition : HierarchyDefinition) : Except String String := do
  let some boundary := definition.nodes.find? (strField? · "id" == some definition.outputNode)
    | throw s!"hierarchy: definition '{definition.id}' output boundary '{definition.outputNode}' is missing"
  unless strField? boundary "kind" == some "module_output" do
    throw s!"hierarchy: definition '{definition.id}' output id '{definition.outputNode}' is not a module_output"
  let sources := inputSources boundary "in"
  let [source] := sources.toList
    | throw s!"hierarchy: definition '{definition.id}' output boundary must have exactly one source"
  pure source

private def validateDefinitionShape (definition : HierarchyDefinition) : Except String Unit := do
  let parameterNames := definition.parameters.map (·.name)
  unless parameterNames.zipIdx.all (fun (name, index) =>
      !(parameterNames.extract 0 index).contains name) do
    throw s!"hierarchy: definition '{definition.id}' repeats a parameter name"
  let some input := definition.nodes.find? (strField? · "id" == some definition.inputNode)
    | throw s!"hierarchy: definition '{definition.id}' input boundary '{definition.inputNode}' is missing"
  unless strField? input "kind" == some "module_input" do
    throw s!"hierarchy: definition '{definition.id}' input id '{definition.inputNode}' is not a module_input"
  unless (objField input "in").isEmpty do
    throw s!"hierarchy: definition '{definition.id}' input boundary cannot have incoming wires"
  for node in definition.nodes do
    let id := strField? node "id" |>.getD "?"
    match strField? node "kind" with
    | some "module_input" => unless id == definition.inputNode do
        throw s!"hierarchy: definition '{definition.id}' has undeclared module_input '{id}'"
    | some "module_output" => unless id == definition.outputNode do
        throw s!"hierarchy: definition '{definition.id}' has undeclared module_output '{id}'"
    | some _ => pure ()
    | none => throw s!"hierarchy: definition '{definition.id}' node '{id}' has no kind"
  let _ ← outputSourceOf definition
  pure ()

private def boundContext (definition : HierarchyDefinition) (inst : Json)
    (parent : Array BoundParameter) (publicId : String) : Except String (Array BoundParameter) := do
  let bindings := objField inst "bindings"
  let params := objField inst "params"
  let mut out := #[]
  for parameter in definition.parameters do
    match bindings[parameter.name]? with
    | some (.str parentName) =>
      let some inherited := lookupBound parent parentName
        | throw s!"hierarchy: binding '{parameter.name}={parentName}' at '{publicId}' names no outer parameter"
      out := out.push { inherited with localName := parameter.name }
    | some _ =>
      throw s!"hierarchy: binding '{publicId}.{parameter.name}' must name an outer parameter"
    | none =>
      out := out.push {
        localName := parameter.name
        globalName := s!"{publicId}.{parameter.name}"
        value := (params[parameter.name]?).getD parameter.defaultValue }
  pure out

private def bindAtomicParams (node : Json) (context : Array BoundParameter) : Except String Json := do
  let bindings := objField node "bindings"
  let mut params := objField node "params"
  let mut aliases : Std.TreeMap.Raw String Json compare := {}
  for (localName, binding) in bindings.toList do
    let .str outerName := binding
      | throw s!"hierarchy: atomic binding '{localName}' must be a parameter name"
    let some bound := lookupBound context outerName
      | throw s!"hierarchy: atomic binding '{localName}={outerName}' names no module parameter"
    params := params.insert localName bound.value
    aliases := aliases.insert localName (.str bound.globalName)
  pure <| jsonErase
    (jsonSet (jsonSet node "params" (.obj params)) "param_aliases" (.obj aliases))
    "bindings"

private def translatedInputs (node : Json)
    (outputs : Array (String × Array String)) : Except String (Array (String × Array String)) := do
  let mut translated := #[]
  for (port, value) in (objField node "in").toList do
    let .arr sourcesJson := value
      | throw s!"hierarchy: inlet '{port}' must be an ordered source array"
    let mut sources := #[]
    for sourceJson in sourcesJson do
      let .str source := sourceJson
        | throw s!"hierarchy: inlet '{port}' contains a non-string source"
      let some resolved := lookupOutput outputs source
        | throw s!"hierarchy: source '{source}' is not earlier in definition order"
      sources := sources ++ resolved
    translated := translated.push (port, sources)
  pure translated

/-- Elaborate a v3 document to the unchanged flat patch-graph envelope. Flat
    legacy graphs pass through structurally; legacy Phaser nodes first migrate
    to the installed hierarchy definition at this same boundary. -/
def elaboratePatchHierarchy (document : Json) : Except String Json := do
  let originalVersion? := (field? document "version").bind nat?
  let legacyNodes := arrField document "nodes"
  let document := if originalVersion? != some 3 &&
      legacyNodes.any (strField? · "kind" == some "phaser") then
    let migratedNodes := legacyNodes.map fun node =>
      if strField? node "kind" == some "phaser" then
        jsonSet
          (jsonSet
            (jsonSet node "kind" (.str "module"))
            "definition" (.str "tropical.modal.phaser"))
          "definition_version" (Lean.toJson 1)
      else node
    let migrated := Json.mkObj [
      ("version", Lean.toJson 3),
      ("definitions", .arr #[]),
      ("scene", Json.mkObj [
        ("nodes", .arr migratedNodes),
        ("out", (field? document "out").getD (.str ""))]),
      ("taps", (field? document "taps").getD (.arr #[]))]
    migrated
  else document
  let version? := (field? document "version").bind nat?
  if version? != some 3 then return document
  let authored ← (arrField document "definitions").mapM decodeDefinition
  let authoredKeys := authored.map fun definition => (definition.id, definition.version)
  unless authoredKeys.zipIdx.all (fun (key, index) =>
      !(authoredKeys.extract 0 index).contains key) do
    throw "hierarchy: duplicate definition id/version"
  let shipped ← shippedDefinitionJsons.mapM decodeDefinition
  let shippedKeys := shipped.map fun definition => (definition.id, definition.version)
  for key in authoredKeys do
    if shippedKeys.contains key then
      throw s!"hierarchy: document cannot replace installed definition '{key.1}' v{key.2}; detach it under a new stable id"
  let definitions ← (shipped ++ authored).mapM fun definition => do
    let nodes ← orderedNodes s!"definition '{definition.id}' v{definition.version}"
      definition.nodes
    let definition := { definition with nodes }
    validateDefinitionShape definition
    pure definition
  validateDefinitionReferences definitions
  let definitionKeys := definitions.map fun definition => (definition.id, definition.version)
  unless definitionKeys.zipIdx.all (fun (key, index) =>
      !(definitionKeys.extract 0 index).contains key) do
    throw "hierarchy: duplicate definition id/version"
  let some scene := field? document "scene"
    | throw "hierarchy: v3 document is missing object field 'scene'"
  let some unorderedSceneNodes := (field? scene "nodes").bind arr?
    | throw "hierarchy: v3 scene is missing array field 'nodes'"
  let sceneNodes ← orderedNodes "scene" unorderedSceneNodes

  let rec expandModule : Nat → Json → Array String → Array BoundParameter →
      Array String → String → Except String ExpandedGraph
    | 0, _, _, _, _, publicId =>
      throw s!"hierarchy: definition-reference cycle at '{publicId}'"
    | fuel + 1, inst, outerInputs, parentContext, path, publicId => do
      let some definitionId := strField? inst "definition"
        | throw s!"hierarchy: module '{publicId}' is missing definition id"
      let some definitionVersion := (field? inst "definition_version").bind nat?
        | throw s!"hierarchy: module '{publicId}' is missing definition version"
      let some definition := definitionOf definitions definitionId definitionVersion
        | throw s!"hierarchy: module '{publicId}' references unavailable definition '{definitionId}' v{definitionVersion}"
      let context ← boundContext definition inst parentContext publicId
      let outputSource ← outputSourceOf definition
      let mut expanded : ExpandedGraph := if outerInputs.size > 1 then
        let fanInId := hygienicId path "__module_input_fanin__"
        let fanIn := Json.mkObj [
          ("id", .str fanInId), ("kind", .str "modalmix"),
          ("params", Json.mkObj []),
          ("in", Json.mkObj [("in", .arr (outerInputs.map Json.str))])]
        {
          nodes := #[fanIn]
          outputs := #[(definition.inputNode, #[fanInId])]
          sourceMap := #[Json.mkObj [
            ("expanded", .str fanInId),
            ("definition", .str definition.id),
            ("definition_version", Lean.toJson definition.version),
            ("local", .str definition.inputNode),
            ("path", .arr (path.map Json.str))]] }
      else {
        outputs := #[(definition.inputNode, outerInputs)] }
      for node in definition.nodes do
        let some localId := strField? node "id"
          | throw s!"hierarchy: definition '{definition.id}' contains a node without id"
        let some kind := strField? node "kind"
          | throw s!"hierarchy: definition '{definition.id}' node '{localId}' has no kind"
        if kind == "module_input" then
          unless localId == definition.inputNode do
            throw s!"hierarchy: definition '{definition.id}' has an undeclared module_input '{localId}'"
        else if kind == "module_output" then
          unless localId == definition.outputNode do
            throw s!"hierarchy: definition '{definition.id}' has an undeclared module_output '{localId}'"
        else
          let inputs ← translatedInputs node expanded.outputs
          let nodePublicId := if localId == outputSource then publicId
            else hygienicId path localId
          if kind == "module" then
            let child ← expandModule fuel (setInputs node inputs)
              (inputs.find? (·.1 == "in") |>.map (·.2) |>.getD #[])
              context (path.push localId) nodePublicId
            expanded := {
              nodes := expanded.nodes ++ child.nodes
              sourceMap := expanded.sourceMap ++ child.sourceMap
              outputs := expanded.outputs.push (localId,
                (lookupOutput child.outputs "__module_output__").getD #[nodePublicId]) }
          else
            let bound ← bindAtomicParams (setInputs node inputs) context
            let flat := jsonSet bound "id" (.str nodePublicId)
            expanded := {
              nodes := expanded.nodes.push flat
              sourceMap := expanded.sourceMap.push <| Json.mkObj [
                ("expanded", .str nodePublicId),
                ("definition", .str definition.id),
                ("definition_version", Lean.toJson definition.version),
                ("local", .str localId),
                ("path", .arr (path.map Json.str))]
              outputs := expanded.outputs.push (localId, #[nodePublicId]) }
      let some result := lookupOutput expanded.outputs outputSource
        | throw s!"hierarchy: definition '{definition.id}' output source '{outputSource}' did not elaborate"
      pure { expanded with outputs := expanded.outputs.push ("__module_output__", result) }

  let mut flatNodes := #[]
  let mut sourceMap := #[]
  let mut outputs : Array (String × Array String) := #[]
  for node in sceneNodes do
    let some id := strField? node "id"
      | throw "hierarchy: scene contains a node without id"
    let some kind := strField? node "kind"
      | throw s!"hierarchy: scene node '{id}' has no kind"
    let inputs ← translatedInputs node outputs
    if kind == "module" then
      let module ← expandModule (definitions.size + 1) (setInputs node inputs)
        (inputs.find? (·.1 == "in") |>.map (·.2) |>.getD #[])
        #[] #[id] id
      flatNodes := flatNodes ++ module.nodes
      sourceMap := sourceMap ++ module.sourceMap
      outputs := outputs.push (id,
        (lookupOutput module.outputs "__module_output__").getD #[id])
    else
      flatNodes := flatNodes.push (setInputs node inputs)
      outputs := outputs.push (id, #[id])

  let sceneOut := strField? scene "out" |>.getD ""
  let flatIds := flatNodes.filterMap (strField? · "id")
  unless flatIds.zipIdx.all (fun (id, index) =>
      !(flatIds.extract 0 index).contains id) do
    throw "hierarchy: hygienic expansion id collides with an authored node id"
  let flatOut := (lookupOutput outputs sceneOut).bind (·[0]?) |>.getD sceneOut
  let taps := match field? document "taps" with
    | some (.arr requested) => .arr <| requested.filterMap fun value => do
        let source ← str? value
        let resolved ← lookupOutput outputs source
        let id ← resolved[0]?
        some (.str id)
    | some other => other
    | none => .arr #[]
  pure <| Json.mkObj [
    ("nodes", .arr flatNodes),
    ("out", .str flatOut),
    ("taps", taps),
    ("source_map", .arr sourceMap)]

end Tropical.Playground
