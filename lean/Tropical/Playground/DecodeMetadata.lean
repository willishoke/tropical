import Tropical.Playground.VocabularyMetadata

/-!
# Playground decode metadata

Pure graph-shape decoding, validation, vocabulary serialization, and parameter
table construction.  Scalar and modal expression construction deliberately do
not appear here.
-/

namespace Tropical.Playground.Metadata

open Lean (Json JsonNumber)

def masterVelocityParam : String := "master.velocity"
def masterTauBaseParam : String := "master.tau_base"
def masterGainParam : String := "master.gain"
def masterGainDefault : JsonNumber := ⟨37, 1⟩

structure Raw where
  id : String
  kind : String
  sel : Json
  params : Json
  inObj : Json
  paramAliases : Json

def decodeRaw (node : Json) : Option Raw :=
  match node.getObjVal? "id", node.getObjVal? "kind" with
  | .ok (.str id), .ok (.str kind) =>
    let sel := (node.getObjVal? "sel").toOption.getD (Json.mkObj [])
    let params := (node.getObjVal? "params").toOption.getD (Json.mkObj [])
    let inObj := (node.getObjVal? "in").toOption.getD (Json.mkObj [])
    let paramAliases :=
      (node.getObjVal? "param_aliases").toOption.getD (Json.mkObj [])
    some { id, kind, sel, params, inObj, paramAliases }
  | _, _ => none

def rawsOf (json : Json) : Array Raw :=
  match (json.getObjVal? "nodes").toOption.bind (·.getArr?.toOption) with
  | some nodes => nodes.filterMap decodeRaw
  | none => #[]

def paramNameOf (raw : Raw) (knob : String) : String :=
  match raw.paramAliases.getObjVal? knob with
  | .ok (.str name) => name
  | _ => s!"{raw.id}.{knob}"

def knobNamesOf (kind : String) : Array String :=
  (portSpecs kind).filterMap fun port =>
    if port.knob.isSome then some port.name else none

def domStr : PortDomain → String
  | .signal => "signal"
  | .modal => "modal"
  | .control => "control"

def discStr : Discipline → String
  | .raw => "raw"
  | .glide => "glide"
  | .anchor => "anchor"

def checkEdgeTypes (raws : Array Raw) : Except String Unit := do
  for raw in raws do
    for port in portSpecs raw.kind do
      let sources := portSources raw.inObj port.name
      if port.accepts.isEmpty && port.knob.isSome && !sources.isEmpty then
        throw s!"connection error: '{raw.id}' ({raw.kind}) has a wire into '{port.name}', which is a knob (a set value), not an inlet — set it via its param slot (or wire its owner port), do not wire the knob itself"
      unless port.accepts.isEmpty do
        if !port.multi && sources.size > 1 then
          throw s!"connection arity error: '{raw.id}' ({raw.kind}) inlet '{port.name}' accepts one source but has {sources.size} authored wires"
        for sourceId in sources do
          match raws.find? (·.id == sourceId) with
          | none =>
            throw s!"connection error: '{raw.id}' ({raw.kind}) inlet '{port.name}' is wired from '{sourceId}', which is not a node in the patch — a wire must name an existing node"
          | some source =>
            match outletOf source.kind with
            | none =>
              throw s!"connection type error: '{source.id}' ({source.kind}) has no outlet but is wired into '{raw.id}' ({raw.kind}) inlet '{port.name}'"
            | some domain =>
              unless port.accepts.contains domain do
                let accepted :=
                  String.intercalate "/" (port.accepts.toList.map domStr)
                throw s!"connection type error: '{source.id}' ({source.kind}, {domStr domain} outlet) → '{raw.id}' ({raw.kind}) inlet '{port.name}' which accepts {accepted} — outlet.color ∉ inlet.accepts (modal→signal realizes; signal→modal is a type error)"
  pure ()

def checkOutTarget (json : Json) (raws : Array Raw) : Except String Unit :=
  match (json.getObjVal? "out").toOption with
  | some (.str outputId) =>
    if outputId == "" || raws.any (·.id == outputId) then pure ()
    else
      throw s!"output target error: the top-level \"out\" names node '{outputId}', which is not in the patch — route the dac from an existing node (or omit \"out\" for a silent patch)"
  | _ => pure ()

def checkServedKinds (raws : Array Raw) : Except String Unit := do
  for raw in raws do
    if withheldKinds.contains raw.kind then
      throw s!"unserved kind: '{raw.id}' has kind '{raw.kind}', which the engine builds but WITHHOLDS from the surface vocabulary (conditioning is guarded, but arbitrary-table factor landing, profile bounds, cost, and live-beta topology are not yet a served contract) — not available as a patch node"
    unless vocabularyKinds.contains raw.kind ||
        hierarchyAtomKinds.contains raw.kind do
      throw s!"unknown kind: '{raw.id}' has kind '{raw.kind}', which is not a served node kind — see get_vocabulary for the {vocabularyKinds.size} kinds the surface builds"
  pure ()

def vocabularySchema : String := "tropical_vocabulary"
def vocabularySchemaVersion : Nat := 1

def fnv1a64 (bytes : ByteArray) : UInt64 := Id.run do
  let mut hash : UInt64 := 14695981039346656037
  for byte in bytes do
    hash := (hash ^^^ byte.toUInt64) * 1099511628211
  return hash

private def hexDigit (value : Nat) : Char :=
  if value < 10 then Char.ofNat ('0'.toNat + value)
  else Char.ofNat ('a'.toNat + value - 10)

def uint64Hex (value : UInt64) : String := Id.run do
  let mut output := ""
  for index in [0:16] do
    let shift := 4 * (15 - index)
    let nibble := ((value >>> shift.toUInt64) &&& 0xf).toNat
    output := output.push (hexDigit nibble)
  return output

def vocabularyPayloadJson : Json :=
  let portJson := fun (port : PortSpec) => Json.mkObj <|
    [("name", Json.str port.name)]
    ++ (if port.accepts.isEmpty then [] else
        [("accepts", Json.arr (port.accepts.map (Json.str ∘ domStr))),
         ("multi", Json.bool port.multi)])
    ++ (match port.knob with
        | some (mantissa, exponent) =>
          [("default", Json.num ⟨mantissa, exponent⟩),
           ("discipline", Json.str (discStr port.discipline))]
        | none => [])
    ++ (match port.display with
        | some metadata =>
          [("min", Lean.toJson metadata.min), ("max", Lean.toJson metadata.max),
           ("log", Json.bool metadata.log), ("unit", Json.str metadata.unit)]
        | none => [])
    ++ (match port.ownerPort with
        | some owner => [("owner", Json.str owner)]
        | none => [])
  Json.mkObj [
    ("rule", Json.str
      "outlet→inlet valid iff outlet.color ∈ inlet.accepts; modal→signal realizes, signal→modal is a type error"),
    ("colors", Json.arr #[Json.str "signal", Json.str "modal", Json.str "control"]),
    ("kinds", Json.arr (vocabularyKinds.map fun kind =>
      Json.mkObj [
        ("kind", Json.str kind),
        ("outlet", match outletOf kind with
          | some domain => Json.str (domStr domain)
          | none => Json.null),
        ("ports", Json.arr ((portSpecs kind).map portJson))]))]

def vocabularyFingerprint : String :=
  s!"fnv1a64:{uint64Hex (fnv1a64 vocabularyPayloadJson.compress.toUTF8)}"

def vocabularyJson : Json :=
  match vocabularyPayloadJson with
  | .obj fields => .obj <|
      fields.insert "schema" (Json.str vocabularySchema)
        |>.insert "schema_version" (Lean.toJson vocabularySchemaVersion)
        |>.insert "fingerprint" (Json.str vocabularyFingerprint)
  | payload => Json.mkObj [
      ("schema", Json.str vocabularySchema),
      ("schema_version", Lean.toJson vocabularySchemaVersion),
      ("fingerprint", Json.str vocabularyFingerprint),
      ("payload", payload)]

def collectParams (raws : Array Raw) : Array (String × JsonNumber) := Id.run do
  let mut parameters : Array (String × JsonNumber) := #[
    (masterVelocityParam, ⟨1, 0⟩),
    (masterTauBaseParam, ⟨0, 0⟩),
    (masterGainParam, masterGainDefault)]
  for raw in raws do
    if raw.kind == "out" then continue
    for port in portSpecs raw.kind do
      if port.knob.isNone then continue
      let knob := port.name
      let selfWired := !(portSources raw.inObj knob).isEmpty
      let ownerWired := match port.ownerPort with
        | some owner => !(portSources raw.inObj owner).isEmpty
        | none => false
      if !selfWired && !ownerWired then
        let base := paramNameOf raw knob
        let dflt := (jNum? raw.params knob).getD ⟨0, 0⟩
        let alreadyRegistered := parameters.any fun (name, _) =>
          name == base || name == s!"{base}#v0" || name == s!"{base}#phase"
        if alreadyRegistered then continue
        if isGlided raw.kind knob then
          parameters := parameters.push (s!"{base}#v0", dflt)
          parameters := parameters.push (s!"{base}#v1", dflt)
          parameters := parameters.push (s!"{base}#t0", ⟨0, 0⟩)
          for index in [0:4] do
            parameters := parameters.push (s!"{base}#t0#u{index}", ⟨0, 0⟩)
        else
          parameters := parameters.push (base, dflt)
          if isAnchored raw.kind knob then
            parameters := parameters.push (s!"{base}#phase", ⟨0, 0⟩)
    if raw.kind == "resonator" &&
        (jNum? raw.params "partials_max").isSome then
      parameters := parameters.push
        (s!"{raw.id}.partials", (jNum? raw.params "partials").getD ⟨6, 0⟩)
  return parameters

end Tropical.Playground.Metadata
