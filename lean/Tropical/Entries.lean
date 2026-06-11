import Tropical.Ir.Nodes
import Tropical.Ir.Codec
import Tropical.Ir.Emit

/-!
# Catalog entries — engine-side rendering (Phase 6 stage 6f)

Port of the compiler service's `concreteEntry` / `genericEntry`: the
port-metadata shape `ProgMeta.fromEntry` consumes (and list_programs /
get_info surface), rendered from the engine's own typed store instead
of a service response. `type` is the display string
(`portTypeToString`), `type_obj` the structured IR PortType JSON
(scalar / alias-with-decl / array), `default` the raw wire-format
ExprNode lowered from the resolved input default (`literalDefault`),
and `registers` carry the slot-table names with the array-init shape
lift (`ensureSlots`' override).

`resolved` is the program's own `tropical_resolved_1` encoding —
`SessionSt.adoptResolved` decodes it back into the store, preserving
the round-trip discipline the service path had (the adopted copy is
the codec image, not the strata output by identity).
-/

namespace Tropical.Entries

open Lean (Json JsonNumber toJson)
open Tropical.Ir

private def jsonNull : Json := Json.null

private def aliasJson (arena : Arena) (td : TypeDefIdx) : Json :=
  match arena.typeDef? td with
  | some (.alias name base) => Json.mkObj
      [("op", Json.str "aliasTypeDef"), ("name", Json.str name),
       ("base", Json.str base.wire)]
  | _ => jsonNull

private def elemJson (arena : Arena) : ScalarOrAlias → Json
  | .scalar k => Json.str k.wire
  | .alias td => aliasJson arena td

private def elemName (arena : Arena) : ScalarOrAlias → String
  | .scalar k => k.wire
  | .alias td =>
    match arena.typeDef? td with
    | some t => t.name
    | none => "?"

private def dimJson (arena : Arena) : ShapeDim → Json
  | .lit n => Json.num n
  | .typeParam i =>
    match arena.typeParam? i with
    | some tp => Json.mkObj <|
        [("op", Json.str "typeParamDecl"), ("name", Json.str tp.name)]
        ++ (match tp.default? with | some d => [("default", Json.num d)] | none => [])
    | none => jsonNull

private def dimStr : ShapeDim → String
  | .lit n => Tropical.Ir.Emit.jsNumString n
  | .typeParam _ => "?"

/-- The structured `type_obj` (the serialized IR `PortType`). -/
def portTypeObj (arena : Arena) : PortType → Json
  | .scalar k => Json.mkObj [("kind", Json.str "scalar"), ("scalar", Json.str k.wire)]
  | .alias td => Json.mkObj [("kind", Json.str "alias"), ("alias", aliasJson arena td)]
  | .array element shape => Json.mkObj
      [("kind", Json.str "array"), ("element", elemJson arena element),
       ("shape", Json.arr (shape.map (dimJson arena)))]

/-- `portTypeToString` (the display string). -/
def portTypeStr (arena : Arena) : PortType → String
  | .scalar k => k.wire
  | .alias td =>
    match arena.typeDef? td with
    | some t => t.name
    | none => "?"
  | .array element shape =>
    s!"{elemName arena element}[{String.intercalate "," (shape.map dimStr).toList}]"

private def wireOpName : Expr → String
  | .binary tag .. => tag.wire
  | .unary tag _ => tag.wire
  | .clamp .. => "clamp" | .select .. => "select"
  | .arraySet .. => "arraySet" | .index .. => "index"
  | .zeros _ => "zeros"
  | .inputRef _ => "inputRef" | .regRef _ => "regRef" | .paramRef _ => "paramRef"
  | .typeParamRef _ => "typeParamRef" | .bindingRef _ => "bindingRef"
  | .nestedOut .. => "nestedOut"
  | .sampleRate => "sampleRate" | .sampleIndex => "sampleIndex"
  | .fold .. => "fold" | .scan .. => "scan" | .generate .. => "generate"
  | .iterate .. => "iterate" | .chain .. => "chain"
  | .map2 .. => "map2" | .zipWith .. => "zipWith" | .letIn .. => "let"
  | .tag .. => "tag" | .match_ .. => "match"
  | .num _ => "num" | .bool _ => "bool" | .arr _ => "arr"

/-- Port of `literalDefault`: lower a resolved input default to the raw
    wire-format ExprNode (literal-class forms only). -/
partial def literalDefault (portName : String) : Expr → Except String Json
  | .num n => .ok (Json.num n)
  | .bool b => .ok (Json.bool b)
  | .arr items => do
    .ok (Json.arr (← items.mapM (literalDefault portName)))
  | .binary tag a b => opArgs tag.wire #[a, b]
  | .unary tag a => opArgs tag.wire #[a]
  | .clamp a b c => opArgs "clamp" #[a, b, c]
  | .select a b c => opArgs "select" #[a, b, c]
  | .index a b => opArgs "index" #[a, b]
  | .arraySet a b c => opArgs "arraySet" #[a, b, c]
  | .sampleRate => .ok (Json.mkObj [("op", Json.str "sampleRate")])
  | .sampleIndex => .ok (Json.mkObj [("op", Json.str "sampleIndex")])
  | .zeros count => do
    .ok (Json.mkObj [("op", Json.str "zeros"),
      ("count", ← literalDefault portName count)])
  | e =>
    .error (s!"Compiled: input '{portName}' default has op '{wireOpName e}' that's not a literal-class form; "
      ++ "defaults shouldn't reference decls or run combinators")
where
  opArgs (op : String) (args : Array Expr) : Except String Json := do
    .ok (Json.mkObj [("op", Json.str op),
      ("args", Json.arr (← args.mapM (literalDefault portName)))])

/-- A register's display type: the declared scalar/alias, overridden
    to `float[N]` for array-init regs (`ensureSlots`' shape lift). -/
private def regPortType (init : Expr) (type? : Option ScalarOrAlias) :
    Option PortType :=
  let declared : Option PortType := match type? with
    | none => none
    | some (.scalar k) => some (.scalar k)
    | some (.alias td) => some (.alias td)
  let arrayOf : Nat → PortType := fun n =>
    .array (.scalar .float) #[.lit (JsonNumber.fromNat n)]
  match init with
  | .arr items => some (arrayOf items.size)
  | .zeros (.num n) => some (arrayOf n.toFloat.toUInt64.toNat)
  | _ => declared

/-- The service's `concreteEntry`, off the typed store. -/
def concreteEntry (arena : Arena) (entryName : String) (idx : ProgramIdx) :
    Except String Json := do
  let some prog := arena.program? idx
    | .error s!"entry render: program pool index {idx.idx} out of range"
  let resolved ← Codec.encodeResolved arena idx
  let inputs ← prog.inputs.mapM fun d => do
    let (t, tObj) := match d.type? with
      | some pt => (Json.str (portTypeStr arena pt), portTypeObj arena pt)
      | none => (jsonNull, jsonNull)
    let dflt ← match d.default? with
      | some e => literalDefault d.name e
      | none => pure jsonNull
    .ok <| Json.mkObj [("name", Json.str d.name), ("type", t),
      ("type_obj", tObj), ("default", dflt)]
  let outputs := prog.outputs.map fun d =>
    let (t, tObj) := match d.type? with
      | some pt => (Json.str (portTypeStr arena pt), portTypeObj arena pt)
      | none => (jsonNull, jsonNull)
    Json.mkObj [("name", Json.str d.name), ("type", t), ("type_obj", tObj)]
  let registers := prog.regs.filterMap fun d =>
    match d with
    | .reg name init _ type? _ =>
      let (t, tObj) := match regPortType init type? with
        | some pt => (Json.str (portTypeStr arena pt), portTypeObj arena pt)
        | none => (jsonNull, jsonNull)
      some <| Json.mkObj [("name", Json.str name), ("type", t), ("type_obj", tObj)]
    | _ => none
  .ok <| Json.mkObj [
    ("program_name", Json.str entryName),
    ("generic", Json.bool false),
    ("type_params", jsonNull),
    ("resolved", resolved),
    ("inputs", Json.arr inputs),
    ("outputs", Json.arr outputs),
    ("registers", Json.arr registers)]

/-- The service's `genericEntry` (raw templates; `resolved: null`). -/
def genericEntry (arena : Arena) (entryName : String) (idx : ProgramIdx) : Json :=
  let prog := (arena.program? idx).getD {name := entryName}
  let typeParams := Json.mkObj <| prog.typeParams.toList.filterMap fun i =>
    match arena.typeParam? i with
    | some tp => some (tp.name, Json.mkObj <|
        [("type", Json.str "int")]
        ++ (match tp.default? with | some d => [("default", Json.num d)] | none => []))
    | none => none
  Json.mkObj [
    ("program_name", Json.str entryName),
    ("generic", Json.bool true),
    ("resolved", jsonNull),
    ("type_params", typeParams),
    ("inputs", Json.arr (prog.inputs.map fun d => Json.mkObj
      [("name", Json.str d.name), ("type", jsonNull), ("type_obj", jsonNull),
       ("default", jsonNull)])),
    ("outputs", Json.arr (prog.outputs.map fun d => Json.mkObj
      [("name", Json.str d.name), ("type", jsonNull), ("type_obj", jsonNull)])),
    ("registers", Json.arr #[])]

end Tropical.Entries
