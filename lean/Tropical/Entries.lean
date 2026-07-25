import Tropical.Ir.Nodes
import Tropical.Ir.Codec
import Tropical.Ir.Emit

/-!
# Catalog entries — engine-side rendering

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

/-- The structured `type_obj` (the serialized IR `PortType`). -/
def portTypeObj : PortType → Json
  | .scalar k => Json.mkObj [("kind", Json.str "scalar"), ("scalar", Json.str k.wire)]
  | .array element shape => Json.mkObj
      [("kind", Json.str "array"), ("element", Json.str element.wire),
       ("shape", Json.arr (shape.map Json.num))]

/-- `portTypeToString` (the display string). -/
def portTypeStr : PortType → String
  | .scalar k => k.wire
  | .array element shape =>
    s!"{element.wire}[{String.intercalate "," (shape.map Tropical.Ir.Emit.jsNumString).toList}]"

private def wireOpName : ENode → String
  | .binary tag .. => tag.wire
  | .unary tag _ => tag.wire
  | .clamp .. => "clamp" | .select .. => "select"
  | .arraySet .. => "arraySet" | .index .. => "index"
  | .inputRef _ => "inputRef" | .paramRef _ => "paramRef"
  | .nestedOut .. => "nestedOut"
  | .sampleRate => "sampleRate" | .sampleIndex => "sampleIndex"
  | .loopIdx _ => "loopIdx" | .bankSum .. => "bankSum"
  | .num _ => "num" | .bool _ => "bool" | .arr _ => "arr"

/-- Port of `literalDefault`: lower a resolved input default (an `ExprId` into
    the arena's DAG) to the raw wire-format ExprNode (literal-class forms only). -/
partial def literalDefault (ea : ExprArena) (portName : String) (id : ExprId) :
    Except String Json :=
  match ea.deref id with
  | none => .error s!"Compiled: input '{portName}' default references a dangling ExprId {id.idx}"
  | some node => match node with
    | .num n => .ok (Json.num n)
    | .bool b => .ok (Json.bool b)
    | .arr items => do
      .ok (Json.arr (← items.mapM (literalDefault ea portName)))
    | .binary tag a b => opArgs tag.wire #[a, b]
    | .unary tag a => opArgs tag.wire #[a]
    | .clamp a b c => opArgs "clamp" #[a, b, c]
    | .select a b c => opArgs "select" #[a, b, c]
    | .index a b => opArgs "index" #[a, b]
    | .arraySet a b c => opArgs "arraySet" #[a, b, c]
    | .sampleRate => .ok (Json.mkObj [("op", Json.str "sampleRate")])
    | .sampleIndex => .ok (Json.mkObj [("op", Json.str "sampleIndex")])
    | e =>
      .error (s!"Compiled: input '{portName}' default has op '{wireOpName e}' that's not a literal-class form; "
        ++ "defaults shouldn't reference decls or run combinators")
where
  opArgs (op : String) (args : Array ExprId) : Except String Json := do
    .ok (Json.mkObj [("op", Json.str op),
      ("args", Json.arr (← args.mapM (literalDefault ea portName)))])

/-- The service's `concreteEntry`, off the typed store. -/
def concreteEntry (arena : Arena) (entryName : String) (idx : ProgramIdx) :
    Except String Json := do
  let some prog := arena.program? idx
    | .error s!"entry render: program pool index {idx.idx} out of range"
  let resolved ← Codec.encodeResolved arena idx
  let inputs ← prog.inputs.mapM fun d => do
    let (t, tObj) := match d.type? with
      | some pt => (Json.str (portTypeStr pt), portTypeObj pt)
      | none => (jsonNull, jsonNull)
    let dflt ← match d.default? with
      | some e => literalDefault arena.exprs d.name e
      | none => pure jsonNull
    .ok <| Json.mkObj [("name", Json.str d.name), ("type", t),
      ("type_obj", tObj), ("default", dflt)]
  let outputs := prog.outputs.map fun d =>
    let (t, tObj) := match d.type? with
      | some pt => (Json.str (portTypeStr pt), portTypeObj pt)
      | none => (jsonNull, jsonNull)
    Json.mkObj [("name", Json.str d.name), ("type", t), ("type_obj", tObj)]
  -- CF-only: programs have no reg decls, so the registers list is empty.
  let registers : Array Json := #[]
  .ok <| Json.mkObj [
    ("program_name", Json.str entryName),
    ("generic", Json.bool false),
    ("type_params", jsonNull),
    ("resolved", resolved),
    ("inputs", Json.arr inputs),
    ("outputs", Json.arr outputs),
    ("registers", Json.arr registers)]

end Tropical.Entries
