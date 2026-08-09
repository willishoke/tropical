import Tropical.Ir.Nodes
import Tropical.Ir.Codec
import Tropical.Ir.Emit

/-!
# Catalog entries — engine-side rendering

The Lean frontend renders the port-metadata shape consumed by
`ProgMeta.fromEntry` and exposed through `list_programs` / `get_info` directly
from its typed store. `type` is the display string
(`portTypeToString`), `type_obj` the structured IR PortType JSON
(scalar / alias-with-decl / array), `default` the raw wire-format
ExprNode lowered from the resolved input default (`literalDefault`),
and the retained `registers` carrier is empty for current production
programs.

`resolved` is the program's own `tropical_resolved_1` encoding —
`SessionSt.adoptResolved` decodes it back into the store, preserving
the codec boundary between registration metadata and the session arena.
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
  | .routedSum .. => "routedSum"
  | .num _ => "num" | .bool _ => "bool" | .arr _ => "arr"

private def mkOpNode (op : String) (args : Array Json) : Json :=
  Json.mkObj [("op", Json.str op), ("args", Json.arr args)]

/-- Port of `literalDefault`: lower a resolved input default (an `ExprId` into
    the arena's DAG) to the raw wire-format ExprNode (literal-class forms only).
    Total by the frozen-arena wf: `hw` certifies children sit strictly below
    their parent, so the walk descends `id.idx`. -/
def literalDefault (ea : ExprArena) (hw : ea.wf = true) (portName : String)
    (id : ExprId) : Except String Json :=
  match hd : ea.deref id with
  | none => .error s!"Compiled: input '{portName}' default references a dangling ExprId {id.idx}"
  | some node => match node with
    | .num n => .ok (Json.num n)
    | .bool b => .ok (Json.bool b)
    | .arr items => do
      .ok (Json.arr (← items.attach.mapM fun ⟨x, _⟩ => literalDefault ea hw portName x))
    | .binary tag a b => do
      .ok (mkOpNode tag.wire #[← literalDefault ea hw portName a, ← literalDefault ea hw portName b])
    | .unary tag a => do
      .ok (mkOpNode tag.wire #[← literalDefault ea hw portName a])
    | .clamp a b c => do
      .ok (mkOpNode "clamp" #[← literalDefault ea hw portName a,
        ← literalDefault ea hw portName b, ← literalDefault ea hw portName c])
    | .select a b c => do
      .ok (mkOpNode "select" #[← literalDefault ea hw portName a,
        ← literalDefault ea hw portName b, ← literalDefault ea hw portName c])
    | .index a b => do
      .ok (mkOpNode "index" #[← literalDefault ea hw portName a,
        ← literalDefault ea hw portName b])
    | .arraySet a b c => do
      .ok (mkOpNode "arraySet" #[← literalDefault ea hw portName a,
        ← literalDefault ea hw portName b, ← literalDefault ea hw portName c])
    | .sampleRate => .ok (Json.mkObj [("op", Json.str "sampleRate")])
    | .sampleIndex => .ok (Json.mkObj [("op", Json.str "sampleIndex")])
    | e =>
      .error (s!"Compiled: input '{portName}' default has op '{wireOpName e}' that's not a literal-class form; "
        ++ "defaults shouldn't reference decls or run combinators")
termination_by id.idx
decreasing_by
  all_goals
    apply Tropical.Ir.ExprArena.forall_children_lt hw ‹Tropical.Ir.ExprArena.deref _ _ = some _›
    simp_all [Tropical.Ir.ENode.children]

/-- A concrete registration entry rendered from the typed store. -/
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
      | some e =>
        if hw : arena.exprs.wf then literalDefault arena.exprs hw d.name e
        else .error "entry render: expression arena failed its well-formedness sweep"
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
