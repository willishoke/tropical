import Lean.Data.Json
import Tropical.Parse.OrderedJson

/-!
# ParsedProgram — the typed parser AST (port of compiler/parse/nodes.ts)

The Lean mirror of the strict discriminated-union node types the TS
parser and the `raise` JSON adapter emit. This is the free seam of the
port: plain serializable data, `nameRef` placeholders everywhere a name
awaits resolution, zero scope analysis.

Design decisions (versus the TS shapes):

- **Numbers are `JsonNumber`.** TS works in IEEE-754 doubles; the
  differential comparators compare numbers by double bit pattern.
  `JsonNumber` preserves the decimal text the JSON parser saw, so a
  decode∘encode round trip re-emits the same decimal and parses back
  to the identical double on the TS side. (Emitting `Float` instead
  would force a shortest-round-trip printer; carrying the decimal is
  both lossless and simpler.)

- **`NestedOut.output` is a name, not a number.** The TS elaborator
  tolerates `output` being a raw number (legacy slack), but no
  ParsedProgram *producer* emits one: `raise.ts` stringifies via
  `nameRef(String(node.output))`, the surface parser and
  `session_to_parsed.ts` always emit `nameRef`. The codec therefore
  decodes `output` strictly as a NameRef node.

- **No `bounds` field.** `ProgramPortSpec.bounds` is parser-internal:
  `lowerBoundsToClamps` strips it before any ParsedProgram escapes the
  parser (and the Zod schema strips it from JSON input before `raise`
  runs), so serialized ParsedProgram JSON can never carry it.

- **Absence is `Option.none`, never `null`.** Optional fields that TS
  leaves `undefined` are simply omitted from the emitted JSON, so
  round trips are structural-equal. `some #[]` and `none` are distinct
  (present-empty vs absent) for the array-valued options.

The encoder: `Program.toJson : Program → Json` (the `diffcli raise`
inspection surface). Ingest is `Parse/Raise.lean` (`tropical_program_2`
JSON); the standalone ParsedProgram decoder was deleted with its last
consumer (the TS differential harness).
-/

namespace Tropical.Parse

open Lean (Json JsonNumber)

-- ─────────────────────────────────────────────────────────────
-- Op tags
-- ─────────────────────────────────────────────────────────────

inductive BinaryOpTag where
  | add | sub | mul | div | mod
  | lt | lte | gt | gte | eq | neq
  | and | or
  | bitAnd | bitOr | bitXor | lshift | rshift
deriving BEq, Repr

def BinaryOpTag.wire : BinaryOpTag → String
  | .add => "add" | .sub => "sub" | .mul => "mul" | .div => "div" | .mod => "mod"
  | .lt => "lt" | .lte => "lte" | .gt => "gt" | .gte => "gte"
  | .eq => "eq" | .neq => "neq"
  | .and => "and" | .or => "or"
  | .bitAnd => "bitAnd" | .bitOr => "bitOr" | .bitXor => "bitXor"
  | .lshift => "lshift" | .rshift => "rshift"

def BinaryOpTag.ofWire? : String → Option BinaryOpTag
  | "add" => some .add | "sub" => some .sub | "mul" => some .mul
  | "div" => some .div | "mod" => some .mod
  | "lt" => some .lt | "lte" => some .lte | "gt" => some .gt | "gte" => some .gte
  | "eq" => some .eq | "neq" => some .neq
  | "and" => some .and | "or" => some .or
  | "bitAnd" => some .bitAnd | "bitOr" => some .bitOr | "bitXor" => some .bitXor
  | "lshift" => some .lshift | "rshift" => some .rshift
  | _ => none

theorem BinaryOpTag.ofWire_wire (t : BinaryOpTag) :
    BinaryOpTag.ofWire? t.wire = some t := by
  cases t <;> rfl

inductive UnaryOpTag where
  | neg | not | bitNot
deriving BEq, Repr

def UnaryOpTag.wire : UnaryOpTag → String
  | .neg => "neg" | .not => "not" | .bitNot => "bitNot"

def UnaryOpTag.ofWire? : String → Option UnaryOpTag
  | "neg" => some .neg | "not" => some .not | "bitNot" => some .bitNot
  | _ => none

theorem UnaryOpTag.ofWire_wire (t : UnaryOpTag) :
    UnaryOpTag.ofWire? t.wire = some t := by
  cases t <;> rfl

-- ─────────────────────────────────────────────────────────────
-- ParsedExpr — value-producing universe
-- ─────────────────────────────────────────────────────────────

/-- Port of `ParsedExpr` / `ParsedExprOp`. `num`/`bool`/`arr` are the
    literal layer (TS: bare JSON values); the rest are the op-tagged
    nodes. The op string exists only in the codec. Trunk vocabulary
    only: the combinator/sum-type/binder tail (`let`, `fold`, `scan`,
    `generate`, `iterate`, `chain`, `map2`, `zipWith`, `tag`, `match`,
    `binding`) was retired with its producers, and its spellings are
    refused at ingest — no layer past the codec can carry them. -/
inductive ParsedExpr where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array ParsedExpr)
  | binary (tag : BinaryOpTag) (lhs rhs : ParsedExpr)
  | unary (tag : UnaryOpTag) (arg : ParsedExpr)
  | call (callee : ParsedExpr) (args : Array ParsedExpr)
  | nameRef (name : String)
  /-- `inst.port` — both components are NameRefs on the wire. -/
  | nestedOut (ref output : String)
  | index (arr idx : ParsedExpr)
deriving Inhabited, Repr

-- ─────────────────────────────────────────────────────────────
-- Ports and port types
-- ─────────────────────────────────────────────────────────────

/-- Port type: bare scalar/alias NameRef, or array of element + literal
    shape (type-param shape dims were retired with generics). -/
inductive PortTypeDecl where
  | scalar (name : String)
  | array (element : String) (shape : Array JsonNumber)
deriving Inhabited, Repr

structure ProgramPortSpec where
  name : String
  type? : Option PortTypeDecl := none
  default? : Option ParsedExpr := none
deriving Inhabited, Repr

/-- Port entry: bare-name short form or full spec. -/
inductive ProgramPort where
  | bare (name : String)
  | spec (s : ProgramPortSpec)
deriving Inhabited, Repr

inductive ScalarKind where
  | float | int | bool
deriving BEq, Inhabited, Repr

def ScalarKind.wire : ScalarKind → String
  | .float => "float" | .int => "int" | .bool => "bool"

def ScalarKind.ofWire? : String → Option ScalarKind
  | "float" => some .float | "int" => some .int | "bool" => some .bool
  | _ => none

theorem ScalarKind.ofWire_wire (k : ScalarKind) :
    ScalarKind.ofWire? k.wire = some k := by
  cases k <;> rfl

structure ProgramPorts where
  inputs : Option (Array ProgramPort) := none
  outputs : Option (Array ProgramPort) := none
deriving Inhabited, Repr

-- ─────────────────────────────────────────────────────────────
-- BodyAssign — wires pinning a value to a port
-- ─────────────────────────────────────────────────────────────

inductive BodyAssign where
  | output (name : String) (expr : ParsedExpr)
deriving Inhabited, Repr

-- ─────────────────────────────────────────────────────────────
-- BodyDecl + Block + Program
-- ─────────────────────────────────────────────────────────────

mutual

inductive BodyDecl where
  | param (name : String) (value? : Option JsonNumber)
  /-- `program` is a NameRef; `inputs` keeps entry order (each entry's
      key is a NameRef on the wire). -/
  | inst (name program : String)
      (inputs : Option (Array (String × ParsedExpr)))
  | prog (name : String) (program : Program)
deriving Inhabited, Repr

inductive Block where
  | mk (decls : Array BodyDecl) (assigns : Array BodyAssign)
deriving Inhabited, Repr

inductive Program where
  | mk (name : String)
      (ports : Option ProgramPorts)
      (body : Block)
      (breaksCycles : Option Bool)
deriving Inhabited, Repr

end

def Block.decls : Block → Array BodyDecl
  | .mk d _ => d

def Block.assigns : Block → Array BodyAssign
  | .mk _ a => a

def Program.name : Program → String
  | .mk n _ _ _ => n

def Program.ports : Program → Option ProgramPorts
  | .mk _ p _ _ => p

def Program.body : Program → Block
  | .mk _ _ b _ => b

def Program.breaksCycles : Program → Option Bool
  | .mk _ _ _ bc => bc

-- ─────────────────────────────────────────────────────────────
-- Encode — typed AST → Lean.Json
-- ─────────────────────────────────────────────────────────────

namespace Encode

private def jStr (s : String) : Json := Json.str s

private def jNameRef (name : String) : Json :=
  Json.mkObj [("op", jStr "nameRef"), ("name", jStr name)]

/-- Append an optional field only when present (TS-absent stays absent). -/
private def optField (key : String) : Option Json → List (String × Json)
  | some v => [(key, v)]
  | none => []

def expr : ParsedExpr → Json
  | .num n => Json.num n
  | .bool b => Json.bool b
  | .arr items => Json.arr (items.attach.map fun ⟨x, _⟩ => expr x)
  | .binary tag lhs rhs =>
    Json.mkObj [("op", jStr tag.wire), ("args", Json.arr #[expr lhs, expr rhs])]
  | .unary tag arg =>
    Json.mkObj [("op", jStr tag.wire), ("args", Json.arr #[expr arg])]
  | .call callee args =>
    Json.mkObj [("op", jStr "call"), ("callee", expr callee),
                ("args", Json.arr (args.attach.map fun ⟨x, _⟩ => expr x))]
  | .nameRef name => jNameRef name
  | .nestedOut ref output =>
    Json.mkObj [("op", jStr "nestedOut"), ("ref", jNameRef ref),
                ("output", jNameRef output)]
  | .index arr idx =>
    Json.mkObj [("op", jStr "index"), ("args", Json.arr #[expr arr, expr idx])]
termination_by e => sizeOf e
decreasing_by
  all_goals first
    | (simp; omega)
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ args›; simp; omega)

def portTypeDecl : PortTypeDecl → Json
  | .scalar name => jNameRef name
  | .array element shape =>
    Json.mkObj [("kind", jStr "array"), ("element", jNameRef element),
                ("shape", Json.arr (shape.map Json.num))]

def programPort : ProgramPort → Json
  | .bare name => jStr name
  | .spec s =>
    Json.mkObj <|
      [("name", jStr s.name)]
      ++ optField "type" (s.type?.map portTypeDecl)
      ++ optField "default" (s.default?.map expr)

def programPorts (p : ProgramPorts) : Json :=
  Json.mkObj <|
    optField "inputs" (p.inputs.map fun ps => Json.arr (ps.map programPort))
    ++ optField "outputs" (p.outputs.map fun ps => Json.arr (ps.map programPort))

def bodyAssign : BodyAssign → Json
  | .output name e =>
    Json.mkObj [("op", jStr "outputAssign"), ("name", jStr name), ("expr", expr e)]

mutual

def bodyDecl : BodyDecl → Json
  | .param name value? =>
    Json.mkObj <|
      [("op", jStr "paramDecl"), ("name", jStr name)]
      ++ optField "value" (value?.map Json.num)
  | .inst name progName inputs =>
    Json.mkObj <|
      [("op", jStr "instanceDecl"), ("name", jStr name),
       ("program", jNameRef progName)]
      ++ optField "inputs" (inputs.map fun entries =>
           Json.arr (entries.map fun (port, value) =>
             Json.mkObj [("port", jNameRef port), ("value", expr value)]))
  | .prog name p =>
    Json.mkObj [("op", jStr "programDecl"), ("name", jStr name),
                ("program", program p)]
termination_by d => sizeOf d
decreasing_by simp; omega

def block : Block → Json
  | .mk decls assigns =>
    Json.mkObj [("op", jStr "block"),
                ("decls", Json.arr (decls.attach.map fun ⟨d, _⟩ => bodyDecl d)),
                ("assigns", Json.arr (assigns.map bodyAssign))]
termination_by b => sizeOf b
decreasing_by have := Array.sizeOf_lt_of_mem ‹_ ∈ decls›; simp; omega

def program : Program → Json
  | .mk name ports body breaksCycles =>
    Json.mkObj <|
      [("op", jStr "program"), ("name", jStr name), ("body", block body)]
      ++ optField "ports" (ports.map programPorts)
      ++ optField "breaks_cycles" (breaksCycles.map Json.bool)
termination_by p => sizeOf p
decreasing_by simp; omega

end

end Encode

def Program.toJson (p : Program) : Json := Encode.program p

end Tropical.Parse
