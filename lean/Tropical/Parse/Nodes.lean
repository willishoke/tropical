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

- **Record fields that stay records** (`let.bind`, `type_params`) are
  ordered association arrays so source order survives decode∘encode;
  records the TS layer converts to entry arrays (`inputs`,
  `type_args`, `payload`, `arms`) are arrays of typed entries.

The codec: `decodeProgram : JsonV → Except String Program` (strict —
unknown keys and malformed nodes are errors, never silently dropped)
and `Program.toJson : Program → Json`. `decode ∘ encode = id`, and
`encode ∘ decode` preserves structural equality with the TS
`JSON.stringify` shape (object key order may differ; every comparator
in the harness is key-order-insensitive).
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

inductive UnaryOpTag where
  | neg | not | bitNot
deriving BEq, Repr

def UnaryOpTag.wire : UnaryOpTag → String
  | .neg => "neg" | .not => "not" | .bitNot => "bitNot"

def UnaryOpTag.ofWire? : String → Option UnaryOpTag
  | "neg" => some .neg | "not" => some .not | "bitNot" => some .bitNot
  | _ => none

-- ─────────────────────────────────────────────────────────────
-- ParsedExpr — value-producing universe
-- ─────────────────────────────────────────────────────────────

mutual

/-- Port of `ParsedExpr` / `ParsedExprOp`. `num`/`bool`/`arr` are the
    literal layer (TS: bare JSON values); the rest are the op-tagged
    nodes. The op string exists only in the codec. -/
inductive ParsedExpr where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array ParsedExpr)
  | binary (tag : BinaryOpTag) (lhs rhs : ParsedExpr)
  | unary (tag : UnaryOpTag) (arg : ParsedExpr)
  | call (callee : ParsedExpr) (args : Array ParsedExpr)
  | nameRef (name : String)
  | binding (name : String)
  /-- `inst.port` — both components are NameRefs on the wire. -/
  | nestedOut (ref output : String)
  | index (arr idx : ParsedExpr)
  /-- `let { k: e, ... } in body`. `bind` keeps source key order. -/
  | letIn (bind : Array (String × ParsedExpr)) (body : ParsedExpr)
  | fold (over init : ParsedExpr) (accVar elemVar : String) (body : ParsedExpr)
  | scan (over init : ParsedExpr) (accVar elemVar : String) (body : ParsedExpr)
  | generate (count : ParsedExpr) (var : String) (body : ParsedExpr)
  | iterate (count : ParsedExpr) (var : String) (init body : ParsedExpr)
  | chain (count : ParsedExpr) (var : String) (init body : ParsedExpr)
  | map2 (over : ParsedExpr) (elemVar : String) (body : ParsedExpr)
  | zipWith (a b : ParsedExpr) (xVar yVar : String) (body : ParsedExpr)
  /-- Sum constructor. `payload` is `none` when absent in the JSON,
      `some entries` when present (raise omits empty payloads). -/
  | tag (variant : String) (payload : Option (Array TagPayloadEntry))
  | match_ (scrutinee : ParsedExpr) (arms : Array MatchArm)
deriving Inhabited, Repr

/-- `{ field: expr }` in tag construction; `field` is a NameRef. -/
inductive TagPayloadEntry where
  | mk (field : String) (value : ParsedExpr)
deriving Inhabited, Repr

/-- One match arm. `binds` pairs are `(payload field NameRef, binder)`;
    always present on the wire (empty array for payload-less arms). -/
inductive MatchArm where
  | mk (variant : String) (binds : Array (String × String)) (body : ParsedExpr)
deriving Inhabited, Repr

end

def TagPayloadEntry.field : TagPayloadEntry → String
  | .mk f _ => f

def TagPayloadEntry.value : TagPayloadEntry → ParsedExpr
  | .mk _ v => v

def MatchArm.variant : MatchArm → String
  | .mk v _ _ => v

def MatchArm.binds : MatchArm → Array (String × String)
  | .mk _ b _ => b

def MatchArm.body : MatchArm → ParsedExpr
  | .mk _ _ b => b

-- ─────────────────────────────────────────────────────────────
-- Ports, types, type defs
-- ─────────────────────────────────────────────────────────────

/-- Array-shape dimension: integer literal or NameRef (type-param). -/
inductive ShapeDim where
  | lit (n : JsonNumber)
  | ref (name : String)
deriving Inhabited, Repr

/-- Port type: bare scalar/alias NameRef, or array of element + shape. -/
inductive PortTypeDecl where
  | scalar (name : String)
  | array (element : String) (shape : Array ShapeDim)
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

structure StructField where
  name : String
  scalarType : ScalarKind
deriving Inhabited, Repr

structure SumVariant where
  name : String
  payload : Array StructField
deriving Inhabited, Repr

inductive TypeDef where
  | struct (name : String) (fields : Array StructField)
  | sum (name : String) (variants : Array SumVariant)
  /-- `base` is a NameRef on the wire. -/
  | alias (name : String) (base : String)
deriving Inhabited, Repr

structure ProgramPorts where
  inputs : Option (Array ProgramPort) := none
  outputs : Option (Array ProgramPort) := none
  typeDefs : Option (Array TypeDef) := none
deriving Inhabited, Repr

/-- One `type_params` entry value: `{type:'int', default?}`. The `type`
    discriminator is always the literal `int`, so only the default is
    carried. -/
structure TypeParamSpec where
  default? : Option JsonNumber := none
deriving Inhabited, Repr

-- ─────────────────────────────────────────────────────────────
-- BodyAssign — wires pinning a value to a port
-- ─────────────────────────────────────────────────────────────

inductive NextTargetKind where
  | reg | delay
deriving BEq, Inhabited, Repr

def NextTargetKind.wire : NextTargetKind → String
  | .reg => "reg" | .delay => "delay"

inductive BodyAssign where
  | output (name : String) (expr : ParsedExpr)
  | next (kind : NextTargetKind) (name : String) (expr : ParsedExpr)
deriving Inhabited, Repr

-- ─────────────────────────────────────────────────────────────
-- BodyDecl + Block + Program
-- ─────────────────────────────────────────────────────────────

mutual

inductive BodyDecl where
  /-- `type?` is a NameRef on the wire. -/
  | reg (name : String) (init : ParsedExpr) (type? : Option String)
  | delay (name : String) (update init : ParsedExpr) (type? : Option String)
  | param (name : String) (value? : Option JsonNumber)
  /-- `program` is a NameRef; `typeArgs`/`inputs` keep entry order
      (each entry's key is a NameRef on the wire). -/
  | inst (name program : String)
      (typeArgs : Option (Array (String × JsonNumber)))
      (inputs : Option (Array (String × ParsedExpr)))
  | prog (name : String) (program : Program)
deriving Inhabited, Repr

inductive Block where
  | mk (decls : Array BodyDecl) (assigns : Array BodyAssign)
deriving Inhabited, Repr

inductive Program where
  | mk (name : String)
      (typeParams : Option (Array (String × TypeParamSpec)))
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
  | .mk n _ _ _ _ => n

def Program.typeParams : Program → Option (Array (String × TypeParamSpec))
  | .mk _ tp _ _ _ => tp

def Program.ports : Program → Option ProgramPorts
  | .mk _ _ p _ _ => p

def Program.body : Program → Block
  | .mk _ _ _ b _ => b

def Program.breaksCycles : Program → Option Bool
  | .mk _ _ _ _ bc => bc

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

mutual

partial def expr : ParsedExpr → Json
  | .num n => Json.num n
  | .bool b => Json.bool b
  | .arr items => Json.arr (items.map expr)
  | .binary tag lhs rhs =>
    Json.mkObj [("op", jStr tag.wire), ("args", Json.arr #[expr lhs, expr rhs])]
  | .unary tag arg =>
    Json.mkObj [("op", jStr tag.wire), ("args", Json.arr #[expr arg])]
  | .call callee args =>
    Json.mkObj [("op", jStr "call"), ("callee", expr callee),
                ("args", Json.arr (args.map expr))]
  | .nameRef name => jNameRef name
  | .binding name => Json.mkObj [("op", jStr "binding"), ("name", jStr name)]
  | .nestedOut ref output =>
    Json.mkObj [("op", jStr "nestedOut"), ("ref", jNameRef ref),
                ("output", jNameRef output)]
  | .index arr idx =>
    Json.mkObj [("op", jStr "index"), ("args", Json.arr #[expr arr, expr idx])]
  | .letIn bind body =>
    Json.mkObj [("op", jStr "let"),
                ("bind", Json.mkObj (bind.toList.map fun (k, v) => (k, expr v))),
                ("in", expr body)]
  | .fold over init accVar elemVar body =>
    Json.mkObj [("op", jStr "fold"), ("over", expr over), ("init", expr init),
                ("acc_var", jStr accVar), ("elem_var", jStr elemVar),
                ("body", expr body)]
  | .scan over init accVar elemVar body =>
    Json.mkObj [("op", jStr "scan"), ("over", expr over), ("init", expr init),
                ("acc_var", jStr accVar), ("elem_var", jStr elemVar),
                ("body", expr body)]
  | .generate count var body =>
    Json.mkObj [("op", jStr "generate"), ("count", expr count),
                ("var", jStr var), ("body", expr body)]
  | .iterate count var init body =>
    Json.mkObj [("op", jStr "iterate"), ("count", expr count),
                ("var", jStr var), ("init", expr init), ("body", expr body)]
  | .chain count var init body =>
    Json.mkObj [("op", jStr "chain"), ("count", expr count),
                ("var", jStr var), ("init", expr init), ("body", expr body)]
  | .map2 over elemVar body =>
    Json.mkObj [("op", jStr "map2"), ("over", expr over),
                ("elem_var", jStr elemVar), ("body", expr body)]
  | .zipWith a b xVar yVar body =>
    Json.mkObj [("op", jStr "zipWith"), ("a", expr a), ("b", expr b),
                ("x_var", jStr xVar), ("y_var", jStr yVar), ("body", expr body)]
  | .tag variant payload =>
    Json.mkObj <|
      [("op", jStr "tag"), ("variant", jNameRef variant)]
      ++ optField "payload" (payload.map fun entries =>
           Json.arr (entries.map tagPayloadEntry))
  | .match_ scrutinee arms =>
    Json.mkObj [("op", jStr "match"), ("scrutinee", expr scrutinee),
                ("arms", Json.arr (arms.map matchArm))]

partial def tagPayloadEntry : TagPayloadEntry → Json
  | .mk field value =>
    Json.mkObj [("field", jNameRef field), ("value", expr value)]

partial def matchArm : MatchArm → Json
  | .mk variant binds body =>
    Json.mkObj [
      ("variant", jNameRef variant),
      ("binds", Json.arr (binds.map fun (field, bind) =>
        Json.mkObj [("field", jNameRef field), ("bind", jStr bind)])),
      ("body", expr body)]

end

def shapeDim : ShapeDim → Json
  | .lit n => Json.num n
  | .ref name => jNameRef name

def portTypeDecl : PortTypeDecl → Json
  | .scalar name => jNameRef name
  | .array element shape =>
    Json.mkObj [("kind", jStr "array"), ("element", jNameRef element),
                ("shape", Json.arr (shape.map shapeDim))]

def programPort : ProgramPort → Json
  | .bare name => jStr name
  | .spec s =>
    Json.mkObj <|
      [("name", jStr s.name)]
      ++ optField "type" (s.type?.map portTypeDecl)
      ++ optField "default" (s.default?.map expr)

def structField (f : StructField) : Json :=
  Json.mkObj [("name", jStr f.name), ("scalar_type", jStr f.scalarType.wire)]

def typeDef : TypeDef → Json
  | .struct name fields =>
    Json.mkObj [("kind", jStr "struct"), ("name", jStr name),
                ("fields", Json.arr (fields.map structField))]
  | .sum name variants =>
    Json.mkObj [("kind", jStr "sum"), ("name", jStr name),
                ("variants", Json.arr (variants.map fun v =>
                  Json.mkObj [("name", jStr v.name),
                              ("payload", Json.arr (v.payload.map structField))]))]
  | .alias name base =>
    Json.mkObj [("kind", jStr "alias"), ("name", jStr name), ("base", jNameRef base)]

def programPorts (p : ProgramPorts) : Json :=
  Json.mkObj <|
    optField "inputs" (p.inputs.map fun ps => Json.arr (ps.map programPort))
    ++ optField "outputs" (p.outputs.map fun ps => Json.arr (ps.map programPort))
    ++ optField "type_defs" (p.typeDefs.map fun ds => Json.arr (ds.map typeDef))

def typeParams (tps : Array (String × TypeParamSpec)) : Json :=
  Json.mkObj <| tps.toList.map fun (name, spec) =>
    (name, Json.mkObj <|
      [("type", jStr "int")] ++ optField "default" (spec.default?.map Json.num))

def bodyAssign : BodyAssign → Json
  | .output name e =>
    Json.mkObj [("op", jStr "outputAssign"), ("name", jStr name), ("expr", expr e)]
  | .next kind name e =>
    Json.mkObj [("op", jStr "nextUpdate"),
                ("target", Json.mkObj [("kind", jStr kind.wire), ("name", jStr name)]),
                ("expr", expr e)]

mutual

partial def bodyDecl : BodyDecl → Json
  | .reg name init type? =>
    Json.mkObj <|
      [("op", jStr "regDecl"), ("name", jStr name), ("init", expr init)]
      ++ optField "type" (type?.map jNameRef)
  | .delay name update init type? =>
    Json.mkObj <|
      [("op", jStr "delayDecl"), ("name", jStr name),
       ("update", expr update), ("init", expr init)]
      ++ optField "type" (type?.map jNameRef)
  | .param name value? =>
    Json.mkObj <|
      [("op", jStr "paramDecl"), ("name", jStr name)]
      ++ optField "value" (value?.map Json.num)
  | .inst name progName typeArgs inputs =>
    Json.mkObj <|
      [("op", jStr "instanceDecl"), ("name", jStr name),
       ("program", jNameRef progName)]
      ++ optField "type_args" (typeArgs.map fun entries =>
           Json.arr (entries.map fun (param, value) =>
             Json.mkObj [("param", jNameRef param), ("value", Json.num value)]))
      ++ optField "inputs" (inputs.map fun entries =>
           Json.arr (entries.map fun (port, value) =>
             Json.mkObj [("port", jNameRef port), ("value", expr value)]))
  | .prog name p =>
    Json.mkObj [("op", jStr "programDecl"), ("name", jStr name),
                ("program", program p)]

partial def block (b : Block) : Json :=
  Json.mkObj [("op", jStr "block"),
              ("decls", Json.arr (b.decls.map bodyDecl)),
              ("assigns", Json.arr (b.assigns.map bodyAssign))]

partial def program (p : Program) : Json :=
  Json.mkObj <|
    [("op", jStr "program"), ("name", jStr p.name), ("body", block p.body)]
    ++ optField "type_params" (p.typeParams.map typeParams)
    ++ optField "ports" (p.ports.map programPorts)
    ++ optField "breaks_cycles" (p.breaksCycles.map Json.bool)

end

end Encode

def Program.toJson (p : Program) : Json := Encode.program p

-- ─────────────────────────────────────────────────────────────
-- Decode — JsonV → typed AST (strict)
-- ─────────────────────────────────────────────────────────────

namespace Decode

/-- Decoding errors carry a JSONPath-ish location. -/
private def err {α} (path msg : String) : Except String α :=
  .error s!"ParsedProgram decode: {path}: {msg}"

/-- Strictness: every consumed object must contain only known keys, so
    a field this codec doesn't model can never be dropped silently. -/
private def expectKeys (path : String) (j : JsonV) (allowed : List String) :
    Except String Unit := do
  for k in j.keys do
    if !allowed.contains k then
      err path s!"unexpected key '{k}'"

private def reqField (path : String) (j : JsonV) (k : String) : Except String JsonV :=
  match j.getField? k with
  | some v => pure v
  | none => err path s!"missing field '{k}'"

private def reqStr (path : String) (j : JsonV) (k : String) : Except String String := do
  match ← reqField path j k with
  | .str s => pure s
  | _ => err path s!"field '{k}' must be a string"

private def optStr (path : String) (j : JsonV) (k : String) :
    Except String (Option String) :=
  match j.getField? k with
  | none => pure none
  | some (.str s) => pure (some s)
  | some _ => err path s!"field '{k}' must be a string"

private def optNum (path : String) (j : JsonV) (k : String) :
    Except String (Option JsonNumber) :=
  match j.getField? k with
  | none => pure none
  | some (.num n) => pure (some n)
  | some _ => err path s!"field '{k}' must be a number"

private def reqArr (path : String) (j : JsonV) (k : String) :
    Except String (Array JsonV) := do
  match ← reqField path j k with
  | .arr items => pure items
  | _ => err path s!"field '{k}' must be an array"

/-- A `{op:'nameRef', name}` node. -/
def nameRefNode (path : String) (j : JsonV) : Except String String := do
  let .obj _ := j | err path "expected a nameRef node"
  if j.opOf? != some "nameRef" then
    err path s!"expected op 'nameRef', got '{(j.opOf?).getD "<none>"}'"
  expectKeys path j ["op", "name"]
  reqStr path j "name"

mutual

partial def expr (path : String) (j : JsonV) : Except String ParsedExpr := do
  match j with
  | .num n => pure (.num n)
  | .bool b => pure (.bool b)
  | .arr items => do
    let mut out : Array ParsedExpr := #[]
    for h : i in [0:items.size] do
      out := out.push (← expr s!"{path}[{i}]" items[i])
    pure (.arr out)
  | .str _ => err path "string is not a ParsedExpr"
  | .null => err path "null is not a ParsedExpr"
  | .obj _ =>
    let some op := j.opOf?
      | err path "expression object missing string 'op'"
    if let some tag := BinaryOpTag.ofWire? op then
      expectKeys path j ["op", "args"]
      let args ← reqArr path j "args"
      let #[l, r] := args
        | err path s!"'{op}' requires exactly 2 args, got {args.size}"
      pure (.binary tag (← expr s!"{path}.args[0]" l) (← expr s!"{path}.args[1]" r))
    else if let some tag := UnaryOpTag.ofWire? op then
      expectKeys path j ["op", "args"]
      let args ← reqArr path j "args"
      let #[x] := args
        | err path s!"'{op}' requires exactly 1 arg, got {args.size}"
      pure (.unary tag (← expr s!"{path}.args[0]" x))
    else match op with
    | "call" => do
      expectKeys path j ["op", "callee", "args"]
      let callee ← expr s!"{path}.callee" (← reqField path j "callee")
      let args ← reqArr path j "args"
      let mut out : Array ParsedExpr := #[]
      for h : i in [0:args.size] do
        out := out.push (← expr s!"{path}.args[{i}]" args[i])
      pure (.call callee out)
    | "nameRef" => do
      pure (.nameRef (← nameRefNode path j))
    | "binding" => do
      expectKeys path j ["op", "name"]
      pure (.binding (← reqStr path j "name"))
    | "nestedOut" => do
      expectKeys path j ["op", "ref", "output"]
      let ref ← nameRefNode s!"{path}.ref" (← reqField path j "ref")
      let output ← nameRefNode s!"{path}.output" (← reqField path j "output")
      pure (.nestedOut ref output)
    | "index" => do
      expectKeys path j ["op", "args"]
      let args ← reqArr path j "args"
      let #[a, i] := args
        | err path s!"'index' requires exactly 2 args, got {args.size}"
      pure (.index (← expr s!"{path}.args[0]" a) (← expr s!"{path}.args[1]" i))
    | "let" => do
      expectKeys path j ["op", "bind", "in"]
      let .obj bindFields ← reqField path j "bind"
        | err path "'let' bind must be an object"
      let mut bind : Array (String × ParsedExpr) := #[]
      for (k, v) in bindFields do
        bind := bind.push (k, ← expr s!"{path}.bind.{k}" v)
      pure (.letIn bind (← expr s!"{path}.in" (← reqField path j "in")))
    | "fold" => do
      expectKeys path j ["op", "over", "init", "acc_var", "elem_var", "body"]
      pure (.fold
        (← expr s!"{path}.over" (← reqField path j "over"))
        (← expr s!"{path}.init" (← reqField path j "init"))
        (← reqStr path j "acc_var") (← reqStr path j "elem_var")
        (← expr s!"{path}.body" (← reqField path j "body")))
    | "scan" => do
      expectKeys path j ["op", "over", "init", "acc_var", "elem_var", "body"]
      pure (.scan
        (← expr s!"{path}.over" (← reqField path j "over"))
        (← expr s!"{path}.init" (← reqField path j "init"))
        (← reqStr path j "acc_var") (← reqStr path j "elem_var")
        (← expr s!"{path}.body" (← reqField path j "body")))
    | "generate" => do
      expectKeys path j ["op", "count", "var", "body"]
      pure (.generate
        (← expr s!"{path}.count" (← reqField path j "count"))
        (← reqStr path j "var")
        (← expr s!"{path}.body" (← reqField path j "body")))
    | "iterate" => do
      expectKeys path j ["op", "count", "var", "init", "body"]
      pure (.iterate
        (← expr s!"{path}.count" (← reqField path j "count"))
        (← reqStr path j "var")
        (← expr s!"{path}.init" (← reqField path j "init"))
        (← expr s!"{path}.body" (← reqField path j "body")))
    | "chain" => do
      expectKeys path j ["op", "count", "var", "init", "body"]
      pure (.chain
        (← expr s!"{path}.count" (← reqField path j "count"))
        (← reqStr path j "var")
        (← expr s!"{path}.init" (← reqField path j "init"))
        (← expr s!"{path}.body" (← reqField path j "body")))
    | "map2" => do
      expectKeys path j ["op", "over", "elem_var", "body"]
      pure (.map2
        (← expr s!"{path}.over" (← reqField path j "over"))
        (← reqStr path j "elem_var")
        (← expr s!"{path}.body" (← reqField path j "body")))
    | "zipWith" => do
      expectKeys path j ["op", "a", "b", "x_var", "y_var", "body"]
      pure (.zipWith
        (← expr s!"{path}.a" (← reqField path j "a"))
        (← expr s!"{path}.b" (← reqField path j "b"))
        (← reqStr path j "x_var") (← reqStr path j "y_var")
        (← expr s!"{path}.body" (← reqField path j "body")))
    | "tag" => do
      expectKeys path j ["op", "variant", "payload"]
      let variant ← nameRefNode s!"{path}.variant" (← reqField path j "variant")
      let payload ← match j.getField? "payload" with
        | none => pure none
        | some (.arr entries) => do
          let mut out : Array TagPayloadEntry := #[]
          for h : i in [0:entries.size] do
            out := out.push (← tagPayloadEntry s!"{path}.payload[{i}]" entries[i])
          pure (some out)
        | some _ => err path "'tag' payload must be an array"
      pure (.tag variant payload)
    | "match" => do
      expectKeys path j ["op", "scrutinee", "arms"]
      let scrutinee ← expr s!"{path}.scrutinee" (← reqField path j "scrutinee")
      let armsJson ← reqArr path j "arms"
      let mut arms : Array MatchArm := #[]
      for h : i in [0:armsJson.size] do
        arms := arms.push (← matchArm s!"{path}.arms[{i}]" armsJson[i])
      pure (.match_ scrutinee arms)
    | other => err path s!"unknown expression op '{other}'"

partial def tagPayloadEntry (path : String) (j : JsonV) :
    Except String TagPayloadEntry := do
  expectKeys path j ["field", "value"]
  let field ← nameRefNode s!"{path}.field" (← reqField path j "field")
  let value ← expr s!"{path}.value" (← reqField path j "value")
  pure (.mk field value)

partial def matchArm (path : String) (j : JsonV) : Except String MatchArm := do
  expectKeys path j ["variant", "binds", "body"]
  let variant ← nameRefNode s!"{path}.variant" (← reqField path j "variant")
  let bindsJson ← reqArr path j "binds"
  let mut binds : Array (String × String) := #[]
  for h : i in [0:bindsJson.size] do
    let b := bindsJson[i]
    let bPath := s!"{path}.binds[{i}]"
    expectKeys bPath b ["field", "bind"]
    let field ← nameRefNode s!"{bPath}.field" (← reqField bPath b "field")
    binds := binds.push (field, ← reqStr bPath b "bind")
  let body ← expr s!"{path}.body" (← reqField path j "body")
  pure (.mk variant binds body)

end

def shapeDim (path : String) (j : JsonV) : Except String ShapeDim := do
  match j with
  | .num n => pure (.lit n)
  | .obj _ => pure (.ref (← nameRefNode path j))
  | _ => err path "shape dim must be a number or nameRef"

def portTypeDecl (path : String) (j : JsonV) : Except String PortTypeDecl := do
  let .obj _ := j | err path "port type must be an object"
  match j.getStr? "kind" with
  | some "array" => do
    expectKeys path j ["kind", "element", "shape"]
    let element ← nameRefNode s!"{path}.element" (← reqField path j "element")
    let shapeJson ← reqArr path j "shape"
    let mut shape : Array ShapeDim := #[]
    for h : i in [0:shapeJson.size] do
      shape := shape.push (← shapeDim s!"{path}.shape[{i}]" shapeJson[i])
    pure (.array element shape)
  | some other => err path s!"unknown port type kind '{other}'"
  | none => pure (.scalar (← nameRefNode path j))

def programPort (path : String) (j : JsonV) : Except String ProgramPort := do
  match j with
  | .str name => pure (.bare name)
  | .obj _ => do
    expectKeys path j ["name", "type", "default"]
    let name ← reqStr path j "name"
    let type? ← match j.getField? "type" with
      | none => pure none
      | some t => pure (some (← portTypeDecl s!"{path}.type" t))
    let default? ← match j.getField? "default" with
      | none => pure none
      | some d => pure (some (← expr s!"{path}.default" d))
    pure (.spec { name, type?, default? })
  | _ => err path "port must be a string or a spec object"

def structField (path : String) (j : JsonV) : Except String StructField := do
  expectKeys path j ["name", "scalar_type"]
  let name ← reqStr path j "name"
  let st ← reqStr path j "scalar_type"
  let some scalarType := ScalarKind.ofWire? st
    | err path s!"unknown scalar_type '{st}'"
  pure { name, scalarType }

def typeDef (path : String) (j : JsonV) : Except String TypeDef := do
  let .obj _ := j | err path "type def must be an object"
  match j.getStr? "kind" with
  | some "struct" => do
    expectKeys path j ["kind", "name", "fields"]
    let name ← reqStr path j "name"
    let fieldsJson ← reqArr path j "fields"
    let mut fields : Array StructField := #[]
    for h : i in [0:fieldsJson.size] do
      fields := fields.push (← structField s!"{path}.fields[{i}]" fieldsJson[i])
    pure (.struct name fields)
  | some "sum" => do
    expectKeys path j ["kind", "name", "variants"]
    let name ← reqStr path j "name"
    let variantsJson ← reqArr path j "variants"
    let mut variants : Array SumVariant := #[]
    for h : i in [0:variantsJson.size] do
      let v := variantsJson[i]
      let vPath := s!"{path}.variants[{i}]"
      expectKeys vPath v ["name", "payload"]
      let vName ← reqStr vPath v "name"
      let payloadJson ← reqArr vPath v "payload"
      let mut payload : Array StructField := #[]
      for h : k in [0:payloadJson.size] do
        payload := payload.push (← structField s!"{vPath}.payload[{k}]" payloadJson[k])
      variants := variants.push { name := vName, payload }
    pure (.sum name variants)
  | some "alias" => do
    expectKeys path j ["kind", "name", "base"]
    let name ← reqStr path j "name"
    let base ← nameRefNode s!"{path}.base" (← reqField path j "base")
    pure (.alias name base)
  | some other => err path s!"unknown type def kind '{other}'"
  | none => err path "type def missing 'kind'"

def programPorts (path : String) (j : JsonV) : Except String ProgramPorts := do
  expectKeys path j ["inputs", "outputs", "type_defs"]
  let decodePorts (k : String) : Except String (Option (Array ProgramPort)) := do
    match j.getField? k with
    | none => pure none
    | some (.arr items) => do
      let mut out : Array ProgramPort := #[]
      for h : i in [0:items.size] do
        out := out.push (← programPort s!"{path}.{k}[{i}]" items[i])
      pure (some out)
    | some _ => err path s!"'{k}' must be an array"
  let inputs ← decodePorts "inputs"
  let outputs ← decodePorts "outputs"
  let typeDefs ← match j.getField? "type_defs" with
    | none => pure none
    | some (.arr items) => do
      let mut out : Array TypeDef := #[]
      for h : i in [0:items.size] do
        out := out.push (← typeDef s!"{path}.type_defs[{i}]" items[i])
      pure (some out)
    | some _ => err path "'type_defs' must be an array"
  pure { inputs, outputs, typeDefs }

def typeParams (path : String) (j : JsonV) :
    Except String (Array (String × TypeParamSpec)) := do
  let .obj fields := j | err path "'type_params' must be an object"
  let mut out : Array (String × TypeParamSpec) := #[]
  for (name, spec) in fields do
    let sPath := s!"{path}.{name}"
    expectKeys sPath spec ["type", "default"]
    if spec.getStr? "type" != some "int" then
      err sPath "type param 'type' must be the literal 'int'"
    out := out.push (name, { default? := ← optNum sPath spec "default" })
  pure out

def bodyAssign (path : String) (j : JsonV) : Except String BodyAssign := do
  let .obj _ := j | err path "body assign must be an object"
  match j.opOf? with
  | some "outputAssign" => do
    expectKeys path j ["op", "name", "expr"]
    pure (.output (← reqStr path j "name")
      (← expr s!"{path}.expr" (← reqField path j "expr")))
  | some "nextUpdate" => do
    expectKeys path j ["op", "target", "expr"]
    let target ← reqField path j "target"
    expectKeys s!"{path}.target" target ["kind", "name"]
    let kind ← match target.getStr? "kind" with
      | some "reg" => pure NextTargetKind.reg
      | some "delay" => pure NextTargetKind.delay
      | _ => err s!"{path}.target" "kind must be 'reg' or 'delay'"
    pure (.next kind (← reqStr s!"{path}.target" target "name")
      (← expr s!"{path}.expr" (← reqField path j "expr")))
  | some other => err path s!"unknown body assign op '{other}'"
  | none => err path "body assign missing string 'op'"

mutual

partial def bodyDecl (path : String) (j : JsonV) : Except String BodyDecl := do
  let .obj _ := j | err path "body decl must be an object"
  match j.opOf? with
  | some "regDecl" => do
    expectKeys path j ["op", "name", "init", "type"]
    let type? ← match j.getField? "type" with
      | none => pure none
      | some t => pure (some (← nameRefNode s!"{path}.type" t))
    pure (.reg (← reqStr path j "name")
      (← expr s!"{path}.init" (← reqField path j "init")) type?)
  | some "delayDecl" => do
    expectKeys path j ["op", "name", "update", "init", "type"]
    let type? ← match j.getField? "type" with
      | none => pure none
      | some t => pure (some (← nameRefNode s!"{path}.type" t))
    pure (.delay (← reqStr path j "name")
      (← expr s!"{path}.update" (← reqField path j "update"))
      (← expr s!"{path}.init" (← reqField path j "init")) type?)
  | some "paramDecl" => do
    expectKeys path j ["op", "name", "value"]
    pure (.param (← reqStr path j "name") (← optNum path j "value"))
  | some "instanceDecl" => do
    expectKeys path j ["op", "name", "program", "type_args", "inputs"]
    let name ← reqStr path j "name"
    let progName ← nameRefNode s!"{path}.program" (← reqField path j "program")
    let typeArgs ← match j.getField? "type_args" with
      | none => pure none
      | some (.arr entries) => do
        let mut out : Array (String × JsonNumber) := #[]
        for h : i in [0:entries.size] do
          let e := entries[i]
          let ePath := s!"{path}.type_args[{i}]"
          expectKeys ePath e ["param", "value"]
          let param ← nameRefNode s!"{ePath}.param" (← reqField ePath e "param")
          let value ← match e.getField? "value" with
            | some (.num n) => pure n
            | _ => err ePath "type arg 'value' must be a number"
          out := out.push (param, value)
        pure (some out)
      | some _ => err path "'type_args' must be an array"
    let inputs ← match j.getField? "inputs" with
      | none => pure none
      | some (.arr entries) => do
        let mut out : Array (String × ParsedExpr) := #[]
        for h : i in [0:entries.size] do
          let e := entries[i]
          let ePath := s!"{path}.inputs[{i}]"
          expectKeys ePath e ["port", "value"]
          let port ← nameRefNode s!"{ePath}.port" (← reqField ePath e "port")
          let value ← expr s!"{ePath}.value" (← reqField ePath e "value")
          out := out.push (port, value)
        pure (some out)
      | some _ => err path "'inputs' must be an array"
    pure (.inst name progName typeArgs inputs)
  | some "programDecl" => do
    expectKeys path j ["op", "name", "program"]
    pure (.prog (← reqStr path j "name")
      (← program s!"{path}.program" (← reqField path j "program")))
  | some other => err path s!"unknown body decl op '{other}'"
  | none => err path "body decl missing string 'op'"

partial def block (path : String) (j : JsonV) : Except String Block := do
  let .obj _ := j | err path "block must be an object"
  if j.opOf? != some "block" then
    err path "block must have op 'block'"
  expectKeys path j ["op", "decls", "assigns"]
  let declsJson ← reqArr path j "decls"
  let mut decls : Array BodyDecl := #[]
  for h : i in [0:declsJson.size] do
    decls := decls.push (← bodyDecl s!"{path}.decls[{i}]" declsJson[i])
  let assignsJson ← reqArr path j "assigns"
  let mut assigns : Array BodyAssign := #[]
  for h : i in [0:assignsJson.size] do
    assigns := assigns.push (← bodyAssign s!"{path}.assigns[{i}]" assignsJson[i])
  pure (.mk decls assigns)

partial def program (path : String) (j : JsonV) : Except String Program := do
  let .obj _ := j | err path "program must be an object"
  if j.opOf? != some "program" then
    err path "program must have op 'program'"
  expectKeys path j ["op", "name", "body", "type_params", "ports", "breaks_cycles"]
  let name ← reqStr path j "name"
  let tps ← match j.getField? "type_params" with
    | none => pure none
    | some t => pure (some (← typeParams s!"{path}.type_params" t))
  let ports ← match j.getField? "ports" with
    | none => pure none
    | some p => pure (some (← programPorts s!"{path}.ports" p))
  let body ← block s!"{path}.body" (← reqField path j "body")
  let breaksCycles ← match j.getField? "breaks_cycles" with
    | none => pure none
    | some (.bool b) => pure (some b)
    | some _ => err path "'breaks_cycles' must be a boolean"
  pure (.mk name tps ports body breaksCycles)

end

end Decode

/-- Decode a serialized ParsedProgram (strict). -/
def decodeProgram (j : JsonV) : Except String Program :=
  Decode.program "$" j

end Tropical.Parse
