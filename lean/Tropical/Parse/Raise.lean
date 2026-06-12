import Tropical.Parse.Nodes

/-!
# raise — legacy `tropical_program_2` JSON → typed ParsedProgram

Line-faithful port of three TS layers, in the order the TS entry path
composes them:

1. **`normalizeProgramFile`** (compiler/session.ts) — schema-tag check
   (exact TS message) and the split of the deprecated top-level
   `params` / `audio_outputs` metadata from the program node.
2. **`parseProgramV2`** (compiler/schema.ts) — the Zod validation,
   ported structurally: same accepted shapes, same *strip* semantics
   (z.object drops unknown keys — notably `bounds` and output-port
   `default`s never reach raise), shallow `ExprNodeSchema` checking
   (object exprs are only checked for a string `op`; children of
   object exprs are not recursed into — only array elements are).
   Error *text* is best-effort Zod mimicry; no recorded MCP script
   exercises a validation failure, so only the schema-tag message is
   contractual.
3. **`raiseProgram`** (compiler/parse/raise.ts) — the op mappings,
   desugarings (`{zeros: N}`, `{typeParam: n}`), and error strings,
   ported exactly, followed by `lowerBoundsToClamps`
   (compiler/parse/lower_bounds.ts).

Strictness divergences (all unreachable on inputs the TS path handles
*successfully*): where TS casts a field and would silently emit
`undefined`-valued junk (e.g. a missing `name` on a decl, a non-string
binder), this port raises a decode error instead — the typed AST has
nowhere to put junk. `String(...)` coercion sites that TS makes total
(`String(node.output)` in nestedOut, `String(obj.op)` in unknown-op
messages) stay total here via `JsonV.jsString`.

`lowerBoundsToClamps` notes: the parsed AST carries no `bounds` field
(the Zod layer strips explicit bounds before raise, and raise never
copies one), so only the built-in port-type alias bounds (`signal`,
`bipolar`, `unipolar`, `phase`, `freq`) apply here. The TS
`alreadyWrapped` also recognizes elaborator-shaped direct ops
(`{op:'clamp'}`); those are unrepresentable in a ParsedProgram, so
only the parser-level `call(nameRef('clamp'|'select'), ...)` shapes
are checked.
-/

namespace Tropical.Parse.Raise

open Lean (Json JsonNumber)
open Tropical.Parse

-- ─────────────────────────────────────────────────────────────
-- Top-level metadata (deprecated `params` / `audio_outputs`)
-- ─────────────────────────────────────────────────────────────

/-- One stripped top-level `params` entry. -/
structure LegacyParam where
  name : String
  value : Option JsonNumber := none
  timeConst : Option JsonNumber := none
  /-- `'param' | 'trigger'`. -/
  ptype : Option String := none

def LegacyParam.toJson (p : LegacyParam) : Json :=
  Json.mkObj <|
    [("name", Json.str p.name)]
    ++ (match p.value with | some v => [("value", Json.num v)] | none => [])
    ++ (match p.timeConst with | some v => [("time_const", Json.num v)] | none => [])
    ++ (match p.ptype with | some t => [("type", Json.str t)] | none => [])

/-- One stripped top-level `audio_outputs` entry. The `expr` arm keeps
    the (shallowly validated) wire expression verbatim, mirroring how
    the TS layer passes it through untouched. -/
inductive AudioOutput where
  | ref (inst : String) (output : JsonV)
  | expr (e : JsonV)

def AudioOutput.toJson : AudioOutput → Json
  | .ref inst output =>
    Json.mkObj [("instance", Json.str inst), ("output", output.toJson)]
  | .expr e => Json.mkObj [("expr", e.toJson)]

structure TopLevel where
  params : Option (Array LegacyParam) := none
  audioOutputs : Option (Array AudioOutput) := none

-- ─────────────────────────────────────────────────────────────
-- Schema validation (port of compiler/schema.ts parseProgramV2)
-- ─────────────────────────────────────────────────────────────

namespace Schema

/-- `Invalid program (v2): at 'path': message` (formatZodError shape;
    fail-fast on the first issue where Zod would collect several). -/
private def zerr {α} (path msg : String) : Except String α :=
  .error <|
    "Invalid program (v2): " ++
    (if path.isEmpty then msg else s!"at '{path}': {msg}")

/-- Zod's `received` vocabulary. -/
private def received : JsonV → String
  | .null => "null"
  | .bool _ => "boolean"
  | .num _ => "number"
  | .str _ => "string"
  | .arr _ => "array"
  | .obj _ => "object"

private def sub (path k : String) : String :=
  if path.isEmpty then k else s!"{path}.{k}"

private def reqStr (path : String) (j : JsonV) (k : String) : Except String String :=
  match j.getField? k with
  | none => zerr (sub path k) "Required"
  | some (.str s) => pure s
  | some other => zerr (sub path k) s!"Expected string, received {received other}"

private def optBool (path : String) (j : JsonV) (k : String) :
    Except String (Option Bool) :=
  match j.getField? k with
  | none => pure none
  | some (.bool b) => pure (some b)
  | some other => zerr (sub path k) s!"Expected boolean, received {received other}"

private def isInt (n : JsonNumber) : Bool :=
  let f := n.toFloat
  f.isFinite && f == f.floor

/-- Shallow `ExprNodeSchema`: number | boolean | array (recursed) |
    object with a string `op` (children not recursed). -/
partial def exprNode (path : String) (v : JsonV) : Except String Unit := do
  match v with
  | .num _ | .bool _ => pure ()
  | .arr items =>
    for h : i in [0:items.size] do
      exprNode (sub path (toString i)) items[i]
  | .obj _ =>
    match v.getStr? "op" with
    | some _ => pure ()
    | none => zerr path "Invalid input"
  | _ => zerr path "Invalid input"

/-- `PortTypeDeclSchema`: bare string (no brackets) or
    `{kind:'array', element, shape}`. Returns the stripped value. -/
def portType (path : String) (v : JsonV) : Except String JsonV := do
  match v with
  | .str s =>
    if s.toList.contains '[' then
      zerr path "Bracketed array types like \"float[4]\" are not supported. Use {kind:\"array\", element:\"float\", shape:[4]}."
    else
      pure v
  | .obj _ => do
    if v.getStr? "kind" != some "array" then
      zerr path "Invalid input"
    let element ← reqStr path v "element"
    let some (JsonV.arr dims) := v.getField? "shape"
      | zerr (sub path "shape") "Required"
    if dims.isEmpty then
      zerr (sub path "shape") "Array must contain at least 1 element(s)"
    let mut outDims : Array JsonV := #[]
    for h : i in [0:dims.size] do
      match dims[i] with
      | .num n =>
        if !isInt n then
          zerr (sub (sub path "shape") (toString i)) "Expected integer, received float"
        if n.toFloat < 0 then
          zerr (sub (sub path "shape") (toString i)) "Number must be greater than or equal to 0"
        outDims := outDims.push dims[i]
      | dim@(.obj _) => do
        if dim.getStr? "op" != some "typeParam" then
          zerr (sub (sub path "shape") (toString i)) "Invalid input"
        let name ← reqStr (sub (sub path "shape") (toString i)) dim "name"
        outDims := outDims.push (.obj #[("op", .str "typeParam"), ("name", .str name)])
      | _ => zerr (sub (sub path "shape") (toString i)) "Invalid input"
    pure (.obj #[("kind", .str "array"), ("element", .str element),
                 ("shape", .arr outDims)])
  | _ => zerr path "Invalid input"

/-- `ProgramInputSchema` / `ProgramOutputSchema`: bare string or spec
    object. `allowDefault` is true for inputs only — output specs are
    stripped to `{name, type?}` (the Zod schema has no `default` key
    on outputs, so a present one is silently dropped, exactly as Zod
    strips it). -/
def port (path : String) (v : JsonV) (allowDefault : Bool) : Except String JsonV := do
  match v with
  | .str _ => pure v
  | .obj _ => do
    let name ← reqStr path v "name"
    let mut fields : Array (String × JsonV) := #[("name", .str name)]
    match v.getField? "type" with
    | none => pure ()
    | some t => fields := fields.push ("type", ← portType (sub path "type") t)
    if allowDefault then
      match v.getField? "default" with
      | none => pure ()
      | some d => do
        exprNode (sub path "default") d
        fields := fields.push ("default", d)
    pure (.obj fields)
  | _ => zerr path "Invalid input"

def structFields (path : String) (v : JsonV) (key : String) : Except String JsonV := do
  let some (JsonV.arr items) := v.getField? key
    | zerr (sub path key) "Required"
  let mut out : Array JsonV := #[]
  for h : i in [0:items.size] do
    let f := items[i]
    let fPath := sub (sub path key) (toString i)
    let name ← reqStr fPath f "name"
    let st ← reqStr fPath f "scalar_type"
    if st != "float" && st != "int" && st != "bool" then
      zerr (sub fPath "scalar_type") "Invalid input"
    out := out.push (.obj #[("name", .str name), ("scalar_type", .str st)])
  pure (.arr out)

def typeDef (path : String) (v : JsonV) : Except String JsonV := do
  let .obj _ := v | zerr path "Invalid input"
  match v.getStr? "kind" with
  | some "struct" => do
    let name ← reqStr path v "name"
    let fields ← structFields path v "fields"
    pure (.obj #[("kind", .str "struct"), ("name", .str name), ("fields", fields)])
  | some "sum" => do
    let name ← reqStr path v "name"
    let some (JsonV.arr items) := v.getField? "variants"
      | zerr (sub path "variants") "Required"
    let mut variants : Array JsonV := #[]
    for h : i in [0:items.size] do
      let vt := items[i]
      let vPath := sub (sub path "variants") (toString i)
      let vName ← reqStr vPath vt "name"
      let payload ← structFields vPath vt "payload"
      variants := variants.push (.obj #[("name", .str vName), ("payload", payload)])
    pure (.obj #[("kind", .str "sum"), ("name", .str name), ("variants", .arr variants)])
  | some "alias" => do
    let name ← reqStr path v "name"
    let base ← reqStr path v "base"
    pure (.obj #[("kind", .str "alias"), ("name", .str name), ("base", .str base)])
  | _ => zerr path "Invalid input"

def ports (path : String) (v : JsonV) : Except String JsonV := do
  let .obj _ := v | zerr path s!"Expected object, received {received v}"
  let mut fields : Array (String × JsonV) := #[]
  let portArray (k : String) (allowDefault : Bool) :
      Except String (Option JsonV) := do
    match v.getField? k with
    | none => pure none
    | some (.arr items) => do
      let mut out : Array JsonV := #[]
      for h : i in [0:items.size] do
        out := out.push (← port (sub (sub path k) (toString i)) items[i] allowDefault)
      pure (some (.arr out))
    | some other => zerr (sub path k) s!"Expected array, received {received other}"
  match ← portArray "inputs" true with
  | some a => fields := fields.push ("inputs", a)
  | none => pure ()
  match ← portArray "outputs" false with
  | some a => fields := fields.push ("outputs", a)
  | none => pure ()
  match v.getField? "type_defs" with
  | none => pure ()
  | some (.arr items) => do
    let mut out : Array JsonV := #[]
    for h : i in [0:items.size] do
      out := out.push (← typeDef (sub (sub path "type_defs") (toString i)) items[i])
    fields := fields.push ("type_defs", .arr out)
  | some other =>
    zerr (sub path "type_defs") s!"Expected array, received {received other}"
  pure (.obj fields)

def typeParams (path : String) (v : JsonV) : Except String JsonV := do
  let .obj entries := v | zerr path s!"Expected object, received {received v}"
  let mut out : Array (String × JsonV) := #[]
  for (name, spec) in entries do
    let sPath := sub path name
    let .obj _ := spec | zerr sPath s!"Expected object, received {received spec}"
    if spec.getStr? "type" != some "int" then
      zerr (sub sPath "type") "Invalid literal value, expected \"int\""
    let mut fields : Array (String × JsonV) := #[("type", .str "int")]
    match spec.getField? "default" with
    | none => pure ()
    | some (.num n) =>
      if !isInt n then
        zerr (sub sPath "default") "Expected integer, received float"
      fields := fields.push ("default", .num n)
    | some other =>
      zerr (sub sPath "default") s!"Expected number, received {received other}"
    out := out.push (name, .obj fields)
  pure (.obj out)

def block (path : String) (v : JsonV) : Except String JsonV := do
  let .obj _ := v | zerr path s!"Expected object, received {received v}"
  if v.getStr? "op" != some "block" then
    zerr (sub path "op") "Invalid literal value, expected \"block\""
  let mut fields : Array (String × JsonV) := #[("op", .str "block")]
  let exprArray (k : String) : Except String (Option JsonV) := do
    match v.getField? k with
    | none => pure none
    | some (.arr items) => do
      for h : i in [0:items.size] do
        exprNode (sub (sub path k) (toString i)) items[i]
      pure (some (.arr items))
    | some other => zerr (sub path k) s!"Expected array, received {received other}"
  match ← exprArray "decls" with
  | some a => fields := fields.push ("decls", a)
  | none => pure ()
  match ← exprArray "assigns" with
  | some a => fields := fields.push ("assigns", a)
  | none => pure ()
  match v.getField? "value" with
  | none => pure ()
  | some .null => fields := fields.push ("value", .null)
  | some e => do
    exprNode (sub path "value") e
    fields := fields.push ("value", e)
  pure (.obj fields)

def topParams (v : JsonV) : Except String (Array LegacyParam) := do
  let .arr items := v | zerr "params" s!"Expected array, received {received v}"
  let mut out : Array LegacyParam := #[]
  for h : i in [0:items.size] do
    let p := items[i]
    let pPath := sub "params" (toString i)
    let name ← reqStr pPath p "name"
    let numOpt (k : String) : Except String (Option JsonNumber) :=
      match p.getField? k with
      | none => pure none
      | some (.num n) => pure (some n)
      | some other => zerr (sub pPath k) s!"Expected number, received {received other}"
    let value ← numOpt "value"
    let timeConst ← numOpt "time_const"
    let ptype ← match p.getField? "type" with
      | none => pure none
      | some (.str "param") => pure (some "param")
      | some (.str "trigger") => pure (some "trigger")
      | some other =>
        zerr (sub pPath "type")
          s!"Invalid enum value. Expected 'param' | 'trigger', received {other.compress}"
    out := out.push { name, value, timeConst, ptype }
  pure out

def topAudioOutputs (v : JsonV) : Except String (Array AudioOutput) := do
  let .arr items := v | zerr "audio_outputs" s!"Expected array, received {received v}"
  let mut out : Array AudioOutput := #[]
  for h : i in [0:items.size] do
    let entry := items[i]
    let ePath := sub "audio_outputs" (toString i)
    -- Zod union: try {instance, output} first, then {expr}.
    let refArm : Except String AudioOutput := do
      let inst ← reqStr ePath entry "instance"
      match entry.getField? "output" with
      | some o@(.str _) | some o@(.num _) => pure (.ref inst o)
      | _ => zerr ePath "Invalid input"
    let exprArm : Except String AudioOutput := do
      match entry.getField? "expr" with
      | some e => do
        exprNode (sub ePath "expr") e
        pure (.expr e)
      | none => zerr ePath "Invalid input"
    match refArm with
    | .ok a => out := out.push a
    | .error _ =>
      match exprArm with
      | .ok a => out := out.push a
      | .error _ => zerr ePath "Invalid input"
  pure out

end Schema

/-- Port of `normalizeProgramFile` + `parseProgramV2`: schema-tag check
    (exact TS message), structural validation with Zod strip semantics,
    and the program-node / top-level-metadata split. The returned node
    carries `op:'program'` plus the stripped program fields. -/
def normalizeProgramFile (raw : JsonV) : Except String (JsonV × TopLevel) := do
  if raw.getField? "schema" != some (.str "tropical_program_2") then
    let shown := match raw.getField? "schema" with
      | some v => v.jsString
      | none => "undefined"
    throw s!"Unknown schema '{shown}'. Expected 'tropical_program_2'."
  let name ← Schema.reqStr "" raw "name"
  if name.isEmpty then
    Schema.zerr "name" "String must contain at least 1 character(s)"
  let mut fields : Array (String × JsonV) :=
    #[("op", .str "program"), ("name", .str name)]
  match raw.getField? "type_params" with
  | none => pure ()
  | some t => fields := fields.push ("type_params", ← Schema.typeParams "type_params" t)
  match raw.getField? "sample_rate" with
  | none => pure ()
  | some (.num n) =>
    if n.toFloat <= 0 then
      Schema.zerr "sample_rate" "Number must be greater than 0"
    fields := fields.push ("sample_rate", .num n)
  | some other =>
    Schema.zerr "sample_rate" s!"Expected number, received {Schema.received other}"
  match ← Schema.optBool "" raw "breaks_cycles" with
  | none => pure ()
  | some b => fields := fields.push ("breaks_cycles", .bool b)
  match raw.getField? "ports" with
  | none => pure ()
  | some p => fields := fields.push ("ports", ← Schema.ports "ports" p)
  match raw.getField? "body" with
  | none => Schema.zerr "body" "Required"
  | some b => fields := fields.push ("body", ← Schema.block "body" b)
  let mut top : TopLevel := {}
  match raw.getField? "params" with
  | none => pure ()
  | some p => top := { top with params := some (← Schema.topParams p) }
  match raw.getField? "audio_outputs" with
  | none => pure ()
  | some a => top := { top with audioOutputs := some (← Schema.topAudioOutputs a) }
  pure (.obj fields, top)

-- ─────────────────────────────────────────────────────────────
-- Op classification (raise.ts tables, verbatim)
-- ─────────────────────────────────────────────────────────────

/-- Legacy ops collapsing to `nameRef(<carried name>)`. -/
private def refOpsName : List String :=
  ["input", "reg", "typeParam", "param", "trigger",
   "paramExpr", "triggerParamExpr"]

private def builtinNullaryOps : List String := ["sampleRate", "sampleIndex"]

/-- N-ary builtins raising to `call(nameRef(op), args)` — camelCase
    canon plus the snake_case spellings older fixtures use. -/
private def builtinCallOps : List String :=
  ["select", "clamp", "round", "ldexp", "floorDiv",
   "sqrt", "abs", "floatExponent", "arraySet",
   "floor", "ceil", "toInt", "toBool", "toFloat",
   "floor_div", "float_exponent", "to_int", "to_bool", "to_float"]

-- ─────────────────────────────────────────────────────────────
-- Expression raising
-- ─────────────────────────────────────────────────────────────

private def rerr {α} (msg : String) : Except String α := .error s!"raise: {msg}"

/-- TS reads a field and casts it `as string`; a non-string would flow
    junk into the output, which the typed AST cannot represent — error
    instead (unreachable on inputs TS raises successfully). -/
private def fieldStr (node : JsonV) (k ctx : String) : Except String String :=
  match node.getField? k with
  | some (.str s) => pure s
  | _ => rerr s!"{ctx} requires string '{k}', got {JsonV.stringifyOpt (node.getField? k)}"

mutual

partial def raiseExpr (e : JsonV) : Except String ParsedExpr := do
  match e with
  | .num n => pure (.num n)
  | .bool b => pure (.bool b)
  | .arr items => do
    let mut out : Array ParsedExpr := #[]
    for item in items do
      out := out.push (← raiseExpr item)
    pure (.arr out)
  | .obj _ => raiseOpNode e
  | other => rerr s!"invalid expr value: {other.compress}"

/-- TS indexes `args[i]` and feeds the result to `raiseExpr`; a missing
    element is `undefined`, which raiseExpr rejects with this exact
    message. -/
partial def raiseArgAt (args : Array JsonV) (i : Nat) : Except String ParsedExpr :=
  match args[i]? with
  | some a => raiseExpr a
  | none => rerr "invalid expr value: undefined"

partial def raiseExprOpt (e : Option JsonV) : Except String ParsedExpr :=
  match e with
  | some v => raiseExpr v
  | none => rerr "invalid expr value: undefined"

partial def raiseArgs (node : JsonV) (op : String) : Except String (Array JsonV) :=
  match node.getField? "args" with
  | some (.arr a) => pure a
  | _ => rerr s!"'{op}' requires an args array, got {JsonV.stringifyOpt (node.getField? "args")}"

partial def raiseOpNode (node : JsonV) : Except String ParsedExpr := do
  let some op := node.opOf?
    | match node.getField? "op" with
      | none => rerr s!"expression object missing 'op' field: {node.compress}"
      | some _ => rerr s!"expression object missing 'op' field: {node.compress}"

  -- ── Reference collapse ────────────────────────────────────
  if refOpsName.contains op then
    return .nameRef (← fieldStr node "name" s!"'{op}'")
  if op == "delayRef" then
    return .nameRef (← fieldStr node "id" "'delayRef'")
  if op == "binding" then
    return .binding (← fieldStr node "name" "'binding'")

  -- ── Builtin → call ───────────────────────────────────────
  if builtinNullaryOps.contains op then
    return .call (.nameRef op) #[]
  if builtinCallOps.contains op then
    let args ← raiseArgs node op
    let mut out : Array ParsedExpr := #[]
    for a in args do
      out := out.push (← raiseExpr a)
    return .call (.nameRef op) out

  -- ── Pass-through binary / unary ──────────────────────────
  if let some tag := BinaryOpTag.ofWire? op then
    let args ← raiseArgs node op
    return .binary tag (← raiseArgAt args 0) (← raiseArgAt args 1)
  if let some tag := UnaryOpTag.ofWire? op then
    let args ← raiseArgs node op
    return .unary tag (← raiseArgAt args 0)

  -- ── Structured / ADT ─────────────────────────────────────
  match op with
  | "nestedOut" => do
    let ref ← fieldStr node "ref" "'nestedOut'"
    let output := match node.getField? "output" with
      | some v => v.jsString
      | none => "undefined"
    pure (.nestedOut ref output)
  | "index" => do
    let args ← raiseArgs node op
    pure (.index (← raiseArgAt args 0) (← raiseArgAt args 1))
  | "tag" => do
    let variant ← fieldStr node "variant" "'tag'"
    let payload ← match node.getField? "payload" with
      | none => pure none
      | some (.obj entries) => do
        let mut out : Array TagPayloadEntry := #[]
        for (field, value) in entries do
          out := out.push (.mk field (← raiseExpr value))
        pure (if out.isEmpty then none else some out)
      | some other => rerr s!"'tag' payload must be an object, got {other.compress}"
    pure (.tag variant payload)
  | "match" => do
    let some (JsonV.obj armEntries) := node.getField? "arms"
      | rerr s!"'match' requires an arms object, got {JsonV.stringifyOpt (node.getField? "arms")}"
    let mut arms : Array MatchArm := #[]
    for (variant, arm) in armEntries do
      -- Legacy arms carry bind name(s) only; payload field labels are
      -- not in the schema, so raise emits `_unknown` placeholders.
      let bindNames : Array String ← match arm.getField? "bind" with
        | none => pure #[]
        | some (.str s) => pure #[s]
        | some (.arr bs) => do
          let mut out : Array String := #[]
          for b in bs do
            match b with
            | .str s => out := out.push s
            | other => rerr s!"'match' arm bind must be a string, got {other.compress}"
          pure out
        | some other => rerr s!"'match' arm bind must be a string or array, got {other.compress}"
      let body ← raiseExprOpt (arm.getField? "body")
      arms := arms.push (.mk variant (bindNames.map ("_unknown", ·)) body)
    let scrutinee ← raiseExprOpt (node.getField? "scrutinee")
    pure (.match_ scrutinee arms)
  | "let" => do
    let some (JsonV.obj bindEntries) := node.getField? "bind"
      | rerr s!"'let' requires a bind object, got {JsonV.stringifyOpt (node.getField? "bind")}"
    let mut bind : Array (String × ParsedExpr) := #[]
    for (k, v) in bindEntries do
      bind := bind.push (k, ← raiseExpr v)
    pure (.letIn bind (← raiseExprOpt (node.getField? "in")))
  | "fold" => do
    let over ← raiseExprOpt (node.getField? "over")
    let init ← raiseExprOpt (node.getField? "init")
    pure (.fold over init
      (← fieldStr node "acc_var" "'fold'") (← fieldStr node "elem_var" "'fold'")
      (← raiseExprOpt (node.getField? "body")))
  | "scan" => do
    let over ← raiseExprOpt (node.getField? "over")
    let init ← raiseExprOpt (node.getField? "init")
    pure (.scan over init
      (← fieldStr node "acc_var" "'scan'") (← fieldStr node "elem_var" "'scan'")
      (← raiseExprOpt (node.getField? "body")))
  | "generate" => do
    let count ← raiseExprOpt (node.getField? "count")
    pure (.generate count (← fieldStr node "var" "'generate'")
      (← raiseExprOpt (node.getField? "body")))
  | "iterate" => do
    let count ← raiseExprOpt (node.getField? "count")
    let init ← raiseExprOpt (node.getField? "init")
    pure (.iterate count (← fieldStr node "var" "'iterate'") init
      (← raiseExprOpt (node.getField? "body")))
  | "chain" => do
    let count ← raiseExprOpt (node.getField? "count")
    let init ← raiseExprOpt (node.getField? "init")
    pure (.chain count (← fieldStr node "var" "'chain'") init
      (← raiseExprOpt (node.getField? "body")))
  | "map2" => do
    let over ← raiseExprOpt (node.getField? "over")
    pure (.map2 over (← fieldStr node "elem_var" "'map2'")
      (← raiseExprOpt (node.getField? "body")))
  | "zipWith" => do
    let a ← raiseExprOpt (node.getField? "a")
    let b ← raiseExprOpt (node.getField? "b")
    pure (.zipWith a b
      (← fieldStr node "x_var" "'zipWith'") (← fieldStr node "y_var" "'zipWith'")
      (← raiseExprOpt (node.getField? "body")))
  | other => rerr s!"unknown expression op '{other}'"

end

-- ─────────────────────────────────────────────────────────────
-- Ports + type defs
-- ─────────────────────────────────────────────────────────────

private def raiseShapeDim (d : JsonV) : Except String ShapeDim := do
  match d with
  | .num n => pure (.lit n)
  | .obj _ => pure (.ref (← fieldStr d "name" "shape dim"))
  | other => rerr s!"invalid shape dim: {other.compress}"

private def raisePortType (pt : JsonV) : Except String PortTypeDecl := do
  match pt with
  | .str s => pure (.scalar s)
  | .obj _ => do
    let element ← fieldStr pt "element" "array port type"
    let some (JsonV.arr dims) := pt.getField? "shape"
      | rerr s!"array port type requires a shape array, got {JsonV.stringifyOpt (pt.getField? "shape")}"
    let mut shape : Array ShapeDim := #[]
    for d in dims do
      shape := shape.push (← raiseShapeDim d)
    pure (.array element shape)
  | other => rerr s!"invalid port type: {other.compress}"

private def raisePort (p : JsonV) : Except String ProgramPort := do
  match p with
  | .str name => pure (.bare name)
  | .obj _ => do
    let name ← fieldStr p "name" "port spec"
    let type? ← match p.getField? "type" with
      | none => pure none
      | some t => pure (some (← raisePortType t))
    let default? ← match p.getField? "default" with
      | none => pure none
      | some d => pure (some (← raiseExpr d))
    pure (.spec { name, type?, default? })
  | other => rerr s!"invalid port: {other.compress}"

private def raiseStructField (f : JsonV) (ctx : String) : Except String StructField := do
  let name ← fieldStr f "name" ctx
  let st ← fieldStr f "scalar_type" ctx
  let some scalarType := ScalarKind.ofWire? st
    | rerr s!"{ctx} has unknown scalar_type '{st}'"
  pure { name, scalarType }

private def raiseTypeDef (td : JsonV) : Except String TypeDef := do
  let name ← fieldStr td "name" "type def"
  match td.getStr? "kind" with
  | some "alias" => do
    pure (.alias name (← fieldStr td "base" "alias type def"))
  | some "sum" => do
    let some (JsonV.arr items) := td.getField? "variants"
      | rerr s!"sum type def requires a variants array, got {JsonV.stringifyOpt (td.getField? "variants")}"
    let mut variants : Array SumVariant := #[]
    for v in items do
      let vName ← fieldStr v "name" "sum variant"
      let some (JsonV.arr payloadItems) := v.getField? "payload"
        | rerr s!"sum variant requires a payload array, got {JsonV.stringifyOpt (v.getField? "payload")}"
      let mut payload : Array StructField := #[]
      for f in payloadItems do
        payload := payload.push (← raiseStructField f "sum variant payload field")
      variants := variants.push { name := vName, payload }
    pure (.sum name variants)
  | _ => do
    -- TS falls through to the struct shape for any other kind.
    let some (JsonV.arr items) := td.getField? "fields"
      | rerr s!"struct type def requires a fields array, got {JsonV.stringifyOpt (td.getField? "fields")}"
    let mut fields : Array StructField := #[]
    for f in items do
      fields := fields.push (← raiseStructField f "struct field")
    pure (.struct name fields)

private def raisePorts (ports : JsonV) : Except String ProgramPorts := do
  let portArray (k : String) : Except String (Option (Array ProgramPort)) := do
    match ports.getField? k with
    | none => pure none
    | some (.arr items) => do
      let mut out : Array ProgramPort := #[]
      for p in items do
        out := out.push (← raisePort p)
      pure (some out)
    | some other => rerr s!"ports.{k} must be an array, got {other.compress}"
  let inputs ← portArray "inputs"
  let outputs ← portArray "outputs"
  let typeDefs ← match ports.getField? "type_defs" with
    | none => pure none
    | some (.arr items) => do
      let mut out : Array TypeDef := #[]
      for td in items do
        out := out.push (← raiseTypeDef td)
      pure (some out)
    | some other => rerr s!"ports.type_defs must be an array, got {other.compress}"
  pure { inputs, outputs, typeDefs }

private def raiseTypeParams (tps : JsonV) :
    Except String (Array (String × TypeParamSpec)) := do
  -- TS passes `legacy.type_params` through by reference; the typed AST
  -- requires the (Zod-shaped) `{type:'int', default?}` form, which is
  -- what the validated top level always supplies.
  let .obj entries := tps
    | rerr s!"type_params must be an object, got {tps.compress}"
  let mut out : Array (String × TypeParamSpec) := #[]
  for (name, spec) in entries do
    if spec.getStr? "type" != some "int" then
      rerr s!"type_params.{name} must have type 'int'"
    let default? ← match spec.getField? "default" with
      | none => pure none
      | some (.num n) => pure (some n)
      | some other => rerr s!"type_params.{name} default must be a number, got {other.compress}"
    out := out.push (name, { default? })
  pure out

-- ─────────────────────────────────────────────────────────────
-- Bounds lowering (port of compiler/parse/lower_bounds.ts)
-- ─────────────────────────────────────────────────────────────

/-- `[lo, hi]`, either side open. Built-in alias bounds are all
    integral, and explicit `bounds` cannot survive the Zod strip, so
    `Int` is exact here. -/
private abbrev Bounds := Option Int × Option Int

private def builtinPortBounds : String → Option Bounds
  | "signal" | "bipolar" => some (some (-1), some 1)
  | "unipolar" | "phase" => some (some 0, some 1)
  | "freq" => some (some 0, none)
  | _ => none

private def aliasBounds : Option PortTypeDecl → Option Bounds
  | some (.scalar name) => builtinPortBounds name
  | _ => none

private def exprIsNum (e : ParsedExpr) (i : Int) : Bool :=
  match e with
  | .num n => n.toFloat.toBits == (Float.ofInt i).toBits
  | _ => false

private def callArgsTo (e : ParsedExpr) (callee : String) : Option (Array ParsedExpr) :=
  match e with
  | .call (.nameRef n) args => if n == callee then some args else none
  | _ => none

/-- True if `e` already enforces `bounds` (idempotency guard). Checks
    the parser-level call shapes; the elaborator-level direct-op shapes
    the TS guard also accepts are unrepresentable in a ParsedProgram. -/
private def alreadyWrapped (e : ParsedExpr) (bounds : Bounds) : Bool :=
  let (lo, hi) := bounds
  let clampMatch :=
    match callArgsTo e "clamp" with
    | some args =>
      args.size == 3 &&
      (match lo with | some l => exprIsNum args[1]! l | none => false) &&
      (match hi with | some h => exprIsNum args[2]! h | none => false)
    | none => false
  let selectMatch :=
    match callArgsTo e "select" with
    | some args =>
      args.size == 3 &&
      (match args[0]!, lo, hi with
       | .binary .gt _ rhs, some l, none => exprIsNum rhs l && exprIsNum args[2]! l
       | .binary .lt _ rhs, none, some h => exprIsNum rhs h && exprIsNum args[2]! h
       | _, _, _ => false)
    | none => false
  clampMatch || selectMatch

private def wrapWithBound (e : ParsedExpr) (bounds : Bounds) : ParsedExpr :=
  if alreadyWrapped e bounds then e
  else
    let lit (i : Int) : ParsedExpr := .num (JsonNumber.fromInt i)
    match bounds with
    | (some lo, some hi) => .call (.nameRef "clamp") #[e, lit lo, lit hi]
    | (some lo, none) =>
      .call (.nameRef "select") #[.binary .gt e (lit lo), e, lit lo]
    | (none, some hi) =>
      .call (.nameRef "select") #[.binary .lt e (lit hi), e, lit hi]
    | (none, none) => e

/-- Port of `lowerBoundsToClamps`: wrap bounded input defaults and
    bounded-output assigns in clamp/select chains. Pure (the TS version
    mutates in place). Bounded inputs without defaults drop their
    bounds silently — there is nothing to wrap. -/
partial def lowerBounds (p : Program) : Program :=
  let ports := p.ports
  -- Nested programs first (mirrors the TS recursion).
  let decls := p.body.decls.map fun d =>
    match d with
    | .prog n inner => .prog n (lowerBounds inner)
    | other => other
  -- Inputs: wrap defaults where the type alias carries bounds.
  let inputs' := ports.bind (·.inputs) |>.map fun ins =>
    ins.map fun port =>
      match port with
      | .spec s =>
        match aliasBounds s.type?, s.default? with
        | some b, some dflt =>
          ProgramPort.spec { s with default? := some (wrapWithBound dflt b) }
        | _, _ => ProgramPort.spec s
      | bare => bare
  -- Outputs: collect bounds by port name.
  let outputBounds : Array (String × Bounds) :=
    (ports.bind (·.outputs) |>.getD #[]).filterMap fun port =>
      match port with
      | .spec s => (aliasBounds s.type?).map (s.name, ·)
      | .bare _ => none
  let assigns :=
    if outputBounds.isEmpty then p.body.assigns
    else p.body.assigns.map fun a =>
      match a with
      | .output name e =>
        match outputBounds.find? (·.1 == name) with
        | some (_, b) => .output name (wrapWithBound e b)
        | none => .output name e
      | other => other
  let ports' := ports.map fun pp => { pp with inputs := inputs' }
  .mk p.name p.typeParams ports' (.mk decls assigns) p.breaksCycles

-- ─────────────────────────────────────────────────────────────
-- Body decls + assigns + program
-- ─────────────────────────────────────────────────────────────

/-- JS `String(obj.op)` for the unknown-op error messages. -/
private def opString (obj : JsonV) : String :=
  match obj.getField? "op" with
  | some v => v.jsString
  | none => "undefined"

/-- `{zeros: N}` / inner `{typeParam: n}` legacy sugar (Delay.json). -/
private def raiseZerosArg (arg : JsonV) : Except String ParsedExpr := do
  match arg with
  | .obj _ =>
    if (arg.getField? "op").isNone then
      match arg.getStr? "typeParam" with
      | some name => pure (.nameRef name)
      | none => raiseExpr arg
    else raiseExpr arg
  | _ => raiseExpr arg

private def raiseRegInit (init : JsonV) : Except String ParsedExpr := do
  match init with
  | .obj _ =>
    if (init.getField? "op").isNone then
      match init.getField? "zeros" with
      | some n => do
        pure (.call (.nameRef "zeros") #[← raiseZerosArg n])
      | none => raiseExpr init
    else raiseExpr init
  | _ => raiseExpr init

mutual

partial def raiseBodyDecl (decl : JsonV) : Except String BodyDecl := do
  let .obj _ := decl
    | rerr s!"body decl must be an object, got {decl.compress}"
  match decl.opOf?.getD (opString decl) with
  | "regDecl" => do
    let init ← match decl.getField? "init" with
      | some i => pure i
      | none =>
        let shownName := match decl.getField? "name" with
          | some v => v.jsString
          | none => "undefined"
        rerr s!"regDecl '{shownName}' missing init"
    let name ← fieldStr decl "name" "regDecl"
    let type? := decl.getStr? "type"   -- only a *string* type raises (TS: typeof check)
    pure (.reg name (← raiseRegInit init) type?)
  | "delayDecl" => do
    let name ← fieldStr decl "name" "delayDecl"
    let update ← raiseExprOpt (decl.getField? "update")
    let init ← raiseExprOpt (decl.getField? "init")
    let type? := decl.getStr? "type"
    pure (.delay name update init type?)
  | "paramDecl" => do
    let name ← fieldStr decl "name" "paramDecl"
    let value? := match decl.getField? "value" with
      | some (.num n) => some n   -- TS: only `typeof value === 'number'`
      | _ => none
    pure (.param name value?)
  | "instanceDecl" => do
    let name ← fieldStr decl "name" "instanceDecl"
    let progName ← fieldStr decl "program" "instanceDecl"
    let typeArgs ← match decl.getField? "type_args" with
      | none => pure none
      | some (.obj entries) => do
        let mut out : Array (String × JsonNumber) := #[]
        for (param, value) in entries do
          match value with
          | .num n => out := out.push (param, n)
          | other => rerr s!"instanceDecl type_args.{param} must be a number, got {other.compress}"
        pure (if out.isEmpty then none else some out)
      | some other => rerr s!"instanceDecl type_args must be an object, got {other.compress}"
    let inputs ← match decl.getField? "inputs" with
      | none => pure none
      | some (.obj entries) => do
        let mut out : Array (String × ParsedExpr) := #[]
        for (port, value) in entries do
          out := out.push (port, ← raiseExpr value)
        pure (if out.isEmpty then none else some out)
      | some other => rerr s!"instanceDecl inputs must be an object, got {other.compress}"
    pure (.inst name progName typeArgs inputs)
  | "programDecl" => do
    let name ← fieldStr decl "name" "programDecl"
    let some inner := decl.getField? "program"
      | rerr "programDecl missing program"
    pure (.prog name (← raiseProgram inner))
  | other => rerr s!"unknown body decl op '{other}'"

partial def raiseBodyAssign (a : JsonV) : Except String BodyAssign := do
  let .obj _ := a
    | rerr s!"body assign must be an object, got {a.compress}"
  match a.opOf?.getD (opString a) with
  | "outputAssign" => do
    let name ← fieldStr a "name" "outputAssign"
    pure (.output name (← raiseExprOpt (a.getField? "expr")))
  | "nextUpdate" => do
    let some target := a.getField? "target"
      | rerr "nextUpdate missing target"
    let kind ← match target.getStr? "kind" with
      | some "reg" => pure NextTargetKind.reg
      | some "delay" => pure NextTargetKind.delay
      | _ => rerr s!"nextUpdate target kind must be 'reg' or 'delay', got {JsonV.stringifyOpt (target.getField? "kind")}"
    let name ← fieldStr target "name" "nextUpdate target"
    pure (.next kind name (← raiseExprOpt (a.getField? "expr")))
  | other => rerr s!"unknown body assign op '{other}'"

/-- Port of `raiseProgram`: raise body decls/assigns, lift ports and
    type params, then run the bounds lowering. Pure; errors carry the
    TS message strings. -/
partial def raiseProgram (legacy : JsonV) : Except String Program := do
  let body := legacy.getField? "body"
  let mut decls : Array BodyDecl := #[]
  match body.bind (·.getField? "decls") with
  | none | some .null => pure ()
  | some (.arr items) =>
    for d in items do
      decls := decls.push (← raiseBodyDecl d)
  | some other => rerr s!"body decls must be an array, got {other.compress}"
  let mut assigns : Array BodyAssign := #[]
  match body.bind (·.getField? "assigns") with
  | none | some .null => pure ()
  | some (.arr items) =>
    for a in items do
      assigns := assigns.push (← raiseBodyAssign a)
  | some other => rerr s!"body assigns must be an array, got {other.compress}"

  let name ← fieldStr legacy "name" "program"
  let typeParams ← match legacy.getField? "type_params" with
    | none => pure none
    | some tps => pure (some (← raiseTypeParams tps))
  let ports ← match legacy.getField? "ports" with
    | none => pure none
    | some p => pure (some (← raisePorts p))
  -- Only a literal `true` raises (TS: `legacy.breaks_cycles === true`).
  let breaksCycles :=
    if legacy.getField? "breaks_cycles" == some (.bool true) then some true else none

  pure <| lowerBounds (.mk name typeParams ports (.mk decls assigns) breaksCycles)

end

-- ─────────────────────────────────────────────────────────────
-- Entry
-- ─────────────────────────────────────────────────────────────

/-- The full TS ingest path: schema check → validation/strip → split →
    raise. -/
def raiseFile (raw : JsonV) : Except String (Program × TopLevel) := do
  let (node, top) ← normalizeProgramFile raw
  pure (← raiseProgram node, top)

/-- The diff-harness wrapper shape: `{program, params?, audio_outputs?}`
    (absent metadata stays absent). -/
def raisedFileJson (prog : Program) (top : TopLevel) : Json :=
  Json.mkObj <|
    [("program", prog.toJson)]
    ++ (match top.params with
        | some ps => [("params", Json.arr (ps.map LegacyParam.toJson))]
        | none => [])
    ++ (match top.audioOutputs with
        | some os => [("audio_outputs", Json.arr (os.map AudioOutput.toJson))]
        | none => [])

end Tropical.Parse.Raise
