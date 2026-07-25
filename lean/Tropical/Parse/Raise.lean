import Tropical.Parse.Nodes
import Tropical.Parse.BoundLower

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
3. **`raiseProgram`** (compiler/parse/raise.ts) — the op mappings and
   error strings, followed by `lowerBoundsToClamps`
   (compiler/parse/lower_bounds.ts).

This is the front door of the trunk grammar: retired constructs
(combinators, sum types, binders, generics, state ops) are REFUSED
here, at ingest, with the retirement message — the grammar you can
spell is the language that compiles.

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
def exprNode (path : String) (v : JsonV) : Except String Unit := do
  match v with
  | .num _ | .bool _ => pure ()
  | .arr items =>
    items.attach.zipIdx.forM fun (⟨x, _⟩, i) =>
      exprNode (sub path (toString i)) x
  | .obj _ =>
    match v.getStr? "op" with
    | some _ => pure ()
    | none => zerr path "Invalid input"
  | _ => zerr path "Invalid input"
termination_by sizeOf v
decreasing_by have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp_all <;> omega

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
  if (v.getField? "type_defs").isSome then
    zerr (sub path "type_defs")
      "type_defs are retired — sum/struct/alias type defs left with the surface language and generics"
  pure (.obj fields)

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
  if (raw.getField? "type_params").isSome then
    Schema.zerr "type_params" "type_params are retired — generics left with the surface language"
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
  ["input", "param", "trigger", "paramExpr", "triggerParamExpr"]

/-- The front-door refusal: a retired construct spelled in JSON dies at
    ingest — nothing downstream can represent it, let alone lower it. -/
private def retiredOp (op : String) : String :=
  s!"'{op}' is not a trunk construct — combinator/sum-type lowering was retired with its producers (the literate surface language and generics); a summing indexed family is authored as bankSum"

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

/-- Sized view of the `args` array: same behavior and error string as
    the old `raiseArgs`, but the result carries the `sizeOf` bound that
    `raiseExpr?`'s termination measure descends through. -/
private def raiseArgsD (node : JsonV) (op : String) :
    Except String {args : Array JsonV // sizeOf args < sizeOf node} :=
  match h : node.getField? "args" with
  | some (.arr a) =>
    pure ⟨a, by have := JsonV.sizeOf_lt_of_getField h; simp at this; omega⟩
  | _ => rerr s!"'{op}' requires an args array, got {JsonV.stringifyOpt (node.getField? "args")}"

/-- The whole `raiseExpr` / `raiseOpNode` / `raiseArgAt` / `raiseExprOpt`
    family as ONE recursive function over `Option JsonV`. `undefined`
    (`none`) was always in the domain — TS feeds `args[i]` and missing
    fields straight into raiseExpr, and rejects them with the exact
    message below — and folding it in gives every recursive call a
    strictly smaller `sizeOf`, with no lexicographic measure and no
    mutual block. The public names below are thin views of this one. -/
def raiseExpr? : Option JsonV → Except String ParsedExpr
  | none => rerr "invalid expr value: undefined"
  | some node => do
  match _hn : node with
  | .num n => pure (.num n)
  | .bool b => pure (.bool b)
  | .arr items => do
    let out ← items.attach.mapM fun ⟨x, _⟩ => raiseExpr? (some x)
    pure (.arr out)
  | .obj _ => do
    let some op := node.opOf?
      | match node.getField? "op" with
        | none => rerr s!"expression object missing 'op' field: {node.compress}"
        | some _ => rerr s!"expression object missing 'op' field: {node.compress}"

    -- ── Reference collapse ────────────────────────────────────
    if refOpsName.contains op then
      return .nameRef (← fieldStr node "name" s!"'{op}'")

    -- ── Builtin → call ───────────────────────────────────────
    if builtinNullaryOps.contains op then
      return .call (.nameRef op) #[]
    if builtinCallOps.contains op then
      let ⟨args, _⟩ ← raiseArgsD node op
      let out ← args.attach.mapM fun ⟨a, _⟩ => raiseExpr? (some a)
      return .call (.nameRef op) out

    -- ── Pass-through binary / unary ──────────────────────────
    if let some tag := BinaryOpTag.ofWire? op then
      let ⟨args, _⟩ ← raiseArgsD node op
      return .binary tag (← raiseExpr? args[0]?) (← raiseExpr? args[1]?)
    if let some tag := UnaryOpTag.ofWire? op then
      let ⟨args, _⟩ ← raiseArgsD node op
      return .unary tag (← raiseExpr? args[0]?)

    -- ── Structured / ADT ─────────────────────────────────────
    match op with
    | "nestedOut" => do
      let ref ← fieldStr node "ref" "'nestedOut'"
      let output := match node.getField? "output" with
        | some v => v.jsString
        | none => "undefined"
      pure (.nestedOut ref output)
    | "index" => do
      let ⟨args, _⟩ ← raiseArgsD node op
      pure (.index (← raiseExpr? args[0]?) (← raiseExpr? args[1]?))
    | "let" | "fold" | "scan" | "generate" | "iterate" | "chain"
    | "map2" | "zipWith" | "tag" | "match" | "binding" =>
      rerr (retiredOp op)
    | "reg" | "delayRef" | "delayValue" =>
      rerr s!"'{op}' is retired — there is no state primitive; kernels are closed-form f(τ, params)"
    | other => rerr s!"unknown expression op '{other}'"
  | other => rerr s!"invalid expr value: {other.compress}"
termination_by o => sizeOf o
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp_all <;> omega)
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ args›; simp_all <;> omega)
    | (refine Nat.lt_of_le_of_lt (sizeOf_getElem?_le _ _) ?_;
       simp_all <;> omega)

def raiseExpr (e : JsonV) : Except String ParsedExpr := raiseExpr? (some e)

def raiseExprOpt (e : Option JsonV) : Except String ParsedExpr := raiseExpr? e

-- ─────────────────────────────────────────────────────────────
-- Ports + type defs
-- ─────────────────────────────────────────────────────────────

private def raisePortType (pt : JsonV) : Except String PortTypeDecl := do
  match pt with
  | .str s => pure (.scalar s)
  | .obj _ => do
    let element ← fieldStr pt "element" "array port type"
    let some (JsonV.arr dims) := pt.getField? "shape"
      | rerr s!"array port type requires a shape array, got {JsonV.stringifyOpt (pt.getField? "shape")}"
    let mut shape : Array JsonNumber := #[]
    for d in dims do
      match d with
      | .num n => shape := shape.push n
      | other => rerr s!"invalid shape dim: {other.compress} (type-param dims are retired — shapes are literal)"
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
  if (ports.getField? "type_defs").isSome then
    rerr "ports.type_defs are retired — sum/struct/alias type defs left with the surface language and generics"
  pure { inputs, outputs }

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

/-- Widen the integral alias bounds to the shared `JsonNumber` bound pair, so
    the raise path folds them through the same idempotent `wrapWithBound` as
    the surface parser (`Parse/BoundLower.lean`). -/
private def boundPair : Bounds → Tropical.Parse.BoundPair
  | (lo, hi) => (lo.map (JsonNumber.fromInt ·), hi.map (JsonNumber.fromInt ·))

/-- Port of `lowerBoundsToClamps`: wrap bounded input defaults and
    bounded-output assigns in clamp/select chains. Pure (the TS version
    mutates in place). Bounded inputs without defaults drop their
    bounds silently — there is nothing to wrap. -/
def lowerBounds : Program → Program
  | .mk pName ports (.mk pDecls pAssigns) pBreaksCycles =>
    -- Nested programs first (mirrors the TS recursion).
    let decls := pDecls.attach.map fun ⟨d, _⟩ =>
      match _hd : d with
      | .prog n inner => .prog n (lowerBounds inner)
      | other => other
    -- Inputs: wrap defaults where the type alias carries bounds.
    let inputs' := ports.bind (·.inputs) |>.map fun ins =>
      ins.map fun port =>
        match port with
        | .spec s =>
          match aliasBounds s.type?, s.default? with
          | some b, some dflt =>
            ProgramPort.spec { s with default? := some (wrapWithBound dflt (boundPair b)) }
          | _, _ => ProgramPort.spec s
        | bare => bare
    -- Outputs: collect bounds by port name.
    let outputBounds : Array (String × Bounds) :=
      (ports.bind (·.outputs) |>.getD #[]).filterMap fun port =>
        match port with
        | .spec s => (aliasBounds s.type?).map (s.name, ·)
        | .bare _ => none
    let assigns :=
      if outputBounds.isEmpty then pAssigns
      else pAssigns.map fun a =>
        match a with
        | .output name e =>
          match outputBounds.find? (·.1 == name) with
          | some (_, b) => .output name (wrapWithBound e (boundPair b))
          | none => .output name e
    let ports' := ports.map fun pp => { pp with inputs := inputs' }
    .mk pName ports' (.mk decls assigns) pBreaksCycles
termination_by p => sizeOf p
decreasing_by have := Array.sizeOf_lt_of_mem ‹_ ∈ pDecls›; simp_all <;> omega

-- ─────────────────────────────────────────────────────────────
-- Body decls + assigns + program
-- ─────────────────────────────────────────────────────────────

/-- JS `String(obj.op)` for the unknown-op error messages. -/
private def opString (obj : JsonV) : String :=
  match obj.getField? "op" with
  | some v => v.jsString
  | none => "undefined"

def raiseBodyAssign (a : JsonV) : Except String BodyAssign := do
  let .obj _ := a
    | rerr s!"body assign must be an object, got {a.compress}"
  match a.opOf?.getD (opString a) with
  | "outputAssign" => do
    let name ← fieldStr a "name" "outputAssign"
    pure (.output name (← raiseExprOpt (a.getField? "expr")))
  | other => rerr s!"unknown body assign op '{other}'"

mutual

def raiseBodyDecl (decl : JsonV) : Except String BodyDecl := do
  if !decl.isObj then
    rerr s!"body decl must be an object, got {decl.compress}"
  match decl.opOf?.getD (opString decl) with
  | "paramDecl" => do
    let name ← fieldStr decl "name" "paramDecl"
    let value? := match decl.getField? "value" with
      | some (.num n) => some n   -- TS: only `typeof value === 'number'`
      | _ => none
    pure (.param name value?)
  | "instanceDecl" => do
    let name ← fieldStr decl "name" "instanceDecl"
    let progName ← fieldStr decl "program" "instanceDecl"
    if (decl.getField? "type_args").isSome then
      rerr s!"instanceDecl '{name}': type_args are retired — generics left with the surface language"
    let inputs ← match decl.getField? "inputs" with
      | none => pure none
      | some (.obj entries) => do
        let mut out : Array (String × ParsedExpr) := #[]
        for (port, value) in entries do
          out := out.push (port, ← raiseExpr value)
        pure (if out.isEmpty then none else some out)
      | some other => rerr s!"instanceDecl inputs must be an object, got {other.compress}"
    pure (.inst name progName inputs)
  | "programDecl" => do
    let name ← fieldStr decl "name" "programDecl"
    match _hi : decl.getField? "program" with
    | none => rerr "programDecl missing program"
    | some inner => pure (.prog name (← raiseProgram inner))
  | other => rerr s!"unknown body decl op '{other}'"
termination_by sizeOf decl
decreasing_by have := JsonV.sizeOf_lt_of_getField _hi; omega

/-- Port of `raiseProgram`: raise body decls/assigns, lift ports and
    type params, then run the bounds lowering. Pure; errors carry the
    TS message strings. -/
def raiseProgram (legacy : JsonV) : Except String Program := do
  let decls : Array BodyDecl ←
    match _hb : legacy.getField? "body" with
    | none => pure #[]
    | some bodyV =>
      match _hd : bodyV.getField? "decls" with
      | none | some .null => pure #[]
      | some (.arr items) =>
        items.attach.mapM fun ⟨d, _⟩ => raiseBodyDecl d
      | some other => rerr s!"body decls must be an array, got {other.compress}"
  let body := legacy.getField? "body"
  let mut assigns : Array BodyAssign := #[]
  match body.bind (·.getField? "assigns") with
  | none | some .null => pure ()
  | some (.arr items) =>
    for a in items do
      assigns := assigns.push (← raiseBodyAssign a)
  | some other => rerr s!"body assigns must be an array, got {other.compress}"

  let name ← fieldStr legacy "name" "program"
  let ports ← match legacy.getField? "ports" with
    | none => pure none
    | some p => pure (some (← raisePorts p))
  -- Only a literal `true` raises (TS: `legacy.breaks_cycles === true`).
  let breaksCycles :=
    if legacy.getField? "breaks_cycles" == some (.bool true) then some true else none

  pure <| lowerBounds (.mk name ports (.mk decls assigns) breaksCycles)
termination_by sizeOf legacy
decreasing_by
  have := JsonV.sizeOf_lt_of_getField _hb
  have := JsonV.sizeOf_lt_of_getField _hd
  have := Array.sizeOf_lt_of_mem ‹_ ∈ items›
  simp_all <;> omega

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
