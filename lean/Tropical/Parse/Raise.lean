import Tropical.Parse.Nodes

/-!
# raise — `tropical_program_2` JSON → the patch-bay node + top-level metadata

What survives of the raise layer: `normalizeProgramFile` — the
schema-tag check (exact TS message), the Zod-strip structural
validation, and the program-node / top-level-metadata split. The wire
is a PATCH BAY (instances + wiring + params of registered types): the
ingest (`Engine/ProgramIO.lean`) walks the returned `JsonV` node
directly, wire expressions are validated by the session grammar
(`Tropical.Expr.validateExpr`), and program definitions over the wire
are refused at ingest.

The rest of raise — expression raising, port raising, the bounds
lowering, `raiseProgram`/`raiseFile`, the retired-op tables — died with
the elaborator (2026-07-26): with `ParsedExpr` gone there is nothing to
raise INTO. The refusal discipline lives on at the two surviving
boundaries: the schema/shape checks here, and the ingest's programDecl
retirement message.
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

/-- One stripped top-level `audio_outputs` entry. The `expr` arm keeps
    the (shallowly validated) wire expression verbatim, mirroring how
    the TS layer passes it through untouched. -/
inductive AudioOutput where
  | ref (inst : String) (output : JsonV)
  | expr (e : JsonV)

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

end Tropical.Parse.Raise
