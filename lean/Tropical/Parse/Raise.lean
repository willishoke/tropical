import Tropical.Parse.Nodes

/-!
# raise — `tropical_program_2` JSON → the patch-bay node

What survives of the raise layer: `normalizeProgramFile` — the
schema-tag check, structural validation, and canonical program-node
normalization. The wire
is a PATCH BAY (instances + wiring + params of registered types): the
ingest (`Engine/ProgramIO.lean`) walks the returned `JsonV` node
directly, wire expressions are validated by the session grammar
(`Tropical.WireExpr`'s decoder), and program definitions over the wire
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

end Schema

/-- Validate and normalize a Plan-2 patch document. The returned node carries
    `op:'program'` plus the canonical program fields. Retired top-level
    parameter/output carriers fail with their migration spelling. -/
def normalizeProgramFile (raw : JsonV) : Except String JsonV := do
  if raw.getField? "schema" != some (.str "tropical_program_2") then
    let shown := match raw.getField? "schema" with
      | some v => v.jsString
      | none => "undefined"
    throw s!"Unknown schema '{shown}'. Expected 'tropical_program_2'."
  let name ← Schema.reqStr "" raw "name"
  if name.isEmpty then
    Schema.zerr "name" "String must contain at least 1 character(s)"
  if (raw.getField? "params").isSome then
    Schema.zerr "params"
      "top-level params are retired; declare body.decls paramDecl entries instead"
  if (raw.getField? "audio_outputs").isSome then
    Schema.zerr "audio_outputs"
      "audio_outputs is retired; declare body.assigns outputAssign{name:'dac.out',expr:{op:'ref',...}} instead"
  if (raw.getField? "breaks_cycles").isSome then
    Schema.zerr "breaks_cycles"
      "breaks_cycles is retired; tropical patch graphs are closed-form and acyclic"
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
  match raw.getField? "ports" with
  | none => pure ()
  | some p => fields := fields.push ("ports", ← Schema.ports "ports" p)
  match raw.getField? "body" with
  | none => Schema.zerr "body" "Required"
  | some b => fields := fields.push ("body", ← Schema.block "body" b)
  pure (.obj fields)

end Tropical.Parse.Raise
