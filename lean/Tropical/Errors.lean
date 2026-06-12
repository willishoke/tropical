import Lean.Data.Json

/-!
# The tool-error envelope (spec: mcp/ERRORS.md)

Native port of the envelope the TS engine produces. Every tool failure
is a structured value — machine-readable `code`, human `message`,
`retryable`, the offending `param`/`value`, a `valid` descriptor of
what the param accepts, and a nearest-match `suggestion` (Levenshtein).

A failure is either an envelope constructed here or a verbatim envelope
relayed from the compiler service (kept as raw Json so nothing is lost
in a decode/re-encode round trip).
-/

namespace Tropical

open Lean (Json ToJson toJson)

inductive ErrorCode where
  | unknownProgram | unknownInstance | unknownInput | unknownOutput
  | unknownParam | unknownDevice
  | instanceExists | invalidTypeArgs
  | typeMismatch | shapeMismatch | lengthMismatch | arityError
  | missingArgument | invalidValue | invalidState
  | compileFailed | audioError | internalError
deriving BEq, Repr

def ErrorCode.wire : ErrorCode → String
  | .unknownProgram  => "unknown_program"
  | .unknownInstance => "unknown_instance"
  | .unknownInput    => "unknown_input"
  | .unknownOutput   => "unknown_output"
  | .unknownParam    => "unknown_param"
  | .unknownDevice   => "unknown_device"
  | .instanceExists  => "instance_exists"
  | .invalidTypeArgs => "invalid_type_args"
  | .typeMismatch    => "type_mismatch"
  | .shapeMismatch   => "shape_mismatch"
  | .lengthMismatch  => "length_mismatch"
  | .arityError      => "arity_error"
  | .missingArgument => "missing_argument"
  | .invalidValue    => "invalid_value"
  | .invalidState    => "invalid_state"
  | .compileFailed   => "compile_failed"
  | .audioError      => "audio_error"
  | .internalError   => "internal_error"

/-- A `FieldSpec` for `valid.record` descriptors. -/
structure FieldSpec where
  type     : String           -- "int" | "float" | "string" | "bool"
  required : Bool
  min      : Option Float := none
  max      : Option Float := none

def FieldSpec.toJson (f : FieldSpec) : Json :=
  Json.mkObj <|
    [("type", Json.str f.type), ("required", Json.bool f.required)]
    ++ (match f.min with | some m => [("min", Lean.toJson m)] | none => [])
    ++ (match f.max with | some m => [("max", Lean.toJson m)] | none => [])

inductive Valid where
  | enum (options : Array String)
  | record (fields : List (String × FieldSpec))
  | predicate (predicate : String) (expected got : Option Json)

def Valid.toJson : Valid → Json
  | .enum options => Json.mkObj [
      ("kind", Json.str "enum"),
      ("options", Json.arr (options.map Json.str))]
  | .record fields => Json.mkObj [
      ("kind", Json.str "record"),
      ("fields", Json.mkObj (fields.map fun (n, f) => (n, f.toJson)))]
  | .predicate p expected got => Json.mkObj <|
      [("kind", Json.str "predicate"), ("predicate", Json.str p)]
      ++ (match expected with | some e => [("expected", e)] | none => [])
      ++ (match got with | some g => [("got", g)] | none => [])

structure ErrorEnvelope where
  code       : ErrorCode
  message    : String
  retryable  : Bool := false
  param      : Option String := none
  value      : Option Json := none
  valid      : Option Valid := none
  suggestion : Option Json := none

def ErrorEnvelope.toJson (e : ErrorEnvelope) : Json :=
  Json.mkObj <|
    [("code", Json.str e.code.wire),
     ("message", Json.str e.message),
     ("retryable", Json.bool e.retryable)]
    ++ (match e.param with | some p => [("param", Json.str p)] | none => [])
    ++ (match e.value with | some v => [("value", v)] | none => [])
    ++ (match e.valid with | some v => [("valid", v.toJson)] | none => [])
    ++ (match e.suggestion with | some s => [("suggestion", s)] | none => [])

/-- A tool failure: an envelope built Lean-side, or one relayed verbatim
    from the compiler service. -/
inductive Failure where
  | env (e : ErrorEnvelope)
  | raw (j : Json)

def Failure.toJson : Failure → Json
  | .env e => e.toJson
  | .raw j => j

/-- The handler monad: tool logic that can fail with an envelope. -/
abbrev EngineM := ExceptT Failure IO

-- ── Levenshtein suggestions ──────────────────────────────────────────────────

def levenshtein (a b : String) : Nat := Id.run do
  let sa := a.toList.toArray
  let sb := b.toList.toArray
  let m := sa.size
  let n := sb.size
  let mut prev : Array Nat := Array.range (n + 1)
  for i in [1:m+1] do
    let mut cur : Array Nat := Array.replicate (n + 1) 0
    cur := cur.set! 0 i
    for j in [1:n+1] do
      let cost := if sa[i-1]! == sb[j-1]! then prev[j-1]! else prev[j-1]! + 1
      cur := cur.set! j (Nat.min cost (Nat.min (prev[j]! + 1) (cur[j-1]! + 1)))
    prev := cur
  return prev[n]!

/-- Nearest candidate within `max(2, ⌊len/3⌋)` edits; first wins ties. -/
def nearestMatch (value : String) (candidates : Array String) : Option String := Id.run do
  if candidates.isEmpty then return none
  let mut best := candidates[0]!
  let mut bestD := levenshtein value best
  for c in candidates[1:] do
    let d := levenshtein value c
    if d < bestD then
      best := c
      bestD := d
  return if bestD ≤ Nat.max 2 (value.length / 3) then some best else none

-- ── Throw helpers (mirror failEnum / failRecord / failPredicate / failBare) ──

def throwEnum {α} (code : ErrorCode) (param : String) (value : Json)
    (options : Array String) (message : Option String := none) : EngineM α := do
  let suggestion := match value with
    | .str s => nearestMatch s options
    | _      => none
  let msg := message.getD <|
    s!"Invalid {param}: {value.compress}" ++
    (match suggestion with | some s => s!". Did you mean '{s}'?" | none => "")
  throw <| .env {
    code, message := msg, retryable := false,
    param := some param, value := some value,
    valid := some (.enum options),
    suggestion := suggestion.map Json.str }

def throwRecord {α} (code : ErrorCode) (param : String) (value : Json)
    (fields : List (String × FieldSpec)) (message : Option String := none) : EngineM α :=
  throw <| .env {
    code, message := message.getD s!"Invalid {param}", retryable := false,
    param := some param, value := some value,
    valid := some (.record fields) }

def throwPredicate {α} (code : ErrorCode) (param : String) (value : Json)
    (predicate : String) (expected got : Option Json)
    (message : Option String := none) : EngineM α :=
  throw <| .env {
    code, message := message.getD s!"{predicate} failed on {param}",
    retryable := false,
    param := some param, value := some value,
    valid := some (.predicate predicate expected got) }

def throwBare {α} (code : ErrorCode) (message : String)
    (retryable : Bool := false) (param : Option String := none)
    (value : Option Json := none) : EngineM α :=
  throw <| .env { code, message, retryable, param, value }

def internalError {α} (message : String) : EngineM α :=
  throwBare .internalError message

-- ── ToolResult framing (the MCP TextContent envelope) ────────────────────────

def okResult (data : Json) : Json :=
  let payload := Json.mkObj [("status", Json.str "ok"), ("data", data)]
  Json.mkObj [("content", Json.arr #[
    Json.mkObj [("type", Json.str "text"), ("text", Json.str payload.compress)]])]

def failResult (f : Failure) : Json :=
  let payload := Json.mkObj [("status", Json.str "error"), ("error", f.toJson)]
  Json.mkObj [("content", Json.arr #[
    Json.mkObj [("type", Json.str "text"), ("text", Json.str payload.compress)]]),
    ("isError", Json.bool true)]

/-- Run a handler, framing success/failure as a ToolResult. IO exceptions
    map to `internal_error`, matching the TS `wrap`'s catch-all. -/
def wrap (m : EngineM Json) : IO Json := do
  match ← (try m.run catch e => pure (.error (.env
      { code := .internalError, message := toString e, retryable := false }))) with
  | .ok data   => pure (okResult data)
  | .error f   => pure (failResult f)

end Tropical
