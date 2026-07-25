import Lean.Data.Json

/-!
# Order-preserving JSON

`Lean.Json` stores objects in a tree map sorted by key, which destroys
the source key order. That is fine everywhere the differential harness
compares values (the comparators are key-order-insensitive), but the
TS `raise` adapter's semantics *observe* key order: `Object.entries`
over `instanceDecl.inputs` / `type_args`, `tag.payload`, `match.arms`,
and `let.bind` iterates keys in JSON-source insertion order, and raise
turns several of those records into **arrays**, where order is
load-bearing. A faithful Lean port therefore needs a JSON value type
whose objects are ordered association arrays.

`JsonV` is exactly `Lean.Json` with `obj : Array (String × JsonV)`.
The parser is a line-for-line adaptation of `Lean.Json.Parser`
(string/number parsing is reused from it verbatim, so numeric fidelity
is identical to `Lean.Json.parse`).

Known divergence from JS, documented rather than replicated: JS object
iteration hoists integer-like keys ("0", "1", …) ahead of string keys
in ascending numeric order. No tropical record (ports, arms, inputs,
binders) uses integer-like keys, so `JsonV` keeps pure insertion order.
-/

namespace Tropical.Parse

open Lean (Json JsonNumber)

/-- Termination workhorse for folds over `Array (α × β)` fields (ordered
    objects, `let` binders, tag payloads): membership of the pair bounds
    the second component. `Array.sizeOf_lt_of_mem` alone stops at the
    pair; this adds the projection step. -/
theorem sizeOf_lt_of_mem_snd {α β} [SizeOf α] [SizeOf β]
    {a : α} {b : β} {ps : Array (α × β)} (h : (a, b) ∈ ps) :
    sizeOf b < sizeOf ps := by
  have := Array.sizeOf_lt_of_mem h
  simp at this
  omega

/-- Indexing form of `Array.sizeOf_lt_of_mem`. -/
theorem sizeOf_lt_of_getElem {α} [SizeOf α] {as : Array α} {i : Nat}
    (h : i < as.size) : sizeOf as[i] < sizeOf as :=
  Array.sizeOf_lt_of_mem (Array.getElem_mem h)

/-- `Option`-level bound for recursion through `as[i]?` (a missing
    element still decreases, because an array's size is at least 2). -/
theorem sizeOf_getElem?_le {α} [SizeOf α] (as : Array α) (i : Nat) :
    sizeOf as[i]? ≤ sizeOf as := by
  match h : as[i]? with
  | none => cases as; simp
  | some a =>
    have := Array.sizeOf_lt_of_mem (Array.mem_of_getElem? h)
    simp
    omega

inductive JsonV where
  | null
  | bool (b : Bool)
  | num (n : JsonNumber)
  | str (s : String)
  | arr (elems : Array JsonV)
  | obj (fields : Array (String × JsonV))
deriving Inhabited, Repr, BEq

namespace JsonV

-- ── Accessors ────────────────────────────────────────────────────────────────

def getField? : JsonV → String → Option JsonV
  | .obj fields, k => (fields.find? (·.1 == k)).map (·.2)
  | _, _ => none

/-- A field's value is smaller than the object it came from — the fact
    that lets decoders recurse through `getField?` under a `sizeOf`
    termination measure. -/
theorem sizeOf_lt_of_getField {j v : JsonV} {k : String}
    (h : j.getField? k = some v) : sizeOf v < sizeOf j := by
  match j with
  | .null | .bool _ | .num _ | .str _ | .arr _ => simp [getField?] at h
  | .obj fields =>
    simp only [getField?, Option.map_eq_some_iff] at h
    obtain ⟨⟨a, b⟩, hf, rfl⟩ := h
    have := sizeOf_lt_of_mem_snd (Array.mem_of_find?_eq_some hf)
    simp
    omega

/-- `Option`-level bound for recursion through `getField?` itself (a
    missing field still decreases against any `JsonV`). -/
theorem sizeOf_getField_le (j : JsonV) (k : String) :
    sizeOf (j.getField? k) ≤ sizeOf j := by
  match h : j.getField? k with
  | none => cases j <;> simp <;> omega
  | some v =>
    have := sizeOf_lt_of_getField h
    simp
    omega

def getStr? (j : JsonV) (k : String) : Option String :=
  match j.getField? k with
  | some (.str s) => some s
  | _ => none

def opOf? (j : JsonV) : Option String := j.getStr? "op"

def isObj : JsonV → Bool
  | .obj _ => true
  | _ => false

/-- Object key set, in insertion order. -/
def keys : JsonV → Array String
  | .obj fields => fields.map (·.1)
  | _ => #[]

-- ── Conversion to Lean.Json (for printing; key order is surrendered) ─────────

def toJson : JsonV → Json
  | .null => Json.null
  | .bool b => Json.bool b
  | .num n => Json.num n
  | .str s => Json.str s
  | .arr elems => Json.arr (elems.attach.map fun ⟨x, _⟩ => toJson x)
  | .obj fields =>
    Json.mkObj (fields.attach.toList.map fun ⟨(k, v), _⟩ => (k, toJson v))
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem ‹_ ∈ elems›; simp; omega
  · have := sizeOf_lt_of_mem_snd ‹_ ∈ fields›; simp; omega

-- ── JS-flavoured rendering (for error-message parity with TS) ────────────────

/-- `JSON.stringify`-compatible compact rendering. Unlike
    `Json.compress`, object keys keep source order — which is what
    `JSON.stringify` does, and what the TS error strings embed. -/
def compress : JsonV → String
  | .null => "null"
  | .bool b => toString b
  | .num n => toString n
  | .str s => (Json.str s).compress
  | .arr elems =>
    "[" ++ String.intercalate "," (elems.attach.map fun ⟨x, _⟩ => compress x).toList ++ "]"
  | .obj fields =>
    "{" ++ String.intercalate ","
      (fields.attach.map fun ⟨(k, v), _⟩ =>
        s!"{(Json.str k).compress}:{compress v}").toList ++ "}"
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem ‹_ ∈ elems›; simp; omega
  · have := sizeOf_lt_of_mem_snd ‹_ ∈ fields›; simp; omega

/-- JS `String(x)` for the value positions TS stringifies into names and
    error messages (`String(node.output)`, `String(obj.op)`). Array/object
    cases fall back to `compress`; JS would render "1,2" / "[object
    Object]" there, but no caller reaches them on well-formed input. -/
def jsString : JsonV → String
  | .str s => s
  | .num n => toString n
  | .bool b => toString b
  | .null => "null"
  | j => j.compress

/-- `JSON.stringify(x)` where `x` may be a missing field: JS renders
    `undefined` (bare, unquoted) inside template literals. -/
def stringifyOpt : Option JsonV → String
  | some j => j.compress
  | none => "undefined"

-- ── Parser (adapted from Lean.Json.Parser, objects kept ordered) ─────────────

section Parser

open Std.Internal.Parsec
open Std.Internal.Parsec.String
open Lean.Json.Parser (lookahead)

/-- Aliases dodging the collision with the `JsonV.num`/`JsonV.str`
    constructors in scope. -/
private def pNum : Parser JsonNumber := Lean.Json.Parser.num
private def pStr : Parser String := Lean.Json.Parser.str

/- Deliberately `partial`: these recurse on the parser STATE, not on a
   JsonV — the discharging measure would be "remaining input shrinks",
   a fact about `Std.Internal.Parsec` internals that core does not
   expose. Every structural fold in this module is total; only the
   three parser combinators below are exempt. -/
mutual

private partial def arrayCore (acc : Array JsonV) : Parser (Array JsonV) := do
  let hd ← anyCore
  let acc' := acc.push hd
  let c ← any
  if c == ']' then
    ws
    return acc'
  else if c == ',' then
    ws
    arrayCore acc'
  else
    fail "unexpected character in array"

private partial def objectCore (acc : Array (String × JsonV)) :
    Parser (Array (String × JsonV)) := do
  lookahead (fun c => c == '"') "\""; skip
  let k ← pStr; ws
  lookahead (fun c => c == ':') ":"; skip; ws
  let v ← anyCore
  let c ← any
  if c == '}' then
    ws
    return acc.push (k, v)
  else if c == ',' then
    ws
    objectCore (acc.push (k, v))
  else
    fail "unexpected character in object"

private partial def anyCore : Parser JsonV := do
  let c ← peek!
  if c == '[' then
    skip; ws
    let c ← peek!
    if c == ']' then
      skip; ws
      return .arr #[]
    else
      return .arr (← arrayCore #[])
  else if c == '{' then
    skip; ws
    let c ← peek!
    if c == '}' then
      skip; ws
      return .obj #[]
    else
      return .obj (← objectCore #[])
  else if c == '"' then
    skip
    let s ← pStr
    ws
    return .str s
  else if c == 'f' then
    skipString "false"; ws
    return .bool false
  else if c == 't' then
    skipString "true"; ws
    return .bool true
  else if c == 'n' then
    skipString "null"; ws
    return .null
  else if c == '-' || ('0' <= c && c <= '9') then
    let n ← pNum
    ws
    return .num n
  else
    fail "unexpected input"

end

private def anyWithEof : Parser JsonV := do
  ws
  let res ← anyCore
  eof
  return res

def parse (s : String) : Except String JsonV :=
  Parser.run anyWithEof s

end Parser

end JsonV

end Tropical.Parse
