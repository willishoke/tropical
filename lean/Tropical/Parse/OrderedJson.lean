import Lean.Data.Json

/-!
# Order-preserving JSON

`Lean.Json` stores objects in a `Std.TreeMap.Raw` sorted by key. `JsonV`
keeps them as insertion-ordered association arrays instead, which today
buys two things:

- **Provable recursion.** An `Array (String × JsonV)` object admits the
  `sizeOf` lemmas below (`sizeOf_lt_of_mem_snd`, `sizeOf_lt_of_getField`)
  that the codec's total decode DFS hangs its termination on; core ships
  no such lemmas for tree-map lookups. This is the load-bearing property.
- **Key order**, the original motivation (the TS `raise` adapter observed
  `Object.entries` order over several record shapes — all retired with
  the surface language). Audited 2026-07-26: order is now semantically
  inert except as the trailing tiebreak for unknown-port wires in the
  ingest's declared-port sort (`Engine/ProgramIO`), wires the compiler
  never reads. Two of the three parse entries feed this parser from
  `Lean.Json.compress` output — already key-sorted — and nothing minds.

`JsonV` is exactly `Lean.Json` with `obj : Array (String × JsonV)`.
The parser is a total lexer + pushdown fold (see the Parser section);
string/number literals are decoded by `Lean.Json.parse` on the lexed
spans, so literal fidelity is identical to `Lean.Json.parse`.

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

-- ── Parser: total lexer + pushdown fold, objects kept ordered ────────────────

section Parser

/- Ingress without fuel: the text is consumed in ITS OWN order. A single
   lexer scan whose position strictly advances (measure `utf8ByteSize −
   byteIdx`, discharged by `Pos.Raw.byteIdx_lt_byteIdx_next`) produces
   the token array; one fold over that array with an explicit stack of
   suspended containers builds the tree. Recursion depth became a data
   stack, so termination is structural over the input, and every error
   branch names a real malformed input — there is no exhaustion case.
   Leaf LITERALS (strings with their escapes, numbers) are decoded by
   `Lean.Json.parse` on the lexed span, so literal semantics — numeric
   fidelity included — are core's exactly. -/

private inductive Tok where
  | lbrack | rbrack | lbrace | rbrace | comma | colon
  | str (s : String) | num (n : JsonNumber) | bool (b : Bool) | null

private def Tok.describe : Tok → String
  | .lbrack => "'['" | .rbrack => "']'" | .lbrace => "'{'" | .rbrace => "'}'"
  | .comma => "','"  | .colon => "':'"
  | .str _ => "a string" | .num _ => "a number"
  | .bool b => toString b | .null => "null"

/-- Position just past the closing quote of a string literal, `p` sitting
    just after the opening quote. A backslash skips the char after it —
    enough to find the FIRST unescaped quote, which is also where core's
    decoder stops (a `\uXXXX` tail is hex digits, never a bare `"`). -/
private def strEnd (s : String) (p : String.Pos.Raw) :
    Except String {q : String.Pos.Raw // p.byteIdx < q.byteIdx} :=
  if hp : p.byteIdx < s.utf8ByteSize then
    have h1 := String.Pos.Raw.byteIdx_lt_byteIdx_next s p
    let c := String.Pos.Raw.get s p
    if c == '"' then
      .ok ⟨p.next s, h1⟩
    else if c == '\\' then
      have h2 := String.Pos.Raw.byteIdx_lt_byteIdx_next s (p.next s)
      match strEnd s ((p.next s).next s) with
      | .ok ⟨q, _hq⟩ => .ok ⟨q, by omega⟩
      | .error e => .error e
    else
      match strEnd s (p.next s) with
      | .ok ⟨q, _hq⟩ => .ok ⟨q, by omega⟩
      | .error e => .error e
  else
    .error s!"offset {p.byteIdx}: unterminated string"
termination_by s.utf8ByteSize - p.byteIdx
decreasing_by all_goals omega

private def isNumChar (c : Char) : Bool :=
  c.isDigit || c == '-' || c == '+' || c == 'e' || c == 'E' || c == '.'

/-- Position just past the maximal run of number-literal chars. On valid
    JSON this is exactly the span core's number parser consumes: a legal
    follower of a number is whitespace, `,`, `]`, or `}` — never in the
    run set. -/
private def numEnd (s : String) (p : String.Pos.Raw) :
    {q : String.Pos.Raw // p.byteIdx ≤ q.byteIdx} :=
  if hp : p.byteIdx < s.utf8ByteSize then
    if isNumChar (String.Pos.Raw.get s p) then
      have h1 := String.Pos.Raw.byteIdx_lt_byteIdx_next s p
      let ⟨q, _hq⟩ := numEnd s (p.next s)
      ⟨q, by omega⟩
    else ⟨p, Nat.le_refl _⟩
  else ⟨p, Nat.le_refl _⟩
termination_by s.utf8ByteSize - p.byteIdx
decreasing_by omega

/-- One scan, source order. Structural tokens push directly; string and
    number spans are delimited here and their VALUES decoded by
    `Lean.Json.parse` on the span (a span that fails to decode is a
    malformed literal). Each token carries its byte offset for errors. -/
private def lex (s : String) (p : String.Pos.Raw) (acc : Array (Tok × Nat)) :
    Except String (Array (Tok × Nat)) :=
  if _hp : p.byteIdx < s.utf8ByteSize then
    have _h1 := String.Pos.Raw.byteIdx_lt_byteIdx_next s p
    let c := String.Pos.Raw.get s p
    if c == ' ' || c == '\t' || c == '\n' || c == '\r' then
      lex s (p.next s) acc
    else if c == '[' then lex s (p.next s) (acc.push (.lbrack, p.byteIdx))
    else if c == ']' then lex s (p.next s) (acc.push (.rbrack, p.byteIdx))
    else if c == '{' then lex s (p.next s) (acc.push (.lbrace, p.byteIdx))
    else if c == '}' then lex s (p.next s) (acc.push (.rbrace, p.byteIdx))
    else if c == ',' then lex s (p.next s) (acc.push (.comma, p.byteIdx))
    else if c == ':' then lex s (p.next s) (acc.push (.colon, p.byteIdx))
    else if c == '"' then
      match strEnd s (p.next s) with
      | .error e => .error e
      | .ok ⟨q, _hq⟩ =>
        match Lean.Json.parse (String.Pos.Raw.extract s p q) with
        | .ok (Json.str v) => lex s q (acc.push (.str v, p.byteIdx))
        | _ => .error s!"offset {p.byteIdx}: invalid string literal"
    else if c == '-' || c.isDigit then
      let ⟨q, _hq⟩ := numEnd s (p.next s)
      match Lean.Json.parse (String.Pos.Raw.extract s p q) with
      | .ok (Json.num n) => lex s q (acc.push (.num n, p.byteIdx))
      | _ => .error s!"offset {p.byteIdx}: invalid number literal"
    else if c == 't' then
      if String.Pos.Raw.extract s p ⟨p.byteIdx + 4⟩ == "true" then
        lex s ⟨p.byteIdx + 4⟩ (acc.push (.bool true, p.byteIdx))
      else .error s!"offset {p.byteIdx}: unexpected input"
    else if c == 'f' then
      if String.Pos.Raw.extract s p ⟨p.byteIdx + 5⟩ == "false" then
        lex s ⟨p.byteIdx + 5⟩ (acc.push (.bool false, p.byteIdx))
      else .error s!"offset {p.byteIdx}: unexpected input"
    else if c == 'n' then
      if String.Pos.Raw.extract s p ⟨p.byteIdx + 4⟩ == "null" then
        lex s ⟨p.byteIdx + 4⟩ (acc.push (.null, p.byteIdx))
      else .error s!"offset {p.byteIdx}: unexpected input"
    else
      .error s!"offset {p.byteIdx}: unexpected character '{c}'"
  else .ok acc
termination_by s.utf8ByteSize - p.byteIdx
decreasing_by all_goals omega

/-- A suspended container: the parent awaiting the value under
    construction. The continuation, defunctionalized. -/
private inductive Ctx where
  | arr (items : Array JsonV)
  | obj (fields : Array (String × JsonV)) (key : String)

/-- What the next token must be. Together with the `Ctx` stack this is a
    zipper over the partial tree; a container's own frame is suspended
    onto the stack only when a CHILD value begins, so no state ever
    destructures a stack it might not have — every arm is live. -/
private inductive Mode where
  | rootVal                                                    -- expecting the root value
  | rootDone (v : JsonV)                                       -- root complete
  | arrFirst                                                   -- after `[`: value or `]`
  | arrNext  (items : Array JsonV)                             -- after `,`: value
  | arrAfter (items : Array JsonV)                             -- after element: `,` or `]`
  | objFirst                                                   -- after `{`: key or `}`
  | objKey   (fields : Array (String × JsonV))                 -- after `,`: key
  | objColon (fields : Array (String × JsonV)) (key : String)  -- after key: `:`
  | objVal   (fields : Array (String × JsonV)) (key : String)  -- after `:`: value
  | objAfter (fields : Array (String × JsonV))                 -- after pair: `,` or `}`

/-- Complete a value: hand it to the innermost suspended container, or
    crown it the root. Pops at most one frame — closings never cascade. -/
private def complete (v : JsonV) : List Ctx → List Ctx × Mode
  | [] => ([], .rootDone v)
  | .arr items :: rest => (rest, .arrAfter (items.push v))
  | .obj fields key :: rest => (rest, .objAfter (fields.push (key, v)))

private def scalar? : Tok → Option JsonV
  | .str s => some (.str s)
  | .num n => some (.num n)
  | .bool b => some (.bool b)
  | .null  => some .null
  | _ => none

/-- Dispatch a value-starting token; `stack` already holds the container
    the value belongs to (or is empty at root). -/
private def beginValue (stack : List Ctx) (tok : Tok) (off : Nat) :
    Except String (List Ctx × Mode) :=
  match scalar? tok with
  | some v => .ok (complete v stack)
  | none =>
    match tok with
    | .lbrack => .ok (stack, .arrFirst)
    | .lbrace => .ok (stack, .objFirst)
    | t => .error s!"offset {off}: expected a value, got {t.describe}"

private def step (st : List Ctx × Mode) (tok : Tok) (off : Nat) :
    Except String (List Ctx × Mode) :=
  let (stack, mode) := st
  match mode with
  | .rootVal => beginValue stack tok off
  | .arrFirst =>
    match tok with
    | .rbrack => .ok (complete (.arr #[]) stack)
    | t => beginValue (.arr #[] :: stack) t off
  | .arrNext items => beginValue (.arr items :: stack) tok off
  | .arrAfter items =>
    match tok with
    | .comma  => .ok (stack, .arrNext items)
    | .rbrack => .ok (complete (.arr items) stack)
    | t => .error s!"offset {off}: expected ',' or ']' in array, got {t.describe}"
  | .objFirst =>
    match tok with
    | .rbrace => .ok (complete (.obj #[]) stack)
    | .str k  => .ok (stack, .objColon #[] k)
    | t => .error s!"offset {off}: expected a key or '}' in object, got {t.describe}"
  | .objKey fields =>
    match tok with
    | .str k => .ok (stack, .objColon fields k)
    | t => .error s!"offset {off}: expected a key in object, got {t.describe}"
  | .objColon fields key =>
    match tok with
    | .colon => .ok (stack, .objVal fields key)
    | t => .error s!"offset {off}: expected ':' after key, got {t.describe}"
  | .objVal fields key => beginValue (.obj fields key :: stack) tok off
  | .objAfter fields =>
    match tok with
    | .comma  => .ok (stack, .objKey fields)
    | .rbrace => .ok (complete (.obj fields) stack)
    | t => .error s!"offset {off}: expected ',' or '}' in object, got {t.describe}"
  | .rootDone _ => .error s!"offset {off}: expected end of input, got {tok.describe}"

def parse (s : String) : Except String JsonV := do
  let toks ← lex s ⟨0⟩ #[]
  let (_, mode) ← toks.foldlM (fun st t => step st t.1 t.2)
    (([] : List Ctx), Mode.rootVal)
  match mode with
  | .rootDone v => .ok v
  | _ => .error s!"offset {s.utf8ByteSize}: unexpected end of input"

end Parser

end JsonV

end Tropical.Parse
