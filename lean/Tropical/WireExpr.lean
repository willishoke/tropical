import Tropical.Ir.Nodes
import Tropical.Parse.OrderedJson

/-!
# WireExpr — the typed session wire grammar

The wire-expression grammar as a closed inductive, replacing raw
`Lean.Json` in the session store. The constructor set is derived from
the READERS — what the two lowerings (`Engine.wireExprToResolved`,
`Ir.WireProgram.translate`) can compile, plus the forms other parts of
the engine legitimately construct or tolerate:

- `broadcastTo` — constructed by the wiring adapter (`adaptInputExpr`)
  on scalar→array connections; still refused at both lowerings.
- `input` / `nestedOut` — export-file forms (`export_program` writes
  them; loading such a file stores them); refused at both lowerings.
- `sessionSlot` / `sessionArraySlot` — service-state-dump legacy reads
  (the alias quotient consumes `sessionArraySlot`; both pretty-print).

The DECODER is the refusal site: state ops (`delay`, `reg`,
`delayValue`, `delayRef`) die here with the closed-form-only message,
and everything else — retired combinators (`fold`, `scan`), the
TS-era array/functional ops (`matmul`, `map`, `call`, `function`,
`matrix`, `zeros`, `ones`, `fill`, `reshape`, `transpose`, `slice`,
`reduce`, `arrayPack`), `binding`, `nestedOutput` — dies with
`unknown op '<op>'` (the format the retirement gates pin). The grammar
you can spell is the language that compiles — this module makes that
sentence true at the wire layer, where it previously wasn't (the old
`validateExpr` accepted ~20 ops no lowering could compile).

Alias collapses at decode (round-trips canonicalize):
- `{op:'array'|'arrayLiteral', items|values}` and bare JSON arrays →
  `arr`.
- `paramExpr` → `param`, `triggerParamExpr` → `trigger` (this also
  makes them compile uniformly on both lowering paths; previously the
  session path refused what the lift path accepted).
- `sampleClock` / `sample_clock` / `clock` → `clock`.
- `array_set` / `arraySet` → `arraySet`.

Decoding is total over `JsonV` (the array-backed twin with the
`sizeOf` lemmas); the `Lean.Json` entry reparses through `JsonV`
(`Json → compress → JsonV.parse`), confining the tree-map opacity of
`Lean.Json.obj` to one boundary hop that is semantically the identity.
-/

namespace Tropical

open Lean (Json JsonNumber)
open Tropical.Ir (BinaryOpTag UnaryOpTag)
open Tropical.Parse (JsonV)

/-- A `ref`/`nestedOut` output designator: port name, or positional
    index (the save path emits indices for dac assigns). The index
    keeps its `JsonNumber` so lexical form and sign survive round-trips
    (negative indices reproduce today's out-of-range refusals rather
    than wrapping). -/
inductive RefOut where
  | name (s : String)
  | index (n : JsonNumber)
deriving BEq, Repr, Inhabited

inductive WireExpr where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array WireExpr)
  | ref (inst : String) (output : RefOut)
  | param (name : String)
  | trigger (name : String)
  | binary (tag : BinaryOpTag) (l r : WireExpr)
  | unary (tag : UnaryOpTag) (a : WireExpr)
  | clamp (a b c : WireExpr)
  | select (a b c : WireExpr)
  | index (a b : WireExpr)
  | arraySet (a b c : WireExpr)
  | sampleRate
  | sampleIndex
  | clock
  | broadcastTo (a : WireExpr) (shape : Array Nat)
  | input (name : String)
  | nestedOut (ref : String) (output : RefOut)
  | sessionSlot (idx : Nat)
  | sessionArraySlot (idx : Nat) (size : Option Nat)
deriving BEq, Repr, Inhabited

namespace WireExpr

/-- The wire-format op name (error messages; `unsupported op '<op>'`
    parity at the lowerings' refusal arms). -/
def opName : WireExpr → String
  | .num _ => "num" | .bool _ => "bool" | .arr _ => "array"
  | .ref .. => "ref" | .param _ => "param" | .trigger _ => "trigger"
  | .binary tag .. => tag.wire | .unary tag _ => tag.wire
  | .clamp .. => "clamp" | .select .. => "select"
  | .index .. => "index" | .arraySet .. => "arraySet"
  | .sampleRate => "sampleRate" | .sampleIndex => "sampleIndex"
  | .clock => "clock" | .broadcastTo .. => "broadcastTo"
  | .input _ => "input" | .nestedOut .. => "nestedOut"
  | .sessionSlot _ => "sessionSlot" | .sessionArraySlot .. => "sessionArraySlot"

-- ── Encoder (the canonical wire-format Json) ─────────────────────────────────

private def refOutJson : RefOut → Json
  | .name s => Json.str s
  | .index n => Json.num n

private def opNode (op : String) (fields : List (String × Json) := []) : Json :=
  Json.mkObj (("op", Json.str op) :: fields)

private def opArgs (op : String) (args : Array Json) : Json :=
  opNode op [("args", Json.arr args)]

def toJson : WireExpr → Json
  | .num n => Json.num n
  | .bool b => Json.bool b
  | .arr items => Json.arr (items.attach.map fun ⟨x, _⟩ => toJson x)
  | .ref inst output =>
    opNode "ref" [("instance", Json.str inst), ("output", refOutJson output)]
  | .param name => opNode "param" [("name", Json.str name)]
  | .trigger name => opNode "trigger" [("name", Json.str name)]
  | .binary tag l r => opArgs tag.wire #[toJson l, toJson r]
  | .unary tag a => opArgs tag.wire #[toJson a]
  | .clamp a b c => opArgs "clamp" #[toJson a, toJson b, toJson c]
  | .select a b c => opArgs "select" #[toJson a, toJson b, toJson c]
  | .index a b => opArgs "index" #[toJson a, toJson b]
  | .arraySet a b c => opArgs "arraySet" #[toJson a, toJson b, toJson c]
  | .sampleRate => opNode "sampleRate"
  | .sampleIndex => opNode "sampleIndex"
  | .clock => opNode "clock"
  | .broadcastTo a shape =>
    opNode "broadcastTo" [("args", Json.arr #[toJson a]),
      ("shape", Json.arr (shape.map fun n => Lean.toJson n))]
  | .input name => opNode "input" [("name", Json.str name)]
  | .nestedOut r output =>
    opNode "nestedOut" [("ref", Json.str r), ("output", refOutJson output)]
  | .sessionSlot idx => opNode "sessionSlot" [("index", Lean.toJson idx)]
  | .sessionArraySlot idx size =>
    opNode "sessionArraySlot" <|
      [("index", Lean.toJson idx)]
      ++ (match size with | some s => [("size", Lean.toJson s)] | none => [])
termination_by e => sizeOf e
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

-- ── Queries (the walkers the raw-Json helpers used to be) ────────────────────

/-- Instance names referenced via `ref`, first-encounter order
    (replaces `Expr.exprDependencies` / `Lowering.collectInstanceRefs`). -/
def depsInto (acc : Array String) : WireExpr → Array String
  | .ref inst _ => if acc.contains inst then acc else acc.push inst
  | .arr items => items.attach.foldl (fun a ⟨x, _⟩ => depsInto a x) acc
  | .binary _ l r => depsInto (depsInto acc l) r
  | .unary _ a | .broadcastTo a _ => depsInto acc a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    depsInto (depsInto (depsInto acc a) b) c
  | .index a b => depsInto (depsInto acc a) b
  | _ => acc
termination_by e => sizeOf e
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

def deps (e : WireExpr) : Array String := depsInto #[] e

/-- The first sub-expression no lowering can compile, if any.

    Five constructors are in the grammar because something OTHER than an
    agent builds them — `broadcastTo` from the wiring adapter, `input` /
    `nestedOut` from `export_program`'s serializer, `sessionSlot` /
    `sessionArraySlot` from legacy state-dump reads — and BOTH lowerings
    (`Engine.wireExprToResolved`, `Ir.WireProgram.translate`) refuse all
    five. So "it decodes" was never the same as "it compiles", and a wire
    carrying one used to reach the session store and detonate at the next
    `syncCompile` — poisoning the session for every later mutation and
    getting persisted by `save` into a file that can never load.

    Every boundary that ADMITS a wire checks this, which is what makes the
    decoder's promise true: the grammar you can spell is the language that
    compiles. -/
def uncompilableOp? : WireExpr → Option String
  | .broadcastTo .. => some "broadcastTo"
  | .input _ => some "input"
  | .nestedOut .. => some "nestedOut"
  | .sessionSlot _ => some "sessionSlot"
  | .sessionArraySlot .. => some "sessionArraySlot"
  | .arr items => items.attach.findSome? fun ⟨x, _⟩ => uncompilableOp? x
  | .binary _ l r => (uncompilableOp? l).orElse fun _ => uncompilableOp? r
  | .unary _ a => uncompilableOp? a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    ((uncompilableOp? a).orElse fun _ => uncompilableOp? b).orElse fun _ =>
      uncompilableOp? c
  | .index a b => (uncompilableOp? a).orElse fun _ => uncompilableOp? b
  | _ => none
termination_by e => sizeOf e
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

/-- The refusal message for `uncompilableOp?`, shared by every admitting
    boundary so the wording does not drift. -/
def uncompilableMessage (path op : String) : String :=
  s!"{path}: '{op}' is not a wire a patch can carry — it is a form the engine " ++
  "builds internally (wiring adapter / export serializer / legacy state dump) " ++
  "and no lowering compiles. Wire an instance output, a param, or a closed-form " ++
  "expression over them instead."

/-- `param`/`trigger` names, first-encounter order (replaces
    `Engine.collectWireParams`). -/
def paramNamesInto (acc : Array String) : WireExpr → Array String
  | .param name | .trigger name =>
    if acc.contains name then acc else acc.push name
  | .arr items => items.attach.foldl (fun a ⟨x, _⟩ => paramNamesInto a x) acc
  | .binary _ l r => paramNamesInto (paramNamesInto acc l) r
  | .unary _ a | .broadcastTo a _ => paramNamesInto acc a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    paramNamesInto (paramNamesInto (paramNamesInto acc a) b) c
  | .index a b => paramNamesInto (paramNamesInto acc a) b
  | _ => acc
termination_by e => sizeOf e
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

/-- Note: unlike `deps`, the collector the session root uses appends
    duplicates freely and dedups at the call site; this one dedups
    inline — same resulting name set and order. -/
def paramNames (e : WireExpr) : Array String := paramNamesInto #[] e

/-- Does the wire need lifting into an anonymous instance? True exactly
    when an array literal appears anywhere (replaces
    `Lowering.needsWireLift`: bare arrays, `array`/`arrayLiteral` ops,
    and recursion through `args`/`items` all collapse to "contains
    `arr`"). -/
def needsLift : WireExpr → Bool
  | .arr _ => true
  | .binary _ l r => needsLift l || needsLift r
  | .unary _ a | .broadcastTo a _ => needsLift a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    needsLift a || needsLift b || needsLift c
  | .index a b => needsLift a || needsLift b
  | _ => false

-- ── Pretty-printing (list_wiring / get_info) ─────────────────────────────────

/-- Infix symbols (the old `prettyExpr` table; camelCase bit ops and
    `floorDiv` deliberately miss and render in call notation). -/
private def infixSym : BinaryOpTag → Option String
  | .add => some "+" | .sub => some "-" | .mul => some "*" | .div => some "/"
  | .mod => some "%"
  | .lt => some "<" | .lte => some "<=" | .gt => some ">" | .gte => some ">="
  | .eq => some "==" | .neq => some "!="
  | .lshift => some "<<" | .rshift => some ">>"
  | _ => none

private def refOutStr (lookupOutputs : String → Option (Array String))
    (inst : String) : RefOut → String
  | .name s => s
  | .index n =>
    match lookupOutputs inst with
    | some names => (names[n.toFloat.toUInt64.toNat]?).getD n.toString
    | none => n.toString

/-- Render a wire expression as a human-readable string (the old
    `Expr.prettyExpr`, now total and exhaustive over the grammar).
    `lookupOutputs` resolves an instance name to its output-port names
    for numeric `ref` outputs. -/
def pretty (lookupOutputs : String → Option (Array String)) : WireExpr → String
  | .num n => n.toString
  | .bool b => toString b
  | .arr items =>
    "[" ++ String.intercalate ", " (items.attach.map fun ⟨x, _⟩ =>
      pretty lookupOutputs x).toList ++ "]"
  | .ref inst output => s!"{inst}.{refOutStr lookupOutputs inst output}"
  | .param name => s!"param({name})"
  | .trigger name => s!"trigger({name})"
  | .binary tag l r =>
    let ls := pretty lookupOutputs l
    let rs := pretty lookupOutputs r
    match infixSym tag with
    | some sym => s!"({ls} {sym} {rs})"
    | none => s!"{tag.wire}({ls}, {rs})"
  | .unary tag a =>
    let as' := pretty lookupOutputs a
    if tag == .neg then s!"-{as'}" else s!"{tag.wire}({as'})"
  | .clamp a b c =>
    s!"clamp({pretty lookupOutputs a}, {pretty lookupOutputs b}, {pretty lookupOutputs c})"
  | .select a b c =>
    s!"select({pretty lookupOutputs a}, {pretty lookupOutputs b}, {pretty lookupOutputs c})"
  | .index a b => s!"{pretty lookupOutputs a}[{pretty lookupOutputs b}]"
  | .arraySet a b c =>
    s!"array_set({pretty lookupOutputs a}, {pretty lookupOutputs b}, {pretty lookupOutputs c})"
  | .sampleRate => "sampleRate"
  | .sampleIndex => "sampleIndex"
  | .clock => "clock()"
  | .broadcastTo a _ => s!"broadcastTo({pretty lookupOutputs a})"
  | .input name => s!"input({name})"
  | .nestedOut r output => s!"{r}.{refOutStr lookupOutputs r output}"
  | .sessionSlot idx => s!"delay(slot:{idx})"
  | .sessionArraySlot idx _ => s!"delay(array_slot:{idx})"
termination_by e => sizeOf e
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

-- ── Decoder (the refusal site) ───────────────────────────────────────────────

private def jsTypeof : JsonV → String
  | .null => "object" | .bool _ => "boolean" | .num _ => "number"
  | .str _ => "string" | .arr _ => "object" | .obj _ => "object"

/-- State ops die at decode with the closed-form-only message (formerly
    `WireProgram.translate`'s refusal, now moved to ingest). -/
private def stateOps : List String := ["delay", "reg", "delayValue", "delayRef"]

private def reqField (path : String) (j : JsonV) (k : String) (usage : String) :
    Except String {v : JsonV // sizeOf v < sizeOf j} :=
  match hf : j.getField? k with
  | some v => pure ⟨v, Tropical.Parse.JsonV.sizeOf_lt_of_getField hf⟩
  | none => throw s!"{path}: missing field '{k}'. Use {usage}"

private def argsExact (path : String) (j : JsonV) (op : String) (n : Nat) :
    Except String {items : Array JsonV // sizeOf items < sizeOf j} := do
  match hf : j.getField? "args" with
  | some (.arr items) =>
    if items.size = n then
      pure ⟨items, by
        have := Tropical.Parse.JsonV.sizeOf_lt_of_getField hf
        simp at this; omega⟩
    else
      throw s!"{path}: '{op}' requires exactly {n} arg{if n == 1 then "" else "s"}, got {items.size}"
  | some v =>
    throw s!"{path}: '{op}' requires 'args' array, got {jsTypeof v}"
  | none =>
    throw s!"{path}: '{op}' requires 'args' array, got undefined"

private def elemBound {items : Array JsonV} {j : JsonV}
    (h : sizeOf items < sizeOf j) {i : Nat} (hi : i < items.size) :
    sizeOf items[i] < sizeOf j :=
  Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem hi) h

private def reqNat (path : String) (j : JsonV) (k : String) : Except String Nat :=
  match j.getField? k with
  | some (.num n) => pure n.toFloat.toUInt64.toNat
  | _ => throw s!"{path}: '{k}' must be a number"

/-- Decode a wire expression off ordered JSON. Total by descent on
    `sizeOf j` (the codec-decoder pattern). Threads `path` for the
    error-position prefix the old `validateExpr` messages carried. -/
def ofJsonV (j : JsonV) (path : String := "expr") : Except String WireExpr := do
  match hj : j with
  | .num n => pure (.num n)
  | .bool b => pure (.bool b)
  | .arr items =>
    let out ← items.attach.zipIdx.mapM fun (⟨x, _⟩, i) =>
      ofJsonV x s!"{path}[{i}]"
    pure (.arr out)
  | .null => throw s!"{path}: expected number, boolean, array, or \{op: ...}, got object"
  | .str _ => throw s!"{path}: expected number, boolean, array, or \{op: ...}, got string"
  | .obj _ =>
    let some op := j.opOf?
      | throw s!"{path}: missing or non-string 'op' field (got {(j.compress.take 100)})"
    if let some tag := BinaryOpTag.ofWire? op then
      let ⟨a, ha⟩ ← argsExact path j op 2
      if h2 : a.size = 2 then
        pure (.binary tag
          (← ofJsonV (a[0]'(by omega)) s!"{path}.args[0]")
          (← ofJsonV (a[1]'(by omega)) s!"{path}.args[1]"))
      else throw s!"{path}: internal arity"
    else if let some tag := UnaryOpTag.ofWire? op then
      let ⟨a, ha⟩ ← argsExact path j op 1
      if h1 : a.size = 1 then
        pure (.unary tag (← ofJsonV (a[0]'(by omega)) s!"{path}.args[0]"))
      else throw s!"{path}: internal arity"
    else match op with
    | "clamp" | "select" | "arraySet" | "array_set" =>
      let ⟨a, ha⟩ ← argsExact path j op 3
      if h3 : a.size = 3 then
        let x ← ofJsonV (a[0]'(by omega)) s!"{path}.args[0]"
        let y ← ofJsonV (a[1]'(by omega)) s!"{path}.args[1]"
        let z ← ofJsonV (a[2]'(by omega)) s!"{path}.args[2]"
        pure (match op with
          | "clamp" => .clamp x y z
          | "select" => .select x y z
          | _ => .arraySet x y z)
      else throw s!"{path}: internal arity"
    | "index" =>
      let ⟨a, ha⟩ ← argsExact path j op 2
      if h2 : a.size = 2 then
        pure (.index
          (← ofJsonV (a[0]'(by omega)) s!"{path}.args[0]")
          (← ofJsonV (a[1]'(by omega)) s!"{path}.args[1]"))
      else throw s!"{path}: internal arity"
    | "ref" =>
      let inst ← match j.getField? "instance" with
        | some (.str s) => pure s
        | f => throw <| s!"{path}: 'ref' requires 'instance' (string), got " ++
            s!"{match f with | none => "undefined" | some v => jsTypeof v}. " ++
            "Use {op: \"ref\", instance: \"name\", output: \"port\"}"
      match j.getField? "output" with
      | some (.str s) => pure (.ref inst (.name s))
      | some (.num n) => pure (.ref inst (.index n))
      | some v => throw s!"{path}: 'ref' output must be a string port name or index, got {jsTypeof v}"
      | none =>
        throw s!"{path}: 'ref' requires 'output'. Use \{op: \"ref\", instance: \"{inst}\", output: \"port_name\"}"
    | "param" | "paramExpr" | "trigger" | "triggerParamExpr" =>
      let some name := j.getStr? "name"
        | throw s!"{path}: '{op}' requires 'name' (string)"
      pure (if op == "param" || op == "paramExpr" then .param name else .trigger name)
    | "array" | "arrayLiteral" =>
      let itemsKey := if (j.getField? "items").isSome then "items" else "values"
      match hf : j.getField? itemsKey with
      | some (.arr items) =>
        let out ← items.attach.zipIdx.mapM fun (⟨x, _⟩, i) =>
          ofJsonV x s!"{path}.{itemsKey}[{i}]"
        pure (.arr out)
      | _ => throw s!"{path}: '{op}' requires items: ExprNode[]"
    | "sampleRate" => pure .sampleRate
    | "sampleIndex" => pure .sampleIndex
    | "clock" | "sampleClock" | "sample_clock" => pure .clock
    | "broadcastTo" =>
      let ⟨a, ha⟩ ← argsExact path j op 1
      let shape ← match j.getField? "shape" with
        | some (.arr dims) => dims.mapM fun d => match d with
          | .num n => pure n.toFloat.toUInt64.toNat
          | _ => throw s!"{path}: 'broadcastTo' shape must be number[]"
        | _ => throw s!"{path}: 'broadcastTo' requires shape: number[]"
      if h1 : a.size = 1 then
        pure (.broadcastTo (← ofJsonV (a[0]'(by omega)) s!"{path}.args[0]") shape)
      else throw s!"{path}: internal arity"
    | "input" =>
      let some name := j.getStr? "name"
        | throw s!"{path}: 'input' requires 'name' (string)"
      pure (.input name)
    | "nestedOut" =>
      let some r := j.getStr? "ref"
        | throw s!"{path}: 'nestedOut' requires 'ref' (string instance name)"
      match j.getField? "output" with
      | some (.str s) => pure (.nestedOut r (.name s))
      | some (.num n) => pure (.nestedOut r (.index n))
      | _ => throw s!"{path}: 'nestedOut' requires 'output' (string or index)"
    | "sessionSlot" => pure (.sessionSlot (← reqNat path j "index"))
    | "sessionArraySlot" =>
      let idx ← reqNat path j "index"
      let size ← match j.getField? "size" with
        | some (.num n) => pure (some n.toFloat.toUInt64.toNat)
        | some _ => throw s!"{path}: 'sessionArraySlot' size must be a number"
        | none => pure none
      pure (.sessionArraySlot idx size)
    | _ =>
      if stateOps.contains op then
        throw <| s!"{path}: '{op}' is unsupported — tropical is closed-form-only " ++
          "and has no per-sample state. Express the wire as a closed-form " ++
          "function of the time coordinate instead."
      else
        throw s!"{path}: unknown op '{op}'"
termination_by sizeOf j
decreasing_by
  all_goals first
    | (subst hj
       exact Nat.lt_trans (Tropical.Parse.sizeOf_lt_of_getElem (by omega)) ha)
    | (have := Tropical.Parse.JsonV.sizeOf_lt_of_getField hf
       have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp_all <;> omega)
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp_all <;> omega)

/-- Decode off `Lean.Json` — the MCP tool-argument door. Reparses
    through `JsonV` (semantically the identity; `Lean.Json.obj`'s
    tree-map admits no total structural walk, so the hop is confined
    here). -/
def ofJson (j : Json) (path : String := "expr") : Except String WireExpr := do
  match Tropical.Parse.JsonV.parse j.compress with
  | .error e => throw s!"{path}: internal reparse failure: {e}"
  | .ok jv => ofJsonV jv path

end WireExpr

end Tropical
