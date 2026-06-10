import Lean.Data.Json

/-!
# ExprNode utilities

Ports of the TS wire-expression helpers the engine layer needs:

- `validateExpr`   — structural check of the closed op set (compiler/expr.ts)
- `exprDependencies` — instance names referenced via `ref` ops (compiler/compiler.ts)
- `unwrapDelay` / `wrapInUnitDelay` — the auto-delay convention (compiler/session.ts)
- `prettyExpr`     — human-readable rendering (compiler/session.ts)

Expressions stay as raw `Json` — the wire format is the type. The Lean
session stores wires canonically pre-extraction, so `sessionSlot` /
`sessionArraySlot` reads only appear in the pretty-printer's fallback
arm (wires adopted from a service state dump can carry them when the
slot registry was malformed; they render as `delay(slot:i)`).
-/

namespace Tropical.Expr

open Lean (Json)

-- Op sets from compiler/expr.ts (validation) ----------------------------------

private def vBinaryOps : List String :=
  ["add", "sub", "mul", "div", "floorDiv", "mod",
   "lt", "lte", "gt", "gte", "eq", "neq",
   "bitAnd", "bitOr", "bitXor", "lshift", "rshift",
   "and", "or", "ldexp"]

private def vUnaryOps : List String :=
  ["neg", "abs", "not", "bitNot",
   "sqrt", "floor", "ceil", "round",
   "floatExponent", "toInt", "toBool", "toFloat"]

private def vTernaryOps : List String := ["clamp", "select"]

private def vLeafOps : List String :=
  ["input", "reg", "sampleRate", "sampleIndex",
   "delayValue", "delayRef", "nestedOutput", "nestedOut",
   "binding", "param", "paramExpr"]

-- Helpers ----------------------------------------------------------------------

def getField? (j : Json) (k : String) : Option Json :=
  (j.getObjVal? k).toOption

def getStrField? (j : Json) (k : String) : Option String :=
  match getField? j k with
  | some (.str s) => some s
  | _ => none

def opOf? (j : Json) : Option String :=
  match j with
  | .obj _ => getStrField? j "op"
  | _ => none

private def argsOf (j : Json) : Array Json :=
  match getField? j "args" with
  | some (.arr a) => a
  | _ => #[]

/-- `typeof x` for the TS error messages. -/
private def jsTypeof : Json → String
  | .null    => "object"
  | .bool _  => "boolean"
  | .num _   => "number"
  | .str _   => "string"
  | .arr _   => "object"
  | .obj _   => "object"

-- validateExpr -----------------------------------------------------------------

/-- Structural validation of an ExprNode. Mirrors compiler/expr.ts
    `validateExpr` — same checks, same message strings. -/
partial def validateExpr (node : Json) (path : String := "expr") : Except String Unit := do
  match node with
  | .num _ | .bool _ => pure ()
  | .arr items =>
    for h : i in [0:items.size] do
      validateExpr items[i] s!"{path}[{i}]"
  | .null => throw s!"{path}: expected number, boolean, array, or \{op: ...}, got object"
  | .str _ => throw s!"{path}: expected number, boolean, array, or \{op: ...}, got string"
  | .obj _ =>
    let some op := getStrField? node "op"
      | throw s!"{path}: missing or non-string 'op' field (got {(node.compress.take 100)})"
    let args := argsOf node
    let hasArgs := (getField? node "args").any (fun j => match j with | .arr _ => true | _ => false)

    let validateArgsExact (n : Nat) (usage : String) : Except String Unit := do
      if !hasArgs then
        throw s!"{path}: '{op}' requires 'args' array, got {match getField? node "args" with | none => "undefined" | some j => jsTypeof j}. Use {usage}"
      if args.size != n then
        throw s!"{path}: '{op}' requires exactly {n} arg{if n == 1 then "" else "s"}, got {args.size}"
      for h : i in [0:args.size] do
        validateExpr args[i] s!"{path}.args[{i}]"

    if vBinaryOps.contains op then
      validateArgsExact 2 s!"\{op: \"{op}\", args: [left, right]}"
    else if vUnaryOps.contains op then
      validateArgsExact 1 s!"\{op: \"{op}\", args: [x]}"
    else if vTernaryOps.contains op then
      if !hasArgs then
        throw s!"{path}: '{op}' requires 'args' array. Use \{op: \"{op}\", args: [a, b, c]}"
      if args.size != 3 then
        throw s!"{path}: '{op}' requires exactly 3 args, got {args.size}"
      for h : i in [0:args.size] do
        validateExpr args[i] s!"{path}.args[{i}]"
    else if op == "index" then
      if !hasArgs || args.size != 2 then
        throw s!"{path}: 'index' requires args: [array, index]"
      for h : i in [0:args.size] do
        validateExpr args[i] s!"{path}.args[{i}]"
    else if op == "arraySet" then
      if !hasArgs || args.size != 3 then
        throw s!"{path}: 'arraySet' requires args: [array, index, value]"
      for h : i in [0:args.size] do
        validateExpr args[i] s!"{path}.args[{i}]"
    else if op == "ref" then
      match getField? node "instance" with
      | some (.str _) => pure ()
      | f => throw s!"{path}: 'ref' requires 'instance' (string), got {match f with | none => "undefined" | some j => jsTypeof j}. Use \{op: \"ref\", instance: \"name\", output: \"port\"}"
      if (getField? node "output").isNone then
        let inst := (getStrField? node "instance").getD ""
        throw s!"{path}: 'ref' requires 'output'. Use \{op: \"ref\", instance: \"{inst}\", output: \"port_name\"}"
    else if op == "array" then
      match getField? node "items" with
      | some (.arr items) =>
        for h : i in [0:items.size] do
          validateExpr items[i] s!"{path}.items[{i}]"
      | _ => pure ()
    else if op == "arrayPack" then
      for h : i in [0:args.size] do
        validateExpr args[i] s!"{path}.args[{i}]"
    else if op == "matmul" then
      if !hasArgs || args.size != 2 then
        throw s!"{path}: 'matmul' requires args: [a, b]"
      for h : i in [0:args.size] do
        validateExpr args[i] s!"{path}.args[{i}]"
    else if op == "call" then
      match getField? node "callee" with
      | some c => validateExpr c s!"{path}.callee"
      | none => pure ()
      for h : i in [0:args.size] do
        validateExpr args[i] s!"{path}.args[{i}]"
    else if vLeafOps.contains op then
      pure ()
    else if op == "delay" then
      if !hasArgs || args.size != 1 then
        throw s!"{path}: 'delay' requires args: [expr] — the expression whose value will be read next sample"
      validateExpr args[0]! s!"{path}.args[0]"
      match getField? node "init" with
      | some (.num _) | none => pure ()
      | some j => throw s!"{path}: 'delay' init must be a number, got {jsTypeof j}"
      match getField? node "id" with
      | some (.str _) | none => pure ()
      | some j => throw s!"{path}: 'delay' id must be a string, got {jsTypeof j}"
    else if op == "zeros" || op == "ones" then
      match getField? node "shape" with
      | some (.arr _) => pure ()
      | _ => throw s!"{path}: '{op}' requires shape: number[]"
    else if op == "fill" then
      match getField? node "shape" with
      | some (.arr _) => pure ()
      | _ => throw s!"{path}: 'fill' requires shape: number[]"
      match getField? node "value" with
      | some v => validateExpr v s!"{path}.value"
      | none => throw s!"{path}: 'fill' requires value: ExprNode"
    else if op == "arrayLiteral" then
      match getField? node "values" with
      | some (.arr vs) =>
        for h : i in [0:vs.size] do
          validateExpr vs[i] s!"{path}.values[{i}]"
      | _ => throw s!"{path}: 'arrayLiteral' requires values: ExprNode[]"
    else if op == "reshape" || op == "transpose" then
      if !hasArgs || args.size < 1 then
        throw s!"{path}: '{op}' requires args: [arr]"
      validateExpr args[0]! s!"{path}.args[0]"
    else if op == "slice" then
      if !hasArgs || args.size < 1 then
        throw s!"{path}: 'slice' requires args: [arr]"
      validateExpr args[0]! s!"{path}.args[0]"
      match getField? node "start" with
      | some (.num _) => pure ()
      | _ => throw s!"{path}: 'slice' requires start: number"
      match getField? node "end" with
      | some (.num _) => pure ()
      | _ => throw s!"{path}: 'slice' requires end: number"
    else if op == "reduce" then
      if !hasArgs || args.size < 1 then
        throw s!"{path}: 'reduce' requires args: [arr]"
      validateExpr args[0]! s!"{path}.args[0]"
      match getField? node "reduce_op" with
      | some (.str _) => pure ()
      | _ => throw s!"{path}: 'reduce' requires reduce_op: string"
    else if op == "broadcastTo" then
      if !hasArgs || args.size < 1 then
        throw s!"{path}: 'broadcastTo' requires args: [arr]"
      validateExpr args[0]! s!"{path}.args[0]"
    else if op == "map" then
      if !hasArgs || args.size < 1 then
        throw s!"{path}: 'map' requires args: [arr]"
      validateExpr args[0]! s!"{path}.args[0]"
      match getField? node "callee" with
      | some c => validateExpr c s!"{path}.callee"
      | none => pure ()
    else if op == "matrix" || op == "function" then
      pure ()
    else if op == "sessionSlot" || op == "sessionArraySlot" then
      -- Compile-internal slot reads; can appear in wires adopted from a
      -- service state dump whose registry could not be fully resolved.
      pure ()
    else
      throw s!"{path}: unknown op '{op}'"

-- exprDependencies ---------------------------------------------------------------

/-- Instance names referenced via `ref` ops anywhere in the tree. The walk
    is generic over every object field and array element — a superset of
    the TS per-op child traversal that finds the same refs. -/
partial def exprDependencies (node : Json) : Array String :=
  let rec go (j : Json) (acc : Array String) : Array String :=
    match j with
    | .arr items => items.foldl (fun a e => go e a) acc
    | .obj _ =>
      if opOf? j == some "ref" then
        match getStrField? j "instance" with
        | some i => if acc.contains i then acc else acc.push i
        | none => acc
      else
        match j with
        | .obj m => m.toArray.foldl (fun a (kv : String × Json) =>
            if kv.1 == "op" then a else go kv.2 a) acc
        | _ => acc
    | _ => acc
  go node #[]

-- Auto-delay convention ----------------------------------------------------------

/-- Strip a top-level `delay()` wrapper if present. -/
def unwrapDelay (expr : Json) : Json :=
  if opOf? expr == some "delay" then
    match getField? expr "args" with
    | some (.arr #[inner]) => inner
    | _ => expr
  else expr

/-- `{op:'delay', args:[expr], init, id}` — the session-level unit delay. -/
def wrapInUnitDelay (expr : Json) (init : Json) (id : String) : Json :=
  Json.mkObj [("op", Json.str "delay"), ("args", Json.arr #[expr]),
              ("init", init), ("id", Json.str id)]

/-- Drop `id` fields from every `delay` node, recursively. The TS engine
    echoes wires through `reconstructWireDelays`, which rebuilds delays
    without their ids; `get_info` matches that canonical echo shape. -/
partial def stripDelayIds (j : Json) : Json :=
  match j with
  | .arr items => .arr (items.map stripDelayIds)
  | .obj _ =>
    let isDelay := opOf? j == some "delay"
    match j with
    | .obj m =>
      Json.mkObj <| m.toArray.toList.filterMap fun (kv : String × Json) =>
        if isDelay && kv.1 == "id" then none
        else some (kv.1, stripDelayIds kv.2)
    | _ => j
  | _ => j

-- prettyExpr ----------------------------------------------------------------------

-- Op sets from compiler/session.ts (pretty-printing; narrower than the
-- validation sets — ops outside them render via dedicated arms or throw).

private def pBinaryOps : List String :=
  ["add", "sub", "mul", "div", "floorDiv", "mod",
   "lt", "lte", "gt", "gte", "eq", "neq",
   "bitAnd", "bitOr", "bitXor", "lshift", "rshift"]

private def pUnaryOps : List String :=
  ["neg", "abs", "sin", "cos", "exp", "log", "tanh", "not", "bitNot"]

/-- Infix symbols (snake_case keys, mirroring the TS table — camelCase ops
    like `floorDiv` deliberately miss and fall back to call notation). -/
private def binaryInfix : List (String × String) :=
  [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/"),
   ("floor_div", "//"), ("mod", "%"), ("pow", "**"), ("matmul", "@"),
   ("lt", "<"), ("lte", "<="), ("gt", ">"), ("gte", ">="),
   ("eq", "=="), ("neq", "!="),
   ("bit_and", "&"), ("bit_or", "|"), ("bit_xor", "^"),
   ("lshift", "<<"), ("rshift", ">>")]

/-- Render an ExprNode as a human-readable string. `lookupOutputs`
    resolves an instance name to its output-port names (for numeric
    output indices in `ref` nodes). Mirrors compiler/session.ts
    `prettyExpr`; the Lean session has no delay-slot registry, so
    `sessionSlot` reads use the `delay(slot:i)` fallback. -/
partial def prettyExpr (node : Json) (lookupOutputs : String → Option (Array String)) : String :=
  let rec go (node : Json) : String :=
    match node with
    | .num n  => n.toString  -- JsonNumber preserves the lexical form
    | .bool b => toString b
    | .arr items => "[" ++ String.intercalate ", " (items.map go).toList ++ "]"
    | .str s => s
    | .null => "null"
    | .obj _ =>
      let op := (opOf? node).getD "?"
      let args := argsOf node
      let arg (i : Nat) : String := match args[i]? with | some a => go a | none => "?"
      if op == "ref" then
        let inst := (getStrField? node "instance").getD "?"
        let outStr := match getField? node "output" with
          | some (.num n) =>
            match lookupOutputs inst with
            | some names =>
              match n.toFloat.toUInt64.toNat, names[n.toFloat.toUInt64.toNat]? with
              | _, some nm => nm
              | _, none => n.toString
            | none => n.toString
          | some (.str s) => s
          | some j => j.compress
          | none => "undefined"
        s!"{inst}.{outStr}"
      else if op == "input" then s!"input({(getStrField? node "name").getD ""})"
      else if op == "param" then s!"param({(getStrField? node "name").getD ""})"
      else if op == "binding" then s!"${(getStrField? node "name").getD ""}"
      else if op == "sampleRate" then "sampleRate"
      else if op == "sampleIndex" then "sampleIndex"
      else if op == "float" || op == "int" || op == "bool" then
        match getField? node "value" with
        | some v => go v
        | none => "undefined"
      else if pBinaryOps.contains op then
        match (binaryInfix.lookup op) with
        | some sym => s!"({arg 0} {sym} {arg 1})"
        | none     => s!"{op}({arg 0}, {arg 1})"
      else if pUnaryOps.contains op then
        if op == "neg" then s!"-{arg 0}" else s!"{op}({arg 0})"
      else if op == "clamp" || op == "select" then
        s!"{op}(" ++ String.intercalate ", " (args.map go).toList ++ ")"
      else if op == "index" then s!"{arg 0}[{arg 1}]"
      else if op == "arraySet" then
        "array_set(" ++ String.intercalate ", " (args.map go).toList ++ ")"
      else if op == "array" then
        match getField? node "items" with
        | some (.arr items) => "[" ++ String.intercalate ", " (items.map go).toList ++ "]"
        | _ => "[]"
      else if op == "matrix" then
        s!"matrix({((getField? node "rows").getD .null).compress})"
      else if op == "delay" then
        let init := match getField? node "init" with | some j => go j | none => "0"
        s!"delay({arg 0}, {init})"
      else if op == "delayRef" then
        s!"delay_ref({(getStrField? node "id").getD ""})"
      else if op == "sessionSlot" then
        let idx := ((getField? node "index").getD .null).compress
        s!"delay(slot:{idx})"
      else if op == "sessionArraySlot" then
        let idx := ((getField? node "index").getD .null).compress
        s!"delay(array_slot:{idx})"
      else if op == "nestedOut" then
        let r := match getField? node "ref" with | some j => go j | none => "?"
        let o := match getField? node "output" with | some (.str s) => s | some j => j.compress | none => "?"
        s!"{r}.{o}"
      else
        s!"{op}(" ++ String.intercalate ", " (args.map go).toList ++ ")"
  go node

end Tropical.Expr
