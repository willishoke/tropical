import Lean.Data.Json
import Tropical.Expr
import Tropical.Ir.Nodes

/-!
# Wire-program lift — port of compiler/ir/wire_program.ts (Phase 5 stage 6b)

Lift a wire `ExprNode` (engine Json wire form) to a raw resolved
`Program`: one `InputDecl` per free instance-output ref (sorted by
canonical `instance:port` key), one output `out`, and inline
`ParamDecl`s for `param`/`trigger` refs. Shape-identical to a
user-authored single-assign program; the strata pipeline accepts it
unmodified.

Pure and total over its inputs; every error string is the TS message
byte-exact (they surface as `internal_error` envelopes, the same
mapping the service relay produced when `liftWiresToInstances`
threw).
-/

namespace Tropical.Ir.WireProgram

open Lean (Json JsonNumber)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Parse (ScalarKind)

/-- JS `typeof`, for the freeRefs non-string-output message. -/
private def jsTypeof : Json → String
  | .num _ => "number"
  | .bool _ => "boolean"
  | .str _ => "string"
  | .null => "object"
  | .arr _ => "object"
  | .obj _ => "object"

/-- Render a Json value the way the TS error sites render it
    (`String(outVal)` over a JSON-parsed value): bare strings unquoted,
    numbers/bools/null via their JSON form. -/
private def jsString : Json → String
  | .str s => s
  | .null => "null"
  | j => j.compress

/-- Port of `freeRefs`: walk the wire expression and collect every
    `ref(instance, output)`, deduplicated by canonical key in
    first-encounter order. Wire-form ops carry children at `args`
    and/or `items`; other shapes aren't recursed into (the translator
    rejects them later, as TS does). -/
partial def freeRefs (expr : Json) : Except String (Array (String × String)) := do
  let mut out : Array (String × String) := #[]
  out ← walk out expr
  return out
where
  walk (acc : Array (String × String)) (e : Json) :
      Except String (Array (String × String)) := do
    match e with
    | .num _ | .bool _ | .str _ | .null => return acc
    | .arr items =>
      let mut acc := acc
      for item in items do
        acc ← walk acc item
      return acc
    | .obj _ =>
      if opOf? e == some "ref" then
        let some inst := getStrField? e "instance"
          | throw "freeRefs: ref node missing string 'instance' field"
        match getField? e "output" with
        | some (.str outName) =>
          let acc := if acc.any (fun (i, o) => i == inst && o == outName)
                     then acc else acc.push (inst, outName)
          return acc
        | outVal =>
          -- Numeric output indices are post-elaboration; wire-form is string.
          throw <| s!"freeRefs: ref({inst}, {jsString (outVal.getD .null)}) — output must be a " ++
            s!"string port name in wire-form, got {jsTypeof (outVal.getD .null)}"
      else
        let mut acc := acc
        if let some (.arr args) := getField? e "args" then
          for a in args do
            acc ← walk acc a
        if let some (.arr items) := getField? e "items" then
          for a in items do
            acc ← walk acc a
        return acc

/-- Port of `inferOutputPortType`: array-shaped wires (bare literals,
    `{op:'array'/'arrayLiteral', items}`, possibly inside the
    shape-polymorphic `delay()` auto-wrap) get an array-typed output;
    `none` means scalar. -/
partial def inferOutputPortType (expr : Json) : Option PortType :=
  match expr with
  | .arr items =>
    some (.array (.scalar .float) #[.lit ⟨items.size, 0⟩])
  | .obj _ =>
    let op := opOf? expr
    if (op == some "array" || op == some "arrayLiteral") then
      match getField? expr "items" with
      | some (.arr items) => some (.array (.scalar .float) #[.lit ⟨items.size, 0⟩])
      | _ => none
    else if op == some "delay" then
      match getField? expr "args" with
      | some (.arr args) =>
        if h : args.size = 1 then inferOutputPortType args[0] else none
      | _ => none
    else none
  | _ => none

/-- Translation state: param decls (insertion-ordered, position =
    `ParamIdx`) plus the shared expression DAG the translated ids intern into.
    CF-only — the lifted body has no reg decls. -/
private structure Ctx where
  params : Array String := #[]
  exprs : ExprArena := {}

/-- Intern a node into the ctx's DAG, returning its id and the advanced ctx. -/
private def internW (n : ENode) (ctx : Ctx) : ExprId × Ctx :=
  let (id, ex) := (eintern n).run ctx.exprs
  (id, { ctx with exprs := ex })

private def wireKeyOf (inst port : String) : String := s!"{inst}:{port}"

/-- Port of `translateExpr`: wire ExprNode Json → resolved id, interning into
    the ctx's DAG and threading the param accumulator. -/
private partial def translate (refToInput : Array (String × Nat))
    (e : Json) (ctx : Ctx) : Except String (ExprId × Ctx) := do
  match e with
  | .num n => return internW (.num n) ctx
  | .bool b => return internW (.bool b) ctx
  | .arr items =>
    let mut ctx := ctx
    let mut out : Array ExprId := #[]
    for item in items do
      let (t, ctx') ← translate refToInput item ctx
      out := out.push t
      ctx := ctx'
    return internW (.arr out) ctx
  | .str _ | .null =>
    throw s!"liftWireToProgram: invalid expr value: {e.compress}"
  | .obj _ =>
    let some op := opOf? e
      | throw "liftWireToProgram: expression missing op tag"
    let args : Array Json := match getField? e "args" with
      | some (.arr a) => a
      | _ => #[]
    -- N-ary helper: translate exactly `n` positional args.
    let nArgs (n : Nat) (ctx : Ctx) : Except String (Array ExprId × Ctx) := do
      unless args.size ≥ n do
        throw s!"liftWireToProgram: '{op}' requires {n} args, got {args.size}"
      let mut ctx := ctx
      let mut out : Array ExprId := #[]
      for i in [0:n] do
        let (t, ctx') ← translate refToInput args[i]! ctx
        out := out.push t
        ctx := ctx'
      return (out, ctx)
    match op with
    | "ref" =>
      let inst := (getStrField? e "instance").getD "undefined"
      let outName := match getField? e "output" with
        | some j => jsString j
        | none => "undefined"
      let key := wireKeyOf inst outName
      match refToInput.find? (·.1 == key) with
      | some (_, i) => return internW (.inputRef ⟨i⟩) ctx
      | none =>
        throw <| s!"liftWireToProgram: ref {key} not in freeRefSet — " ++
          "pass the same set returned by freeRefs(expr)"
    | "param" | "paramExpr" | "trigger" | "triggerParamExpr" =>
      let name := (getStrField? e "name").getD "undefined"
      let ctx := if ctx.params.contains name then ctx
                 else { ctx with params := ctx.params.push name }
      let some pi := ctx.params.idxOf? name
        | throw "liftWireToProgram: param accumulator lost a name (internal)"
      return internW (.paramRef ⟨pi⟩) ctx
    | "sampleRate" => return internW .sampleRate ctx
    | "sampleIndex" => return internW .sampleIndex ctx
    | "index" =>
      let (a, ctx) ← nArgs 2 ctx
      return internW (.index a[0]! a[1]!) ctx
    | "array" =>
      let some (Json.arr items) := getField? e "items"
        | throw s!"liftWireToProgram: 'array' requires items[], got {e.compress}"
      let mut ctx := ctx
      let mut out : Array ExprId := #[]
      for item in items do
        let (t, ctx') ← translate refToInput item ctx
        out := out.push t
        ctx := ctx'
      return internW (.arr out) ctx
    | "delay" =>
      -- CF-only: session-level `delay()` would synthesize a per-sample
      -- state register (the per-wire 1-sample latency). State has been
      -- removed from the language, so `delay()` wire ops are rejected.
      throw <|
        "liftWireToProgram: 'delay' wire op is unsupported — tropical is " ++
        "closed-form-only and has no per-sample state. Express the wire as a " ++
        "closed-form function of the time coordinate instead."
    | _ =>
      match BinaryOpTag.ofWire? op with
      | some tag =>
        let (a, ctx) ← nArgs 2 ctx
        return internW (.binary tag a[0]! a[1]!) ctx
      | none =>
      match UnaryOpTag.ofWire? op with
      | some tag =>
        let (a, ctx) ← nArgs 1 ctx
        return internW (.unary tag a[0]!) ctx
      | none =>
      match op with
      | "clamp" =>
        let (a, ctx) ← nArgs 3 ctx
        return internW (.clamp a[0]! a[1]! a[2]!) ctx
      | "select" =>
        let (a, ctx) ← nArgs 3 ctx
        return internW (.select a[0]! a[1]! a[2]!) ctx
      | "arraySet" =>
        let (a, ctx) ← nArgs 3 ctx
        return internW (.arraySet a[0]! a[1]! a[2]!) ctx
      | _ => throw s!"liftWireToProgram: unhandled wire-form op '{op}'"

/-- Port of `liftWireToProgram`: build the raw lifted `Program`.
    Returns the program AND the sorted free refs (the caller wires
    each `instance__port` input back to its source). -/
def lift (expr : Json) (synthName : String) (exprs0 : ExprArena := {}) :
    Except String (Program × Array (String × String) × ExprArena) := do
  let refs ← freeRefs expr
  -- Sort by canonical key — deterministic input order across calls.
  let sortedRefs := refs.qsort fun a b =>
    wireKeyOf a.1 a.2 < wireKeyOf b.1 b.2
  let inputDecls : Array InputDecl := sortedRefs.map fun (inst, port) =>
    -- Double-underscore separator avoids collisions with user port
    -- names; dots in instance paths flatten to underscores.
    { name := s!"{inst.replace "." "_"}__{port}" }
  let refToInput := sortedRefs.mapIdx fun i (inst, port) =>
    (wireKeyOf inst port, i)
  let outputDecl : OutputDecl := { name := "out", type? := inferOutputPortType expr }
  let (translated, ctx) ← translate refToInput expr { exprs := exprs0 }
  let prog : Program := {
    name := synthName
    typeParams := #[]
    inputs := inputDecls
    outputs := #[outputDecl]
    typeDefs := #[]
    decls := ctx.params.map (BodyDecl.param · none)
    assigns := #[{ target := .port ⟨0⟩, expr := translated }]
    binderCount := 0
    registry := #[] }
  return (prog, sortedRefs, ctx.exprs)

end Tropical.Ir.WireProgram
