import Tropical.Parse.Surface.Expr
import Tropical.Parse.Nodes

/-!
# Body-item parsers (port of `compiler/parse/statements.ts`)

The leaf decl/assign parsers — reg, param, next, dac.out, and the
instance-vs-output-assign disambiguation. None of these recurse into the
declaration layer, so they live here as plain defs; `parseBodyItem` /
`parseProgram` (the body↔program mutual knot) live in `Declarations.lean`,
which dispatches to these.
-/

namespace Tropical.Parse.Surface

open Tropical.Parse (ParsedExpr BodyDecl BodyAssign TypeDef NextTargetKind)
open Lean (JsonNumber)

/-- A parsed body item, sorted into the block's `decls`/`assigns` (and
    `typeDefs`, produced by the declaration layer) by `parseBodyItems`. -/
inductive BodyItem where
  | decl (d : BodyDecl)
  | assign (a : BodyAssign)
  | typeDef (t : TypeDef)

private def isCapitalized (s : String) : Bool :=
  match s.toList with
  | c :: _ => c.isUpper
  | [] => false

-- ── Decls ────────────────────────────────────────────────────────────────────

/-- `reg name [: type] = init` -/
def parseRegDecl : P BodyDecl := do
  let _ ← consume .kreg "reg keyword"
  let name := (← consume .ident "reg name").sval
  let type? ← (do
    if (← eat .colon).isSome then pure (some (← consume .ident "reg type name").sval)
    else pure none)
  let _ ← consume .assign "reg `=` before init"
  let init ← parseTopExpr
  pure (.reg name init type?)

/-- `param name: smoothed [= default]` (default must be a number literal). -/
def parseParamDecl : P BodyDecl := do
  let _ ← consume .kparam "param keyword"
  let name := (← consume .ident "param name").sval
  let _ ← consume .colon "param `:` before kind"
  let kindTok ← consume .ident "param kind (smoothed)"
  if kindTok.sval != "smoothed" then
    throw s!"param kind must be 'smoothed', got '{kindTok.sval}'"
  let value? ← (do
    if (← eat .assign).isSome then
      match (← parseTopExpr) with
      | .num n => pure (some n)
      | _ => throw "param default must be a number literal"
    else pure none)
  pure (.param name value?)

-- ── Assigns ──────────────────────────────────────────────────────────────────

/-- `next name = expr` (target kind is always `reg`). -/
def parseNextUpdate : P BodyAssign := do
  let _ ← consume .knext "next keyword"
  let name := (← consume .ident "next target name").sval
  let _ ← consume .assign "next `=` before expression"
  let expr ← parseTopExpr
  pure (.next .reg name expr)

/-- `dac.out = expr` — the boundary-leaf wire. -/
def parseDacOutAssign : P BodyAssign := do
  let dacTok ← consume .ident "dac"
  if dacTok.sval != "dac" then throw "expected 'dac' for boundary-leaf wire"
  let _ ← consume .dot "dac `.`"
  let portTok ← consume .ident "dac port name"
  if portTok.sval != "out" then
    throw s!"dac has only one output port: 'out'. Got '{portTok.sval}'"
  let _ ← consume .assign "dac.out `=` before expression"
  let expr ← parseTopExpr
  pure (.output "dac.out" expr)

-- ── Instance decls ───────────────────────────────────────────────────────────

/-- `<key=value, ...>` (opening `<` already consumed; consumes through `>`). -/
def parseTypeArgs : P (Array (String × JsonNumber)) := do
  let entries ← commaList .gt (do
    let k := (← consume .ident "type-arg name").sval
    let _ ← consume .assign "`=` after type-arg name"
    let vTok ← consume .num "type-arg value (number literal)"
    match vTok.val with
    | .num n => pure (k, n)
    | _ => throw "type-arg value missing")
  let _ ← consume .gt "closing `>` of type-args"
  pure entries

/-- `(port: expr, ...)` — keyword-arg inputs (`(` already consumed; stops at `)`). -/
def parseInstanceInputs : P (Array (String × ParsedExpr)) :=
  commaList .rpar (do
    let port := (← consume .ident "instance input port name").sval
    let _ ← consume .colon "`:` after input port"
    let v ← parseTopExpr
    pure (port, v))

/-- `ProgType[<typeArgs>](port: expr, ...)` — RHS of an instance decl. -/
def parseInstanceRhs (name : String) : P BodyDecl := do
  let programName := (← consume .ident "program type name").sval
  let typeArgs? ← (do
    if (← eat .lt).isSome then pure (some (← parseTypeArgs)) else pure none)
  let _ ← consume .lpar s!"`(` after program type '{programName}'"
  let inputs ← parseInstanceInputs
  let _ ← consume .rpar s!"closing `)` of '{programName}' inputs"
  let typeArgs? := match typeArgs? with
    | some ta => if ta.isEmpty then none else some ta
    | none => none
  let inputs? := if inputs.isEmpty then none else some inputs
  pure (.inst name programName typeArgs? inputs?)

/-- `name = ...` — instanceDecl when the RHS is `Type<...>(...)` / `Type(...)` /
    `type(port: ...)`, otherwise an outputAssign. Total multi-token lookahead. -/
def parseAssignOrInstance : P BodyItem := do
  let name := (← consume .ident "statement target name").sval
  let _ ← consume .assign s!"`=` after '{name}'"
  let t ← peek 0
  let t2 ← peek 1
  if t.kind == .ident && t2.kind == .lt then
    pure (.decl (← parseInstanceRhs name))
  else if t.kind == .ident && t2.kind == .lpar then
    if isCapitalized t.sval then
      pure (.decl (← parseInstanceRhs name))
    else
      let t3 ← peek 2
      let t4 ← peek 3
      if t3.kind == .ident && t4.kind == .colon then
        pure (.decl (← parseInstanceRhs name))
      else
        pure (.assign (.output name (← parseTopExpr)))
  else
    pure (.assign (.output name (← parseTopExpr)))

end Tropical.Parse.Surface
