import Tropical.Parse.Surface.Statements
import Tropical.Parse.Surface.Bounds

/-!
# Declaration parser (port of `compiler/parse/declarations.ts`)

Program header, type params, ports, ADTs, and the body↔program mutual knot.
Per the port plan (R4) the TS callback injection is dropped: `parseBodyItem`
and `parseProgramFromCtx` sit in one `mutual` block, so nested `program`
decls and the body are directly mutually recursive.

Explicit `in [lo, hi]` bounds are parsed into a per-program sidecar (the Lean
AST carries no `bounds` field) and handed to `lowerBounds`, which folds them —
together with built-in alias bounds derived from port types — into `clamp`/
`select` on input defaults and output assigns. `lowerBounds` lives in
`Bounds.lean`; here it is applied at the end of each `parseProgramFromCtx`.
-/

namespace Tropical.Parse.Surface

open Tropical.Parse
  (Program ProgramPorts ProgramPort ProgramPortSpec PortTypeDecl ShapeDim
   ScalarKind TypeDef StructField SumVariant TypeParamSpec Block BodyDecl)
open Lean (JsonNumber)

-- `BoundPair`, `portName`, and `lowerBounds` come from `Bounds.lean`.

-- ── Type params: `<N: int [= default], ...>` ────────────────────────────────

def parseTypeParams : P (Array (String × TypeParamSpec)) := do
  let _ ← consume .lt "opening `<` of type params"
  let entries ← commaList .gt (do
    let name := (← consume .ident "type-param name").sval
    let _ ← consume .colon s!"`:` after type-param '{name}'"
    let typeName := (← consume .ident s!"type-param '{name}' type").sval
    if typeName != "int" then
      throw s!"type-param '{name}' type must be 'int', got '{typeName}'"
    let default? ← (do
      if (← eat .assign).isSome then
        match (← consume .num s!"default for type-param '{name}'").val with
        | .num n => pure (some n)
        | _ => throw "type-param default missing"
      else pure none)
    pure (name, ({ default? := default? } : TypeParamSpec)))
  let _ ← consume .gt "closing `>` of type params"
  pure entries

-- ── Port types ───────────────────────────────────────────────────────────────

def parseShapeDim : P ShapeDim := do
  let t ← cur
  match t.kind with
  | .num =>
    advance
    match t.val with | .num n => pure (.lit n) | _ => throw "shape dim missing value"
  | .ident => advance; pure (.ref t.sval)
  | _ => throw "expected number or identifier in array shape"

def parsePortType : P PortTypeDecl := do
  let element := (← consume .ident "port type name").sval
  if (← peekKind) != .lbrack then pure (.scalar element)
  else do
    advance  -- consume `[`
    let shape ← commaList .rbrack parseShapeDim
    let _ ← consume .rbrack "closing `]` of array type"
    if shape.isEmpty then throw "array type must have at least one shape dim"
    pure (.array element shape)

-- ── Bounds (`in [lo, hi]`) ───────────────────────────────────────────────────

def parseBound : P (Option JsonNumber) := do
  let t ← cur
  if isCtxKw t "null" then advance; pure none
  else do
    let neg ← (do if (← eat .minus).isSome then pure true else pure false)
    let numTok ← cur
    match numTok.val with
    | .num n => advance; pure (some (if neg then { n with mantissa := -n.mantissa } else n))
    | _ => throw "bound must be a number literal or 'null'"

def parseBounds : P BoundPair := do
  let _ ← consume .lbrack "`[` opening bounds"
  let lo ← parseBound
  let _ ← consume .comma "`,` between bound lo/hi"
  let hi ← parseBound
  let _ ← consume .rbrack "`]` closing bounds"
  pure (lo, hi)

-- ── Port specs ───────────────────────────────────────────────────────────────

/-- One port: bare name, or `name: type`, `name = default`, `name: type =
    default`, each optionally `in [lo, hi]`. Returns the AST port plus its
    explicit bound (if any) for the lowering sidecar. -/
def parsePortSpec (allowDefault : Bool) : P (ProgramPort × Option BoundPair) := do
  let name := (← consume .ident "port name").sval
  let k ← peekKind
  if k != .colon && k != .assign then
    pure (.bare name, none)
  else do
    let type? ← (do
      if k == .colon then advance; pure (some (← parsePortType)) else pure none)
    let default? ← (do
      if (← peekKind) == .assign then
        if !allowDefault then throw "output ports cannot have a default value"
        advance
        pure (some (← parseTopExpr))
      else pure none)
    let bounds? ← (do
      if (← peekKind) == .kin then advance; pure (some (← parseBounds)) else pure none)
    pure (.spec { name := name, type? := type?, default? := default? }, bounds?)

def parsePortList (allowDefault : Bool) :
    P (Array ProgramPort × Array (String × BoundPair)) := do
  let entries ← commaList .rpar (parsePortSpec allowDefault)
  let ports := entries.map (·.1)
  let bounds := entries.filterMap fun e => e.2.map fun b => (portName e.1, b)
  pure (ports, bounds)

-- ── ADTs: struct / enum / type alias ─────────────────────────────────────────

def parseScalarKind (what : String) : P ScalarKind := do
  let t ← consume .ident what
  match ScalarKind.ofWire? t.sval with
  | some sk => pure sk
  | none => throw s!"{what}: expected float/int/bool, got '{t.sval}'"

def parseStructDecl : P TypeDef := do
  let _ ← consume .kstruct "struct keyword"
  let name := (← consume .ident "struct name").sval
  let _ ← consume .lbrace s!"`\{` after struct '{name}'"
  let fields ← commaList .rbrace (do
    let fieldName := (← consume .ident "struct field name").sval
    let _ ← consume .colon "`:` after field name"
    let st ← parseScalarKind "struct field type"
    pure ({ name := fieldName, scalarType := st } : StructField))
  let _ ← consume .rbrace s!"`}` closing struct '{name}'"
  pure (.struct name fields)

def parseSumVariant (enumName : String) : P SumVariant := do
  let variantName := (← consume .ident s!"enum '{enumName}': variant name").sval
  if (← peekKind) != .lpar then pure { name := variantName, payload := #[] }
  else do
    advance  -- consume `(`
    let payload ← commaList .rpar (do
      let pname := (← consume .ident s!"variant '{variantName}' field name").sval
      let _ ← consume .colon s!"variant '{variantName}' `:` after field name"
      let st ← parseScalarKind s!"variant '{variantName}' field type"
      pure ({ name := pname, scalarType := st } : StructField))
    let _ ← consume .rpar s!"closing `)` of variant '{variantName}' payload"
    pure { name := variantName, payload := payload }

def parseEnumDecl : P TypeDef := do
  let _ ← consume .kenum "enum keyword"
  let name := (← consume .ident "enum name").sval
  let _ ← consume .lbrace s!"`\{` after enum '{name}'"
  let variants ← commaList .rbrace (parseSumVariant name)
  let _ ← consume .rbrace s!"`}` closing enum '{name}'"
  pure (.sum name variants)

def parseAliasDecl : P TypeDef := do
  let _ ← consume .ktype "type keyword"
  let name := (← consume .ident "alias name").sval
  let _ ← consume .assign s!"`=` after alias '{name}'"
  let base := (← consume .ident s!"base type for alias '{name}'").sval
  pure (.alias name base)

-- ── The body↔program mutual knot ─────────────────────────────────────────────

mutual

partial def parseProgramFromCtx : P Program := do
  let _ ← consume .kprogram "program keyword"
  let name := (← consume .ident "program name").sval
  let typeParams? ← (do
    if (← peekKind) == .lt then pure (some (← parseTypeParams)) else pure none)
  let _ ← consume .lpar s!"`(` after program name '{name}'"
  let (inputs, inputBounds) ← parsePortList true
  let _ ← consume .rpar s!"closing `)` of inputs for '{name}'"
  let (outputs?, outputBounds) ← (do
    if (← eat .arrow).isSome then
      let _ ← consume .lpar s!"`(` after `->` for '{name}'"
      let (outs, ob) ← parsePortList false
      let _ ← consume .rpar s!"closing `)` of outputs for '{name}'"
      pure (some outs, ob)
    else pure (none, #[]))
  let breaksCycles ← (do
    if isCtxKw (← cur) "breaks_cycles" then advance; pure true else pure false)
  let _ ← consume .lbrace s!"`\{` opening body of '{name}'"
  let (decls, assigns, typeDefs) ← parseBodyItems
  let _ ← consume .rbrace s!"`}` closing body of '{name}'"
  let typeParams? := match typeParams? with
    | some tp => if tp.isEmpty then none else some tp
    | none => none
  let ports : ProgramPorts := {
    inputs := if inputs.isEmpty then none else some inputs,
    outputs := match outputs? with
      | some o => if o.isEmpty then none else some o
      | none => none,
    typeDefs := if typeDefs.isEmpty then none else some typeDefs }
  let ports? := if ports.inputs.isSome || ports.outputs.isSome || ports.typeDefs.isSome
    then some ports else none
  let breaks? := if breaksCycles then some true else none
  let prog := Program.mk name typeParams? ports? (Block.mk decls assigns) breaks?
  pure (lowerBounds prog inputBounds outputBounds)

partial def parseProgramDecl : P BodyDecl := do
  let inner ← parseProgramFromCtx
  pure (.prog inner.name inner)

partial def parseBodyItem : P BodyItem := do
  let t ← cur
  match t.kind with
  | .kreg => BodyItem.decl <$> parseRegDecl
  | .kparam => BodyItem.decl <$> parseParamDecl
  | .knext => BodyItem.assign <$> parseNextUpdate
  | .kprogram => BodyItem.decl <$> parseProgramDecl
  | .kstruct => BodyItem.typeDef <$> parseStructDecl
  | .kenum => BodyItem.typeDef <$> parseEnumDecl
  | .ktype => BodyItem.typeDef <$> parseAliasDecl
  | .ident =>
    if t.sval == "dac" then BodyItem.assign <$> parseDacOutAssign
    else parseAssignOrInstance
  | _ => throw "expected body item"

partial def parseBodyItems : P (Array BodyDecl × Array BodyAssign × Array TypeDef) := do
  let mut decls : Array BodyDecl := #[]
  let mut assigns : Array BodyAssign := #[]
  let mut typeDefs : Array TypeDef := #[]
  while (← peekKind) != .rbrace && (← peekKind) != .eof do
    match (← parseBodyItem) with
    | .decl d => decls := decls.push d
    | .assign a => assigns := assigns.push a
    | .typeDef td => typeDefs := typeDefs.push td
    let _ ← eat .semi
  pure (decls, assigns, typeDefs)

end

/-- Parse a top-level program declaration; error on trailing input. -/
def parseProgram (src : String) : Except String Program := do
  let toks ← tokenize src
  let (node, c) ← parseProgramFromCtx.run { toks := toks }
  let trailing := c.toks[min c.i (c.toks.size - 1)]!
  if trailing.kind != .eof then throw "unexpected trailing input after program"
  pure node

end Tropical.Parse.Surface
