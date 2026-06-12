import Tropical.Parse.Surface.Cursor
import Tropical.Parse.Nodes

/-!
# Surface expression parser (port of `compiler/parse/expressions.ts`)

Precedence-climbing infix (10 table-driven levels, left-associative), prefix
unary with `-literal` constant-folding, postfix (`.`→nestedOut, `[]`→index,
`()`→call/combinator), and primaries (num/bool/paren/array/let/match/tag).
Bare identifiers resolve to `binding` when lexically bound (combinator/let
binders threaded through `Ctx.binders`), else `nameRef`.

One closed mutual block: expressions are the lowest layer and never call up
into statements/declarations.
-/

namespace Tropical.Parse.Surface

open Tropical.Parse (ParsedExpr BinaryOpTag UnaryOpTag TagPayloadEntry MatchArm)
open Lean (JsonNumber)

-- ── Precedence table (weakest → strongest), exactly the TS order ─────────────

def infixLevels : Array (Array (TokKind × BinaryOpTag)) := #[
  #[(.orOr, .or)],
  #[(.andAnd, .and)],
  #[(.pipe, .bitOr)],
  #[(.caret, .bitXor)],
  #[(.amp, .bitAnd)],
  #[(.eqEq, .eq), (.ne, .neq)],
  #[(.lt, .lt), (.le, .lte), (.gt, .gt), (.ge, .gte)],
  #[(.shl, .lshift), (.shr, .rshift)],
  #[(.plus, .add), (.minus, .sub)],
  #[(.star, .mul), (.slash, .div), (.percent, .mod)]
]

def unaryOp? : TokKind → Option UnaryOpTag
  | .minus => some .neg | .bang => some .not | .tilde => some .bitNot
  | _ => none

private def isCapitalized (s : String) : Bool :=
  match s.toList with
  | c :: _ => c.isUpper
  | [] => false

private def negNum (n : JsonNumber) : JsonNumber := { n with mantissa := -n.mantissa }

-- ── The parser (one mutual block) ────────────────────────────────────────────

mutual

partial def parseTopExpr : P ParsedExpr := parseInfix 0

partial def parseInfix (level : Nat) : P ParsedExpr := do
  if level ≥ infixLevels.size then
    parseUnary
  else
    let ops := infixLevels[level]!
    let lhs ← parseInfix (level + 1)
    parseInfixLoop level ops lhs

partial def parseInfixLoop (level : Nat) (ops : Array (TokKind × BinaryOpTag))
    (lhs : ParsedExpr) : P ParsedExpr := do
  match ops.find? (·.1 == (← peekKind)) with
  | none => pure lhs
  | some (_, op) =>
    advance
    let rhs ← parseInfix (level + 1)
    parseInfixLoop level ops (.binary op lhs rhs)

partial def parseUnary : P ParsedExpr := do
  match unaryOp? (← peekKind) with
  | none => parsePostfix
  | some op =>
    advance
    let operand ← parseUnary
    -- Constant-fold `-<literal>` so `[1, -0.5]` matches the corpus.
    match op, operand with
    | .neg, .num n => pure (.num (negNum n))
    | _, _ => pure (.unary op operand)

partial def parsePostfix : P ParsedExpr := do
  parsePostfixLoop (← parsePrimary)

partial def parsePostfixLoop (node : ParsedExpr) : P ParsedExpr := do
  match (← peekKind) with
  | .dot =>
    advance
    let field ← consume .ident "field name after `.`"
    match node with
    | .nameRef nm => parsePostfixLoop (.nestedOut nm field.sval)
    | _ => throw "dot access requires an identifier on the left (got a complex expression)"
  | .lbrack =>
    advance
    let idx ← parseTopExpr
    let _ ← consume .rbrack "closing `]`"
    parsePostfixLoop (.index node idx)
  | .lpar =>
    advance
    match node with
    | .nameRef nm =>
      match (← parseCombinatorCall nm) with
      | some comb => parsePostfixLoop comb
      | none =>
        let args ← commaList .rpar parseTopExpr
        let _ ← consume .rpar "closing `)`"
        parsePostfixLoop (.call node args)
    | _ =>
      let args ← commaList .rpar parseTopExpr
      let _ ← consume .rpar "closing `)`"
      parsePostfixLoop (.call node args)
  | _ => pure node

/-- Known combinator dispatch (callee ident + `(` already consumed). `none`
    falls back to a generic call. Consumes through the closing `)`. -/
partial def parseCombinatorCall (name : String) : P (Option ParsedExpr) := do
  match name with
  | "fold"     => some <$> parseFoldOrScan false
  | "scan"     => some <$> parseFoldOrScan true
  | "generate" => some <$> parseGenerate
  | "iterate"  => some <$> parseIterateOrChain false
  | "chain"    => some <$> parseIterateOrChain true
  | "map2"     => some <$> parseMap2
  | "zipWith"  => some <$> parseZipWith
  | _          => pure none

partial def parseFoldOrScan (isScan : Bool) : P ParsedExpr := do
  let over ← parseTopExpr
  let _ ← consume .comma
  let init ← parseTopExpr
  let _ ← consume .comma
  let binders ← parseLambdaArgs 2
  let body ← parseLambdaBody binders
  let _ ← consume .rpar
  let acc := binders[0]!
  let elem := binders[1]!
  pure (if isScan then .scan over init acc elem body else .fold over init acc elem body)

partial def parseGenerate : P ParsedExpr := do
  let count ← parseTopExpr
  let _ ← consume .comma
  let binders ← parseLambdaArgs 1
  let body ← parseLambdaBody binders
  let _ ← consume .rpar
  pure (.generate count binders[0]! body)

partial def parseIterateOrChain (isChain : Bool) : P ParsedExpr := do
  let count ← parseTopExpr
  let _ ← consume .comma
  let init ← parseTopExpr
  let _ ← consume .comma
  let binders ← parseLambdaArgs 1
  let body ← parseLambdaBody binders
  let _ ← consume .rpar
  pure (if isChain then .chain count binders[0]! init body else .iterate count binders[0]! init body)

partial def parseMap2 : P ParsedExpr := do
  let over ← parseTopExpr
  let _ ← consume .comma
  let binders ← parseLambdaArgs 1
  let body ← parseLambdaBody binders
  let _ ← consume .rpar
  pure (.map2 over binders[0]! body)

partial def parseZipWith : P ParsedExpr := do
  let a ← parseTopExpr
  let _ ← consume .comma
  let b ← parseTopExpr
  let _ ← consume .comma
  let binders ← parseLambdaArgs 2
  let body ← parseLambdaBody binders
  let _ ← consume .rpar
  pure (.zipWith a b binders[0]! binders[1]! body)

/-- `(b1, b2, ...) =>` returning the binder names; arity-checked. -/
partial def parseLambdaArgs (expectedArity : Nat) : P (Array String) := do
  let _ ← consume .lpar "opening `(` for lambda"
  let binders ← commaList .rpar (do pure (← consume .ident "binder name").sval)
  let _ ← consume .rpar "closing `)` of lambda args"
  if binders.size != expectedArity then
    throw s!"lambda expects {expectedArity} binder(s), got {binders.size}"
  let _ ← consume .fatArrow "`=>` after lambda binders"
  pure binders

partial def parseLambdaBody (binders : Array String) : P ParsedExpr :=
  withScope binders.toList parseTopExpr

partial def parsePrimary : P ParsedExpr := do
  let t ← cur
  match t.kind with
  | .num =>
    advance
    match t.val with
    | .num n => pure (.num n)
    | _ => throw "internal: num token without value"
  | .true_ => advance; pure (.bool true)
  | .false_ => advance; pure (.bool false)
  | .lpar =>
    advance
    let inner ← parseTopExpr
    let _ ← consume .rpar "closing `)`"
    pure inner
  | .lbrack => advance; parseArrayLiteral
  | .klet => advance; parseLet
  | .kmatch => advance; parseMatch
  | .ident =>
    advance
    let name := t.sval
    let nextBrace := (← peekKind) == .lbrace
    if isCapitalized name && nextBrace then
      advance  -- consume `{`
      let payload ← parseTagPayload name
      let _ ← consume .rbrace s!"closing `}}` of tag '{name}' payload"
      pure (.tag name (if payload.isEmpty then none else some payload))
    else if (← isBound name) then
      pure (.binding name)
    else
      pure (.nameRef name)
  | _ => throw "unexpected token in expression"

partial def parseArrayLiteral : P ParsedExpr := do
  let items ← commaList .rbrack parseTopExpr
  let _ ← consume .rbrack "closing `]` of array literal"
  pure (.arr items)

partial def parseTagPayload (variant : String) : P (Array TagPayloadEntry) := do
  if (← peekKind) == .rbrace then pure #[]
  else parseTagPayloadLoop variant #[]

partial def parseTagPayloadLoop (variant : String) (acc : Array TagPayloadEntry) :
    P (Array TagPayloadEntry) := do
  let fnameTok ← consume .ident s!"tag '{variant}' payload field name"
  let fname := fnameTok.sval
  if acc.any (·.field == fname) then
    throw s!"tag '{variant}': duplicate payload field '{fname}'"
  let _ ← consume .colon s!"tag '{variant}' `:` after field name"
  let value ← parseTopExpr
  let acc := acc.push (.mk fname value)
  if (← peekKind) == .rbrace then pure acc
  else do
    let _ ← consume .comma s!"tag '{variant}' `,` between payload fields"
    parseTagPayloadLoop variant acc

partial def parseLet : P ParsedExpr := do
  let _ ← consume .lbrace "let: opening `{`"
  let binds ← parseLetBinds #[]
  let _ ← consume .rbrace "let: closing `}`"
  let _ ← consume .kin "let: `in`"
  let order := binds.toList.map (·.1)
  let body ← withScope order parseTopExpr
  pure (.letIn binds body)

partial def parseLetBinds (acc : Array (String × ParsedExpr)) :
    P (Array (String × ParsedExpr)) := do
  if (← peekKind) == .rbrace then pure acc
  else
    let nameTok ← consume .ident "let binding name"
    let name := nameTok.sval
    if acc.any (·.1 == name) then throw s!"let: duplicate binding name '{name}'"
    let _ ← consume .colon "let binding `:`"
    let value ← parseTopExpr
    let acc := acc.push (name, value)
    -- separators `;` or `,`, optional before `}`
    let sawSep ← (do
      if (← eat .semi).isSome then pure true
      else if (← eat .comma).isSome then pure true
      else pure false)
    if sawSep then parseLetBinds acc else pure acc

partial def parseMatch : P ParsedExpr := do
  let scrutinee ← parseTopExpr
  let _ ← consume .lbrace "`{` after match scrutinee"
  let arms ← parseMatchArms #[]
  pure (.match_ scrutinee arms)

partial def parseMatchArms (acc : Array MatchArm) : P (Array MatchArm) := do
  if (← peekKind) == .rbrace then
    let _ ← consume .rbrace
    pure acc
  else
    let variantTok ← consume .ident "match arm variant name"
    let variant := variantTok.sval
    if acc.any (·.variant == variant) then
      throw s!"match: duplicate arm for variant '{variant}'"
    let binds ← (do
      if (← eat .lbrace).isSome then
        let pairs ← commaList .rbrace parseMatchBind
        let _ ← consume .rbrace s!"arm '{variant}' closing `}}` of pattern"
        pure pairs
      else pure #[])
    let _ ← consume .fatArrow s!"arm '{variant}' `=>` after pattern"
    let bindNames := binds.toList.map (·.2)
    let body ← withScope bindNames parseTopExpr
    let acc := acc.push (.mk variant binds body)
    if (← peekKind) == .rbrace then
      let _ ← consume .rbrace
      pure acc
    else do
      let _ ← consume .comma "match: `,` between arms"
      parseMatchArms acc

partial def parseMatchBind : P (String × String) := do
  let fname ← consume .ident "arm field name"
  let _ ← consume .colon "arm `:` after field name"
  let localName ← consume .ident "arm bind name"
  pure (fname.sval, localName.sval)

end

/-- Parse a complete expression; error on trailing input. -/
def parseExpr (src : String) : Except String ParsedExpr := do
  let toks ← tokenize src
  let (node, c) ← parseTopExpr.run { toks := toks }
  let trailing := c.toks[min c.i (c.toks.size - 1)]!
  if trailing.kind != .eof then
    throw "unexpected trailing input"
  pure node

end Tropical.Parse.Surface
