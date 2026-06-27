import Std.Internal.Parsec
import Std.Internal.Parsec.String
import Lean.Data.Json

/-!
# Surface lexer (port of `compiler/parse/lexer.ts`)

A combinator lexer over `Std.Internal.Parsec.String`, producing a flat
`Array Tok` that the parser layers (`Expr`, `Statements`, `Declarations`)
consume by index with total lookahead. The TS lexer is an index-based
sticky-regex scanner; this is the same token set realized with combinators
(the "combinators absorb lexing" half of the two-stage design).

Numbers reuse `Lean.Json.parse`: the matched surface literal is normalized
to JSON form (a leading `.5` gets a `0` prefix) and parsed, so the resulting
`JsonNumber` is bit-identical to what the committed `stdlib/parsed/*.json`
corpus decodes to. Tokens carry no source position — the differential gate
never exercises error paths over the 33 stdlib files, and spot tests assert
on structure, not on `line:col` strings.
-/

namespace Tropical.Parse.Surface

open Std.Internal.Parsec
open Std.Internal.Parsec.String

-- ── Token kinds ──────────────────────────────────────────────────────────────

inductive TokKind where
  -- literals + identifiers
  | num | ident | str
  -- boolean keywords (own kinds, like TS)
  | true_ | false_
  -- binders + control flow
  | klet | kin | kif | kelse | kmatch
  -- declarations
  | kprogram | kreg | kparam | knext
  -- ADTs
  | kstruct | kenum | ktype
  -- brackets
  | lpar | rpar | lbrack | rbrack | lbrace | rbrace
  -- separators / accessors
  | comma | dot | semi | colon
  -- assignment + arrows
  | assign | fatArrow | arrow
  -- arithmetic
  | plus | minus | star | slash | percent
  -- comparison
  | lt | le | gt | ge | eqEq | ne
  -- bitwise
  | shl | shr | amp | pipe | caret | tilde
  -- logical
  | andAnd | orOr | bang
  -- end of stream
  | eof
deriving DecidableEq, Repr, Inhabited

/-- Payload for value-bearing tokens: `num`, `ident`, `str`. -/
inductive TokVal where
  | none
  | num (n : Lean.JsonNumber)
  | str (s : String)
deriving Repr, Inhabited

structure Tok where
  kind : TokKind
  val : TokVal := .none
deriving Repr, Inhabited

/-- The `str` payload of an `ident`/`str` token (empty otherwise). -/
def Tok.sval (t : Tok) : String :=
  match t.val with | .str s => s | _ => ""

-- ── Keyword + punctuation tables ─────────────────────────────────────────────

def keywordKind? : String → Option TokKind
  | "true" => some .true_ | "false" => some .false_
  | "let" => some .klet | "in" => some .kin
  | "if" => some .kif | "else" => some .kelse | "match" => some .kmatch
  | "program" => some .kprogram
  | "reg" => some .kreg | "param" => some .kparam
  | "next" => some .knext
  | "struct" => some .kstruct | "enum" => some .kenum | "type" => some .ktype
  | _ => none

def punct1? : Char → Option TokKind
  | '(' => some .lpar | ')' => some .rpar
  | '[' => some .lbrack | ']' => some .rbrack
  | '{' => some .lbrace | '}' => some .rbrace
  | ',' => some .comma | '.' => some .dot
  | ';' => some .semi | ':' => some .colon
  | '=' => some .assign
  | '+' => some .plus | '-' => some .minus | '*' => some .star
  | '/' => some .slash | '%' => some .percent
  | '<' => some .lt | '>' => some .gt
  | '&' => some .amp | '|' => some .pipe | '^' => some .caret
  | '~' => some .tilde | '!' => some .bang
  | _ => none

-- ── Whitespace + comments ────────────────────────────────────────────────────

/-- Skip a block-comment body, having already consumed the opening `/*`. -/
partial def skipBlock : Parser Unit := do
  match (← peek?) with
  | none => fail "unterminated block comment"
  | some '*' =>
    skip
    match (← peek?) with
    | some '/' => skip
    | _ => skipBlock
  | some _ => skip; skipBlock

/-- Whitespace and line/block comments, repeated. -/
partial def wsc : Parser Unit := do
  ws
  match (← peek?) with
  | some '/' =>
    if (← (do let _ ← attempt (skipString "//"); pure true) <|> pure false) then
      let _ ← manyChars (satisfy (· != '\n'))
      wsc
    else if (← (do let _ ← attempt (skipString "/*"); pure true) <|> pure false) then
      skipBlock
      wsc
    else
      pure ()
  | _ => pure ()

-- ── Numbers ──────────────────────────────────────────────────────────────────

private def isIdentStart (c : Char) : Bool := c.isAlpha || c == '_'
private def isIdentCont (c : Char) : Bool := c.isAlphanum || c == '_'

/-- `[0-9]+ ('.' [0-9]+)?` — the integer-led form. The fractional part is
    `attempt`ed so `1.foo` lexes as `1`, `.`, `foo` (consume `.`, fail on the
    missing digits, backtrack). -/
private def intDotFrac : Parser String := do
  let i ← many1Chars digit
  let f ← (attempt do skipChar '.'; let d ← many1Chars digit; pure ("." ++ d)) <|> pure ""
  pure (i ++ f)

/-- `'.' [0-9]+` — the leading-dot form (`.5`). -/
private def dotFrac : Parser String := do
  skipChar '.'
  let d ← many1Chars digit
  pure ("." ++ d)

/-- `[eE] [+-]? [0-9]*` — exponent (zero digits captured so `1e` reaches the
    JSON parser and errors as a malformed number, matching TS). -/
private def expPart : Parser String := do
  let _ ← (pchar 'e' <|> pchar 'E')
  let sgn ← ((fun c => c.toString) <$> (pchar '+' <|> pchar '-')) <|> pure ""
  let d ← manyChars digit
  pure ("e" ++ sgn ++ d)

def lexNumber : Parser Tok := do
  let mant ← intDotFrac <|> dotFrac
  let exp ← expPart <|> pure ""
  let s := mant ++ exp
  let normalized := if s.startsWith "." then "0" ++ s else s
  match Lean.Json.parse normalized with
  | .ok (.num n) => pure { kind := .num, val := .num n }
  | _ => fail s!"malformed number: {s}"

-- ── Identifiers / keywords ───────────────────────────────────────────────────

def lexIdentOrKw : Parser Tok := do
  let c ← satisfy isIdentStart
  let rest ← manyChars (satisfy isIdentCont)
  let name := c.toString ++ rest
  match keywordKind? name with
  | some k => pure { kind := k }
  | none => pure { kind := .ident, val := .str name }

-- ── Strings ──────────────────────────────────────────────────────────────────

private def escapeChar? : Char → Option Char
  | 'n' => some '\n' | 't' => some '\t' | 'r' => some '\r'
  | '\\' => some '\\' | '\'' => some '\'' | '"' => some '"'
  | _ => none

partial def stringBody (q : Char) (acc : String) : Parser String := do
  match (← peek?) with
  | none => fail "unterminated string literal"
  | some '\n' => fail "unterminated string literal"
  | some c =>
    if c == q then pure acc
    else if c == '\\' then
      skip
      match (← peek?) with
      | none => fail "unterminated string literal"
      | some e =>
        skip
        match escapeChar? e with
        | some r => stringBody q (acc.push r)
        | none => fail s!"unknown escape: \\{e}"
    else
      skip; stringBody q (acc.push c)

def lexString : Parser Tok := do
  let q ← (pchar '"' <|> pchar '\'')
  let body ← stringBody q ""
  skipChar q
  pure { kind := .str, val := .str body }

-- ── Punctuation ──────────────────────────────────────────────────────────────

def lexPunct : Parser Tok := do
  let c ← any
  let n ← peek?
  match c, n with
  | '<', some '=' => skip; pure { kind := .le }
  | '>', some '=' => skip; pure { kind := .ge }
  | '=', some '=' => skip; pure { kind := .eqEq }
  | '!', some '=' => skip; pure { kind := .ne }
  | '<', some '<' => skip; pure { kind := .shl }
  | '>', some '>' => skip; pure { kind := .shr }
  | '&', some '&' => skip; pure { kind := .andAnd }
  | '|', some '|' => skip; pure { kind := .orOr }
  | '=', some '>' => skip; pure { kind := .fatArrow }
  | '-', some '>' => skip; pure { kind := .arrow }
  | _, _ =>
    match punct1? c with
    | some k => pure { kind := k }
    | none => fail s!"unexpected character: {c}"

-- ── Driver ───────────────────────────────────────────────────────────────────

/-- One token. `lexNumber` is `attempt`ed so a leading `.` that isn't a number
    (`.foo`) backtracks to the punctuation rule. Each earlier alternative fails
    without consuming, so `<|>` falls through cleanly. -/
def lexToken : Parser Tok :=
  (attempt lexNumber) <|> lexIdentOrKw <|> lexString <|> lexPunct

partial def lexAll (acc : Array Tok) : Parser (Array Tok) := do
  wsc
  if (← isEof) then
    pure (acc.push { kind := .eof })
  else
    let t ← lexToken
    lexAll (acc.push t)

def tokenize (src : String) : Except String (Array Tok) :=
  Parser.run (lexAll #[]) src

end Tropical.Parse.Surface
