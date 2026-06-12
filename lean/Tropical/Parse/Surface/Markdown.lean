import Tropical.Parse.Surface.Declarations

/-!
# Literate extraction + driver (port of `compiler/parse/markdown.ts`)

Extracts the first ```` ```tropical ```` fenced block from a literate `.md`
document and parses it. Fence rules mirror the TS extractor: 3+ backticks at
column 0, first whitespace-delimited info word is the language tag, a closing
fence is a backtick run (≥ the opening length) followed only by whitespace,
and an unterminated fence runs to end-of-document. The stdlib convention is
one tropical block per file, so the driver takes the first.
-/

namespace Tropical.Parse.Surface

open Tropical.Parse (Program)

/-- `(fenceLength, languageTag)` if `line` opens a fence (≥3 backticks). -/
private def fenceLang? (line : String) : Option (Nat × String) :=
  let chars := line.toList
  let bt := (chars.takeWhile (· == '`')).length
  if bt < 3 then none
  else
    let after := (chars.drop bt).dropWhile (·.isWhitespace)
    some (bt, String.ofList (after.takeWhile (fun c => !c.isWhitespace)))

/-- A closing fence: ≥ `openLen` backticks then only whitespace. -/
private def isClosingFence (line : String) (openLen : Nat) : Bool :=
  let chars := line.toList
  let bt := (chars.takeWhile (· == '`')).length
  bt ≥ openLen && (chars.drop bt).all (·.isWhitespace)

private partial def findClose (lines : Array String) (j openLen : Nat) : Nat :=
  if j < lines.size then
    if isClosingFence lines[j]! openLen then j else findClose lines (j + 1) openLen
  else lines.size

private partial def firstTropical (lines : Array String) (i : Nat) : Option String :=
  if i < lines.size then
    match fenceLang? lines[i]! with
    | none => firstTropical lines (i + 1)
    | some (openLen, lang) =>
      let close := findClose lines (i + 1) openLen
      if lang == "tropical" then
        some (String.intercalate "\n" ((lines.toList.drop (i + 1)).take (close - (i + 1))))
      else
        firstTropical lines (close + 1)
  else none

/-- Extract the first tropical block from a literate `.md` document. -/
def firstTropicalBlock (src : String) : Option String :=
  firstTropical (src.splitOn "\n").toArray 0

/-- Parse a literate `.md` document's first tropical block into a `Program`. -/
def parseMarkdownProgram (src : String) : Except String Program :=
  match firstTropicalBlock src with
  | none => throw "no tropical code block found"
  | some body => parseProgram body

end Tropical.Parse.Surface
