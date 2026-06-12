import Tropical.Parse.Surface.Lexer
import Std.Data.HashSet

/-!
# Parser cursor + plumbing (port of `compiler/parse/shared.ts`)

The three parser layers walk one `Array Tok` with index lookahead. The
monad is `StateT Ctx (Except String)`: `Ctx` carries the token array, the
cursor index, and the lexical binder set (so a bare identifier resolves to
`binding` vs `nameRef` — the TS `Ctx.binders` threaded by `withScope`).

`por` is an explicit backtracking alternative (restore the cursor on the
left's failure) for the one site that needs it (instance-decl vs
output-assign). Everywhere else dispatch is by `peek`, no backtracking.
-/

namespace Tropical.Parse.Surface

open Std (HashSet)

structure Ctx where
  toks : Array Tok
  i : Nat := 0
  binders : HashSet String := {}

abbrev P (α : Type) := StateT Ctx (Except String) α

/-- Backtracking alternative: try `p`; on failure rerun `q` from the cursor
    state `p` started in. -/
def por {α : Type} (p q : P α) : P α := StateT.mk fun s =>
  match p.run s with
  | .ok r => .ok r
  | .error _ => q.run s

/-- Lookahead, clamped to the trailing `eof` (mirrors TS `peek`). -/
def peek (offset : Nat := 0) : P Tok := do
  let c ← get
  pure c.toks[min (c.i + offset) (c.toks.size - 1)]!

def peekKind (offset : Nat := 0) : P TokKind := do
  pure (← peek offset).kind

def cur : P Tok := peek 0

def advance : P Unit := modify fun c => { c with i := c.i + 1 }

private def kindName (k : TokKind) : String := toString (repr k)

/-- Require a token of `kind`, advancing past it. -/
def consume (kind : TokKind) (what : String := "") : P Tok := do
  let t ← cur
  if t.kind == kind then
    advance
    pure t
  else
    throw s!"expected {if what.isEmpty then kindName kind else what}, got {kindName t.kind}"

/-- Consume a token of `kind` if present; otherwise leave the cursor put. -/
def eat (kind : TokKind) : P (Option Tok) := do
  let t ← cur
  if t.kind == kind then advance; pure (some t) else pure none

/-- Parse a comma-separated list up to (not including) `terminator`.
    Tolerates an empty list and a trailing comma. -/
def commaList {α : Type} (terminator : TokKind) (parse : P α) : P (Array α) := do
  if (← peekKind) == terminator then return #[]
  let mut items := #[← parse]
  while (← eat .comma).isSome do
    if (← peekKind) == terminator then break
    items := items.push (← parse)
  return items

/-- Run `body` with `names` added to the binder scope, removing only the ones
    this call introduced (nested binders nest cleanly). On a thrown error the
    whole parse aborts, so scope restoration on the error path is moot. -/
def withScope {α : Type} (names : List String) (body : P α) : P α := do
  let c ← get
  let added := names.filter fun n => !c.binders.contains n
  modify fun c => { c with binders := added.foldl (·.insert ·) c.binders }
  let r ← body
  modify fun c => { c with binders := added.foldl (fun s n => s.erase n) c.binders }
  pure r

def isBound (name : String) : P Bool := do
  pure ((← get).binders.contains name)

/-- True iff `t` is an `ident` whose value is `name` — contextual keywords
    (`init`, `null`, `dac`, `smoothed`, `out`, `breaks_cycles`). -/
def isCtxKw (t : Tok) (name : String) : Bool :=
  t.kind == .ident && t.sval == name

end Tropical.Parse.Surface
