import Tropical.Parse.Nodes

/-!
# Bounds → clamp/select lowering (the shared leaf transform)

`in [lo, hi]` and the built-in port-type aliases (`signal`, `bipolar`,
`unipolar`, `phase`, `freq`) are parse-time notation, not IR. Both ingest
front doors — the surface parser (`Parse/Surface/Bounds.lean`) and the JSON
raise (`Parse/Raise.lean`) — fold them into explicit `clamp`/`select` calls on
input defaults and matching output assigns:

  [lo, hi]     → clamp(expr, lo, hi)
  [lo, null]   → select(expr > lo, expr, lo)   (max)
  [null, hi]   → select(expr < hi, expr, hi)   (min)
  [null, null] → expr

`wrapWithBound` is the one leaf transform both callers share, so the two
front doors agree by construction. It is IDEMPOTENT: an expression that
already enforces the bound (a `clamp`/`select` in the matching shape) is left
untouched, so a default a user wrote as its own `clamp` is not double-wrapped.
Bounds are `JsonNumber` (the surface path admits non-integer `in [0.5, …]`;
the raise path's integer alias bounds embed losslessly via `JsonNumber.fromInt`).
-/

namespace Tropical.Parse

open Lean (JsonNumber)

/-- A bound pair; each side `none` for the `null` sentinel. -/
abbrev BoundPair := Option JsonNumber × Option JsonNumber

/-- Float-bit equality of a numeric literal against a bound value (the shape
    the guard compares; matches the raise path's original `Int` comparison for
    integer bounds since `(JsonNumber.fromInt i).toFloat = Float.ofInt i`). -/
private def exprIsNum (e : ParsedExpr) (target : JsonNumber) : Bool :=
  match e with
  | .num n => n.toFloat.toBits == target.toFloat.toBits
  | _ => false

private def callArgsTo (e : ParsedExpr) (callee : String) : Option (Array ParsedExpr) :=
  match e with
  | .call (.nameRef n) args => if n == callee then some args else none
  | _ => none

/-- True if `e` already enforces `bounds` (the idempotency guard). Checks the
    parser-level `clamp`/`select` call shapes; the elaborator-level direct-op
    shapes the TS guard also accepts are unrepresentable in a ParsedProgram. -/
def alreadyWrapped (e : ParsedExpr) (bounds : BoundPair) : Bool :=
  let (lo, hi) := bounds
  let clampMatch :=
    match callArgsTo e "clamp" with
    | some args =>
      args.size == 3 &&
      (match lo with | some l => exprIsNum args[1]! l | none => false) &&
      (match hi with | some h => exprIsNum args[2]! h | none => false)
    | none => false
  let selectMatch :=
    match callArgsTo e "select" with
    | some args =>
      args.size == 3 &&
      (match args[0]!, lo, hi with
       | .binary .gt _ rhs, some l, none => exprIsNum rhs l && exprIsNum args[2]! l
       | .binary .lt _ rhs, none, some h => exprIsNum rhs h && exprIsNum args[2]! h
       | _, _, _ => false)
    | none => false
  clampMatch || selectMatch

/-- Wrap an expression in the bounds-enforcing op chain (parser call shape),
    idempotently — an already-wrapped expression is returned unchanged. -/
def wrapWithBound (e : ParsedExpr) (bounds : BoundPair) : ParsedExpr :=
  if alreadyWrapped e bounds then e
  else match bounds with
    | (some lo, some hi) => .call (.nameRef "clamp") #[e, .num lo, .num hi]
    | (some lo, none)    => .call (.nameRef "select") #[.binary .gt e (.num lo), e, .num lo]
    | (none, some hi)    => .call (.nameRef "select") #[.binary .lt e (.num hi), e, .num hi]
    | (none, none)       => e

end Tropical.Parse
