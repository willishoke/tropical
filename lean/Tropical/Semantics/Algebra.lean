import Tropical.Semantics.Value

/-!
# Carrier-parametric scalar algebra

Lowering preservation is about retaining primitive operations and their order,
not proving analytic facts about a particular implementation of `sqrt`,
`ldexp`, or the numeric conversions.  `Algebra` therefore supplies those
operations explicitly.  It also owns conversion of a dynamic bank count and
the loop-index injection; either may refuse.
-/

namespace Tropical.Semantics

open Lean (JsonNumber)
open Tropical.Ir

structure Algebra (α : Type) where
  literal : JsonNumber → Result α
  unary : UnaryOpTag → Value α → Result α
  binary : BinaryOpTag → Value α → Value α → Result α
  clamp : Value α → Value α → Value α → Result α
  select : Value α → Value α → Value α → Result α
  index : Value α → Value α → Result α
  loopIndex : Nat → Result α
  /-- Convert the runtime count to the signed integer seen by production
      `ReduceBegin`; `bankTrips` performs the lower and capacity clamps. -/
  dynamicCount : Value α → Except Refusal Int
  zero : Result α

/-- Left-to-right binary lifting.  A refusal on the left prevents evaluation
    of the right, matching the structural evaluator's order. -/
def applyBinary (alg : Algebra α) (tag : BinaryOpTag)
    (lhs rhs : Result α) : Result α :=
  match lhs with
  | .error error => .error error
  | .ok lhs =>
    match rhs with
    | .error error => .error error
    | .ok rhs => alg.binary tag lhs rhs

def applyTernary (op : Value α → Value α → Value α → Result α)
    (a b c : Result α) : Result α :=
  match a with
  | .error error => .error error
  | .ok a =>
    match b with
    | .error error => .error error
    | .ok b =>
      match c with
      | .error error => .error error
      | .ok c => op a b c

end Tropical.Semantics
