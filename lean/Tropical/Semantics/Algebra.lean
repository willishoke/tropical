import Tropical.Semantics.Value

/-!
# Carrier-parametric scalar algebra

Lowering preservation is about retaining primitive operations and their order,
not proving analytic facts about a particular implementation of `sqrt`,
`ldexp`, or the numeric conversions.  `Algebra` therefore supplies those
operations explicitly.  It also owns conversion of a dynamic bank count and
the loop-index injection; either may refuse.

**Sorts.** The IR's nodes are unsorted: an integer width is chosen when the
expression is compiled, not stored in the DAG, so `denoteExpr` calls the
unsorted operations. A Plan knows more. Every constant carries its kind,
every instruction its result sort, and every `WriteSlot` its destination
kind, and `int32` and `int` are different meanings of the same operation
(`add` wraps at 2³² in one and 2⁶⁴ in the other). The Plan semantics passes
those sorts through the `sorted*` operations and `atSort`. An algebra whose
carrier has one integer meaning leaves them at their defaults, which forget
the sort; `Semantics.Scalar.algebra` (the concrete reference) gives them the
total integer semantics of `ConstFold`.
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
  /-- A Plan constant of a declared kind. -/
  sortedLiteral : Tropical.Parse.ScalarKind → JsonNumber → Result α :=
    fun _ number => literal number
  /-- A Plan unary op whose result has the given sort. -/
  sortedUnary : Tropical.Parse.ScalarKind → UnaryOpTag → Value α → Result α :=
    fun _ => unary
  /-- A Plan binary op whose result has the given sort. -/
  sortedBinary : Tropical.Parse.ScalarKind → BinaryOpTag → Value α → Value α → Result α :=
    fun _ => binary
  /-- A Plan `clamp` whose result has the given sort. -/
  sortedClamp : Tropical.Parse.ScalarKind → Value α → Value α → Value α → Result α :=
    fun _ => clamp
  /-- A Plan `select` whose result has the given sort. -/
  sortedSelect : Tropical.Parse.ScalarKind → Value α → Value α → Value α → Result α :=
    fun _ => select
  /-- A value stored at a declared kind (a `WriteSlot`'s destination): the
      conversion every backend applies on that store. -/
  atSort : Tropical.Parse.ScalarKind → Value α → Result α := fun _ value => .ok value

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
