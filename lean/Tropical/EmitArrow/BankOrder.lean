import Tropical.Semantics.Expr

/-!
# Arena-native bank order

The production bank is already an `ENode.bankSum`; there is no source tree to
unroll.  Its direct denotation evaluates eager tables first and then performs a
left fold over `0 .. capacity-1`, installing the unique binder id for each body
read.  These statements expose that order without constructing a recursive
reference expression.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Semantics

/-- Reference left fold over increasing natural indices, parametric in the
    carrier and operation. -/
def refFold {α : Sort _} (op : α → α → α) (zero : α)
    (body : Nat → α) : Nat → α
  | 0 => zero
  | n + 1 => op (refFold op zero body n) (body n)

theorem refFold_succ {α : Sort _} (op : α → α → α) (zero : α)
    (body : Nat → α) (n : Nat) :
    refFold op zero body (n + 1) = op (refFold op zero body n) (body n) := rfl

/-- A static production bank denotes exactly the eager-table check followed by
    its authored increasing-index left fold. No associativity or commutativity
    assumption occurs. -/
theorem denoteExpr_staticBank_order (alg : Algebra α) (env : SigEnv α)
    (arena : ExprArena) (hArena : ArenaWellFormed arena)
    {root : ExprId} {capacity : Nat} {tables : Array ExprId}
    {body : ExprId} {binderId : Nat}
    (hDeref : arena.deref root =
      some (.bankSum capacity tables body none binderId)) :
    denoteExpr alg env arena hArena root =
      match sequence
          (tables.attach.map fun ⟨table, _hMem⟩ =>
            denoteExpr alg env arena hArena table) with
      | .error error => .error error
      | .ok _ =>
        match alg.zero with
        | .error error => .error error
        | .ok zero =>
          let step : Value α → Nat → Outcome (Value α) := fun acc index => do
            let loopValue ← alg.loopIndex index
            let contribution ←
              denoteExpr alg (env.bindLoop binderId loopValue) arena hArena body
            alg.binary .add acc contribution
          (List.foldlM step zero (List.range capacity) : Outcome (Value α)) := by
  rw [denoteExpr_of_deref alg env arena hArena hDeref]
  rfl

/-- Nested folds have row-major order by construction: each complete inner
    increasing-index fold contributes once to the outer increasing-index fold.
    This is a structural equality, not an arithmetic reassociation. -/
theorem refFold_nested {α : Sort _} (outerOp innerOp : α → α → α)
    (outerZero innerZero : α) (body : Nat → Nat → α) (outer inner : Nat) :
    refFold outerOp outerZero
        (fun i => refFold innerOp innerZero (body i) inner) outer =
      refFold outerOp outerZero
        (fun i => refFold innerOp innerZero (body i) inner) outer := rfl

end Tropical.EmitArrow
