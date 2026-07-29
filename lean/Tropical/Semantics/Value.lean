import Tropical.Ir.Nodes

/-!
# Semantic values and explicit refusal

The semantic spine never assigns a convenient value to a failed lookup or an
ill-typed scalar operation.  Every such case is represented by `Refusal`.
`Value` is deliberately independent of the runtime's current scalar carrier:
future proofs may instantiate it with exact integers, floats, dyadics, or a
symbolic algebra without changing the production syntax being interpreted.
-/

namespace Tropical.Semantics

/-- A modeled semantic refusal.  The payload is explanatory data, not a proof
    obligation, so equality of denotations remains executable. -/
structure Refusal where
  operation : String
  detail : String
deriving BEq, Repr, Inhabited

/-- Values admitted by the current `Sig` vocabulary.  Arrays may be nested
    because the production syntax does not prohibit nested literals. -/
inductive Value (α : Type) where
  | scalar (value : α)
  | array (items : Array (Value α))
deriving Repr, Inhabited

/-- A computation that either produces `β` or names its refusal. -/
abbrev Outcome (β : Type) := Except Refusal β

/-- Every expression computation either produces a value or names its refusal. -/
abbrev Result (α : Type) := Outcome (Value α)

def refusal (operation detail : String) : Result α :=
  .error { operation, detail }

/-- Sequence array-valued semantic computations from left to right.  This is
    also the evaluation order used for literal arrays and bank tables. -/
def sequence (xs : Array (Result α)) : Outcome (Array (Value α)) :=
  xs.foldlM (fun acc x => acc.push <$> x) #[]

end Tropical.Semantics
