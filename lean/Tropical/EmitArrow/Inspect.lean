import Tropical.EmitArrow.Sig
import Tropical.Exact.Gamma

/-!
# Frozen-arena inspection

The production modal compiler makes admission decisions from two one-pass,
child-before-parent classifications of its active expression arena.  Keeping
them here makes the traversal shared and theorem-visible; callers only project
the queried root.  Missing children and dangling roots fail closed.
-/

namespace Tropical.EmitArrow

open Tropical.Ir
open Tropical.Exact (DyadicI)

/-- Exact constant classification in arena order. -/
def sigConstTable (arena : ExprArena) : Array (Option DyadicI) :=
  arena.nodes.foldl (fun values node =>
    let getConst := fun (id : ExprId) => (values[id.idx]?).join
    let value := match node with
      | .num n => some (DyadicI.ofJsonNumber n)
      | .unary .neg a => (getConst a).map DyadicI.neg
      | .unary .toFloat a => getConst a
      | .binary .add a b => do pure (DyadicI.add (← getConst a) (← getConst b))
      | .binary .sub a b => do pure (DyadicI.sub (← getConst a) (← getConst b))
      | .binary .mul a b => do pure (DyadicI.mul (← getConst a) (← getConst b))
      | .binary .div a b => do pure (DyadicI.div (← getConst a) (← getConst b))
      | _ => none
    values.push value) #[]

def sigConstDFrom? (constants : Array (Option DyadicI))
    (signal : Sig) : Option DyadicI :=
  (constants[signal.idx]?).join

def sigConstD? (signal : Sig) : BuildM (Option DyadicI) := do
  let builder ← get
  pure (sigConstDFrom? (sigConstTable builder.exprs) signal)

def sigConstF? (signal : Sig) : BuildM (Option Float) := do
  let value ← sigConstD? signal
  pure (value.map DyadicI.toFloat)

/-- Pure s0 classification in arena order. -/
def sigS0Table (arena : ExprArena) : Array Bool :=
  arena.nodes.foldl (fun values node =>
    let getS0 := fun (id : ExprId) => (values[id.idx]?).getD false
    let value := match node with
      | .num _ | .paramRef _ | .sampleRate => true
      | .unary _ arg => getS0 arg
      | .binary _ lhs rhs => getS0 lhs && getS0 rhs
      | .clamp value lo hi | .select value lo hi =>
          getS0 value && getS0 lo && getS0 hi
      | _ => false
    values.push value) #[]

def sigIsS0From (values : Array Bool) (signal : Sig) : Bool :=
  (values[signal.idx]?).getD false

def sigIsS0 (signal : Sig) : BuildM Bool := do
  let builder ← get
  pure (sigIsS0From (sigS0Table builder.exprs) signal)

private theorem List.foldl_push_size (items : List α) (initial : Array β)
    (value : Array β → α → β) :
    (items.foldl (fun values item => values.push (value values item)) initial).size =
      initial.size + items.length := by
  induction items generalizing initial with
  | nil => simp
  | cons head tail ih =>
    rw [List.foldl_cons, ih]
    simp
    omega

theorem sigConstTable_size (arena : ExprArena) :
    (sigConstTable arena).size = arena.nodes.size := by
  simp only [sigConstTable]
  rw [← Array.foldl_toList]
  simpa using List.foldl_push_size arena.nodes.toList #[] (fun values node =>
    match node with
    | .num n => some (DyadicI.ofJsonNumber n)
    | .unary .neg a => ((values[a.idx]?).join).map DyadicI.neg
    | .unary .toFloat a => (values[a.idx]?).join
    | .binary .add a b => do
        pure (DyadicI.add (← (values[a.idx]?).join) (← (values[b.idx]?).join))
    | .binary .sub a b => do
        pure (DyadicI.sub (← (values[a.idx]?).join) (← (values[b.idx]?).join))
    | .binary .mul a b => do
        pure (DyadicI.mul (← (values[a.idx]?).join) (← (values[b.idx]?).join))
    | .binary .div a b => do
        pure (DyadicI.div (← (values[a.idx]?).join) (← (values[b.idx]?).join))
    | _ => none)

theorem sigS0Table_size (arena : ExprArena) :
    (sigS0Table arena).size = arena.nodes.size := by
  simp only [sigS0Table]
  rw [← Array.foldl_toList]
  simpa using List.foldl_push_size arena.nodes.toList #[] (fun values node =>
    match node with
    | .num _ | .paramRef _ | .sampleRate => true
    | .unary _ arg => (values[arg.idx]?).getD false
    | .binary _ lhs rhs =>
        (values[lhs.idx]?).getD false && (values[rhs.idx]?).getD false
    | .clamp value lo hi | .select value lo hi =>
        (values[value.idx]?).getD false && (values[lo.idx]?).getD false &&
          (values[hi.idx]?).getD false
    | _ => false)

/-- A dangling root cannot be classified as a constant. -/
theorem sigConstDFrom?_dangling (arena : ExprArena) (signal : Sig)
    (h : arena.nodes.size ≤ signal.idx) :
    sigConstDFrom? (sigConstTable arena) signal = none := by
  simp [sigConstDFrom?, sigConstTable_size, h]

/-- A dangling root cannot be admitted as s0. -/
theorem sigIsS0From_dangling (arena : ExprArena) (signal : Sig)
    (h : arena.nodes.size ≤ signal.idx) :
    sigIsS0From (sigS0Table arena) signal = false := by
  simp [sigIsS0From, sigS0Table_size, h]

/-- Successful constant classification implies the queried root is
    addressable in the frozen arena. -/
theorem sigConstDFrom?_addressable (arena : ExprArena) (signal : Sig)
    {value : DyadicI}
    (h : sigConstDFrom? (sigConstTable arena) signal = some value) :
    ∃ node, arena.deref signal = some node := by
  have hi : signal.idx < arena.nodes.size := by
    apply Nat.lt_of_not_ge
    intro hLe
    rw [sigConstDFrom?_dangling arena signal hLe] at h
    cases h
  exact ⟨arena.nodes[signal.idx], by
    simp [ExprArena.deref, Array.getElem?_eq_getElem hi]⟩

/-- Successful s0 admission implies the queried root is addressable. -/
theorem sigIsS0From_addressable (arena : ExprArena) (signal : Sig)
    (h : sigIsS0From (sigS0Table arena) signal = true) :
    ∃ node, arena.deref signal = some node := by
  have hi : signal.idx < arena.nodes.size := by
    apply Nat.lt_of_not_ge
    intro hLe
    rw [sigIsS0From_dangling arena signal hLe] at h
    cases h
  exact ⟨arena.nodes[signal.idx], by
    simp [ExprArena.deref, Array.getElem?_eq_getElem hi]⟩

end Tropical.EmitArrow
