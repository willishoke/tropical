import Tropical.EmitArrow.Sig

/-!
# Arena-native clock algebra (phase 4)

The recursive `ClockAlgebra` theorem talks about an authored `Sig` tree. The
production authoring path no longer has such a tree: it interns each operation
immediately and retains only an `ExprId` in one frozen `ExprArena`.

This module restates the clock-rail judgment directly on that representation.
Every constructor carries the exact `ExprArena.deref` equation for its node;
children remain ids in the same arena, and `denoteClock` recurses on the rail
derivation rather than rebuilding an expression tree. The five structural
clock laws therefore quantify over arena nodes and require no lowering or
tree-to-id correspondence theorem.

`CLOCK_RAIL_IS_EXACT` remains the same backend boundary: per sample, LLVM and
MSL must compute the two's-complement i64 image of this `Int` denotation. This
module proves the frontend algebra only.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- Values supplied at the two leaves of the integer clock rail. A boundary is
    keyed by its source `ExprId`; structural interning makes equal source
    subgraphs use the same key. -/
structure ClockEnv where
  tick : Int
  boundary : ExprId → Int

/-- The Q32.32 clock fragment, stated on ids in one frozen expression arena.

    The judgment is intentionally `Type`-valued: its derivation is the total
    recursion argument for `denoteClock`. Shift amounts are literal natural
    numbers, and bit masks have the production low-bits shape `2^k - 1`.
    Unsupported and dangling nodes cannot produce a derivation. -/
inductive OnClockRail (arena : ExprArena) : ExprId → Type where
  | intLit (id : ExprId) (m : Int)
      (node : arena.deref id = some (.num ⟨m, 0⟩)) :
      OnClockRail arena id
  | tick (id : ExprId)
      (node : arena.deref id = some .sampleIndex) :
      OnClockRail arena id
  | boundary (id source : ExprId)
      (node : arena.deref id = some (.unary .toInt source)) :
      OnClockRail arena id
  | neg {id arg : ExprId}
      (node : arena.deref id = some (.unary .neg arg))
      (argRail : OnClockRail arena arg) :
      OnClockRail arena id
  | add {id lhs rhs : ExprId}
      (node : arena.deref id = some (.binary .add lhs rhs))
      (lhsRail : OnClockRail arena lhs) (rhsRail : OnClockRail arena rhs) :
      OnClockRail arena id
  | sub {id lhs rhs : ExprId}
      (node : arena.deref id = some (.binary .sub lhs rhs))
      (lhsRail : OnClockRail arena lhs) (rhsRail : OnClockRail arena rhs) :
      OnClockRail arena id
  | mul {id lhs rhs : ExprId}
      (node : arena.deref id = some (.binary .mul lhs rhs))
      (lhsRail : OnClockRail arena lhs) (rhsRail : OnClockRail arena rhs) :
      OnClockRail arena id
  | lshift {id arg amount : ExprId} (m : Int) (k : Nat)
      (hm : m = (k : Int))
      (node : arena.deref id = some (.binary .lshift arg amount))
      (amountNode : arena.deref amount = some (.num ⟨m, 0⟩))
      (argRail : OnClockRail arena arg) :
      OnClockRail arena id
  | rshift {id arg amount : ExprId} (m : Int) (k : Nat)
      (hm : m = (k : Int))
      (node : arena.deref id = some (.binary .rshift arg amount))
      (amountNode : arena.deref amount = some (.num ⟨m, 0⟩))
      (argRail : OnClockRail arena arg) :
      OnClockRail arena id
  | mask {id arg amount : ExprId} (m : Int) (k : Nat)
      (hm : m = 2 ^ k - 1)
      (node : arena.deref id = some (.binary .bitAnd arg amount))
      (amountNode : arena.deref amount = some (.num ⟨m, 0⟩))
      (argRail : OnClockRail arena arg) :
      OnClockRail arena id

/-- Total `Int` denotation of an arena-native clock-rail derivation. -/
def denoteClock {arena : ExprArena} :
    {id : ExprId} → OnClockRail arena id → ClockEnv → Int
  | _, .intLit _ m _, _ => m
  | _, .tick _ _, env => env.tick
  | _, .boundary _ source _, env => env.boundary source
  | _, .neg _ argRail, env => -(denoteClock argRail env)
  | _, .add _ lhsRail rhsRail, env =>
      denoteClock lhsRail env + denoteClock rhsRail env
  | _, .sub _ lhsRail rhsRail, env =>
      denoteClock lhsRail env - denoteClock rhsRail env
  | _, .mul _ lhsRail rhsRail, env =>
      denoteClock lhsRail env * denoteClock rhsRail env
  | _, .lshift _ k _ _ _ argRail, env => denoteClock argRail env <<< k
  | _, .rshift _ k _ _ _ argRail, env => denoteClock argRail env >>> k
  | _, .mask _ k _ _ _ argRail, env => denoteClock argRail env % 2 ^ k

-- The laws receive dereference equations for the newly interned parent nodes.
-- This is the native replacement for constructing a larger recursive `Sig`
-- in the theorem statement.

/-- Law 1, arena-native: `(c + δ) - δ = c`. -/
theorem warp_inv {arena : ExprArena} {c delta sum root : ExprId}
    (clockRail : OnClockRail arena c) (deltaRail : OnClockRail arena delta)
    (sumNode : arena.deref sum = some (.binary .add c delta))
    (rootNode : arena.deref root = some (.binary .sub sum delta))
    (env : ClockEnv) :
    denoteClock (.sub rootNode (.add sumNode clockRail deltaRail) deltaRail) env =
      denoteClock clockRail env := by
  simp [denoteClock]

/-- Law 1 in the other direction: `(c - δ) + δ = c`. -/
theorem warp_inv' {arena : ExprArena} {c delta difference root : ExprId}
    (clockRail : OnClockRail arena c) (deltaRail : OnClockRail arena delta)
    (differenceNode : arena.deref difference = some (.binary .sub c delta))
    (rootNode : arena.deref root = some (.binary .add difference delta))
    (env : ClockEnv) :
    denoteClock
        (.add rootNode (.sub differenceNode clockRail deltaRail) deltaRail) env =
      denoteClock clockRail env := by
  simp [denoteClock]

/-- Law 2, arena-native: `(c - δ₁) - δ₂ = c - (δ₁ + δ₂)`. -/
theorem warp_assoc {arena : ExprArena}
    {c delta1 delta2 leftInner leftRoot rightInner rightRoot : ExprId}
    (clockRail : OnClockRail arena c)
    (delta1Rail : OnClockRail arena delta1)
    (delta2Rail : OnClockRail arena delta2)
    (leftInnerNode : arena.deref leftInner = some (.binary .sub c delta1))
    (leftRootNode : arena.deref leftRoot = some (.binary .sub leftInner delta2))
    (rightInnerNode : arena.deref rightInner = some (.binary .add delta1 delta2))
    (rightRootNode : arena.deref rightRoot = some (.binary .sub c rightInner))
    (env : ClockEnv) :
    denoteClock
        (.sub leftRootNode (.sub leftInnerNode clockRail delta1Rail) delta2Rail) env =
      denoteClock
        (.sub rightRootNode clockRail
          (.add rightInnerNode delta1Rail delta2Rail)) env := by
  simp [denoteClock]
  omega

/-- Law 4, arena-native: `-(-c) = c`. -/
theorem rev_involution {arena : ExprArena} {c inner root : ExprId}
    (clockRail : OnClockRail arena c)
    (innerNode : arena.deref inner = some (.unary .neg c))
    (rootNode : arena.deref root = some (.unary .neg inner))
    (env : ClockEnv) :
    denoteClock (.neg rootNode (.neg innerNode clockRail)) env =
      denoteClock clockRail env := by
  simp [denoteClock]

/-- Law 5, arena-native: `-(c - δ) = (-c) + δ`. -/
theorem rev_swap {arena : ExprArena}
    {c delta leftInner leftRoot rightInner rightRoot : ExprId}
    (clockRail : OnClockRail arena c) (deltaRail : OnClockRail arena delta)
    (leftInnerNode : arena.deref leftInner = some (.binary .sub c delta))
    (leftRootNode : arena.deref leftRoot = some (.unary .neg leftInner))
    (rightInnerNode : arena.deref rightInner = some (.unary .neg c))
    (rightRootNode : arena.deref rightRoot = some (.binary .add rightInner delta))
    (env : ClockEnv) :
    denoteClock (.neg leftRootNode (.sub leftInnerNode clockRail deltaRail)) env =
      denoteClock
        (.add rightRootNode (.neg rightInnerNode clockRail) deltaRail) env := by
  simp [denoteClock]
  omega

/-- The exact split identity used by the native phase reducers, including for
    negative relative clocks. -/
theorem rail_split_identity (x : Int) :
    (x >>> (32 : Nat)) * 2 ^ 32 + x % 2 ^ 32 = x := by
  rw [Int.shiftRight_eq_div_pow]
  omega

end Tropical.EmitArrow
