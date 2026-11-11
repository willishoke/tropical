import Tropical.Ir.ConstFold
import Init.Data.Int.Bitwise.Lemmas
import Init.Data.Int.DivMod.Lemmas

/-!
# ConstFoldLaws — the integer constant folder as i64 and i32 algebras

`ConstFold.lean` says its integer paths reproduce i64 and i32 two's-complement
execution. This module makes the integer half of that sentence checked:

* `wrap64` is the canonical representative in `[-2^63, 2^63)`, fixes every
  value already in range, is idempotent, and preserves the residue modulo
  `2^64`;
* wrapping commutes with add, sub, mul, and neg, so the arbitrary-precision
  `Int` implementation realizes the quotient ring `ℤ/2^64ℤ`;
* `toU64` is exactly the nonnegative residue and `bit64` always returns an
  i64-range value;
* every integer-only `foldOp` route is pinned to its intended equation, with
  no hypothesis — the integer semantics is total: arithmetic (a zero divisor
  gives 0, `MIN / -1` wraps to `MIN`), comparisons, truthiness, bitwise ops,
  shifts (the amount reduced modulo 64), casts, clamp, and select;
* the same for `int32`: `wrap32` is the canonical representative modulo
  `2^32` and a ring homomorphism, `wrap32 ∘ wrap64 = wrap32` (an i32 result is
  the low word of the i64 one), each int32 `foldOp` route is pinned (shifts
  reduce the amount modulo 32, `INT32_MIN / -1 = INT32_MIN`,
  `INT32_MIN % -1 = 0`), and the wrap edge cases are evaluated outright;
* float → int conversion is saturating: `foldOp` routes it through
  `truncSatToInt`/`truncSatToInt32`, whose results always lie in the target
  word (`truncSatToInt_isI64`, `truncSatToInt32_isI32`).

No other Float claim is made here: `toFloat`, Float comparisons,
transcendentals, and which integer a finite in-range float truncates to
remain outside this proof boundary.
-/

namespace Tropical.Ir

open Tropical.Plan (PlanOp)

/-- `2^63`, the positive exclusive bound of a signed i64. -/
def i64Half : Int := 9223372036854775808

/-- `2^64`, the modulus of two's-complement i64 arithmetic. -/
def i64Modulus : Int := 18446744073709551616

/-- The mathematical signed-i64 range. -/
def IsI64 (x : Int) : Prop := -i64Half ≤ x ∧ x < i64Half

private theorem scalar_int_beq_int :
    ((Tropical.Parse.ScalarKind.int == Tropical.Parse.ScalarKind.int) = true) := by
  decide

private theorem scalar_int_beq_bool :
    ((Tropical.Parse.ScalarKind.int == Tropical.Parse.ScalarKind.bool) = false) := by
  decide

@[local simp] private theorem scalar_int_beq_int32 :
    ((Tropical.Parse.ScalarKind.int == Tropical.Parse.ScalarKind.int32) = false) := by
  decide

@[local simp] private theorem scalar_bool_beq_int32 :
    ((Tropical.Parse.ScalarKind.bool == Tropical.Parse.ScalarKind.int32) = false) := by
  decide

@[local simp] private theorem scalar_int32_beq_int32 :
    ((Tropical.Parse.ScalarKind.int32 == Tropical.Parse.ScalarKind.int32) = true) := by
  decide

@[local simp] private theorem scalar_int32_beq_int :
    ((Tropical.Parse.ScalarKind.int32 == Tropical.Parse.ScalarKind.int) = false) := by
  decide

@[local simp] private theorem scalar_float_beq_int :
    ((Tropical.Parse.ScalarKind.float == Tropical.Parse.ScalarKind.int) = false) := by
  decide

@[local simp] private theorem scalar_float_beq_int32 :
    ((Tropical.Parse.ScalarKind.float == Tropical.Parse.ScalarKind.int32) = false) := by
  decide

-- ─────────────────────────────────────────────────────────────
-- wrap64 — canonical quotient representative
-- ─────────────────────────────────────────────────────────────

/-- Every wrapped integer is in the signed-i64 range. -/
theorem wrap64_isI64 (x : Int) : IsI64 (wrap64 x) := by
  constructor
  · unfold i64Half wrap64
    have h := Int.emod_nonneg
      (x + 9223372036854775808)
      (by omega : (18446744073709551616 : Int) ≠ 0)
    omega
  · unfold i64Half wrap64
    have h := Int.emod_lt_of_pos
      (x + 9223372036854775808)
      (by omega : (0 : Int) < 18446744073709551616)
    omega

/-- Wrapping is the identity on values already representable as i64. -/
theorem wrap64_eq_self {x : Int} (hx : IsI64 x) : wrap64 x = x := by
  unfold IsI64 i64Half at hx
  unfold wrap64
  rw [Int.emod_eq_of_lt] <;> omega

/-- Wrapping preserves exactly the residue modulo `2^64`. -/
theorem wrap64_emod (x : Int) :
    wrap64 x % i64Modulus = x % i64Modulus := by
  unfold i64Modulus wrap64
  rw [Int.emod_sub_emod]
  simp

/-- Equal residues have equal canonical signed representatives. -/
theorem wrap64_eq_of_emod_eq {x y : Int}
    (h : x % i64Modulus = y % i64Modulus) :
    wrap64 x = wrap64 y := by
  unfold i64Modulus at h
  unfold wrap64
  rw [← Int.emod_add_emod x 18446744073709551616 9223372036854775808,
      ← Int.emod_add_emod y 18446744073709551616 9223372036854775808, h]

/-- `wrap64` is a canonicalization: applying it twice changes nothing. -/
theorem wrap64_idem (x : Int) : wrap64 (wrap64 x) = wrap64 x := by
  exact wrap64_eq_self (wrap64_isI64 x)

/-- Addition in the canonical representatives is addition modulo `2^64`. -/
theorem wrap64_add (a b : Int) :
    wrap64 (wrap64 a + wrap64 b) = wrap64 (a + b) := by
  apply wrap64_eq_of_emod_eq
  calc
    (wrap64 a + wrap64 b) % i64Modulus =
        (wrap64 a % i64Modulus +
          wrap64 b % i64Modulus) % i64Modulus :=
      Int.add_emod _ _ _
    _ = (a % i64Modulus +
          b % i64Modulus) % i64Modulus := by
      rw [wrap64_emod, wrap64_emod]
    _ = (a + b) % i64Modulus := (Int.add_emod _ _ _).symm

/-- Subtraction in the canonical representatives is subtraction modulo `2^64`. -/
theorem wrap64_sub (a b : Int) :
    wrap64 (wrap64 a - wrap64 b) = wrap64 (a - b) := by
  apply wrap64_eq_of_emod_eq
  calc
    (wrap64 a - wrap64 b) % i64Modulus =
        (wrap64 a % i64Modulus -
          wrap64 b % i64Modulus) % i64Modulus :=
      Int.sub_emod _ _ _
    _ = (a % i64Modulus -
          b % i64Modulus) % i64Modulus := by
      rw [wrap64_emod, wrap64_emod]
    _ = (a - b) % i64Modulus := (Int.sub_emod _ _ _).symm

/-- Multiplication in the canonical representatives is multiplication modulo `2^64`. -/
theorem wrap64_mul (a b : Int) :
    wrap64 (wrap64 a * wrap64 b) = wrap64 (a * b) := by
  apply wrap64_eq_of_emod_eq
  calc
    (wrap64 a * wrap64 b) % i64Modulus =
        (wrap64 a % i64Modulus *
          (wrap64 b % i64Modulus)) % i64Modulus :=
      Int.mul_emod _ _ _
    _ = (a % i64Modulus *
          (b % i64Modulus)) % i64Modulus := by
      rw [wrap64_emod, wrap64_emod]
    _ = (a * b) % i64Modulus := (Int.mul_emod _ _ _).symm

/-- Negation in the canonical representatives is negation modulo `2^64`. -/
theorem wrap64_neg (a : Int) :
    wrap64 (-wrap64 a) = wrap64 (-a) := by
  apply wrap64_eq_of_emod_eq
  rw [Int.neg_emod_eq_sub_emod, Int.neg_emod_eq_sub_emod]
  exact (Int.emod_sub_cancel_left i64Modulus).2 (wrap64_emod a)

-- ─────────────────────────────────────────────────────────────
-- Unsigned image and bitwise result range
-- ─────────────────────────────────────────────────────────────

/-- `toU64` is exactly the least nonnegative residue modulo `2^64`. -/
theorem ofNat_toU64 (x : Int) :
    Int.ofNat (toU64 x) = x % i64Modulus := by
  unfold i64Modulus toU64
  have hx0 : 0 ≤ x % (18446744073709551616 : Int) :=
    Int.emod_nonneg _ (by omega)
  rw [Int.add_emod]
  simp only [Int.emod_emod, Int.emod_self, Int.add_zero]
  exact Int.toNat_of_nonneg hx0

/-- The unsigned image fits in exactly 64 bits. -/
theorem toU64_lt (x : Int) : toU64 x < 18446744073709551616 := by
  have hxlt : x % i64Modulus < i64Modulus := by
    unfold i64Modulus
    exact Int.emod_lt_of_pos _ (by omega)
  rw [← ofNat_toU64] at hxlt
  exact Int.ofNat_lt.mp hxlt

/-- Any natural bit-operation, reinterpreted through `bit64`, lands in i64. -/
theorem bit64_isI64 (f : Nat → Nat → Nat) (a b : Int) :
    IsI64 (bit64 f a b) := by
  exact wrap64_isI64 _

-- ─────────────────────────────────────────────────────────────
-- foldOp — arithmetic and zero guards
-- ─────────────────────────────────────────────────────────────

theorem foldOp_add_int (a b : Int) :
    foldOp PlanOp.add.name .int #[.i a, .i b] =
      some (.i (wrap64 (a + b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_sub_int (a b : Int) :
    foldOp PlanOp.sub.name .int #[.i a, .i b] =
      some (.i (wrap64 (a - b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_mul_int (a b : Int) :
    foldOp PlanOp.mul.name .int #[.i a, .i b] =
      some (.i (wrap64 (a * b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_div_int (a b : Int) :
    foldOp PlanOp.div.name .int #[.i a, .i b] =
      some (.i (if b == 0 then 0 else wrap64 (Int.tdiv a b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_mod_int (a b : Int) :
    foldOp PlanOp.mod.name .int #[.i a, .i b] =
      some (.i (if b == 0 then 0 else wrap64 (Int.tmod a b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_floorDiv_int (a b : Int) :
    foldOp PlanOp.floorDiv.name .int #[.i a, .i b] =
      some (.i (if b == 0 then 0 else wrap64 (Int.tdiv a b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_neg_int (a : Int) :
    foldOp PlanOp.neg.name .int #[.i a] =
      some (.i (wrap64 (-a))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_abs_int (a : Int) :
    foldOp PlanOp.abs.name .int #[.i a] =
      some (.i (if a < 0 then wrap64 (-a) else a)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

-- ─────────────────────────────────────────────────────────────
-- foldOp — integer comparisons and truthiness
-- ─────────────────────────────────────────────────────────────

theorem foldOp_less_int (a b : Int) :
    foldOp PlanOp.less.name .bool #[.i a, .i b] =
      some (.b (decide (a < b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_lessEq_int (a b : Int) :
    foldOp PlanOp.lessEq.name .bool #[.i a, .i b] =
      some (.b (decide (a ≤ b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_greater_int (a b : Int) :
    foldOp PlanOp.greater.name .bool #[.i a, .i b] =
      some (.b (decide (a > b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_greaterEq_int (a b : Int) :
    foldOp PlanOp.greaterEq.name .bool #[.i a, .i b] =
      some (.b (decide (a ≥ b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_equal_int (a b : Int) :
    foldOp PlanOp.equal.name .bool #[.i a, .i b] =
      some (.b (a == b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_notEqual_int (a b : Int) :
    foldOp PlanOp.notEqual.name .bool #[.i a, .i b] =
      some (.b (a != b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_and_int (a b : Int) :
    foldOp PlanOp.and.name .bool #[.i a, .i b] =
      some (.b ((a != 0) && (b != 0))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_bool]

theorem foldOp_or_int (a b : Int) :
    foldOp PlanOp.or.name .bool #[.i a, .i b] =
      some (.b ((a != 0) || (b != 0))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_bool]

theorem foldOp_not_int (a : Int) :
    foldOp PlanOp.not.name .bool #[.i a] =
      some (.b (a == 0)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?]

theorem foldOp_toInt_int (a : Int) :
    foldOp PlanOp.toInt.name .int #[.i a] = some (.i a) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_toBool_int (a : Int) :
    foldOp PlanOp.toBool.name .bool #[.i a] = some (.b (a != 0)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_bool]

-- ─────────────────────────────────────────────────────────────
-- foldOp — bitwise ops and shifts
-- ─────────────────────────────────────────────────────────────

theorem foldOp_bitAnd_int (a b : Int) :
    foldOp PlanOp.bitAnd.name .int #[.i a, .i b] =
      some (.i (bit64 (· &&& ·) a b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_bitOr_int (a b : Int) :
    foldOp PlanOp.bitOr.name .int #[.i a, .i b] =
      some (.i (bit64 (· ||| ·) a b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_bitXor_int (a b : Int) :
    foldOp PlanOp.bitXor.name .int #[.i a, .i b] =
      some (.i (bit64 (· ^^^ ·) a b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_bitNot_int (a : Int) :
    foldOp PlanOp.bitNot.name .int #[.i a] =
      some (.i (bit64 (· ^^^ ·) a (-1))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- A left shift takes its amount modulo 64 (`Int`'s `%` is the least
    nonnegative residue, so `-1` shifts by 63). -/
theorem foldOp_lshift_int (a sh : Int) :
    foldOp PlanOp.lshift.name .int #[.i a, .i sh] =
      some (.i (wrap64 (a <<< (sh % 64).toNat))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- An arithmetic right shift takes its amount modulo 64. -/
theorem foldOp_rshift_int (a sh : Int) :
    foldOp PlanOp.rshift.name .int #[.i a, .i sh] =
      some (.i (a >>> (sh % 64).toNat)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- Within `[0, 63]` the reduction is the identity: the ordinary shift. -/
theorem foldOp_lshift_int_inRange {a sh : Int} (h0 : 0 ≤ sh) (h63 : sh ≤ 63) :
    foldOp PlanOp.lshift.name .int #[.i a, .i sh] =
      some (.i (wrap64 (a <<< sh.toNat))) := by
  rw [foldOp_lshift_int, Int.emod_eq_of_lt h0 (by omega)]

theorem foldOp_rshift_int_inRange {a sh : Int} (h0 : 0 ≤ sh) (h63 : sh ≤ 63) :
    foldOp PlanOp.rshift.name .int #[.i a, .i sh] =
      some (.i (a >>> sh.toNat)) := by
  rw [foldOp_rshift_int, Int.emod_eq_of_lt h0 (by omega)]

/-- The reduced amount is always a valid shift for the word. -/
theorem shiftAmount64_lt (sh : Int) : (sh % 64).toNat < 64 := by
  have h0 := Int.emod_nonneg sh (by decide : (64 : Int) ≠ 0)
  have h1 := Int.emod_lt_of_pos sh (by decide : (0 : Int) < 64)
  omega

-- ─────────────────────────────────────────────────────────────
-- foldOp — integer clamp and select
-- ─────────────────────────────────────────────────────────────

theorem foldOp_clamp_int (v lo hi : Int) :
    foldOp PlanOp.clamp.name .int #[.i v, .i lo, .i hi] =
      some (.i (let lc := if v > lo then v else lo
        if lc < hi then lc else hi)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
    scalar_int_beq_int]

theorem foldOp_select_int (c t f : Int) :
    foldOp PlanOp.select.name .int #[.i c, .i t, .i f] =
      some (.i (if c != 0 then t else f)) := by
  by_cases hc : c = 0 <;>
    simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty,
      scalar_int_beq_int, hc]

-- ─────────────────────────────────────────────────────────────
-- int32 — wrap32 as the canonical representative modulo 2^32
-- ─────────────────────────────────────────────────────────────

/-- `2^31`, the positive exclusive bound of a signed i32. -/
def i32Half : Int := 2147483648

/-- `2^32`, the modulus of two's-complement i32 arithmetic. -/
def i32Modulus : Int := 4294967296

/-- The mathematical signed-i32 range. -/
def IsI32 (x : Int) : Prop := -i32Half ≤ x ∧ x < i32Half

theorem wrap32_isI32 (x : Int) : IsI32 (wrap32 x) := by
  constructor
  · unfold i32Half wrap32
    have h := Int.emod_nonneg (x + 2147483648) (by omega : (4294967296 : Int) ≠ 0)
    omega
  · unfold i32Half wrap32
    have h := Int.emod_lt_of_pos (x + 2147483648) (by omega : (0 : Int) < 4294967296)
    omega

theorem wrap32_eq_self {x : Int} (hx : IsI32 x) : wrap32 x = x := by
  unfold IsI32 i32Half at hx
  unfold wrap32
  rw [Int.emod_eq_of_lt] <;> omega

theorem wrap32_emod (x : Int) : wrap32 x % i32Modulus = x % i32Modulus := by
  unfold i32Modulus wrap32
  rw [Int.emod_sub_emod]
  simp

theorem wrap32_eq_of_emod_eq {x y : Int} (h : x % i32Modulus = y % i32Modulus) :
    wrap32 x = wrap32 y := by
  unfold i32Modulus at h
  unfold wrap32
  rw [← Int.emod_add_emod x 4294967296 2147483648,
      ← Int.emod_add_emod y 4294967296 2147483648, h]

theorem wrap32_idem (x : Int) : wrap32 (wrap32 x) = wrap32 x :=
  wrap32_eq_self (wrap32_isI32 x)

theorem wrap32_add (a b : Int) : wrap32 (wrap32 a + wrap32 b) = wrap32 (a + b) := by
  apply wrap32_eq_of_emod_eq
  rw [Int.add_emod, wrap32_emod, wrap32_emod, ← Int.add_emod]

theorem wrap32_sub (a b : Int) : wrap32 (wrap32 a - wrap32 b) = wrap32 (a - b) := by
  apply wrap32_eq_of_emod_eq
  rw [Int.sub_emod, wrap32_emod, wrap32_emod, ← Int.sub_emod]

theorem wrap32_mul (a b : Int) : wrap32 (wrap32 a * wrap32 b) = wrap32 (a * b) := by
  apply wrap32_eq_of_emod_eq
  rw [Int.mul_emod, wrap32_emod, wrap32_emod, ← Int.mul_emod]

theorem wrap32_neg (a : Int) : wrap32 (-wrap32 a) = wrap32 (-a) := by
  apply wrap32_eq_of_emod_eq
  rw [Int.neg_emod_eq_sub_emod, Int.neg_emod_eq_sub_emod]
  exact (Int.emod_sub_cancel_left i32Modulus).2 (wrap32_emod a)

/-- An i32 result is the low word of the i64 one: `2^32` divides `2^64`, so
    wrapping to i64 first changes nothing modulo `2^32`. With the ring laws
    above, this is why truncating an int computation to int32 equals doing
    it in int32 (for add, sub, mul and neg). -/
theorem wrap32_wrap64 (x : Int) : wrap32 (wrap64 x) = wrap32 x := by
  apply wrap32_eq_of_emod_eq
  have h := wrap64_emod x
  unfold i64Modulus at h
  unfold i32Modulus
  have hd : (4294967296 : Int) ∣ 18446744073709551616 := ⟨4294967296, by rfl⟩
  rw [← Int.emod_emod_of_dvd (wrap64 x) hd, ← Int.emod_emod_of_dvd x hd, h]

theorem ofNat_toU32 (x : Int) : Int.ofNat (toU32 x) = x % i32Modulus := by
  unfold i32Modulus toU32
  have hx0 : 0 ≤ x % (4294967296 : Int) := Int.emod_nonneg _ (by omega)
  rw [Int.add_emod]
  simp only [Int.emod_emod, Int.emod_self, Int.add_zero]
  exact Int.toNat_of_nonneg hx0

theorem bit32_isI32 (f : Nat → Nat → Nat) (a b : Int) : IsI32 (bit32 f a b) :=
  wrap32_isI32 _

-- ─────────────────────────────────────────────────────────────
-- foldOp — the int32 routes
-- ─────────────────────────────────────────────────────────────

theorem foldOp_add_int32 (a b : Int) :
    foldOp PlanOp.add.name .int32 #[.i32 a, .i32 b] = some (.i32 (wrap32 (a + b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_sub_int32 (a b : Int) :
    foldOp PlanOp.sub.name .int32 #[.i32 a, .i32 b] = some (.i32 (wrap32 (a - b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_mul_int32 (a b : Int) :
    foldOp PlanOp.mul.name .int32 #[.i32 a, .i32 b] = some (.i32 (wrap32 (a * b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_neg_int32 (a : Int) :
    foldOp PlanOp.neg.name .int32 #[.i32 a] = some (.i32 (wrap32 (-a))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_abs_int32 (a : Int) :
    foldOp PlanOp.abs.name .int32 #[.i32 a] =
      some (.i32 (if a < 0 then wrap32 (-a) else a)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- int32 division: 0 for a zero divisor, else the truncated quotient
    wrapped to the word — total, like `int`'s. -/
theorem foldOp_div_int32 (a b : Int) :
    foldOp PlanOp.div.name .int32 #[.i32 a, .i32 b] =
      some (.i32 (if b == 0 then 0 else wrap32 (Int.tdiv a b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_mod_int32 (a b : Int) :
    foldOp PlanOp.mod.name .int32 #[.i32 a, .i32 b] =
      some (.i32 (if b == 0 then 0 else wrap32 (Int.tmod a b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_floorDiv_int32 (a b : Int) :
    foldOp PlanOp.floorDiv.name .int32 #[.i32 a, .i32 b] =
      some (.i32 (if b == 0 then 0 else wrap32 (Int.tdiv a b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- The overflowing quotient wraps: `INT32_MIN / -1 = INT32_MIN`. -/
theorem foldOp_div_int32_min_neg_one :
    foldOp PlanOp.div.name .int32 #[.i32 (-2147483648), .i32 (-1)] =
      some (.i32 (-2147483648)) := by
  rw [foldOp_div_int32]; rfl

/-- … and its remainder is 0: `INT32_MIN % -1 = 0`. -/
theorem foldOp_mod_int32_min_neg_one :
    foldOp PlanOp.mod.name .int32 #[.i32 (-2147483648), .i32 (-1)] =
      some (.i32 0) := by
  rw [foldOp_mod_int32]; rfl

/-- The i64 rule is the same: `INT64_MIN / -1 = INT64_MIN`, `INT64_MIN % -1 = 0`. -/
theorem foldOp_div_int_min_neg_one :
    foldOp PlanOp.div.name .int #[.i (-9223372036854775808), .i (-1)] =
      some (.i (-9223372036854775808)) := by
  rw [foldOp_div_int]; rfl

theorem foldOp_mod_int_min_neg_one :
    foldOp PlanOp.mod.name .int #[.i (-9223372036854775808), .i (-1)] =
      some (.i 0) := by
  rw [foldOp_mod_int]; rfl

theorem foldOp_less_int32 (a b : Int) :
    foldOp PlanOp.less.name .bool #[.i32 a, .i32 b] = some (.b (decide (a < b))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_equal_int32 (a b : Int) :
    foldOp PlanOp.equal.name .bool #[.i32 a, .i32 b] = some (.b (a == b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_bitAnd_int32 (a b : Int) :
    foldOp PlanOp.bitAnd.name .int32 #[.i32 a, .i32 b] =
      some (.i32 (bit32 (· &&& ·) a b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_bitOr_int32 (a b : Int) :
    foldOp PlanOp.bitOr.name .int32 #[.i32 a, .i32 b] =
      some (.i32 (bit32 (· ||| ·) a b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_bitXor_int32 (a b : Int) :
    foldOp PlanOp.bitXor.name .int32 #[.i32 a, .i32 b] =
      some (.i32 (bit32 (· ^^^ ·) a b)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_bitNot_int32 (a : Int) :
    foldOp PlanOp.bitNot.name .int32 #[.i32 a] =
      some (.i32 (bit32 (· ^^^ ·) a (-1))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- An int32 left shift takes its amount modulo 32. -/
theorem foldOp_lshift_int32 (a sh : Int) :
    foldOp PlanOp.lshift.name .int32 #[.i32 a, .i32 sh] =
      some (.i32 (wrap32 (a <<< (sh % 32).toNat))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- An int32 arithmetic right shift takes its amount modulo 32. -/
theorem foldOp_rshift_int32 (a sh : Int) :
    foldOp PlanOp.rshift.name .int32 #[.i32 a, .i32 sh] =
      some (.i32 (a >>> (sh % 32).toNat)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_lshift_int32_inRange {a sh : Int} (h0 : 0 ≤ sh) (h31 : sh ≤ 31) :
    foldOp PlanOp.lshift.name .int32 #[.i32 a, .i32 sh] =
      some (.i32 (wrap32 (a <<< sh.toNat))) := by
  rw [foldOp_lshift_int32, Int.emod_eq_of_lt h0 (by omega)]

theorem foldOp_rshift_int32_inRange {a sh : Int} (h0 : 0 ≤ sh) (h31 : sh ≤ 31) :
    foldOp PlanOp.rshift.name .int32 #[.i32 a, .i32 sh] =
      some (.i32 (a >>> sh.toNat)) := by
  rw [foldOp_rshift_int32, Int.emod_eq_of_lt h0 (by omega)]

theorem shiftAmount32_lt (sh : Int) : (sh % 32).toNat < 32 := by
  have h0 := Int.emod_nonneg sh (by decide : (32 : Int) ≠ 0)
  have h1 := Int.emod_lt_of_pos sh (by decide : (0 : Int) < 32)
  omega

/-- `toInt` of an int32 sign-extends: the value is unchanged. -/
theorem foldOp_toInt_int32 (a : Int) :
    foldOp PlanOp.toInt.name .int #[.i32 a] = some (.i a) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- `toInt32` of an int keeps its low 32 bits. -/
theorem foldOp_toInt32_int (a : Int) :
    foldOp PlanOp.toInt32.name .int32 #[.i a] = some (.i32 (wrap32 a)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

-- ─────────────────────────────────────────────────────────────
-- Saturating float → int conversion
-- ─────────────────────────────────────────────────────────────

theorem clampInt_ge (lo hi x : Int) : lo ≤ clampInt lo hi x := by
  unfold clampInt; omega

theorem clampInt_le (lo hi x : Int) (h : lo ≤ hi) : clampInt lo hi x ≤ hi := by
  unfold clampInt; omega

/-- A saturated conversion to `int` always lands in the i64 range — for
    every float, NaN and the infinities included. -/
theorem truncSatToInt_isI64 (v : Float) : IsI64 (truncSatToInt v) := by
  unfold IsI64 i64Half truncSatToInt
  exact ⟨clampInt_ge _ _ _, by
    have := clampInt_le (-9223372036854775808) 9223372036854775807
      (if v.isNaN then 0
        else if v >= 9.223372036854775808e18 then 9223372036854775807
        else if v <= -9.223372036854775808e18 then -9223372036854775808
        else if v >= 0.0 then Int.ofNat v.toUInt64.toNat
        else -(Int.ofNat (-v).toUInt64.toNat)) (by decide)
    omega⟩

/-- … and a saturated conversion to `int32` in the i32 range. -/
theorem truncSatToInt32_isI32 (v : Float) : IsI32 (truncSatToInt32 v) := by
  unfold IsI32 i32Half truncSatToInt32
  exact ⟨clampInt_ge _ _ _, by
    have := clampInt_le (-2147483648) 2147483647 (truncSatToInt v) (by decide)
    omega⟩

/-- `toInt` of a float folds, for every float, to the saturated truncation. -/
theorem foldOp_toInt_float (x : Float) :
    foldOp PlanOp.toInt.name .int #[.f x] = some (.i (truncSatToInt x)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

/-- `toInt32` of a float folds, for every float, to the saturated truncation. -/
theorem foldOp_toInt32_float (x : Float) :
    foldOp PlanOp.toInt32.name .int32 #[.f x] = some (.i32 (truncSatToInt32 x)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

theorem foldOp_clamp_int32 (v lo hi : Int) :
    foldOp PlanOp.clamp.name .int32 #[.i32 v, .i32 lo, .i32 hi] =
      some (.i32 (let lc := if v > lo then v else lo
        if lc < hi then lc else hi)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty]

-- The wrap edge cases, evaluated.

/-- `INT32_MIN · -1` wraps back to `INT32_MIN`. -/
example : foldOp "Mul" .int32 #[.i32 (-2147483648), .i32 (-1)] =
    some (.i32 (-2147483648)) := by
  rw [show "Mul" = PlanOp.mul.name from rfl, foldOp_mul_int32]; rfl

/-- `INT32_MAX + 1` wraps to `INT32_MIN`. -/
example : foldOp "Add" .int32 #[.i32 2147483647, .i32 1] = some (.i32 (-2147483648)) := by
  rw [show "Add" = PlanOp.add.name from rfl, foldOp_add_int32]; rfl

/-- `2^16 · 2^16` overflows to 0: only the low word survives. -/
example : foldOp "Mul" .int32 #[.i32 65536, .i32 65536] = some (.i32 0) := by
  rw [show "Mul" = PlanOp.mul.name from rfl, foldOp_mul_int32]; rfl

/-- `46341²` overflows past `2^31` into the negative range. -/
example : foldOp "Mul" .int32 #[.i32 46341, .i32 46341] = some (.i32 (-2147479015)) := by
  rw [show "Mul" = PlanOp.mul.name from rfl, foldOp_mul_int32]; rfl

/-- `-INT32_MIN` is `INT32_MIN`. -/
example : foldOp "Neg" .int32 #[.i32 (-2147483648)] = some (.i32 (-2147483648)) := by
  rw [show "Neg" = PlanOp.neg.name from rfl, foldOp_neg_int32]; rfl

/-- `1 << 31` is `INT32_MIN`; `INT32_MIN >> 31` is `-1` (arithmetic). -/
example : foldOp "LShift" .int32 #[.i32 1, .i32 31] = some (.i32 (-2147483648)) := by
  rw [show "LShift" = PlanOp.lshift.name from rfl, foldOp_lshift_int32]; rfl

example : foldOp "RShift" .int32 #[.i32 (-2147483648), .i32 31] = some (.i32 (-1)) := by
  rw [show "RShift" = PlanOp.rshift.name from rfl, foldOp_rshift_int32]; rfl

/-- A shift by the word size is a shift by 0; by 33, a shift by 1; by -1, a
    shift by 31. -/
example : foldOp "LShift" .int32 #[.i32 5, .i32 32] = some (.i32 5) := by
  rw [show "LShift" = PlanOp.lshift.name from rfl, foldOp_lshift_int32]; rfl

example : foldOp "LShift" .int32 #[.i32 5, .i32 33] = some (.i32 10) := by
  rw [show "LShift" = PlanOp.lshift.name from rfl, foldOp_lshift_int32]; rfl

example : foldOp "LShift" .int32 #[.i32 1, .i32 (-1)] = some (.i32 (-2147483648)) := by
  rw [show "LShift" = PlanOp.lshift.name from rfl, foldOp_lshift_int32]; rfl

example : foldOp "RShift" .int #[.i (-8), .i 65] = some (.i (-4)) := by
  rw [show "RShift" = PlanOp.rshift.name from rfl, foldOp_rshift_int]; rfl

/-- A zero divisor gives 0 in both widths. -/
example : foldOp "Div" .int32 #[.i32 7, .i32 0] = some (.i32 0) := by
  rw [show "Div" = PlanOp.div.name from rfl, foldOp_div_int32]; rfl

example : foldOp "Mod" .int #[.i 7, .i 0] = some (.i 0) := by
  rw [show "Mod" = PlanOp.mod.name from rfl, foldOp_mod_int]; rfl

/-- `toInt32 (2^32 + 5) = 5` and `toInt32 2^31 = INT32_MIN`. -/
example : foldOp "ToInt32" .int32 #[.i 4294967301] = some (.i32 5) := by
  rw [show "ToInt32" = PlanOp.toInt32.name from rfl, foldOp_toInt32_int]; rfl

example : foldOp "ToInt32" .int32 #[.i 2147483648] = some (.i32 (-2147483648)) := by
  rw [show "ToInt32" = PlanOp.toInt32.name from rfl, foldOp_toInt32_int]; rfl

end Tropical.Ir
