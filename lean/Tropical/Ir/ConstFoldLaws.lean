import Tropical.Ir.ConstFold
import Init.Data.Int.Bitwise.Lemmas
import Init.Data.Int.DivMod.Lemmas

/-!
# ConstFoldLaws — the integer constant folder as an i64 algebra

`ConstFold.lean` says its integer path reproduces i64 two's-complement
execution. This module makes the integer half of that sentence checked:

* `wrap64` is the canonical representative in `[-2^63, 2^63)`, fixes every
  value already in range, is idempotent, and preserves the residue modulo
  `2^64`;
* wrapping commutes with add, sub, mul, and neg, so the arbitrary-precision
  `Int` implementation realizes the quotient ring `ℤ/2^64ℤ`;
* `toU64` is exactly the nonnegative residue and `bit64` always returns an
  i64-range value;
* every integer-only `foldOp` route is pinned to its intended equation:
  arithmetic (including zero guards), comparisons, truthiness, bitwise ops,
  shifts (including the `[0,63]` refusal), casts, clamp, and select.

No Float claim is made here. In particular, `toFloat`, Float comparisons,
transcendentals, and `truncToInt?` remain outside this proof boundary.
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

theorem foldOp_lshift_int {a sh : Int} (h0 : 0 ≤ sh) (h63 : sh ≤ 63) :
    foldOp PlanOp.lshift.name .int #[.i a, .i sh] =
      some (.i (wrap64 (a <<< sh.toNat))) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty, h0, h63]

theorem foldOp_lshift_int_refuses {a sh : Int} (h : sh < 0 ∨ 63 < sh) :
    foldOp PlanOp.lshift.name .int #[.i a, .i sh] = none := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty, h]

theorem foldOp_rshift_int {a sh : Int} (h0 : 0 ≤ sh) (h63 : sh ≤ 63) :
    foldOp PlanOp.rshift.name .int #[.i a, .i sh] =
      some (.i (a >>> sh.toNat)) := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty, h0, h63]

theorem foldOp_rshift_int_refuses {a sh : Int} (h : sh < 0 ∨ 63 < sh) :
    foldOp PlanOp.rshift.name .int #[.i a, .i sh] = none := by
  simp [foldOp, PlanOp.name, PlanOp.ofString?, CVal.coerce?, CVal.ty, h]

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

end Tropical.Ir
