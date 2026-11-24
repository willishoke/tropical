import Tropical.Ir.ConstFold

/-!
# Per-op laws for the symbolic carrier (int32 lane)

`Semantics.Symbolic` prints an int32 operation with a symbolic operand as an
SMT-LIB bit-vector term. These lemmas state, for each primitive the stdlib
integer kernels use, that the bit-vector operation on the two's-complement
images of in-range values has, as its signed value (`toInt`), exactly what
`ConstFold.foldOp` computes on `Int` at sort `int32`. Since `toInt` is
injective, the term and the reference agree as words.

What is NOT here (the open part of the carrier's obligation): the same laws
for `bvnot`, and `bvor`/`bvxor`/`bvshl` at 64 bits, masked runtime shift
amounts, the bool<->int coercions, and the theorem that `Symbolic.symOp`'s
dispatch (which sort, which coercion) composes these primitives into exactly
`foldOp` at every sort. Also trusted: SMT-LIB's `bvadd` etc. mean Lean's
`BitVec` ops, and z3 decides them correctly.
-/

namespace Tropical.Semantics.SymbolicLaws

open Tropical.Ir

abbrev InI32 (a : Int) : Prop := -2147483648 ≤ a ∧ a < 2147483648
abbrev InI64 (a : Int) : Prop := -9223372036854775808 ≤ a ∧ a < 9223372036854775808

theorem wrap32_eq_bmod (x : Int) : wrap32 x = x.bmod (2 ^ 32) := by
  rw [Int.bmod_def]; unfold wrap32; simp only [Nat.reducePow]; split <;> omega

theorem wrap64_eq_bmod (x : Int) : wrap64 x = x.bmod (2 ^ 64) := by
  rw [Int.bmod_def]; unfold wrap64; simp only [Nat.reducePow]; split <;> omega

theorem bmod32_of_in {a : Int} (h : InI32 a) : a.bmod (2 ^ 32) = a := by
  rw [Int.bmod_def]; simp only [Nat.reducePow]; split <;> omega

theorem bmod64_of_in {a : Int} (h : InI64 a) : a.bmod (2 ^ 64) = a := by
  rw [Int.bmod_def]; simp only [Nat.reducePow]; split <;> omega

theorem toInt_img32 {a : Int} (h : InI32 a) : (BitVec.ofInt 32 a).toInt = a := by
  rw [BitVec.toInt_ofInt, bmod32_of_in h]

theorem toInt_img64 {a : Int} (h : InI64 a) : (BitVec.ofInt 64 a).toInt = a := by
  rw [BitVec.toInt_ofInt, bmod64_of_in h]

/-- A `toInt` is already a canonical int32 value, so `wrap32` fixes it. -/
theorem wrap32_toInt (x : BitVec 32) : wrap32 x.toInt = x.toInt := by
  rw [wrap32_eq_bmod]; exact bmod32_of_in ⟨by have := BitVec.le_toInt x; simpa using this,
    by have := BitVec.toInt_lt (x := x); simpa using this⟩

variable {a b : Int}

-- bvadd / bvsub / bvmul / bvneg: `wrap32` of the Int ring op
theorem add32 (ha : InI32 a) (hb : InI32 b) :
    (BitVec.ofInt 32 a + BitVec.ofInt 32 b).toInt = wrap32 (a + b) := by
  rw [BitVec.toInt_add, toInt_img32 ha, toInt_img32 hb, wrap32_eq_bmod]

theorem sub32 (ha : InI32 a) (hb : InI32 b) :
    (BitVec.ofInt 32 a - BitVec.ofInt 32 b).toInt = wrap32 (a - b) := by
  rw [BitVec.toInt_sub, toInt_img32 ha, toInt_img32 hb, wrap32_eq_bmod]

theorem mul32 (ha : InI32 a) (hb : InI32 b) :
    (BitVec.ofInt 32 a * BitVec.ofInt 32 b).toInt = wrap32 (a * b) := by
  rw [BitVec.toInt_mul, toInt_img32 ha, toInt_img32 hb, wrap32_eq_bmod]

theorem neg32 (ha : InI32 a) : (-BitVec.ofInt 32 a).toInt = wrap32 (-a) := by
  rw [BitVec.toInt_neg, toInt_img32 ha, wrap32_eq_bmod]

-- bvashr by a constant k (< 32): `foldOp` is `a >>> (sh % 32)`, Int's floor shift
theorem ashr32 (ha : InI32 a) (k : Nat) :
    ((BitVec.ofInt 32 a).sshiftRight k).toInt = a >>> k := by
  rw [BitVec.toInt_sshiftRight, toInt_img32 ha]

-- bvsdiv / bvsrem guarded by `ite (= b 0) 0 _`: `foldOp`'s total quotient
theorem sdiv32 (ha : InI32 a) (hb : InI32 b) :
    ((BitVec.ofInt 32 a).sdiv (BitVec.ofInt 32 b)).toInt = wrap32 (a.tdiv b) := by
  rw [BitVec.toInt_sdiv, toInt_img32 ha, toInt_img32 hb, wrap32_eq_bmod]

theorem srem32 (ha : InI32 a) (hb : InI32 b) :
    ((BitVec.ofInt 32 a).srem (BitVec.ofInt 32 b)).toInt = wrap32 (a.tmod b) := by
  have h := BitVec.toInt_srem (BitVec.ofInt 32 a) (BitVec.ofInt 32 b)
  rw [toInt_img32 ha, toInt_img32 hb] at h
  rw [← h, wrap32_toInt]

-- (_ sign_extend 32): `int32 → int` keeps the value
theorem signExtend32 (ha : InI32 a) :
    (BitVec.signExtend 64 (BitVec.ofInt 32 a)).toInt = a := by
  rw [BitVec.toInt_signExtend, toInt_img32 ha]; exact bmod32_of_in ha

-- bvslt on the sign-extended images: `foldOp`'s comparisons compare the
-- int views (`decide (a < b)`)
theorem slt32 (ha : InI32 a) (hb : InI32 b) :
    (BitVec.signExtend 64 (BitVec.ofInt 32 a)).slt (BitVec.signExtend 64 (BitVec.ofInt 32 b)) =
      decide (a < b) := by
  rw [Bool.eq_iff_iff, BitVec.slt_iff_toInt_lt, signExtend32 ha, signExtend32 hb]; simp

-- bvslt at int32 itself (the clamp's comparisons)
theorem slt32' (ha : InI32 a) (hb : InI32 b) :
    (BitVec.ofInt 32 a).slt (BitVec.ofInt 32 b) = decide (a < b) := by
  rw [Bool.eq_iff_iff, BitVec.slt_iff_toInt_lt, toInt_img32 ha, toInt_img32 hb]; simp

-- (_ extract 31 0) of an int: `int → int32` is `wrap32`
theorem extract32 :
    (BitVec.setWidth 32 (BitVec.ofInt 64 a)).toInt = wrap32 a := by
  rw [BitVec.toInt_setWidth, BitVec.toNat_ofInt, wrap32_eq_bmod]
  simp only [Nat.reducePow]
  rw [Int.bmod_def, Int.bmod_def]
  have h0 : 0 ≤ a % ((18446744073709551616 : Nat) : Int) := Int.emod_nonneg _ (by decide)
  rw [Int.toNat_of_nonneg h0]; split <;> split <;> omega

-- int64 ring ops, same shape
theorem add64 (ha : InI64 a) (hb : InI64 b) :
    (BitVec.ofInt 64 a + BitVec.ofInt 64 b).toInt = wrap64 (a + b) := by
  rw [BitVec.toInt_add, toInt_img64 ha, toInt_img64 hb, wrap64_eq_bmod]

theorem mul64 (ha : InI64 a) (hb : InI64 b) :
    (BitVec.ofInt 64 a * BitVec.ofInt 64 b).toInt = wrap64 (a * b) := by
  rw [BitVec.toInt_mul, toInt_img64 ha, toInt_img64 hb, wrap64_eq_bmod]

theorem ashr64 (ha : InI64 a) (k : Nat) :
    ((BitVec.ofInt 64 a).sshiftRight k).toInt = a >>> k := by
  rw [BitVec.toInt_sshiftRight, toInt_img64 ha]

-- bvand: `foldOp`'s `bit32 (· &&& ·)` through the u32 images
theorem and32 :
    (BitVec.ofInt 32 a &&& BitVec.ofInt 32 b).toInt = bit32 (· &&& ·) a b := by
  rw [BitVec.toInt_eq_toNat_bmod, BitVec.toNat_and, BitVec.toNat_ofInt, BitVec.toNat_ofInt]
  unfold bit32 toU32; rw [wrap32_eq_bmod]
  have e : ∀ x : Int, ((x % 4294967296 + 4294967296) % 4294967296) = x % ((2 ^ 32 : Nat) : Int) := by
    intro x; simp only [Nat.reducePow]; omega
  rw [e a, e b]; rfl

-- bvshl by a constant k: `wrap32 (a <<< k)`
theorem shl32 (k : Nat) :
    ((BitVec.ofInt 32 a) <<< k).toInt = wrap32 (a <<< k) := by
  rw [BitVec.toInt_shiftLeft, BitVec.toNat_ofInt, Nat.shiftLeft_eq, Int.shiftLeft_eq, wrap32_eq_bmod]
  have h0 : 0 ≤ a % ((2 ^ 32 : Nat) : Int) := Int.emod_nonneg _ (by decide)
  rw [Int.natCast_mul, Int.toNat_of_nonneg h0]
  have e : ((2 ^ k : Nat) : Int) = (2 : Int) ^ k := Int.natCast_pow 2 k
  rw [e, ← Int.emod_bmod (x := a % _ * _), ← Int.emod_bmod (x := a * _)]
  congr 1
  rw [Int.mul_emod, Int.emod_emod, ← Int.mul_emod]

theorem or32 :
    (BitVec.ofInt 32 a ||| BitVec.ofInt 32 b).toInt = bit32 (· ||| ·) a b := by
  rw [BitVec.toInt_eq_toNat_bmod, BitVec.toNat_or, BitVec.toNat_ofInt, BitVec.toNat_ofInt]
  unfold bit32 toU32; rw [wrap32_eq_bmod]
  have e : ∀ x : Int, ((x % 4294967296 + 4294967296) % 4294967296) = x % ((2 ^ 32 : Nat) : Int) := by
    intro x; simp only [Nat.reducePow]; omega
  rw [e a, e b]; rfl

theorem xor32 :
    (BitVec.ofInt 32 a ^^^ BitVec.ofInt 32 b).toInt = bit32 (· ^^^ ·) a b := by
  rw [BitVec.toInt_eq_toNat_bmod, BitVec.toNat_xor, BitVec.toNat_ofInt, BitVec.toNat_ofInt]
  unfold bit32 toU32; rw [wrap32_eq_bmod]
  have e : ∀ x : Int, ((x % 4294967296 + 4294967296) % 4294967296) = x % ((2 ^ 32 : Nat) : Int) := by
    intro x; simp only [Nat.reducePow]; omega
  rw [e a, e b]; rfl

theorem and64 :
    (BitVec.ofInt 64 a &&& BitVec.ofInt 64 b).toInt = bit64 (· &&& ·) a b := by
  rw [BitVec.toInt_eq_toNat_bmod, BitVec.toNat_and, BitVec.toNat_ofInt, BitVec.toNat_ofInt]
  unfold bit64 toU64; rw [wrap64_eq_bmod]
  have e : ∀ x : Int, ((x % 18446744073709551616 + 18446744073709551616) % 18446744073709551616)
      = x % ((2 ^ 64 : Nat) : Int) := by
    intro x; simp only [Nat.reducePow]; omega
  rw [e a, e b]; rfl

theorem sub64 (ha : InI64 a) (hb : InI64 b) :
    (BitVec.ofInt 64 a - BitVec.ofInt 64 b).toInt = wrap64 (a - b) := by
  rw [BitVec.toInt_sub, toInt_img64 ha, toInt_img64 hb, wrap64_eq_bmod]

theorem neg64 (ha : InI64 a) : (-BitVec.ofInt 64 a).toInt = wrap64 (-a) := by
  rw [BitVec.toInt_neg, toInt_img64 ha, wrap64_eq_bmod]

end Tropical.Semantics.SymbolicLaws
