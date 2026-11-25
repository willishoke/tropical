/-!
# Rational clocks — the ring model and its laws

An independent model of `RatioClock` and `ClockMultDivQ15`
(`EmitArrow/Stdlib.lean`; both read `ratioRingPhase` of one Ψ) over
big integers, written without the IR: the gate in `Tropicaltest/ClockMultDiv`
renders the kernel and compares it to these definitions word for word. The
theorems state, over the model:

* (a) `360360 / d` is exact for every `d ∈ [1, 15]`, so `m = n · (360360 / d)`
  is an integer, `d · m = n · 360360`, and `m < 2²³` (it fits the `int32`
  product with room to spare);
* (b) recurrence: one wrap of `Ψ` (one universal cycle, 360360 master
  cycles) returns every output to its phase, and in it output `i` turns
  exactly `mᵢ` whole cycles;
* (c) the lock: the phase is `m·Ψ + o` on ℤ/2⁶⁴, so for any two outputs
  `nⱼ·dᵢ·(Pᵢ − oᵢ) = nᵢ·dⱼ·(Pⱼ − oⱼ)` at every `Ψ`;
* the realization: the 32-bit limb products the kernel spells
  (`mulWideU32`, `mulLoU32By64`, `mulLo64`) and the clock split of
  `ringPosition64` are the ring product and the exact floor.

What is not proved here: that the IR the builders emit computes these
formulas (that is the gate's job, against the JIT), and the `int32` wrap
semantics of each limb product (`ConstFoldLaws` states those).
-/

namespace Tropical.Testing.ClockMultDivLaws

/-- `lcm(1, …, 15)`. -/
def universalCycle : Nat := 360360

def ringSize : Nat := 2 ^ 64

/-- A knob word's low four bits, as the kernel reads it (`x & 15`). -/
def knob (x : Int) : Nat := (Int.emod x 16).toNat

/-- `m = n · (360360 / d)`, and 0 at `d = 0`. -/
def multiplier (n d : Nat) : Nat := if d = 0 then 0 else n * (universalCycle / d)

/-- `K = round(2⁶⁴ / (360360 · rate))` (ties cannot occur: the quotient has
    the odd factor 45045 in its denominator). -/
def incScale (rate : Nat) : Int :=
  (((2 * ringSize + universalCycle * rate) / (2 * universalCycle * rate) : Nat) : Int)

/-- The increment per sample on ℤ/2⁶⁴ for an `int32` Q16.16 Hz word:
    `⌊F · K / 2¹⁶⌋`. -/
def increment (rate : Nat) (freqQ16 : Int) : Int :=
  Int.fdiv (freqQ16 * incScale rate) (2 ^ 16)

/-- `Ψ = ⌊I · τ / 2¹⁶⌋ mod 2⁶⁴` for a Q48.16 clock `τ`. -/
def position (rate : Nat) (freqQ16 tau : Int) : Nat :=
  (Int.emod (Int.fdiv (increment rate freqQ16 * tau) (2 ^ 16)) ringSize).toNat

/-- Output phase in Q0.64 cycles: `m·Ψ + o·2³²` mod 2⁶⁴, `o` a Q0.32 word. -/
def phase (rate : Nat) (freqQ16 tau : Int) (n d : Nat) (offsetQ32 : Int) : Nat :=
  (Int.emod (multiplier n d * position rate freqQ16 tau + offsetQ32 * 2 ^ 32)
    ringSize).toNat

/-- The phase's top word as the kernel's signed `int32` (the `phase` port). -/
def topWord (p : Nat) : Int :=
  let w : Nat := p / 2 ^ 32
  if w ≥ 2 ^ 31 then (w : Int) - 2 ^ 32 else (w : Int)

/-- The gate port's word: 32768 (1.0 in Q17.15) for the first half cycle,
    0 for the second, 0 when `d = 0`. -/
def gateWord (d : Nat) (p : Nat) : Int :=
  if d = 0 then 0 else if p < 2 ^ 63 then 32768 else 0

/-! ## (a) the multiplier is an integer -/

theorem universalCycle_dvd : ∀ d : Fin 16, d.val ≠ 0 → d.val ∣ universalCycle := by
  decide

theorem multiplier_exact (n d : Nat) (h1 : 1 ≤ d) (h15 : d ≤ 15) :
    d * multiplier n d = n * universalCycle := by
  have hd : d ∣ universalCycle :=
    universalCycle_dvd ⟨d, by omega⟩ (by simp; omega)
  have hne : d ≠ 0 := by omega
  simp only [multiplier, hne, if_false]
  rw [Nat.mul_left_comm, Nat.mul_div_cancel' hd]

theorem multiplier_lt (n d : Nat) (hn : n ≤ 15) : multiplier n d < 2 ^ 23 := by
  unfold multiplier
  split
  · decide
  · have : universalCycle / d ≤ universalCycle := Nat.div_le_self _ _
    have : n * (universalCycle / d) ≤ 15 * universalCycle :=
      Nat.mul_le_mul hn this
    simp only [universalCycle] at *
    omega

/-! ## (b) recurrence -/

/-- One wrap of `Ψ` returns every output to its phase. -/
theorem phase_recurs (m o psi : Nat) :
    (m * (psi + ringSize) + o) % ringSize = (m * psi + o) % ringSize := by
  rw [Nat.mul_add, Nat.add_right_comm, Nat.add_mul_mod_self_right]

/-- In one universal cycle (360360 master cycles) output `n/d` turns exactly
    `multiplier n d` whole cycles: `(n/d) · 360360 = m`. -/
theorem cycles_per_universal (n d : Nat) (h1 : 1 ≤ d) (h15 : d ≤ 15) :
    n * universalCycle = d * multiplier n d :=
  (multiplier_exact n d h1 h15).symm

/-- `Ψ ≡ 0` puts every output at its offset. -/
theorem phase_at_zero (m o : Nat) : (m * 0 + o) % ringSize = o % ringSize := by
  simp

/-! ## (c) the lock -/

/-- The lock invariant on ℤ/2⁶⁴: with `Pᵢ = mᵢ·Ψ + oᵢ` and `dᵢ·mᵢ = nᵢ·Λ`,
    the relation between any two outputs holds at every `Ψ`. -/
theorem lock (Λ psi mi mj ni nj di dj oi oj : BitVec 64)
    (hi : di * mi = ni * Λ) (hj : dj * mj = nj * Λ) :
    nj * di * ((mi * psi + oi) - oi) = ni * dj * ((mj * psi + oj) - oj) := by
  simp only [BitVec.add_sub_cancel]
  calc nj * di * (mi * psi) = nj * (di * mi) * psi := by grind
    _ = nj * (ni * Λ) * psi := by rw [hi]
    _ = ni * (nj * Λ) * psi := by grind
    _ = ni * (dj * mj) * psi := by rw [hj]
    _ = ni * dj * (mj * psi) := by grind

/-! ## The realization in 32-bit multiplies -/

/-- `mulWideU32`: four 16×16 limb products are the full 32×32 product. -/
theorem mulWideU32_eq (a b : Nat) (ha : a < 2 ^ 32) (hb : b < 2 ^ 32) :
    a % 2 ^ 16 * (b % 2 ^ 16)
      + (a % 2 ^ 16 * (b / 2 ^ 16 % 2 ^ 16) + a / 2 ^ 16 % 2 ^ 16 * (b % 2 ^ 16)) * 2 ^ 16
      + a / 2 ^ 16 % 2 ^ 16 * (b / 2 ^ 16 % 2 ^ 16) * 2 ^ 32 = a * b := by
  have ha1 : a / 2 ^ 16 % 2 ^ 16 = a / 2 ^ 16 := Nat.mod_eq_of_lt (by omega)
  have hb1 : b / 2 ^ 16 % 2 ^ 16 = b / 2 ^ 16 := Nat.mod_eq_of_lt (by omega)
  rw [ha1, hb1]
  have da := Nat.div_add_mod a (2 ^ 16)
  have db := Nat.div_add_mod b (2 ^ 16)
  generalize a / 2 ^ 16 = a1 at *
  generalize a % 2 ^ 16 = a0 at *
  generalize b / 2 ^ 16 = b1 at *
  generalize b % 2 ^ 16 = b0 at *
  subst da db
  grind

/-- `mulLoU32By64`: `a · b_lo` in full plus `(a · b_hi mod 2³²) · 2³²` is
    `a · b` mod 2⁶⁴. -/
theorem mulLoU32By64_eq (a b : Nat) :
    (a * (b % 2 ^ 32) + (a * (b / 2 ^ 32)) % 2 ^ 32 * 2 ^ 32) % 2 ^ 64
      = a * b % 2 ^ 64 := by
  have db := Nat.div_add_mod b (2 ^ 32)
  have dc := Nat.div_add_mod (a * (b / 2 ^ 32)) (2 ^ 32)
  generalize hq : a * (b / 2 ^ 32) = q at *
  have : a * b = a * (b % 2 ^ 32) + q * 2 ^ 32 := by
    rw [← hq]; conv => lhs; rw [← db]
    grind
  rw [this]
  conv => rhs; rw [← dc]
  have : (2 ^ 32 * (q / 2 ^ 32) + q % 2 ^ 32) * 2 ^ 32
      = q % 2 ^ 32 * 2 ^ 32 + 2 ^ 64 * (q / 2 ^ 32) := by grind
  rw [this, ← Nat.add_assoc, Nat.add_mul_mod_self_left]

/-- `mulLo64`: adding `(a_hi · b_lo mod 2³²) · 2³²` to `a_lo · b` gives
    `a · b` mod 2⁶⁴. -/
theorem mulLo64_eq (a b : Nat) :
    ((a % 2 ^ 32) * b % 2 ^ 64 + (a / 2 ^ 32 * (b % 2 ^ 32)) % 2 ^ 32 * 2 ^ 32) % 2 ^ 64
      = a * b % 2 ^ 64 := by
  have da := Nat.div_add_mod a (2 ^ 32)
  have db := Nat.div_add_mod b (2 ^ 32)
  -- a·b = a_lo·b + a_hi·b·2³², and a_hi·b·2³² ≡ (a_hi·b_lo mod 2³²)·2³² (mod 2⁶⁴)
  have key : a * b = a % 2 ^ 32 * b + (a / 2 ^ 32 * (b % 2 ^ 32)) * 2 ^ 32
      + 2 ^ 64 * (a / 2 ^ 32 * (b / 2 ^ 32)) := by
    conv => lhs; rw [← da]; arg 2; rw [← db]
    grind
  rw [key, Nat.add_mul_mod_self_left]
  have dc := Nat.div_add_mod (a / 2 ^ 32 * (b % 2 ^ 32)) (2 ^ 32)
  generalize a / 2 ^ 32 * (b % 2 ^ 32) = q at *
  generalize a % 2 ^ 32 * b = p
  conv => rhs; rw [← dc]
  have : (2 ^ 32 * (q / 2 ^ 32) + q % 2 ^ 32) * 2 ^ 32
      = q % 2 ^ 32 * 2 ^ 32 + 2 ^ 64 * (q / 2 ^ 32) := by grind
  rw [this, ← Nat.add_assoc, Nat.add_mul_mod_self_left, Nat.add_mod, Nat.mod_mod,
    ← Nat.add_mod]

/-- `ringPosition64`'s clock split: with `τ = whole·2¹⁶ + frac` and
    `inc = hi·2¹⁶ + lo` (floor division, `0 ≤ frac, lo < 2¹⁶`),
    `⌊inc·τ / 2¹⁶⌋ = inc·whole + hi·frac + ⌊lo·frac / 2¹⁶⌋`. -/
theorem position_split (inc tau : Int) :
    (inc * tau) / 2 ^ 16
      = inc * (tau / 2 ^ 16) + inc / 2 ^ 16 * (tau % 2 ^ 16)
        + (inc % 2 ^ 16 * (tau % 2 ^ 16)) / 2 ^ 16 := by
  have dt := Int.mul_ediv_add_emod tau (2 ^ 16)
  have di := Int.mul_ediv_add_emod inc (2 ^ 16)
  have : inc * tau = inc % 2 ^ 16 * (tau % 2 ^ 16)
      + (inc * (tau / 2 ^ 16) + inc / 2 ^ 16 * (tau % 2 ^ 16)) * 2 ^ 16 := by
    conv => lhs; rw [← dt]; arg 1; rw [← di]
    grind
  rw [this, Int.add_mul_ediv_right _ _ (by decide)]
  omega

end Tropical.Testing.ClockMultDivLaws
