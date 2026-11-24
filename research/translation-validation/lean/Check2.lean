-- Cut-point proof that the lifted thumbv6m object computes sinOscQ15, composed in Lean.
-- (Spec.lean, Lifted.lean and LiftedCuts.lean are concatenated in by run_lean.sh.)

-- (a) up to the cut at 0x5c, r0 holds sinX of the spec's phase, r1 holds 1
theorem cut_x (F P : BitVec 32) (tau : BitVec 64) :
    thumbX F P tau = sinX (phaseQ32Of32 (hzQ16ToIncQ32 F) P tau) := by
  unfold thumbX sinX phaseQ32Of32 hzQ16ToIncQ32 asr32 asr64 lo32
  bv_decide (config := { acNf := true, timeout := 600 })

theorem cut_r1 (F P : BitVec 32) (tau : BitVec 64) : thumbR1 F P tau = 1 := by
  unfold thumbR1; bv_decide

-- (b) from the cut, for every x and every value of the other registers
theorem tail (x a b c d e : BitVec 32) : thumbTail x a b c d e = sinTail x := by
  unfold thumbTail sinTail clamp32 asr32
  bv_decide (config := { acNf := true, timeout := 600 })

-- composition: the whole lifted object is the tail run on the cut state (definitional)
theorem compose (F P : BitVec 32) (tau : BitVec 64) :
    thumbKernel F P tau =
      thumbTail (thumbX F P tau) (thumbR2 F P tau) (thumbR4 F P tau) (thumbR5 F P tau)
        (thumbR6 F P tau) (thumbR7 F P tau) := by
  rfl

theorem thumb_eq_spec' (F P : BitVec 32) (tau : BitVec 64) :
    thumbKernel F P tau = sinOscQ15 F P tau := by
  rw [compose, tail, cut_x]; rfl

#print axioms thumb_eq_spec'
