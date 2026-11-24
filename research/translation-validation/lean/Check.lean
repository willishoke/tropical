import Std.Tactic.BVDecide
-- run: lean Check.lean (with Spec.lean and Lifted.lean concatenated in, see run_lean.sh)

-- the end-to-end statement about the lifted thumbv6m object, monolithic
theorem thumb_eq_spec (F P : BitVec 32) (tau : BitVec 64) :
    thumbKernel F P tau = sinOscQ15 F P tau := by
  unfold thumbKernel sinOscQ15 fixedSinQ15 sinTail sinX phaseQ32Of32 hzQ16ToIncQ32 clamp32 asr32 asr64 lo32
  bv_decide (config := { timeout := 900 })
