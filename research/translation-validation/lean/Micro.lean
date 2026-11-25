import Std.Tactic.BVDecide
set_option maxHeartbeats 0
-- where does bit-blasting struggle?  each with a 60 s SAT budget
theorem m1 (a b : BitVec 32) : a * b = b * a := by bv_decide (config := { timeout := 60 })
theorem m2 (a b : BitVec 32) : a * b = b * a := by bv_decide (config := { acNf := true, timeout := 60 })
theorem m3 (x : BitVec 32) : (x * x + ((1#32) <<< (13#32))).sshiftRight' (14#32) = (x * x + 8192).sshiftRight 14 := by
  bv_decide (config := { timeout := 60 })
-- same multiplier inputs reached by different (equal) expressions: the cut-3a situation
theorem m4 (h : BitVec 32) :
    let n := (h + 16384).sshiftRight 15
    let x1 := (1 - 2 * (n &&& 1)) * (h - (n <<< 15))
    let x2 := (1 - (((h + 16384) >>> 14) &&& 2)) * (h - ((h + 16384) &&& 0x18000))
    (h &&& 0xffff = h) → x1 * x1 = x2 * x2 := by
  intro n x1 x2; bv_decide (config := { timeout := 60 })
