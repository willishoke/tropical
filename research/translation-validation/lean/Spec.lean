/-
SinOscQ15 at rate 48000, written by hand from lean/Tropical/EmitArrow/Numerics.lean
(hzQ16ToIncQ32, phaseQ32Of32, fixedSinQ15Sig) with each op read at its sort through
ConstFold.foldOp: int32 add/sub/mul wrap (BitVec ring ops), int32 rshift is arithmetic
(Int >>> = floor = sshiftRight), bitAnd on the two's-complement image, int -> int32 keeps
the low word, clamp v lo hi = (lc := v > lo ? v : lo; lc < hi ? lc : hi).
M = round(2^48 / 48000) = 5864062015 is the f64 constant of the artifact (not derived here).
-/
def asr32 (x : BitVec 32) (n : Nat) : BitVec 32 := x.sshiftRight n
def asr64 (x : BitVec 64) (n : Nat) : BitVec 64 := x.sshiftRight n
def lo32 (x : BitVec 64) : BitVec 32 := x.truncate 32

def hzQ16ToIncQ32 (F : BitVec 32) : BitVec 32 :=
  let m : BitVec 64 := 5864062015
  let q := lo32 (asr64 m 32)
  let mLow := m &&& 4294967295
  let mh := lo32 (asr64 mLow 16)
  let ml := lo32 (mLow &&& 65535)
  let fh := asr32 F 16
  let fl := F &&& 65535
  ((q * F + fh * mh) + asr32 (fh * ml) 16) + (asr32 (fl * mh) 16 &&& 65535)

def phaseQ32Of32 (inc offset : BitVec 32) (clk : BitVec 64) : BitVec 32 :=
  let whole := lo32 (asr64 clk 16)
  let frac := lo32 (clk &&& 65535)
  let incHi := asr32 inc 16
  let incLo := inc &&& 65535
  ((inc * whole + incHi * frac) + (asr32 (incLo * frac) 16 &&& 65535)) + offset

def clamp32 (v lo hi : BitVec 32) : BitVec 32 :=
  let lc := if BitVec.slt lo v then v else lo
  if BitVec.slt lc hi then lc else hi

def sinX (phaseQ : BitVec 32) : BitVec 32 :=
  let h := asr32 (phaseQ + 32768) 16 &&& 65535
  let n := asr32 (h + 16384) 15
  let r := h - (n <<< 15)
  let sign := 1 - 2 * (n &&& 1)
  sign * r

def sinTail (x : BitVec 32) : BitVec 32 :=
  let z := asr32 (x * x + 8192) 14
  let step (c acc : BitVec 32) : BitVec 32 := c - asr32 (acc * z + 8192) 14
  let acc1 := step 51474 (step 21169 (step 2604 141))
  clamp32 (asr32 (x * acc1 + 8192) 14) (-32767) 32767

def fixedSinQ15 (phaseQ : BitVec 32) : BitVec 32 := sinTail (sinX phaseQ)

def sinOscQ15 (F P : BitVec 32) (tau : BitVec 64) : BitVec 32 :=
  fixedSinQ15 (phaseQ32Of32 (hzQ16ToIncQ32 F) P tau)
