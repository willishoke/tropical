"""SinOscQ15 specification as a 32/64-bit bit-vector function.

Written from lean/Tropical/EmitArrow/Stdlib.lean (buildSinOscQ15) and
lean/Tropical/EmitArrow/Numerics.lean (hzQ16ToIncQ32, phaseQ32Of32,
fixedSinQ15Sig), with each node read through the reference semantics
lean/Tropical/Ir/ConstFold.lean `foldOp` at its sort:

  int32 add/sub/mul  = wrap32 of the Int op       -> bvadd/bvsub/bvmul (32)
  int32 rshift       = Int >>> (sh % 32)          -> bvashr (32)
  int32 lshift       = wrap32 (a <<< (sh % 32))   -> bvshl (32)
  int32 bitAnd       = via the u32 image          -> bvand (32)
  int   rshift/bitAnd                             -> bvashr/bvand (64)
  int -> int32       = wrap32 (low 32 bits)       -> Extract(31,0)
  int32 clamp v lo hi= lc := v>lo ? v : lo ; lc<hi ? lc : hi

The Int(wrap32) <-> BitVec correspondence of those ops is the standard
two's-complement one; it is assumed here, not proven (in Lean it is a
handful of BitVec.toInt_* lemmas).

NOT transcribed from the emitted LLVM: node order and constant spelling
follow the builder (e.g. `mul (lit 2) parity`, `step c acc`).
"""
import math
from z3 import BitVecVal, Extract, If

def i32(n):
    return BitVecVal(n & 0xFFFFFFFF, 32)

def i64(n):
    return BitVecVal(n & 0xFFFFFFFFFFFFFFFF, 64)

def to_int32(x64):           # int -> int32 keeps the low 32 bits
    return Extract(31, 0, x64)

def M_const(rate):
    """m = toInt(round(2^48 / rate)) in f64: Python float is IEEE binary64;
    round half away from zero (= ConstFold .round = llvm.round)."""
    v = 281474976710656.0 / float(rate)
    r = math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)
    # (v + 0.5 is exact enough here: v ~ 5.86e9, far from a tie; asserted below)
    assert abs(v - round(v)) != 0.5
    return int(r)  # truncSatToInt: in range, so plain truncation

def hzQ16ToIncQ32(F, rate):
    m = i64(M_const(rate))
    q = to_int32(m >> 32)                       # int rshift (arith), toInt32
    mLow = m & i64(4294967295)
    mh = to_int32(mLow >> 16)
    ml = to_int32(mLow & i64(65535))
    fh = F >> 16                                # int32 rshift = ashr
    fl = F & i32(65535)
    whole = q * F
    hh = fh * mh
    hl = (fh * ml) >> 16
    lh = ((fl * mh) >> 16) & i32(65535)
    return ((whole + hh) + hl) + lh

def phaseQ32Of32(inc, offset, clk):
    whole = to_int32(clk >> 16)                 # clockFracBits = 16
    frac = to_int32(clk & i64(65535))           # clockFracMask = 0xffff
    incHi = inc >> 16
    incLo = inc & i32(65535)
    pWhole = inc * whole
    pHi = incHi * frac
    pLo = ((incLo * frac) >> 16) & i32(65535)
    return ((pWhole + pHi) + pLo) + offset

def clamp32(v, lo, hi):
    lc = If(v > lo, v, lo)                      # signed compare (int32 order)
    return If(lc < hi, lc, hi)

def fixedSinQ15(phaseQ):
    half14 = i32(8192)
    h = ((phaseQ + i32(32768)) >> 16) & i32(65535)
    n = (h + i32(16384)) >> 15
    r = h - (n << 15)
    parity = n & i32(1)
    sign = i32(1) - i32(2) * parity
    x = sign * r
    z = ((x * x) + half14) >> 14
    def step(c, acc):
        return c - (((acc * z) + half14) >> 14)
    acc7 = i32(141)
    acc5 = step(i32(2604), acc7)
    acc3 = step(i32(21169), acc5)
    acc1 = step(i32(51474), acc3)
    y = ((x * acc1) + half14) >> 14
    return clamp32(y, i32(-32767), i32(32767))

def sinOscQ15(F, P, tau, rate=48000):
    inc = hzQ16ToIncQ32(F, rate)
    phase = phaseQ32Of32(inc, P, tau)
    return fixedSinQ15(phase)
