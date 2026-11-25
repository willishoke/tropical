"""Spec-level properties of SinOscQ15 that involve wide arithmetic (the
claims made in Numerics.lean doc comments), as solver obligations.
usage: venv/bin/python props.py [name ...]"""
import sys
import z3, spec
from spec import i32
from tv import prove

def SX(x, w): return z3.SignExt(w - x.size(), x)

def p_phase_exact():
    """phaseQ32Of32(inc, off, tau) == low32(floor(inc*tau / 2^16)) + off, all inc, off, tau."""
    inc, off = z3.BitVecs('inc off', 32); tau = z3.BitVec('tau', 64)
    wide = (SX(inc, 96) * SX(tau, 96)) >> 16
    return 'phase split exact (96-bit product)', spec.phaseQ32Of32(inc, off, tau) == z3.Extract(31, 0, wide) + off

def p_phase_exact_frac():
    """the only non-modular piece: floor(lo*frac/2^16) == ((lo*frac)>>16)&0xffff (32-bit)."""
    inc = z3.BitVec('inc', 32); tau = z3.BitVec('tau', 64)
    lo = z3.ZeroExt(32, inc & i32(0xffff)); fr = z3.ZeroExt(32, z3.Extract(31, 0, tau) & i32(0xffff))
    exact = z3.Extract(31, 0, z3.LShR(lo * fr, 16))
    return 'phase: low partial product needs no wrap', exact == ((((inc & i32(0xffff)) * (z3.Extract(31, 0, tau) & i32(0xffff))) >> 16) & i32(0xffff))

def p_inc_error():
    """(floor(F*M/2^32) - inc) mod 2^32 in [0, 3]: at most 3 LSBs low (doc of hzQ16ToIncQ32)."""
    F = z3.BitVec('F', 32); M = spec.M_const(48000)
    exact = z3.Extract(31, 0, (SX(F, 128) * z3.BitVecVal(M, 128)) >> 32)
    d = exact - spec.hzQ16ToIncQ32(F, 48000)
    return 'inc within 3 LSB below floor(F*M/2^32)', z3.ULE(d, i32(3))

def p_sine_nowrap():
    """below the phase rounding no int32 op in fixedSinQ15 wraps: each op recomputed in
    64 bits on sign-extended operands equals the sign-extension of the 32-bit result."""
    ph = z3.BitVec('ph', 32)
    oks = []
    def chk(r32, r64):
        oks.append(SX(r32, 64) == r64)
    h = ((ph + i32(32768)) >> 16) & i32(65535)      # the rounding add wraps by design
    H = SX(h, 64)
    def W(n): return z3.BitVecVal(n, 64)
    n = (h + i32(16384)) >> 15; chk(h + i32(16384), H + W(16384))
    r = h - (n << 15); chk(n << 15, SX(n, 64) << 15); chk(r, H - (SX(n, 64) << 15))
    sign = i32(1) - i32(2) * (n & i32(1))
    x = sign * r; chk(x, SX(sign, 64) * SX(r, 64))
    xx = x * x; chk(xx, SX(x, 64) * SX(x, 64)); chk(xx + i32(8192), SX(xx, 64) + W(8192))
    z = (xx + i32(8192)) >> 14
    acc = i32(141)
    for c in (2604, 21169, 51474):
        p = acc * z; chk(p, SX(acc, 64) * SX(z, 64)); chk(p + i32(8192), SX(p, 64) + W(8192))
        t = (p + i32(8192)) >> 14; acc2 = i32(c) - t; chk(acc2, W(c) - SX(t, 64)); acc = acc2
    y0 = x * acc; chk(y0, SX(x, 64) * SX(acc, 64)); chk(y0 + i32(8192), SX(y0, 64) + W(8192))
    return 'sine: no wrap after the phase rounding', z3.And(oks)

def p_clamp_inert():
    """the final clamp never fires (y already in [-32767, 32767])."""
    ph = z3.BitVec('ph', 32)
    x = None
    h = ((ph + i32(32768)) >> 16) & i32(65535)
    n = (h + i32(16384)) >> 15; r = h - (n << 15); x = (i32(1) - i32(2) * (n & i32(1))) * r
    z = ((x * x) + i32(8192)) >> 14
    acc = i32(141)
    for c in (2604, 21169, 51474): acc = i32(c) - (((acc * z) + i32(8192)) >> 14)
    y = ((x * acc) + i32(8192)) >> 14
    return 'sine: clamp is inert', z3.And(y >= -32767, y <= 32767)

ALL = dict(phase_exact=p_phase_exact, phase_exact_frac=p_phase_exact_frac, inc_error=p_inc_error,
           sine_nowrap=p_sine_nowrap, clamp_inert=p_clamp_inert)
if __name__ == '__main__':
    for k in (sys.argv[1:] or ALL):
        name, claim = ALL[k](); prove(name, claim)
