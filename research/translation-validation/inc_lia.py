"""hzQ16ToIncQ32's error claim, stated over Int exactly as ConstFold.foldOp computes it
(wrap32 after each int32 op, >>> = floor division, & 65535 = mod 2^16).  All products
are variable x constant (M is a constant of the artifact), so this is linear integer
arithmetic with div/mod by constants: decidable, and z3 handles it."""
import time, z3, spec
M = spec.M_const(48000); q = M >> 32; mLow = M & 0xffffffff; mh = mLow >> 16; ml = mLow & 0xffff
def wrap32(x): return ((x + 2**31) % 2**32) - 2**31
def floordiv(x, k): return x / k          # z3 Int '/' is floor division for positive k
F = z3.Int('F')
fh = floordiv(F, 65536); fl = F % 65536
whole = wrap32(q * F); hh = wrap32(fh * mh); hl = floordiv(wrap32(fh * ml), 65536)
lh = floordiv(wrap32(fl * mh), 65536) % 65536
inc = wrap32(wrap32(wrap32(whole + hh) + hl) + lh)
exact = floordiv(F * M, 2**32)
d = (exact - inc) % 2**32
s = z3.Solver(); s.set('timeout', 300000)
s.add(F >= -2**31, F < 2**31, z3.Not(z3.And(d >= 0, d <= 3)))
t = time.time(); r = s.check(); print('inc within 3 LSB below floor(F*M/2^32) [LIA]:', r, '%.2fs' % (time.time() - t))
if r == z3.sat: print(s.model())
# tightness: is 3 attained?
s2 = z3.Solver(); s2.set('timeout', 300000); s2.add(F >= -2**31, F < 2**31, d == 3)
print('d == 3 attainable:', s2.check())
for k in (2, 1, 0):
    s3 = z3.Solver(); s3.set('timeout', 300000); s3.add(F >= -2**31, F < 2**31, d == k)
    print('d == %d attainable:' % k, s3.check())
