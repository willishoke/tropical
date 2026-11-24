"""Finer cuts inside the sine: x = sign*r (r0@0x5c), then the Horner tail from a free x."""
import os, subprocess
import z3, spec, thumbsym
from spec import i32
from tv import prove, OBJDUMP, HERE
dis = subprocess.run([OBJDUMP, '-d', '--triple=thumbv6m-none-eabi',
                      os.path.join(HERE, 'lfo_knob2-m0.o')], capture_output=True, text=True).stdout
callee = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}
def junk(**kw):
    d = {r: z3.BitVec('junk_' + r, 32) for r in ('r1', 'r2', 'r4', 'r5', 'r6', 'r7')}
    d.update(callee); d['r3'] = ('ptr', 'out', 0); d.update(kw); return d
def x_spec(ph):   # fixedSinQ15 up to x, from spec.py's definitions
    h = ((ph + i32(32768)) >> 16) & i32(65535)
    n = (h + i32(16384)) >> 15
    r = h - (n << 15)
    sign = i32(1) - i32(2) * (n & i32(1))
    return sign * r
def tail_spec(x):  # fixedSinQ15 after x
    half14 = i32(8192); z = ((x * x) + half14) >> 14
    step = lambda c, acc: c - (((acc * z) + half14) >> 14)
    acc1 = step(i32(51474), step(i32(21169), step(i32(2604), i32(141))))
    return spec.clamp32(((x * acc1) + half14) >> 14, i32(-32767), i32(32767))
ph = z3.BitVec('ph', 32)
assert z3.simplify(tail_spec(x_spec(ph)) == spec.fixedSinQ15(ph)) is not None
prove('split is the spec: tail(x(ph)) == fixedSinQ15(ph)', tail_spec(x_spec(ph)) == spec.fixedSinQ15(ph))
(r,) = thumbsym.run(dis, 'tropical_kernel', junk(r0=ph), {}, start=0x3e, stop=0x5c, stack0=callee)
prove('cut 3a: x(spec)(ph) == r0@0x5c (r1 == 1 too)', z3.And(x_spec(ph) == r[2]['r0'], r[2]['r1'] == 1))
X = z3.BitVec('x', 32)
res = thumbsym.run(dis, 'tropical_kernel', junk(r0=X, r1=i32(1)), {}, start=0x5c, stack0=callee)
prove('cut 3b: tail(spec)(x) == thumb from 0x5c (free x)', tail_spec(X) == thumbsym.merge(res, ('out', 0)))
