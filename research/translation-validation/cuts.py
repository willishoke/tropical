"""Compositional (cut-point) version of the spec == thumbv6m proof.
Cuts: inc (r1 at 0x1e), phase (r0 at 0x3e); then the sine from a free phase."""
import os, subprocess, time, resource
import z3, spec, thumbsym
from tv import prove, OBJDUMP, HERE
dis = subprocess.run([OBJDUMP, '-d', '--triple=thumbv6m-none-eabi',
                      os.path.join(HERE, 'lfo_knob2-m0.o')], capture_output=True, text=True).stdout
F = z3.BitVec('freq', 32); P = z3.BitVec('phase', 32); T = z3.BitVec('tau', 64)
callee = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}
regs0 = dict(callee, r0=('ptr', 'tau', 0), r1=('ptr', 'in1', 0), r2=('ptr', 'in2', 0), r3=('ptr', 'out', 0))
loads = {('in1', 0): F, ('in2', 0): P, ('tau', 0): z3.Extract(31, 0, T), ('tau', 4): z3.Extract(63, 32, T)}

inc_s = spec.hzQ16ToIncQ32(F, 48000)
(r,) = thumbsym.run(dis, 'tropical_kernel', regs0, loads, stop=0x1e)
prove('cut 1: inc(spec) == r1@0x1e (freq only)', inc_s == r[2]['r1'])
(r,) = thumbsym.run(dis, 'tropical_kernel', regs0, loads, stop=0x3e)
prove('cut 2: phase(spec) == r0@0x3e (all inputs)', spec.phaseQ32Of32(inc_s, P, T) == r[2]['r0'])
INC = z3.BitVec('inc', 32)
r2 = dict(r[2]); # replace r1 (inc) by a free variable? r1 is consumed; check phase from free inc:
ph = z3.BitVec('ph', 32)
regs = dict(callee, r0=ph, r1=z3.BitVec('junk1', 32), r2=z3.BitVec('junk2', 32), r3=('ptr', 'out', 0),
            r4=z3.BitVec('junk4', 32), r5=z3.BitVec('junk5', 32), r6=z3.BitVec('junk6', 32), r7=z3.BitVec('junk7', 32))
res = thumbsym.run(dis, 'tropical_kernel', regs, {}, start=0x3e, stack0=callee)
prove('cut 3: sine(spec)(ph) == thumb from 0x3e (free ph)', spec.fixedSinQ15(ph) == thumbsym.merge(res, ('out', 0)))
