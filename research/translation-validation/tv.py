"""Translation validation of one tropical kernel (SinOscQ15, frames=1, rate
48000) across three artifacts:

   spec (spec.py, from the Lean builder + ConstFold reference semantics)
   == emitted LLVM text (llsym.py executes lfo_knob2-m0.o.ll)
   == thumbv6m object (thumbsym.py executes llvm-objdump of lfo_knob2-m0.o)

for ALL freq (int32), phase (int32), tau (int64).
usage: venv/bin/python tv.py [frames]
"""
import os, subprocess, sys, time, resource
import z3
import spec, llsym, thumbsym

HERE = os.path.dirname(os.path.abspath(__file__))
OBJDUMP = '/opt/homebrew/opt/llvm@22/bin/llvm-objdump'
z3.set_param('memory_max_size', 4000)   # MB
z3.set_param('timeout', int(os.environ.get('TV_TIMEOUT_MS', '600000')))         # ms

def prove(name, claim):
    s = z3.SolverFor('QF_BV')
    s.add(z3.Not(claim))
    t = time.time()
    r = s.check()
    dt = time.time() - t
    mem = z3.get_param('memory_max_size')
    print('%-48s %-8s %7.3fs  (z3 max rss so far %.0f MB)' % (
        name, 'PROVED' if r == z3.unsat else 'REFUTED' if r == z3.sat else 'UNKNOWN', dt,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))
    if r == z3.sat:
        print('  counterexample:', s.model())
    return r

def main(frames=1, stem='lfo_knob2-m0'):
    ll = open(os.path.join(HERE, stem + '.o.ll')).read()
    F = z3.BitVec('freq', 32); P = z3.BitVec('phase', 32)
    taus = [z3.BitVec('tau%d' % i, 64) for i in range(frames)]

    # --- LLVM text ---
    t = time.time()
    taumem = {(8 * i, 'i64'): taus[i] for i in range(frames)}
    outs = llsym.run(ll, {'%tau': taumem, '%in_1': {(0, 'i32'): F},
                          '%in_2': {(0, 'i32'): P}, '%out': 'out'})
    print('llvm symbolic execution: %.2fs, %d output words' % (time.time() - t, len(outs)))
    assert sorted(outs) == [4 * i for i in range(frames)], outs

    # --- thumbv6m object ---
    dis = subprocess.run([OBJDUMP, '-d', '--triple=thumbv6m-none-eabi',
                          os.path.join(HERE, stem + '.o')], capture_output=True, text=True).stdout
    callee = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}
    regs0 = dict(callee, r0=('ptr', 'tau', 0), r1=('ptr', 'in1', 0),
                 r2=('ptr', 'in2', 0), r3=('ptr', 'out', 0))
    loads = {('in1', 0): F, ('in2', 0): P}
    for i, tv in enumerate(taus):   # little-endian i64 = two words
        loads[('tau', 8 * i)] = z3.Extract(31, 0, tv)
        loads[('tau', 8 * i + 4)] = z3.Extract(63, 32, tv)
    t = time.time()
    res = thumbsym.run(dis, 'tropical_kernel', regs0, loads)
    print('thumb symbolic execution: %.2fs, %d paths' % (time.time() - t, len(res)))

    for i in range(frames):
        sp = spec.sinOscQ15(F, P, taus[i])
        lv = outs[4 * i]
        th = thumbsym.merge(res, ('out', 4 * i))
        prove('frame %d: spec == LLVM text' % i, sp == lv)
        prove('frame %d: LLVM text == thumbv6m object' % i, lv == th)
        prove('frame %d: spec == thumbv6m object' % i, sp == th)
    # ABI hygiene: callee-saved registers restored on every path
    prove('thumb: r4-r7 restored on every path',
          z3.And([z3.And([R[r] == callee[r] for r in ('r4', 'r5', 'r6', 'r7')]) for _, _, R in res]))
    # sanity: the check can fail -- a perturbed spec is refuted
    bad = spec.fixedSinQ15(spec.phaseQ32Of32(spec.hzQ16ToIncQ32(F, 48000), P, taus[0]) + 1)
    prove('negative control: spec(phase+1) == LLVM (expect REFUTED)', bad == outs[0])

if __name__ == '__main__':
    fr = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(fr, sys.argv[2] if len(sys.argv) > 2 else 'lfo_knob2-m0')
