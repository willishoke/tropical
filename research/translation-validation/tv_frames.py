"""Generated spec vs a thumbv6m object compiled with --frames N (the frame loop).

usage: tv_frames.py --spec <spec.smt2> --obj <obj.o> [--cuts]
The spec is one frame of the plan semantics (diffcli emit-spec); the claim is
that for ALL free inputs and ALL clocks tau_0..tau_{N-1}, the object writes
out[k] = spec(params, tau_k) for every frame k (frame-major, one channel per
spec output), stores nothing else outside its own stack, restores r4-r7 and
returns to lr."""
import argparse, json, subprocess, time
import z3, thumbsym2
from tv import prove, OBJDUMP

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True); ap.add_argument('--obj', required=True)
ap.add_argument('--triple', default='thumbv6m-none-eabi')
ap.add_argument('--cuts', action='store_true')
ap.add_argument('--halves', action='store_true', help='autocut may merge into 32-bit words of 64-bit spec nodes')
ap.add_argument('--cut-timeout', type=int, default=3000, help='ms per candidate merge')
ap.add_argument('--channels', default='', help='comma-separated subset of channels')
a = ap.parse_args()
manifest = json.load(open(a.obj + '.manifest.json'))
frames = manifest['object']['frames']
spec = {}
for asrt in z3.parse_smt2_string(open(a.spec).read()):
    spec[int(asrt.arg(0).decl().name().split('.')[1])] = asrt.arg(1)
nch = len(spec)
ports = manifest['object']['ports']
callee = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}
regs0 = dict(callee); loads = {}; out_base = None
taus = [z3.BitVec('tau@%d' % k, 64) for k in range(frames)]
for i, p in enumerate(ports):
    base = 'arg%d' % i
    if i < 4: regs0['r%d' % i] = ('ptr', base, 0)
    else: loads[('sp', 4 * (i - 4))] = ('ptr', base, 0)     # AAPCS: stacked arguments at sp on entry
    if p['name'] == 'tau':
        for k, t in enumerate(taus):
            loads[(base, 8 * k)] = z3.Extract(31, 0, t); loads[(base, 8 * k + 4)] = z3.Extract(63, 32, t)
    elif p['name'] == 'out': out_base = base
    elif p['elem'] == 'i32': loads[(base, 0)] = z3.BitVec(p['name'], 32)
    else: raise SystemExit('port %s: %s not modeled' % (p['name'], p['elem']))
dis = subprocess.run([OBJDUMP, '-dr', '--triple=' + a.triple, a.obj], capture_output=True, text=True).stdout
t = time.time()
st, npaths, steps = thumbsym2.run(dis, manifest['object']['symbol'], regs0, loads)
print('thumb lift (%d frames): %.2fs, %d returning state(s), %d steps' % (frames, time.time() - t, npaths, steps))
written = sorted(k for k in st.mem if k[0] != 'sp')
want = [(out_base, 4 * i) for i in range(frames * nch)]
print('stores outside the stack:', 'exactly out[0..%d)' % (frames * nch) if written == want else written)
assert written == want
tau = z3.BitVec('tau', 64)
for k in range(frames):
    for ch, term in sorted(spec.items()):
        if a.channels and ch not in [int(c) for c in a.channels.split(',')]: continue
        s_k = z3.substitute(term, (tau, taus[k]))
        o_k = st.mem[(out_base, 4 * (k * nch + ch))]
        if a.cuts:
            import autocut
            autocut.prove_with_cuts('frame %d ch %d: spec(tau_%d) == object' % (k, ch, k), s_k, o_k, halves=a.halves, cut_timeout_ms=a.cut_timeout)
        else:
            prove('frame %d ch %d: spec(tau_%d) == object' % (k, ch, k), s_k == o_k)
prove('r4-r7 restored, returns to lr',
      z3.And([st.R[r] == callee[r] for r in ('r4', 'r5', 'r6', 'r7')] + [st.ret == callee['lr']]))
