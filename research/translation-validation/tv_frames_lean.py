"""Kernel-checked version of tv_frames.py: for a --frames N object, emit one
Lean file per frame k proving  out[k] (lifted by thumbsym2) = spec(params, tau_k),
then check them with `lake env lean` in parallel. The structural facts (stores
only to out[0..N) and the stack, r4-r7 restored, return to lr) stay z3/lifter
claims (tv_frames.py); the per-frame values are the Lean theorems.
usage: tv_frames_lean.py --spec S --obj O [--jobs J] [--frames-only K1,K2,...]"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import z3, thumbsym2, tolean2
from tv import OBJDUMP

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True); ap.add_argument('--obj', required=True)
ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--out-dir', default=os.path.join(HERE, 'gen', 'frames'))
ap.add_argument('--frames-only', default='')
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
manifest = json.load(open(a.obj + '.manifest.json'))
frames = manifest['object']['frames']
spec = z3.parse_smt2_string(open(a.spec).read())[0].arg(1)
ports = manifest['object']['ports']
callee = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}
regs0 = dict(callee); loads = {}; out_base = None
taus = [z3.BitVec('tau@%d' % k, 64) for k in range(frames)]
for i, p in enumerate(ports):
    base = 'arg%d' % i; regs0['r%d' % i] = ('ptr', base, 0)
    if p['name'] == 'tau':
        for k, t in enumerate(taus):
            loads[(base, 8 * k)] = z3.Extract(31, 0, t); loads[(base, 8 * k + 4)] = z3.Extract(63, 32, t)
    elif p['name'] == 'out': out_base = base
    else: loads[(base, 0)] = z3.BitVec(p['name'], 32)
dis = subprocess.run([OBJDUMP, '-dr', '--triple=thumbv6m-none-eabi', a.obj], capture_output=True, text=True).stdout
st, _, _ = thumbsym2.run(dis, manifest['object']['symbol'], regs0, loads)
tau = z3.BitVec('tau', 64)
ks = [int(x) for x in a.frames_only.split(',')] if a.frames_only else list(range(frames))
files = []
t0 = time.time()
for k in ks:
    f = os.path.join(a.out_dir, 'frame%03d.lean' % k)
    tolean2.build(z3.substitute(spec, (tau, taus[k])), st.mem[(out_base, 4 * k)], f,
                  'frame %d of %s against %s' % (k, a.obj, a.spec))
    files.append(f)
print('generated %d files in %.0fs' % (len(files), time.time() - t0))
LEAN_DIR = os.path.join(HERE, '../../lean')
def check(f):
    r = subprocess.run(['lake', 'env', 'lean', '--tstack=200000', os.path.abspath(f)], cwd=LEAN_DIR,
                       capture_output=True, text=True,
                       env=dict(os.environ, PATH=os.path.expanduser('~/.elan/bin') + ':' + os.environ['PATH']))
    out = r.stdout + r.stderr
    ok = r.returncode == 0 and 'error' not in out and 'sorry' not in out and "depends on axioms" in out
    return f, ok, out
t0 = time.time()
with ThreadPoolExecutor(a.jobs) as ex:
    res = list(ex.map(check, files))
bad = [f for f, ok, _ in res if not ok]
print('lean: %d/%d frame theorems checked in %.0fs' % (len(res) - len(bad), len(res), time.time() - t0))
for f, ok, out in res:
    if not ok: print('FAILED', f, out[:600])
