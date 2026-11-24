"""Translation validation with a GENERATED specification (phase 2).

    spec   = `diffcli emit-spec <patch>`: the Plan reference semantics run over
             the symbolic carrier Tropical.Semantics.Symbolic (no hand-written spec)
    object = the thumbv6m object from `diffcli compile-object --frames 1`,
             lifted by thumbsym.py, its arguments bound from the manifest's
             `object.ports` (one pointer per port, in port order, AAPCS).

usage: tv_gen.py <patch.json> [--rate R] [--cpu C] [--triple T] [--diffcli PATH]
                 [--compare-handspec] [--cuts]
Proves, for every output channel and ALL values of the free inputs (each
`param:` port and tau), spec(...) == object(...). Kernel-agnostic: nothing in
here names a kernel.
"""
import argparse, json, os, subprocess, sys, time
import z3, thumbsym
from tv import prove, OBJDUMP

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patch')
    ap.add_argument('--rate', default='48000')
    ap.add_argument('--triple', default='thumbv6m-none-eabi')
    ap.add_argument('--cpu', default='cortex-m0plus')
    ap.add_argument('--diffcli', default=os.path.join(HERE, '../../lean/.lake/build/bin/diffcli'))
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'gen'))
    ap.add_argument('--compare-handspec', action='store_true')
    ap.add_argument('--cuts', action='store_true', help='also try automatic cut points (autocut.py)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, os.path.splitext(os.path.basename(a.patch))[0])
    env = dict(os.environ, TROPICAL_KERNEL_CACHE_ROOT=os.environ.get(
        'TROPICAL_KERNEL_CACHE_ROOT', os.path.join(HERE, '.cache')))

    t = time.time()
    r = subprocess.run([a.diffcli, 'emit-spec', a.patch, '--rate', a.rate],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit('emit-spec failed: ' + r.stderr.strip())
    open(stem + '.spec.smt2', 'w').write(r.stdout)
    print('emit-spec: %.2fs, %d lines -> %s' % (time.time() - t, r.stdout.count('\n'), stem + '.spec.smt2'))
    r = subprocess.run([a.diffcli, 'compile-object', a.patch, '--out', stem + '.o', '--frames', '1',
                        '--rate', a.rate, '--triple', a.triple, '--cpu', a.cpu, '--emit-ir'],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit('compile-object failed: ' + r.stderr.strip())
    manifest = json.load(open(stem + '.o.manifest.json'))

    spec = {}
    for asrt in z3.parse_smt2_string(r_text := open(stem + '.spec.smt2').read()):
        lhs, rhs = asrt.arg(0), asrt.arg(1)
        spec[int(lhs.decl().name().split('.')[1])] = rhs
    print('spec outputs:', sorted(spec))

    ports = manifest['object']['ports']
    if len(ports) > 4:
        sys.exit('more than four ports: stack-passed pointers are not modeled')
    regs0 = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}
    callee = dict(regs0)
    loads, out_base = {}, None
    for i, p in enumerate(ports):
        base = 'arg%d' % i
        regs0['r%d' % i] = ('ptr', base, 0)
        if p['name'] == 'tau':
            tau = z3.BitVec('tau', 64)
            loads[(base, 0)] = z3.Extract(31, 0, tau); loads[(base, 4)] = z3.Extract(63, 32, tau)
        elif p['name'] == 'out':
            out_base = base
        elif p['elem'] == 'i32':
            loads[(base, 0)] = z3.BitVec(p['name'], 32)
        else:
            sys.exit('port %s: element %s not modeled' % (p['name'], p['elem']))
    dis = subprocess.run([OBJDUMP, '-d', '--triple=' + a.triple, stem + '.o'],
                         capture_output=True, text=True).stdout
    t = time.time()
    res = thumbsym.run(dis, manifest['object']['symbol'], regs0, loads)
    print('thumb lift: %.2fs, %d paths' % (time.time() - t, len(res)))
    for ch, term in sorted(spec.items()):
        obj = thumbsym.merge(res, (out_base, 4 * ch))
        if a.cuts:
            import autocut
            autocut.prove_with_cuts('channel %d: generated spec == object' % ch, term, obj)
        else:
            prove('channel %d: generated spec == object' % ch, term == obj)
    prove('callee-saved r4-r7 restored on every path',
          z3.And([z3.And([R[r] == callee[r] for r in ('r4', 'r5', 'r6', 'r7')]) for _, _, R in res]))
    if a.compare_handspec:
        import spec as hand
        F, P, T = z3.BitVec('param:freq', 32), z3.BitVec('param:phase', 32), z3.BitVec('tau', 64)
        prove('generated spec == hand-written spec.py', spec[0] == hand.sinOscQ15(F, P, T))

if __name__ == '__main__':
    main()
