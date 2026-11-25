"""Where does spec == object get hard? After autocut, list the lowest spec
nodes that no object node was merged into although all their spec children
were matched (or are leaves): the first places where the two computations
part ways. usage: diag_unmatched.py <spec.smt2> <obj.o> <channel> [--halves]"""
import json, subprocess, sys, collections
import z3, thumbsym2, autocut
from tv import OBJDUMP
spec_file, obj_file, ch = sys.argv[1], sys.argv[2], int(sys.argv[3])
halves = '--halves' in sys.argv
m = json.load(open(obj_file + '.manifest.json'))
specs = {int(x.arg(0).decl().name().split('.')[1]): x.arg(1) for x in z3.parse_smt2_string(open(spec_file).read())}
regs0 = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}; loads = {}
for i, p in enumerate(m['object']['ports']):
    base = 'arg%d' % i
    if i < 4: regs0['r%d' % i] = ('ptr', base, 0)
    else: loads[('sp', 4 * (i - 4))] = ('ptr', base, 0)
    if p['name'] == 'tau':
        t = z3.BitVec('tau', 64); loads[(base, 0)] = z3.Extract(31, 0, t); loads[(base, 4)] = z3.Extract(63, 32, t)
    elif p['name'] == 'out': out_base = base
    else: loads[(base, 0)] = z3.BitVec(p['name'], 32)
dis = subprocess.run([OBJDUMP, '-dr', '--triple=thumbv6m-none-eabi', obj_file], capture_output=True, text=True).stdout
st, _, _ = thumbsym2.run(dis, m['object']['symbol'], regs0, loads)
spec, obj = specs[ch], st.mem[(out_base, 4 * ch)]
r, cuts = autocut.prove_with_cuts('diag', spec, obj, return_cuts=True, halves=halves, cut_timeout_ms=3000)
matched = {s.get_id() for s, _ in cuts}
for s, _ in cuts:            # a matched half matches its 64-bit parent's word
    if s.decl().kind() == z3.Z3_OP_EXTRACT: matched.add(s.get_id())
nodes = autocut.nodes_of(spec, {})
def is_leaf(t): return z3.is_const(t) or z3.is_bv_value(t)
def ok(t): return is_leaf(t) or t.get_id() in matched
first = [t for t in nodes.values() if not ok(t) and all(ok(c) for c in t.children())]
print('%d spec nodes, %d matched, %d first-unmatched' % (len(nodes), len(matched & set(nodes)), len(first)))
print(collections.Counter('%s/%d' % (t.decl().name(), t.size() if z3.is_bv(t) else 1) for t in first).most_common(20))
for t in first[:8]: print('  ', t.sexpr()[:160].replace('\n', ' '))
