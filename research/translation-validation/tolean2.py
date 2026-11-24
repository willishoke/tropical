"""Turn an autocut proof (generated spec == lifted object, with merges) into a
Lean file whose theorems are checked by the Lean kernel (bv_decide's LRAT
certificates, plus rfl/simp), so z3 leaves the trusted base: z3 is used only
as an oracle that picks the merges and the shape of each small obligation.

Every DAG node becomes a Lean def `n<k>` over the free inputs. For an object
node o, `p<k>` is its merged form (the spec node it was merged with, or its op
over merged children). Theorems:
  a<k> : op_o(merged children) = n<s>        -- one per merge, bv_decide
  b<k> : n<o> = p<o>                          -- every object node, rewriting
  main : n<object root> = n<spec root>
usage: tolean2.py <spec.smt2> <obj.o> <out.lean>   (one frame, ports from the manifest)
"""
import json, subprocess, sys
import z3, thumbsym, autocut
from tv import OBJDUMP

def lean_name(v):
    return 'v_' + ''.join(c if c.isalnum() else '_' for c in v)

class Emit:
    def __init__(self, inputs):
        self.inputs = inputs  # list of z3 consts
        self.args = ' '.join(lean_name(i.decl().name()) for i in inputs)
        self.sig = ' '.join('(%s : %s)' % (lean_name(i.decl().name()), self.ty(i)) for i in inputs)

    def ty(self, t):
        return 'BitVec %d' % t.size() if z3.is_bv(t) else 'Bool'

    def op(self, t, a):
        """Lean spelling of node t applied to child expressions a."""
        K = t.decl().kind()
        if K == z3.Z3_OP_BADD: return ' + '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_BSUB: return '(%s) - (%s)' % tuple(a)
        if K == z3.Z3_OP_BMUL: return ' * '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_BAND: return ' &&& '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_BOR: return ' ||| '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_BXOR: return ' ^^^ '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_BNOT: return '~~~(%s)' % a[0]
        if K == z3.Z3_OP_BNEG: return '-(%s)' % a[0]
        if K == z3.Z3_OP_BASHR: return "BitVec.sshiftRight' (%s) (%s)" % tuple(a)
        if K == z3.Z3_OP_BLSHR: return '(%s) >>> (%s)' % tuple(a)
        if K == z3.Z3_OP_BSHL: return '(%s) <<< (%s)' % tuple(a)
        if K == z3.Z3_OP_EXTRACT:
            hi, lo = t.decl().params(); return "BitVec.extractLsb' %d %d (%s)" % (lo, hi - lo + 1, a[0])
        if K == z3.Z3_OP_CONCAT: return ' ++ '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_SIGN_EXT: return 'BitVec.signExtend %d (%s)' % (t.size(), a[0])
        if K == z3.Z3_OP_ZERO_EXT: return 'BitVec.zeroExtend %d (%s)' % (t.size(), a[0])
        if K == z3.Z3_OP_ITE: return 'if (%s) then (%s) else (%s)' % tuple(a)
        if K == z3.Z3_OP_EQ: return '(%s) == (%s)' % tuple(a)
        if K == z3.Z3_OP_DISTINCT: return '(%s) != (%s)' % tuple(a)
        if K == z3.Z3_OP_NOT: return '!(%s)' % a[0]
        if K == z3.Z3_OP_AND: return ' && '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_OR: return ' || '.join('(%s)' % x for x in a)
        if K == z3.Z3_OP_SLT: return 'BitVec.slt (%s) (%s)' % tuple(a)
        if K == z3.Z3_OP_SGT: return 'BitVec.slt (%s) (%s)' % (a[1], a[0])
        if K == z3.Z3_OP_SLEQ: return 'BitVec.sle (%s) (%s)' % tuple(a)
        if K == z3.Z3_OP_SGEQ: return 'BitVec.sle (%s) (%s)' % (a[1], a[0])
        if K == z3.Z3_OP_ULT: return 'BitVec.ult (%s) (%s)' % tuple(a)
        if K == z3.Z3_OP_UGEQ: return '!(BitVec.ult (%s) (%s))' % tuple(a)
        raise Exception('no Lean spelling for %s' % t.decl())

def leaf(e, t):
    if z3.is_bv_value(t): return '(%d#%d)' % (t.as_long(), t.size())
    if z3.is_true(t): return 'true'
    if z3.is_false(t): return 'false'
    if z3.is_const(t): return lean_name(t.decl().name())
    return None

def main(spec_file, obj_file, out_file):
    manifest = json.load(open(obj_file + '.manifest.json'))
    spec = z3.parse_smt2_string(open(spec_file).read())[0].arg(1)
    ports = manifest['object']['ports']
    callee = {r: z3.BitVec('orig_' + r, 32) for r in ('r4', 'r5', 'r6', 'r7', 'lr')}
    regs0 = dict(callee); loads = {}; out_base = None
    for i, p in enumerate(ports):
        base = 'arg%d' % i; regs0['r%d' % i] = ('ptr', base, 0)
        if p['name'] == 'tau':
            t = z3.BitVec('tau', 64); loads[(base, 0)] = z3.Extract(31, 0, t); loads[(base, 4)] = z3.Extract(63, 32, t)
        elif p['name'] == 'out': out_base = base
        else: loads[(base, 0)] = z3.BitVec(p['name'], 32)
    dis = subprocess.run([OBJDUMP, '-d', '--triple=thumbv6m-none-eabi', obj_file], capture_output=True, text=True).stdout
    obj = thumbsym.merge(thumbsym.run(dis, manifest['object']['symbol'], regs0, loads), (out_base, 0))
    build(spec, obj, out_file, '%s and %s' % (spec_file, obj_file))

def build(spec, obj, out_file, source):
    """Emit the Lean file proving obj == spec (both z3 terms over the same inputs)."""
    r, cuts = autocut.prove_with_cuts('oracle', spec, obj, return_cuts=True)
    assert r == z3.unsat
    allnodes = {}
    autocut.nodes_of(spec, allnodes); autocut.nodes_of(obj, allnodes)
    specids = set(autocut.nodes_of(spec, {}))
    objids = set(autocut.nodes_of(obj, {}))
    inputs = sorted({t for t in allnodes.values() if leaf(None, t) and z3.is_const(t)
                     and not z3.is_bv_value(t) and not z3.is_true(t) and not z3.is_false(t)},
                    key=lambda t: t.decl().name())
    E = Emit(inputs)
    if obj.get_id() not in specids and all(o.get_id() != obj.get_id() for _, o in cuts):
        cuts = cuts + [(spec, obj)]          # the final obligation, as one more merge
    merged = {o.get_id(): s.get_id() for s, o in cuts}
    order = []  # topological (children first)
    seen = set()
    def topo(t):
        if t.get_id() in seen: return
        seen.add(t.get_id())
        for c in t.children(): topo(c)
        order.append(t)
    topo(spec); topo(obj)

    def ref(t):           # reference to the original node
        l = leaf(None, t)
        return l if l is not None else 'n%d %s' % (t.get_id(), E.args)
    def pref(t):          # reference to the merged form
        k = t.get_id()
        if leaf(None, t) is not None or k in specids: return ref(t)
        return 'p%d %s' % (k, E.args)
    out = ['import Std.Tactic.BVDecide', 'set_option maxRecDepth 40000', 'set_option maxHeartbeats 0', 'set_option linter.unusedSimpArgs false', 'set_option linter.unusedVariables false',
           '-- GENERATED by tolean2.py from %s' % source,
           'namespace TV', '']
    for t in order:
        if leaf(None, t) is not None: continue
        out.append('def n%d %s : %s := %s' % (t.get_id(), E.sig, E.ty(t),
                   E.op(t, ['(%s)' % ref(c) for c in t.children()])))
    for t in order:
        k = t.get_id()
        if leaf(None, t) is not None or k in specids or k not in objids: continue
        if k in merged:
            body = 'n%d %s' % (merged[k], E.args)
        else:
            body = E.op(t, ['(%s)' % pref(c) for c in t.children()])
        out.append('def p%d %s : %s := %s' % (k, E.sig, E.ty(t), body))
    out.append('')
    # a: merge obligations. Choose which spec defs to unfold with z3 as an oracle:
    # the frontier = spec nodes referenced by the merged children; unfold spec
    # nodes under s down to the frontier; if that abstraction is not valid,
    # unfold everything to the inputs.
    pdefs = ', '.join('p%d' % t.get_id() for t in order
                     if leaf(None, t) is None and t.get_id() not in specids and t.get_id() in objids)
    def spec_below(s, stop):
        acc, st = set(), [s]
        while st:
            x = st.pop()
            if x.get_id() in acc or leaf(None, x) is not None or x.get_id() in stop: continue
            acc.add(x.get_id()); st.extend(x.children())
        return acc
    def merged_expr_frontier(o):
        fr, st = set(), list(o.children())
        while st:
            x = st.pop(); k = x.get_id()
            if leaf(None, x) is not None: continue
            if k in specids: fr.add(k)
            elif k in merged: fr.add(merged[k])
            else: st.extend(x.children())
        return fr
    subs_all = [(o, s_node) for s_node, o in cuts]
    def obj_pdefs(o):
        """p defs needed to unfold o's merged children down to spec atoms."""
        acc, st = [], list(o.children())
        while st:
            x = st.pop(); k = x.get_id()
            if leaf(None, x) is not None or k in specids: continue
            acc.append('p%d' % k)
            if k not in merged: st.extend(x.children())
        return acc
    nfull = 0
    def views(o, s_node, unfold):
        """z3 terms for exactly what the Lean goal looks like after
        `simp only [p-defs, n-defs in unfold]`: p-chains expanded, merged
        nodes replaced by their spec node, spec nodes outside `unfold`
        left as opaque atoms."""
        atoms, memo = {}, {}
        def vs(t):
            k = t.get_id()
            if leaf(None, t) is not None: return t
            if k not in unfold:
                if k not in atoms: atoms[k] = z3.FreshConst(t.sort(), 'atom')
                return atoms[k]
            if ('s', k) not in memo: memo[('s', k)] = t.decl()(*[vs(c) for c in t.children()])
            return memo[('s', k)]
        def vo(t):
            k = t.get_id()
            if leaf(None, t) is not None: return t
            if k in specids: return vs(t)
            if k in merged and t is not o_root[0]: return vs(allnodes[merged[k]])
            if ('o', k) not in memo: memo[('o', k)] = t.decl()(*[vo(c) for c in t.children()])
            return memo[('o', k)]
        o_root = [o]
        lhs = o.decl()(*[vo(c) for c in o.children()]) if o.num_args() else o
        return lhs, vs(s_node), atoms
    for i, (s_node, o) in enumerate(cuts):
        k, sk = o.get_id(), s_node.get_id()
        unfold = {sk}
        for depth in range(8):
            lhs_z, rhs_z, atoms = views(o, s_node, unfold)
            sv = z3.SolverFor('QF_BV'); sv.set('timeout', 5000); sv.add(lhs_z != rhs_z)
            if sv.check() == z3.unsat: break
            unfold |= set(atoms)          # unfold every opaque spec node one more level
        else:
            nfull += 1
            unfold = spec_below(s_node, set())
            for f in merged_expr_frontier(o): unfold |= spec_below(allnodes[f], set())
        names = ', '.join(list(dict.fromkeys(obj_pdefs(o))) + ['n%d' % u for u in sorted(unfold)])
        lhs = E.op(o, ['(%s)' % pref(c) for c in o.children()])
        out.append('theorem a%d %s : (%s) = n%d %s := by' % (k, E.sig, lhs, sk, E.args))
        if names: out.append('  try simp only [%s]' % names)
        comm = ' | exact BitVec.mul_comm _ _' if o.decl().kind() == z3.Z3_OP_BMUL and o.num_args() == 2 else ''
        # no plain `rfl`: definitional unfolding of BitVec arithmetic over opaque
        # atoms can take minutes or hit the recursion limit (which `first`
        # cannot catch); bv_decide settles these in well under a second
        rfl = ''
        out.append('  all_goals first | with_reducible rfl%s%s | bv_decide (config := { acNf := true, timeout := 300 })' % (comm, rfl))
    print('merge obligations: %d with an opaque frontier, %d fully unfolded' % (len(cuts) - nfull, nfull))
    # b: original object nodes equal their merged forms, bottom-up
    for t in order:
        k = t.get_id()
        if leaf(None, t) is not None or k in specids or k not in objids: continue
        kids = list({c.get_id(): c for c in t.children() if leaf(None, c) is None and c.get_id() not in specids}.values())
        bs = ' '.join('b%d' % c.get_id() for c in kids)
        out.append('theorem b%d %s : n%d %s = p%d %s := by' % (k, E.sig, k, E.args, k, E.args))
        if k in merged:
            out.append('  rw [n%d%s, p%d]; exact a%d %s' % (k, ''.join(', b%d' % c.get_id() for c in kids), k, k, E.args))
        else:
            out.append('  rw [n%d%s, p%d]' % (k, ''.join(', b%d' % c.get_id() for c in kids), k))
    ok, sk = obj.get_id(), spec.get_id()
    out.append('')
    out.append('/-- The lifted object computes the generated specification, for all inputs. -/')
    out.append('theorem main %s : n%d %s = n%d %s := by' % (E.sig, ok, E.args, sk, E.args))
    if ok == sk:
        out.append('  rfl')
    elif merged.get(ok) == sk:
        out.append('  rw [b%d, p%d]' % (ok, ok))
    else:
        # the object root was merged into another spec node s' that equals the
        # spec root (e.g. the root is `s' + 0`): close s' = root with s' opaque
        s2 = merged[ok]
        names = ', '.join('n%d' % u for u in sorted(spec_below(spec, {s2})))
        main_line = out.pop()
        out.append('theorem fin %s : n%d %s = n%d %s := by' % (E.sig, s2, E.args, sk, E.args))
        out.append('  try simp only [%s]' % names)
        out.append('  all_goals first | with_reducible rfl | bv_decide (config := { acNf := true, timeout := 300 })')
        out.append(main_line)
        out.append('  rw [b%d, p%d, fin]' % (ok, ok))
    out.append('')
    out.append('#print axioms main')
    out.append('end TV')
    open(out_file, 'w').write('\n'.join(out) + '\n')
    print('wrote %s: %d defs, %d merge theorems' % (out_file, sum(1 for l in out if l.startswith('def ')), len(cuts)))

if __name__ == '__main__':
    main(*sys.argv[1:4])
