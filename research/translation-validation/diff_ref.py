"""Differential check of the symbolic carrier against the concrete reference:
evaluate the generated SMT-LIB spec (z3) on the vectors that tools/RefOfPlanJson.lean
ran through Semantics.observeFlatPlan over Semantics.Scalar.algebra (the sink image).
usage: diff_ref.py <spec.smt2> <vectors> <ref_out> <param names in slot order...>
A bounded check, not a proof."""
import sys, z3
spec, vecs, refs, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
terms = {int(a.arg(0).decl().name().split('.')[1]): a.arg(1) for a in z3.parse_smt2_string(open(spec).read())}
tau = z3.BitVec('tau', 64); ps = [z3.BitVec(n, 32) for n in names]
bad = 0; n = 0
for vl, rl in zip(open(vecs), open(refs)):
    w = [int(x) for x in vl.split()]
    sub = [(tau, z3.BitVecVal(w[0] % 2**64, 64))] + [(p, z3.BitVecVal(v % 2**32, 32)) for p, v in zip(ps, w[1:])]
    got = [z3.simplify(z3.substitute(terms[ch], *sub)).as_signed_long() for ch in sorted(terms)]
    want = [float(x) for x in rl.split()]
    n += 1
    if [float(g) for g in got] != want:
        bad += 1
        if bad < 5: print('MISMATCH', w, got, want)
print('%d vectors, %d mismatches' % (n, bad))
