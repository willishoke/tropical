"""Automatic cut points for spec == object (combinational equivalence checking
in the style of SAT sweeping / fraiging): random simulation proposes pairs of
internal spec/object nodes with identical value signatures; each pair is proved
equal with z3 (over the original inputs), then the object node is MERGED into
the spec node: every later occurrence of the object node is replaced by the
spec node. The final obligation is spec == object-with-merges.

Soundness: each merge replaces a subterm by a term proved equal to it for all
inputs, so object-with-merges denotes the same function as the object.
(Replacing both sides by a shared fresh variable instead -- the first version
-- is also sound but incomplete: the fresh variables forget correlations, and
the final query came back with a spurious counterexample.)"""
import random, time
import z3
from tv import prove

M = lambda w: (1 << w) - 1

def sx(v, w):
    v &= M(w)
    return v - (1 << w) if v >> (w - 1) else v

def nodes_of(e, acc):
    seen = {}
    stack = [e]
    while stack:
        t = stack.pop()
        if t.get_id() in seen: continue
        seen[t.get_id()] = t
        stack.extend(t.children())
    acc.update(seen)
    return acc

def evaluate(t, env, memo):
    k = t.get_id()
    if k in memo: return memo[k]
    K = t.decl().kind()
    if z3.is_bv_value(t): v = t.as_long()
    elif z3.is_true(t): v = 1
    elif z3.is_false(t): v = 0
    elif z3.is_const(t): v = env[t.decl().name()]
    else:
        a = [evaluate(c, env, memo) for c in t.children()]
        w = t.size() if z3.is_bv(t) else 1
        cw = t.children()[0].size() if t.num_args() and z3.is_bv(t.children()[0]) else 1
        if K == z3.Z3_OP_BADD: v = sum(a)
        elif K == z3.Z3_OP_BSUB: v = a[0] - a[1]
        elif K == z3.Z3_OP_BMUL:
            v = 1
            for x in a: v *= x
        elif K == z3.Z3_OP_BAND:
            v = M(w)
            for x in a: v &= x
        elif K == z3.Z3_OP_BOR:
            v = 0
            for x in a: v |= x
        elif K == z3.Z3_OP_BXOR:
            v = 0
            for x in a: v ^= x
        elif K == z3.Z3_OP_BNOT: v = ~a[0]
        elif K == z3.Z3_OP_BNEG: v = -a[0]
        elif K == z3.Z3_OP_BSHL: v = a[0] << a[1] if a[1] < w else 0
        elif K == z3.Z3_OP_BLSHR: v = a[0] >> a[1] if a[1] < w else 0
        elif K == z3.Z3_OP_BASHR: v = sx(a[0], w) >> min(a[1], w - 1)
        elif K == z3.Z3_OP_EXTRACT:
            hi, lo = t.decl().params(); v = a[0] >> lo
        elif K == z3.Z3_OP_CONCAT:
            v = 0
            for c, x in zip(t.children(), a): v = (v << c.size()) | x
        elif K == z3.Z3_OP_SIGN_EXT: v = sx(a[0], cw)
        elif K == z3.Z3_OP_ZERO_EXT: v = a[0]
        elif K == z3.Z3_OP_ITE: v = a[1] if a[0] else a[2]
        elif K == z3.Z3_OP_EQ: v = int(a[0] == a[1])
        elif K == z3.Z3_OP_DISTINCT: v = int(a[0] != a[1])
        elif K == z3.Z3_OP_NOT: v = 1 - a[0]
        elif K == z3.Z3_OP_AND: v = int(all(a))
        elif K == z3.Z3_OP_OR: v = int(any(a))
        elif K == z3.Z3_OP_SLT: v = int(sx(a[0], cw) < sx(a[1], cw))
        elif K == z3.Z3_OP_SLEQ: v = int(sx(a[0], cw) <= sx(a[1], cw))
        elif K == z3.Z3_OP_SGT: v = int(sx(a[0], cw) > sx(a[1], cw))
        elif K == z3.Z3_OP_SGEQ: v = int(sx(a[0], cw) >= sx(a[1], cw))
        elif K == z3.Z3_OP_ULT: v = int(a[0] < a[1])
        elif K == z3.Z3_OP_ULEQ: v = int(a[0] <= a[1])
        elif K == z3.Z3_OP_UGT: v = int(a[0] > a[1])
        elif K == z3.Z3_OP_UGEQ: v = int(a[0] >= a[1])
        elif K == z3.Z3_OP_BUREM: v = a[0] if a[1] == 0 else a[0] % a[1]
        elif K == z3.Z3_OP_BSREM:
            x, y = sx(a[0], cw), sx(a[1], cw)
            v = x if y == 0 else (abs(x) % abs(y)) * (1 if x >= 0 else -1)
        elif K == z3.Z3_OP_BSDIV:
            x, y = sx(a[0], cw), sx(a[1], cw)
            if y == 0: v = -1 if x >= 0 else 1          # SMT-LIB bvsdiv by zero
            else: v = (abs(x) // abs(y)) * (1 if (x < 0) == (y < 0) else -1)
        else: raise Exception('autocut: no evaluator for %s' % t.decl())
        v &= M(w)
    memo[k] = v
    return v

def prove_with_cuts(name, spec, obj, samples=48, cut_timeout_ms=3000, seed=1, return_cuts=False, halves=False):
    t0 = time.time()
    rng = random.Random(seed)
    allnodes = {}
    nodes_of(spec, allnodes); nodes_of(obj, allnodes)
    specids = set(nodes_of(spec, {}))
    if halves:
        # 32-bit words of 64-bit spec nodes as extra merge targets, so an
        # object that computes a 64-bit ring op in two words (adds/adcs,
        # limb products) can be aligned word by word
        for k in list(specids):
            t = allnodes[k]
            if z3.is_bv(t) and t.size() == 64 and not z3.is_const(t):
                for hl in (z3.Extract(31, 0, t), z3.Extract(63, 32, t)):
                    allnodes[hl.get_id()] = hl; specids.add(hl.get_id())
    inputs = {t.decl().name(): t.size() if z3.is_bv(t) else 1
              for t in allnodes.values() if z3.is_const(t) and not z3.is_bv_value(t)
              and not z3.is_true(t) and not z3.is_false(t)}
    edge = [0, 1, -1, 2**31 - 1, -2**31, 2**15, 2**16 - 1]
    envs = []
    for i in range(samples):
        envs.append({n: (rng.choice(edge) if i < 8 else rng.getrandbits(w)) & M(w) for n, w in inputs.items()})
    sig = {}
    for k, t in allnodes.items():
        if not z3.is_bv(t) or z3.is_const(t): continue
        sig[k] = (t.size(),)
    for env in envs:
        memo = {}
        for k in sig:
            sig[k] = sig[k] + (evaluate(allnodes[k], env, memo),)
    by_sig = {}
    for k in specids:
        if k in sig: by_sig.setdefault(sig[k], []).append(k)
    # object nodes bottom-up (small terms first)
    size = {}
    def tsize(t):
        k = t.get_id()
        if k not in size: size[k] = 1 + sum(tsize(c) for c in t.children())
        return size[k]
    cands = sorted((k for k in sig if k not in specids and sig[k] in by_sig), key=lambda k: tsize(allnodes[k]))
    cuts = []  # (spec node, obj node)
    tried = proved = 0
    for k in cands:
        o = allnodes[k]
        subs = [(c[1], c[0]) for c in cuts]
        o_c = z3.substitute(o, *subs) if subs else o
        for sk in by_sig[sig[k]][:2]:
            s_node = allnodes[sk]
            s_c = s_node
            tried += 1
            sv = z3.SolverFor('QF_BV'); sv.set('timeout', cut_timeout_ms)
            sv.add(s_c != o_c)
            if sv.check() == z3.unsat:
                cuts.append((s_node, o)); proved += 1
                break
    subs = [(c[1], c[0]) for c in cuts]
    s_f = spec
    o_f = z3.substitute(obj, *subs) if subs else obj
    print('autocut: %d candidate pairs tried, %d merges proved, %.2fs' % (tried, proved, time.time() - t0))
    r = prove(name + ' [auto cuts]', s_f == o_f)
    return (r, cuts) if return_cuts else r
