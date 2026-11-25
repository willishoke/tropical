"""Symbolic executor for the subset of textual LLVM IR that EmitLlvm emits for
an integer kernel.  Reads the .ll text as written by `diffcli compile-object
--emit-ir` (the very string handed to the object compiler), executes
@tropical_kernel with symbolic memory behind the pointer arguments, and
returns the z3 terms stored to %out.

Scope is deliberately narrow and refuses loudly: any opcode, flag (nsw/nuw/
exact/fast-math), type, or symbolic branch outside the subset raises.
Control flow is executed with concrete branch conditions (loop bounds are
static in these kernels), so loops are unrolled by execution.
Poison checks: constant shift amounts must be < width (else LangRef poison).
"""
import re
import z3

RNE = z3.RNE()
F64 = z3.Float64()

class Unsupported(Exception):
    pass

def parse_fn(text, name):
    m = re.search(r'define void @%s\((.*?)\)[^{]*\{\n(.*?)\n\}' % re.escape(name), text, re.S)
    if not m:
        raise Unsupported('no function ' + name)
    params = re.findall(r'ptr[^,]*?(%[\w.]+)(?:,|$)', m.group(1))
    blocks, order, cur = {}, [], None
    for line in m.group(2).split('\n'):
        s = line.strip()
        if not s:
            continue
        lm = re.match(r'^([\w.]+):$', s)
        if lm:
            cur = lm.group(1); blocks[cur] = []; order.append(cur); continue
        blocks[cur].append(s)
    return params, blocks, order

def ty_width(t):
    if t in ('i1', 'i8', 'i16', 'i32', 'i64'):
        return int(t[1:])
    raise Unsupported('type ' + t)

def fp_const(tok):
    if tok.startswith('0x'):
        bits = int(tok, 16)
        return z3.fpBVToFP(z3.BitVecVal(bits, 64), F64)
    raise Unsupported('fp const ' + tok)

def fptosi_sat(x, w):
    lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    return z3.If(z3.fpIsNaN(x), z3.BitVecVal(0, w),
           z3.If(z3.fpGEQ(x, z3.FPVal(float(hi + 1), F64)), z3.BitVecVal(hi, w),
           z3.If(z3.fpLEQ(x, z3.FPVal(float(lo), F64)), z3.BitVecVal(lo, w),
                 z3.fpToSBV(z3.RTZ(), x, z3.BitVecSort(w)))))

def run(text, args, fn='tropical_kernel'):
    """args: param name -> ('mem', {(byte_off, type): z3 term}) for inputs,
    or 'out' for the output region.  Returns {byte_off: term} stored to out."""
    params, blocks, order = parse_fn(text, fn)
    env, mem = {}, {}      # mem: (region, byte_off, type) -> term
    regions = {}
    for p in params:
        regions[p] = p
        env[p] = ('ptr', p, 0)
        for (off, t), v in args[p].items() if args[p] != 'out' else []:
            mem[(p, off, t)] = v
    outs = {}
    sizes = {'i32': 4, 'i64': 8, 'double': 8, 'ptr': 8}

    def val(tok, t):
        tok = tok.strip()
        if tok.startswith('%'):
            return env[tok]
        if t == 'double':
            return fp_const(tok)
        if t.startswith('i'):
            return z3.BitVecVal(int(tok) & ((1 << ty_width(t)) - 1), ty_width(t))
        raise Unsupported('operand ' + tok)

    def conc(v):
        s = z3.simplify(v)
        if z3.is_true(s): return True
        if z3.is_false(s): return False
        if z3.is_bv_value(s): return s.as_long()
        raise Unsupported('symbolic control value %s' % s)

    pred, blk, steps = None, order[0], 0
    while True:
        insts = blocks[blk]
        # phis first, all reading pred-block values simultaneously
        phis = {}
        i = 0
        while i < len(insts) and ' = phi ' in insts[i]:
            m = re.match(r'(%[\w.]+) = phi (\w+) (.*)$', insts[i])
            for vv, bb in re.findall(r'\[\s*([^,]+),\s*%([\w.]+)\s*\]', m.group(3)):
                if bb == pred:
                    phis[m.group(1)] = val(vv, m.group(2))
            i += 1
        env.update(phis)
        nxt = None
        for s in insts[i:]:
            steps += 1
            if steps > 200000: raise Unsupported('step budget')
            if s.startswith('br '):
                m = re.match(r'br label %([\w.]+)$', s)
                if m: nxt = m.group(1); break
                m = re.match(r'br i1 (%[\w.]+), label %([\w.]+), label %([\w.]+)$', s)
                nxt = m.group(2) if conc(env[m.group(1)]) else m.group(3); break
            if s == 'ret void':
                return outs
            m = re.match(r'store (\w+) ([^,]+), ptr (%[\w.]+), align \d+$', s)
            if m:
                t, v, p = m.groups(); _, reg, off = env[p]
                if args.get(reg) == 'out':
                    outs[off] = val(v, t)
                else:
                    # a store kills every overlapping entry of any type
                    for k in [k for k in mem if k[0] == reg and
                              k[1] < off + sizes[t] and off < k[1] + sizes[k[2]]]:
                        del mem[k]
                    mem[(reg, off, t)] = val(v, t)
                continue
            m = re.match(r'(%[\w.]+) = (.*)$', s)
            if not m: raise Unsupported(s)
            dst, rhs = m.groups()
            mm = re.match(r'alloca (\w+), i64 (\d+), align \d+$', rhs)
            if mm: env[dst] = ('ptr', dst, 0); continue
            mm = re.match(r'getelementptr inbounds (\w+), ptr (%[\w.]+), (i\d+) (\S+)$', rhs)
            if mm:
                t, p, it, ix = mm.groups(); _, reg, off = env[p]
                env[dst] = ('ptr', reg, off + sizes[t] * conc(val(ix, it))); continue
            mm = re.match(r'load (\w+), ptr (%[\w.]+), align \d+$', rhs)
            if mm:
                t, p = mm.groups(); _, reg, off = env[p]
                if (reg, off, t) not in mem: raise Unsupported('load of unwritten %s+%d:%s' % (reg, off, t))
                env[dst] = mem[(reg, off, t)]; continue
            mm = re.match(r'(add|sub|mul|and|or|xor|shl|ashr|lshr) (i\d+) ([^,]+), (.+)$', rhs)
            if mm:
                op, t, a, b = mm.groups()
                if ' ' in a: raise Unsupported('flagged op: ' + s)  # nsw/nuw/exact
                A, Bv = val(a, t), val(b, t); w = ty_width(t)
                if op in ('shl', 'ashr', 'lshr'):
                    sb = z3.simplify(Bv)
                    if not z3.is_bv_value(sb) or sb.as_long() >= w:
                        raise Unsupported('shift amount not a constant < width (poison): ' + s)
                env[dst] = {'add': lambda: A + Bv, 'sub': lambda: A - Bv, 'mul': lambda: A * Bv,
                            'and': lambda: A & Bv, 'or': lambda: A | Bv, 'xor': lambda: A ^ Bv,
                            'shl': lambda: A << Bv, 'ashr': lambda: A >> Bv,
                            'lshr': lambda: z3.LShR(A, Bv)}[op](); continue
            mm = re.match(r'(trunc|sext|zext) (i\d+) (\S+) to (i\d+)$', rhs)
            if mm:
                op, t, a, t2 = mm.groups(); A = val(a, t); w1, w2 = ty_width(t), ty_width(t2)
                env[dst] = (z3.Extract(w2 - 1, 0, A) if op == 'trunc' else
                            z3.SignExt(w2 - w1, A) if op == 'sext' else z3.ZeroExt(w2 - w1, A)); continue
            mm = re.match(r'icmp (eq|ne|slt|sgt|sle|sge|ult|ugt|ule|uge) (i\d+) ([^,]+), (.+)$', rhs)
            if mm:
                c, t, a, b = mm.groups(); A, Bv = val(a, t), val(b, t)
                env[dst] = {'eq': A == Bv, 'ne': A != Bv, 'slt': A < Bv, 'sgt': A > Bv,
                            'sle': A <= Bv, 'sge': A >= Bv, 'ult': z3.ULT(A, Bv),
                            'ugt': z3.UGT(A, Bv), 'ule': z3.ULE(A, Bv), 'uge': z3.UGE(A, Bv)}[c]; continue
            mm = re.match(r'select i1 (\S+), (\w+) ([^,]+), \w+ (.+)$', rhs)
            if mm:
                c, t, a, b = mm.groups(); env[dst] = z3.If(env[c], val(a, t), val(b, t)); continue
            mm = re.match(r'sitofp (i\d+) (\S+) to double$', rhs)
            if mm:
                env[dst] = z3.fpSignedToFP(RNE, val(mm.group(2), mm.group(1)), F64); continue
            mm = re.match(r'fdiv double ([^,]+), (.+)$', rhs)
            if mm:
                env[dst] = z3.fpDiv(RNE, val(mm.group(1), 'double'), val(mm.group(2), 'double')); continue
            mm = re.match(r'fcmp oeq double ([^,]+), (.+)$', rhs)
            if mm:
                env[dst] = z3.fpEQ(val(mm.group(1), 'double'), val(mm.group(2), 'double')); continue
            mm = re.match(r'call double @llvm\.round\.f64\(double (\S+)\)$', rhs)
            if mm:
                env[dst] = z3.fpRoundToIntegral(z3.RNA(), val(mm.group(1), 'double')); continue
            mm = re.match(r'call (i\d+) @llvm\.fptosi\.sat\.i\d+\.f64\(double (\S+)\)$', rhs)
            if mm:
                env[dst] = fptosi_sat(val(mm.group(2), 'double'), ty_width(mm.group(1))); continue
            raise Unsupported(s)
        pred, blk = blk, nxt
