"""Symbolic executor for Thumb-1 (ARMv6-M) kernel objects WITH the frame loop,
read from `llvm-objdump -d` text. Successor of thumbsym.py (straight-line only).

Model
* Registers hold z3 32-bit terms or abstract pointers ('ptr', base, byte_off).
  Pointer arithmetic is allowed only with a concrete offset.
* Memory is a map (base, byte_off) -> 32-bit term; input regions are seeded by
  the caller (`loads`), `sp` is a region of its own (push/pop/spills), every
  store is recorded. Distinct bases do not alias (the kernel ABI's noalias).
* Control: a conditional branch whose condition simplifies to a constant takes
  one edge; otherwise both, with path conditions. States waiting at the same
  pc with the same concrete skeleton (pointers, concrete words) are MERGED
  with If on the symbolic values, so a clamp diamond costs nothing and the
  loop does not multiply paths. Scheduling is by (back-edges taken, pc), so
  both sides of a diamond reach the join before either moves on.
* Flags: N, Z, C, V are computed for cmp (register or immediate); any other
  flag-setting instruction clobbers them (the kernels branch only after cmp).
Refuses any other mnemonic or form.
"""
import re
import z3

class Unsupported(Exception):
    pass

def bv(n): return z3.BitVecVal(n & 0xFFFFFFFF, 32)

def parse(objdump_text, fn):
    insts, words, relocs, on = {}, {}, {}, False
    for l in objdump_text.split('\n'):
        if re.match(r'^[0-9a-f]+ <%s>:$' % re.escape(fn), l):
            on = True; continue
        if on and (re.match(r'^[0-9a-f]+ <', l) or l.startswith('Disassembly')):
            break
        if not on: continue
        m = re.match(r'^\s+([0-9a-f]+):\s+R_ARM_\S+\s+(\S+)', l)
        if m:
            relocs[int(m.group(1), 16)] = m.group(2); continue
        m = re.match(r'^\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s*\.word\s+0x([0-9a-f]+)', l)
        if m:
            words[int(m.group(1), 16)] = int(m.group(3), 16); continue
        m = re.match(r'^\s+([0-9a-f]+):\s+([0-9a-f]{4})( [0-9a-f]{4})?\s+(\S+)\s*(.*?)\s*$', l)
        if m:
            size = 4 if m.group(3) else 2
            insts[int(m.group(1), 16)] = (m.group(4), m.group(5).split('@')[0].strip(), size)
    return insts, words, relocs

def w64(lo, hi): return z3.Concat(hi, lo)

# Run-time ABI helpers (AEABI, ARM IHI 0043) a call may reach, by their
# specified effect on r0-r3. Their implementations are TRUSTED to meet this.
def aeabi(sym, R):
    a = w64(R['r0'], R['r1'])
    if sym == '__aeabi_lmul':
        r = a * w64(R['r2'], R['r3'])
    elif sym in ('__aeabi_llsl', '__aeabi_llsr', '__aeabi_lasr'):
        n = z3.ZeroExt(32, R['r2'] & 0xFF)      # shift count in r2 (low byte); >= 64 is unspecified
        r = a << n if sym == '__aeabi_llsl' else z3.LShR(a, n) if sym == '__aeabi_llsr' else a >> n
    else:
        raise Unsupported('call to %s: no contract modeled' % sym)
    return z3.Extract(31, 0, r), z3.Extract(63, 32, r)

def imm(t): return int(t.strip().lstrip('#'), 0)

def split_ops(ops):
    return [x.strip() for x in re.split(r',(?![^\[{]*[\]}])', ops) if x.strip()]

class State:
    def __init__(self, pc, R, mem, cond, flags, back):
        self.pc, self.R, self.mem, self.cond, self.flags, self.back = pc, R, mem, cond, flags, back
    def copy(self):
        return State(self.pc, dict(self.R), dict(self.mem), self.cond, self.flags, self.back)

def is_ptr(v): return isinstance(v, tuple)

def skeleton(v):
    if is_ptr(v): return v
    s = z3.simplify(v)
    return ('c', s.as_long()) if z3.is_bv_value(s) else 'sym'

def mergeable(a, b):
    if a.pc != b.pc or a.back != b.back or set(a.R) != set(b.R) or set(a.mem) != set(b.mem):
        return False
    if (a.flags is None) != (b.flags is None): return False
    for k in a.R:
        sa, sb = skeleton(a.R[k]), skeleton(b.R[k])
        if (is_ptr(a.R[k]) or is_ptr(b.R[k]) or sa != 'sym' and sb != 'sym') and sa != sb: return False
    for k in a.mem:
        if is_ptr(a.mem[k]) != is_ptr(b.mem[k]) or is_ptr(a.mem[k]) and a.mem[k] != b.mem[k]: return False
    return True

def _complementary(x, y):
    return z3.eq(y, z3.Not(x)) or z3.eq(x, z3.Not(y))

def merge2(a, b):
    """Merge two states at the same pc. Path conditions are tuples of
    conjuncts; the If uses only the conjuncts past the common prefix, and
    two sides of one fork merge back to the prefix, so conditions do not
    grow from frame to frame."""
    n = 0
    while n < min(len(a.cond), len(b.cond)) and z3.eq(a.cond[n], b.cond[n]): n += 1
    ra, rb = a.cond[n:], b.cond[n:]
    c = z3.And(*ra) if ra else z3.BoolVal(True)
    if len(ra) == 1 and len(rb) == 1 and _complementary(ra[0], rb[0]):
        cond = a.cond[:n]
    else:
        cond = a.cond[:n] + (z3.Or(c, z3.And(*rb) if rb else z3.BoolVal(True)),)
    def m(x, y):
        if is_ptr(x): return x
        return x if x.eq(y) else z3.If(c, x, y)
    R = {k: m(a.R[k], b.R[k]) for k in a.R}
    mem = {k: m(a.mem[k], b.mem[k]) for k in a.mem}
    flags = None if a.flags is None else tuple(None if x is None or y is None else z3.If(c, x, y)
                                               for x, y in zip(a.flags, b.flags))
    return State(a.pc, R, mem, cond, flags, a.back)

def run(objdump_text, fn, regs0, loads, max_steps=200000):
    """regs0: initial registers ('sp' is set here); loads: {(base, off): BV32}
    for the read-only input regions.  Returns the merged final state(s)."""
    insts, words, relocs = parse(objdump_text, fn)
    R = dict(regs0); R['sp'] = ('ptr', 'sp', 0)
    pending = [State(0, R, {}, (), None, 0)]
    done, steps = [], 0
    while pending:
        pending.sort(key=lambda s: (s.back, s.pc))
        st = pending.pop(0)
        # merge every waiting state at the same pc and skeleton
        rest = []
        for o in pending:
            if mergeable(st, o): st = merge2(st, o)
            else: rest.append(o)
        pending = rest
        while True:
            steps += 1
            if steps > max_steps: raise Unsupported('step budget')
            if st.pc not in insts: raise Unsupported('pc %x' % st.pc)
            mn, ops, size = insts[st.pc]
            o = split_ops(ops)
            nxt = st.pc + size
            R = st.R
            def g(r):
                v = R[r]
                if is_ptr(v): raise Unsupported('word op on pointer %s at %x' % (r, st.pc))
                return v
            def conc(v):
                s = z3.simplify(v)
                if not z3.is_bv_value(s): raise Unsupported('symbolic address offset at %x' % st.pc)
                return s.as_long()
            def addr(base_reg, off):
                p = R[base_reg]
                if not is_ptr(p): raise Unsupported('address from a non-pointer at %x' % st.pc)
                return (p[1], p[2] + off)
            def load(a):
                if a in st.mem: return st.mem[a]
                if a in loads: return loads[a]
                raise Unsupported('load from unknown %s at %x' % (a, st.pc))
            def store(a, v):
                st.mem = dict(st.mem); st.mem[a] = v
            def add_any(x, y, sub=False):
                if is_ptr(x) and not is_ptr(y):
                    d = conc(y); return (x[0], x[1], x[2] + (-d if sub else d))
                if is_ptr(x) or is_ptr(y): raise Unsupported('pointer + pointer at %x' % st.pc)
                return x - y if sub else x + y
            branch = None
            nzcv = None
            flagset = mn in ('movs', 'muls', 'adds', 'subs', 'asrs', 'lsls', 'lsrs', 'ands', 'orrs', 'eors', 'mvns', 'rsbs')
            if mn == 'push':
                rs = re.findall(r'r\d+|lr', o[0])
                sp = R['sp']; base = sp[2] - 4 * len(rs)
                for i, r in enumerate(rs): store(('sp', base + 4 * i), R[r])
                R['sp'] = ('ptr', 'sp', base)
            elif mn == 'pop':
                rs = re.findall(r'r\d+|pc', o[0])
                sp = R['sp']
                vals = [load(('sp', sp[2] + 4 * i)) for i in range(len(rs))]
                R['sp'] = ('ptr', 'sp', sp[2] + 4 * len(rs))
                for r, v in zip(rs, vals):
                    if r != 'pc': R[r] = v
                if 'pc' in rs:
                    st.ret = vals[rs.index('pc')]
                    done.append(st); break
            elif mn in ('sub', 'add') and o[0] == 'sp':
                R['sp'] = add_any(R['sp'], bv(imm(o[-1])), sub=(mn == 'sub'))
            elif mn == 'ldr':
                m = re.match(r'\[pc, #(-?0x[0-9a-f]+|-?\d+)\]', o[1])
                if m:
                    R[o[0]] = bv(words[((st.pc + 4) & ~3) + int(m.group(1), 0)])
                else:
                    m = re.match(r'\[(r\d+|sp)(?:, #(0x[0-9a-f]+|\d+))?\]$', o[1])
                    if m:
                        R[o[0]] = load(addr(m.group(1), int(m.group(2), 0) if m.group(2) else 0))
                    else:
                        m = re.match(r'\[(r\d+), (r\d+)\]$', o[1])
                        if not m: raise Unsupported('ldr form %s' % ops)
                        R[o[0]] = load(addr(m.group(1), conc(g(m.group(2)))))
            elif mn == 'str':
                m = re.match(r'\[(r\d+|sp)(?:, #(0x[0-9a-f]+|\d+))?\]$', o[1])
                if m:
                    store(addr(m.group(1), int(m.group(2), 0) if m.group(2) else 0), R[o[0]])
                else:
                    m = re.match(r'\[(r\d+), (r\d+)\]$', o[1])
                    if not m: raise Unsupported('str form %s' % ops)
                    store(addr(m.group(1), conc(g(m.group(2)))), R[o[0]])
            elif mn == 'ldm':
                base = o[0].rstrip('!'); rs = re.findall(r'r\d+', o[1])
                p = R[base]
                for i, r in enumerate(rs): R[r] = load((p[1], p[2] + 4 * i))
                if o[0].endswith('!') and base not in rs:
                    R[base] = (p[0], p[1], p[2] + 4 * len(rs))
            elif mn == 'stm':
                base = o[0].rstrip('!'); rs = re.findall(r'r\d+', o[1])
                p = R[base]
                for i, r in enumerate(rs): store((p[1], p[2] + 4 * i), R[r])
                if o[0].endswith('!'):
                    R[base] = (p[0], p[1], p[2] + 4 * len(rs))
            elif mn == 'bl':
                sym = relocs.get(st.pc)
                if sym is None: raise Unsupported('bl without a relocation at %x' % st.pc)
                lo, hi = aeabi(sym, R)
                st.calls = getattr(st, 'calls', set()) | {sym}
                R['r0'], R['r1'] = lo, hi
                for r in ('r2', 'r3', 'r12', 'lr'):  # caller-saved and the link register: clobbered
                    R[r] = z3.FreshConst(z3.BitVecSort(32), 'clobber_' + r)
                st.flags = None
            elif mn == 'mov':
                R[o[0]] = R[o[1]]
            elif mn == 'movs' and o[1].startswith('#'):
                R[o[0]] = bv(imm(o[1]))
            elif mn == 'muls':
                R[o[0]] = g(o[1]) * g(o[2])
            elif mn in ('adds', 'subs'):
                if len(o) == 2:            # Rdn, #imm8
                    x, y = R[o[0]], bv(imm(o[1]))
                else:
                    x, y = R[o[1]], (bv(imm(o[2])) if o[2].startswith('#') else R[o[2]])
                R[o[0]] = add_any(x, y, sub=(mn == 'subs'))
                if not is_ptr(x) and not is_ptr(y):
                    res = R[o[0]]
                    if mn == 'adds':
                        C = z3.ULT(res, x)
                        V = z3.Extract(31, 31, ~(x ^ y) & (x ^ res)) == 1
                    else:
                        C = z3.UGE(x, y)
                        V = z3.Extract(31, 31, (x ^ y) & (x ^ res)) == 1
                    nzcv = (C, V)
            elif mn in ('asrs', 'lsls', 'lsrs') and len(o) == 3 and o[2].startswith('#'):
                a, n = g(o[1]), imm(o[2])
                if not (0 <= n <= 31 or (mn != 'lsls' and n == 32)): raise Unsupported('shift %d' % n)
                R[o[0]] = (a >> n) if mn == 'asrs' else (a << n) if mn == 'lsls' else z3.LShR(a, n)
            elif mn == 'uxth':
                R[o[0]] = g(o[1]) & bv(0xFFFF)
            elif mn in ('ands', 'orrs', 'eors') and len(o) == 2:
                a, b = g(o[0]), g(o[1])
                R[o[0]] = a & b if mn == 'ands' else a | b if mn == 'orrs' else a ^ b
            elif mn == 'cmp':
                a = g(o[0]); b = bv(imm(o[1])) if o[1].startswith('#') else g(o[1])
                res = a - b
                N = z3.Extract(31, 31, res) == 1
                Z = res == 0
                C = z3.UGE(a, b)
                V = z3.Extract(31, 31, (a ^ b) & (a ^ res)) == 1
                st.flags = (N, Z, C, V)
            elif mn in ('b', 'bgt', 'blt', 'bge', 'ble', 'beq', 'bne', 'bhs', 'blo', 'bhi', 'bls'):
                tgt = int(o[0].split()[0], 16)
                if mn == 'b':
                    if tgt <= st.pc: st.back += 1
                    st.pc = tgt
                    break_to_pending = True
                    pending.append(st); break
                if st.flags is None: raise Unsupported('branch without cmp at %x' % st.pc)
                N, Z, C, V = st.flags
                need = {'bgt': 'ZNV', 'blt': 'NV', 'bge': 'NV', 'ble': 'ZNV', 'beq': 'Z', 'bne': 'Z',
                        'bhs': 'C', 'blo': 'C', 'bhi': 'CZ', 'bls': 'CZ'}[mn]
                if ('C' in need and C is None) or ('V' in need and V is None):
                    raise Unsupported('%s reads a flag this model does not set at %x' % (mn, st.pc))
                c = {'bgt': z3.And(z3.Not(Z), N == V), 'blt': N != V, 'bge': N == V,
                     'ble': z3.Or(Z, N != V), 'beq': Z, 'bne': z3.Not(Z), 'bhs': C, 'blo': z3.Not(C),
                     'bhi': z3.And(C, z3.Not(Z)), 'bls': z3.Or(z3.Not(C), Z)}[mn]
                cs = z3.simplify(c)
                taken = st.copy(); taken.pc = tgt
                if tgt <= st.pc: taken.back += 1
                fall = st.copy(); fall.pc = nxt
                if z3.is_true(cs): pending.append(taken)
                elif z3.is_false(cs): pending.append(fall)
                else:
                    taken.cond = st.cond + (c,); fall.cond = st.cond + (z3.Not(c),)
                    pending.extend([taken, fall])
                break
            else:
                raise Unsupported('%x: %s %s' % (st.pc, mn, ops))
            if flagset:
                st.flags = None
                d = R.get(o[0])
                if d is not None and not is_ptr(d):
                    N = z3.Extract(31, 31, d) == 1; Z = d == 0
                    if mn in ('adds', 'subs') and nzcv is not None:
                        st.flags = (N, Z) + nzcv
                    else:
                        st.flags = (N, Z, None, None)   # C, V not modeled for these
            st.pc = nxt
    # merge the returned states
    out = done[0]
    for d in done[1:]:
        if set(d.mem) != set(out.mem): raise Unsupported('paths stored different addresses')
        c = z3.And(*d.cond) if d.cond else z3.BoolVal(True)
        out = State(out.pc, {k: (v if is_ptr(v) else (v if v.eq(d.R[k]) else z3.If(c, d.R[k], v)))
                             for k, v in out.R.items()},
                    {k: (v if is_ptr(v) else (v if v.eq(d.mem[k]) else z3.If(c, d.mem[k], v)))
                     for k, v in out.mem.items()}, (), None, 0)
    return out, len(done), steps
