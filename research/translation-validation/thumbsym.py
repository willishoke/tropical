"""Symbolic executor for the Thumb-1 (ARMv6-M) subset in a straight-line
tropical kernel object, read from `llvm-objdump -d` text (so the
disassembler and my reading of the ARMv6-M ARM semantics are trusted).

Pointers are kept abstract: ('ptr', base, byte_off); the caller binds the
argument registers r0..r3 to bases and supplies the input words.  Paths
fork at conditional branches and are merged with If at return.
Refuses any mnemonic outside the subset.
"""
import re
import z3

class Unsupported(Exception):
    pass

def parse(objdump_text, fn):
    lines = objdump_text.split('\n')
    insts, words, on = {}, {}, False
    for l in lines:
        if re.match(r'^[0-9a-f]+ <%s>:$' % re.escape(fn), l):
            on = True; continue
        if on and re.match(r'^[0-9a-f]+ <', l) or (on and l.startswith('Disassembly')):
            break
        if not on: continue
        m = re.match(r'^\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s*\.word\s+0x([0-9a-f]+)', l)
        if m:
            words[int(m.group(1), 16)] = int(m.group(3), 16); continue
        m = re.match(r'^\s+([0-9a-f]+):\s+[0-9a-f]{4}(?: [0-9a-f]{4})?\s+(\S+)\s*(.*?)\s*$', l)
        if m:
            insts[int(m.group(1), 16)] = (m.group(2), m.group(3))
    return insts, words

def bv(n): return z3.BitVecVal(n & 0xFFFFFFFF, 32)

def imm(t):
    return int(t.lstrip('#'), 0)

def run(objdump_text, fn, regs0, loads, start=0, stop=None, stack0=None):
    """regs0: initial register values (BV or ('ptr',base,off));
    loads: {(base, off): BV32}.  Returns (stores, final regs) merged."""
    insts, words = parse(objdump_text, fn)
    results = []   # (pathcond, stores, regs)
    work = [(start, dict(regs0), dict(stack0 or {}), z3.BoolVal(True), None, {})]
    while work:
        pc, R, stack, cond, flags, stores = work.pop()
        while True:
            if pc == stop:
                results.append((cond, stores, R)); break
            if pc not in insts: raise Unsupported('pc %x' % pc)
            mn, ops = insts[pc]
            size = 2  # all Thumb-1 16-bit here; bl would be 4 (refused below)
            o = [x.strip() for x in re.split(r',(?![^\[{]*[\]}])', ops.split('@')[0]) if x.strip()]
            nxt = pc + size
            def g(r):
                v = R[r]
                if isinstance(v, tuple): raise Unsupported('arith on pointer %s at %x' % (r, pc))
                return v
            if mn == 'push':
                for r in re.findall(r'r\d+|lr', o[0]): stack[r] = R[r]
            elif mn == 'pop':
                rs = re.findall(r'r\d+|pc', o[0])
                for r in rs:
                    if r != 'pc': R[r] = stack[r]
                if 'pc' in rs:
                    results.append((cond, stores, R)); break
            elif mn == 'ldr':
                m = re.match(r'\[pc, #(0x[0-9a-f]+|\d+)\]', o[1])
                if m:
                    addr = ((pc + 4) & ~3) + int(m.group(1), 0)
                    R[o[0]] = bv(words[addr])
                else:
                    m = re.match(r'\[(r\d+)(?:, #(0x[0-9a-f]+|\d+))?\]', o[1])
                    p = R[m.group(1)]; off = int(m.group(2), 0) if m.group(2) else 0
                    if not isinstance(p, tuple): raise Unsupported('load via non-pointer')
                    key = (p[1], p[2] + off)
                    if key not in loads: raise Unsupported('load %s' % (key,))
                    R[o[0]] = loads[key]
            elif mn == 'str':
                m = re.match(r'\[(r\d+)(?:, #(0x[0-9a-f]+|\d+))?\]', o[1])
                p = R[m.group(1)]; off = int(m.group(2), 0) if m.group(2) else 0
                stores = dict(stores); stores[(p[1], p[2] + off)] = g(o[0])
            elif mn == 'mov':           # T1 MOV (register): no flags
                R[o[0]] = R[o[1]]
            elif mn == 'movs' and o[1].startswith('#'):
                R[o[0]] = bv(imm(o[1]))
            elif mn == 'muls':          # MULS Rd, Rn, Rd  (low 32 bits)
                R[o[0]] = g(o[1]) * g(o[2])
            elif mn in ('adds', 'subs') and len(o) == 3 and not o[2].startswith('#'):
                a, b = g(o[1]), g(o[2])
                R[o[0]] = a + b if mn == 'adds' else a - b
            elif mn in ('asrs', 'lsls', 'lsrs') and o[2].startswith('#'):
                a, n = g(o[1]), imm(o[2])
                if not (0 <= n <= 31 or (mn != 'lsls' and n == 32)): raise Unsupported('shift %d' % n)
                R[o[0]] = (a >> n) if mn == 'asrs' else (a << n) if mn == 'lsls' else z3.LShR(a, n)
            elif mn == 'uxth':
                R[o[0]] = g(o[1]) & bv(0xFFFF)
            elif mn == 'ands' and len(o) == 2:   # ANDS Rdn, Rm
                R[o[0]] = g(o[0]) & g(o[1])
            elif mn == 'cmp' and len(o) == 2 and not o[1].startswith('#'):
                a, b = g(o[0]), g(o[1])
                res = a - b
                N = z3.Extract(31, 31, res) == 1
                Z = res == 0
                V = z3.Extract(31, 31, (a ^ b) & (a ^ res)) == 1
                flags = (N, Z, V)
            elif mn in ('bgt', 'blt', 'bge', 'ble', 'beq', 'bne'):
                if flags is None: raise Unsupported('branch without cmp')
                N, Z, V = flags
                c = {'bgt': z3.And(z3.Not(Z), N == V), 'blt': N != V, 'bge': N == V,
                     'ble': z3.Or(Z, N != V), 'beq': Z, 'bne': z3.Not(Z)}[mn]
                tgt = int(o[0].split()[0], 16)
                work.append((tgt, dict(R), dict(stack), z3.And(cond, c), flags, dict(stores)))
                cond = z3.And(cond, z3.Not(c))
            else:
                raise Unsupported('%x: %s %s' % (pc, mn, ops))
            # any other flag-setting op clobbers the cmp flags (only cmp feeds branches)
            if mn in ('movs', 'muls', 'adds', 'subs', 'asrs', 'lsls', 'lsrs', 'ands'):
                flags = None
            pc = nxt
    return results

def merge(results, key):
    out = None
    for cond, stores, _ in results:
        v = stores[key]
        out = v if out is None else z3.If(cond, v, out)
    return out
