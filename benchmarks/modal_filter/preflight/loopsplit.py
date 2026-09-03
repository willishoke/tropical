#!/usr/bin/env python3
"""Post-O2, which ops live INSIDE the sample loop vs its preheader?

If LLVM's LICM already hoisted the scalar pole math (the 1607 intercept) to
the preheader, the remaining ~640% wall clock is entirely the per-mode
composition, and the question sharpens to: what pins THAT inside the loop?
"""
import re, sys
from pathlib import Path

LABEL = re.compile(r"^([A-Za-z_.][\w.]*):")
OPS = ("fdiv", "fmul", "fadd", "llvm.sqrt", "llvm.fabs",
       "load ", "store ", "= phi ")

def blocks(path: Path):
    cur, out = ("__entry__", []), []
    for line in path.read_text().splitlines():
        m = LABEL.match(line)
        if m:
            out.append(cur)
            cur = (m.group(1), [])
        else:
            cur[1].append(line)
    out.append(cur)
    return out

def analyze(path: Path):
    bs = blocks(path)
    idx = {name: i for i, (name, _) in enumerate(bs)}
    # outer loop = the backward branch with the largest block span
    best = None
    for i, (name, lines) in enumerate(bs):
        for l in lines:
            for t in re.findall(r"label %([\w.]+)", l):
                j = idx.get(t)
                if j is not None and j <= i and (best is None or (i - j) > (best[1] - best[0])):
                    best = (j, i)
    lo, hi = best
    def histo(rng):
        h = {op: 0 for op in OPS}
        n = 0
        for name, lines in rng:
            for l in lines:
                if re.match(r"^\s+\S", l) and not l.strip().startswith(";"):
                    n += 1
                for op in OPS:
                    if op in l:
                        h[op] += 1
        return n, h
    pre_n, pre_h = histo(bs[:lo])
    in_n, in_h = histo(bs[lo:hi + 1])
    print(f"  outer loop: blocks[{lo}..{hi}] = {bs[lo][0]} .. {bs[hi][0]}")
    for tag, n, h in (("preheader+entry", pre_n, pre_h), ("inside loop", in_n, in_h)):
        compact = {k.strip(" ="): v for k, v in h.items() if v}
        print(f"  {tag:>16}: {n:>6} lines  {compact}")

if __name__ == "__main__":
    for tag in sys.argv[1:]:
        d = Path(__file__).resolve().parent / ".work" / tag / "post"
        f = max(d.glob("postpipeline_*.ll"), key=lambda q: q.stat().st_size)
        print(f"== {tag} ==")
        analyze(f)
