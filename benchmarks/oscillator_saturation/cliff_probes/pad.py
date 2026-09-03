#!/usr/bin/env python3
"""Pad the per-sample loop body with independent, short-lived integer adds
that the optimizer cannot remove: same work, more executed code."""
import sys
src, dst, p = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = open(src).read()
anchor = "  %s_next = add i64 %s, 1\n"
assert t.count(anchor) == 1
lines = []
for i in range(p):
    lines.append(f"  %pad{i} = add i64 %current_idx, {i}\n")
    lines.append(f'  call void asm sideeffect "", "r"(i64 %pad{i})\n')
open(dst, "w").write(t.replace(anchor, "".join(lines) + anchor))
print(f"padded {p} units")
