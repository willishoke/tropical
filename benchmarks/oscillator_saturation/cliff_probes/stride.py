#!/usr/bin/env python3
"""Relocate every %slots access by a constant stride: identical instruction
stream, identical semantics, larger touched footprint. Page bases hidden
behind an opaque identity so immediates still fit."""
import json, re, sys
src_ll, src_man, dst_ll, dst_man, stride = sys.argv[1:6]
stride = int(stride)
text = open(src_ll).read()
pat = re.compile(r"getelementptr inbounds double, ptr %slots, i64 (\d+)")
pages = set()
def sub(m):
    idx = int(m.group(1)) * stride
    k = idx // 4096
    if k == 0:
        return f"getelementptr inbounds double, ptr %slots, i64 {idx}"
    pages.add(k)
    return f"getelementptr inbounds double, ptr %slots_p{k}, i64 {idx - k*4096}"
out = pat.sub(sub, text)
d = ["  %slots_int = ptrtoint ptr %slots to i64\n"]
for k in sorted(pages):
    d.append(f"  %sp{k}a = add i64 %slots_int, {k*4096*8}\n")
    d.append(f'  %sp{k}b = call i64 asm "", "=r,0"(i64 %sp{k}a)\n')
    d.append(f"  %slots_p{k} = inttoptr i64 %sp{k}b to ptr\n")
out = out.replace("entry:\n", "entry:\n" + "".join(d), 1)
open(dst_ll, "w").write(out)
man = json.loads(open(src_man).read())
old = man["slot_count"]; new = (old - 1) * stride + 1
man["slot_count"] = new
man["slot_defaults"] = (man["slot_defaults"] + [0.0] * new)[:new]
man["slot_names"] = (man["slot_names"] + [f"pad{i}" for i in range(new)])[:new]
open(dst_man, "w").write(json.dumps(man))
print(f"stride={stride} slots {old}->{new} ({new*8} B) pages={sorted(pages)}")
