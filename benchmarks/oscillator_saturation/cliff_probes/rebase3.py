#!/usr/bin/env python3
"""Page-rebase %slots GEPs behind an opaque (zero-instruction) identity asm,
so InstCombine cannot reassociate the page base away."""
import re, sys
PAGE = 4096  # doubles
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
pat = re.compile(r"getelementptr inbounds double, ptr %slots, i64 (\d+)")
pages = set()
def sub(m):
    idx = int(m.group(1)); k = idx // PAGE
    if k == 0: return m.group(0)
    pages.add(k)
    return f"getelementptr inbounds double, ptr %slots_p{k}, i64 {idx - k*PAGE}"
out = pat.sub(sub, text)
if pages:
    d = ["  %slots_int = ptrtoint ptr %slots to i64\n"]
    for k in sorted(pages):
        d.append(f"  %sp{k}a = add i64 %slots_int, {k*PAGE*8}\n")
        d.append(f'  %sp{k}b = call i64 asm "", "=r,0"(i64 %sp{k}a)\n')
        d.append(f"  %slots_p{k} = inttoptr i64 %sp{k}b to ptr\n")
    out = out.replace("entry:\n", "entry:\n" + "".join(d), 1)
open(dst, "w").write(out)
print(f"pages={sorted(pages)} rewrote={len(pat.findall(text))}")
