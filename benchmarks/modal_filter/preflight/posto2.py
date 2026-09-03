#!/usr/bin/env python3
"""What does -O2 leave in the per-sample loop?  The preflight counted the
EMITTER's IR; the JIT then runs its O2 pipeline (LICM included) before
codegen.  If LICM were rescuing the tau-free composition, the wall clock would
not show ~640% -- but "LICM fails" is a claim about the post-pipeline module,
so measure that module."""
import json, os, re, statistics, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marginal
from marginal import ROOT, BENCH, env, ir_loop_body

def post_o2(tag: str):
    d = marginal.WORK / tag
    dump = d / "post"
    dump.mkdir(parents=True, exist_ok=True)
    for f in dump.glob("postpipeline_*.ll"):
        f.unlink()
    args = [str(BENCH), "--ir", str(d / "audio.ll"),
            "--manifest", str(d / "manifest.json"),
            "--buffer", "512", "--blocks", "20", "--warmup", "4", "--rate", "44100"]
    coeff = d / "coeff.ll"
    if coeff.exists() and coeff.stat().st_size:
        args += ["--coeff", str(coeff)]
    e = env(TROPICAL_KERNEL_CACHE_ROOT=str(marginal.WORK / "cache-post" / tag),
            TROPICAL_KERNEL_CACHE_DISABLE="1",
            TROPICAL_JIT_TRACE="1", TROPICAL_JIT_TRACE_DUMP=str(dump))
    subprocess.run(args, cwd=str(ROOT), env=e, capture_output=True, check=True)
    lls = sorted(dump.glob("postpipeline_*.ll"), key=lambda q: q.stat().st_size)
    audio = lls[-1]                     # audio module dwarfs the coeff module
    body = ir_loop_body(audio)
    text = audio.read_text()
    m = re.search(r"^loop_body.*?(?=^loop_end|\Z)", text, re.S | re.M)
    ops = {}
    for op in ("fdiv", "fmul", "fadd", "llvm.sqrt", "llvm.fabs", "load", "store"):
        ops[op] = len(re.findall(re.escape(op), m.group(0))) if m else -1
    return body, ops

if __name__ == "__main__":
    print(f"{'tag':>12} {'pre-O2':>8} {'post-O2':>8}  ops")
    for m_ in (16, 64):
        for stage in (None, "filter"):
            tag = f"m{m_}{stage or 'plain'}"
            pre = ir_loop_body(marginal.WORK / tag / "audio.ll")
            body, ops = post_o2(tag)
            print(f"{tag:>12} {pre:>8} {body:>8}  {ops}", flush=True)
