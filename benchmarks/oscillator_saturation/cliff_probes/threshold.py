#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys, statistics, time, shutil, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slotknee import run as emit, env
ROOT=str(pathlib.Path(__file__).resolve().parents[3])
SP=os.path.dirname(os.path.abspath(__file__))
WORK=str(pathlib.Path(__file__).resolve().parents[1] / ".work" / "cliff")
BENCH=f"{ROOT}/build/tropical_runtime_bench"; L="/opt/homebrew/opt/llvm/bin"

def bench(ir, man, tag):
    cache=f"{WORK}/th/{tag}"; shutil.rmtree(cache, ignore_errors=True); os.makedirs(cache, exist_ok=True)
    e={k:v for k,v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update({"TROPICAL_STAGE0":"1","TROPICAL_JIT_OPT_LEVEL":"O2",
              "TROPICAL_KERNEL_CACHE_DISABLE":"0","TROPICAL_KERNEL_CACHE_ROOT":cache})
    time.sleep(1.0)
    p=json.loads(subprocess.run([BENCH,"--ir",ir,"--manifest",man,"--buffer","512",
        "--blocks","300","--warmup","32","--rate","44100"],cwd=ROOT,env=e,
        capture_output=True,check=True).stdout)
    o=max(glob.glob(f"{cache}/*/*.o"), key=os.path.getsize)
    tb=int(re.search(r"__text\s+(\d+)", subprocess.run([f"{L}/llvm-size","--format=sysv",o],
        capture_output=True,text=True).stdout).group(1))
    return statistics.median(p["process_ns"]), tb

stages=int(sys.argv[1]); counts=[int(c) for c in sys.argv[2].split(",")]
print(f"{'N':>5} {'slots':>6} {'stock_text':>11} {'stock ns/v':>11} {'reb_text':>9} {'reb ns/v':>9}")
for n in counts:
    r=emit(n, stages, measure=False)
    d=f"{WORK}/slotknee/s{stages}/n{n}"
    subprocess.run(["python3", f"{SP}/rebase3.py", f"{d}/audio.ll", f"{d}/reb.ll"],
                   check=True, capture_output=True)
    a,ta=bench(f"{d}/audio.ll", f"{d}/manifest.json", f"s{stages}n{n}s")
    b,tb=bench(f"{d}/reb.ll",   f"{d}/manifest.json", f"s{stages}n{n}r")
    print(f"{n:>5} {r['slot_count']:>6} {ta:>11} {a/n:>11.1f} {tb:>9} {b/n:>9.1f}", flush=True)
