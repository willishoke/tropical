#!/usr/bin/env python3
"""Build and check a tropical kernel as relocatable objects for a no-FPU MCU.

`diffcli compile-object` emits the objects; this script is the gate and the
report around them:

  * libcalls — `llvm-nm -u` on the thumbv6m object must list nothing (no
    __aeabi_*, __mul*, __div*, soft-float helpers). Exit 1 otherwise.
  * cycles — a static count of the hot loop from `llvm-objdump`, with the
    Cortex-M0+ table (ALU and MULS 1, loads/stores 2, PUSH/POP 1+N, taken
    branch 2, BL 3). Upper bound: every instruction in the loop body once,
    every conditional branch taken. Lower bound: conditional branches not
    taken, forward-skipped instructions still counted (they are clamp arms of
    one or two instructions), so the true path lies between the two.
  * sizes, sections, stack depth (.stack_sizes), ARM build attributes.

    tools/embedded_object.py --diffcli lean/.lake/build/bin/diffcli \
        --llvm-bin /opt/homebrew/opt/llvm@22/bin --patch P --out-dir D \
        [--frames N] [--rate R] [--check-only]
"""
import argparse, json, os, re, subprocess, sys

TRIPLES = [("host", ""), ("thumbv6m", "thumbv6m-none-eabi"),
           ("wasm32", "wasm32-unknown-unknown")]
LIBCALL = re.compile(r"__aeabi_|__mul|__div|__mod|__udiv|__add|__sub|__fix|__float|__extend|__trunc|__cmp|__eq|__ne|__lt|__le|__gt|__ge|__un|memcpy|memset")

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

def m0_cycles(mn, ops):
    mn = mn.split(".")[0]
    if mn in ("push", "pop", "ldm", "stm", "ldmia", "stmia"):
        regs = ops[ops.find("{") + 1: ops.find("}")]
        n = 0
        for part in regs.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-"); n += int(b.strip()[1:]) - int(a.strip()[1:]) + 1
            elif part:
                n += 1
        return 1 + n + (2 if (mn == "pop" and "pc" in regs) else 0)
    if mn.startswith("ldr") or mn.startswith("str"):
        return 2
    if mn == "bl":
        return 3
    if mn in ("bx", "blx"):
        return 2
    if mn == "b":
        return 2
    if re.fullmatch(r"b(eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)", mn):
        return None  # conditional: 2 taken, 1 not
    return 1

def disasm(llvm_bin, obj, fn="tropical_kernel"):
    out = run([f"{llvm_bin}/llvm-objdump", "-d", "--no-show-raw-insn", obj])
    insns, cur = [], None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(\w+)>:", line)
        if m:
            cur = m.group(1); continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m and cur == fn and not m.group(2).startswith("."):
            insns.append((int(m.group(1), 16), m.group(2), m.group(3)))
    return insns

def loop_report(insns):
    # The sample loop: the backward branch whose target is the lowest address.
    back = []
    for addr, mn, ops in insns:
        m = re.match(r"0x([0-9a-f]+)", ops)
        if mn.startswith("b") and mn not in ("bl", "bx", "blx") and m:
            tgt = int(m.group(1), 16)
            if tgt <= addr:
                back.append((tgt, addr))
    lo = min(t for t, _ in back); hi = max(a for _, a in back)
    body = [(a, mn, ops) for a, mn, ops in insns if lo <= a <= hi]
    up = low = 0
    for _, mn, ops in body:
        c = m0_cycles(mn, ops)
        if c is None:
            up += 2; low += 1
        else:
            up += c; low += c
    outside = sum((m0_cycles(mn, ops) or 2) for a, mn, ops in insns if not (lo <= a <= hi))
    muls = sum(1 for _, mn, _ in body if mn.startswith("mul"))
    calls = [ops for _, mn, ops in body if mn == "bl"]
    return {"loop_range": [hex(lo), hex(hi)], "loop_instructions": len(body),
            "muls_per_frame": muls, "calls_in_loop": len(calls),
            "frame_cycles_lower": low, "frame_cycles_upper": up,
            "outside_loop_cycles_upper": outside}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffcli", required=True)
    ap.add_argument("--llvm-bin", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--out-format", default="Q1.15")
    ap.add_argument("--check-only", action="store_true",
                    help="thumbv6m object only; fail on any undefined symbol")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.patch))[0]
    objs = {}
    for tag, triple in (TRIPLES[1:2] if a.check_only else TRIPLES):
        o = os.path.join(a.out_dir, f"{name}-{tag}.o")
        cmd = [a.diffcli, "compile-object", a.patch, "--out", o, "--frames", str(a.frames),
               "--rate", str(a.rate), "--out-elem", "i32", "--out-format", a.out_format]
        if triple:
            cmd += ["--triple", triple]
        if tag == "thumbv6m":
            cmd += ["--cpu", "cortex-m0plus"]
        run(cmd)
        objs[tag] = o
    thumb = objs["thumbv6m"]
    undef = [l.split()[-1] for l in run([f"{a.llvm_bin}/llvm-nm", "-u", thumb]).splitlines() if l.strip()]
    if a.check_only:
        if undef:
            print(f"FAIL embedded-object: {thumb} references {undef}")
            return 1
        print(f"PASS embedded-object: thumbv6m object has no undefined symbols")
        return 0
    insns = disasm(a.llvm_bin, thumb)
    rep = loop_report(insns)
    N = a.frames
    rep["frames"] = N
    rep["call_cycles_upper"] = rep["outside_loop_cycles_upper"] + N * rep["frame_cycles_upper"]
    rep["call_cycles_lower"] = rep["outside_loop_cycles_upper"] + N * rep["frame_cycles_lower"]
    rep["undefined_symbols"] = undef
    rep["libcalls"] = [u for u in undef if LIBCALL.search(u)]
    sizes = {}
    for line in run([f"{a.llvm_bin}/llvm-size", "-A", thumb]).splitlines()[2:]:
        p = line.split()
        if len(p) >= 2 and p[0].startswith("."):
            sizes[p[0]] = int(p[1])
    rep["sections"] = sizes
    rep["flash_bytes"] = sizes.get(".text", 0) + sizes.get(".rodata", 0) + sizes.get(".data", 0)
    rep["static_ram_bytes"] = sizes.get(".data", 0) + sizes.get(".bss", 0)
    st = run([f"{a.llvm_bin}/llvm-readobj", "--stack-sizes", thumb])
    rep["stack_bytes"] = {m.group(1): int(m.group(2), 0) for m in
                          re.finditer(r"Functions: \[(\w+)\]\s+Size: (0x[0-9A-Fa-f]+|\d+)", st)}
    attrs = run([f"{a.llvm_bin}/llvm-readelf", "-A", thumb])
    rep["arm_attributes"] = {m.group(1).strip(): m.group(2).strip() for m in
                             re.finditer(r"TagName: (\w+)\s*\n\s*Description: ([^\n]+)", attrs)}
    # Fold the measured facts into the thumbv6m manifest's object block.
    mpath = thumb + ".manifest.json"
    with open(mpath) as f:
        man = json.load(f)
    man["object"]["stack_bytes"] = rep["stack_bytes"].get("tropical_kernel")
    man["object"]["sections"] = sizes
    man["object"]["m0plus_cycles"] = {
        "frame_lower": rep["frame_cycles_lower"], "frame_upper": rep["frame_cycles_upper"],
        "call_upper": rep["call_cycles_upper"], "frames": N}
    man["object"]["float_abi"] = "soft (no FP instructions; Tag_FP_arch absent)"
    with open(mpath, "w") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(a.out_dir, "cycles-thumbv6m.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    return 1 if undef else 0

if __name__ == "__main__":
    sys.exit(main())
