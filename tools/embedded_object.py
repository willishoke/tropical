#!/usr/bin/env python3
"""Build and check a tropical kernel as relocatable objects for a microcontroller.

`diffcli compile-object` emits the objects; this script is the gate and the
report around them. Two cores (`--core`):

  * `m0plus` (default) — Cortex-M0+, `thumbv6m-none-eabi`, no FPU: an integer
    program at the default float format. Sizes, libcalls and cycles.
  * `m4f` — Cortex-M4F, `thumbv7em-none-eabihf`, FPv4-SP-D16 (binary32 only):
    the artifact is compiled at `--float f32`. Sizes and libcalls only. The
    cycle table below is the M0+'s; for any other core the script counts no
    cycles and says so, and `--wait-states` on one exits non-zero rather than
    print a number that does not hold.

What it checks and reports:

  * libcalls — `llvm-nm -u` on the thumb object must list nothing (no
    __aeabi_*, __mul*, __div*, soft-float helpers). Exit 1 otherwise.
    `compile-object` already refuses such an object, naming the op; a
    refusal is reported as a failure with its message.
  * cycles — a static count from `llvm-objdump` on the Cortex-M0+ timing
    table (ARM DDI 0484, zero wait states), assuming the single-cycle
    multiplier (the M0+'s 32-cycle iterative multiplier is a build option;
    the report states the assumption as `timing_model`): ALU and MULS 1, LDR/STR 2,
    PUSH/POP 1+N, POP with PC 3+N (N counting PC), B 2, B<cond> 2 taken and
    1 not taken, BL 3, BX 2. The sample loop is found from its back edges and
    entered at the block the prologue branches to. Every path once around the
    loop is enumerated, so the per-frame range is the cheapest and the
    dearest path, data-dependent branches (the output clamp) included; a call
    is entry path + (N - 1) frames + the last frame, which leaves through the
    exit branch into the epilogue. The enumeration is exhaustive or the
    script exits non-zero: a path that revisits an instruction (an inner
    loop) or more paths than the limit means no bound, so none is printed.
    Calibrated against an ARMv6-M emulator on
    the TRM table: the #252 LFO object measured 20 + 114/frame + 10 (worst
    block 1902), the int32 LFO 17 + 86/frame + 10 (worst 1451); both lie in
    the computed ranges.
  * wait states (`--wait-states W`) — an estimate, not a bound, for code in
    flash: each flash access (a 32-bit instruction fetch, or a literal-pool
    load) costs W more cycles, and an instruction fetch hides behind the
    execution of the word before it when that takes long enough. The low end
    assumes that overlap, the high end none.
  * sizes, sections, stack depth (.stack_sizes), ARM build attributes.

    tools/embedded_object.py --diffcli lean/.lake/build/bin/diffcli \\
        --llvm-bin /opt/homebrew/opt/llvm@22/bin --patch P --out-dir D \\
        [--core m0plus|m4f] [--frames N] [--rate R] [--wait-states W] [--check-only]
    tools/embedded_object.py --llvm-bin ... --object O.o [--frames N]

With `--object` the core is read from the object's own ARM build attributes
(`Tag_CPU_arch`), not from a flag.

Port element types and fixed-point formats come from the program, not from
flags (`design/typed-kernel.md`, "Typed ports").
"""
import argparse, json, os, re, subprocess, sys

# Per core: the object's tag, triple, cpu and float format (None: the
# compiler's default), and the Tag_CPU_arch its object carries.
CORES = {
    "m0plus": {"tag": "thumbv6m", "triple": "thumbv6m-none-eabi", "cpu": "cortex-m0plus",
               "float": None, "cpu_arch": "ARM v6S-M"},
    "m4f": {"tag": "thumbv7em", "triple": "thumbv7em-none-eabihf", "cpu": "cortex-m4",
            "float": "f32", "cpu_arch": "ARM v7E-M"},
}
# The objects built per patch besides the core's own: the host and wasm32,
# at the core's float format, for comparison runs.
OTHER_TRIPLES = [("host", ""), ("wasm32", "wasm32-unknown-unknown")]
LIBCALL = re.compile(r"__aeabi_|__mul|__div|__mod|__udiv|__add|__sub|__fix|__float|__extend|__trunc|__cmp|__eq|__ne|__lt|__le|__gt|__ge|__un|memcpy|memset")
# The cycle table the counts assume, stated in every report: the Cortex-M0+
# multiplier is a build-time option, and the counts hold only for the
# single-cycle one.
TIMING_MODEL = ("Cortex-M0+ (ARM DDI 0484), zero wait states, single-cycle "
                "multiplier: MULS = 1 cycle. A part built with the small iterative "
                "multiplier takes 32 cycles per MULS; add 31 x muls_per_frame per frame.")
COND = re.compile(r"b(eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)$")


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def reglist_len(ops):
    regs = ops[ops.find("{") + 1: ops.find("}")]
    n = 0
    for part in regs.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            n += int(b.strip()[1:]) - int(a.strip()[1:]) + 1
        elif part:
            n += 1
    return n, "pc" in regs


def cost(mn, ops):
    """Zero-wait cycles of one instruction; a conditional branch's cost is
    decided by the path (2 taken, 1 not), so it returns None."""
    mn = mn.split(".")[0]
    if mn in ("push", "pop", "ldm", "stm", "ldmia", "stmia"):
        n, pc = reglist_len(ops)
        return 1 + n + (2 if (mn == "pop" and pc) else 0)
    if mn.startswith("ldr") or mn.startswith("str"):
        return 2
    if mn == "bl":
        return 3
    if mn in ("bx", "blx", "b"):
        return 2
    if COND.match(mn):
        return None
    return 1


class Insn:
    def __init__(self, addr, mn, ops):
        self.addr, self.mn, self.ops = addr, mn.split(".")[0], ops
        m = re.match(r"0x([0-9a-f]+)", ops)
        self.target = int(m.group(1), 16) if (self.mn.startswith("b") and m
                                              and self.mn not in ("bl", "bx", "blx")) else None
        self.cond = bool(COND.match(self.mn))
        self.returns = (self.mn == "pop" and "pc" in ops) or (self.mn == "bx" and "lr" in ops)
        self.literal = self.mn.startswith("ldr") and "[pc" in ops
        self.size = 2  # Thumb-1; BL is 4 (handled by the address of the next insn)


def disasm(llvm_bin, obj, fn="tropical_kernel"):
    out = run([f"{llvm_bin}/llvm-objdump", "-d", "--no-show-raw-insn", obj])
    insns, cur = [], None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(\w+)>:", line)
        if m:
            cur = m.group(1); continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m and cur == fn and not m.group(2).startswith("."):
            insns.append(Insn(int(m.group(1), 16), m.group(2), m.group(3)))
    return insns


class Cfg:
    def __init__(self, insns):
        self.insns = insns
        self.at = {i.addr: k for k, i in enumerate(insns)}

    def nxt(self, k):
        return k + 1 if k + 1 < len(self.insns) else None

    def succ(self, k):
        """(successor index, cycles of insn k on that edge, taken?) pairs."""
        i = self.insns[k]
        if i.returns:
            return [(None, cost(i.mn, i.ops), False)]
        if i.target is not None and not i.cond:
            return [(self.at[i.target], 2, True)]
        if i.cond:
            return [(self.at[i.target], 2, True), (self.nxt(k), 1, False)]
        return [(self.nxt(k), cost(i.mn, i.ops), False)]


def paths(cfg, start, stop, region=None, limit=4096, what="path"):
    """Every path from index `start` to a member of `stop` (or a return), as
    lists of (index, cycles, taken). Paths leaving `region` end there with
    the leaving edge as their last step.

    The enumeration is exhaustive or it fails: a path that comes back to an
    instruction it already ran (a loop other than the one `stop` closes)
    has no bounded cost, and past `limit` paths the set is not known to be
    complete. Either way the range computed from it would not be a bound, so
    the script exits non-zero instead of reporting one."""
    out, stack = [], [(start, [], frozenset({start}))]
    while stack:
        k, path, seen = stack.pop()
        for s, c, taken in cfg.succ(k):
            step = path + [(k, c, taken)]
            if s is None or s in stop or (region is not None and not region(s)):
                out.append((step, s))
                if len(out) > limit:
                    raise SystemExit(
                        f"embedded_object: {what}: more than {limit} paths from "
                        f"{hex(cfg.insns[start].addr)}; the cycle range would not be a bound")
            elif s in seen:
                raise SystemExit(
                    f"embedded_object: {what}: the path from {hex(cfg.insns[start].addr)} "
                    f"revisits {hex(cfg.insns[s].addr)} (an inner loop); its cost is "
                    f"unbounded, so the cycle range would not be a bound")
            else:
                stack.append((s, step, seen | {s}))
    return out


def loop_report(insns, frames, wait):
    cfg = Cfg(insns)
    back = [(i.target, i.addr) for i in insns if i.target is not None and i.target <= i.addr]
    lo = min(t for t, _ in back); hi = max(a for _, a in back)
    inside = lambda k: k is not None and lo <= cfg.insns[k].addr <= hi
    # The header: the loop block the prologue enters.
    entry_paths = paths(cfg, 0, set(), region=lambda k: not inside(k), what="prologue")
    heads = {s for p, s in entry_paths if s is not None and inside(s)}
    if len(heads) != 1:
        raise SystemExit(f"embedded_object: expected one loop entry, found {sorted(heads)}")
    head = heads.pop()
    entry = [p for p, s in entry_paths if s == head]
    around, leave = [], []
    loop_paths = paths(cfg, head, {head}, region=inside, what="frame")
    for p, s in loop_paths:
        (around if s == head else leave).append((p, s))
    around = [p for p, _ in around]
    if not around or not leave:
        raise SystemExit("embedded_object: the loop has no back path or no exit")
    total = lambda p: sum(c for _, c, _ in p)
    # Epilogue: from the exit target to the return, its cheapest and its
    # dearest path.
    def epilogue(s):
        if s is None:
            return 0, 0
        costs = [total(p) for p, _ in paths(cfg, s, set(), what="epilogue")]
        return min(costs), max(costs)
    frame = sorted(total(p) for p in around)
    exits_lo, exits_hi = [], []
    for p, s in leave:
        e_lo, e_hi = epilogue(s)
        exits_lo.append(total(p) + e_lo)
        exits_hi.append(total(p) + e_hi)
    leave = [p for p, _ in leave]
    entry_c = [total(p) for p in entry]
    # The exit cost is what the last frame adds over a frame on the same data
    # path: the exit branch taken instead of not (+1) and the epilogue.
    exit_extra = min(exits_lo) - frame[0]
    exit_extra_hi = max(exits_hi) - frame[-1]
    lo_call = min(entry_c) + (frames - 1) * frame[0] + min(exits_lo)
    hi_call = max(entry_c) + (frames - 1) * frame[-1] + max(exits_hi)
    body = [i for i in insns if lo <= i.addr <= hi]
    rep = {"loop_range": [hex(lo), hex(hi)], "loop_head": hex(cfg.insns[head].addr),
           "loop_instructions": len(body),
           "muls_per_frame": sum(1 for i in body if i.mn.startswith("mul")),
           "calls_in_loop": sum(1 for i in body if i.mn == "bl"),
           "frame_paths": len(around),
           "entry_cycles": min(entry_c), "entry_cycles_upper": max(entry_c),
           "frame_cycles_lower": frame[0],
           "frame_cycles_upper": frame[-1], "exit_cycles": exit_extra,
           "exit_cycles_upper": exit_extra_hi,
           "frames": frames, "call_cycles_lower": lo_call, "call_cycles_upper": hi_call,
           "timing_model": TIMING_MODEL}
    if wait:
        rep["wait_states"] = wait
        rep.update(wait_estimate(cfg, entry, around, leave, epilogue, frames, wait))
    return rep


def flash_cost(cfg, path, wait):
    """Extra cycles of `path` at `wait` flash wait states: (low, high). One
    32-bit fetch per word the path enters; a literal load is a flash data
    access. High: every access stalls `wait`. Low: a fetch stalls only by what
    the previous word's instructions (on this path) do not cover, and a
    taken branch's refetch always stalls."""
    lit = sum(1 for k, _, _ in path if cfg.insns[k].literal)
    words, cover, prev_word, prev_taken = [], {}, None, True
    for k, c, taken in path:
        w = cfg.insns[k].addr // 4
        if w != prev_word:
            words.append((w, prev_taken))
            prev_word = w
        cover[w] = cover.get(w, 0) + c
        prev_taken = taken
    high = wait * (len(words) + lit)
    low = wait * lit
    for idx, (w, after_branch) in enumerate(words):
        if after_branch or idx == 0:
            low += wait
        else:
            low += max(0, 1 + wait - cover[words[idx - 1][0]])
    return low, high


def wait_estimate(cfg, entry, around, leave, epilogue, frames, wait):
    fl = lambda ps: [flash_cost(cfg, p, wait) for p in ps]
    e = fl(entry); f = fl(around); x = fl(leave)
    total = lambda p: sum(c for _, c, _ in p)
    base_f = [total(p) for p in around]
    lo_f = min(b + l for b, (l, _) in zip(base_f, f))
    hi_f = max(b + h for b, (_, h) in zip(base_f, f))
    lo_e = min(total(p) + l for p, (l, _) in zip(entry, e))
    hi_e = max(total(p) + h for p, (_, h) in zip(entry, e))
    lo_x = min(total(p) + l for p, (l, _) in zip(leave, x)) + 1 + wait
    hi_x = max(total(p) + h for p, (_, h) in zip(leave, x)) + 2 + 2 * wait
    return {"wait_frame_cycles_estimate": [lo_f, hi_f],
            "wait_call_cycles_estimate": [lo_e + (frames - 1) * lo_f + lo_x,
                                          hi_e + (frames - 1) * hi_f + hi_x]}


M0PLUS_ARCH = CORES["m0plus"]["cpu_arch"]


def cycles_refusal(attrs):
    """Why this object gets no cycle count, or None for a Cortex-M0+ one."""
    arch = attrs.get("CPU_arch")
    if arch == M0PLUS_ARCH:
        return None
    cpu = attrs.get("CPU_name", "an unnamed core")
    return (f"no cycle count: the timing table is the Cortex-M0+'s ({M0PLUS_ARCH}); "
            f"this object is for {cpu} ({arch or 'no Tag_CPU_arch'}), whose pipeline, "
            f"multiplier and FPU timings it does not model")


def object_report(a, thumb, frames):
    undef = [l.split()[-1] for l in run([f"{a.llvm_bin}/llvm-nm", "-u", thumb]).splitlines() if l.strip()]
    attrs_text = run([f"{a.llvm_bin}/llvm-readelf", "-A", thumb])
    attrs = {m.group(1).strip(): m.group(2).strip() for m in
             re.finditer(r"TagName: (\w+)\s*\n\s*(?:Value|Description): ([^\n]+)", attrs_text)}
    refusal = cycles_refusal(attrs)
    if refusal is None:
        rep = loop_report(disasm(a.llvm_bin, thumb), frames, a.wait_states)
    else:
        if a.wait_states:
            raise SystemExit(f"embedded_object: --wait-states: {refusal}")
        print(f"embedded_object: {thumb}: {refusal}", file=sys.stderr)
        rep = {"cycles": None, "cycles_refused": refusal, "frames": frames}
    rep["undefined_symbols"] = undef
    rep["libcalls"] = [u for u in undef if LIBCALL.search(u)]
    sizes = {}
    for line in run([f"{a.llvm_bin}/llvm-size", "-A", thumb]).splitlines()[2:]:
        p = line.split()
        if len(p) >= 2 and p[0].startswith("."):
            sizes[p[0]] = int(p[1])
    rep["sections"] = sizes
    def total(*prefixes):
        return sum(v for k, v in sizes.items() if any(k == p or k.startswith(p + ".") for p in prefixes))
    rep["flash_bytes_kernel"] = sizes.get(".text.tropical_kernel", 0) + total(".rodata", ".data")
    rep["flash_bytes_all"] = total(".text", ".rodata", ".data")
    rep["static_ram_bytes"] = total(".data", ".bss")
    st = run([f"{a.llvm_bin}/llvm-readobj", "--stack-sizes", thumb])
    rep["stack_bytes"] = {m.group(1): int(m.group(2), 0) for m in
                          re.finditer(r"Functions: \[(\w+)\]\s+Size: (0x[0-9A-Fa-f]+|\d+)", st)}
    rep["arm_attributes"] = attrs
    return rep, undef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffcli")
    ap.add_argument("--llvm-bin", required=True)
    ap.add_argument("--patch", action="append", default=[])
    ap.add_argument("--object", help="report on an existing thumb object; no compile")
    ap.add_argument("--core", choices=sorted(CORES), default="m0plus",
                    help="the microcontroller to build for (default m0plus)")
    ap.add_argument("--out-dir")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--wait-states", type=int, default=0)
    ap.add_argument("--check-only", action="store_true",
                    help="the core's objects only; fail on any undefined symbol")
    a = ap.parse_args()
    if a.object:
        rep, undef = object_report(a, a.object, a.frames)
        print(json.dumps(rep, indent=2))
        return 1 if undef else 0
    if not (a.diffcli and a.patch and a.out_dir):
        ap.error("--diffcli, --patch and --out-dir are required without --object")
    core = CORES[a.core]
    if a.wait_states and a.core != "m0plus":
        raise SystemExit(f"embedded_object: --wait-states: the timing table is the "
                         f"Cortex-M0+'s; --core {a.core} gets no cycle count")
    os.makedirs(a.out_dir, exist_ok=True)
    status = 0
    for patch in a.patch:
        name = os.path.splitext(os.path.basename(patch))[0]
        objs = {}
        builds = [(core["tag"], core["triple"], core["cpu"])]
        if not a.check_only:
            builds += [(tag, triple, None) for tag, triple in OTHER_TRIPLES]
        failed = False
        for tag, triple, cpu in builds:
            o = os.path.join(a.out_dir, f"{name}-{tag}.o")
            cmd = [a.diffcli, "compile-object", patch, "--out", o, "--frames", str(a.frames),
                   "--rate", str(a.rate)]
            if triple:
                cmd += ["--triple", triple]
            if cpu:
                cmd += ["--cpu", cpu]
            if core["float"]:
                cmd += ["--float", core["float"]]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                # compile-object refuses an object that needs a libcall, or a
                # float format the target cannot realize, and says why.
                print(f"FAIL embedded-object: {patch} for {tag}: {r.stderr.strip()}")
                failed = True
                break
            objs[tag] = o
        if failed:
            status = 1
            continue
        thumb = objs[core["tag"]]
        if a.check_only:
            undef = [l.split()[-1] for l in run([f"{a.llvm_bin}/llvm-nm", "-u", thumb]).splitlines() if l.strip()]
            if undef:
                print(f"FAIL embedded-object: {thumb} references {undef}")
                status = 1
            else:
                print(f"PASS embedded-object: {thumb} has no undefined symbols")
            continue
        rep, undef = object_report(a, thumb, a.frames)
        mpath = thumb + ".manifest.json"
        with open(mpath) as f:
            man = json.load(f)
        man["object"]["stack_bytes"] = rep["stack_bytes"].get("tropical_kernel")
        man["object"]["sections"] = rep["sections"]
        if a.core == "m0plus":
            man["object"]["m0plus_cycles"] = {
                "entry": rep["entry_cycles"], "frame_lower": rep["frame_cycles_lower"],
                "frame_upper": rep["frame_cycles_upper"], "exit": rep["exit_cycles"],
                "call_lower": rep["call_cycles_lower"], "call_upper": rep["call_cycles_upper"],
                "frames": a.frames, "wait_states": 0, "timing_model": TIMING_MODEL}
            man["object"]["float_abi"] = "soft (no FP instructions; Tag_FP_arch absent)"
        with open(mpath, "w") as f:
            json.dump(man, f, indent=2)
        with open(os.path.join(a.out_dir, f"{name}-report-{core['tag']}.json"), "w") as f:
            json.dump(rep, f, indent=2)
        print(json.dumps(rep, indent=2))
        if undef:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
