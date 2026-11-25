#!/usr/bin/env python3
"""The M0+ cycle bounds of tools/embedded_object.py.

The bounds come from a shortest/longest-path pass over the loop-free control-
flow graph (`bounds`). These tests hold it to the path enumeration it
replaced, kept below verbatim as `enumerated_loop_report`, on every graph
small enough to enumerate, and to closed forms past that:

  * synthetic kernels — random loop-free frame bodies (diamonds, skips,
    branches to the next instruction, early exits) report exactly what the
    enumeration reports, the wait-state estimate included;
  * twenty diamonds in a row — 2^20 paths, which the enumeration refuses,
    and a range known in closed form;
  * an inner loop, and a cycle in the epilogue, are still refused;
  * the thumbv6m objects of the fixture patches, when `TROPICAL_DIFFCLI` and
    `TROPICAL_LLVM_BIN` name a diffcli and an LLVM bin directory (`make
    validate` and CI set both): the report equals the enumeration's on each
    LFO, and on a chain of selects, which the enumeration refuses, the frame
    range equals a top-down recursion's.

    TROPICAL_DIFFCLI=lean/.lake/build/bin/diffcli \\
    TROPICAL_LLVM_BIN=/opt/homebrew/opt/llvm@22/bin python3 tools/test_embedded_object.py
"""
import importlib.util
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("embedded_object", HERE / "embedded_object.py")
EO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EO)

# The M0+ objects of these fixtures are compared with the enumeration.
FIXTURES = ["tests/fixtures/embedded_lfo.json", "tests/fixtures/embedded_lfo_knob.json"]
# Fourteen selects on one LFO's output into a second LFO's frequency: Thumb-1
# has no conditional move, so each select is a branch, and the frame loop
# has 2^16 paths. The enumeration refuses it; a top-down recursion, a second
# way to the same extremes, checks the bounds.
BRANCHY = "tests/fixtures/embedded_select_chain.json"


def topdown(cfg, start, stop, region):
    """{end: (lo, hi, n)} by memoized recursion from `start`, independent of
    `bounds`' topological order. Loop-free graphs only."""
    memo = {}
    def go(k):
        if k in memo:
            return memo[k]
        acc = {}
        for s, c, _ in cfg.succ(k):
            sub = {s: (0, 0, 1)} if EO.ends(s, stop, region) else go(s)
            for e, (lo, hi, n) in sub.items():
                a = acc.get(e)
                v = (lo + c, hi + c, n)
                acc[e] = v if a is None else (min(a[0], v[0]), max(a[1], v[1]), a[2] + n)
        memo[k] = acc
        return acc
    return go(start)


def enumerated_loop_report(insns, frames, wait):
    """`loop_report` as it was before the path DP (e00f76e), unchanged: the
    oracle. It enumerates every path and so refuses past `paths`' limit."""
    paths, Cfg, wait_estimate, TIMING_MODEL = EO.paths, EO.Cfg, EO.wait_estimate, EO.TIMING_MODEL
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


def assemble(lines):
    """Thumb-1 lines, as llvm-objdump prints them, into `Insn`s at 2 bytes
    each. `label:` names the next instruction; a branch operand `@label`
    becomes its address."""
    labels, body = {}, []
    for line in lines:
        if line.endswith(":"):
            labels[line[:-1]] = 2 * len(body)
        else:
            body.append(line)
    insns = []
    for k, line in enumerate(body):
        mn, _, ops = line.partition(" ")
        if ops.startswith("@"):
            t = labels[ops[1:]]
            ops = f"0x{t:x} <tropical_kernel+0x{t:x}>"
        insns.append(EO.Insn(2 * k, mn, ops))
    return insns


def without_note(rep):
    return {k: v for k, v in rep.items() if k != "cycle_bounds"}


def diamonds(n):
    """A frame loop whose body is `n` diamonds in a row: each a compare and
    a conditional branch over one arm into the other. Per diamond the
    fall-through arm costs cmp 1 + bge not taken 1 + movs 1 + b 2 = 5 and the
    taken arm cmp 1 + bge taken 2 + movs 1 = 4."""
    lines = ["push {r4, lr}", "movs r3, #16", "head:"]
    for d in range(n):
        lines += ["cmp r0, r1", f"bge @else{d}", "movs r2, #1", f"b @join{d}",
                  f"else{d}:", "movs r2, #2", f"join{d}:"]
    lines += ["subs r3, #1", "bne @head", "pop {r4, pc}"]
    return assemble(lines)


ALU = ["adds r0, r1", "lsls r2, r2, #3", "muls r0, r1, r0", "ldr r1, [r2, #4]",
       "str r0, [sp, #8]", "ldr r4, [pc, #12]", "asrs r1, r1, #2", "eors r0, r2"]
CONDS = ["beq", "bne", "bge", "blt", "bhi", "bls", "bmi", "bpl"]


def random_kernel(rng, body_len, early_exits):
    """A prologue, a frame loop of `body_len` instructions with forward
    branches only (so loop-free but for the back edge), optional early exits
    into the epilogue, and a branchy epilogue."""
    lines = ["push {r4, r5, r6, r7, lr}", "sub sp, #12"]
    # Prologue: forward branches within itself; it falls into the head.
    pro = rng.randint(0, 4)
    for k in range(pro):
        lines.append(f"p{k}:")
        if rng.random() < 0.4:
            lines.append(f"{rng.choice(CONDS)} @p{rng.randint(k + 1, pro)}")
        else:
            lines.append(rng.choice(ALU))
    lines.append(f"p{pro}:")
    lines.append("head:")
    for k in range(body_len):
        lines.append(f"l{k}:")
        r = rng.random()
        if r < 0.25:
            lines.append(f"{rng.choice(CONDS)} @l{rng.randint(k + 1, body_len)}")
        elif r < 0.32:
            lines.append(f"b @l{rng.randint(k + 1, body_len)}")
        elif r < 0.37:
            lines.append(f"{rng.choice(CONDS)} @l{k + 1}")  # both edges to one place
        elif r < 0.37 + early_exits:
            lines.append(f"{rng.choice(CONDS)} @e{rng.randint(0, 3)}")
        else:
            lines.append(rng.choice(ALU))
    lines += [f"l{body_len}:", "subs r3, #1", "bne @head"]
    for k in range(4):
        lines.append(f"e{k}:")
        if rng.random() < 0.5:
            lines.append(f"{rng.choice(CONDS)} @e{rng.randint(k + 1, 4)}")
        else:
            lines.append(rng.choice(ALU))
    lines += ["e4:", "add sp, #12", "pop {r4, r5, r6, r7, pc}"]
    return assemble(lines)


class Synthetic(unittest.TestCase):
    def test_random_kernels_match_the_enumeration(self):
        rng = random.Random(256)
        compared = 0
        for trial in range(400):
            insns = random_kernel(rng, rng.randint(1, 30), rng.choice([0, 0.05, 0.1]))
            try:
                want = enumerated_loop_report(insns, 16, 0)
            except SystemExit as e:
                self.assertIn("more than 4096 paths", str(e))
                continue
            got = EO.loop_report(insns, 16, 0)
            self.assertEqual(without_note(got), want, f"trial {trial}")
            self.assertEqual(without_note(EO.loop_report(insns, 7, 2)),
                             enumerated_loop_report(insns, 7, 2), f"trial {trial}, W=2")
            compared += 1
        self.assertGreater(compared, 300)

    def test_twenty_diamonds_in_closed_form(self):
        n, frames = 20, 16
        insns = diamonds(n)
        with self.assertRaises(SystemExit) as e:
            enumerated_loop_report(insns, frames, 0)
        self.assertIn("more than 4096 paths", str(e.exception))
        rep = EO.loop_report(insns, frames, 0)
        # Around: the diamonds, subs 1, bne taken 2. Leaving: bne not taken 1,
        # then pop {r4, pc} = 1 + 2 + 2. Entry: push {r4, lr} 3, movs 1.
        self.assertEqual(rep["frame_paths"], 2 ** n)
        self.assertEqual(rep["frame_cycles_lower"], 4 * n + 3)
        self.assertEqual(rep["frame_cycles_upper"], 5 * n + 3)
        self.assertEqual((rep["entry_cycles"], rep["entry_cycles_upper"]), (4, 4))
        self.assertEqual((rep["exit_cycles"], rep["exit_cycles_upper"]), (4, 4))
        self.assertEqual(rep["call_cycles_lower"], 4 + (frames - 1) * (4 * n + 3) + (4 * n + 2 + 5))
        self.assertEqual(rep["call_cycles_upper"], 4 + (frames - 1) * (5 * n + 3) + (5 * n + 2 + 5))
        # The wait-state estimate walks paths, so it refuses here.
        with self.assertRaises(SystemExit) as e:
            EO.loop_report(insns, frames, 2)
        self.assertIn("--wait-states frame: more than 4096 paths", str(e.exception))

    def test_small_diamonds_match_the_enumeration(self):
        for n in range(0, 12):
            insns = diamonds(n)
            self.assertEqual(without_note(EO.loop_report(insns, 16, 0)),
                             enumerated_loop_report(insns, 16, 0))

    def test_an_inner_loop_is_refused(self):
        insns = assemble(["push {r4, lr}", "head:", "movs r2, #4", "inner:", "subs r2, #1",
                          "bne @inner", "subs r3, #1", "bne @head", "pop {r4, pc}"])
        with self.assertRaises(SystemExit) as e:
            EO.loop_report(insns, 16, 0)
        self.assertIn("frame: the graph from 0x2 has a cycle through", str(e.exception))
        self.assertIn("unbounded", str(e.exception))

    def test_a_cycle_past_the_frame_loop_is_refused(self):
        # bounds() on its own: a loop-free run into a two-instruction cycle.
        insns = assemble(["movs r0, #1", "top:", "adds r0, #1", "bne @top", "bx lr"])
        cfg = EO.Cfg(insns)
        with self.assertRaises(SystemExit) as e:
            EO.bounds(cfg, 0, set(), what="epilogue")
        self.assertIn("epilogue: the graph from 0x0 has a cycle through 0x", str(e.exception))
        # A cycle back to the start itself.
        with self.assertRaises(SystemExit):
            EO.bounds(cfg, 1, set(), what="epilogue")

    def test_branch_edges_carry_their_own_cost(self):
        # bne to the next instruction: two edges, 2 (taken) and 1, one target.
        insns = assemble(["push {r4, lr}", "head:", "bne @n", "n:", "subs r3, #1",
                          "bne @head", "pop {r4, pc}"])
        rep = EO.loop_report(insns, 16, 0)
        self.assertEqual(rep["frame_paths"], 2)
        self.assertEqual((rep["frame_cycles_lower"], rep["frame_cycles_upper"]), (4, 5))


@unittest.skipUnless(os.environ.get("TROPICAL_DIFFCLI") and os.environ.get("TROPICAL_LLVM_BIN"),
                     "set TROPICAL_DIFFCLI and TROPICAL_LLVM_BIN to compare compiled objects")
class Objects(unittest.TestCase):
    def test_fixture_objects_match_the_enumeration(self):
        diffcli = os.environ["TROPICAL_DIFFCLI"]
        llvm = os.environ["TROPICAL_LLVM_BIN"]
        core = EO.CORES["m0plus"]
        with tempfile.TemporaryDirectory() as d:
            for patch in FIXTURES:
                with self.subTest(patch=patch):
                    o = os.path.join(d, Path(patch).stem + ".o")
                    subprocess.run([diffcli, "compile-object", str(ROOT / patch), "--out", o,
                                    "--frames", "16", "--rate", "48000", "--triple",
                                    core["triple"], "--cpu", core["cpu"]],
                                   check=True, capture_output=True, text=True)
                    insns = EO.disasm(llvm, o)
                    got = EO.loop_report(insns, 16, 0)
                    self.assertEqual(without_note(got), enumerated_loop_report(insns, 16, 0))
                    self.assertEqual(without_note(EO.loop_report(insns, 16, 2)),
                                     enumerated_loop_report(insns, 16, 2))
                    self.assertGreaterEqual(got["frame_cycles_upper"], got["frame_cycles_lower"])
                    print(f"{patch}: {got['frame_paths']} frame paths, "
                          f"frame {got['frame_cycles_lower']}-{got['frame_cycles_upper']}, "
                          f"call {got['call_cycles_lower']}-{got['call_cycles_upper']}")

    def test_a_branchy_object_past_the_enumeration(self):
        diffcli = os.environ["TROPICAL_DIFFCLI"]
        llvm = os.environ["TROPICAL_LLVM_BIN"]
        core = EO.CORES["m0plus"]
        with tempfile.TemporaryDirectory() as d:
            o = os.path.join(d, "branchy.o")
            subprocess.run([diffcli, "compile-object", str(ROOT / BRANCHY), "--out", o,
                            "--frames", "16", "--rate", "48000", "--triple", core["triple"],
                            "--cpu", core["cpu"]], check=True, capture_output=True, text=True)
            insns = EO.disasm(llvm, o)
        with self.assertRaises(SystemExit) as e:
            enumerated_loop_report(insns, 16, 0)
        self.assertIn("frame: more than 4096 paths", str(e.exception))
        rep = EO.loop_report(insns, 16, 0)
        self.assertGreater(rep["frame_paths"], 4096)
        cfg = EO.Cfg(insns)
        lo, hi = (int(a, 16) for a in rep["loop_range"])
        inside = lambda k: k is not None and lo <= cfg.insns[k].addr <= hi
        head = cfg.at[int(rep["loop_head"], 16)]
        f_lo, f_hi, n = topdown(cfg, head, {head}, inside)[head]
        self.assertEqual((rep["frame_cycles_lower"], rep["frame_cycles_upper"], rep["frame_paths"]),
                         (f_lo, f_hi, n))
        print(f"{BRANCHY}: {rep['frame_paths']} frame paths, "
              f"frame {rep['frame_cycles_lower']}-{rep['frame_cycles_upper']}, "
              f"call {rep['call_cycles_lower']}-{rep['call_cycles_upper']}")


if __name__ == "__main__":
    unittest.main()
