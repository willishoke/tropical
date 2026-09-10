#!/usr/bin/env python3
"""Modal source through a resonant lowpass — the composition benchmark.

The bare-oscillator table (../faust_comparison) measures per-voice synthesis;
nothing in it exercises COMPOSITION, which is where tropical's design is
supposed to pay. This fixture measures it directly, per the design handoff:

  metric 1  marginal cost of the filter: cost(source→filter) − cost(source),
            per system — cancels each system's source realization and
            isolates what adding a filter costs.
  metric 2  total cost per point, alongside — marginal alone is exactly the
            framing that would make this a rigged benchmark.
  metric 3  quality under cutoff sweep, each system against ITS OWN
            finer-grained reference:
              tropical — the same graph with the knob written every SAMPLE
                (--buffer 1) vs every BLOCK; prices coefficient-update
                quantization (settled knobs are coefficient-time).
              Faust    — the same dsp at 8x the rate, compared at common
                instants (the sweep runs on absolute time); prices the svf's
                finite-rate coefficient tracking AND its discretization, so
                the static row is the baseline to read the sweep rows against.

Fairness: per-voice filters on both sides; the source mirrors resonatorBank's
partial law (see dsp/gen_modal.py) but by each system's own paradigm
(recurrent vs closed form) — stated, and cancelled by metric 1; `fi.svf`
because a direct-form biquad under modulation is a strawman.
"""
import json, math, os, statistics, struct, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAUST = "/opt/homebrew/bin/faust"
CXX = "/opt/homebrew/opt/llvm/bin/clang++"
BENCH_ARCH = ROOT / "benchmarks/faust_comparison/bench_arch.cpp"
RENDER_ARCH = HERE / "render_arch.cpp"
GEN = HERE / "dsp/gen_modal.py"
DIFFCLI = ROOT / "lean/.lake/build/bin/diffcli"
RUNTIME_BENCH = ROOT / "build/tropical_runtime_bench"
WORK = HERE / ".work"
RATE = 44100
BUFFER = 512
SCHEMA = "modal_filter_comparison_1"


def clean_env(**over):
    e = {k: v for k, v in os.environ.items() if not k.startswith("TROPICAL_")}
    e.update({"TROPICAL_STAGE0": "1", "TROPICAL_JIT_OPT_LEVEL": "O2",
              "TROPICAL_KERNEL_CACHE_DISABLE": "1"})
    e.update(over)
    return e


def sh(args, env=None, timeout=1800):
    p = subprocess.run([str(a) for a in args], capture_output=True,
                       timeout=timeout, env=env, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError(f"{args[0]}: {p.stderr.decode()[:300]}")
    return p.stdout


# ─────────────────────────────── tropical ────────────────────────────────

def graph(p: int, m: int, filtered: bool) -> dict:
    nodes, tips = [], []
    for v in range(p):
        f0 = 220.0 * (1.0 + 0.03 * v)
        nodes.append({"id": f"src{v}", "kind": "resonator",
                      "params": {"freq": f0, "decay": 4, "partials": m}})
        tip = f"src{v}"
        if filtered:
            nodes.append({"id": f"flt{v}", "kind": "filter",
                          "params": {"cutoff": 800, "resonance": 0.5},
                          "in": {"in": [tip]}})
            tip = f"flt{v}"
        tips.append(tip)
    if p == 1:
        nodes.append({"id": "out", "kind": "out", "in": {"in": [tips[0]]}})
    else:
        nodes.append({"id": "mix", "kind": "mix", "in": {"in": tips}})
        nodes.append({"id": "out", "kind": "out", "in": {"in": ["mix"]}})
    return {"nodes": nodes, "out": "out"}


def tropical_point(p: int, m: int, filtered: bool, blocks=200, warmup=16) -> float:
    tag = f"t_p{p}m{m}{'f' if filtered else 'b'}"
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    src = d / "graph.json"
    src.write_text(json.dumps(graph(p, m, filtered), separators=(",", ":")))
    env = clean_env(TROPICAL_STAGE0_DUMP=str(d))
    sh([DIFFCLI, "render-graph", src, "--frames", 0, "--buffer", BUFFER], env=env)
    args = [RUNTIME_BENCH, "--ir", d / "audio.ll", "--manifest", d / "manifest.json",
            "--buffer", BUFFER, "--blocks", blocks, "--warmup", warmup, "--rate", RATE]
    coeff = d / "coeff.ll"
    if coeff.exists() and coeff.stat().st_size:
        args += ["--coeff", coeff]
    time.sleep(0.8)
    probe = json.loads(sh(args, env=clean_env()))
    return statistics.median(probe["process_ns"])


def tropical_sweep(m: int, rate_hz: float, buffer: int, seconds: float) -> bytes:
    d = WORK / f"tsweep_m{m}"
    d.mkdir(parents=True, exist_ok=True)
    src = d / "graph.json"
    src.write_text(json.dumps(graph(1, m, True), separators=(",", ":")))
    frames = int(seconds * RATE) // buffer
    return sh([DIFFCLI, "render-sweep", src, "--param=flt0.cutoff",
               "--center=800", "--octaves=2", f"--rate-hz={rate_hz}",
               "--frames", frames, "--buffer", buffer], env=clean_env())


# ──────────────────────────────── faust ──────────────────────────────────

def faust_build(dsp: Path, exe: Path, arch: Path):
    cpp = exe.with_suffix(".cpp")
    sh([FAUST, "-double", "-t", "1800", "-a", arch, "-o", cpp, dsp])
    sh([CXX, "-O3", "-std=c++17", "-DFAUSTFLOAT=double", "-o", exe, cpp])


def faust_point(p: int, m: int, filtered: bool, blocks=200, warmup=16) -> float:
    tag = f"f_p{p}m{m}{'f' if filtered else 'b'}"
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    dsp, exe = d / "v.dsp", d / "v"
    if not exe.exists():
        sh([sys.executable, GEN, "filtered" if filtered else "bare", dsp, p, m])
        faust_build(dsp, exe, BENCH_ARCH)
    time.sleep(0.8)
    out = json.loads(sh([exe, "--buffer", BUFFER, "--blocks", blocks,
                         "--warmup", warmup, "--rate", RATE, "--label", tag]))
    return out["median_ns"]


def faust_sweep(m: int, rate_hz: float, rate: int, seconds: float) -> bytes:
    d = WORK / f"fsweep_m{m}_r{rate_hz}"
    d.mkdir(parents=True, exist_ok=True)
    dsp, exe = d / "v.dsp", d / "v"
    if not exe.exists():
        sh([sys.executable, GEN, "swept", dsp, m, rate_hz, 2.0])
        faust_build(dsp, exe, RENDER_ARCH)
    frames = int(seconds * rate) // BUFFER
    return sh([exe, "--buffer", BUFFER, "--frames", frames, "--rate", rate])


# ──────────────────────────────── metrics ────────────────────────────────

def doubles(raw: bytes):
    return struct.unpack(f"<{len(raw) // 8}d", raw)


def snr_db(ref, dut) -> float:
    n = min(len(ref), len(dut))
    sig = sum(ref[i] * ref[i] for i in range(n))
    err = sum((ref[i] - dut[i]) ** 2 for i in range(n))
    if err == 0.0:
        return float("inf")
    return 10.0 * math.log10(sig / err) if sig > 0 else float("-inf")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []

    print("== metrics 1+2: marginal and total cost (ns / 512-frame block) ==")
    print(f"{'P':>3} {'M':>4} {'sys':>9} {'bare':>10} {'filtered':>10} "
          f"{'marginal':>9} {'marg%':>7}")
    for p in (1, 8):
        for m in (16, 64):
            for name, fn in (("tropical", tropical_point), ("faust", faust_point)):
                b = fn(p, m, False)
                f = fn(p, m, True)
                marg = f - b
                print(f"{p:>3} {m:>4} {name:>9} {b:>10.0f} {f:>10.0f} "
                      f"{marg:>9.0f} {100.0 * marg / b:>6.1f}%", flush=True)
                rows.append({"metric": "cost", "system": name, "P": p, "M": m,
                             "bare_ns": b, "filtered_ns": f, "marginal_ns": marg})

    print("\n== metric 3: sweep quality, SNR dB vs own fine-grained reference ==")
    print("   (tropical ref: knob written per SAMPLE; faust ref: 4x rate (ma.SR clamps at 192k).")
    print("    faust's static row is its discretization baseline — read the")
    print("    sweep rows against it, not against 0.)")
    seconds, m = 2.0, 64
    print(f"{'rate Hz':>8} {'tropical b512':>13} {'tropical b64':>13} {'faust 1x/8x':>12}")
    for rate_hz in (0.0, 2.0, 20.0):
        t_ref = doubles(tropical_sweep(m, rate_hz, 1, seconds))
        t512 = doubles(tropical_sweep(m, rate_hz, 512, seconds))
        t64 = doubles(tropical_sweep(m, rate_hz, 64, seconds))
        # 4x, NOT 8x: Faust's `ma.SR` clamps to [1, 192000], so a 352.8 kHz
        # render silently computes absolute time against 192 kHz and produces
        # a different signal entirely (measured: fully uncorrelated, -4.5 dB
        # at STATIC cutoff, M- and Q-independent — the tell was
        # cos(...) == 0.5000 exactly). 176.4 kHz stays under the clamp.
        f1 = doubles(faust_sweep(m, rate_hz, RATE, seconds))
        f8 = doubles(faust_sweep(m, rate_hz, 4 * RATE, seconds))
        f8d = f8[::4]
        s512, s64, sf = snr_db(t_ref, t512), snr_db(t_ref, t64), snr_db(f8d, f1)
        print(f"{rate_hz:>8.1f} {s512:>13.1f} {s64:>13.1f} {sf:>12.1f}", flush=True)
        rows.append({"metric": "sweep", "M": m, "rate_hz": rate_hz,
                     "tropical_b512_snr_db": s512, "tropical_b64_snr_db": s64,
                     "faust_vs_4x_snr_db": sf})

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "data" / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.strip()
    out.write_text("\n".join(
        [json.dumps({"schema": SCHEMA, "commit": commit, "recorded": tag})]
        + [json.dumps(r) for r in rows]) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
