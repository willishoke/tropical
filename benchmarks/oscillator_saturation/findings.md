# Oscillator saturation — findings

**Date:** 2026-08-29
**Host:** Apple M1 Pro (`MacBookPro18,1`, 16 Metal cores, macOS 26.3)
**Commit:** `4b436ce` (working tree dirty: this benchmark was untracked at run time)
**Row:** [`data/patch-fused-m1pro-20260829.jsonl`](data/patch-fused-m1pro-20260829.jsonl)
**Fixture:** `--fixture patch --mode fused` — N `FixedSinOsc` instances summed
into one `SoftClip`, 512-frame block at 44.1 kHz (11.610 ms deadline)

## TL;DR

For plain time-domain oscillators the GPU is **not** the win it is for fat
modal patches, and the two backends land within 4% of each other at the
threshold.

1. **JIT reaches 50% saturation between 1536 and 2048 oscillators**
   (44.51% → 59.92%). Interpolated estimate 1702 — an estimate, not a
   measured count.
2. **Metal reaches 50% between the same two counts** (42.24% → 57.63%),
   interpolated 1776.
3. **Metal only becomes cheaper at N=1536**, and then only by ~2 points. Below
   that the JIT is strictly better — by **16×** at N=8, where Metal is pinned
   at its submission floor.
4. **Metal refuses N=3072 outright**: `pipeline state failed: Compute function
   exceeds available stack space`. The ceiling is structural and sits between
   2048 and 3072 unrolled voices. It does not degrade; it fails to build.

## Results

```
   N    jit_sat   metal_sat    jit ns/osc   metal ns/osc
   1      0.02%      1.70%          2291         197458
   8      0.12%      1.94%          1688          28203
  32      0.50%      2.60%          1771           9438
 128      2.15%      7.05%          1948           6392
 256      4.61%      7.30%          2090           3309
 512      8.61%     14.19%          1951           3218
1024     19.19%     27.13%          2176           3076
1536     44.51%     42.24%          3365           3193   <- crossover
2048     59.92%     57.63%          3397           3267   <- both over 50%
3072     86.38%    REFUSED          3265              -
```

## The mechanism

The per-oscillator column explains the whole curve, and it is two flat regimes
plus one cliff:

- **JIT holds a flat ~1,950 ns/voice from N=8 through N=1024**, then cliffs to
  **~3,365 ns/voice at N=1536** — a 55% jump for a 1.5× count increase — and
  stays there. The work per voice has not changed; this is a code-size effect
  in a single straight-line kernel (at N=1536 the fused kernel is ~126k
  instructions).
- **Metal has no cliff.** It starts as pure overhead — 197 µs at N=1, which is
  the flat submission toll `../gpu_time_partition/findings.md` measured at
  ~200–330 µs — and amortizes monotonically to a flat **~3,100–3,270 ns/voice**
  from N=256 onward.

So the crossover at N=1536 is not the GPU getting better. It is the **JIT
getting worse at exactly that point**, falling from 1,950 to 3,365 ns/voice and
landing just above Metal's flat rate. Above the cliff the two backends cost the
same per voice (3,265–3,397 vs 3,193–3,267); the JIT's real advantage is
entirely the sub-cliff regime, where it is 1.6× cheaper per voice *and* pays no
fixed toll.

**Why the GPU does not pull ahead.** The Metal kernel parallelizes over the 512
samples in a render tile, not over voices — one thread per time index, each
thread evaluating the whole kernel. Adding a voice therefore adds work to
*every* thread. There is no voice-axis parallelism to exploit, so the GPU
scales in N at roughly the CPU's per-voice rate and can only win on the fixed
overhead it has already paid down. This is the structural difference from the
modal case, where `gpu_time_partition` saw up to 7.8×: there the per-block work
was large enough that the toll amortized while the CPU had far more arithmetic
to chew through.

## Compile time

Emission is near-identical between backends (MSL adds ~0.2 s) but superlinear
in N at the top of the sweep:

```
   N     emit     ratio per 2x
 512     5.3s
1024    12.7s     2.4x
2048    49.9s     3.9x
3072   142.4s     (2.9x for 1.5x)
```

At N=3072 that is 142 s to compile ~252k instructions. The cost is LLVM's
SelectionDAG list scheduler: a 15-second stack sample of a slow compile put
**99.9% of samples in `ScheduleDAGRRList::Schedule()` and 83% in
`RegReductionPriorityQueue::pop()`**, which selects by linear scan over the
ready list. This is not optimization — `TROPICAL_JIT_OPT_LEVEL=O0` is the
*slowest* setting measured (283 s vs 213 s at O2 on the graph fixture), because
O0 hands the scheduler a larger DAG.

`--mode microkernel` was measured against `fused` at every count from 64 to
1024 and is **identical** (4.1 s vs 4.2 s at N=1024). It is not a lever here.

## An unresolved anomaly worth filing separately

The `--fixture graph` route (playground `source` nodes) is **~100× worse per
instruction** than the patch route, and the gap is not explained by scale:

| route | instructions (1 block) | JIT compile |
|---|---|---|
| patch, N=1024 | ~88,000 | 4.1 s |
| graph, N=256 | 32,295 | 212 s |

The graph route detonates on 2.7× *fewer* instructions. Ruled out by
measurement: MSL emission (JIT and Metal within 2%), opt level (O0 slowest),
`TROPICAL_STAGE0` (no effect), instruction count (1.5×), basic-block count and
max block size (both routes are one ~identical giant block), register pressure
(a flat 15 simultaneously-live values at N=64/128/256 — the emitter already
accumulates the sum incrementally), and instruction mix (no exotic opcodes, no
calls). The profile locates it in the scheduler; what provokes the scheduler on
one lowering and not the other is **not yet identified**.

Repro: `--fixture graph --counts 256 --backend jit` versus
`--fixture patch --counts 1024 --backend jit`.

## Caveats

- Metal's `process_ns` is a synchronous worker-tile wait, so it measures GPU
  **throughput**, not callback cost. With the worker pipelined the callback
  only copies from a ready tile; `gpu_time_partition` measured that path at
  ~4–6 µs. These numbers say "the GPU can sustain this block rate", not "the
  callback costs this much".
- The GPU was shared with the desktop compositor (WindowServer at ~36% CPU
  during some runs). A reserved-GPU measurement would be needed before any
  production claim, as `gpu_time_partition` also notes.
- Voices here are **unrolled**, not banked. A `bankSum` region would make the
  emitted IR O(1) in N and remove the compile-time question entirely; the
  playground `source` and patch `FixedSinOsc` paths simply never construct one,
  because banking is chosen at authoring time by the arrow modal builder
  (`Ir/BanksFlag.lean`). A banked oscillator family is the obvious follow-on.
- Single host, single row. No other hardware is inferred.
