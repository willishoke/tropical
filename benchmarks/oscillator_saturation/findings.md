# Oscillator saturation — findings

**Date:** 2026-08-29
**Host:** Apple M1 Pro (`MacBookPro18,1`, 16 Metal cores, macOS 26.3)
**Commit:** `4b436ce` (working tree dirty: this benchmark was untracked at run time)
**Row:** [`data/patch-fused-m1pro-20260829.jsonl`](data/patch-fused-m1pro-20260829.jsonl)
**Fixture:** `--fixture patch --mode fused` — N `FixedSinOsc` instances summed
into one `SoftClip`, 512-frame block at 44.1 kHz (11.610 ms deadline)

> **CORRECTION (same day, after review).** The headline below is measured at
> the default Metal render tile of 512 frames, and at that tile the dispatch is
> **one threadgroup**, which occupies **one of the M1 Pro's 16 GPU cores**. The
> comparison is therefore between a full CPU core and ~1/16 of the GPU. Widening
> the tile flips the result: at N=512 voices Metal goes from 6.201 ns/voice-sample
> (1.5x *slower* than the JIT's 4.091) to **1.290 at an 8192-frame tile — 3.2x
> faster**. The 512->1024 step is exactly 2.0x, as an occupancy explanation
> predicts. See "The dispatch is one threadgroup" below. Read every Metal number
> in this row as *"Metal at Rgpu=512"*, not as *"Metal"*.

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
   at its submission floor. *(At the default 512-frame tile only — see the
   correction above; with a wider tile Metal leads from far lower counts.)*
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

## The dispatch is one threadgroup

The non-cooperative kernel is emitted as one thread per sample
(`uint s [[thread_position_in_grid]]`, `Ir/EmitMsl.lean`) and dispatched
(`engine/metal/MetalKernel.mm:213-217`) as:

```objc
threads = min(pso.maxTotalThreadsPerThreadgroup, frames);
dispatchThreads(MTLSize(frames,1,1), threadsPerThreadgroup: MTLSize(threads,1,1));
```

With `frames = 512` and a threadgroup width up to 512, the whole grid is a
single threadgroup — and a threadgroup runs on a single GPU core. Fifteen of
sixteen cores are idle for the entire dispatch.

Measured at N=512 voices, sweeping `TROPICAL_METAL_RENDER_TILE_FRAMES`:

```
  tile   ns/voice-sample   vs jit (4.091)
   512        6.201          1.5x slower     <- the default, and this row's data
  1024        3.251          1.26x faster    <- exactly 2.0x the 512 result
  2048        1.818          2.25x faster
  4096        1.847          (plateau)
  8192        1.290          3.2x faster
```

The clean 2.0x at the first doubling is what an occupancy-bound dispatch
predicts and is hard to explain otherwise.

Two consequences:

1. **The "GPU is not a win for oscillators" conclusion does not survive.** It
   was a statement about a 512-frame tile, not about the workload.
2. **The fix may not require a larger tile at all.** Capping
   `threadsPerThreadgroup` (at, say, 32) would split the *same* 512-frame tile
   into 16 threadgroups and let it span all 16 cores. That is a one-line change
   in `MetalKernel.mm` and is the obvious first experiment.

Even at 8192 the kernel is roughly **0.5% of the device's arithmetic peak**, and
throughput was still improving at the largest tile measured — so occupancy is
the first bound, not the last. Per-thread register pressure is the likely next
one: a thread evaluates all N voices, and the N=3072 failure is precisely a
per-thread stack overflow.

**Caveat on the large-tile numbers.** These sum `process_ns` over 400 blocks, so
they measure the rate the pipeline delivers samples to the consumer. Where the
worker renders ahead asynchronously (the large tiles), some GPU time is
overlapped rather than waited on, so those figures are throughput-as-delivered
and may flatter the device relative to a pure GPU-busy measure. The 512 and 1024
points have little render-ahead and are the cleanest comparison.

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

## The graph-route anomaly, bisected

The `--fixture graph` route compiles ~100x slower per instruction than the
patch route. The bisect below rules out the obvious causes and localises the
remainder to tropical's JIT invocation rather than to the program.

**Not the voice body.** Holding the route, the summing structure and the
constants fixed and swapping only the oscillator:

```
  FixedSinOsc  N=256   26,150 instrs   1.1s
  MorphOsc     N=256   26,150 instrs   1.2s     (the morph voice, patch route)
  graph source N=256   32,295 instrs   212s
```

**Not param-slot reads.** The playground `source` reads freq/morph from param
slots where the patch fixture folds constants. Reproducing that on the patch
route (one `paramDecl` per voice, freq and then morph via `{"op":"param"}`)
changes nothing: 1.2s at N=256 for all three of const / param / param+morph.

**Not the IR.** This is the decisive one. Standalone LLVM on the *same*
`audio.ll` files, composing exactly what the JIT composes:

```
                       llc -O2 alone   opt -O2 then llc -O2   tropical
  graph  (32,295)         73.9s              3.05s             214.5s
  patch  (26,150)         75.4s              1.28s               2.8s
```

`-mcpu=generic` and `-mcpu=apple-m1` are within noise of each other, and both
modules are 0.07-0.08s at `llc -O0`. Reproduced through one loader with the
kernel cache disabled (`tropical_runtime_bench`, `load_ns`): graph 214.5s,
patch 2.8s.

Two conclusions:

1. **Feeding the scheduler unoptimized IR is what costs, and it is general.**
   `llc -O2` on raw IR is ~74s for either module; running `opt -O2` first drops
   codegen to ~1.4s. The IR pipeline pays for itself roughly 50x over on this
   shape. That matches the pathology the engine already documents at
   `engine/jit/OrcJitEngine.cpp:324`, where an unoptimized sibling JIT
   (`CodeGenOptLevel::None`, linear source-order scheduler) exists precisely
   because "the default backend's scheduler/regalloc go superlinear on exactly
   that shape". Only the stage-0 coefficient kernel is routed to it; the audio
   kernel is not.

2. **The graph module's 214s is not justified by its content.** The identical
   file goes through `opt -O2 && llc -O2` in 3.05s. Something in the JIT path
   is ~70x off the standalone equivalent for this module and not for the patch
   module -- and it is not size, since the patch route compiles 88,000
   instructions (N=1024) in 4.1s. Root cause unidentified; the next step is to
   establish whether the IR transform layer actually runs for this module, which
   needs instrumentation in `OrcJitEngine`.

Minimal repro, no benchmark needed:

```sh
# 3.05s standalone
opt -O2 -S -o /tmp/g_opt.ll <graph audio.ll> && llc -O2 -filetype=obj -o /tmp/g.o /tmp/g_opt.ll
# 214s through the JIT, same file
TROPICAL_KERNEL_CACHE_DISABLE=1 build/tropical_runtime_bench \
  --ir <graph audio.ll> --manifest <manifest.json> --coeff <coeff.ll> \
  --buffer 512 --blocks 5 --warmup 1 --rate 44100
```

Generate the two `audio.ll` files with `--fixture graph --counts 256` and
`--fixture patch --counts 256`, both `--backend jit`, under
`TROPICAL_STAGE0_DUMP`.

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
