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
  stays there. The work per voice has not changed. Swept finely this is a ramp
  beginning just past N=1024, and it is two stacked emission pathologies:
  AArch64 slot offsets leaving the scaled-immediate window, and the per-sample
  straight-line body outgrowing the 192 KB L1 instruction cache. Both are
  measured in "The runtime kernel-size cliff: resolved" below.
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

## The graph-route anomaly: root cause and fix

The `--fixture graph` route compiled ~100x slower per instruction than the
patch route. Bisected to LLVM's pre-RA instruction scheduler, and fixed behind
a knob.

**Not the voice body.** Holding route, summing and constants fixed and swapping
only the oscillator: `MorphOsc` on the patch route is 1.2s at N=256 (26,150
instrs) against the graph route's 212s (32,295 instrs).

**Not param-slot reads.** The playground `source` reads freq/morph from param
slots where the patch fixture folds constants. Reproducing that on the patch
route (a `paramDecl` per voice; freq, then also morph, via `{"op":"param"}`)
changes nothing: 1.2s for const / param / param+morph alike.

**Not the IR pipeline.** Instrumenting the JIT (`TROPICAL_JIT_TRACE=1`, added
in `engine/jit/OrcJitEngine.cpp`) splits the compile wall in two and settles
it. The pipeline runs, and is fast:

```
  patch256   ir-pipeline 636ms  27,438 -> 16,418 instrs   codegen   2.1s
  graph256   ir-pipeline 716ms  32,301 -> 26,398 instrs   codegen 214.7s
```

Codegen is the entire gap: 1.6x the instructions for 100x the time.

**It is the pre-RA scheduler.** Dumping the exact post-pipeline module
(`TROPICAL_JIT_TRACE_DUMP=<dir>`) and compiling it standalone at each
scheduler, on one 26,398-instruction straight-line block:

```
  -pre-RA-sched=   wall        object
  linearize        1.53s        93,176B
  fast             1.82s       103,536B
  list-ilp         2.55s       109,248B
  source          55.39s       103,608B   <- the default on this target
  list-hybrid   2955.19s       100,760B
  list-burr     3594.30s       100,328B
```

`source` is `src_ls_rr_sort`, which is exactly the frame the earlier profile
named (83% of samples in `RegReductionPriorityQueue::pop()`); it selects by
**linear scan over the ready list**, and one enormous straight-line block of
mutually independent voices keeps that list at its worst case. `fast` is ~30x
cheaper for an object 0.07% *smaller*, so the scheduling quality the default
buys is not visible in code size at this shape.

**The fix, measured end to end** (`TROPICAL_JIT_PRERA_SCHED`, 200 blocks):

```
  kernel     scheduler   compile    runtime median
  graph256   default     215.29s      717.8us
  graph256   fast          7.79s      795.5us   (+10.8%)
  graph256   list-ilp      9.52s      938.2us   (+30.7%)
  patch512   default        8.34s    1002.1us
  patch512   fast           7.56s    1040.6us   (+3.8%)
```

`fast` removes the pathology (27.6x on the compile wall) and brings the graph
kernel in line with the patch kernel. It is **not free**: ~4-11% of runtime on
these two kernels, which is real money in a realtime synth, so it is a knob and
not a new default. Left unset the engine keeps LLVM's default and behaves
exactly as before; `ctest` 5/5 and `tropicaltest` 137/137 (byte-for-byte audio
goldens) pass with the instrumentation in place.

**Still unexplained:** why this cliff is shape-dependent rather than
size-dependent. graph256 (26,398 post-pipeline instrs) detonates while
patch512 (larger) does not, and the patch route compiles 88,000 instructions at
N=1024 in 4.1s. The scheduler is the mechanism; what puts one block's ready set
in the pathological regime and not another's is not established. With the knob
in place this is a tuning question rather than a wall.

**Diagnostic surface added** (all default-off, zero cost unset):

- `TROPICAL_JIT_TRACE=1` — per compile: tier, IR bytes, instruction counts
  either side of the IR pipeline, and the wall of each phase, plus a call
  counter that distinguishes one slow compile from many.
- `TROPICAL_JIT_TRACE_DUMP=<dir>` — write the post-pipeline module, so the
  exact IR the codegen layer receives can be compiled standalone.
- `TROPICAL_JIT_PRERA_SCHED=<name>` — select the pre-RA scheduler.

## Metal-target compile time is the JIT, not the shader compiler

Splitting a Metal load with the same instrumentation (patch fixture, current
binaries, cache disabled), against the mandatory dual-loaded JIT:

```
    N    metal total   jit portion   metal shader   msl size
  256        3.28s      3.11s (95%)      0.17s        928KB
  512        8.86s      8.63s (97%)      0.23s        1.9MB
 1024       18.44s     17.89s (97%)      0.55s        3.8MB
 2048       46.55s     45.73s (98%)      0.82s        7.8MB
```

**Apple's Metal shader compiler is not the bottleneck** — 0.82s for 7.8MB of
MSL, scaling sublinearly. Tropical's own MSL emission is ~0.2s. Between them
they are ~2% of the wall. (This is the flat contradiction of the first
diagnosis offered in this investigation, which blamed the MSL emitter; that
was wrong.)

**~98% of a Metal target's compile wall is the dual-loaded JIT** — and on a
Metal session the JIT kernel does not produce the audio. It exists for
`render_window` and reference comparisons. So nearly the entire Metal load is
spent compiling a kernel that is off the audio path.

That makes the largest Metal-specific opportunity a **scheduling** question,
not a codegen one — with the caveat that scheduling moves work rather than
removing it. Three distinct things could be meant, and only one is a saving:

- **Elimination.** A session that never opens the scope or calls
  `render_window` never needs the reference at all, so the compile is avoided
  rather than deferred and total CPU genuinely drops. Conditional on how often
  the reference is used, which is not measured here.
- **Deferral to first use.** No saving of any kind: the same 45.7s moves from
  load onto the first `render_window` call, i.e. onto an interactive moment.
  Strictly worse than paying it at load. Not worth doing alone.
- **Overlap.** Compile the reference on a background thread after publishing
  the Metal kernel. Total CPU is unchanged and nothing gets faster, but audio
  starts when the GPU pipeline is up (~0.8s here instead of 46.55s) and the
  scope arrives when it is ready.

Overlap is the one worth building. It is the same move `../gpu_time_partition`
used to hide the GPU submission toll: latency off the critical path, throughput
untouched. ORC already supports it — `compile_ir_text` calls `addIRModule` and
then immediately `lookup(symbol)`, and the instrumentation shows the `lookup`
is what triggers materialization, so the eagerness is tropical's choice rather
than LLVM's.

Not attempted. Two risks want measuring first: every hot-swap re-arms the
compile, so repeated topology edits followed by opening the scope could pay it
repeatedly; and `../metal_live`'s qualification takes reference checkpoints at
specific moments, so the gates stay valid but their timing assumptions need
re-examining.

Note that none of this reduces the compile cost itself. The levers that do are
a `bankSum` region (smaller IR) and, for the pathological shape only, the
pre-RA scheduler knob below.

**The pre-RA scheduler knob does not help the Metal path**, because these
modules are not on the cliff:

```
  N=1024   default 16.80s   fast 18.45s     gpu runtime 3132.0 -> 3082.7us
  N=2048   default 45.42s   fast 50.60s     gpu runtime 6785.5 -> 6844.8us
```

Marginally worse, and GPU runtime is unchanged as expected — the knob affects
the CPU kernel's codegen, and on Metal that kernel is the reference, not the
audio. `fast` is for the pathological shape only; it removes a cliff rather
than speeding up healthy compiles.

## Improving JIT compile time: the measured menu

The compile wall is codegen, not the IR pipeline, and codegen quality is a
dial separate from the optimization that produces runtime performance. All
figures below are cold-cache, `tropical_runtime_bench`, 200 blocks.

**The object cache already covers repeats.** graph256 cold 212.3s, warm 0.83s
(256x), keyed on IR hash plus build id. The wall is a FIRST-compile cost. That
still bites interactively -- every topology edit is new IR -- but a replayed
patch is nearly free, and every measurement in this document disables the cache
deliberately to expose the underlying cost.

**Codegen opt level, independent of the IR pipeline**
(`TROPICAL_JIT_CODEGEN_OPT`) — **now defaulted to `none`.**

An earlier revision of this file reported +29% / +19% runtime for `none`.
Those were CONTENTION, measured in a loop against other work — the same trap
the harness settles for. Isolated, repeated, 300 blocks, the penalty is gone:

```
  kernel              default      none        delta
  oscillators N=64     130.6us    130.8us      +0.2%
  oscillators N=256    536.1us    539.5us      +0.6%
  oscillators N=512   1003.2us   1010.5us      +0.7%
  oscillators N=768   1503.9us   1500.2us      -0.2%
  oscillators N=1024    19.21%     19.50% (saturation)
  oscillators N=1536    44.78%     44.80% (saturation)
  oscillators N=2048    59.99%     59.99% (saturation)
  graph256 (x2 reps)   736.6us    736.3us  /  717.0us  716.6us
  gong                 116.8us    117.0us      +0.2%
  resonator->reverb   3576.7us   3590.3us      +0.4%
  256-partial bank    1631.5us   1630.3us      -0.1%
```

The bank row matters most: it is a `bankSum` REGION — a loop, not straight
line — which is the shape where scheduling and loop-aware register allocation
would most plausibly pay. It does not.

Compile, meanwhile:

```
  kernel              default      none
  oscillators N=512      8.40s     1.46s   (5.8x)
  oscillators N=1024    17.35s     2.83s   (6.1x)
  oscillators N=2048    47.83s     5.92s   (8.1x)
  graph256 (worst)     213.30s     0.92s   (232x)
  `less`               214.44s             (no help: same scheduler)
```

Why: a closed-form kernel is long straight-line f64 arithmetic (or one bounded
reduction) that is dependency- and memory-bound. There is little for an
aggressive scheduler or a graph-colouring allocator to win, so the expensive
dial buys nothing measurable while costing 6x-232x of compile.

Audio goldens are byte-identical under the new default (tropicaltest 137/137,
ctest 5/5) — codegen opt level does not perturb FP semantics.
`TROPICAL_JIT_CODEGEN_OPT=default` restores LLVM's choice with no rebuild.
If a future kernel shape (tight loops, high register pressure, branchy control
flow) does reward real codegen, re-measure before assuming this still holds.

**Pre-RA scheduler only** (`TROPICAL_JIT_PRERA_SCHED=fast`), a gentler point on
the same curve: graph256 7.79s / +10.8%, patch512 7.56s / +3.8%.

So there is a spectrum, not a single answer:

```
  codegen=none        232x compile   +29% runtime    (fast ISel, regalloc-fast)
  sched=fast           27x compile   +11% runtime    (good ISel/RA, cheap sched)
  default               1x           baseline
```

**Tiered compilation is NOT needed for this.** The obvious design — compile
fast, start audio, recompile at quality in the background, hot-swap — assumes
the fast tier is a compromise you want to escape. It is not: at these kernel
shapes `none` and `default` produce indistinguishable runtime, so there is no
second tier worth upgrading to. Defaulting the dial gets the whole win with a
flag instead of a background compiler, a swap protocol, and the failure modes
both carry. Tiering stays available if a future shape actually rewards
codegen — but on this evidence it would be machinery built to recover nothing.

**On the LLVM-vs-Metal asymmetry.** At `codegen=none` the JIT compiles
N=2048-class work in ~1.5s, the same order as Metal's 0.82s shader compile. The
gap was never LLVM being slower than a Metal assembler -- Apple's Metal
compiler *is* LLVM (MSL -> AIR is Clang; the driver lowers AIR at PSO
creation). The asymmetry is that a CPU backend schedules for a superscalar
out-of-order core with 32 architectural registers, while a GPU backend hides
latency with SMT and allocates from a huge register file, so aggressive pre-RA
scheduling buys it little. Like for like the two are comparable; tropical was
simply asking for the expensive dial.

## The runtime kernel-size cliff: resolved

**Date:** 2026-09-01 · **Commit:** `77bd626` · same host.
Probes: [`cliff_probes/`](cliff_probes/) · row
[`data/knee-fine-20260901T072327Z.jsonl`](data/knee-fine-20260901T072327Z.jsonl)

The per-voice step reported above (and independently in
`benchmarks/faust_comparison`) is **two separate emission pathologies stacked
at nearby counts**, not one. Both are named below, both are measured with
instruction-level counters taken from the actual emitted object, and both are
fixable. Neither is the compile-side scheduler cliff that `2ed2466` defused.

The 1.5x bracket in the original sweep hid the shape. Swept finely, the "cliff"
is a **ramp** that begins just past N=1024 and saturates near N=1400:

```
   N     ns/voice   emitted __text
  512      2409        118,724
  768      2341        177,940
  896      2354        207,600
 1024      2378        237,268
 1056      2640        246,456
 1088      3000        255,532
 1104      3429        264,012
 1120      3522        268,548
 1136      3771        273,068
 1152      3985        277,596
 1280      4110        313,860
 1536      4256        386,396
 2048      4451        576,988
```

`min` steps with `median` (2338 -> 3762 ns/voice at the same points), so this is
structural per block, not thermal drift or a tail artifact.

### Cause 1 — the AArch64 scaled-immediate window on slot offsets

`EmitLlvm` gives every intermediate its own `%slots` index and addresses it with
a constant GEP (`loadSlotF64`/`storeSlotF64`, `Ir/EmitLlvm.lean:197-204`). A
64-bit AArch64 `ldr`/`str` encodes an unsigned offset scaled by 8 in 12 bits, so
the last addressable slot is index **4095** (byte 32,760). Past it the backend
must materialise the offset in a register.

The patch fixture allocates `4N+3` slots, so the window closes at N=1023. The
emitted object agrees exactly, in two fixtures with different slots-per-voice:

```
  register-offset ldr/str  =  slots_per_voice x (N - N_wall),  N_wall = (4095-3)/spv

  FixedSinOsc          (4 slots/voice, N_wall=1023):  N=1024 -> 4     N=1536 -> 2052
  FixedSinOsc+SoftClip (7 slots/voice, N_wall=584.6): N=608  -> 164   N=768  -> 1284
```

Every emitted object pins `max_imm` at exactly `0x7ff8`. The damage is not the
addressing form itself but what it does to `RegAllocFast`: the extra live
offsets evict the `%slots` base pointer, which is then reloaded from the stack
**once per voice for the whole kernel**, not just past the wall.

```
  FixedSinOsc+SoftClip     N=576      N=608
  slot_count               4,035      4,259     <- crosses 4,095
  base-pointer reloads         1        608
  loads from sp              588      8,068     <- 13.7x
```

Rewriting the GEPs against page-relative bases (`cliff_probes/rebase3.py`, page
= 4096 doubles, base hidden behind a zero-instruction `asm ""` identity so
InstCombine cannot reassociate it away) removes the pathology completely —
`regoff` 516 -> 1, base reloads 1149 -> 1, `sp` loads 2828 -> 1167 — and is
worth:

```
                       stock ns/voice   rebased ns/voice
  4 slots/voice N=1152      3,985            3,504    -12%
  7 slots/voice N=608       5,322            3,534    -34%
  7 slots/voice N=640       6,397            3,533    -45%
```

Real, and larger the more slots a voice holds. **It is not the cliff**: with it
removed the ramp is still there, and every voice now emits an identical 57.9
machine instructions on both sides of it.

### Cause 2 — the per-sample body outgrowing the L1 instruction cache

The kernel is one straight-line basic block executed once per sample. With
cause 1 repaired the per-voice instruction stream is byte-identical across the
ramp, so the only remaining variable is how much of it there is.

Measured directly, holding the voice count fixed at 896 and padding the loop
body with independent integer adds that do no useful work
(`cliff_probes/pad.py`):

```
   pad insns    __text     ns/sample   marginal cost per byte of code
          0    207,600       4,200.2
      2,048    215,788       4,249.4        6.0 ps/B
      4,096    223,980       4,388.3       17.0 ps/B
      6,144    240,360       5,328.0       57.4 ps/B   <-- onset
      8,192    256,744       7,557.4      136.1 ps/B
     12,288    289,508       9,934.7       72.6 ps/B
     16,384    322,272      11,280.7       41.1 ps/B
     24,576    387,800      13,892.5       39.9 ps/B
     32,768    453,328      16,100.0       33.7 ps/B
```

At fixed work and fixed data footprint, an instruction that computes nothing
costs ~0.02 cycles while the body is under ~224 KB and ~1 cycle once it is over
~240 KB — a 10-20x change in the price of code. `hw.perflevel0.l1icachesize` on
this host is **196,608 B**. The onset sits modestly above nominal capacity and
the degradation is gradual rather than a step, which is what a non-LRU
replacement policy plus L2 next-line prefetch absorbing the first tens of KB of
overflow would produce.

That threshold predicts both fixtures' knees from their emitted size alone:

```
                          flat through          degrading from
  4 slots/voice (232 B/voice)   N=1024, 237,260 B    N=1056, 244,648 B
  7 slots/voice (340 B/voice)   N=640,  217,344 B    N=672,  228,184 B
```

Of the 1,599 ns/voice excess at N=1152, ~480 (30%) is cause 1 and ~1,120 (70%)
is cause 2.

### Ruled out by measurement

- **L1d capacity on temps/slots** (handoff hypothesis 1) — striding every
  `%slots` access so the touched array grows 32,792 -> 131,144 B, with the
  instruction stream held byte-identical (`cliff_probes/stride.py`), costs
  **3%**. The working set is not the constraint; the arithmetic that made
  ~98 KB look suspicious was coincidence.

```
   stride   slots bytes   ns/voice
        1        32,792     2,353
        2        65,576     2,415
        4       131,144     2,424
```

- **Stage-0 changing character at scale** (hypothesis 3) — `coeff.ll` is empty
  at every count for this fixture; the split has nothing to do here.
- **P-core to E-core migration.** The ~1.7x ratio looks like one, but forced
  background scheduling (`taskpolicy -b`) is **9x** slower, not 1.7x.
- **Thermal or DVFS drift** — the `min` block steps with the median.
- **The IR pipeline.** With `TROPICAL_JIT_OPT_LEVEL=O0` there is no knee at all
  (5333 -> 5943 ns/voice from N=768 to 1280, smooth): the kernel is uniformly
  ~2.4x slower and instruction fetch is hidden behind it. The knee is what
  happens when good code runs out of instruction cache.

### Does banking remove it?

Not demonstrated, and the standing claim in *Caveats* was too strong. Measured
on the `resonator` bank, 256 -> 12,288 partials shows no knee at all — but not
because the emitted IR is O(1) in N:

```
  partials   per-sample IR insns   per partial   ns/partial
       256                 1,557          6.08        6,443
     2,048                12,524          6.12        6,387
     8,192                49,388          6.03        6,423
    12,288                73,964          6.02        6,846
```

`bankSum` banks the **reduction** — the `rd_body` region is a flat 192
instructions at every count — while per-partial evaluation stays straight-line
in the per-sample block. By 12,288 partials that block is well past the
instruction-cache threshold, and it still does not knee, because the resonator
runs at ~2 ns per per-sample IR instruction (vs 0.08 for `FixedSinOsc`):
latency-bound arithmetic hides the fetch cost entirely.

So the honest answer is: **banking moves the wall in proportion to the
per-sample straight-line code it removes per voice, and does not remove it.**
Six instructions per partial instead of 86 per voice is a ~14x shift — from
~1,000 unrolled voices to an extrapolated ~14,000 banked partials. A bank whose
per-sample body is genuinely O(1) (a loop over an array, not an unrolled
evaluation feeding a banked sum) would remove it; the resonator's is not that
shape.

### What to do

1. **Page-relative slot bases in `EmitLlvm`** — emit one
   `getelementptr double, ptr %slots, i64 4096k` per page in `entry` and index
   from it, keeping every immediate inside the window. Proven above by IR
   rewriting to be worth 12-45% past the wall, semantics unchanged (identical
   IR after `-O2`, `nonfinite_count` 0). Not implemented here: it changes every
   emitted kernel and wants `make validate` behind it.
2. **Per-sample straight-line code is the scaling budget, and it is ~230 KB.**
   That is the number a banked oscillator family is buying room against — a
   concrete runtime argument for banking, not only the compile-time one.

## Caveats

- Metal's `process_ns` is a synchronous worker-tile wait, so it measures GPU
  **throughput**, not callback cost. With the worker pipelined the callback
  only copies from a ready tile; `gpu_time_partition` measured that path at
  ~4–6 µs. These numbers say "the GPU can sustain this block rate", not "the
  callback costs this much".
- The GPU was shared with the desktop compositor (WindowServer at ~36% CPU
  during some runs). A reserved-GPU measurement would be needed before any
  production claim, as `gpu_time_partition` also notes.
- Voices here are **unrolled**, not banked; the playground `source` and patch
  `FixedSinOsc` paths never construct a `bankSum` region, because banking is
  chosen at authoring time by the arrow modal builder (`Ir/BanksFlag.lean`). A
  banked oscillator family is the obvious follow-on. An earlier revision of this
  line claimed a `bankSum` region would make the emitted IR O(1) in N; measured,
  it does not — it banks the reduction and leaves per-partial evaluation
  straight-line. See "Does banking remove it?" above.
- Single host, single row. No other hardware is inferred.
