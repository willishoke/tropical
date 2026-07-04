# GPU time-partition — findings

**Date:** 2026-07-03
**Host:** Apple M1 Pro (arm64, unified memory), macOS 26.3, Metal
**Question:** For a stateless closed-form-in-τ kernel, a realtime block of `B` samples is
one parallel dispatch of `B` independent threads. Does that dispatch complete inside the
audio deadline (`B / sampleRate`) on unified memory — i.e., is a *realtime* GPU backend
feasible at all? (Offline/throughput is explicitly out of scope: if you drop realtime you
throw away the stateless design and render 100× cheaper with plain feedback. Realtime is
the only regime where the design earns its FLOP cost, so it is the only regime worth
testing.)

## TL;DR

**Feasible on UMA, but it's a targeted tool, not a blanket win.**

1. ✅ Every tested config clears its deadline at p99 — the per-block round trip
   (encode + commit + wait) fits inside `B / 48000` with margin.
2. ⚠️ The GPU pays a **flat ~200–330 µs submission toll per block**, nearly independent
   of `B` and `K`. It's overhead-bound, not compute-bound, at these sizes. So the GPU is
   a *loss* for light kernels — the CPU does `K=16` in 4.5 µs, the GPU floor is ~205 µs.
3. ✅ It **wins, and widens fast, once the per-block compute is heavy enough to amortize
   the toll** — crossover around `K·B ≈ 10^5` sines/block. At `K=256, B=2048` the GPU is
   **7.8× faster** than the single-thread CPU baseline (279 µs vs 2188 µs).
4. ⚠️ The one realtime hazard is the **jitter tail**: worst-case dispatch ~1.0–1.6 ms.
   That only threatens the tightest deadline (`B=64`, 1.33 ms), where a rare stall would
   drop a buffer. At `B≥128` even the max fits. Pipelining (compute block *n+1* while
   playing *n*) hides the toll entirely; the synchronous round trip measured here is the
   worst case.

## Setup

`gpu_time_partition.mm` — one Objective-C++ translation unit, Metal reached at runtime
(no `.metal` step). Kernel: modal residue sum, `out(t) = Σ_m a_m·e^(−d_m·t)·sin(2π f_m t)`
at `t = t0 + n·dt`, one thread per time index `n` — the same kernel shape as the CPU
sibling `../simd_time_partition.cpp`, whose 8×8 rich kernel is 64 sines/sample (`K=64`
here). Buffers are `StorageModeShared` (zero-copy on UMA). We measure the full round trip
a realtime callback pays — `[cb commit]; [cb waitUntilCompleted]` — over 2000 iterations
after 64 warm-ups, and report median / p99 / max against the deadline, with a single-
thread `-O3` CPU baseline (auto-vectorized across `n`) for the crossover.

Verdict column: `PASS` = p99 < deadline; `p99>dl` = median fits but the jitter tail
overruns; `FAIL` = median ≥ deadline.

## Results

```
K    B        gpu_med    gpu_p99    gpu_max    cpu_med   deadline   verdict
16   64        206.2u     689.6u    1450.1u       4.5u    1333.3u   PASS
16   128       204.2u     553.5u    1458.5u       8.8u    2666.7u   PASS
16   2048      222.1u     571.0u    1610.4u     155.7u   42666.7u   PASS
64   64        225.5u     685.9u    1464.9u      18.8u    1333.3u   PASS
64   1024      230.2u     675.9u    1462.0u     301.8u   21333.3u   PASS   <- GPU passes CPU
64   2048      217.6u     352.0u    1113.8u     579.4u   42666.7u   PASS   (2.7x)
256  64        231.4u     624.7u     834.2u      83.6u    1333.3u   PASS
256  256       285.0u     691.9u    1412.2u     296.7u    5333.3u   PASS   <- crossover
256  512       280.9u     730.5u    1565.9u     583.9u   10666.7u   PASS   (2.1x)
256  1024      333.8u     805.7u    1570.2u    1118.5u   21333.3u   PASS   (3.4x)
256  2048      279.0u     740.4u    1285.2u    2188.4u   42666.7u   PASS   (7.8x)
```

(Full sweep in `data/results.txt`.)

## Interpretation

- **The original latency worry is answered: on UMA it works.** The per-block round trip
  is a flat few-hundred-µs, and for any block ≥128 samples it fits the deadline including
  the worst case. Round-trip / PCIe fears do not apply when host and device share memory.
- **But the ~200 µs floor is the story, not the compute.** At these problem sizes the
  benchmark measures Metal command-buffer *submission* overhead — one command buffer per
  block, created/encoded/committed/waited synchronously. That toll, not the arithmetic,
  is what the GPU spends. It is not fundamental: persistent/indirect command buffers and
  pipelined submission are the obvious next reductions.
- **GPU is the right tool only above the crossover.** The dividend the statelessness pays
  for in CPU FLOPs is refunded *in realtime* exactly when the per-block work is large —
  fat modal sums (`K` big) at reasonable blocks (`B≥512`). Below that, the NEON time-
  partition path from the sibling benchmark is strictly better. This is a routing
  decision, not a replacement: heavy closed-form patches that a CPU can't fill in time
  become realtime on the GPU; light ones stay on the CPU.

## Caveats / next

- Single command buffer per block is the naive submission path; the flat floor is its
  ceiling on improvement. Measure indirect command buffers + pipelined double-buffering.
- `waitUntilCompleted` here also serializes with an idle GPU (no display contention). A
  realtime audio thread shares the GPU with the compositor; robustness under contention
  needs a reserved-GPU test (headless / Jetson) before any production claim.
- This is a hand-written stand-in for the real kernel — it validates the *dispatch model*,
  not tropical's emitted IR. Lowering the actual arrow graph to Metal (or NVPTX via the
  existing LLVM path) is the follow-on if the model is worth pursuing.
