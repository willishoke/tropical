# Oscillator saturation — how many voices before a backend runs out of budget

## The question

The modal work — resonator banks, reverbs, the exact-product circuit — measures
graphs that are heavy by construction. This benchmark measures the other end of
the vocabulary: the plain `source` oscillator, the cheapest real signal the
patch bay has. How many can run at once before a backend spends half its
realtime budget producing one block, and where does the GPU overtake the JIT?

Saturation is defined per block, and it is the only metric here:

```
saturation(N) = per-block production wall / block deadline
block deadline = buffer_frames / rate          (512 / 44100 = 11.610 ms)
```

**Capacity at threshold T** is the largest oscillator count whose saturation
stays at or below `T` (default `T = 0.50`). The harness *brackets* that count
between two adjacent measured points and never promotes an unmeasured count to
a measured one; a log-linear midpoint is recorded separately under
`interpolated_estimate` and is labelled as an estimate everywhere it appears.

## The fixtures

Two spellings of the same question. **`--fixture patch` is the default** and is
the one that reaches useful counts.

**`patch`** — `N` `FixedSinOsc` instances (the minimal oscillator the stdlib
registers) summed through instance inputs into one `SoftClip`, as a
`tropical_program_2` patch. `dac.out` must be a bare `ref`, so the sum is
folded through inputs and terminated at a shaper. This route reaches
`--mode`, so `fused` and `microkernel` realizations are both measurable, and
it compiles linearly to at least N=1024.

**`graph`** — `N` playground `source` nodes (`freq`/`morph`/`pm`, a heavier
morph oscillator) through one `mix` into `out`. Kept because it is what a patch
built by hand in the playground actually produces — but it hits a compile-time
pathology above ~32k instructions that makes counts past 256 impractical. See
[`findings.md`](findings.md); treat it as a repro, not a sweep.

Voices are **not** banked in either fixture — this is the unrolled path, `N`
distinct kernels in one block. Frequencies are spread by a fixed fraction per
voice (`--spread`, default 1.7%) so no two voices share a coefficient:
identical voices would let common-subexpression elimination collapse the graph,
and the sweep would scale *instance count* without scaling *work*. The gate
`test_frequencies_are_distinct_so_cse_cannot_collapse_voices` pins it.

## Run

```sh
make build && make lean

# quick shape check, JIT only, a few minutes
benchmarks/oscillator_saturation/run.sh --suite smoke --backend jit

# the full sweep, both backends (~35 min; this produced findings.md)
benchmarks/oscillator_saturation/run.sh \
    --counts 1,2,4,8,16,32,64,128,256,512,768,1024,1536,2048,3072

# a specific question
benchmarks/oscillator_saturation/run.sh --counts 256,384,512 --backend jit \
    --saturation 0.5 --statistic p99
```

Useful flags: `--fixture patch|graph`, `--mode fused|microkernel`,
`--saturation` (threshold), `--statistic median|p95|p99`, `--buffer`,
`--blocks`, `--warmup`, `--morph` (graph fixture only), `--emit-timeout`
(per-count artifact budget).

Focused gates, which compile and measure nothing:

```sh
python3 benchmarks/oscillator_saturation/test_run.py
```

## What is actually being timed

Both backends go through the same emit (`diffcli compile --mode` then a
`render-bytes`/`render-metal` artifact dump for the patch fixture;
`render-graph` for the graph fixture) and the same `tropical_runtime_bench`
block loop, so the only difference between a JIT row and a Metal row is the
backend under test. `process_ns` is a single
`tropical_runtime_process_offline` call:

- **JIT** — the CPU kernel computes the block. The wall is compute.
- **Metal** — the caller waits for the worker's *exact next tile*. The wall is
  synchronous GPU production, including the per-dispatch submission cost that
  `../gpu_time_partition/findings.md` measured at a flat ~200–330 µs.

Because Metal's number is synchronous, it is a **throughput** measure, which is
what saturation means. It is not the cost an audio callback pays: with the
worker pipelined, the callback only copies from a ready tile.
`gpu_time_partition` measured that critical path at ~4–6 µs. A Metal
saturation figure here therefore says "the GPU can sustain this block rate",
not "the callback costs this much".

## Measurement hygiene

Adjacent runs contend for the same cores and GPU queue. An unsettled sweep
inflates the tail without moving the median, which silently corrupts any
p99-derived capacity — an early probe produced a 305% p99 at N=512 that a
settled rerun put at 60% with zero overruns. The harness therefore sleeps
`--settle-seconds` (default 1.0) between emit and measure, runs every
measurement in its own subprocess, and records `overrun_count` and
`nonfinite_count` per point.

Every inherited `TROPICAL_*` variable is removed before the explicit controls
are set and recorded, and the kernel cache is redirected to a harness-owned
root under `.work/<tag>/cache` so a sweep never reads or writes the ordinary
`~/.cache/tropical/kernels` tree.

## Compile time, and backends that refuse

Artifact emission is recorded per point (`emit_ns`) because at high counts it
becomes the binding constraint rather than a setup cost — it is superlinear in
unrolled voice count for *both* backends (they are within ~2% of each other).
`--emit-timeout` bounds a sweep; a point that exceeds it is recorded as
`emit_timed_out`.

A backend may also **refuse** a count outright: Metal rejects a kernel
exceeding its per-thread stack somewhere between N=2048 and N=3072. That is the
backend's structural ceiling and is recorded as `emit_failed` with the error.
Timed-out and refused points are excluded from the capacity bracket and the
crossover rather than guessed at, and the sweep stops that backend and keeps
every point it already measured — a failure at the top of a sweep must never
discard the row.

## Output

One JSONL record per run in `data/<tag>.jsonl`, carrying the environment
manifest (commit, hardware, toolchain, controls), every measured point with its
full order statistics and artifact digests, the per-backend capacity bracket,
and the JIT/Metal crossover. Interpretation lives in
[`findings.md`](findings.md).
