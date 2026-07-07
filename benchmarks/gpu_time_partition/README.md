# GPU time-partition latency

## The question

`../simd_time_partition.cpp` established the CPU half: for a closed-form-in-τ kernel every
output sample is an independent function of τ, so `W` consecutive samples fill `W` SIMD
lanes at once — capped at NEON's 4. This benchmark takes the **same time-partition axis to
the GPU**, where a realtime block of `B` samples is one parallel dispatch of `B` threads,
one per time index, no cross-thread state.

The only thing that gates a *realtime* GPU backend on unified memory: does that per-block
dispatch finish inside the audio deadline, `B / sampleRate`?

## Why this matters for tropical

Stateful DSP fills a buffer serially — `sample[n]` needs the state left by `sample[n-1]` —
which is the actual reason audio doesn't go on the GPU. Tropical's kernels are stateless
closed forms `f(τ, params)`, so the recurrence is gone and *time itself parallelizes*: the
whole block is embarrassingly parallel. Statelessness costs FLOPs per sample (a closed-form
modal sum instead of a cheap difference equation), which on the CPU caps how heavy a patch
can be and stay realtime. The GPU is where that cost could be *refunded in realtime* — but
only if the dispatch round trip fits the deadline. That is the feasibility question this
answers before any IR-lowering work is committed.

Offline/throughput is deliberately **out of scope**: drop realtime and the stateless design
is pure overhead you'd replace with feedback at 100× lower cost. Realtime is the only regime
where the design earns its keep, so it's the only one worth benchmarking.

## Run

```
make run        # Apple Silicon + Metal; builds and sweeps B × K, prints the table
```

No `.metal` step — the shader is compiled at runtime via `newLibraryWithSource`. Requires
the command-line tools (`clang++`, Metal + Foundation frameworks).

## Results — Apple M1 Pro (UMA)

Feasible, with a caveat: every config clears its deadline at p99, but the GPU pays a flat
~200–330 µs submission toll per block, so it only *beats the CPU* once the per-block work is
heavy enough to amortize it (crossover ≈ `K·B = 10^5` sines/block; up to 7.8× at
`K=256, B=2048`). It's a targeted tool for fat closed-form patches, not a blanket
replacement — light patches stay on the NEON path. See `findings.md` for the full read and
`data/results.txt` for the raw sweep.
