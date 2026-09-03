# tropical vs Faust

An external sanity check: is tropical in the same order of magnitude as a
mature DSP compiler, on a workload both can express?

Faust is the right first comparison because it is also a compiler — a whole
graph lowered to a single sample loop, the same fusion philosophy as
tropical's kernel — rather than a runtime graph of precompiled UGens.
Comparing against an interpreter would conflate fusion with per-sample math.

## Run

```sh
make build && make lean          # tropical side
brew install faust               # comparison side

benchmarks/faust_comparison/run.sh \
    --counts 64,256,512,1024,1536,2048 --precision double
```

Flags: `--precision double|single` (double matches tropical's f64 JIT; Faust
users often ship float, which vectorises better), `--counts`, `--buffer`,
`--blocks`, `--warmup`, `--settle-seconds`, `--faust-timeout` (Faust aborts
its own compile after `-t` seconds, default 120, which a 2048-term expression
exceeds), `--skip-tropical`.

## Method

Three Faust variants separate two axes that a naive comparison conflates —
see [`findings.md`](findings.md). Both sides are measured identically: same
block size, rate, warmup, an isolated process per point, median over N blocks,
and `saturation = median block wall / (buffer / rate)`, the same metric as
`../oscillator_saturation`. Faust is compiled by the same LLVM the tropical
JIT uses, so codegen quality is not a hidden variable.

`bench_arch.cpp` is a Faust architecture file that wraps the generated DSP in
the same block loop `tropical_runtime_bench` uses. `dsp/gen_dsp.py` emits the
variants at a given voice count.

## Caveat that governs the whole table

tropical's `FixedSin` is a fixed-point Q31 polynomial; Faust's `os.osc` is an
interpolated wavetable. Different numerical methods with different accuracy
and aliasing behaviour, so "ns per voice" is not the whole story, and the
headline loss is mostly a property of that choice rather than of the
architecture. `findings.md` says which single row is actually about the
design.
