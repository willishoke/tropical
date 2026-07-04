# ModalHeavy64 on Metal — findings

**Date:** 2026-07-03 · **Host:** Apple M1 Pro (UMA), macOS 26.3, Metal
**Patch:** `ModalHeavy64` — a 64-partial modal voice, `out(t) = gain · Σ wᵢ·sin(2π·freqᵢ·t)`,
authored in `tropical_program_2` (`modal_heavy64.json`: 64 `Sin` + 64 `VCA` instances),
compiled by `diffcli compile`, rendered by the real engine. `modal_params.h` is generated
from the patch's own coefficients; `mh64_ref.bin` is tropical's f64 render.

This is the de-risk experiment from `design/metal-emitter-evaluation.md`: run a genuine
tropical patch on real Metal hardware, judge fitness by its own goldens (SNR vs tropical's
f64 render, **not** bit-exactness), and measure the realtime win.

## TL;DR

1. ✅ **A real tropical patch runs on real Metal and sounds right.** f32 GPU vs tropical's
   f64 render: **105.8 dB SNR** over the first 0.1 s (17.6 effective bits, correlation
   1.000000) — far past CD quality (~96 dB). f32 is not the problem the scoping doc feared.
2. ✅ **f32 is a sufficient substrate once phase is fixed-point — demonstrated, not argued.**
   The error decomposes cleanly into *value* (f32, fine) and *time* (must be fixed-point):
   naive f32 phase drifts 106→52 dB over 30 s, but a fixed-point-phase kernel holds a **flat
   130 dB across the whole 30 s**, at ~zero perf cost. Time and value are different objects.
3. ✅ **The GPU win is real on a real patch.** vs the CPU-f64 baseline of the same patch:
   crossover ~B=256, then **1.8× at B=512, 3.3× at B=1024**, all realtime-feasible
   (pipelined callback cost ~3.5 µs). Matches the synthetic `gpu_time_partition` crossover
   (`K·B ≈ 10⁵`). *(Caveat: the CPU baseline is scalar f64 `std::sin`, not tropical's
   NEON-vectorized polynomial `sin` — a fair CPU baseline would narrow this; and the GPU is
   still overhead-bound at these sizes. See below.)*

## Fitness — phase substrate comparison, f32 GPU vs tropical f64 render (30 s)

```
phase        SNR       max_abs    SNR@t=0    SNR@t=30s    corr
naive f32   56.2 dB   2.71e-04   105.8 dB     52.1 dB    0.999999
fixed      130.2 dB   3.08e-08   130.1 dB    130.2 dB    1.000000
```

**Why naive drifts.** The naive kernel evaluates `sin(2π·freq·t)` with `t=(start+n)/SR` in
f32 — phase on the *line*. For a high partial (freq≈2980 Hz) at t=30 s, `freq·t ≈ 89400`,
where an f32 ULP is ≈0.008 cycle ≈ 3° of phase error before the sine is taken. f32 precision
is *relative*, so it degrades as the monotone coordinate grows.

**The fix, implemented and measured — fixed-point phase.** Phase is the circle 𝕋 = ℝ/ℤ, a
compact object; represent it as a 64-bit fixed-point fraction (ℤ/2⁶⁴ ↪ 𝕋, a subgroup
inclusion — uniform resolution everywhere, and a group homomorphism, so the rotation action
is exact). Per partial, quantize the increment once: `incr = round((freq/SR)·2⁶⁴)`. Then

```
φ(n) = (nabs · incr) mod 2⁶⁴      // u64 multiply + wrap — exact, uniform, STATELESS
phf  = (φ >> 40) / 2²⁴  ∈ [0,1)   // drop to a bounded f32 only at the sine leaf
```

This is **not** a fixed-point accumulator (that would be state) — φ(n) is a pure function of
the exact integer sample index, so it's exact random-access phase (scrub/jump/play all land
identically) *and* stateless, preserving the ideology. The f32 only ever sees a bounded
[0,1) argument, where it was already scoring 130 dB. Result: uniform 130 dB (~21.6 bits)
across all t — the drift is gone. This is the empirical proof behind the fixed-point-time
refactor: keep the monotone unbounded coordinate in uniform-precision integers, drop to f32
only for the bounded value. (The 130 dB ceiling is the f32 *value* floor, not the time
floor — exactly the point.)

## Performance — M1 Pro

```
B       sync_med  pipe_call  pipe_sust   cpu_f64   deadline   rt     speedup
64        211 u      3.4 u      74 u       15 u      1451 u   PASS   0.20x
128       179 u      3.4 u      70 u       31 u      2903 u   PASS   0.45x
256       174 u      3.4 u      74 u       68 u      5805 u   PASS   0.91x   <- crossover
512       181 u      3.7 u      75 u      148 u     11610 u   PASS   1.97x
1024      190 u      3.5 u      79 u      301 u     23220 u   PASS   3.83x
```

- `sync_med` = single-shot round trip; `pipe_call` = encode+commit on the audio thread
  (pipelined, GPU overlapped); `pipe_sust` = sustained block interval; `speedup` =
  cpu_f64 / pipe_sust.
- Every block size clears its deadline. Below B=256 the GPU's fixed submission cost loses to
  the cheap CPU baseline; above it the GPU wins and widens (3.83× at B=1024). For a *heavier*
  patch (more partials / deeper graph) the crossover moves left and the win grows — the
  synthetic sweep showed 7.8× at `K=256, B=2048`.

## Verdict

The de-risk experiment answers both open questions on a genuine patch: **(a) the heavy-patch
GPU win is real** (2–4× here at reasonable blocks, realtime via pipelining), and **(b) f32 is
viable** — inaudible error short-term, and the long-sustain phase drift is a bounded, known
fix (f64 base phase per block on the host, f32 fine phase on the GPU) that keeps the stateless
design intact. Nothing here argues against building `EmitMsl`; the phase-precision split is
the one concrete constraint it (and the Metal host runtime) must respect.

## Reproduce

```
# reference (from repo root, with diffcli built):
diffcli compile benchmarks/metal_patch/modal_heavy64.json > /tmp/mh64_plan.json
diffcli render-bytes /tmp/mh64_plan.json --frames 5168 --buffer 256 > benchmarks/metal_patch/mh64_ref.bin
# then:
cd benchmarks/metal_patch && make run
```
