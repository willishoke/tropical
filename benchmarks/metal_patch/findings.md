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
2. ⚠️ **But naive f32 phase drifts with time.** Over 30 s the SNR falls from 105.8 dB (t=0)
   to 52.1 dB (t=30 s); aggregate 56.2 dB. Root cause below — it's fixable and does **not**
   need f64 on the GPU.
3. ✅ **The GPU win is real on a real patch.** vs the CPU-f64 baseline of the same patch:
   crossover ~B=256, then **1.97× at B=512, 3.83× at B=1024**, all realtime-feasible
   (pipelined callback cost ~3.5 µs). Matches the synthetic `gpu_time_partition` crossover
   (`K·B ≈ 10⁵`).

## Fitness — f32 GPU vs tropical f64 render

```
window            SNR        notes
first 0.1 s     105.8 dB     ~17.6 effective bits, corr 1.000000 — inaudible error
last 0.1 s @30s  52.1 dB     phase-drifted
30 s aggregate   56.2 dB     ~9.3 effective bits
```

**Why it drifts.** The kernel evaluates the stateless closed form `sin(2π·freq·t)` with
`t = (start+n)/SR` in f32. For a high partial (freq≈2980 Hz) at t=30 s, `freq·t ≈ 89400`,
where an f32 ULP is ≈0.008 — i.e. ~0.008 cycle ≈ 3° of phase error before the sine is even
taken. tropical stays exact only because it computes this in **f64** (ULP ≈1e-8 there). This
is fundamental to *stateless* f32 phase, not a bug: you can't accumulate phase incrementally
without reintroducing per-sample state, which the whole design forbids.

**The fix (keeps statelessness, stays on f32 GPU).** Split the phase by timescale: the
control plane computes a per-block **base phase** per partial in f64 —
`φ₀ᵢ = frac(freqᵢ · start / SR)`, once per block, off the audio-rate path — and the GPU
kernel adds only the **within-block** fine phase in f32, `frac(φ₀ᵢ + freqᵢ · n / SR)` with
`n < blockSize`. Because `n` is bounded (≤ a few thousand), `freqᵢ·n/SR` never grows large,
so f32 keeps full precision indefinitely. This is a real design note for `EmitMsl` + the
Metal host runtime: coarse phase in f64 on the host (cheap, once per block), fine phase in
f32 on the GPU. It also matches how tropical's own `Sin` does range reduction — a faithful
`EmitMsl` emitting the reduced-phase arithmetic (rather than a naive `sin(2πft)`) would
already be closer to this.

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
