# Metal backend — live findings (V2 phase 6)

## 2026-07-27 sprint qualification update

The new qualification surface is implemented and has produced two distinct
evidence classes:

- The full short latency matrix passed on the canonical M1 Pro:
  B=128/256/512 × D=1/2/3 × raw/glide/anchor/velocity. Every one of the 36
  Metal rows first reflected the impulsive write after exactly D blocks; the
  three JIT reference rows reflected it in block zero. Thus observed transport
  latency exactly matched `D×B/44100`, from 2.90 ms (D1/B128) through 34.83 ms
  (D3/B512). Multi-slot dispatch took 41–417 ns median across these short
  rows. This measures captured-snapshot transport, not the deliberately gradual
  audible onset of a glide.
- CTest now proves the legacy D=3 spelling, explicit D=1/2/3 precedence,
  default synchronous mode, invalid-depth refusal, exact D-block live-column
  lag, clock-jump draining, and hot-swap re-prime on real Metal hardware.

The first 10-second default-device smoke recorded one RtAudio underrun despite
0 callback budget overruns (859 callbacks, 0.147 ms average, 4.889 ms max).
That row is a real qualification failure and is retained. Following the staff
stop-line protocol, the harness was split into cumulative snapshots and rerun:
a 15-second diagnostic recorded zero underruns at startup/warm-up, clean
post-reset baseline, after writes, after the clock jump, after hot-swap, and
after stop. Its measured window had 1293 callbacks, 0.096 ms average, 2.723 ms
max, and zero overruns. This permits a reset-bounded 30-minute measurement; it
does not erase the original one-underrun row.

The long-row status and raw-data links will be frozen here after the stable
implementation commit. Until that row completes, Metal remains
**qualification pending**, not release-qualified.

**Date:** 2026-07-07
**Host:** Apple M1 Pro, macOS 26.3, 44.1 kHz, B=512 (engine boot default)
**Patches:** `modal_fixed{128,256,512}.json` — production-style fat additive
voices: N × `FixedSinOsc` (integer phase, Q2.30 datapath) × `VCA`, golden-ratio
frequency spread, 1/k^1.3 weights. Generated here; the honest post-scope-A
heavy voice (unlike `metal_patch/modal_heavy64`, the pre-scope-A
unreduced-radian relic kept as a canary in `metal_vs_jit`).

## TL;DR

1. **The GPU wins on the real engine at every size tested** — sync-dispatch
   render is 2.1×/2.2×/3.2× the JIT at 128/256/512 partials; the crossover is
   below 128 partials (far below the K·B ≈ 10⁵ microbenchmark estimate,
   because the real per-sample graph is much fatter than a bare sine).
2. **Pipelined dispatch decouples the audio thread from synthesis cost
   entirely**: callback cost is a constant ~0.1 ms (max 0.73 ms measured,
   including a mid-playback hot-swap) at every patch size, zero dropouts.
   The 512-partial patch that the JIT cannot play (drops) is a non-event.
3. **Dual-load is free**: hot-swap latency with MSL+JIT ≈ JIT alone
   (3.52 s vs 3.55 s at 512 partials — dominated by session compile + LLVM).
4. **Correctness held throughout**: `metal_vs_jit` SNRs identical in sync and
   pipelined modes; ~140 dB (the f32 output floor) on production patches,
   flat at τ+2⁴⁰.

## Offline render throughput (per 512-sample block, compile separated)

| partials | JIT | Metal (sync) | speedup | JIT headroom | Metal headroom |
|---|---|---|---|---|---|
| 128 | 1.158 ms | 0.553 ms | 2.1× | 10.0× | 21.0× |
| 256 | 2.347 ms | 1.057 ms | 2.2× |  4.9× | 11.0× |
| 512 | 7.850 ms | 2.462 ms | 3.2× |  1.5× |  4.7× |

Deadline 11.61 ms (B=512 @ 44.1k). Sync Metal still pays the ~200 µs
round-trip toll per block; the sustained GPU cost implies ~2000+ partials
before the GPU itself saturates at this block size.

## Live sessions (8 s play, host CPU sampled, DAC stats)

| patch | mode | load (dual compile) | proc CPU | cb avg | cb max | drops |
|---|---|---|---|---|---|---|
| 128 | jit        | 0.88 s | 1.3% | 1.496 ms | 3.53 ms | 0 |
| 128 | metal-pipe | 1.00 s | 1.3% | 0.097 ms | 0.24 ms | 0 |
| 256 | jit        | 1.75 s | 2.1% | 2.714 ms | 6.71 ms | 0 |
| 256 | metal-pipe | 1.71 s | 1.4% | 0.095 ms | 0.45 ms | 0 |
| 512 | jit        | 3.55 s | 6.3% | 7.732 ms | 14.64 ms | **4** |
| 512 | metal      | 3.52 s | 2.9% | 4.015 ms | 22.95 ms | **8** |
| 512 | metal-pipe | 3.37 s | 1.5% | **0.098 ms** | **0.23 ms** | **0** |

- The JIT at 512 partials is over the cliff in real use (67% audio-thread
  load, max over deadline → dropouts).
- **Sync Metal's jitter tail is real** (max 22.95 ms under normal desktop
  contention — the reserved-GPU caveat from `gpu_time_partition/findings.md`,
  reproduced live). Pipelining removes it from the critical path entirely.
- Pipelined callback cost is patch-size-INDEPENDENT: the audio thread only
  copies a completed buffer and enqueues the next future block.

## The pipeline's trade (and why it's cheap here)

`TROPICAL_METAL_PIPELINE=1` pre-renders D=3 FUTURE blocks — legal because
kernels are closed-form: block S+kB is a pure function of its sample index
and the slot snapshot at enqueue time. So unlike a stream-DSP pipeline there
is ZERO audio-position latency; the cost is **param-change latency of up to
D blocks** (34.8 ms at B=512, 8.7 ms at B=128). Clock jumps
(scrub/`set_sample_index`) re-prime the ring at the requested position;
hot-swap primes at the carried `sample_index` (measured seamless, max
0.73 ms callback through a swap).

## Recommendation

Default the live engine to `TROPICAL_BACKEND=metal` + pipeline on Apple
hardware; keep B=128–256 if the D-block param latency matters (8.7–17.4 ms —
comparable to typical controller→audio latency), B=512 for maximum headroom.
The JIT remains the correctness reference, the scope path (`render_window`),
and the portability fallback — dual-load makes that free.

## Pending

- 30-min soak (leaks/@autoreleasepool discipline, SNR drift) — not yet run.
- B-sweep of live sessions (engine buffer length is fixed at boot; needs a
  boot flag before B=128/256 live rows can be measured).
- Knob-latency measurement under pipeline (bounded by D·B by construction;
  not yet measured empirically).
