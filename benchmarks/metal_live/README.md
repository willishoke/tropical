# Metal live qualification

The current harness exercises the real emitted Metal kernel, the off-RT epoch
render worker, and the mandatory dual-loaded JIT reference. It has two modes:

```sh
# Bdev=128/256/512, Rgpu=512, all four write disciplines
benchmarks/metal_live/run.sh --mode latency

# Authorized release-candidate row on the canonical M1 Pro
benchmarks/metal_live/run.sh --mode soak \
  --duration-seconds 600 --buffer 512 --render-frames 512
```

`Bdev` is the exact RtAudio callback quantum. `Rgpu` is the Metal tile render
quantum and must be a positive multiple of `Bdev`. The runtime owns two banks
of four preallocated `Rgpu` tiles. A dedicated worker submits and waits for
Metal; the callback only validates exact epoch/device/source tags and copies
one `Bdev` slice from a ready tile.

`TROPICAL_METAL_RENDER_TILE_FRAMES` selects `Rgpu` for qualification. The
engine boot block length is independently selectable with
`TROPICAL_BUFFER_LENGTH` (16..16384) before Runtime/DAC construction. Both
default to 512 in the live product configuration. The retired future-block
pipeline controls and depth diagnostic are absent.

## Current support envelope

The current release-qualified Live-Metal envelope is limited to
Bdev=512/Rgpu=512 with a four-tile worker capacity on the canonical Apple M1
Pro (`MacBookPro18,1`, 16 Metal cores, macOS 26.3). Candidate `8d92a64` has a
retained
[`600-second qualification row`](data/reverse-crossing-fix-soak-b512-r512-600s-8d92a64-m1pro-20260730.jsonl)
with all 35 acceptance gates true across 51,681 measured callbacks. Its four
Metal/JIT checkpoints measure 143.947, 144.070, 142.615, and 143.968 dB; all
120 clock jumps and 20 hot-swaps were acknowledged; queue, callback,
activation, ownership, and device-continuity faults were zero; and 285 valid
post-warmup RSS samples showed no material growth. No other hardware or
device/render quantum is inferred.

An earlier Bdev=512/Rgpu=512 600-second row on the same M1 Pro blocked on one
latched render starvation observed at the first measured poll. The preserved
[`artifact`](data/epoch-worker-soak-b512-r512-600s-29e0f7de0ada-m1pro-20260729.jsonl)
records exact Bdev=512/Rgpu=512, a four-tile capacity, zero Metal dispatch
failures, zero tag mismatches, and zero activation failures. It failed closed
before the scheduled clock-jump, A/B-swap, reference, and RSS gates, so none
of those results is inferred.

A subsequent
[`diagnostic`](data/diagnostic-prime-drain-b512-r512-45s-328d537-m1pro-20260729.jsonl)
located the fault before DAC start: eight tight generic warm-up calls drained
the four-tile primed window, and the fifth call wrapped to a still-free tile
0. This was a benchmark/DAC priming defect, not a Metal command-buffer
failure.

Candidate `8f7eecb` repairs that defect with off-RT exact-tile waits for
unpaced rendering and a non-consuming Metal readiness barrier for DAC start,
device switch, and reconnect. Its retained
[`60-second hardware row`](data/hardened-worker-smoke-b512-r512-60s-8f7eecb-m1pro-20260729.jsonl)
records zero starvation before start, through startup, and across 5,170
measured callbacks; its separate 1,000-block offline support row also records
zero starvation. Worker-stage telemetry is complete and ordered.

That retained row remains blocked on a distinct final-reference correctness
gate: three Metal/JIT checkpoints measured above 142 dB, while a final reverse
clock jump across the last velocity anchor measured 78.585 dB. A deterministic
no-DAC reproducer has since isolated the cause: the glide start coordinate
crossed the ordinary f32 Metal slot ABI before subtraction, so an absolute
timestamp near 2^40 lost the entire 20 ms ramp's resolution. The fix transports
that coordinate as four exact 16-bit limbs and performs the integer subtraction
before converting the bounded elapsed delta to float. The reproducer's full
reverse arm now measures 143.021 dB, with its exact/stale oracle control at
142.380/83.429 dB.

Candidate `4263faf` now has a clean retained
[`60-second hardware row`](data/reverse-crossing-fix-smoke-b512-r512-60s-4263faf-m1pro-20260730.jsonl).
Its start, post-2^40-jump, post-swap, and final checkpoints measure 144.028,
144.163, 142.638, and 142.954 dB. All 35 acceptance gates pass, including zero
starvation at every startup/measured boundary, zero queue/callback/activation
faults, complete ordered worker telemetry, and full clock-jump/hot-swap
acknowledgement. This short validation closes the reverse-crossing blocker and
warranted a new 600-second release-qualification row. The subsequent passing
`8d92a64` long row now qualifies only the exact envelope stated above.

The retained B=128/D=3 and B=512/D=3 failures in
[`findings.md`](findings.md) describe the superseded callback-owned
future-dispatch implementation. They remain valuable causal evidence but do
not qualify or disqualify this different runtime architecture. No document
claims that a worker always meets a hardware deadline.

## What a row gates

The soak compiles the real 512-partial bank. The addressed bank is the
reference signal behind a 1e-8 trim, while a 55 Hz clock canary is bounded at
1e-12. Every reference checkpoint requires the bank to contribute at least
99% of the conservative signal-energy lower bound and Metal/JIT SNR to exceed
100 dB, so the canary cannot mask the heavy graph.

The harness records and gates:

- exact requested and negotiated `Bdev`, `Rgpu`, and four-tile worker
  capacity;
- real callback count, exact maximum, and p50/p95/p99 bounds from a
  preallocated one-microsecond histogram;
- zero measured-window RtAudio underruns/overruns, non-finite samples, device
  continuity events, and runtime ownership failures;
- zero Metal dispatch failures, render starvations, tag mismatches, activation
  failures, and callback-thread Metal provenance violations;
- ordered worker stage timestamps from request receipt through old-epoch
  retirement, plus activation latency min/mean/max and worker CPU/wall time;
- every requested periodic clock jump and hot-swap reaching an acknowledged
  activation epoch, with the final reference after writes stop and the last
  activation is acknowledged;
- exact control `effective_sample_index` replay by an isolated JIT runtime at
  start, after repeated post-2^40 jumps, after A/B swaps, and at the end;
- resident memory every two seconds, with at least three positive post-warmup
  samples and no material monotonic growth;
- distinct original/replacement IR and MSL artifacts, hot-swap wall time,
  dispatch wall time, and a separate offline supporting row.

A row blocks if any required evidence is missing, the exact callback maximum
reaches the Bdev deadline, callback p99 reaches half the deadline, measured
callback coverage leaves [0.99, 1.01], a reference checkpoint is at or below
100 dB, or any sticky fault counter is nonzero. Startup status is separated by
a stats epoch applied at an actual callback boundary. The callback performs no
allocation, lock, I/O, Metal submission, or control-state packing for the
qualification surface.

At Bdev=512 and 44.1 kHz, every callback has an 11.610 ms hard processing
deadline. Activation latency, individual GPU tile duration, render-ahead
reserve, and callback execution are distinct measurements; none enlarges that
deadline.

Before every subprocess, inherited `TROPICAL_*` variables are removed and the
benchmark controls are explicitly set and recorded.

## Deterministic supporting gates

The no-DAC large-clock discriminator reproduces the heavy graph, all 20
retained post-swap velocity epochs, and the final `E - 1,536` reverse capture:

```sh
python3 benchmarks/metal_live/run_velocity_oracle_discriminator.py
```

Its full reverse, capture-after-`E`, no-velocity, and no-swap arms must remain
above 140 dB on Metal versus JIT. It records f64/f32 `tau_base`, includes
glide/anchor causal controls, and forces a callback boundary between production
dispatches: exact-epoch replay must remain above 140 dB while the obsolete
batch-start oracle must fail below 100 dB.

`engine/tests/test_metal_kernel.cpp` additionally covers queue ownership and
bank reuse, callback isolation, terminal dispatch failures, retargeting,
rapid A/B/A activation serialization, and 10,000-event clock/swap and
parameter-epoch stress under CPU contention. These finite tests establish
specific invariants; release qualification remains hardware-scoped empirical
evidence.
