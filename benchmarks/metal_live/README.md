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

Live-Metal release support is withheld. The one authorized
Bdev=512/Rgpu=512, 600-second row on the canonical M1 Pro blocked on one
latched render starvation observed at the first measured poll and was not
retried. The preserved
[`artifact`](data/epoch-worker-soak-b512-r512-600s-29e0f7de0ada-m1pro-20260729.jsonl)
records exact Bdev=512/Rgpu=512, a four-tile capacity, zero Metal dispatch
failures, zero tag mismatches, and zero activation failures. It failed closed
before the scheduled clock-jump, A/B-swap, reference, and RSS gates, so none
of those results is inferred. No epoch-worker device/render configuration is
currently release-qualified.

A subsequent
[`diagnostic`](data/diagnostic-prime-drain-b512-r512-45s-328d537-m1pro-20260729.jsonl)
located the fault before DAC start: eight tight generic warm-up calls drained
the four-tile primed window, and the fifth call wrapped to a still-free tile
0. This is a benchmark/DAC priming defect, not evidence of a Metal
command-buffer failure. It requires a new fixed candidate and does not turn
the failed qualification row into a pass.

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

The no-DAC large-clock discriminator reproduces the heavy graph and exact
production control epochs:

```sh
python3 benchmarks/metal_live/run_velocity_oracle_discriminator.py
```

It requires true 1↔0.75 velocity toggles and the velocity-no-op control to
remain above 140 dB on Metal versus JIT. It then forces a callback boundary
between production dispatches: exact-epoch replay must remain above 140 dB
while the obsolete batch-start oracle must fail below 100 dB.

`engine/tests/test_metal_kernel.cpp` additionally covers queue ownership and
bank reuse, callback isolation, terminal dispatch failures, retargeting,
rapid A/B/A activation serialization, and 10,000-event clock/swap and
parameter-epoch stress under CPU contention. These finite tests establish
specific invariants; release qualification remains hardware-scoped empirical
evidence.
