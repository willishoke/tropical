# Metal backend — live findings (V2 phase 6)

## 2026-07-28 reviewed short validation and operating-envelope stop

Independent review authorized one B=512/D=3 validation after the exact-index
oracle correction, followed by one B=128/D=3 operating-envelope row. The
B=512 row is retained as
[`data/reviewed-oracle-fix-smoke-b512-d3-60s-3dc3be4-m1pro-20260728.jsonl`](data/reviewed-oracle-fix-smoke-b512-d3-60s-3dc3be4-m1pro-20260728.jsonl).
It ran from commit `3dc3be45fed602ba8301e44c6f71e66ab339f1ce`;
its SHA-256 is
`f36e87522f0c966b69bb1bb6271d6d49bd63bb2cc8a64d86147af5c941db8fb2`.
That short validation passed all 24 gates, including four reference
checkpoints at 142.543–144.013 dB.

For B=512 at 44.1 kHz, the callback hard deadline is 11.610 ms. The D=3
ordinary-parameter transport measured by the latency matrix is 34.830 ms
(`3×512/44100`) from capture to the first output block. That transport delay
does not provide a 34.8 ms callback processing budget; every callback still
must meet the 11.61 ms hardware period.

The subsequent B=128 row is retained unchanged as
[`data/reviewed-oracle-fix-smoke-b128-d3-60s-a875b68-m1pro-20260728.jsonl`](data/reviewed-oracle-fix-smoke-b128-d3-60s-a875b68-m1pro-20260728.jsonl).
It ran exactly once from commit
`a875b6845c8ebbe9100ae8593b897cb3996b599d`; its SHA-256 is
`5eba153a8637d3cc51f513d383c0912e107d10b580ba553fcff3b50363d40388`.
The row blocked after 19.958 s of measured time with
`post_reset_underrun`: one underrun and one callback overrun were recorded,
and the 5.422916 ms exact callback maximum exceeded the 2.902494 ms B=128
deadline. This is a genuine operating-envelope failure; the evidence does not
establish a cause for the single tail event.

The B=128 row also exercised the correction across a live multi-index batch.
At event block 2754, raw and glide were applied at sample index 442368 while
anchor and velocity were applied at 442496, one 128-sample callback boundary
later. Exact per-dispatch replay remained aligned: the later post-2^40
checkpoint measured 144.135 dB with error below 1e-14. The abort occurred
before hot-swap, so this row makes no post-swap RSS claim.

The approved sequence stopped at the B=128 failure. B=256 and the long-duration
rows were not run and remain untested. The staff product decision scopes the
observed hard-deadline miss to B=128/D=3, which is a known unsupported
configuration on the canonical M1 Pro; it does not block Metal universally.
B=512/D=3 is the supported candidate based on its passing short validation,
but remains pending the required final long soak before release qualification.
B=256 has no support decision until it is tested.

## 2026-07-28 blocked production-dispatch incident

The independently approved 60-second B=512/D=3 actual-DAC smoke is retained as
[`data/approved-short-smoke-b512-d3-60s-b351deb-m1pro-20260727.jsonl`](data/approved-short-smoke-b512-d3-60s-b351deb-m1pro-20260727.jsonl).
It ran from commit `b351deb6b8d9218fce5744e354b72e53bc0c7728`
and returned nonzero, so the planned B=128 and B=256 smokes were not started.
The only manifest status entry is the requested output file itself, created
before the manifest snapshot; the worktree was clean immediately before the
run.

The end reference fell to 88.116 dB with 1.351e-12 maximum error after start,
post-2^40, and post-swap checkpoints of 144.010, 143.940, and 142.664 dB. The
runtime correctly blocked on that row. All other invariants passed: zero
underruns, overruns, non-finite samples, ownership failures, or device
continuity events; 0.281 ms callback p99; 1.00018 callback coverage; exact
B=512/D=3; distinct replacement artifacts; and valid, non-growing RSS.

The incident is a reference-oracle timing defect, not evidence of Metal drift:

- Production dispatch reads the live completed sample boundary independently
  for raw, glide, anchor, and velocity. The old oracle sampled one `live_now`
  before the batch, processed the reference, then reused that stale boundary
  while the four production writes consumed up to 10.264 ms against an
  11.610 ms callback period. Fifteen batches after the clean midpoint consumed
  80.92 ms in total, allowing boundary crossings to accumulate in
  time-dependent anchor/velocity companions.
- The deterministic no-DAC discriminator uses the actual emitted heavy graph,
  fresh replacement defaults, the exact 15-event post-2^40 schedule, and
  synchronous Metal. True 1↔0.75 velocity toggles are bit-identical to JIT
  (999 dB, zero error), as is the velocity=1 no-op control, even though the
  final host `tau_base` is 6,233,064.849705 s and its f32 representation is
  6,233,065. This rejects the competing Metal slot-precision hypothesis.
- With one forced callback crossing inside the production batch, replay at
  each dispatch's exact boundary remains bit-identical. Reusing the obsolete
  batch-start boundary falls below the unchanged 100 dB gate. The checked-in
  `run_velocity_oracle_discriminator.py` reproduces all three cases.

The correction records `applied_sample_index` from each production dispatch
and replays the separate JIT runtime through a qualification-only explicit-now
entry point that shares the exact production math. Production dispatch timing
and the >100 dB acceptance gate are unchanged. Independent review authorized
only the short validation sequence recorded above.

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

The first long attempt was interrupted because review found the original
harness could not prove live SNR, callback p95/p99, or actual event progress.
It is explicitly a rejected diagnostic, not a qualification row. The final
16-second real-DAC harness smoke passed every fail-closed gate with 1379
measured callbacks, zero underruns/overruns, a 0.237 ms p99 upper bound at
1 us resolution, 5.531 ms exact max, 5.41% process CPU/wall, and
144.14–145.19 dB nonzero JIT-reference SNR at start, post-2^40,
midpoint-after-swap, and end. All required event booleans and callback indices
were present, and the ordinary end capture was 37 callbacks after its preceding
write (D+1 required). Three RSS samples after the explicit two-second
post-hot-swap settling boundary were flat and passed the
non-monotonic-growth gate. This validates the harness only; it is not a
long-run memory conclusion.

Raw evidence is intentionally classified rather than blended:

- `data/failed-underrun-smoke-b512-d3-m1pro-20260727.jsonl`: original genuine
  one-underrun failure;
- `data/reset-bounded-diagnostic-b512-d3-m1pro-20260727.jsonl`: clean snapshot
  diagnostic after the startup/reset split;
- `data/interrupted-pre-abort-fix-b512-d3-m1pro-20260727.jsonl` and
  `data/interrupted-review-rejected-b512-d3-m1pro-20260727.jsonl`: manifest-only
  interrupted attempts, never qualification rows;
- `data/pre-hard-gates-diagnostic-b512-d3-m1pro-20260727.jsonl`: pre-threshold
  smoke retained as a diagnostic;
- `data/corrected-harness-smoke-b512-d3-m1pro-20260727.jsonl`: final short
  review smoke with explicit gate results;
- `data/reviewed-oracle-fix-smoke-b512-d3-60s-3dc3be4-m1pro-20260728.jsonl`:
  reviewed passing short validation after the exact-index correction;
- `data/reviewed-oracle-fix-smoke-b128-d3-60s-a875b68-m1pro-20260728.jsonl`:
  reviewed genuine B=128 operating-envelope failure.

The genuine B=128 failure blocks the B=128/D=3 configuration on the canonical
M1 Pro, not Metal universally. B=256 remains untested, while B=512/D=3 remains
a supported candidate pending its required final long soak and is not yet
release-qualified.

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

## Historical recommendation (superseded)

The July 7 recommendation was to default the live engine to
`TROPICAL_BACKEND=metal` + pipeline on Apple
hardware; keep B=128–256 if the D-block param latency matters (8.7–17.4 ms —
comparable to typical controller→audio latency), B=512 for maximum headroom.
The JIT remains the correctness reference, the scope path (`render_window`),
and the portability fallback — dual-load makes that free.

That recommendation is **not current release guidance**. The sprint evidence
supports exact D-block transport and a corrected B=512/D=3 short smoke. The
required final long B=512/D=3 soak is pending, B=256 is untested, and B=128/D=3
is outside the supported envelope on the canonical M1 Pro.

## Pending

- The required final 30-minute B=512/D=3 actual-DAC soak. No run is authorized
  by this document.
- A B=256 support decision; that configuration remains untested. B=128/D=3 is
  not a pending candidate on the canonical M1 Pro.
- Process user+system CPU seconds and measured-wall fraction are recorded.
  Per-core attribution, pipeline queue-depth samples, and Metal
  resource/object counts are not exposed by the current harness.
- Callback p95/p99 are 1 us histogram upper bounds, not exact retained samples;
  the exact max remains available. The fixed histogram has an explicit >=20 ms
  overflow bin.
- The control-latency matrix measures impulsive slot transport for all four
  host write shapes; deliberately glided audible onset remains outside that
  transport claim.
- A second Apple generation and compositor-contention row remain optional,
  untested hardware risks.
