# Metal backend — live findings (V2 phase 6)

## 2026-07-28 final B=512/D=3 qualification failure

The one authorized 30-minute B=512/D=3 actual-DAC soak is retained unchanged
as
[`data/final-soak-b512-d3-1800s-bd7c9bf-m1pro-20260728.jsonl`](data/final-soak-b512-d3-1800s-bd7c9bf-m1pro-20260728.jsonl).
It ran from the clean code candidate
`bd7c9bf56690e383bdd53e4ca64c73defab533f3`; the only manifest status entry
is the output file created before the snapshot. Its SHA-256 is
`ab03fbc7ad849799cb0a770c4858d552ccba5d6c505d13792ae8b532e9e8bfd0`.
The failed artifact is the final qualification result; it was not retried.

The harness aborted at the first scheduled clock jump after 450.050 seconds
and 38,763 measured callbacks. One callback took 21.009750 ms against the
11.609977 ms B=512 deadline, producing
`post_reset_callback_overrun`. The immediately preceding snapshot had a
0.336500 ms maximum. Across the measured window there were zero underruns,
zero ownership failures, zero Metal dispatch failures, zero non-finite
samples, 0.203 ms callback p99, 0.999972 callback coverage, and no material RSS
growth. The start reference passed at 144.013 dB. The abort correctly occurred
before clock-jump progress, hot-swap, and final-reference evidence, so those
gates remain unsatisfied rather than being inferred.

The timing places the failure on the discontinuity re-prime boundary. A
pipelined clock jump drains stale futures, submits D replacement blocks from
the requested coordinate, and synchronously waits for the first replacement
inside the callback. The row proves that this path exposed the callback to a
deadline-breaking tail on the canonical M1 Pro. It does not by itself
distinguish deterministic re-prime work from Metal/system scheduling jitter
within that synchronous wait. Earlier retained B512/D3 jump windows completed
at 2.946 ms and 4.460 ms, so a fixed deterministic 21 ms path cost is not
supported; a tail exposed by the synchronous re-prime is the leading
explanation. The row does not authorize weakening the hard deadline to a
perceptual-latency budget.

The supported-envelope decision is therefore conservative: B=128/D=3 and
B=512/D=3 are not release-qualified on the canonical M1 Pro, B=256 remains
untested, and no pipelined Metal configuration is currently declared supported
from this sprint evidence. Metal remains a valid backend for offline,
synchronous, and explicitly experimental use; this result does not invalidate
the backend's correctness evidence or the steady-state performance result.

## 2026-07-28 reviewed short validation and operating-envelope stop

Independent review authorized one B=512/D=3 validation after the exact-index
oracle correction, followed by one B=128/D=3 operating-envelope row. The
B=512 row is retained as
[`data/reviewed-oracle-fix-smoke-b512-d3-60s-3dc3be4-m1pro-20260728.jsonl`](data/reviewed-oracle-fix-smoke-b512-d3-60s-3dc3be4-m1pro-20260728.jsonl).
It ran from commit `3dc3be4cb2f1727a6243ede8f6707938548dc766`;
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

That approved short sequence stopped at the B=128 failure. B=256 remained
untested; the later final B=512 row is recorded above. The staff product
decision scopes the
observed hard-deadline miss to B=128/D=3, which is a known unsupported
configuration on the canonical M1 Pro; it does not block Metal universally.
B=512/D=3 subsequently failed its final qualification row and is also
unsupported for release on this machine. B=256 has no support decision until
it is tested.

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

That historical correction recorded `applied_sample_index`, which at the time
meant the completed callback boundary observed by each dispatch, and replayed
the separate JIT runtime through a qualification-only explicit-now entry
point. Current evidence replaces that ambiguous field with
`observed_sample_index` and `effective_sample_index`; discipline math and
oracle replay use the latter, the first output sample at which the generation
is audible. Independent review authorized only the short validation sequence
recorded above.

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
- That historical CTest run covered the then-supported D=3 alias and explicit
  depth precedence. The alias is now retired; current tests require explicit
  D=1/2/3, default synchronous mode, invalid-depth refusal, exact D-block
  live-column lag, clock-jump draining, and hot-swap re-prime.

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
M1 Pro, not Metal universally. B=256 remains untested. The later final
B=512/D=3 row recorded above also failed qualification.

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
2. **In these July 7 B=512 fixture rows**, pipelined dispatch decoupled the
   audio thread from synthesis cost: callback cost was ~0.1 ms (max 0.73 ms,
   including a mid-playback hot-swap) at every patch size, with zero recorded
   dropouts. This is retained historical evidence, not a claim about later
   B=128/D=3 qualification; that reviewed row failed as recorded above.
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
- In this fixed-B historical fixture, pipelined callback cost was insensitive
  to patch size: the audio thread copied a completed buffer and enqueued the
  next future block.

## The pipeline's trade (and why it's cheap here)

The July 7 fixture used a D=3 future-block pipeline — legal because
kernels are closed-form: block S+kB is a pure function of its sample index
and the slot snapshot at enqueue time. So unlike a stream-DSP pipeline there
is ZERO audio-position latency; the cost is **param-change latency of up to
D blocks** (34.8 ms at B=512, 8.7 ms at B=128). Clock jumps
(scrub/`set_sample_index`) re-prime the ring at the requested position;
hot-swap primes at the carried `sample_index`. The 0.73 ms swap callback cited
by that historical fixture is not a claim about every later buffer-size row;
the reviewed B=128 operating-envelope failure above recorded a 5.422916 ms
maximum and an underrun.

## Historical recommendation (superseded)

The July 7 recommendation was to default the live engine to
`TROPICAL_BACKEND=metal` + pipeline on Apple
hardware; keep B=128–256 if the D-block param latency matters (8.7–17.4 ms —
comparable to typical controller→audio latency), B=512 for maximum headroom.
The JIT remains the correctness reference, the scope path (`render_window`),
and the portability fallback — dual-load makes that free.

That recommendation is **not current release guidance**. The sprint evidence
supports exact D-block transport and a corrected B=512/D=3 short smoke, but
the final long row exposed a re-prime deadline miss. B=128/D=3 and B=512/D=3
are outside the release-qualified envelope on the canonical M1 Pro; B=256 is
untested.

## Pending

- Redesign or otherwise bound clock-jump/hot-swap re-prime work before another
  B=512/D=3 qualification proposal. The failed final row does not authorize a
  retry. Candidate follow-ups are stage-timing instrumentation and an
  epoch-tagged ring primed off the audio thread; adding depth alone cannot
  preserve a discontinuous coordinate because every queued future is stale.
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
